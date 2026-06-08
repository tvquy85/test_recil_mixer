"""Training CLI for ReCIL-Mixer smoke and real-data runs."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import ReCILDataset, build_context_cache, default_split_indices, load_stockmixer_dataset
from .losses import recil_loss
from .metrics import evaluate_predictions
from .model import ReCILMixer, VALID_VARIANTS


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train ReCIL-Mixer with clean checkpointing and logging")
    parser.add_argument("--dataset", choices=["nasdaq", "sp500", "crypto"], default="nasdaq")
    parser.add_argument("--data-root", default=str(Path(__file__).resolve().parents[2] / "dataset"))
    parser.add_argument("--variant", choices=sorted(VALID_VARIANTS), default="full")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--market-dim", type=int, default=32)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--lookback", type=int, default=16)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--alpha-rank", type=float, default=0.1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--quick-test", action="store_true")
    return parser


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_synthetic_arrays(seed: int, lookback: int, steps: int):
    rng = np.random.default_rng(seed)
    num_assets, trade_dates, num_features = 20, max(64, lookback + steps + 32), 5
    time = np.arange(trade_dates, dtype=np.float32)
    eod = np.zeros((num_assets, trade_dates, num_features), dtype=np.float32)
    asset_bias = rng.normal(0.0, 0.02, size=(num_assets, 1)).astype(np.float32)
    for asset in range(num_assets):
        trend = 0.002 * time + asset_bias[asset]
        seasonal = 0.04 * np.sin(time * (0.07 + asset * 0.001))
        for feature in range(num_features):
            noise = rng.normal(0.0, 0.005, size=trade_dates).astype(np.float32)
            eod[asset, :, feature] = 10.0 + asset * 0.03 + feature * 0.1 + trend + seasonal + noise

    price = eod[:, :, -1].copy()
    gt = np.zeros((num_assets, trade_dates), dtype=np.float32)
    raw_return = np.zeros_like(gt)
    raw_return[:, 1:] = (price[:, 1:] - price[:, :-1]) / np.maximum(price[:, :-1], 1e-8)
    context_signal = np.sin(time * 0.11).astype(np.float32)
    gt[:, :] = raw_return + 0.02 * eod[:, :, 0] / np.maximum(eod[:, :, -1], 1e-8)
    gt += 0.01 * context_signal[None, :]
    gt += rng.normal(0.0, 0.001, size=gt.shape).astype(np.float32)
    mask = np.ones((num_assets, trade_dates), dtype=np.float32)
    valid_index, test_index = 36, 50
    return eod, mask, gt, price, valid_index, test_index


def offsets_for_target_range(start_index: int, end_index: int, lookback: int, steps: int, trade_dates: int) -> np.ndarray:
    start = max(0, int(start_index) - lookback - steps + 1)
    stop = min(trade_dates - lookback - steps + 1, int(end_index) - lookback - steps + 1)
    if stop <= start:
        return np.array([], dtype=np.int64)
    return np.arange(start, stop, dtype=np.int64)


def cap_offsets(offsets: np.ndarray, cap: int | None) -> np.ndarray:
    if cap is None or offsets.shape[0] <= cap:
        return offsets
    return offsets[:cap]


def prepare_data(args):
    if args.synthetic:
        eod, mask, gt, price, valid_index, test_index = make_synthetic_arrays(args.seed, args.lookback, args.steps)
        dataset_name = "synthetic"
    else:
        loaded = load_stockmixer_dataset(args.data_root, args.dataset, steps=args.steps)
        eod, mask, gt, price = loaded["eod_data"], loaded["mask_data"], loaded["gt_data"], loaded["price_data"]
        valid_index, test_index = default_split_indices(loaded["dataset"], eod.shape[1])
        dataset_name = loaded["dataset"]

    trade_dates = int(eod.shape[1])
    train_offsets = offsets_for_target_range(0, valid_index, args.lookback, args.steps, trade_dates)
    val_offsets = offsets_for_target_range(valid_index, test_index, args.lookback, args.steps, trade_dates)
    test_offsets = offsets_for_target_range(test_index, trade_dates, args.lookback, args.steps, trade_dates)
    if args.quick_test:
        train_offsets = cap_offsets(train_offsets, 4)
        val_offsets = cap_offsets(val_offsets, 4)
        test_offsets = cap_offsets(test_offsets, 4)
    if train_offsets.size == 0 or val_offsets.size == 0 or test_offsets.size == 0:
        raise ValueError("empty train/val/test offsets; check lookback, steps, and split indices")

    context_cache = build_context_cache(eod, mask, train_offsets, val_offsets, test_offsets, args.lookback)
    datasets = {
        "train": ReCILDataset(eod, mask, gt, price, context_cache, train_offsets, args.lookback, args.steps),
        "val": ReCILDataset(eod, mask, gt, price, context_cache, val_offsets, args.lookback, args.steps),
        "test": ReCILDataset(eod, mask, gt, price, context_cache, test_offsets, args.lookback, args.steps),
    }
    return dataset_name, datasets


def batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "x": batch["x"].to(device),
        "y": batch["y"].to(device),
        "mask": batch["mask"].to(device),
        "context": batch["context"].to(device),
    }


def train_one_epoch(model, loader, optimizer, device, alpha_rank: float) -> float:
    model.train()
    total = 0.0
    count = 0
    for raw_batch in loader:
        batch = batch_to_device(raw_batch, device)
        optimizer.zero_grad()
        pred, aux = model(batch["x"], batch["context"], mask=batch["mask"])
        loss = recil_loss(pred, batch["y"], batch["mask"], aux=aux, alpha_rank=alpha_rank)
        loss.backward()
        optimizer.step()
        total += float(loss.detach().cpu())
        count += 1
    return total / max(count, 1)


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _stack_or_empty(items: list[np.ndarray], trailing_shape: tuple[int, ...]) -> np.ndarray:
    if not items:
        return np.empty((0,) + trailing_shape, dtype=np.float32)
    return np.concatenate(items, axis=0).astype(np.float32, copy=False)


@torch.no_grad()
def evaluate_loader(model, loader, device, alpha_rank: float):
    model.eval()
    preds: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    contexts: list[np.ndarray] = []
    scale_weights: list[np.ndarray] = []
    router_weights: list[np.ndarray] = []
    context_gates: list[np.ndarray] = []
    total_loss = 0.0
    count = 0
    for raw_batch in loader:
        batch = batch_to_device(raw_batch, device)
        pred, aux = model(batch["x"], batch["context"], mask=batch["mask"])
        loss = recil_loss(pred, batch["y"], batch["mask"], aux=aux, alpha_rank=alpha_rank)
        total_loss += float(loss.detach().cpu())
        count += 1
        preds.append(_to_numpy(pred))
        labels.append(_to_numpy(batch["y"]))
        masks.append(_to_numpy(batch["mask"]))
        contexts.append(_to_numpy(batch["context"]))
        if aux["scale_weights"] is not None:
            scale_weights.append(_to_numpy(aux["scale_weights"]))
        if aux["router_weights"] is not None:
            router_weights.append(_to_numpy(aux["router_weights"]))
        if aux["context_gate"] is not None:
            context_gates.append(_to_numpy(aux["context_gate"]))

    pred_arr = np.concatenate(preds, axis=0)
    label_arr = np.concatenate(labels, axis=0)
    mask_arr = np.concatenate(masks, axis=0)
    context_arr = np.concatenate(contexts, axis=0)
    metrics = evaluate_predictions(pred_arr, label_arr, mask_arr, k=10)
    return {
        "loss": total_loss / max(count, 1),
        "metrics": metrics,
        "predictions": pred_arr.astype(np.float32),
        "labels": label_arr.astype(np.float32),
        "masks": mask_arr.astype(np.float32),
        "contexts": context_arr.astype(np.float32),
        "aux": {
            "scale_weights": _stack_or_empty(scale_weights, (0,)),
            "router_weights": _stack_or_empty(router_weights, (0,)),
            "context_gate": _stack_or_empty(context_gates, (0,)),
        },
    }


def best_score(val_metrics: dict[str, Any], val_loss: float) -> float:
    for key in ("RankIC", "IC"):
        value = float(val_metrics.get(key, float("nan")))
        if np.isfinite(value):
            return value
    return -float(val_loss)


def clean_json(value):
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(clean_json(payload), indent=2, sort_keys=True), encoding="utf-8")


def save_checkpoint(path: Path, model, optimizer, epoch: int, config: dict[str, Any], val_result, test_result, score: float) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": int(epoch),
            "config": config,
            "val_metrics": clean_json(val_result["metrics"]),
            "test_metrics": clean_json(test_result["metrics"]),
            "best_score": float(score),
        },
        path,
    )


def write_train_log(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["epoch", "train_loss", "val_loss", "test_loss", "val_RankIC", "val_IC", "test_RankIC", "test_IC", "best_score"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def run(args) -> Path:
    set_deterministic(args.seed)
    device = resolve_device(args.device)
    dataset_name, datasets = prepare_data(args)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(datasets["val"], batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=False)

    sample = datasets["train"][0]
    model = ReCILMixer(
        num_assets=sample["x"].shape[0],
        num_features=sample["x"].shape[-1],
        d_model=args.d_model,
        context_dim=sample["context"].shape[0],
        market_dim=args.market_dim,
        num_experts=args.num_experts,
        variant=args.variant,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    config = vars(args).copy()
    config.update({"resolved_dataset": dataset_name, "device_resolved": str(device)})
    run_dir = Path(args.output_dir) / dataset_name / args.variant / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(run_dir / "config.json", config)

    best = float("-inf")
    best_test = None
    log_rows: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, args.alpha_rank)
        val_result = evaluate_loader(model, val_loader, device, args.alpha_rank)
        test_result = evaluate_loader(model, test_loader, device, args.alpha_rank)
        score = best_score(val_result["metrics"], val_result["loss"])
        if score > best:
            best = score
            best_test = test_result
            save_checkpoint(run_dir / "best_model.pt", model, optimizer, epoch, config, val_result, test_result, score)
        save_checkpoint(run_dir / "last_model.pt", model, optimizer, epoch, config, val_result, test_result, score)
        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_result["loss"],
                "test_loss": test_result["loss"],
                "val_RankIC": val_result["metrics"]["RankIC"],
                "val_IC": val_result["metrics"]["IC"],
                "test_RankIC": test_result["metrics"]["RankIC"],
                "test_IC": test_result["metrics"]["IC"],
                "best_score": best,
            }
        )

    final_test = best_test if best_test is not None else evaluate_loader(model, test_loader, device, args.alpha_rank)
    write_train_log(run_dir / "train_log.csv", log_rows)
    save_json(run_dir / "metrics.json", {"test": final_test["metrics"], "best_score": best})
    np.save(run_dir / "predictions.npy", final_test["predictions"])
    np.save(run_dir / "labels.npy", final_test["labels"])
    np.save(run_dir / "masks.npy", final_test["masks"])
    np.save(run_dir / "contexts.npy", final_test["contexts"])
    np.savez(run_dir / "aux_outputs.npz", **final_test["aux"])
    return run_dir


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_dir = run(args)
    print(f"saved ReCIL run to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
