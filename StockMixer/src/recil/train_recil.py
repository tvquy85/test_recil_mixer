"""Training CLI for ReCIL-Mixer smoke and real-data runs.

This replacement keeps the original command-line workflow but adds practical
controls needed for paper-grade experiments: AdamW/weight decay, router entropy
regularization, gradient clipping, optional AMP, optional strict determinism,
turnover/cost metrics, active parameter logging, and early stopping.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import ReCILDataset, build_context_cache, default_split_indices, load_stockmixer_dataset
from .losses import recil_loss
from .metrics import evaluate_predictions
from .model import ReCILMixer, VALID_VARIANTS


LOG_FIELDS = [
    "epoch",
    "epoch_sec",
    "train_loss",
    "train_grad_norm",
    "val_loss",
    "test_loss",
    "val_RankIC",
    "val_RankICIR",
    "val_IC",
    "val_ICIR",
    "val_Precision@10",
    "test_RankIC",
    "test_RankICIR",
    "test_IC",
    "test_ICIR",
    "test_Precision@10",
    "test_Sharpe@10",
    "test_Turnover@10",
    "test_CostSharpe@10",
    "best_score",
    "bad_epochs",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train ReCIL-Mixer with clean checkpointing and logging")
    parser.add_argument("--dataset", choices=["nasdaq", "sp500", "crypto"], default="nasdaq")
    parser.add_argument("--data-root", default=str(Path(__file__).resolve().parents[2] / "dataset"))
    parser.add_argument("--variant", choices=sorted(VALID_VARIANTS), default="full")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--market-dim", type=int, default=32)
    parser.add_argument("--rank", type=int, default=0, help="low-rank asset interaction bottleneck; 0 uses an automatic value")
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--lookback", type=int, default=16)
    parser.add_argument("--steps", type=int, default=1, help="forecast horizon in StockMixer data alignment")
    parser.add_argument(
        "--scales",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="non-overlapping temporal pooling scales; each must divide lookback",
    )
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--alpha-rank", type=float, default=0.1)
    parser.add_argument("--lambda-entropy", type=float, default=0.0)
    parser.add_argument("--max-pairs-per-day", type=int, default=4096)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--transaction-cost-bps", type=float, default=0.0)
    parser.add_argument("--gate-init", type=float, default=-2.0)
    parser.add_argument("--scale-gate-temperature", type=float, default=1.0)
    parser.add_argument("--router-init", choices=["zero", "small_normal"], default="zero")
    parser.add_argument("--router-temperature", type=float, default=1.0)
    parser.add_argument(
        "--interaction-warmup-epochs",
        type=int,
        default=0,
        help="linearly ramp interaction branches over this many epochs; 0 keeps full strength",
    )
    parser.add_argument(
        "--no-mask-invalid-assets",
        dest="mask_invalid_assets",
        action="store_false",
        default=True,
        help="disable model-side hidden/prediction masking; losses and metrics remain mask-aware",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true", help="enable CUDA automatic mixed precision")
    parser.add_argument("--strict-deterministic", action="store_true")
    parser.add_argument("--patience", type=int, default=0, help="early-stop patience in epochs; 0 disables")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--run-tag", default="", help="optional suffix for variant directory to avoid overwriting config sweeps")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--quick-test", action="store_true")
    return parser


def set_deterministic(seed: int, strict: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if strict:
        torch.use_deterministic_algorithms(True, warn_only=True)


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
    num_assets, trade_dates, num_features = 20, max(96, lookback + steps + 48), 5
    time_idx = np.arange(trade_dates, dtype=np.float32)
    eod = np.zeros((num_assets, trade_dates, num_features), dtype=np.float32)
    asset_bias = rng.normal(0.0, 0.02, size=(num_assets, 1)).astype(np.float32)
    for asset in range(num_assets):
        trend = 0.002 * time_idx + asset_bias[asset]
        seasonal = 0.04 * np.sin(time_idx * (0.07 + asset * 0.001))
        for feature in range(num_features):
            noise = rng.normal(0.0, 0.005, size=trade_dates).astype(np.float32)
            eod[asset, :, feature] = 10.0 + asset * 0.03 + feature * 0.1 + trend + seasonal + noise
    price = eod[:, :, -1].copy()
    gt = np.zeros((num_assets, trade_dates), dtype=np.float32)
    raw_return = np.zeros_like(gt)
    raw_return[:, 1:] = (price[:, 1:] - price[:, :-1]) / np.maximum(price[:, :-1], 1e-8)
    context_signal = np.sin(time_idx * 0.11).astype(np.float32)
    gt[:, :] = raw_return + 0.02 * eod[:, :, 0] / np.maximum(eod[:, :, -1], 1e-8)
    gt += 0.01 * context_signal[None, :]
    gt += rng.normal(0.0, 0.001, size=gt.shape).astype(np.float32)
    mask = np.ones((num_assets, trade_dates), dtype=np.float32)
    # Inject a few invalid tails to exercise mask-aware model paths.
    mask[-2:, -10:] = 0.0
    valid_index, test_index = int(trade_dates * 0.6), int(trade_dates * 0.8)
    return eod, mask, gt, price, valid_index, test_index


def offsets_for_target_range(start_index: int, end_index: int, lookback: int, steps: int, trade_dates: int) -> np.ndarray:
    start = max(0, int(start_index) - lookback - steps + 1)
    stop = min(trade_dates - lookback - steps + 1, int(end_index) - lookback - steps + 1)
    if stop <= start:
        return np.array([], dtype=np.int64)
    return np.arange(start, stop, dtype=np.int64)


def cap_offsets(offsets: np.ndarray, cap: Optional[int]) -> np.ndarray:
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


def batch_to_device(batch: Dict[str, Any], device: torch.device, non_blocking: bool = False) -> Dict[str, Any]:
    return {
        "x": batch["x"].to(device, non_blocking=non_blocking),
        "y": batch["y"].to(device, non_blocking=non_blocking),
        "mask": batch["mask"].to(device, non_blocking=non_blocking),
        "context": batch["context"].to(device, non_blocking=non_blocking),
    }


def _amp_context(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.amp.autocast(device_type=device.type, enabled=True)


def _make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:  # pragma: no cover - for older PyTorch builds
        return torch.cuda.amp.GradScaler(enabled=enabled)


def train_one_epoch(model, loader, optimizer, device, args, scaler) -> Dict[str, float]:
    model.train()
    total = 0.0
    count = 0
    grad_norms = []
    non_blocking = device.type == "cuda"
    amp_enabled = bool(args.amp and device.type == "cuda")

    for raw_batch in loader:
        batch = batch_to_device(raw_batch, device, non_blocking=non_blocking)
        optimizer.zero_grad(set_to_none=True)
        with _amp_context(device, amp_enabled):
            pred, aux = model(batch["x"], batch["context"], mask=batch["mask"])
            loss = recil_loss(
                pred,
                batch["y"],
                batch["mask"],
                aux=aux,
                alpha_rank=args.alpha_rank,
                lambda_entropy=args.lambda_entropy,
                max_pairs_per_day=args.max_pairs_per_day,
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss: {loss.detach().cpu().item()}")
        scaler.scale(loss).backward()
        if args.grad_clip_norm and args.grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float(args.grad_clip_norm),
                error_if_nonfinite=True,
            )
            grad_norms.append(float(grad_norm.detach().cpu()))
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.detach().cpu())
        count += 1

    return {
        "loss": total / max(count, 1),
        "grad_norm": float(np.nanmean(grad_norms)) if grad_norms else float("nan"),
    }


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _stack_or_empty(items: list[np.ndarray], trailing_shape: tuple[int, ...]) -> np.ndarray:
    if not items:
        return np.empty((0,) + trailing_shape, dtype=np.float32)
    return np.concatenate(items, axis=0).astype(np.float32, copy=False)


def _normalized_entropy(weights: np.ndarray) -> float:
    if weights.ndim != 2 or weights.shape[0] == 0 or weights.shape[1] <= 1:
        return float("nan")
    clipped = np.clip(weights.astype(np.float64), 1e-12, 1.0)
    entropy = -(clipped * np.log(clipped)).sum(axis=1) / np.log(float(weights.shape[1]))
    return float(np.mean(entropy))


def _aux_diagnostics(aux: Dict[str, np.ndarray]) -> Dict[str, float]:
    diagnostics: Dict[str, float] = {}
    router = aux.get("router_weights")
    if router is not None and router.ndim == 2 and router.shape[0] > 0 and router.shape[1] > 0:
        diagnostics.update(
            {
                "router_entropy_norm": _normalized_entropy(router),
                "router_max_mean": float(np.mean(np.max(router, axis=1))),
                "router_std": float(np.std(router)),
            }
        )
    scale = aux.get("scale_weights")
    if scale is not None and scale.ndim == 2 and scale.shape[0] > 0 and scale.shape[1] > 0:
        diagnostics.update(
            {
                "scale_entropy_norm": _normalized_entropy(scale),
                "scale_max_mean": float(np.mean(np.max(scale, axis=1))),
                "scale_std": float(np.std(scale)),
            }
        )
    gate = aux.get("context_gate")
    if gate is not None and gate.size > 0:
        diagnostics.update(
            {
                "context_gate_mean": float(np.mean(gate)),
                "context_gate_std": float(np.std(gate)),
                "context_gate_min": float(np.min(gate)),
                "context_gate_max": float(np.max(gate)),
            }
        )
    return diagnostics


@torch.no_grad()
def evaluate_loader(model, loader, device, args):
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
    non_blocking = device.type == "cuda"
    amp_enabled = bool(args.amp and device.type == "cuda")

    for raw_batch in loader:
        batch = batch_to_device(raw_batch, device, non_blocking=non_blocking)
        with _amp_context(device, amp_enabled):
            pred, aux = model(batch["x"], batch["context"], mask=batch["mask"])
            loss = recil_loss(
                pred,
                batch["y"],
                batch["mask"],
                aux=aux,
                alpha_rank=args.alpha_rank,
                lambda_entropy=args.lambda_entropy,
                max_pairs_per_day=args.max_pairs_per_day,
            )
        total_loss += float(loss.detach().cpu())
        count += 1
        preds.append(_to_numpy(pred))
        labels.append(_to_numpy(batch["y"]))
        masks.append(_to_numpy(batch["mask"]))
        contexts.append(_to_numpy(batch["context"]))
        if aux.get("scale_weights") is not None:
            scale_weights.append(_to_numpy(aux["scale_weights"]))
        if aux.get("router_weights") is not None:
            router_weights.append(_to_numpy(aux["router_weights"]))
        gate_tensor = aux.get("context_gate")
        if gate_tensor is None:
            gate_tensor = aux.get("residual_gate")
        if gate_tensor is not None:
            context_gates.append(_to_numpy(gate_tensor))

    pred_arr = np.concatenate(preds, axis=0)
    label_arr = np.concatenate(labels, axis=0)
    mask_arr = np.concatenate(masks, axis=0)
    context_arr = np.concatenate(contexts, axis=0)
    aux = {
        "scale_weights": _stack_or_empty(scale_weights, (0,)),
        "router_weights": _stack_or_empty(router_weights, (0,)),
        "context_gate": _stack_or_empty(context_gates, (0,)),
    }
    metrics = evaluate_predictions(
        pred_arr,
        label_arr,
        mask_arr,
        k=10,
        transaction_cost_bps=args.transaction_cost_bps,
        include_diagnostics=True,
        include_original_metrics=True,
    )
    metrics.update(_aux_diagnostics(aux))
    return {
        "loss": total_loss / max(count, 1),
        "metrics": metrics,
        "predictions": pred_arr.astype(np.float32),
        "labels": label_arr.astype(np.float32),
        "masks": mask_arr.astype(np.float32),
        "contexts": context_arr.astype(np.float32),
        "aux": aux,
    }


def best_score(val_metrics: Dict[str, Any], val_loss: float) -> float:
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
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(clean_json(payload), indent=2, sort_keys=True), encoding="utf-8")


def save_checkpoint(path: Path, model, optimizer, epoch: int, config: Dict[str, Any], val_result, test_result, score: float) -> None:
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


def write_train_log(path: Path, rows: list[Dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in LOG_FIELDS})


def _metric(metrics: Dict[str, Any], key: str) -> Any:
    return metrics.get(key, "")


def _safe_run_tag(value: str) -> str:
    tag = str(value or "").strip()
    if not tag:
        return ""
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in tag)
    return safe.strip("._-")


def interaction_strength_for_epoch(epoch: int, warmup_epochs: int) -> float:
    warmup_epochs = int(warmup_epochs)
    if warmup_epochs <= 0:
        return 1.0
    return min(1.0, max(0.0, float(epoch) / float(warmup_epochs)))


def run(args) -> Path:
    set_deterministic(args.seed, strict=args.strict_deterministic)
    device = resolve_device(args.device)
    dataset_name, datasets = prepare_data(args)
    generator = torch.Generator().manual_seed(args.seed)
    pin_memory = device.type == "cuda"
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
    }
    train_loader = DataLoader(datasets["train"], shuffle=True, generator=generator, **loader_kwargs)
    val_loader = DataLoader(datasets["val"], shuffle=False, **loader_kwargs)
    test_loader = DataLoader(datasets["test"], shuffle=False, **loader_kwargs)

    sample = datasets["train"][0]
    model = ReCILMixer(
        num_assets=sample["x"].shape[0],
        num_features=sample["x"].shape[-1],
        d_model=args.d_model,
        context_dim=sample["context"].shape[0],
        market_dim=args.market_dim,
        num_experts=args.num_experts,
        rank=args.rank if args.rank and args.rank > 0 else None,
        scales=args.scales,
        dropout=args.dropout,
        variant=args.variant,
        scale_gate_temperature=args.scale_gate_temperature,
        router_init=args.router_init,
        router_temperature=args.router_temperature,
        gate_init=args.gate_init,
        mask_invalid_assets=args.mask_invalid_assets,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = _make_grad_scaler(enabled=bool(args.amp and device.type == "cuda"))

    config = vars(args).copy()
    config.update(
        {
            "resolved_dataset": dataset_name,
            "device_resolved": str(device),
            "amp_resolved": bool(args.amp and device.type == "cuda"),
            "optimizer": "AdamW",
            "parameter_report": model.parameter_report(),
        }
    )
    safe_tag = _safe_run_tag(args.run_tag)
    variant_dir = args.variant if not safe_tag else f"{args.variant}__{safe_tag}"
    run_dir = Path(args.output_dir) / dataset_name / variant_dir / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(run_dir / "config.json", config)

    best = float("-inf")
    best_test = None
    bad_epochs = 0
    log_rows: list[Dict[str, Any]] = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, args.epochs + 1):
        start_time = time.perf_counter()
        model.set_interaction_strength(interaction_strength_for_epoch(epoch, args.interaction_warmup_epochs))
        train_stats = train_one_epoch(model, train_loader, optimizer, device, args, scaler)
        val_result = evaluate_loader(model, val_loader, device, args)
        test_result = evaluate_loader(model, test_loader, device, args)
        epoch_sec = time.perf_counter() - start_time
        score = best_score(val_result["metrics"], val_result["loss"])
        if score > best:
            best = score
            best_test = test_result
            bad_epochs = 0
            save_checkpoint(run_dir / "best_model.pt", model, optimizer, epoch, config, val_result, test_result, score)
        else:
            bad_epochs += 1
        save_checkpoint(run_dir / "last_model.pt", model, optimizer, epoch, config, val_result, test_result, score)

        log_rows.append(
            {
                "epoch": epoch,
                "epoch_sec": epoch_sec,
                "train_loss": train_stats["loss"],
                "train_grad_norm": train_stats["grad_norm"],
                "val_loss": val_result["loss"],
                "test_loss": test_result["loss"],
                "val_RankIC": _metric(val_result["metrics"], "RankIC"),
                "val_RankICIR": _metric(val_result["metrics"], "RankICIR"),
                "val_IC": _metric(val_result["metrics"], "IC"),
                "val_ICIR": _metric(val_result["metrics"], "ICIR"),
                "val_Precision@10": _metric(val_result["metrics"], "Precision@10"),
                "test_RankIC": _metric(test_result["metrics"], "RankIC"),
                "test_RankICIR": _metric(test_result["metrics"], "RankICIR"),
                "test_IC": _metric(test_result["metrics"], "IC"),
                "test_ICIR": _metric(test_result["metrics"], "ICIR"),
                "test_Precision@10": _metric(test_result["metrics"], "Precision@10"),
                "test_Sharpe@10": _metric(test_result["metrics"], "Sharpe@10"),
                "test_Turnover@10": _metric(test_result["metrics"], "Turnover@10"),
                "test_CostSharpe@10": _metric(test_result["metrics"], "CostSharpe@10"),
                "best_score": best,
                "bad_epochs": bad_epochs,
            }
        )
        if args.patience and args.patience > 0 and bad_epochs >= args.patience:
            break

    final_test = best_test if best_test is not None else evaluate_loader(model, test_loader, device, args)
    write_train_log(run_dir / "train_log.csv", log_rows)
    peak_gpu_memory_mb = None
    if device.type == "cuda":
        peak_gpu_memory_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
    save_json(
        run_dir / "metrics.json",
        {
            "test": final_test["metrics"],
            "best_score": best,
            "parameter_report": model.parameter_report(),
            "peak_gpu_memory_mb": peak_gpu_memory_mb,
        },
    )
    np.save(run_dir / "predictions.npy", final_test["predictions"])
    np.save(run_dir / "labels.npy", final_test["labels"])
    np.save(run_dir / "masks.npy", final_test["masks"])
    np.save(run_dir / "contexts.npy", final_test["contexts"])
    np.savez(run_dir / "aux_outputs.npz", **final_test["aux"])
    return run_dir


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_dir = run(args)
    print(f"saved ReCIL run to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
