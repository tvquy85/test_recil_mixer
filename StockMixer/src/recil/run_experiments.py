"""Grid runner for ReCIL experiments.

This version keeps the original subprocess-based workflow but forwards the new
training controls needed for serious ablation/tuning runs.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, List

from .model import VALID_VARIANTS


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ReCIL experiment grids")
    parser.add_argument("--datasets", nargs="+", default=["nasdaq"], choices=["nasdaq", "sp500", "crypto"])
    parser.add_argument("--variants", nargs="+", default=sorted(VALID_VARIANTS), choices=sorted(VALID_VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--market-dim", type=int, default=32)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--lookback", type=int, default=16)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--scales", type=int, nargs="+", default=[1, 2, 4])
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
    parser.add_argument("--interaction-warmup-epochs", type=int, default=0)
    parser.add_argument(
        "--no-mask-invalid-assets",
        dest="mask_invalid_assets",
        action="store_false",
        default=True,
        help="forward model-side mask isolation flag to train_recil",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--strict-deterministic", action="store_true")
    parser.add_argument("--quick-test", action="store_true")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--python", default=sys.executable)
    return parser


def _append_flag(cmd: List[str], name: str, value) -> None:
    if value is None:
        return
    if isinstance(value, str) and value == "":
        return
    if isinstance(value, bool):
        if value:
            cmd.append(name)
        return
    if isinstance(value, (list, tuple)):
        cmd.append(name)
        cmd.extend(str(v) for v in value)
        return
    cmd.extend([name, str(value)])


def validate_args(args) -> None:
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if any(seed < 0 for seed in args.seeds):
        raise ValueError("--seeds must be non-negative")


def build_commands(args) -> List[dict]:
    validate_args(args)
    commands: List[dict] = []
    for dataset, variant, seed in itertools.product(args.datasets, args.variants, args.seeds):
        cmd = [
            args.python,
            "-m",
            "src.recil.train_recil",
            "--dataset",
            dataset,
            "--variant",
            variant,
            "--seed",
            str(seed),
            "--output-dir",
            args.output_dir,
        ]
        passthrough = {
            "--epochs": args.epochs,
            "--batch-size": args.batch_size,
            "--lr": args.lr,
            "--weight-decay": args.weight_decay,
            "--d-model": args.d_model,
            "--market-dim": args.market_dim,
            "--rank": args.rank,
            "--num-experts": args.num_experts,
            "--lookback": args.lookback,
            "--steps": args.steps,
            "--scales": args.scales,
            "--dropout": args.dropout,
            "--alpha-rank": args.alpha_rank,
            "--lambda-entropy": args.lambda_entropy,
            "--max-pairs-per-day": args.max_pairs_per_day,
            "--grad-clip-norm": args.grad_clip_norm,
            "--transaction-cost-bps": args.transaction_cost_bps,
            "--gate-init": args.gate_init,
            "--scale-gate-temperature": args.scale_gate_temperature,
            "--router-init": args.router_init,
            "--router-temperature": args.router_temperature,
            "--interaction-warmup-epochs": args.interaction_warmup_epochs,
            "--num-workers": args.num_workers,
            "--patience": args.patience,
            "--device": args.device,
            "--run-tag": args.run_tag,
        }
        for key, value in passthrough.items():
            _append_flag(cmd, key, value)
        _append_flag(cmd, "--amp", args.amp)
        _append_flag(cmd, "--strict-deterministic", args.strict_deterministic)
        _append_flag(cmd, "--quick-test", args.quick_test)
        _append_flag(cmd, "--synthetic", args.synthetic)
        if not args.mask_invalid_assets:
            cmd.append("--no-mask-invalid-assets")
        commands.append({"dataset": dataset, "variant": variant, "seed": int(seed), "command": cmd})
    return commands


def save_manifest(path: Path, commands: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"commands": [spec["command"] for spec in commands]}, indent=2), encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_run_manifest_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        commands = build_commands(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.dry_run:
        for spec in commands:
            print(" ".join(spec["command"]))
        return 0

    output_dir = Path(args.output_dir)
    save_manifest(output_dir / "experiment_manifest.json", commands)
    run_manifest_path = output_dir / "run_manifest.jsonl"
    for spec in commands:
        cmd = spec["command"]
        print(" ".join(cmd), flush=True)
        start_time = _utc_now()
        start = time.time()
        completed = subprocess.run(cmd, check=False)
        end_time = _utc_now()
        row = {
            "dataset": spec["dataset"],
            "variant": spec["variant"],
            "seed": spec["seed"],
            "command": cmd,
            "start_time": start_time,
            "end_time": end_time,
            "duration_sec": time.time() - start,
            "status": "pass" if completed.returncode == 0 else "fail",
            "return_code": int(completed.returncode),
        }
        _write_run_manifest_row(run_manifest_path, row)
        if completed.returncode != 0 and not args.keep_going:
            return int(completed.returncode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
