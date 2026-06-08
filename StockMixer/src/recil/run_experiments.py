"""Safe experiment runner for ReCIL ablations."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from .model import VALID_VARIANTS


VALID_DATASETS = ("nasdaq", "sp500", "crypto")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or dry-run staged ReCIL ablations")
    parser.add_argument("--datasets", nargs="+", default=["nasdaq"], choices=VALID_DATASETS)
    parser.add_argument("--variants", nargs="+", default=["static", "context_only", "single_gate", "moe", "full"], choices=sorted(VALID_VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quick-test", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--keep-going", action="store_true")
    return parser


def validate_args(args) -> None:
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.max_parallel < 1:
        raise ValueError("--max-parallel must be at least 1")
    if any(seed < 0 for seed in args.seeds):
        raise ValueError("--seeds must be non-negative")


def build_commands(args) -> list[dict]:
    validate_args(args)
    commands = []
    for dataset in args.datasets:
        for variant in args.variants:
            for seed in args.seeds:
                command = [
                    args.python,
                    "-m",
                    "src.recil.train_recil",
                    "--dataset",
                    dataset,
                    "--variant",
                    variant,
                    "--seed",
                    str(seed),
                    "--epochs",
                    str(args.epochs),
                    "--output-dir",
                    args.output_dir,
                ]
                if args.quick_test:
                    command.append("--quick-test")
                if args.device != "auto":
                    command.extend(["--device", args.device])
                commands.append({"dataset": dataset, "variant": variant, "seed": int(seed), "command": command})
    return commands


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_manifest_row(manifest_path: Path, row: dict) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def run_commands(args, commands: list[dict]) -> int:
    if args.max_parallel > 1:
        raise NotImplementedError("--max-parallel > 1 is intentionally not implemented for safe GPU scheduling")

    manifest_path = Path(args.output_dir) / "run_manifest.jsonl"
    exit_code = 0
    for spec in commands:
        start_time = utc_now()
        start = time.time()
        result = subprocess.run(spec["command"], check=False)
        end_time = utc_now()
        duration = time.time() - start
        status = "pass" if result.returncode == 0 else "fail"
        row = {
            "dataset": spec["dataset"],
            "variant": spec["variant"],
            "seed": spec["seed"],
            "command": spec["command"],
            "start_time": start_time,
            "end_time": end_time,
            "duration_sec": duration,
            "status": status,
            "return_code": int(result.returncode),
        }
        write_manifest_row(manifest_path, row)
        if result.returncode != 0:
            exit_code = int(result.returncode)
            if not args.keep_going:
                break
    return exit_code


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        commands = build_commands(args)
        if args.dry_run:
            for spec in commands:
                print(" ".join(spec["command"]))
            return 0
        return run_commands(args, commands)
    except (ValueError, NotImplementedError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
