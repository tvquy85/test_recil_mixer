"""Post-run analysis utilities for ReCIL outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import warnings

import numpy as np
import torch

from .context import CONTEXT_FEATURE_NAMES
from .metrics import evaluate_predictions
from .model import ReCILMixer


METRIC_COLUMNS = (
    "IC",
    "RankIC",
    "ICIR",
    "Precision@10",
    "Sharpe",
    "OriginalIC",
    "OriginalICIR",
    "OriginalPositivePrecision@10",
    "OriginalSharpe@5",
    "OriginalMSE",
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze saved ReCIL experiment outputs")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--regime", action="store_true")
    parser.add_argument("--interpretability", action="store_true")
    parser.add_argument("--efficiency", action="store_true")
    parser.add_argument("--make-tables", action="store_true")
    return parser


def discover_runs(output_dir) -> list[dict]:
    root = Path(output_dir)
    runs = []
    for metrics_path in sorted(root.glob("*/*/seed_*/metrics.json")):
        run_dir = metrics_path.parent
        config_path = run_dir / "config.json"
        if not config_path.exists():
            warnings.warn(f"skipping {run_dir}: missing config.json", RuntimeWarning)
            continue
        seed_text = run_dir.name.removeprefix("seed_")
        try:
            seed = int(seed_text)
        except ValueError:
            warnings.warn(f"skipping {run_dir}: cannot parse seed", RuntimeWarning)
            continue
        runs.append(
            {
                "run_dir": run_dir,
                "dataset": run_dir.parent.parent.name,
                "variant": run_dir.parent.name,
                "seed": seed,
            }
        )
    return runs


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _finite_float(value):
    if value is None or value == "":
        return np.nan
    try:
        out = float(value)
    except (TypeError, ValueError):
        return np.nan
    return out if np.isfinite(out) else np.nan


def _safe_checkpoint_epoch(path: Path):
    if not path.exists():
        return ""
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # pragma: no cover - defensive for corrupt checkpoints.
        warnings.warn(f"could not read checkpoint {path}: {exc}", RuntimeWarning)
        return ""
    return checkpoint.get("epoch", "")


def aggregate_results(output_dir):
    root = Path(output_dir)
    rows = []
    for run in discover_runs(root):
        metrics = read_json(run["run_dir"] / "metrics.json").get("test", {})
        row = {
            "dataset": run["dataset"],
            "variant": run["variant"],
            "seed": run["seed"],
            "best_epoch": _safe_checkpoint_epoch(run["run_dir"] / "best_model.pt"),
        }
        for metric in METRIC_COLUMNS:
            row[metric] = metrics.get(metric, "")
        rows.append(row)

    fields = ["dataset", "variant", "seed", *METRIC_COLUMNS, "best_epoch"]
    write_csv(root / "results_summary.csv", rows, fields)

    grouped = {}
    for row in rows:
        grouped.setdefault((row["dataset"], row["variant"]), []).append(row)
    summary_rows = []
    for (dataset, variant), group_rows in sorted(grouped.items()):
        out = {"dataset": dataset, "variant": variant}
        for metric in METRIC_COLUMNS:
            values = np.asarray([_finite_float(row.get(metric)) for row in group_rows], dtype=np.float64)
            values = values[np.isfinite(values)]
            out[f"{metric}_mean"] = float(values.mean()) if values.size else ""
            out[f"{metric}_std"] = float(values.std(ddof=0)) if values.size else ""
        summary_rows.append(out)
    summary_fields = ["dataset", "variant"]
    for metric in METRIC_COLUMNS:
        summary_fields.extend([f"{metric}_mean", f"{metric}_std"])
    write_csv(root / "results_summary_mean_std.csv", summary_rows, summary_fields)
    return rows


def _load_required_arrays(run_dir: Path):
    arrays = {}
    for name in ("predictions", "labels", "masks", "contexts"):
        path = run_dir / f"{name}.npy"
        if not path.exists():
            warnings.warn(f"missing {path}", RuntimeWarning)
            return None
        arrays[name] = np.load(path)
    return arrays


def _metric_subset(preds, labels, masks):
    if preds.shape[0] == 0:
        return {"IC": np.nan, "RankIC": np.nan, "Precision@10": np.nan, "Sharpe": np.nan, "num_days": 0}
    metrics = evaluate_predictions(preds, labels, masks, k=10)
    return {
        "IC": metrics["IC"],
        "RankIC": metrics["RankIC"],
        "Precision@10": metrics["Precision@10"],
        "Sharpe": metrics["Sharpe"],
        "num_days": metrics["num_days"],
    }


def regime_analysis(output_dir):
    root = Path(output_dir)
    rows = []
    feature_index = {name: idx for idx, name in enumerate(CONTEXT_FEATURE_NAMES)}
    median_features = ("market_volatility", "pca_ratio", "cross_sectional_dispersion")
    for run in discover_runs(root):
        arrays = _load_required_arrays(run["run_dir"])
        if arrays is None:
            continue
        contexts = arrays["contexts"]
        split_specs = []
        for feature in median_features:
            values = contexts[:, feature_index[feature]]
            median = np.nanmedian(values)
            split_specs.append((feature, "low", values <= median))
            split_specs.append((feature, "high", values > median))
        trend = contexts[:, feature_index["market_trend"]]
        split_specs.append(("market_trend", "up_or_flat", trend >= 0.0))
        split_specs.append(("market_trend", "down", trend < 0.0))

        for regime_name, regime_side, selector in split_specs:
            selector = np.asarray(selector, dtype=bool)
            if not np.any(selector):
                continue
            metrics = _metric_subset(arrays["predictions"][selector], arrays["labels"][selector], arrays["masks"][selector])
            rows.append(
                {
                    "dataset": run["dataset"],
                    "variant": run["variant"],
                    "seed": run["seed"],
                    "regime_name": regime_name,
                    "regime_side": regime_side,
                    **metrics,
                }
            )
    fields = ["dataset", "variant", "seed", "regime_name", "regime_side", "IC", "RankIC", "Precision@10", "Sharpe", "num_days"]
    write_csv(root / "regime_results.csv", rows, fields)
    return rows


def pearson_corr_safe(x, y):
    x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[valid]
    y_arr = y_arr[valid]
    if x_arr.size < 2 or x_arr.std() <= 1e-12 or y_arr.std() <= 1e-12:
        return np.nan, int(x_arr.size)
    return float(np.corrcoef(x_arr, y_arr)[0, 1]), int(x_arr.size)


def _plot_timeseries(values, path: Path, title: str) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(values)
    ax.set_title(title)
    ax.set_xlabel("test day")
    ax.set_ylabel("weight")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def interpretability_analysis(output_dir):
    root = Path(output_dir)
    rows = []
    figures_dir = Path("figures")
    feature_index = {name: idx for idx, name in enumerate(CONTEXT_FEATURE_NAMES)}
    context_features = ("market_volatility", "pca_ratio", "cross_sectional_dispersion")
    plotted_any = False
    for run in discover_runs(root):
        arrays = _load_required_arrays(run["run_dir"])
        aux_path = run["run_dir"] / "aux_outputs.npz"
        if arrays is None or not aux_path.exists():
            warnings.warn(f"skipping interpretability for {run['run_dir']}: missing arrays or aux", RuntimeWarning)
            continue
        contexts = arrays["contexts"]
        aux = np.load(aux_path)
        scale_weights = aux["scale_weights"] if "scale_weights" in aux.files else np.empty((0, 0), dtype=np.float32)
        router_weights = aux["router_weights"] if "router_weights" in aux.files else np.empty((0, 0), dtype=np.float32)

        if scale_weights.ndim == 2 and scale_weights.shape[0] == contexts.shape[0] and scale_weights.shape[1] > 0:
            corr, n = pearson_corr_safe(scale_weights[:, 0], contexts[:, feature_index["market_volatility"]])
            rows.append(
                {
                    "dataset": run["dataset"],
                    "variant": run["variant"],
                    "seed": run["seed"],
                    "signal": "scale_0_weight",
                    "context_feature": "market_volatility",
                    "correlation": corr,
                    "num_days": n,
                }
            )
            plotted_any = _plot_timeseries(
                scale_weights,
                figures_dir / f"{run['dataset']}_{run['variant']}_seed{run['seed']}_scale_weights.png",
                f"{run['dataset']} {run['variant']} scale weights",
            ) or plotted_any

        if router_weights.ndim == 2 and router_weights.shape[0] == contexts.shape[0] and router_weights.shape[1] > 0:
            for expert_idx in range(router_weights.shape[1]):
                for feature in context_features:
                    corr, n = pearson_corr_safe(router_weights[:, expert_idx], contexts[:, feature_index[feature]])
                    rows.append(
                        {
                            "dataset": run["dataset"],
                            "variant": run["variant"],
                            "seed": run["seed"],
                            "signal": f"expert_{expert_idx}_weight",
                            "context_feature": feature,
                            "correlation": corr,
                            "num_days": n,
                        }
                    )
            plotted_any = _plot_timeseries(
                router_weights,
                figures_dir / f"{run['dataset']}_{run['variant']}_seed{run['seed']}_router_weights.png",
                f"{run['dataset']} {run['variant']} router weights",
            ) or plotted_any

    if not plotted_any:
        warnings.warn("matplotlib unavailable or no non-empty aux weights; wrote CSV-only interpretability outputs", RuntimeWarning)
    fields = ["dataset", "variant", "seed", "signal", "context_feature", "correlation", "num_days"]
    write_csv(root / "interpretability_correlations.csv", rows, fields)
    return rows


def _read_train_log_epochs(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def _read_mean_epoch_seconds(path: Path) -> float | str:
    if not path.exists():
        return ""
    values = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            value = _finite_float(row.get("epoch_sec"))
            if value == value:
                values.append(value)
    if not values:
        return ""
    return float(np.mean(values))


def _infer_num_params(run_dir: Path, config: dict) -> int | str:
    try:
        preds = np.load(run_dir / "predictions.npy")
        contexts = np.load(run_dir / "contexts.npy")
        num_assets = int(preds.shape[1])
        context_dim = int(contexts.shape[1])
        num_features = int(config.get("num_features", 5))
        model = ReCILMixer(
            num_assets=num_assets,
            num_features=num_features,
            d_model=int(config.get("d_model", 64)),
            context_dim=context_dim,
            market_dim=int(config.get("market_dim", 32)),
            num_experts=int(config.get("num_experts", 4)),
            variant=config.get("variant", run_dir.parent.name),
        )
        return int(sum(param.numel() for param in model.parameters()))
    except Exception as exc:
        warnings.warn(f"could not infer params for {run_dir}: {exc}", RuntimeWarning)
        return ""


def efficiency_analysis(output_dir):
    root = Path(output_dir)
    rows = []
    for run in discover_runs(root):
        config = read_json(run["run_dir"] / "config.json")
        metrics_payload = read_json(run["run_dir"] / "metrics.json")
        metrics = metrics_payload.get("test", {})
        train_log = run["run_dir"] / "train_log.csv"
        rows.append(
            {
                "dataset": run["dataset"],
                "variant": run["variant"],
                "seed": run["seed"],
                "device_resolved": config.get("device_resolved", ""),
                "num_params": _infer_num_params(run["run_dir"], config),
                "train_log_epochs": _read_train_log_epochs(train_log),
                "time_per_epoch_sec": _read_mean_epoch_seconds(train_log),
                "gpu_memory_peak": metrics_payload.get("peak_gpu_memory_mb", ""),
                "RankIC": metrics.get("RankIC", ""),
            }
        )
    fields = ["dataset", "variant", "seed", "device_resolved", "num_params", "train_log_epochs", "time_per_epoch_sec", "gpu_memory_peak", "RankIC"]
    write_csv(root / "efficiency_table.csv", rows, fields)
    return rows


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _format_value(value):
    x = _finite_float(value)
    if not np.isfinite(x):
        return ""
    return f"{x:.4f}"


def _latex_table(rows: list[dict], columns: list[str], metric_to_bold: str | None = None) -> str:
    best = None
    if metric_to_bold:
        values = [_finite_float(row.get(metric_to_bold)) for row in rows]
        finite = [value for value in values if np.isfinite(value)]
        best = max(finite) if finite else None
    lines = ["\\begin{tabular}{" + "l" * len(columns) + "}", " & ".join(columns) + " \\\\", "\\hline"]
    for row in rows:
        cells = []
        for col in columns:
            cell = _format_value(row.get(col)) if col not in {"dataset", "variant", "regime_name", "regime_side", "seed"} else str(row.get(col, ""))
            if best is not None and col == metric_to_bold and np.isclose(_finite_float(row.get(col)), best):
                cell = f"\\textbf{{{cell}}}"
            cells.append(cell)
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\end{tabular}\n")
    return "\n".join(lines)


def make_tables(output_dir):
    root = Path(output_dir)
    table_dir = Path("paper_tables")
    table_dir.mkdir(parents=True, exist_ok=True)
    summary = _read_csv(root / "results_summary.csv")
    mean_std = _read_csv(root / "results_summary_mean_std.csv")
    regime = _read_csv(root / "regime_results.csv")
    efficiency = _read_csv(root / "efficiency_table.csv")
    (table_dir / "main_results_latex.tex").write_text(
        _latex_table(
            summary,
            [
                "dataset",
                "variant",
                "seed",
                "IC",
                "RankIC",
                "ICIR",
                "Precision@10",
                "Sharpe",
                "OriginalIC",
                "OriginalICIR",
                "OriginalPositivePrecision@10",
                "OriginalSharpe@5",
            ],
            metric_to_bold="RankIC",
        ),
        encoding="utf-8",
    )
    (table_dir / "ablation_latex.tex").write_text(
        _latex_table(
            mean_std,
            [
                "dataset",
                "variant",
                "RankIC_mean",
                "IC_mean",
                "Precision@10_mean",
                "Sharpe_mean",
                "OriginalIC_mean",
                "OriginalICIR_mean",
                "OriginalPositivePrecision@10_mean",
                "OriginalSharpe@5_mean",
            ],
            metric_to_bold="RankIC_mean",
        ),
        encoding="utf-8",
    )
    (table_dir / "regime_latex.tex").write_text(
        _latex_table(regime[:40], ["dataset", "variant", "regime_name", "regime_side", "RankIC", "Precision@10", "num_days"], metric_to_bold="RankIC"),
        encoding="utf-8",
    )
    (table_dir / "efficiency_latex.tex").write_text(
        _latex_table(efficiency, ["dataset", "variant", "seed", "num_params", "train_log_epochs", "RankIC"], metric_to_bold="RankIC"),
        encoding="utf-8",
    )


def run(args) -> None:
    actions = [args.aggregate, args.regime, args.interpretability, args.efficiency, args.make_tables]
    run_all = not any(actions)
    if args.aggregate or run_all:
        aggregate_results(args.output_dir)
    if args.regime or run_all:
        regime_analysis(args.output_dir)
    if args.interpretability or run_all:
        interpretability_analysis(args.output_dir)
    if args.efficiency or run_all:
        efficiency_analysis(args.output_dir)
    if args.make_tables or run_all:
        if not (Path(args.output_dir) / "results_summary.csv").exists():
            aggregate_results(args.output_dir)
        if not (Path(args.output_dir) / "regime_results.csv").exists():
            regime_analysis(args.output_dir)
        if not (Path(args.output_dir) / "efficiency_table.csv").exists():
            efficiency_analysis(args.output_dir)
        make_tables(args.output_dir)


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
