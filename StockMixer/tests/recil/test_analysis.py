import csv
import json
import subprocess
import sys

import numpy as np
import torch

from src.recil.analysis import (
    aggregate_results,
    efficiency_analysis,
    interpretability_analysis,
    regime_analysis,
)


def _write_run(root, dataset, variant, seed, aux=True):
    run_dir = root / dataset / variant / f"seed_{seed}"
    run_dir.mkdir(parents=True)
    config = {
        "dataset": dataset,
        "resolved_dataset": dataset,
        "variant": variant,
        "seed": seed,
        "d_model": 8,
        "market_dim": 4,
        "num_experts": 2,
        "device_resolved": "cpu",
    }
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    metrics = {
        "best_score": 0.1,
        "test": {
            "mse": 0.01,
            "IC": 0.2 + seed,
            "RankIC": 0.3 + seed,
            "ICIR": 1.0,
            "Precision@10": 0.5,
            "Sharpe": 2.0,
            "num_valid_days": 4,
            "num_days": 4,
        },
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    with (run_dir / "train_log.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss"])
        writer.writeheader()
        writer.writerow({"epoch": 1, "train_loss": 0.1})
    torch.save({"epoch": 1, "model_state_dict": {}}, run_dir / "best_model.pt")

    preds = np.array(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.4, 0.3, 0.2, 0.1],
            [0.1, 0.3, 0.2, 0.4],
            [0.4, 0.1, 0.3, 0.2],
        ],
        dtype=np.float32,
    )
    labels = np.array(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4],
            [0.4, 0.3, 0.2, 0.1],
            [0.4, 0.3, 0.2, 0.1],
        ],
        dtype=np.float32,
    )
    masks = np.ones_like(preds, dtype=np.float32)
    contexts = np.array(
        [
            [0.0, -0.1, 0.1, 0.2, 0.3, 0.0, 0.0],
            [0.0, 0.2, 0.2, 0.3, 0.4, 0.0, 0.0],
            [0.0, -0.3, 0.3, 0.4, 0.5, 0.0, 0.0],
            [0.0, 0.4, 0.4, 0.5, 0.6, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    np.save(run_dir / "predictions.npy", preds)
    np.save(run_dir / "labels.npy", labels)
    np.save(run_dir / "masks.npy", masks)
    np.save(run_dir / "contexts.npy", contexts)
    if aux:
        np.savez(
            run_dir / "aux_outputs.npz",
            scale_weights=np.array([[0.8, 0.2], [0.7, 0.3], [0.2, 0.8], [0.1, 0.9]], dtype=np.float32),
            router_weights=np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]], dtype=np.float32),
            context_gate=np.ones((4, 8), dtype=np.float32),
        )
    else:
        np.savez(
            run_dir / "aux_outputs.npz",
            scale_weights=np.empty((0, 0), dtype=np.float32),
            router_weights=np.empty((0, 0), dtype=np.float32),
            context_gate=np.empty((0, 0), dtype=np.float32),
        )
    return run_dir


def _read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_aggregate_results_writes_rows_and_mean_std(tmp_path):
    _write_run(tmp_path, "nasdaq", "static", 0, aux=False)
    _write_run(tmp_path, "nasdaq", "full", 0, aux=True)
    rows = aggregate_results(tmp_path)
    assert len(rows) == 2
    summary = _read_csv(tmp_path / "results_summary.csv")
    mean_std = _read_csv(tmp_path / "results_summary_mean_std.csv")
    assert {row["variant"] for row in summary} == {"static", "full"}
    assert len(mean_std) == 2
    assert "RankIC_mean" in mean_std[0]


def test_regime_analysis_writes_regime_rows(tmp_path):
    _write_run(tmp_path, "nasdaq", "full", 0, aux=True)
    rows = regime_analysis(tmp_path)
    assert rows
    regimes = {(row["regime_name"], row["regime_side"]) for row in rows}
    assert ("market_volatility", "low") in regimes
    assert ("market_volatility", "high") in regimes
    assert ("market_trend", "up_or_flat") in regimes
    assert (tmp_path / "regime_results.csv").exists()


def test_interpretability_handles_non_empty_and_empty_aux(tmp_path):
    _write_run(tmp_path, "nasdaq", "static", 0, aux=False)
    _write_run(tmp_path, "nasdaq", "full", 0, aux=True)
    rows = interpretability_analysis(tmp_path)
    assert any(row["signal"] == "scale_0_weight" for row in rows)
    assert any(row["signal"] == "expert_0_weight" for row in rows)
    assert (tmp_path / "interpretability_correlations.csv").exists()


def test_efficiency_reconstructs_model_and_counts_params(tmp_path):
    _write_run(tmp_path, "nasdaq", "full", 0, aux=True)
    rows = efficiency_analysis(tmp_path)
    assert rows
    assert int(rows[0]["num_params"]) > 0
    assert rows[0]["train_log_epochs"] == 1
    assert (tmp_path / "efficiency_table.csv").exists()


def test_analysis_cli_aggregate(tmp_path):
    _write_run(tmp_path, "nasdaq", "full", 0, aux=True)
    result = subprocess.run(
        [sys.executable, "-m", "src.recil.analysis", "--output-dir", str(tmp_path), "--aggregate"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "results_summary.csv").exists()
