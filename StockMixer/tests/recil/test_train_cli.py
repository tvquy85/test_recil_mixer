import json
import subprocess
import sys

import numpy as np
import torch


def test_train_recil_help_works():
    result = subprocess.run(
        [sys.executable, "-m", "src.recil.train_recil", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--synthetic" in result.stdout
    assert "--variant" in result.stdout
    assert "--output-dir" in result.stdout
    assert "--router-init" in result.stdout
    assert "--router-temperature" in result.stdout
    assert "--interaction-warmup-epochs" in result.stdout
    assert "--run-tag" in result.stdout


def test_synthetic_quick_run_creates_outputs(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.recil.train_recil",
            "--synthetic",
            "--variant",
            "full",
            "--epochs",
            "1",
            "--quick-test",
            "--device",
            "cpu",
            "--output-dir",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    run_dir = tmp_path / "synthetic" / "full" / "seed_0"
    required = {
        "config.json",
        "train_log.csv",
        "metrics.json",
        "best_model.pt",
        "last_model.pt",
        "predictions.npy",
        "labels.npy",
        "masks.npy",
        "contexts.npy",
        "aux_outputs.npz",
    }
    assert required == {path.name for path in run_dir.iterdir()}

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    test_metrics = metrics["test"]
    assert {"mse", "IC", "RankIC", "ICIR", "Precision@10", "Sharpe", "num_valid_days", "num_days"} <= set(test_metrics)
    assert {"router_entropy_norm", "router_max_mean", "router_std"} <= set(test_metrics)
    assert {"scale_entropy_norm", "context_gate_mean", "context_gate_std", "context_gate_min", "context_gate_max"} <= set(test_metrics)
    forbidden_key = "R" + "IC"
    assert forbidden_key not in test_metrics

    predictions = np.load(run_dir / "predictions.npy")
    labels = np.load(run_dir / "labels.npy")
    masks = np.load(run_dir / "masks.npy")
    contexts = np.load(run_dir / "contexts.npy")
    assert predictions.shape == labels.shape == masks.shape
    assert predictions.shape[0] == contexts.shape[0] == test_metrics["num_days"]
    assert np.isfinite(predictions).all()

    aux = np.load(run_dir / "aux_outputs.npz")
    assert aux["scale_weights"].shape[0] == predictions.shape[0]
    assert aux["router_weights"].shape[0] == predictions.shape[0]
    assert aux["context_gate"].shape[0] == predictions.shape[0]

    checkpoint = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=False)
    assert "model_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint


def test_run_tag_namespaces_output_without_changing_config_variant(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.recil.train_recil",
            "--synthetic",
            "--variant",
            "moe",
            "--epochs",
            "1",
            "--quick-test",
            "--device",
            "cpu",
            "--run-tag",
            "ne2 temp=0.5",
            "--output-dir",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    run_dir = tmp_path / "synthetic" / "moe__ne2_temp_0.5" / "seed_0"
    assert run_dir.exists()
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert config["variant"] == "moe"
