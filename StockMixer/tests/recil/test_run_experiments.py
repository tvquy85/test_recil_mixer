import subprocess
import sys

import pytest

from src.recil.run_experiments import build_arg_parser, build_commands


def test_build_commands_deterministic_order_and_flags(tmp_path):
    args = build_arg_parser().parse_args(
        [
            "--datasets",
            "nasdaq",
            "--variants",
            "static",
            "full",
            "--seeds",
            "0",
            "1",
            "--epochs",
            "3",
            "--quick-test",
            "--device",
            "cpu",
            "--router-init",
            "small_normal",
            "--router-temperature",
            "0.5",
            "--interaction-warmup-epochs",
            "5",
            "--run-tag",
            "ne2_temp05",
            "--output-dir",
            str(tmp_path),
        ]
    )
    commands = build_commands(args)
    assert [(c["dataset"], c["variant"], c["seed"]) for c in commands] == [
        ("nasdaq", "static", 0),
        ("nasdaq", "static", 1),
        ("nasdaq", "full", 0),
        ("nasdaq", "full", 1),
    ]
    for spec in commands:
        command = spec["command"]
        assert command[1:3] == ["-m", "src.recil.train_recil"]
        assert "--quick-test" in command
        assert "--epochs" in command and "3" in command
        assert "--output-dir" in command and str(tmp_path) in command
        assert "--device" in command and "cpu" in command
        assert "--router-init" in command and "small_normal" in command
        assert "--router-temperature" in command and "0.5" in command
        assert "--interaction-warmup-epochs" in command and "5" in command
        assert "--run-tag" in command and "ne2_temp05" in command


def test_dry_run_prints_commands_and_creates_no_outputs(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.recil.run_experiments",
            "--datasets",
            "nasdaq",
            "--variants",
            "static",
            "full",
            "--seeds",
            "0",
            "1",
            "--epochs",
            "1",
            "--quick-test",
            "--output-dir",
            str(tmp_path),
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert len(lines) == 4
    assert "--dataset nasdaq --variant static --seed 0" in lines[0]
    assert "--dataset nasdaq --variant static --seed 1" in lines[1]
    assert "--dataset nasdaq --variant full --seed 0" in lines[2]
    assert "--dataset nasdaq --variant full --seed 1" in lines[3]
    assert "--quick-test" in result.stdout
    assert not (tmp_path / "run_manifest.jsonl").exists()
    assert not (tmp_path / "nasdaq").exists()


def test_invalid_variant_exits_nonzero():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.recil.run_experiments",
            "--variants",
            "bad_variant",
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "invalid choice" in result.stderr


def test_invalid_runner_values_fail_fast():
    args = build_arg_parser().parse_args(["--epochs", "0", "--dry-run"])
    with pytest.raises(ValueError):
        build_commands(args)
