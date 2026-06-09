# ReCIL model/pipeline optimization patch

This archive is designed to be extracted at the root of `test_recil_mixer`.
It overwrites selected files under `StockMixer/src/recil` and adds a regression
test under `StockMixer/tests/recil`.

## Changed files

- `StockMixer/src/recil/modules.py`
  - Adds `apply_asset_mask` and propagates asset masks through hidden tensors.
  - Makes asset-axis low-rank experts mask-aware.
  - Initializes scale routers uniformly and residual gates near the static path.
  - Keeps the original tensor convention `[B, N, T, F]`.

- `StockMixer/src/recil/model.py`
  - Instantiates only the modules required by the selected variant.
  - Adds `parameter_report()`, `active_parameter_count`, and `all_parameter_count`.
  - Applies masks before/after temporal and stock mixing and zeros invalid predictions.
  - Maintains backward compatibility with `num_features` and original variant names.

- `StockMixer/src/recil/losses.py`
  - Keeps masked MSE + pairwise ranking loss.
  - Exposes router entropy regularization through `lambda_entropy`.
  - Adds finite/shape checks.

- `StockMixer/src/recil/metrics.py`
  - Adds `RankICIR`.
  - Adds top-k `Return`, `Turnover`, `CostReturn`, and `CostSharpe` diagnostics.
  - Keeps the backward-compatible `Sharpe` alias.

- `StockMixer/src/recil/train_recil.py`
  - Uses `AdamW` with configurable `--weight-decay`.
  - Adds `--lambda-entropy`, `--grad-clip-norm`, `--amp`, `--patience`,
    `--transaction-cost-bps`, `--num-workers`, `--strict-deterministic`.
  - Logs epoch time, gradient norm, active parameter report, and peak GPU memory.

- `StockMixer/src/recil/run_experiments.py`
  - Forwards the new training/tuning options to every subprocess.
  - Writes `experiment_manifest.json` with exact commands.

- `StockMixer/tests/recil/test_model_pipeline_optimizations.py`
  - Checks variant-specific parameter counts.
  - Checks invalid assets cannot alter valid predictions.
  - Checks new metrics are emitted.

## Apply

```bash
unzip recil_optimization_patch.zip -d /path/to/test_recil_mixer
cd /path/to/test_recil_mixer/StockMixer
```

## Smoke checks

```bash
PYTHONPATH=. python -m py_compile \
  src/recil/modules.py src/recil/model.py src/recil/metrics.py \
  src/recil/losses.py src/recil/train_recil.py src/recil/run_experiments.py

PYTHONPATH=. python -m pytest -q tests/recil/test_model_pipeline_optimizations.py
```

## Synthetic quick run

```bash
cd StockMixer
PYTHONPATH=. python -m src.recil.train_recil \
  --synthetic --quick-test \
  --variant single_gate \
  --epochs 2 \
  --output-dir outputs_patch_smoke \
  --alpha-rank 0.3 \
  --weight-decay 1e-4 \
  --grad-clip-norm 1.0
```

## Core Stage 3 command template

```bash
cd StockMixer
PYTHONPATH=. python -m src.recil.run_experiments \
  --datasets nasdaq sp500 \
  --variants static context_only single_gate \
  --seeds 0 1 2 \
  --epochs 50 \
  --alpha-rank 0.3 \
  --weight-decay 1e-4 \
  --grad-clip-norm 1.0 \
  --transaction-cost-bps 10 \
  --output-dir outputs_stage3_patch_core
```
