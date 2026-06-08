# Stage 2 Medium Scale Audit

Date: 2026-06-08

Decision: pass

Recommended next stage: plan `stage_3_full_scale` after reviewing the Stage 2
tradeoffs below.

Stage 2 ran the first medium-scale NASDAQ ablation: five ReCIL variants,
seeds `0 1`, and 30 epochs. The stage verifies multi-variant reproducibility,
clean metrics, saved artifacts, analysis outputs, and interpretability arrays.
It is stronger than Stage 1, but it is still NASDAQ-only and should not be
treated as final paper evidence.

## Environment

- Python: `/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe`
- CUDA available: yes
- GPU: NVIDIA GeForce RTX 3090
- Runner mode: sequential, `--max-parallel 1`
- Dataset: NASDAQ
- Variants: `static`, `context_only`, `single_gate`, `moe`, `full`
- Seeds: `0 1`
- Epochs: 30

## Preflight And Dry Run

| Step | Command | Status |
|---|---|---:|
| compileall | `/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe -m compileall src/recil` | pass |
| pytest | `/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe -m pytest tests/recil -q` | pass, 74 tests, 1 known warning |
| dry-run | `/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe -m src.recil.run_experiments --datasets nasdaq --variants static context_only single_gate moe full --seeds 0 1 --epochs 30 --output-dir outputs_stage2 --dry-run` | pass, 10 commands |

The pytest warning is expected: `analysis.py` records that `train_log.csv` does
not yet contain wall-clock epoch time, so `time_per_epoch_sec` is left blank.

## Commands

Training command:

```bash
/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe -m src.recil.run_experiments \
  --datasets nasdaq \
  --variants static context_only single_gate moe full \
  --seeds 0 1 \
  --epochs 30 \
  --output-dir outputs_stage2
```

Analysis command:

```bash
/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe -m src.recil.analysis \
  --output-dir outputs_stage2 \
  --aggregate --regime --interpretability --efficiency --make-tables
```

## Manifest

`outputs_stage2/run_manifest.jsonl` contains exactly 10 rows. All rows have
`status=pass` and `return_code=0`.

| Variant | Seed | Status | Runtime |
|---|---:|---:|---:|
| static | 0 | pass | 361.07s |
| static | 1 | pass | 356.91s |
| context_only | 0 | pass | 347.22s |
| context_only | 1 | pass | 362.18s |
| single_gate | 0 | pass | 426.13s |
| single_gate | 1 | pass | 455.58s |
| moe | 0 | pass | 481.37s |
| moe | 1 | pass | 483.49s |
| full | 0 | pass | 542.62s |
| full | 1 | pass | 521.94s |

## Required Outputs

Each run directory under `outputs_stage2/nasdaq/{variant}/seed_{seed}` contains:

```text
aux_outputs.npz
best_model.pt
config.json
contexts.npy
labels.npy
last_model.pt
masks.npy
metrics.json
predictions.npy
train_log.csv
```

Analysis outputs exist:

```text
outputs_stage2/results_summary.csv
outputs_stage2/results_summary_mean_std.csv
outputs_stage2/regime_results.csv
outputs_stage2/interpretability_correlations.csv
outputs_stage2/efficiency_table.csv
paper_tables/main_results_latex.tex
paper_tables/ablation_latex.tex
paper_tables/regime_latex.tex
paper_tables/efficiency_latex.tex
figures/nasdaq_moe_seed0_router_weights.png
figures/nasdaq_moe_seed1_router_weights.png
figures/nasdaq_full_seed0_router_weights.png
figures/nasdaq_full_seed0_scale_weights.png
figures/nasdaq_full_seed1_router_weights.png
figures/nasdaq_full_seed1_scale_weights.png
```

Checkpoint loadability was verified with
`torch.load(..., map_location="cpu", weights_only=False)`. Every best
checkpoint contains `model_state_dict`.

## Metric Name And Finite Checks

Command:

```bash
rg -n "\"RIC\"|\bRIC\b" src/recil tests/recil outputs_stage2 || true
```

Result: no matches. Stage 2 uses clean ReCIL metric names only.

Metric keys in every `metrics.json`:

```text
mse
IC
RankIC
ICIR
Precision@10
Sharpe
num_valid_days
num_days
```

Read-only inspection verified finite `predictions.npy`, `labels.npy`,
`masks.npy`, `contexts.npy`, and numeric metrics for all 10 runs.

## Per-Run Metrics

| Variant | Seed | Best epoch | RankIC | IC | Precision@10 | Sharpe | mse |
|---|---:|---:|---:|---:|---:|---:|---:|
| static | 0 | 6 | 0.031092 | 0.036135 | 0.038397 | -0.321750 | 0.000516 |
| static | 1 | 9 | 0.030934 | 0.032551 | 0.054008 | 0.376950 | 0.000405 |
| context_only | 0 | 10 | 0.032943 | 0.027136 | 0.051055 | 0.215171 | 0.000431 |
| context_only | 1 | 1 | 0.018994 | 0.014845 | 0.060338 | 0.800250 | 0.000586 |
| single_gate | 0 | 4 | 0.031345 | 0.026389 | 0.051899 | 0.654638 | 0.000613 |
| single_gate | 1 | 3 | 0.032759 | 0.028032 | 0.054430 | 1.101908 | 0.000393 |
| moe | 0 | 27 | 0.017249 | 0.020178 | 0.053165 | 1.283118 | 0.000440 |
| moe | 1 | 30 | 0.022297 | 0.024558 | 0.057806 | 1.078793 | 0.000407 |
| full | 0 | 30 | 0.015124 | 0.017867 | 0.041350 | 1.303740 | 0.000455 |
| full | 1 | 30 | 0.011406 | 0.016614 | 0.056962 | 1.798208 | 0.000402 |

## Mean And Standard Deviation

| Variant | RankIC mean | RankIC std | IC mean | Precision@10 mean | Sharpe mean |
|---|---:|---:|---:|---:|---:|
| static | 0.031013 | 0.000079 | 0.034343 | 0.046203 | 0.027600 |
| context_only | 0.025968 | 0.006975 | 0.020990 | 0.055696 | 0.507710 |
| single_gate | 0.032052 | 0.000707 | 0.027211 | 0.053165 | 0.878273 |
| moe | 0.019773 | 0.002524 | 0.022368 | 0.055485 | 1.180955 |
| full | 0.013265 | 0.001859 | 0.017240 | 0.049156 | 1.550974 |

## Interpretability And Efficiency

Auxiliary outputs are available as expected:

- `static`: empty scale/router/gate arrays.
- `context_only`: empty scale/router/gate arrays.
- `single_gate`: `context_gate` arrays with shape `(237, 64)` per seed.
- `moe`: `router_weights` arrays with shape `(237, 4)` per seed.
- `full`: `scale_weights (237, 3)`, `router_weights (237, 4)`, and
  `context_gate (237, 64)` per seed.

`outputs_stage2/interpretability_correlations.csv` has 50 rows and includes
router/scale correlations where those auxiliary arrays exist.

`outputs_stage2/efficiency_table.csv` has 10 rows. `time_per_epoch_sec` and
`gpu_memory_peak` remain blank because the current training log does not record
wall-clock epoch timing or peak GPU memory.

## Principal ML Scientist Notes

- `single_gate` is the strongest Stage 2 variant by `RankIC_mean`, and its
  two-seed variance is low.
- `static` remains competitive by `RankIC_mean` and is best by `IC_mean`.
- `moe` and `full` improve Sharpe-like portfolio diagnostics, but they are much
  weaker by `RankIC_mean`. This suggests the current gating/expert path may
  favor concentrated top-k returns over broad cross-sectional rank quality.
- `full` being worse than `single_gate` on RankIC is an ablation signal, not a
  final method failure. Before Stage 3 claims, router/scale behavior should be
  inspected and possibly compared against additional seeds or tuned
  regularization.

## Pass Gate

Stage 2 pass criteria:

- Preflight compile/tests passed: pass.
- Dry-run produced 10 commands: pass.
- All 10 training commands completed with `status=pass`: pass.
- Required run artifacts are present: pass.
- Required analysis outputs are present: pass.
- Clean metric keys are used and no `RIC` appears in Stage 2 ReCIL outputs:
  pass.
- Predictions, labels, masks, contexts, and metrics are finite: pass.
- Checkpoints load and include `model_state_dict`: pass.
- Rank-oriented metrics are available for every variant and seed: pass.

## Known Limitations

- Stage 2 is NASDAQ-only with two seeds; it is not yet a paper-final result.
- SP500 generalization is not tested in this stage.
- Training logs lack epoch wall-clock time and peak GPU memory.
- Sharpe is a diagnostic without transaction costs, turnover, slippage, or
  capacity modeling.
- NYSE remains blocked because local audited NYSE files are zero bytes.

## Next Step

Plan `stage_3_full_scale` only after reviewing the Stage 2 tradeoff. The
default Stage 3 should include NASDAQ and SP500, all five variants, seeds
`0 1 2`, and 50 epochs, but the plan should explicitly decide whether to:

1. keep all variants for completeness, or
2. prioritize `static`, `single_gate`, and `full` while running MoE/full
   interpretability diagnostics because `moe` and `full` currently lag on
   RankIC.

Do not write paper-level claims until Stage 3 has its own pass audit.
