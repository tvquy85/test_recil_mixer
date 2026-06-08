# Stage 1 Small Scale Audit

Date: 2026-06-08

Decision: pass

Recommended next stage: `stage_2_medium_scale`

Stage 1 ran the first non-quick-test real-data ReCIL experiment on NASDAQ with
two variants, `static` and `full`, seed `0`, and 5 epochs. This stage verifies
that the full NASDAQ split, checkpointing, metrics, analysis, and audit
artifacts work beyond the four-day smoke runs. It is still a stability gate,
not paper-quality evidence.

## Environment

- Python: `/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe`
- CUDA: available
- GPU used by training: `cuda`
- GPU: NVIDIA GeForce RTX 3090
- Runner mode: sequential, `--max-parallel 1` default
- Scope: NASDAQ only, variants `static` and `full`, seed `0`, 5 epochs

## Preflight

| Step | Command | Status | Rough runtime |
|---|---|---:|---:|
| compileall | `/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe -m compileall src/recil` | pass | <0.1s |
| pytest | `/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe -m pytest tests/recil -q` | pass, 74 tests, 1 known warning | 12.8s |

The warning is expected: efficiency analysis records that `train_log.csv` does
not yet contain wall-clock epoch time, so `time_per_epoch_sec` is blank.

## Commands

Training command:

```bash
/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe -m src.recil.run_experiments \
  --datasets nasdaq \
  --variants static full \
  --seeds 0 \
  --epochs 5 \
  --output-dir outputs_stage1
```

Analysis command:

```bash
/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe -m src.recil.analysis \
  --output-dir outputs_stage1 \
  --aggregate --regime --interpretability --efficiency --make-tables
```

## Manifest

`outputs_stage1/run_manifest.jsonl` contains exactly two rows.

| Dataset | Variant | Seed | Status | Return code | Duration |
|---|---|---:|---:|---:|---:|
| nasdaq | static | 0 | pass | 0 | 70.64s |
| nasdaq | full | 0 | pass | 0 | 89.28s |

No failed command remains.

## Required Outputs

Run directories:

```text
outputs_stage1/nasdaq/static/seed_0
outputs_stage1/nasdaq/full/seed_0
```

Each run contains:

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

Analysis outputs:

```text
outputs_stage1/results_summary.csv
outputs_stage1/results_summary_mean_std.csv
outputs_stage1/regime_results.csv
outputs_stage1/interpretability_correlations.csv
outputs_stage1/efficiency_table.csv
paper_tables/main_results_latex.tex
paper_tables/ablation_latex.tex
paper_tables/regime_latex.tex
paper_tables/efficiency_latex.tex
figures/nasdaq_full_seed0_router_weights.png
figures/nasdaq_full_seed0_scale_weights.png
```

Checkpoint loadability was verified with
`torch.load(..., map_location="cpu", weights_only=False)`. Each best checkpoint
contains `model_state_dict`.

## Metric And Artifact Checks

Command:

```bash
rg -n "\"RIC\"|\bRIC\b" src/recil tests/recil outputs_stage1 || true
```

Result: no matches. Stage 1 uses clean ReCIL metric names only.

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
`masks.npy`, `contexts.npy`, and numeric metrics for both runs.

| Variant | Test days | Best epoch | mse | IC | RankIC | ICIR | Precision@10 | Sharpe |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| static | 237 | 5 | 0.000402 | 0.030487 | 0.029565 | 0.309466 | 0.058228 | 1.211536 |
| full | 237 | 4 | 0.000633 | 0.026248 | 0.028735 | 0.266454 | 0.049789 | 0.989338 |

These one-seed, five-epoch results are stability signals only. They should not
be used as final model-quality claims.

## Shapes And Aux Outputs

| Variant | Predictions | Contexts | Aux outputs |
|---|---:|---:|---|
| static | `(237, 1026)` | `(237, 7)` | scale `(0, 0)`, router `(0, 0)`, gate `(0, 0)` |
| full | `(237, 1026)` | `(237, 7)` | scale `(237, 3)`, router `(237, 4)`, gate `(237, 64)` |

The full variant produced interpretable scale/router/gate arrays for all 237
test days.

## Efficiency Snapshot

`outputs_stage1/efficiency_table.csv` was created.

| Variant | Device | Parameters | Epochs logged | RankIC |
|---|---|---:|---:|---:|
| static | cuda | 474710 | 5 | 0.029565 |
| full | cuda | 474710 | 5 | 0.028735 |

`time_per_epoch_sec` and `gpu_memory_peak` remain blank because the current
training log does not record wall-clock epoch timing or peak memory.

## Pass Gate

Stage 1 pass criteria:

- Both NASDAQ runs completed with `status=pass`: pass.
- `run_manifest.jsonl` has exactly two rows and return code `0`: pass.
- Required training artifacts are present: pass.
- Required analysis CSVs and paper-table snippets are present: pass.
- Clean metric keys are used and no `RIC` appears in Stage 1 ReCIL outputs:
  pass.
- Predictions, labels, masks, contexts, and metrics are finite: pass.
- Checkpoints load and include `model_state_dict`: pass.

## Known Limitations

- Stage 1 uses only one seed and two variants, so it is not sufficient for
  paper claims or statistical comparisons.
- The `full` variant did not outperform `static` in this short run; this is a
  reason to continue controlled ablation, not a conclusion about the method.
- Training logs do not yet capture epoch wall-clock time or peak GPU memory.
- Sharpe remains a diagnostic metric without transaction costs, turnover,
  slippage, or capacity modeling.
- NYSE remains blocked because local audited NYSE files are zero bytes.

## Next Step

Proceed to planning `stage_2_medium_scale`: NASDAQ all five variants
(`static`, `context_only`, `single_gate`, `moe`, `full`), seeds `0 1`, and 30
epochs initially. Do not run Stage 3 or paper-level claims until Stage 2 has its
own pass audit.
