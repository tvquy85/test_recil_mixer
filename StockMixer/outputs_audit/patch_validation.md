# Patch Validation Audit

Date: 2026-06-08 20:11:25 +07

Run ID: `patch_validation`

Scope: validate the partially applied ReCIL optimization patch from `PATCH_README.md` and `patch_code.md` before any Stage 3 expansion.

Decision: `blocked_for_stage3`

Engineering gate passed, but the scientific gate did not pass. The patch is runnable, test-clean, and produces finite artifacts; however, the Stage 2 patch reproduce substantially reduces NASDAQ RankIC for the previous strongest variants. Do not run Stage 3 until the ranking regression is investigated and fixed.

## Environment

- Interpreter: `/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe`
- Working directory: `StockMixer`
- Dataset: NASDAQ from local `StockMixer/dataset`
- Device policy: `--device auto`
- Resolved device in run configs: `cuda`
- Stage output: `outputs_stage2_patch_reproduce`
- Analysis outputs: `outputs_stage2_patch_reproduce/*.csv`, `paper_tables/*.tex`, `figures/*.png`

## Patch Compatibility Fixes Applied

- `losses.py`
  - Restored `[N]` and `[B, N]` loss support.
  - Preserved graph-connected zero for all-invalid masks and no valid ranking pairs.
  - Kept pairwise RankNet loss day-local and deterministic under pair subsampling.
- `metrics.py`
  - Preserved old evaluator contract: lowercase `mse`, `num_valid_days`, `num_days`, `asset_major`, `min_valid`.
  - Added diagnostic metrics: `RankICIR`, `Return@10`, `Turnover@10`, `CostReturn@10`, `CostSharpe@10`.
  - No legacy `RIC` key is emitted.
- `modules.py` / `model.py`
  - Preserved old constructor/forward behavior for Task 09-11 tests.
  - Kept public aux keys exactly: `scale_weights`, `router_weights`, `context_gate`.
  - Added variant-specific parameter reporting and mask-aware output behavior.
- `run_experiments.py`
  - Restored `--dry-run` and old `build_commands` test contract.
  - Kept patch forwarding options and real-run manifest writing.
- `train_recil.py`
  - Kept output artifact contract.
  - Added AdamW, weight decay, entropy option, grad clipping, AMP option, patience, transaction-cost diagnostics, epoch runtime, grad norm, parameter report, and GPU memory logging.
- `analysis.py`
  - Updated efficiency analysis to read `epoch_sec` from `train_log.csv` and `peak_gpu_memory_mb` from `metrics.json`.

## Commands And Results

| Step | Command | Result |
|---|---|---|
| Syntax | `/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe -m py_compile src/recil/modules.py src/recil/model.py src/recil/metrics.py src/recil/losses.py src/recil/train_recil.py src/recil/run_experiments.py tests/recil/test_model_pipeline_optimizations.py` | pass |
| Patch tests | `/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe -m pytest -q tests/recil/test_model_pipeline_optimizations.py` | pass, `3 passed` |
| Full regression | `/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe -m pytest tests/recil -q` | pass, `77 passed` |
| Synthetic smoke | `/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe -m src.recil.train_recil --synthetic --quick-test --variant single_gate --epochs 2 --output-dir outputs_patch_smoke --alpha-rank 0.3 --weight-decay 1e-4 --grad-clip-norm 1.0` | pass |
| Stage 2 reproduce | `/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe -m src.recil.run_experiments --datasets nasdaq --variants static context_only single_gate moe full --seeds 0 1 --epochs 30 --alpha-rank 0.3 --weight-decay 1e-4 --grad-clip-norm 1.0 --transaction-cost-bps 10 --output-dir outputs_stage2_patch_reproduce` | pass, 10/10 runs |
| Analysis | `/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe -m src.recil.analysis --output-dir outputs_stage2_patch_reproduce --aggregate --regime --interpretability --efficiency --make-tables` | pass |
| Artifact inspection | read-only Python inspection over all 10 run dirs | pass |
| Legacy metric scan | `rg -n "\"RIC\"|'RIC'" src/recil tests/recil outputs_stage2_patch_reproduce --glob '!*.pt' --glob '!*.npy' --glob '!*.npz'` | pass, no matches |

## Manifest Check

`outputs_stage2_patch_reproduce/run_manifest.jsonl` has 10 rows and every row has `status=pass`, `return_code=0`.

| Variant | Seed 0 Runtime | Seed 1 Runtime |
|---|---:|---:|
| static | 384.2s | 382.9s |
| context_only | 385.9s | 384.2s |
| single_gate | 415.6s | 415.3s |
| moe | 462.0s | 460.6s |
| full | 538.4s | 535.7s |

## Stage 2 Patch Metrics

Mean over seeds `0 1` on NASDAQ, 30 epochs.

| Variant | RankIC | RankICIR | IC | Precision@10 | Sharpe | Return@10 | Turnover@10 | CostReturn@10 | CostSharpe@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| static | 0.014719 | 0.138802 | 0.011003 | 0.016667 | 1.674755 | 0.001136 | 0.552743 | 0.000583 | 0.846598 |
| context_only | 0.007613 | 0.062170 | 0.004515 | 0.010338 | 1.286392 | 0.000566 | 0.631857 | -0.000066 | -0.145653 |
| single_gate | 0.013176 | 0.121529 | 0.009810 | 0.015823 | 2.323219 | 0.001484 | 0.548734 | 0.000936 | 1.452755 |
| moe | 0.016358 | 0.137641 | 0.013358 | 0.013291 | 1.467217 | 0.000986 | 0.588819 | 0.000397 | 0.602669 |
| full | 0.013263 | 0.114906 | 0.009301 | 0.015823 | 2.330023 | 0.001568 | 0.597257 | 0.000971 | 1.461690 |

Best patch RankIC: `moe`, `0.016358`.

Best patch cost-adjusted Sharpe diagnostic: `full`, `1.461690`, closely followed by `single_gate`, `1.452755`.

## Comparison To Previous Stage 2 Baseline

Previous baseline is `outputs_stage2/results_summary_mean_std.csv`.

| Variant | Old RankIC | Patch RankIC | Delta |
|---|---:|---:|---:|
| static | 0.031013 | 0.014719 | -0.016295 |
| context_only | 0.025968 | 0.007613 | -0.018355 |
| single_gate | 0.032052 | 0.013176 | -0.018876 |
| moe | 0.019773 | 0.016358 | -0.003415 |
| full | 0.013265 | 0.013263 | -0.000002 |

Interpretation:

- The previous best variant, `single_gate`, lost most of its RankIC advantage.
- `static` also lost substantial RankIC, so the issue is not isolated to context gating.
- `moe` is the best patch variant, but its RankIC remains far below the previous best `single_gate` baseline.
- `full` has strong Sharpe diagnostics but no RankIC improvement, so it should not be selected as the primary scientific model on this evidence.

## Artifact And Metric Invariants

All 10 run directories contain:

- `config.json`
- `metrics.json`
- `best_model.pt`
- `last_model.pt`
- `predictions.npy`
- `labels.npy`
- `masks.npy`
- `contexts.npy`
- `aux_outputs.npz`
- `train_log.csv`

Read-only inspection result:

- Run count: 10
- Missing artifacts: 0
- Non-finite arrays: 0
- Non-finite numeric metrics: 0
- Checkpoints load with `torch.load(..., map_location="cpu", weights_only=False)`: pass
- Required metric keys present: `mse`, `IC`, `RankIC`, `ICIR`, `Precision@10`, `Sharpe`, `num_valid_days`, `num_days`, `RankICIR`, `Return@10`, `Turnover@10`, `CostReturn@10`, `CostSharpe@10`
- Legacy `RIC`: absent

## Efficiency

| Variant | Active Params | All Params | Mean GPU Peak MB |
|---|---:|---:|---:|
| static | 103055 | 103055 | 115.7 |
| context_only | 41945 | 41945 | 114.5 |
| single_gate | 107677 | 107677 | 116.0 |
| moe | 308845 | 308845 | 119.9 |
| full | 318640 | 318640 | 121.3 |

`outputs_stage2_patch_reproduce/efficiency_table.csv` now records mean epoch time via `epoch_sec` and GPU peak memory via `peak_gpu_memory_mb`.

## Principal ML Scientist Assessment

The patch improves instrumentation and adds useful diagnostics, but it currently weakens the core ranking objective. Since the paper-facing signal is RankIC and not only Sharpe, the patch should not be promoted to Stage 3 in its current form.

Most likely regression sources to investigate next:

1. Optimization shift: AdamW + `weight_decay=1e-4` + `alpha_rank=0.3` may be too strong relative to the original Stage 2 setup.
2. Output masking or mask-aware interactions may reduce effective cross-sectional signal if applied too early or too aggressively.
3. Gate initialization (`gate_init=-2.0`) may suppress useful interaction paths for `single_gate/full` during short 30-epoch runs.
4. MoE/full add parameters and runtime but do not recover RankIC enough; router/scale diagnostics should be inspected before expanding them.
5. New cost/turnover metrics are useful diagnostics, but they cannot override the RankIC regression for model selection.

## Recommendation

Do not run `outputs_stage3_patch_core` yet.

Next stage should be `patch_debug_small_grid`, not Stage 3:

- Dataset: NASDAQ only.
- Variants: `static`, `single_gate`, `moe`.
- Seeds: `0 1`.
- Epochs: 30.
- Grid:
  - `alpha_rank`: `0.1`, `0.3`
  - `weight_decay`: `0`, `1e-4`
  - `gate_init`: `-2.0`, `0.0` for gated variants
- Acceptance:
  - `single_gate` or `moe` must recover near the previous Stage 2 RankIC range, ideally `>= 0.025` mean RankIC.
  - No NaN/Inf metrics.
  - Cost metrics and runtime logging preserved.
  - Full `tests/recil` remains pass.

Only after `patch_debug_small_grid` passes should Stage 3 core be considered.
