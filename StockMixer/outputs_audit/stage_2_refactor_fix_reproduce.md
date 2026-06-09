# Stage 2 Refactor-Fix Reproduce Audit

Date: 2026-06-09 00:01:10 +07

Run ID: `stage_2_refactor_fix_reproduce`

Decision: `pass -> stage_3_core_plan`

The refactor-fix code passes the Stage 2 reproduce gate. The repaired implementation keeps the patch tooling and restores ranking signal. `context_only` is the best NASDAQ RankIC candidate in this run, while `single_gate` also passes the required `RankIC >= 0.025` gate.

## Environment

- Interpreter: `/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe`
- Working directory: `StockMixer`
- Dataset: NASDAQ
- Device policy: `--device auto`
- Resolved device: `cuda`
- Output dir: `outputs_stage2_refactor_fix_reproduce`
- Scope: NASDAQ only, 5 variants, seeds `0 1`, 30 epochs
- No Stage 3 run was launched.

## Commands

Preflight:

```bash
/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe -m compileall src/recil
/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe -m pytest tests/recil -q
```

Result: pass, `77 passed`.

Dry-run:

```bash
/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe -m src.recil.run_experiments \
  --datasets nasdaq \
  --variants static context_only single_gate moe full \
  --seeds 0 1 \
  --epochs 30 \
  --alpha-rank 0.1 \
  --weight-decay 0 \
  --grad-clip-norm 0 \
  --transaction-cost-bps 10 \
  --output-dir outputs_stage2_refactor_fix_reproduce \
  --dry-run
```

Result: pass, exactly 10 deterministic commands.

Run:

```bash
/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe -m src.recil.run_experiments \
  --datasets nasdaq \
  --variants static context_only single_gate moe full \
  --seeds 0 1 \
  --epochs 30 \
  --alpha-rank 0.1 \
  --weight-decay 0 \
  --grad-clip-norm 0 \
  --transaction-cost-bps 10 \
  --output-dir outputs_stage2_refactor_fix_reproduce
```

Result: pass, 10/10 runs.

Analysis:

```bash
/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe -m src.recil.analysis \
  --output-dir outputs_stage2_refactor_fix_reproduce \
  --aggregate --regime --interpretability --efficiency --make-tables
```

Result: pass.

## Manifest And Runtime

`outputs_stage2_refactor_fix_reproduce/run_manifest.jsonl` has 10 rows and every row has `status=pass`, `return_code=0`.

| Variant | Seed | Runtime Sec | Status |
|---|---:|---:|---|
| static | 0 | 407.9 | pass |
| static | 1 | 422.0 | pass |
| context_only | 0 | 410.3 | pass |
| context_only | 1 | 429.7 | pass |
| single_gate | 0 | 438.0 | pass |
| single_gate | 1 | 441.3 | pass |
| moe | 0 | 498.6 | pass |
| moe | 1 | 439.1 | pass |
| full | 0 | 570.5 | pass |
| full | 1 | 540.1 | pass |

## Seed-Level Metrics

| Variant | Seed | RankIC | IC | Precision@10 | Sharpe | CostSharpe@10 | Turnover@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| static | 0 | 0.022599 | 0.022568 | 0.047257 | 1.052675 | 1.018694 | 0.027426 |
| static | 1 | 0.024982 | 0.025963 | 0.052321 | 0.721032 | 0.635499 | 0.080169 |
| context_only | 0 | 0.032365 | 0.025406 | 0.049789 | 0.578999 | 0.570618 | 0.007173 |
| context_only | 1 | 0.034370 | 0.033751 | 0.050211 | 1.083186 | 0.906038 | 0.167511 |
| single_gate | 0 | 0.026392 | 0.023850 | 0.048101 | 0.797194 | 0.775104 | 0.019409 |
| single_gate | 1 | 0.028898 | 0.025492 | 0.048101 | 0.098687 | 0.058563 | 0.033333 |
| moe | 0 | 0.013935 | 0.019049 | 0.039241 | 0.516247 | 0.455608 | 0.049789 |
| moe | 1 | 0.017170 | 0.020517 | 0.044304 | 0.730425 | 0.637979 | 0.082278 |
| full | 0 | 0.026350 | 0.023445 | 0.051477 | 0.419453 | 0.399788 | 0.016878 |
| full | 1 | 0.026421 | 0.024387 | 0.051055 | 1.417402 | 1.375699 | 0.036709 |

## Mean Metrics

| Variant | RankIC | IC | Precision@10 | Sharpe |
|---|---:|---:|---:|---:|
| context_only | 0.033368 | 0.029578 | 0.050000 | 0.831092 |
| full | 0.026385 | 0.023916 | 0.051266 | 0.918427 |
| moe | 0.015552 | 0.019783 | 0.041772 | 0.623336 |
| single_gate | 0.027645 | 0.024671 | 0.048101 | 0.447940 |
| static | 0.023790 | 0.024266 | 0.049789 | 0.886853 |

Best RankIC: `context_only`, `0.033368`.

Best Precision@10: `full`, `0.051266`, but it does not beat `context_only` on RankIC.

## Comparison To Previous Runs

| Variant | Old Stage 2 RankIC | Patch-Broken RankIC | Refactor-Fix RankIC |
|---|---:|---:|---:|
| static | 0.031013 | 0.014719 | 0.023790 |
| context_only | 0.025968 | 0.007613 | 0.033368 |
| single_gate | 0.032052 | 0.013176 | 0.027645 |
| moe | 0.019773 | 0.016358 | 0.015552 |
| full | 0.013265 | 0.013263 | 0.026385 |

Interpretation:

- The refactor fix restores the ranking signal and passes the scientific gate.
- `context_only` improves beyond its old baseline and is the best Stage 2 refactor-fix variant.
- `single_gate` passes the required gate, but it remains below its old Stage 2 baseline.
- `static` is competitive but just below the preferred `0.025` RankIC threshold.
- `moe` underperforms despite higher parameter count and should not be the main model.
- `full` is now valid and competitive, but it does not beat `context_only` on RankIC.

## Artifact And Metric Checks

Read-only inspection result:

- Run count: 10
- Manifest rows: 10
- Missing required artifacts: 0
- Non-finite arrays: 0
- Non-finite numeric metrics: 0
- Checkpoints load with `torch.load(..., map_location="cpu", weights_only=False)`: pass
- Required metric keys present: `mse`, `IC`, `RankIC`, `ICIR`, `Precision@10`, `Sharpe`, `num_valid_days`, `num_days`, `RankICIR`, `Return@10`, `Turnover@10`, `CostReturn@10`, `CostSharpe@10`
- Legacy `RIC`: absent
- Post-run `pytest tests/recil -q`: pass, `77 passed`

## Efficiency

| Variant | Params | Mean Epoch Sec | GPU Peak MB | RankIC |
|---|---:|---:|---:|---:|
| static | 104231 | 13.69 | 118.3 | 0.023790 |
| context_only | 42063 | 13.85 | 117.0 | 0.033368 |
| single_gate | 108853 | 14.50 | 118.6 | 0.027645 |
| moe | 313195 | 15.48 | 122.5 | 0.015552 |
| full | 322990 | 18.35 | 123.9 | 0.026385 |

Efficiency interpretation:

- `context_only` is the strongest Stage 2 result by RankIC and has the lowest parameter count.
- `single_gate` remains useful as a stronger interaction ablation than static, but not the top main candidate in this run.
- `full` adds substantial complexity and runtime without beating `context_only`.
- `moe` is not justified at this stage.

## Principal ML Scientist Assessment

The correct next scientific direction is not to revert to old code and not to choose the most complex model. The repaired code keeps the useful patch infrastructure and recovers RankIC. The Stage 2 evidence now points to a simpler context-conditioned model as the main candidate.

Recommended model roles for the next stage:

- Main candidate: `context_only`
- Strong ablation: `single_gate`
- Baseline/control: `static`
- Optional diagnostic: `full`
- Deprioritize: `moe`

## Final Decision

Decision: `pass -> stage_3_core_plan`

Proceed to plan Stage 3 core, but do not launch it automatically from this audit.

Recommended Stage 3 core:

```bash
/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe -m src.recil.run_experiments \
  --datasets nasdaq sp500 \
  --variants static context_only single_gate full \
  --seeds 0 1 2 \
  --epochs 50 \
  --alpha-rank 0.1 \
  --weight-decay 0 \
  --grad-clip-norm 0 \
  --transaction-cost-bps 10 \
  --output-dir outputs_stage3_refactor_fix_core
```

Stage 3 acceptance should require:

- NASDAQ and SP500 artifacts clean.
- `context_only` remains competitive on both datasets.
- `single_gate/full` only become main candidates if their RankIC beats `context_only`, not because of Sharpe alone.
- Three-seed mean/std reported before any paper claim.
