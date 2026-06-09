# MoE/Full Router Repair Audit

Date: 2026-06-09

Run ID: `moe_full_router_repair`

Decision: `blocked -> add_router_specialization_objective_before_stage_3`

## Summary

The router repair infrastructure was implemented and regression-tested, but the first real NASDAQ Phase A probe did not pass the scientific gate. `moe/full` should not be promoted to Stage 3 yet.

The core issue remains under-specialized routing. The first Phase A config reduced experts from 4 to 2 and used `router_init=small_normal`, `router_temperature=0.5`, but router entropy still stayed near exact uniform after 30 epochs.

## Implemented Changes

- Added configurable MoE router controls:
  - `--router-init {zero,small_normal}`
  - `--router-temperature FLOAT`
  - `--interaction-warmup-epochs INT`
- Added `--run-tag` to namespace config sweeps as `dataset/variant__tag/seed_x`, preventing overwrite between Phase A configs.
- Added router/gate diagnostics to `metrics.json` when available:
  - `router_entropy_norm`, `router_max_mean`, `router_std`
  - `scale_entropy_norm`, `scale_max_mean`, `scale_std`
  - `context_gate_mean/std/min/max`
- Preserved default behavior:
  - default `router_init=zero`
  - default `router_temperature=1.0`
  - default `interaction_warmup_epochs=0`
  - no dataset/split/loss contract changes

## Verification

Commands:

```bash
/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe -m pytest tests/recil -q
/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe -m compileall src/recil
/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe -m src.recil.train_recil \
  --synthetic --quick-test --variant moe --epochs 2 \
  --num-experts 2 \
  --router-init small_normal \
  --router-temperature 0.5 \
  --interaction-warmup-epochs 1 \
  --device cpu \
  --run-tag ne2_temp05_warm1 \
  --output-dir outputs_router_repair_smoke
```

Results:

- Unit/regression tests: `85 passed`
- `compileall src/recil`: pass
- Synthetic smoke: pass
- Smoke metrics contain router diagnostics and no bare `RIC` key.

## Phase A Probe

Command:

```bash
/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe -m src.recil.run_experiments \
  --datasets nasdaq \
  --variants moe full \
  --seeds 0 1 \
  --epochs 30 \
  --num-experts 2 \
  --router-init small_normal \
  --router-temperature 0.5 \
  --alpha-rank 0.1 \
  --weight-decay 0 \
  --grad-clip-norm 0 \
  --transaction-cost-bps 10 \
  --run-tag ne2_init_small_temp05 \
  --output-dir outputs_moe_full_router_repair
```

Manifest:

- `outputs_moe_full_router_repair/run_manifest.jsonl`
- Rows: `4`
- Status: `4/4 pass`
- Device: CUDA

Mean metrics:

| variant | RankIC | IC | Precision@10 | Sharpe | OriginalIC | OriginalICIR | OriginalPositivePrecision@10 | OriginalSharpe@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full__ne2_init_small_temp05 | 0.027350 | 0.024662 | 0.051899 | 0.697395 | 0.024728 | 0.252010 | 0.509705 | 0.104691 |
| moe__ne2_init_small_temp05 | 0.019696 | 0.023122 | 0.045992 | 0.487729 | 0.023123 | 0.259105 | 0.500844 | 0.849554 |

Diagnostics:

| variant | seed | RankIC | router_entropy_norm | router_max_mean | router_std | scale_entropy_norm | context_gate_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full__ne2_init_small_temp05 | 0 | 0.026875 | 0.999820 | 0.506493 | 0.007888 | 0.861594 | 0.125922 |
| full__ne2_init_small_temp05 | 1 | 0.027825 | 0.999692 | 0.509779 | 0.010337 | 0.840007 | 0.182579 |
| moe__ne2_init_small_temp05 | 0 | 0.016520 | 0.999601 | 0.510503 | 0.011764 |  |  |
| moe__ne2_init_small_temp05 | 1 | 0.022872 | 0.999858 | 0.505734 | 0.007026 |  |  |

## Gate Result

Failed.

- Router entropy target was `[0.4, 0.95]`; observed values remained `~0.9996-0.9999`.
- `moe RankIC_mean` target was `>= 0.025`; observed `0.019696`.
- `full RankIC_mean` improved over previous `0.026385` to `0.027350`, but still below the Stage 3 candidate target near `context_only - 0.002`.
- `full` improvement is real but not enough to justify Stage 3 with full as main candidate.

## Interpretation

Reducing experts from 4 to 2 and lowering router temperature to `0.5` improves `full` slightly but does not solve routing specialization. The router still behaves almost like uniform averaging.

After this run, `small_normal` router initialization was strengthened from `std=1e-3` to `std=0.02`. A static initialization probe still showed near-uniform initial routing for temperature values `0.5`, `1.0`, and `2.0`, so the remaining planned Phase A temperature grid is unlikely to meet the entropy gate by initialization alone.

The next general fix should add an explicit router specialization objective that encourages low per-sample entropy while preserving balanced global expert usage. This follows MoE practice more closely than trying arbitrary init/temperature values.

## Next Required Plan

Do not run Stage 3 yet.

Plan the next repair as:

- Add optional router specialization loss:
  - minimize per-sample router entropy to encourage decisive routing;
  - maximize batch-level mean router entropy to avoid expert collapse;
  - keep default disabled for backward compatibility.
- Add CLI flags:
  - `--lambda-router-specialization`
  - `--lambda-router-balance`
- Run a small NASDAQ probe on `moe/full`, seeds `0 1`, 30 epochs.
- Promote only if:
  - `RankIC_mean` improves, not just Sharpe;
  - router entropy leaves exact-uniform behavior;
  - global expert usage does not collapse to one expert.

