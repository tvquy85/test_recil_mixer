# ReCIL-Mixer Current Status Guide

This repository snapshot is intended for ChatGPT or another reviewer to analyze
the current ReCIL-Mixer implementation, experiment evidence, and next research
direction.

## Project Goal

ReCIL-Mixer is an experimental market-context-aware extension around the
StockMixer stock prediction setup. The original StockMixer code is preserved for
reference, while the new implementation lives in:

```text
StockMixer/src/recil/
```

The ReCIL pipeline includes causal market context, train-only scaling,
StockMixer-compatible data alignment, mask-aware metrics/losses, ReCIL model
variants, training CLI, experiment runner, analysis utilities, and audit files.

## What Is Excluded

The repository intentionally excludes raw/heavy artifacts:

```text
StockMixer/dataset/
*.pt
*.npy
*.npz
*.pkl
```

CSV/JSON/Markdown summaries are included where useful for analysis. To rerun
real-data experiments, restore StockMixer-format datasets under
`StockMixer/dataset/`.

## Key Files To Read

Current high-value files:

```text
StockMixer/src/recil/
StockMixer/tests/recil/
StockMixer/outputs_stage2_refactor_fix_reproduce/results_summary_mean_std.csv
StockMixer/outputs_moe_full_router_repair/results_summary_mean_std.csv
StockMixer/outputs_audit/stage_2_refactor_fix_reproduce.md
StockMixer/outputs_audit/moe_full_failure_analysis.md
StockMixer/outputs_audit/moe_full_router_repair.md
StockMixer/docs/experiment_plan_recil.md
```

Reference/audit context:

```text
StockMixer/docs/repo_audit_recil.md
StockMixer/docs/data_contract_recil.md
StockMixer/outputs_audit/stage_0_sanity_check.md
StockMixer/outputs_audit/final_acceptance_recil.md
StockMixer/outputs_audit/stage_1_small_scale.md
StockMixer/outputs_audit/stage_2_medium_scale.md
```

## Implementation Status

Completed:

```text
tasks 00-16: complete
stage_0_sanity_check: pass
stage_1_small_scale: pass
stage_2_medium_scale: pass
stage_2_refactor_fix_reproduce: pass
moe_full_router_repair phase-A first probe: blocked
```

Current test baseline:

```text
/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe -m pytest tests/recil -q
85 passed
```

The audited interpreter is:

```text
/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe
```

CUDA/RTX 3090 was available for real-data training.

## Metric Families

ReCIL clean metrics:

- `IC`: mask-aware Pearson correlation over valid assets.
- `RankIC`: mask-aware Spearman rank correlation; main scientific metric.
- `ICIR`, `RankICIR`: information ratios over daily IC series.
- `Precision@10`: predicted top-10 overlap with realized top-10.
- `Sharpe`, `Return@10`, `Turnover@10`, cost metrics: diagnostics.

Original-compatible metrics added for fair comparison with
`StockMixer/src/Original/evaluator.py`:

- `OriginalIC`: zero-fill masked Pearson IC, matching Original behavior.
- `OriginalICIR`: Original's legacy `RIC` logic, renamed to avoid ambiguity.
- `OriginalPositivePrecision@10`: positive-return rate inside predicted top-10.
- `OriginalSharpe@5`: Original `sharpe5` logic with multiplier `15.87`.
- `OriginalMSE`: Original masked MSE formula.

Do not use a bare `RIC` key in ReCIL. It is historically ambiguous.

## Stage 2 Refactor-Fix Reproduce

Run:

```text
outputs_stage2_refactor_fix_reproduce
NASDAQ, variants static/context_only/single_gate/moe/full, seeds 0 1, 30 epochs
alpha_rank=0.1, weight_decay=0, grad_clip_norm=0, transaction_cost_bps=10
```

Mean metrics across two seeds:

| variant | RankIC | IC | Precision@10 | Sharpe | OriginalIC | OriginalICIR | OriginalPositivePrecision@10 | OriginalSharpe@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| context_only | 0.033368 | 0.029578 | 0.050000 | 0.831092 | 0.028361 | 0.293250 | 0.510127 | -0.321152 |
| full | 0.026385 | 0.023916 | 0.051266 | 0.918427 | 0.023206 | 0.242166 | 0.510127 | -0.800038 |
| moe | 0.015552 | 0.019783 | 0.041772 | 0.623336 | 0.019568 | 0.234416 | 0.514557 | 1.654950 |
| single_gate | 0.027645 | 0.024671 | 0.048101 | 0.447940 | 0.024046 | 0.249129 | 0.503797 | -0.941287 |
| static | 0.023790 | 0.024266 | 0.049789 | 0.886853 | 0.024231 | 0.259152 | 0.518143 | -0.497729 |

Interpretation:

- `context_only` is currently the strongest Stage 2 candidate by `RankIC`.
- `single_gate` remains a lower-complexity interaction candidate.
- `full` is better than `static/single_gate` on some top-k diagnostics, but not
  better than `context_only` on ranking.
- `moe` has high `OriginalSharpe@5` but weak `RankIC`, so it should not be
  chosen as the main model from Sharpe alone.

## MoE/Full Failure Diagnosis

Audit:

```text
StockMixer/outputs_audit/moe_full_failure_analysis.md
```

Key finding:

```text
moe router_entropy_norm  ~= 0.999927
full router_entropy_norm ~= 0.999820
```

This is not classic expert collapse into one expert. It is under-specialized
routing: the router remains almost exactly uniform, so MoE behaves like averaging
several low-rank stock mixers rather than selecting regime-specific experts.

## Router Repair Infrastructure

Implemented after the diagnosis:

```text
--router-init {zero,small_normal}
--router-temperature FLOAT
--interaction-warmup-epochs INT
--run-tag TAG
```

New diagnostics are written to `metrics.json` when available:

```text
router_entropy_norm
router_max_mean
router_std
scale_entropy_norm
scale_max_mean
scale_std
context_gate_mean
context_gate_std
context_gate_min
context_gate_max
```

Default behavior is backward-compatible:

```text
router_init=zero
router_temperature=1.0
interaction_warmup_epochs=0
```

## Router Repair Phase A Result

Audit:

```text
StockMixer/outputs_audit/moe_full_router_repair.md
```

Run:

```text
outputs_moe_full_router_repair
variants: moe full
seeds: 0 1
epochs: 30
num_experts=2
router_init=small_normal
router_temperature=0.5
run_tag=ne2_init_small_temp05
```

Mean metrics:

| variant | RankIC | IC | Precision@10 | Sharpe | OriginalIC | OriginalICIR | OriginalPositivePrecision@10 | OriginalSharpe@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full__ne2_init_small_temp05 | 0.027350 | 0.024662 | 0.051899 | 0.697395 | 0.024728 | 0.252010 | 0.509705 | 0.104691 |
| moe__ne2_init_small_temp05 | 0.019696 | 0.023122 | 0.045992 | 0.487729 | 0.023123 | 0.259105 | 0.500844 | 0.849554 |

Router diagnostics:

| variant | seed | RankIC | router_entropy_norm | router_max_mean | router_std | scale_entropy_norm | context_gate_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full__ne2_init_small_temp05 | 0 | 0.026875 | 0.999820 | 0.506493 | 0.007888 | 0.861594 | 0.125922 |
| full__ne2_init_small_temp05 | 1 | 0.027825 | 0.999692 | 0.509779 | 0.010337 | 0.840007 | 0.182579 |
| moe__ne2_init_small_temp05 | 0 | 0.016520 | 0.999601 | 0.510503 | 0.011764 |  |  |
| moe__ne2_init_small_temp05 | 1 | 0.022872 | 0.999858 | 0.505734 | 0.007026 |  |  |

Gate result:

```text
blocked -> add_router_specialization_objective_before_stage_3
```

Reason:

- Target router entropy range was `[0.4, 0.95]`.
- Observed entropy stayed near `~0.9996-0.9999`.
- `full` improved slightly from `0.026385` to `0.027350` RankIC, but remains
  below the `context_only` baseline.
- `moe` improved from `0.015552` to `0.019696` RankIC, but still fails the
  `>=0.025` gate.

## Current Scientific Read

The strongest current story is not "the full model wins".

Current evidence says:

1. Context conditioning helps, but the simple `context_only` path is currently
   strongest by RankIC.
2. `single_gate` is still a plausible lightweight interaction candidate.
3. `moe/full` need a router objective, not just more experts, temperature
   changes, or initialization tweaks.
4. Sharpe-style metrics and RankIC disagree; RankIC remains the primary metric
   for stock ranking.
5. Stage 3 should not run `moe/full` as main candidates until router
   specialization is fixed.

## Recommended Next Direction

Plan the next repair around an optional router specialization objective:

```text
per-sample objective: encourage decisive, lower-entropy routing
batch-level objective: keep average expert usage balanced to avoid collapse
default: disabled for backward compatibility
```

Likely CLI flags:

```text
--lambda-router-specialization
--lambda-router-balance
```

Promote `moe/full` only if:

```text
RankIC_mean improves, not only Sharpe
router entropy leaves exact-uniform behavior
global expert usage does not collapse into one expert
full approaches context_only RankIC within roughly 0.002
```

## Suggested Questions For ChatGPT Analysis

Ask ChatGPT to analyze:

1. Is `context_only` currently the best main paper variant, with `single_gate`
   as the interaction ablation?
2. What router specialization/balance loss is most principled for dense
   softmax MoE in this stock-selection setting?
3. Should `moe/full` use sparse/top-k routing, dense routing with entropy
   regularization, or a simpler two-expert regime split?
4. Should Stage 3 run `context_only/static/single_gate` first while `moe/full`
   remain under repair?
5. How should the paper discuss the disagreement between `RankIC` and
   `OriginalSharpe@5`?

## Reproducing Checks

From `StockMixer`:

```bash
/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe -m compileall src/recil
/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe -m pytest tests/recil -q
```

Inspect summaries:

```bash
sed -n '1,20p' outputs_stage2_refactor_fix_reproduce/results_summary_mean_std.csv
sed -n '1,20p' outputs_moe_full_router_repair/results_summary_mean_std.csv
sed -n '1,160p' outputs_audit/moe_full_router_repair.md
```

