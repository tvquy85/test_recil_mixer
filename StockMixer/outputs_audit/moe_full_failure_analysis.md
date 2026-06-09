# MoE/Full RankIC Failure Analysis

Date: 2026-06-09

Run inspected: `outputs_stage2_refactor_fix_reproduce`

## Decision

Do not run Stage 3 with `moe` or `full` as main candidates yet.

Current evidence supports `context_only` as the strongest NASDAQ Stage 2 candidate by RankIC, with `single_gate` as the lower-complexity interaction candidate. `moe` and `full` should enter a targeted repair stage first.

## Local Evidence

Mean Stage 2 refactor-fix metrics:

| variant | RankIC | IC | Precision@10 | Sharpe | OriginalIC | OriginalICIR | OriginalPositivePrecision@10 | OriginalSharpe@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| context_only | 0.033368 | 0.029578 | 0.050000 | 0.831092 | 0.028361 | 0.293250 | 0.510127 | -0.321152 |
| full | 0.026385 | 0.023916 | 0.051266 | 0.918427 | 0.023206 | 0.242166 | 0.510127 | -0.800038 |
| moe | 0.015552 | 0.019783 | 0.041772 | 0.623336 | 0.019568 | 0.234416 | 0.514557 | 1.654950 |
| single_gate | 0.027645 | 0.024671 | 0.048101 | 0.447940 | 0.024046 | 0.249129 | 0.503797 | -0.941287 |
| static | 0.023790 | 0.024266 | 0.049789 | 0.886853 | 0.024231 | 0.259152 | 0.518143 | -0.497729 |

Key diagnostics:

- `moe` router entropy normalized: `0.999927`; mean max router weight: `0.254216`.
- `full` router entropy normalized: `0.999820`; mean max router weight: `0.256215`.
- A healthy 4-expert router should not remain this close to exact uniform if context is creating useful specialization.
- `moe` RankIC is low despite high `OriginalSharpe@5`, so the model is not uniformly broken; it is optimizing/selecting a behavior that does not preserve cross-sectional ordering.
- `full` has `scale_weights` that move away from uniform, but the residual context gate remains modest: mean `0.153713`. Most of the interaction branch is still suppressed.
- Validation RankIC for `moe`, `full`, `single_gate`, and `static` peaks early and degrades by epoch 30; `context_only` is more stable and improves later.

Best validation epochs:

| variant | seed 0 best epoch | seed 1 best epoch |
| --- | ---: | ---: |
| context_only | 28 | 9 |
| full | 4 | 3 |
| moe | 4 | 8 |
| single_gate | 4 | 3 |
| static | 12 | 14 |

## Diagnosis

The main failure mode is not classic MoE expert collapse. It is under-specialized routing plus unstable interaction optimization.

1. `moe` uses four experts but the router is effectively uniform. This averages several randomly initialized low-rank stock mixers instead of selecting a regime-specific expert.
2. `full` combines scale fusion, FiLM, MoE, and context residual. The useful `context_only` signal is present, but extra interaction modules add variance and early validation overfit.
3. The data scale is limited: NASDAQ Stage 2 has two seeds and 30 epochs. StockMixer's own motivation favors simple, easy-to-optimize MLP mixing under limited stock data.
4. Sharpe-style diagnostics and RankIC disagree. `moe` has high `OriginalSharpe@5` but weak RankIC, so it should not be selected as the main ranking model without further evidence.

## Source-Grounded Interpretation

- StockMixer AAAI 2024 emphasizes that complex models can be hard to optimize under limited stock data, while simple MLP indicator/time/stock mixing is strong and efficient.
- Original MoE work uses gating to select expert combinations per example; if the learned gate remains uniform, conditional computation is not actually being used.
- Switch Transformer notes MoE adoption is affected by complexity, communication costs, and training instability. The local evidence is consistent with instability and ineffective routing, not with a clean capacity gain.
- TFT uses gates to suppress unnecessary components; this supports making interaction branches skippable and monitored, not assuming more gates/experts are automatically better.
- FiLM is a lightweight feature-wise affine conditioning layer. The current `context_only` result suggests context conditioning is useful before adding heavier cross-asset expert routing.

Sources:

- StockMixer AAAI 2024: https://mlanthology.org/aaai/2024/fan2024aaai-stockmixer/
- Sparsely-Gated MoE: https://arxiv.org/abs/1701.06538
- Switch Transformer: https://jmlr.org/papers/v23/21-0998.html
- Temporal Fusion Transformer: https://arxiv.org/abs/1912.09363
- FiLM: https://arxiv.org/abs/1709.07871

## General Fix Plan

### Phase 1: Router Specialization Probe

Run `moe` and `full` with router diagnostics and alternative router initialization.

Configs:

- `num_experts=2` instead of `4`.
- router final layer not zero-initialized, but small normal initialization.
- router temperature grid: `0.5`, `1.0`, `2.0`.
- keep `alpha_rank=0.1`, `weight_decay=0`, `grad_clip_norm=0`.

Pass criteria:

- router normalized entropy drops below `0.95` without collapse below `0.4`.
- `moe RankIC_mean >= 0.025`.
- no NaN/Inf and no deterioration in `Precision@10`.

### Phase 2: Interaction Warm-Up

Train first epochs with context/temporal path only, then enable interaction modules.

Configs:

- freeze expert branch for 5 epochs, then unfreeze.
- or ramp residual gate from near-zero to learned over 10 epochs.

Pass criteria:

- validation RankIC no longer peaks at epoch 3-8 and collapses by epoch 30.
- `full RankIC_mean >= context_only RankIC_mean - 0.002` before considering it a main candidate.

### Phase 3: Capacity Control

Reduce interaction branch capacity before adding more regularization.

Configs:

- `rank=8` or `rank=16`.
- `num_experts=2`.
- `dropout=0.0` vs `0.1` as an isolated factor.

Pass criteria:

- `moe/full` improve RankIC, not only Sharpe.
- active parameter count and runtime remain justifiable relative to `context_only`.

### Phase 4: Selection Rule

Model selection should remain:

1. primary: `RankIC`, `IC`, `RankICIR`;
2. compatibility: `OriginalIC`, `OriginalICIR`, `OriginalPositivePrecision@10`;
3. diagnostics: `Sharpe`, `OriginalSharpe@5`, turnover/cost metrics.

Do not choose `moe` or `full` purely from Sharpe if RankIC remains weak.

