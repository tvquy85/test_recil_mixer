# ReCIL-Mixer Current Status Guide

This repository snapshot is intended for ChatGPT or another reviewer to analyze
the current ReCIL-Mixer implementation, results, and next research direction.

## What This Project Is

ReCIL-Mixer is an experimental market-context-aware extension around the
StockMixer stock prediction setup. The implementation keeps the original
StockMixer source for reference and adds a new ReCIL namespace:

```text
StockMixer/src/recil/
```

The ReCIL pipeline includes:

- causal market context features with train-only scaling;
- StockMixer-compatible dataset alignment;
- mask-aware IC, RankIC, Precision@10, Sharpe, and MSE metrics;
- masked MSE and pairwise ranking losses;
- encoder, temporal mixer, FiLM, context gate, low-rank experts, and variant
  assembly;
- training, experiment runner, analysis, audit, and paper-table utilities.

## What Was Intentionally Not Pushed

The snapshot excludes raw data and heavy artifacts:

- `StockMixer/dataset/`
- checkpoints `*.pt`
- raw arrays `*.npy`, `*.npz`
- pickle files `*.pkl`
- local cache/test smoke output trees

The CSV, JSON, Markdown audit, source code, tests, and small figures are enough
to analyze the current behavior without uploading datasets.

## Key Files To Read

Start here:

```text
guide.md
StockMixer/outputs_audit/stage_2_medium_scale.md
StockMixer/outputs_stage2/results_summary_mean_std.csv
StockMixer/outputs_stage2/results_summary.csv
StockMixer/docs/experiment_plan_recil.md
```

Implementation:

```text
StockMixer/src/recil/context.py
StockMixer/src/recil/data.py
StockMixer/src/recil/metrics.py
StockMixer/src/recil/losses.py
StockMixer/src/recil/modules.py
StockMixer/src/recil/model.py
StockMixer/src/recil/train_recil.py
StockMixer/src/recil/run_experiments.py
StockMixer/src/recil/analysis.py
```

Tests:

```text
StockMixer/tests/recil/
```

Reference/audit documents:

```text
StockMixer/docs/repo_audit_recil.md
StockMixer/docs/data_contract_recil.md
StockMixer/outputs_audit/stage_0_sanity_check.md
StockMixer/outputs_audit/final_acceptance_recil.md
StockMixer/outputs_audit/stage_1_small_scale.md
StockMixer/outputs_audit/stage_2_medium_scale.md
```

## Completed Work

All planned implementation tasks `00` through `16` were completed. The final
acceptance gate passed.

Experiment stages completed:

```text
stage_0_sanity_check: pass
stage_1_small_scale: pass
stage_2_medium_scale: pass
```

The audited Python environment used for tests and runs was:

```text
/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe
```

CUDA was available and training used an RTX 3090.

## Stage 2 Main Result

Stage 2 ran NASDAQ, all five variants, seeds `0 1`, 30 epochs.

Mean metrics across two seeds:

```text
variant        RankIC_mean  IC_mean   Precision@10_mean  Sharpe_mean
single_gate    0.032052     0.027211  0.053165           0.878273
static         0.031013     0.034343  0.046203           0.027600
context_only   0.025968     0.020990  0.055696           0.507710
moe            0.019773     0.022368  0.055485           1.180955
full           0.013265     0.017240  0.049156           1.550974
```

Interpretation:

- `single_gate` is currently the strongest variant by `RankIC_mean`, with low
  variance across two seeds.
- `static` remains a strong baseline and is best by `IC_mean`.
- `context_only` and `moe` have competitive `Precision@10`, but weaker RankIC.
- `moe` and `full` improve Sharpe-like portfolio diagnostics, but are much
  weaker by cross-sectional rank quality.
- `full` is not currently better than `single_gate`; adding all gates/experts
  appears to trade off rank stability for concentrated top-k/portfolio behavior.

## Important Metric Meaning

- `IC`: Pearson correlation between prediction and target return over valid
  assets.
- `RankIC`: Spearman rank correlation. This is the most important metric for
  cross-sectional stock ranking.
- `ICIR`: mean IC divided by IC standard deviation across days.
- `Precision@10`: overlap between predicted top-10 valid assets and realized
  top-10 valid assets.
- `Sharpe`: simple diagnostic Sharpe over long-only top-k returns. It does not
  include transaction costs, slippage, turnover, or capacity.
- `mse`: mask-aware mean squared error. Useful for sanity, not the main
  decision metric for stock selection.

## Current Scientific Read

The current strongest paper story is not "the full model wins everything."

The current evidence says:

1. Context adaptation helps most when kept simple: `single_gate`.
2. The static StockMixer-like baseline is strong and must be respected.
3. The full MoE/gated variant may need regularization, router diagnostics, or a
   different objective to improve broad ranking.
4. Sharpe and RankIC disagree for `moe`/`full`, which suggests useful top-k
   behavior but weaker global ranking.

## Suggested Questions For ChatGPT Analysis

Ask ChatGPT or another reviewer:

1. Given Stage 2, should Stage 3 run all variants, or prioritize `static`,
   `single_gate`, and `full`?
2. How should the paper explain the tradeoff where `full` has high Sharpe but
   low RankIC?
3. Should the loss be adjusted to put more weight on ranking for `moe` and
   `full`?
4. Should router entropy, expert collapse, and scale weights be analyzed before
   expanding to SP500?
5. Is `single_gate` a better main ReCIL variant than `full` for the current
   paper framing?

## Reproducing Checks

From `StockMixer`:

```bash
/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe -m compileall src/recil
/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe -m pytest tests/recil -q
```

To inspect Stage 2 summary:

```bash
sed -n '1,20p' outputs_stage2/results_summary_mean_std.csv
sed -n '1,80p' outputs_audit/stage_2_medium_scale.md
```

To rerun Stage 2, raw StockMixer-format datasets must be restored locally under
`StockMixer/dataset/`; they are intentionally not part of this repository
snapshot.
