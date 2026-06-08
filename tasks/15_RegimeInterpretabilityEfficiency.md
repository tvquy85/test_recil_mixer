# 15 - Regime Interpretability Efficiency

## Goal

Implement analysis utilities for aggregate metrics, regime-wise performance, router/scale interpretability, and efficiency diagnostics.

## Sources

- `MoTa.md`
- `ThucThi.md`
- `tasks/07_PrecisionSharpeVaEvaluator.md`
- `tasks/12_TrainingCLICheckpointLogging.md`
- Temporal Fusion Transformer interpretability motivation: https://arxiv.org/abs/1912.09363

## Files

Update:

```text
StockMixer/src/recil/analysis.py
StockMixer/tests/recil/test_analysis.py
```

Create output directories when commands run:

```text
StockMixer/paper_tables
StockMixer/figures
```

## Steps

1. Implement CLI flags:

   ```text
   --output-dir outputs
   --aggregate
   --regime
   --interpretability
   --efficiency
   --make-tables
   ```

2. Aggregate:
   - Read `metrics.json` under `{dataset}/{variant}/seed_{seed}`.
   - Write `results_summary.csv`.
   - Columns:

     ```text
     dataset, variant, seed, IC, RankIC, ICIR, Precision@10, Sharpe, best_epoch
     ```

   - Also compute mean/std by `dataset x variant`.

3. Regime-wise:
   - Load predictions, labels, masks, contexts.
   - Split test days by median of:
     - `market_volatility`
     - `pca_ratio`
     - `cross_sectional_dispersion`
   - Split trend by `market_trend >= 0` vs `< 0`.
   - Write `regime_results.csv` with metric columns and `num_days`.

4. Interpretability:
   - Load `aux_outputs.npz`.
   - Compute correlations:
     - short scale weight vs volatility
     - each expert weight vs volatility, pca_ratio, dispersion
   - Write `interpretability_correlations.csv`.
   - Generate simple PNG time-series plots if `matplotlib` is available; otherwise write CSV-only and document missing plotting dependency.

5. Efficiency:
   - Count params from model config.
   - Read or compute training time per epoch from `train_log.csv`.
   - If CUDA was used, include peak memory if logged.
   - Write `efficiency_table.csv`.

6. Paper tables:
   - Generate simple LaTeX snippets for main results, ablation, regime, efficiency.
   - Bold best per metric if implemented safely.

## Test

```bash
cd StockMixer
python3 -m pytest tests/recil/test_analysis.py -q
python3 -m src.recil.analysis --output-dir outputs_stage0 --aggregate
```

## Pass criteria

- Analysis handles missing optional files with clear warnings.
- Aggregation works on stage 0 outputs.
- No analysis claims are hard-coded; all numbers come from saved outputs.

## Expected output

CSV and optional figure/table artifacts that support the ReCIL paper story.

## Limitations

If router/scale correlations are weak, report them honestly. Do not force interpretability claims.
