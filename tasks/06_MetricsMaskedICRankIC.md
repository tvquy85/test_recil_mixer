# 06 - Metrics Masked IC RankIC

## Goal

Implement correct masked IC, RankIC, and ICIR. This replaces the old evaluator behavior where invalid assets are zero-filled and `RIC` actually means ICIR.

## Sources

- `StockMixer/src/Original/evaluator.py`
- `StockMixer/src/Enhanced/evaluator.py`
- `MoTa.md`
- `ThucThi.md`

## Files

Update:

```text
StockMixer/src/recil/metrics.py
StockMixer/tests/recil/test_metrics.py
```

## Steps

1. Implement:

   ```python
   def pearson_corr_masked(pred, target, mask, eps=1e-8, min_valid=3): ...
   def spearman_corr_masked(pred, target, mask, eps=1e-8, min_valid=3): ...
   def compute_ic_series(preds, targets, masks): ...
   def summarize_ic(ic_series, rankic_series): ...
   ```

2. Shape support:
   - Single day: `[N]`.
   - Multiple days: either `[D, N]` preferred for new code or `[N, D]` if `asset_major=True` is passed. Be explicit in function docs.

3. Rules:
   - Only assets with `mask > 0.5` participate.
   - If valid assets `< min_valid`, return `np.nan` for that day.
   - If pred or target has near-zero std, return `np.nan`.
   - `IC` is Pearson.
   - `RankIC` is Spearman, implemented through ranks then Pearson.
   - `ICIR = nanmean(IC_t) / nanstd(IC_t)`.
   - Do not expose or print `RIC` in new ReCIL metrics.

4. Tests:
   - Perfect ranking gives `IC=1`, `RankIC=1`.
   - Reversed ranking gives negative values.
   - Invalid masked asset does not affect result.
   - Fewer than 3 valid assets returns NaN.
   - Summary ignores NaN days and counts valid days.

## Test

```bash
cd StockMixer
python3 -m pytest tests/recil/test_metrics.py -q
python3 -m compileall src/recil
```

## Pass criteria

- Toy cases match exact expected values.
- No metric function zero-fills invalid assets before correlation.
- No new metric key named `RIC`.

## Expected output

Clean correlation metrics suitable for financial ML review.

## Limitations

Spearman ties must be handled deterministically. If `scipy` is unavailable, implement average-rank fallback and test ties.
