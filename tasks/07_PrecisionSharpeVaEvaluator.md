# 07 - Precision Sharpe Va Evaluator

## Goal

Implement ranking/portfolio metrics and a clean evaluator that reports IC, RankIC, ICIR, Precision@K, Sharpe, and valid-day counts.

## Sources

- `StockMixer/src/Original/evaluator.py`
- `tasks/06_MetricsMaskedICRankIC.md`
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
   def precision_at_k(pred, target, mask, k=10): ...
   def long_only_daily_return(pred, target_return, mask, k=10): ...
   def sharpe_ratio(daily_returns, annualization=252, eps=1e-8): ...
   def evaluate_predictions(preds, targets, masks, k=10, asset_major=False): ...
   ```

2. Precision@K:
   - Select top-K predicted assets among valid assets.
   - Ground-truth top-K is top-K realized return among valid assets.
   - Use `K_eff = min(k, valid_count)`.
   - Return NaN if no valid assets.

3. Long-only daily return:
   - Average realized return of predicted top-K valid assets.
   - Do not assume transaction costs.

4. Sharpe:

   ```text
   Sharpe = mean(daily_returns) / std(daily_returns) * sqrt(annualization)
   ```

   Return finite value with `eps`; do not return NaN for constant positive returns.

5. Evaluator output keys:

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

6. Tests:
   - Perfect top-K overlap gives Precision@K `1.0`.
   - Masked asset cannot be selected.
   - Constant returns Sharpe is finite.
   - Evaluator includes required keys and no `RIC`.

## Test

```bash
cd StockMixer
python3 -m pytest tests/recil/test_metrics.py -q
python3 -m compileall src/recil
```

## Pass criteria

- Metrics are finite where expected.
- Invalid assets are excluded from ranking and portfolio metrics.
- Evaluator output uses correct metric names.

## Expected output

A reviewer-safe evaluator for all later training and analysis tasks.

## Limitations

Sharpe here is a simple diagnostic metric. Do not claim a complete trading system without turnover and transaction-cost analysis.
