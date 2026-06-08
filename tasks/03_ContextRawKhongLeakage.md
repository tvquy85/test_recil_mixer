# 03 - Context Raw Khong Leakage

## Goal

Implement causal raw market context features without normalization. Context must use only historical close prices in the input lookback window and must exclude invalid assets.

## Sources

- `MoTa.md`
- `ThucThi.md`
- `StockMixer/src/Enhanced/preprocess_context.py`
- `StockMixer/src/GatedContext/preprocess.py`
- FiLM conditioning: https://arxiv.org/abs/1709.07871
- Temporal Fusion Transformer gating/interpretability: https://arxiv.org/abs/1912.09363

## Files

Create or update:

```text
StockMixer/src/recil/context.py
StockMixer/tests/recil/test_context.py
```

Do not reuse full-series min-max normalization from:

```text
StockMixer/src/Enhanced/preprocess_context.py
```

## Steps

1. Implement:

   ```python
   def compute_market_context_raw(close_window, valid_mask=None, eps=1e-8):
       """
       close_window: np.ndarray or torch.Tensor, shape [N, T]
       valid_mask: optional shape [N] or [N, T]
       returns: np.ndarray or torch.Tensor, shape [7]
       """
   ```

2. Context features, in this exact order:
   - `market_return`: mean of valid asset returns in the window.
   - `market_trend`: OLS slope of equal-weighted market return series over time.
   - `market_volatility`: std of equal-weighted market return series.
   - `cross_sectional_dispersion`: std of latest valid cross-sectional returns.
   - `pca_ratio`: first principal component variance ratio over valid asset return matrix.
   - `market_breadth`: fraction of valid latest asset returns greater than zero.
   - `downside_volatility`: std of negative equal-weighted market returns, or zero if none.

3. Validity rules:
   - `close_window` is historical only, shape `[N, T]`; do not include target day.
   - If `valid_mask` is `[N, T]`, an asset is valid only if all values in its window are valid.
   - Exclude non-finite and non-positive close values before computing returns.
   - If fewer than 3 valid assets or fewer than 2 return steps remain, return zeros and avoid NaN.

4. Do not normalize in this function. No train/val/test statistics are allowed here.

5. Add tests:
   - Shape is `(7,)`.
   - All outputs are finite.
   - Volatility and downside volatility are non-negative.
   - Breadth and PCA ratio are within `[0, 1]`.
   - Constant prices give near-zero volatility and finite PCA ratio.
   - Invalid asset is excluded by mask.

## Test

```bash
cd StockMixer
python3 -m pytest tests/recil/test_context.py -q
python3 -m compileall src/recil
```

## Pass criteria

- Tests pass.
- `compute_market_context_raw` never normalizes.
- All edge cases return finite values, not NaN/Inf.

## Expected output

Reusable raw context function with seven causal market-state features.

## Limitations

PCA can be expensive for very large `N`. Use the covariance direction that is cheaper for the window size, and document the chosen implementation.
