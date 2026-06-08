# 04 - Context Scaler Train-Only Va Cache

## Goal

Implement train-only standardization and context cache by split, removing the major leakage risk from full-series context normalization.

## Sources

- `StockMixer/src/Enhanced/preprocess_context.py`
- `StockMixer/src/Enhanced/train_ablation.py`
- `tasks/02_MoiTruongVaDataContract.md`
- `tasks/03_ContextRawKhongLeakage.md`

## Files

Update:

```text
StockMixer/src/recil/context.py
StockMixer/src/recil/data.py
StockMixer/tests/recil/test_context.py
StockMixer/tests/recil/test_data.py
```

## Steps

1. Implement in `context.py`:

   ```python
   class TrainOnlyStandardizer:
       def fit(self, x): ...
       def transform(self, x): ...
       def fit_transform(self, x): ...
       def state_dict(self): ...
       @classmethod
       def from_state_dict(cls, state): ...
   ```

2. Rules:
   - `fit` computes mean/std only on train contexts.
   - Ignore rows containing NaN when fitting, but transformed output must be finite.
   - Replace zero std with `1.0`.
   - `transform` before `fit` raises `RuntimeError`.

3. Implement in `data.py`:

   ```python
   def build_context_cache(eod_data, masks, train_range, val_range, test_range, lookback, close_col=-1):
       """
       Compute raw contexts for usable prediction offsets.
       Fit scaler only on train_range.
       Return normalized context array and metadata.
       """
   ```

4. Alignment:
   - For sample `offset`, input window is `offset:offset+lookback`.
   - Context for that sample must use the same historical input close window, not the target day.
   - Do not index `offset-1` unless you intentionally emulate the old `Enhanced` path and document the difference. Preferred ReCIL convention is context aligned to the same input window.

5. Metadata must include:

   ```text
   context_mean
   context_std
   lookback
   train_range
   val_range
   test_range
   feature_names
   alignment
   ```

6. Add synthetic test with `D=40`, `N=5`, `lookback=4`, train offsets covering the train split, val/test finite.

## Test

```bash
cd StockMixer
python3 -m pytest tests/recil/test_context.py tests/recil/test_data.py -q
python3 -m compileall src/recil
```

## Pass criteria

- Train transformed contexts have mean near `0` and std near `1` for non-constant features.
- Val/test transform uses train statistics and remains finite.
- Metadata records alignment and feature names.
- No scaler is fit on val/test.

## Expected output

A context cache that can be reused by dataset/training without recomputing context every batch.

## Limitations

If train contains too few valid context rows, fail fast with a clear error instead of silently fitting bad statistics.
