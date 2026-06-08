# 05 - Dataset Item Va Alignment

## Goal

Implement ReCIL dataset loading and sample alignment using `StockMixer/dataset`, while preserving original target semantics.

## Sources

- `StockMixer/src/Original/train.py`
- `StockMixer/src/Original/load_data.py`
- `StockMixer/src/Enhanced/train_ablation.py`
- `tasks/02_MoiTruongVaDataContract.md`
- `tasks/04_ContextScalerTrainOnlyVaCache.md`

## Files

Update:

```text
StockMixer/src/recil/data.py
StockMixer/tests/recil/test_data.py
```

## Steps

1. Implement dataset loading helpers:

   ```python
   def load_stockmixer_dataset(data_root, dataset, steps=1):
       ...
   ```

2. Dataset rules:
   - `dataset="nasdaq"` loads `eod_data.pkl`, `mask_data.pkl`, `gt_data.pkl`, `price_data.pkl`.
   - `dataset="crypto"` loads the same pickle layout.
   - `dataset="sp500"` loads `SP500.npy`, applies `data = data[:, 915:, :]`, creates all-one mask, computes `gt_data` from close prices as original code does.
   - Do not use `NYSE` as a default because files are zero bytes in the current workspace.

3. Implement a dataset class:

   ```python
   class ReCILDataset(torch.utils.data.Dataset):
       def __getitem__(self, idx):
           return {
               "x": Tensor[N, T, F],
               "y": Tensor[N],
               "mask": Tensor[N],
               "context": Tensor[7],
               "date_index": int,
               "base_price": Tensor[N],
           }
   ```

4. Offset convention:

   ```text
   x          = eod_data[:, offset : offset + lookback, :]
   mask       = min(mask_data[:, offset : offset + lookback + steps], axis=1)
   base_price = price_data[:, offset + lookback - 1]
   y          = gt_data[:, offset + lookback + steps - 1]
   date_index = offset + lookback + steps - 1
   context    = context_cache[offset]
   ```

5. Split defaults:
   - NASDAQ: `valid_index=756`, `test_index=1008`.
   - SP500: use same indices only if resulting trade date length is sufficient; otherwise derive 60/20/20 chronological splits and record in config.
   - Crypto: derive 60/20/20 chronological splits unless explicit indices are provided.

6. Tests:
   - Synthetic dataset returns all keys with correct shapes and dtype `float32`.
   - First item date index matches formula.
   - Mask excludes assets invalid anywhere in input+horizon.
   - Context shape is `[7]`.

## Test

```bash
cd StockMixer
python3 -m pytest tests/recil/test_data.py -q
python3 -m compileall src/recil
```

## Pass criteria

- Dataset item contract is stable.
- No off-by-one mismatch with original StockMixer target convention.
- Loader handles NASDAQ and SP500 paths as described.

## Expected output

Training can consume a single unified item schema for synthetic and real datasets.

## Limitations

This task does not train a model. It only guarantees data access and alignment.
