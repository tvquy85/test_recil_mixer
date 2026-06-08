# ReCIL Data And Environment Contract

Date: 2026-06-08

This contract executes `tasks/02_MoiTruongVaDataContract.md`. It locks the
environment choice and the dataset indexing semantics that later ReCIL code
must preserve.

## Environment

System Python:

```text
/usr/bin/python3
Python 3.12.3
pip 24.0
```

System `python3` dependency status:

```text
numpy MISSING
pandas MISSING
torch MISSING
scipy MISSING
yaml 6.0.1
matplotlib MISSING
pytest MISSING
```

Per `AGENTS.md`, do not install packages blindly. A reusable environment was
found and should be used for dependency-backed checks:

```text
/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe
Python 3.11.5
```

Reusable environment package status:

```text
numpy 1.26.4
pandas 2.1.1
torch 2.5.1+cu121
scipy 1.16.2
yaml 6.0.1
matplotlib 3.8.0
pytest 9.0.3
```

Recommended command pattern for later dependency-backed checks:

```bash
/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe -m pytest ...
/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe -m compileall ...
```

Task files still mention `python3` as the generic command, but in this
workspace `LOBExp/.venv` is the correct dependency-backed interpreter.

## Original StockMixer Constants

From `StockMixer/src/Original/train.py`:

```text
market_name     = NASDAQ
stock_num       = 1026
lookback_length = 16
epochs          = 100
valid_index     = 756
test_index      = 1008
fea_num         = 5
market_num      = 20
steps           = 1
learning_rate   = 0.001
alpha           = 0.1
scale_factor    = 3
activation      = HardSwish
```

These constants match the local StockMixer paper for the main NASDAQ setup:
1026 stocks, 16-day lookback, and 756/252/273 train/validation/test day split.

## Dataset Layout

Use `StockMixer/dataset` as the data root.

NASDAQ:

```text
StockMixer/dataset/NASDAQ/eod_data.pkl
StockMixer/dataset/NASDAQ/mask_data.pkl
StockMixer/dataset/NASDAQ/gt_data.pkl
StockMixer/dataset/NASDAQ/price_data.pkl
```

SP500:

```text
StockMixer/dataset/SP500/SP500.npy
StockMixer/dataset/SP500/SP500market_metrics_16.pkl
StockMixer/dataset/SP500/sp500_ticker.csv
```

Crypto:

```text
StockMixer/dataset/crypto/eod_data.pkl
StockMixer/dataset/crypto/mask_data.pkl
StockMixer/dataset/crypto/gt_data.pkl
StockMixer/dataset/crypto/price_data.pkl
```

NYSE:

```text
StockMixer/dataset/NYSE/eod_data.pkl
StockMixer/dataset/NYSE/mask_data.pkl
StockMixer/dataset/NYSE/gt_data.pkl
StockMixer/dataset/NYSE/price_data.pkl
```

NYSE files are zero bytes in this workspace and must not be used as a default
experiment dataset until repaired.

## Original Data Loading Semantics

### NASDAQ And Pickle Datasets

For non-SP500 datasets, original code loads pickles from:

```text
StockMixer/dataset/{market_name}/eod_data.pkl
StockMixer/dataset/{market_name}/mask_data.pkl
StockMixer/dataset/{market_name}/gt_data.pkl
StockMixer/dataset/{market_name}/price_data.pkl
```

Expected array meanings:

```text
eod_data   -> [N, D, F], historical features
mask_data  -> [N, D], 1 means valid asset/day
gt_data    -> [N, D], realized one-step return target
price_data -> [N, D], base close price
```

Confirmed NASDAQ shapes in this workspace:

```text
eod_data.pkl   -> (1026, 1245, 5), float32
mask_data.pkl  -> (1026, 1245), float32
gt_data.pkl    -> (1026, 1245), float32
price_data.pkl -> (1026, 1245), float32
```

### SP500

Original code special-cases SP500:

```python
data = np.load("../dataset/SP500/SP500.npy")
data = data[:, 915:, :]
price_data = data[:, :, -1]
mask_data = np.ones((data.shape[0], data.shape[1]))
eod_data = data
gt_data = np.zeros((data.shape[0], data.shape[1]))
for ticket in range(data.shape[0]):
    for row in range(1, data.shape[1]):
        gt_data[ticket][row] = (
            data[ticket][row][-1] - data[ticket][row - steps][-1]
        ) / data[ticket][row - steps][-1]
```

Later ReCIL loaders must preserve the `data[:, 915:, :]` slice unless an
explicit experiment config states otherwise.

Confirmed SP500 shape in this workspace:

```text
SP500.npy before slice -> (474, 2526, 5), float64
SP500.npy after slice  -> (474, 1611, 5), float64
```

Confirmed crypto shapes in this workspace:

```text
eod_data.pkl   -> (117, 1035, 5), float32
mask_data.pkl  -> (117, 1035), int8
gt_data.pkl    -> (117, 1035), float32
price_data.pkl -> (117, 1035), float32
```

## Batch Alignment Contract

For a sample whose input window starts at `offset`, with `lookback=16` and
`steps=1`, preserve the original StockMixer convention:

```text
x          = eod_data[:, offset : offset + lookback, :]
mask       = min(mask_data[:, offset : offset + lookback + steps], axis=1)
base_price = price_data[:, offset + lookback - 1]
y          = gt_data[:, offset + lookback + steps - 1]
date_index = offset + lookback + steps - 1
```

Interpretation:

- `x` contains only historical input days.
- `base_price` is the last close price inside the input window.
- `y` is the realized return at the prediction target day.
- `mask` requires validity through both input window and prediction horizon.
- `date_index` points to the target day used for labels/evaluation.

This alignment is the mandatory baseline for ReCIL. Any context feature must be
computed from the historical input window only and must not include the target
day.

## Split Contract

NASDAQ default:

```text
train offsets should be derived before valid_index = 756
validation target days are evaluated over [valid_index, test_index)
test target days are evaluated over [test_index, trade_dates)
```

Original validation loop maps target-day ranges back to input offsets:

```text
begin  = start_index - lookback - steps + 1
finish = end_index   - lookback - steps + 1
```

For `lookback=16` and `steps=1`:

```text
validation offsets start at 740 for start_index=756
test offsets start at 992 for start_index=1008
```

This is expected because the input window must end before the target day.

SP500:

- Paper split is 1006 train days, 253 validation days, 352 test days.
- Original code still contains NASDAQ constants in `Original/train.py`; later
  ReCIL code should derive or configure SP500 split explicitly instead of
  blindly reusing NASDAQ indices.

Crypto:

- No paper split is defined in StockMixer.
- Later ReCIL code should use a chronological split such as 60/20/20 unless an
  explicit config provides indices.

## Shape Contract For ReCIL

Original StockMixer model/training processes one market-day sample at a time:

```text
input to original model: [N, T, F]
prediction:              [N, 1]
evaluator arrays:         [N, num_days]
```

ReCIL may use batched tensors:

```text
x:        [B, N, T, F]
y:        [B, N]
mask:     [B, N]
context:  [B, 7]
pred:     [B, N]
```

Evaluation outputs saved for analysis may use day-major `[D, N]`, but functions
must document the layout. If compatibility with old evaluator arrays is needed,
use explicit `asset_major=True` rather than relying on implicit transposes.

## Metric Contract For Later Tasks

The StockMixer paper defines:

```text
IC  = average Pearson correlation
RIC = average Spearman rank correlation
```

The local original evaluator does not implement this correctly. It reports:

```text
RIC = mean(IC_t) / std(IC_t)
```

Later ReCIL metrics must use the clean names:

```text
IC
RankIC
ICIR
Precision@10
Sharpe
num_valid_days
```

Do not report new ReCIL metrics under `RIC`.

## Context Contract For Later Tasks

Existing `Enhanced/preprocess_context.py` returns 5 normalized context metrics:

```text
mean_ret, slope, real_vol, dispersion, pca_ratio
```

It normalizes using full-series min/max, which is a leakage risk. Later ReCIL
tasks must compute raw historical context first, then fit normalization only on
train contexts and transform validation/test with train statistics.

Target ReCIL context shape:

```text
context_dim = 7
```

Expected raw features:

```text
market_return
market_trend
market_volatility
cross_sectional_dispersion
pca_ratio
market_breadth
downside_volatility
```

## Task 02 Result

Files changed:

```text
StockMixer/docs/data_contract_recil.md
StockMixer/src/recil/__init__.py
```

Tests to run:

```bash
test -f StockMixer/docs/data_contract_recil.md
rg -n "python3|LOBExp|offset|lookback|valid_index|test_index|SP500|915|mask_data|gt_data" StockMixer/docs/data_contract_recil.md
```

Known limitations:

- No loader was implemented in this task.
- No package installation was performed.
- ReCIL implementation must still add tests before using these contracts in
  training.
