# ReCIL Repo Audit

Date: 2026-06-08

This audit executes `tasks/00_Index.md` and `tasks/01_KiemKeRepoVaNguon.md`.
It is a read-only snapshot of the current workspace before any ReCIL model code
is implemented.

## Task 00 Status

- `tasks/00_Index.md` exists and defines the required execution order.
- Task files `00` through `16` are present under `tasks/`.
- Required task headers are present: `Goal`, `Sources`, `Files`, `Test`, and
  `Pass criteria`.
- Stop-on-fail rule is explicit: do not advance to the next task when a task
  fails.

## Repository Status

- Workspace root: `/mnt/d/Conferences/PRICAI/ReCIL-Mixer`
- No `.git` directory was found by `find . -maxdepth 3 -name .git -type d -print`.
- Literal status for task checks: no .git metadata is available in this
  workspace.
- Do not assume branch creation is possible in this workspace.
- Code lives under `StockMixer/src`, not root `src`.
- Future ReCIL code should use `StockMixer/src/recil`.
- `StockMixer/src/Original/*`, `MoTa.md`, and `ThucThi.md` should remain
  unchanged unless a later task explicitly asks for compatibility edits.

## Local Sources Inspected

- `AGENTS.md`
- `MoTa.md`
- `ThucThi.md`
- `StockMixer/paper.md`
- `StockMixer/README.md`
- `StockMixer/requirements.txt`
- `StockMixer/src/Original/train.py`
- `StockMixer/src/Original/load_data.py`
- `StockMixer/src/Original/model.py`
- `StockMixer/src/Original/evaluator.py`
- `StockMixer/src/Enhanced/preprocess_context.py`
- `StockMixer/src/Enhanced/train_ablation.py`
- `StockMixer/dataset/*`

## External Sources Verified

- PRICAI 2026 CFP: proceedings are Springer LNAI; long papers are 12-16 pages
  including references; Machine Learning, Neural Networks & Deep Learning, Data
  Mining & Knowledge Discovery, Explainable AI, and AI applications are in
  scope. Source: https://2026.pricai.org/calls/call-for-papers
- StockMixer AAAI 2024 official proceedings page: title, authors Jinyong Fan
  and Yanyan Shen, DOI `10.1609/aaai.v38i8.28681`, AAAI 2024 publication, and
  abstract confirm indicator mixing, time mixing, and stock mixing. Source:
  https://ojs.aaai.org/index.php/AAAI/article/view/28681
- Official StockMixer GitHub page redirects to `SJTU-DMTai/StockMixer` and
  lists expected environment notes: Python 3.7, torch approximately 1.10.1,
  numpy approximately 1.21.5, PyYAML, pandas, tqdm, matplotlib. Source:
  https://github.com/SJTU-Quant/StockMixer

## StockMixer Paper Evidence

Local paper file: `StockMixer/paper.md`.

Important facts from the paper:

- StockMixer is an MLP-based stock forecasting architecture with indicator
  mixing, time mixing, and stock mixing.
- StockMixer argues that complex RNN/GNN/Transformer hybrids can overfit
  limited daily stock data and be harder to optimize.
- Time mixing uses multi-scale temporal patches. With lookback `T=16`, the
  paper uses temporal scales `k in {1, 2, 4}`.
- Stock mixing decomposes direct stock-to-stock exchange into
  stock-to-market and market-to-stock mixing through a latent market dimension
  `m`, avoiding graph priors.
- Reported datasets in the paper: NASDAQ, NYSE, and S&P500. Dataset sizes:
  NASDAQ has 1026 stocks; NYSE has 1737 stocks; S&P500 has 474 stocks.
- Paper split statistics: NASDAQ/NYSE have 756 train days, 252 validation days,
  273 test days; S&P500 has 1006 train days, 253 validation days, 352 test days.
- Paper implementation details: 16-day lookback, one stock-mixing layer,
  `alpha=0.1`, learning rate `1e-3`, repeated 3 times.
- The paper defines Rank Information Coefficient as Spearman rank correlation.
  This conflicts with the local evaluator implementation described below.

## Dataset Inventory

Filesystem inventory:

```text
StockMixer/dataset/NASDAQ/NASDAQ_market_metrics_16.pkl
StockMixer/dataset/NASDAQ/NASDAQmarket_metrics_16.pkl
StockMixer/dataset/NASDAQ/eod_data.pkl
StockMixer/dataset/NASDAQ/gt_data.pkl
StockMixer/dataset/NASDAQ/mask_data.pkl
StockMixer/dataset/NASDAQ/price_data.pkl
StockMixer/dataset/NYSE/eod_data.pkl
StockMixer/dataset/NYSE/gt_data.pkl
StockMixer/dataset/NYSE/mask_data.pkl
StockMixer/dataset/NYSE/price_data.pkl
StockMixer/dataset/SP500/SP500.npy
StockMixer/dataset/SP500/SP500market_metrics_16.pkl
StockMixer/dataset/SP500/baseline_data_sp500.npy
StockMixer/dataset/SP500/sp500_ticker.csv
StockMixer/dataset/crypto/cryptomarket_metrics_16.pkl
StockMixer/dataset/crypto/eod_data.pkl
StockMixer/dataset/crypto/gt_data.pkl
StockMixer/dataset/crypto/mask_data.pkl
StockMixer/dataset/crypto/price_data.pkl
```

Size/status summary:

| Dataset | Status in workspace | Notes |
|---|---|---|
| NASDAQ | Present | Main equity dataset. Pickles total about 40 MB. |
| SP500 | Present | `SP500.npy` about 46 MB; original code slices `[:, 915:, :]`. |
| crypto | Present | Optional robustness dataset. |
| NYSE | Not usable now | All four pickle files are zero bytes. Do not use until repaired. |

Shape inspection using `/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe`:

| Dataset/file | Shape | dtype |
|---|---:|---|
| NASDAQ `eod_data.pkl` | `(1026, 1245, 5)` | `float32` |
| NASDAQ `mask_data.pkl` | `(1026, 1245)` | `float32` |
| NASDAQ `gt_data.pkl` | `(1026, 1245)` | `float32` |
| NASDAQ `price_data.pkl` | `(1026, 1245)` | `float32` |
| NASDAQ market metrics | `(1229, 5)` | `float32` |
| SP500 `SP500.npy` | `(474, 2526, 5)` | `float64` |
| SP500 after `[:, 915:, :]` | `(474, 1611, 5)` | `float64` |
| SP500 market metrics | `(1595, 5)` | `float32` |
| crypto `eod_data.pkl` | `(117, 1035, 5)` | `float32` |
| crypto `mask_data.pkl` | `(117, 1035)` | `int8` |
| crypto `gt_data.pkl` | `(117, 1035)` | `float32` |
| crypto `price_data.pkl` | `(117, 1035)` | `float32` |
| crypto market metrics | `(1019, 5)` | `float32` |

## Source-Code Inventory

Main original files:

- `StockMixer/src/Original/load_data.py`: raw CSV loader and relation helpers.
- `StockMixer/src/Original/model.py`: original StockMixer model and original
  loss.
- `StockMixer/src/Original/train.py`: hard-coded NASDAQ training script and
  SP500 special case.
- `StockMixer/src/Original/evaluator.py`: original metrics implementation.

Related improved/experimental files:

- `StockMixer/src/Enhanced/preprocess_context.py`: computes 5-D market context
  and normalizes it over the full available series.
- `StockMixer/src/Enhanced/train_ablation.py`: unified ablation script for
  original/gated variants; useful as reference for dataset path handling.
- `StockMixer/src/GatedContext/preprocess.py`: older 5-D market context logic.
- `StockMixer/src/Gated*` and `StockMixer/src/Enhanced/*`: reference variants,
  not the clean ReCIL target namespace.

## Critical Risks For Later Tasks

### Metric Naming And Masking Risk

`StockMixer/src/Original/evaluator.py` computes:

```text
performance['RIC'] = mean(IC) / std(IC)
```

This is ICIR-like behavior, not Spearman RankIC. In addition, original IC is
computed after multiplying prediction and ground truth by `mask`, which
effectively zero-fills invalid assets before correlation. Later ReCIL metrics
must compute:

```text
IC     = Pearson correlation per day on valid assets only
RankIC = Spearman rank correlation per day on valid assets only
ICIR   = mean(IC_t) / std(IC_t)
```

Do not report new ReCIL metric key `RIC`.

### Context Leakage Risk

`StockMixer/src/Enhanced/preprocess_context.py` computes context for all
windows and then normalizes each context column using full-series min/max.
That includes validation/test windows and is a leakage risk. Later ReCIL
context must use:

```text
raw historical context -> fit scaler on train contexts only -> transform train/val/test
```

### Dataset Availability Risk

`StockMixer/dataset/NYSE/*` files are zero bytes. NYSE should be excluded from
the initial ReCIL implementation and experiments until the data files are
recovered.

### Git Workflow Risk

No `.git` directory was found in this workspace. Any task that suggests branch
creation must be interpreted as documentation-only unless the repository is
restored with Git metadata.

## Environment Decision

Following `AGENTS.md`, missing libraries were checked in this order:

1. Current system `python3`.
2. `/mnt/d/Conferences/NIPS/FinEval`.
3. `/mnt/d/LOBProj/LOBExp`.

System `python3` is available but lacks most ML packages. A reusable environment
exists at:

```text
/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe
```

It has the required packages for dependency-backed inspection:

```text
numpy 1.26.4
pandas 2.1.1
torch 2.5.1+cu121
scipy 1.16.2
yaml 6.0.1
matplotlib 3.8.0
pytest 9.0.3
```

No packages were installed during Tasks 00-02.

## Task 01 Result

Files changed:

```text
StockMixer/docs/repo_audit_recil.md
```

Tests to run:

```bash
test -f StockMixer/docs/repo_audit_recil.md
rg -n "StockMixer/src/recil|no \\.git|RIC|leakage|NYSE|PRICAI|StockMixer AAAI" StockMixer/docs/repo_audit_recil.md
```

Known limitations:

- No source code was modified in this task.
- No package installation was performed.
