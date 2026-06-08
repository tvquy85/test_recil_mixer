# 01 - Kiem Ke Repo Va Nguon

## Goal

Audit repo, dataset, code goc va nguon hoc thuat de tao baseline truth truoc khi viet code moi. Task nay phai ghi ro hien trang khong co `.git` trong workspace hien tai, nen khong duoc gia dinh co the tao branch.

## Sources

- `MoTa.md`
- `ThucThi.md`
- `AGENTS.md`
- `StockMixer/README.md`
- `StockMixer/src/Original/train.py`
- `StockMixer/src/Original/load_data.py`
- `StockMixer/src/Original/model.py`
- `StockMixer/src/Original/evaluator.py`
- `StockMixer/src/Enhanced/preprocess_context.py`
- StockMixer AAAI 2024: https://mlanthology.org/aaai/2024/fan2024aaai-stockmixer/
- Official StockMixer repo: https://github.com/SJTU-DMTai/StockMixer
- PRICAI 2026 CFP: https://2026.pricai.org/calls/call-for-papers

## Files

Create:

```text
StockMixer/docs/repo_audit_recil.md
```

Do not edit:

```text
StockMixer/src/Original/*
MoTa.md
ThucThi.md
```

## Steps

1. Check repository roots and whether `.git` exists:

   ```bash
   find . -maxdepth 3 -name .git -type d -print
   ```

2. Inventory data files:

   ```bash
   find StockMixer/dataset -maxdepth 3 -type f -print | sort
   ls -lh StockMixer/dataset/NASDAQ StockMixer/dataset/SP500 StockMixer/dataset/crypto StockMixer/dataset/NYSE
   ```

3. Inventory model, training, evaluator, and context-related files:

   ```bash
   find StockMixer/src -maxdepth 3 -type f -print | sort
   rg -n "market_state|context|RIC|RankIC|ICIR|spearman|market_metrics|valid_index|test_index" StockMixer/src
   ```

4. Document key facts:
   - Code lives under `StockMixer/src`, not root `src`.
   - New ReCIL namespace should be `StockMixer/src/recil`.
   - Current workspace may not be a Git repo.
   - `Original/evaluator.py` uses `RIC = mean(IC) / std(IC)`, which is ICIR, not RankIC.
   - `Enhanced/preprocess_context.py` normalizes context over the full series, which is a leakage risk.
   - `StockMixer/dataset/NYSE` files are zero bytes in this workspace and should not be a main dataset until repaired.

5. Record trusted sources and why each is used:
   - StockMixer for MLP-based indicator/time/stock mixing.
   - PRICAI CFP for page format and topic fit.
   - FiLM, gMLP, TFT, PatchTST, RankNet for architectural or loss motivation.

## Test

```bash
test -f StockMixer/docs/repo_audit_recil.md
rg -n "StockMixer/src/recil|no \\.git|RIC|leakage|NYSE|PRICAI|StockMixer AAAI" StockMixer/docs/repo_audit_recil.md
```

## Pass criteria

- Audit file exists and includes repo layout, dataset inventory, code inventory, and source list.
- Audit does not claim a Git branch was created unless `.git` actually exists.
- Audit explicitly flags metric naming and context normalization leakage risks.

## Expected output

`StockMixer/docs/repo_audit_recil.md` becomes the source-of-truth snapshot for later tasks.

## Limitations

If Python dependencies are missing, do not force data loading in this task. Record that shape-level inspection must be finalized after Task 02.
