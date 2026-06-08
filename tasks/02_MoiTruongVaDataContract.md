# 02 - Moi Truong Va Data Contract

## Goal

Chuan hoa moi truong chay va viet data contract dua tren loader/training goc. Task nay phai lam ro `python3` la command mac dinh trong workspace hien tai, va dependency can co truoc khi load pickle/numpy.

## Sources

- `AGENTS.md`
- `StockMixer/requirements.txt`
- `StockMixer/src/Original/train.py`
- `StockMixer/src/Original/load_data.py`
- `StockMixer/src/Enhanced/train_ablation.py`
- `StockMixer/dataset`

## Files

Create:

```text
StockMixer/docs/data_contract_recil.md
```

Optionally create if missing:

```text
StockMixer/src/recil/__init__.py
```

Do not implement loaders yet.

## Steps

1. Inspect Python availability:

   ```bash
   which python3
   python3 --version
   python3 -m pip --version
   ```

2. Check package imports without installing anything automatically:

   ```bash
   python3 - <<'PY'
   mods = ["numpy", "pandas", "torch", "scipy", "yaml", "matplotlib"]
   for name in mods:
       try:
           mod = __import__(name)
           print(name, getattr(mod, "__version__", "ok"))
       except Exception as exc:
           print(name, "MISSING", exc)
   PY
   ```

3. If dependencies are missing, document an environment setup command but do not install blindly. Follow `AGENTS.md`: first search current/nearby reusable envs, then install only if needed.

4. Write the real StockMixer data contract:
   - NASDAQ pickle files are under `StockMixer/dataset/NASDAQ`.
   - Original NASDAQ constants: `stock_num=1026`, `lookback_length=16`, `valid_index=756`, `test_index=1008`, `steps=1`, `fea_num=5`.
   - SP500 original loading uses `StockMixer/dataset/SP500/SP500.npy`, then `data = data[:, 915:, :]`.
   - Original batch convention:

     ```text
     x      = eod_data[:, offset : offset + lookback, :]
     mask   = min(mask_data[:, offset : offset + lookback + steps], axis=1)
     price  = price_data[:, offset + lookback - 1]
     y      = gt_data[:, offset + lookback + steps - 1]
     ```

   - Prediction/evaluation arrays are conventionally asset-major: `[N, num_days]`.
   - New ReCIL dataset may internally batch as `[B, N, T, F]`, but must preserve the same target date semantics.

5. Document expected dependency set for ReCIL:
   - Required: `numpy`, `pandas`, `torch`, `pytest`.
   - Recommended: `scipy` for Spearman ranking, with fallback implementation if unavailable.
   - Optional: `matplotlib` for figures.

## Test

```bash
test -f StockMixer/docs/data_contract_recil.md
rg -n "python3|offset|lookback|valid_index|test_index|SP500|915|mask_data|gt_data" StockMixer/docs/data_contract_recil.md
```

## Pass criteria

- Data contract states exact offset formulas.
- Environment section records current dependency status.
- No source code behavior is changed.

## Expected output

A data contract that prevents off-by-one, leakage, and wrong-shape bugs in later tasks.

## Limitations

This task may not load all pickle arrays if dependencies are unavailable. That is acceptable if documented.
