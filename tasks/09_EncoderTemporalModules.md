# 09 - Encoder Temporal Modules

## Goal

Implement basic encoder and temporal modules for ReCIL-Mixer: indicator encoder, market context encoder, time patching, causal temporal mixer, and prediction head.

## Sources

- `StockMixer/src/Original/model.py`
- `StockMixer/src/Enhanced/model_original_refactor.py`
- MLP-Mixer: https://papers.nips.cc/paper/2021/hash/cba0a4ee5ccd02fda0fe3f9a3e7b89fe-Abstract.html
- PatchTST patching idea: https://arxiv.org/abs/2211.14730

## Files

Update:

```text
StockMixer/src/recil/modules.py
StockMixer/tests/recil/test_modules.py
```

## Steps

1. Implement `IndicatorEncoder`:

   ```python
   class IndicatorEncoder(nn.Module):
       def __init__(self, num_features, d_model, dropout=0.1): ...
       def forward(self, x):  # [B, N, T, F] -> [B, N, T, D]
   ```

   Architecture: `Linear(F,D) -> GELU -> Dropout -> Linear(D,D) -> LayerNorm(D)`.

2. Implement `MarketContextEncoder`:

   ```python
   class MarketContextEncoder(nn.Module):
       def __init__(self, context_dim=7, d_model=64, dropout=0.1): ...
       def forward(self, context):  # [B,7] -> [B,D]
   ```

3. Implement `patchify_time(z, scale)`:
   - Input `[B,N,T,D]`.
   - `scale=1` returns original sequence.
   - `scale=2` returns `[B,N,T/2,D]` by reshape+mean.
   - Raise `ValueError` if `T % scale != 0`.

4. Implement `CausalTemporalMixer`:

   ```python
   class CausalTemporalMixer(nn.Module):
       def __init__(self, d_model, patch_len, dropout=0.1): ...
       def forward(self, z):  # [B,N,P,D] -> [B,N,D]
   ```

   Use a lightweight time MLP or causal Conv1D over the lookback window. It may consume the whole historical input window but must not use target-day data.

5. Implement `PredictionHead`:

   ```python
   class PredictionHead(nn.Module):
       def __init__(self, d_model, dropout=0.1): ...
       def forward(self, h):  # [B,N,D] -> [B,N]
   ```

6. Tests:
   - Shapes for `[B=2,N=5,T=16,F=5,D=64]`.
   - Backward pass works for each module.
   - `patchify_time` raises on incompatible scale.
   - Outputs are finite.

## Test

```bash
cd StockMixer
python3 -m pytest tests/recil/test_modules.py -q
python3 -m compileall src/recil
```

## Pass criteria

- All shape and backward tests pass.
- Temporal module output shape is `[B,N,D]`.
- No module returns NaN on random input.

## Expected output

Reusable encoder/temporal building blocks for all ReCIL variants.

## Limitations

Do not over-engineer temporal attention here. ReCIL's main contribution is regime-conditioned interaction, not a heavy temporal backbone.
