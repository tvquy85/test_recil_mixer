# 11 - ReCILMixer Va Variants

## Goal

Assemble the full `ReCILMixer` model and five ablation variants M0-M4 without writing five separate model families.

## Sources

- `tasks/09_EncoderTemporalModules.md`
- `tasks/10_RegimeLowRankExperts.md`
- `ThucThi.md`
- StockMixer AAAI 2024: https://mlanthology.org/aaai/2024/fan2024aaai-stockmixer/

## Files

Update:

```text
StockMixer/src/recil/model.py
StockMixer/tests/recil/test_model.py
```

## Steps

1. Implement:

   ```python
   class ReCILMixer(nn.Module):
       def __init__(
           self,
           num_assets,
           num_features,
           d_model=64,
           context_dim=7,
           market_dim=32,
           num_experts=4,
           scales=(1, 2, 4),
           dropout=0.1,
           variant="full",
       ): ...

       def forward(self, x, context=None, mask=None):
           # x: [B,N,T,F], context: [B,7]
           # returns pred [B,N], aux dict
   ```

2. Supported variants:

   ```text
   static       -> M0 Static-LRI: no context conditioning, static single low-rank interaction
   context_only -> M1 Context-Predictor: context affects prediction representation only
   single_gate  -> M2 Context-Gated Single Expert: one low-rank expert with context gate
   moe          -> M3 MoE-LRI: context-routed low-rank experts, no scale gate
   full         -> M4 ReCIL-Full: scale gate + FiLM + MoE + context gate
   ```

3. Forward behavior:
   - Always encode indicators and temporal scales.
   - If variant does not use context, allow `context=None` and use zero context embedding where needed.
   - For `full`, return aux keys:

     ```text
     scale_weights
     router_weights
     context_gate
     ```

   - For variants without a component, return `None` for the missing aux key, not a fake value.

4. Prediction output:
   - ReCIL should output return scores directly `[B,N]`.
   - Do not output price levels requiring `(prediction - base_price) / base_price`; that old behavior belongs to StockMixer baseline.

5. Tests:
   - Every variant forward passes on `x=[2,20,16,5]`, `context=[2,7]`.
   - Output shape `[2,20]`.
   - Aux shapes match expected component usage.
   - Backward works for every variant.
   - Invalid variant raises `ValueError`.

## Test

```bash
cd StockMixer
python3 -m pytest tests/recil/test_model.py -q
python3 -m compileall src/recil
```

## Pass criteria

- All variants share one stable interface.
- Output is a return score, compatible with new losses/metrics.
- Aux is saved-ready for interpretability.

## Expected output

A configurable ReCIL model usable by training and experiment runner.

## Limitations

This task validates shapes and gradients only. It does not prove predictive quality.
