# 10 - Regime Low-Rank Experts

## Goal

Implement context-conditioned modules: scale fusion, FiLM modulation, low-rank interaction experts, and context-gated residual.

## Sources

- `MoTa.md`
- `ThucThi.md`
- `StockMixer/src/Original/model.py`
- FiLM conditioning: https://arxiv.org/abs/1709.07871
- gMLP gating motivation: https://arxiv.org/abs/2105.08050

## Files

Update:

```text
StockMixer/src/recil/modules.py
StockMixer/tests/recil/test_modules.py
```

## Steps

1. Implement `RegimeGatedScaleFusion`:

   ```python
   class RegimeGatedScaleFusion(nn.Module):
       def __init__(self, d_model, num_scales): ...
       def forward(self, h_scales, context_emb):
           # h_scales: list[[B,N,D]]
           # context_emb: [B,D]
           # returns fused [B,N,D], scale_weights [B,S]
   ```

   `scale_weights = softmax(Linear(context_emb))`.

2. Implement `FiLMModulation`:

   ```python
   class FiLMModulation(nn.Module):
       def __init__(self, d_model): ...
       def forward(self, h, context_emb):  # [B,N,D], [B,D] -> [B,N,D]
   ```

   Use `h * (1 + gamma) + beta`. Initialize final FiLM projection to zeros so initial behavior is close to identity.

3. Implement `RegimeConditionedLowRankExperts`:

   ```python
   class RegimeConditionedLowRankExperts(nn.Module):
       def __init__(self, num_assets, d_model, market_dim=32, num_experts=4, dropout=0.1): ...
       def forward(self, h, context_emb):
           # returns h_inter [B,N,D], router_weights [B,K]
   ```

   Each expert maps across asset axis:

   ```text
   h [B,N,D] -> transpose [B,D,N] -> Linear(N,m) -> GELU -> Linear(m,N) -> [B,N,D]
   ```

   Do not use full `N x N` interaction matrices.

4. Implement `ContextGatedResidual`:

   ```python
   class ContextGatedResidual(nn.Module):
       def __init__(self, d_model): ...
       def forward(self, h_base, h_inter, context_emb):
           # returns h_out [B,N,D], gate [B,D]
   ```

   `h_out = h_base + sigmoid(MLP(context_emb))[:,None,:] * (h_inter - h_base)`.

5. Tests:
   - Scale weights shape `[B,S]` and row sums equal `1`.
   - Router weights shape `[B,K]` and row sums equal `1`.
   - Gate values are in `[0,1]`.
   - FiLM zero-init returns output close to input.
   - Backward pass works.

## Test

```bash
cd StockMixer
python3 -m pytest tests/recil/test_modules.py -q
python3 -m compileall src/recil
```

## Pass criteria

- All module tests pass.
- Low-rank experts scale as `O(K * B * D * N * market_dim)`, not `O(N^2)`.
- Aux weights are returned for later interpretability.

## Expected output

Core adaptive interaction modules for ReCIL-Mixer.

## Limitations

If router collapses to one expert in later experiments, address it in training/analysis with entropy diagnostics rather than changing this module prematurely.
