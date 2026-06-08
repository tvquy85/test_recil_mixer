# tests/test_forward_shapes.py
"""
Unit tests for the refactored StockMixer variants.

These tests verify that the forward pass of each architecture returns the
expected shapes and that the parameter counts differ across variants.  To
run the tests simply execute this file directly with ``python`` or use a
test runner such as ``pytest``.
"""

import torch


def test_forward_shapes():
    N, T, F = 8, 16, 12
    m = 4
    depth = 2
    x = torch.randn(N, T, F)
    # Original architecture
    from model_original import StockMixer as OriginalStockMixer
    model_orig = OriginalStockMixer(stocks=N, time_steps=T, channels=F, market=m, scale=3)
    out_orig = model_orig(x)
    assert out_orig.shape == (N, 1)
    # Gated MLP without context
    from model_gated_nocontext import GatedMLPNoContext
    model_nc = GatedMLPNoContext(stocks=N, time_steps=T, channels=F, market_hidden=m, depth=depth, dropout=0.0)
    out_nc = model_nc(x)
    assert out_nc.shape == (N, 1)
    # Gated MLP with context
    from model_gated_withcontext import GatedMLPWithContext
    model_ctx = GatedMLPWithContext(stocks=N, time_steps=T, channels=F, market_hidden=m, depth=depth, ctx_dim=5, dropout=0.0)
    ctx = torch.randn(5)
    out_ctx = model_ctx(x, ctx)
    assert out_ctx.shape == (N, 1)
    # Parameter counts differ
    params_orig = sum(p.numel() for p in model_orig.parameters())
    params_nc = sum(p.numel() for p in model_nc.parameters())
    params_ctx = sum(p.numel() for p in model_ctx.parameters())
    assert params_orig != params_nc or params_nc != params_ctx, "Parameter counts should differ across variants"
