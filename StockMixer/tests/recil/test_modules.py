import pytest
import torch

from src.recil.modules import (
    CausalTemporalMixer,
    ContextGatedResidual,
    FiLMModulation,
    IndicatorEncoder,
    MarketContextEncoder,
    PredictionHead,
    RegimeConditionedLowRankExperts,
    RegimeGatedScaleFusion,
    patchify_time,
)


BATCH = 2
ASSETS = 5
TIME = 16
FEATURES = 5
D_MODEL = 64


def _assert_backward(output, input_tensor, module):
    loss = output.sum()
    loss.backward()
    assert input_tensor.grad is not None
    grads = [p.grad for p in module.parameters() if p.requires_grad]
    assert grads
    assert all(g is not None for g in grads)


def test_indicator_encoder_shape_finite_and_backward():
    x = torch.randn(BATCH, ASSETS, TIME, FEATURES, requires_grad=True)
    module = IndicatorEncoder(FEATURES, D_MODEL, dropout=0.0)
    out = module(x)
    assert out.shape == (BATCH, ASSETS, TIME, D_MODEL)
    assert torch.isfinite(out).all()
    _assert_backward(out, x, module)


def test_market_context_encoder_shape_finite_and_backward():
    context = torch.randn(BATCH, 7, requires_grad=True)
    module = MarketContextEncoder(context_dim=7, d_model=D_MODEL, dropout=0.0)
    out = module(context)
    assert out.shape == (BATCH, D_MODEL)
    assert torch.isfinite(out).all()
    _assert_backward(out, context, module)


def test_patchify_time_scale_one_returns_original_values():
    z = torch.randn(BATCH, ASSETS, TIME, D_MODEL)
    patched = patchify_time(z, scale=1)
    assert patched.shape == z.shape
    assert patched.data_ptr() == z.data_ptr()
    assert torch.allclose(patched, z)


def test_patchify_time_scale_two_matches_manual_mean():
    z = torch.randn(BATCH, ASSETS, TIME, D_MODEL)
    patched = patchify_time(z, scale=2)
    expected = z.reshape(BATCH, ASSETS, TIME // 2, 2, D_MODEL).mean(dim=3)
    assert patched.shape == (BATCH, ASSETS, TIME // 2, D_MODEL)
    assert torch.allclose(patched, expected)


def test_patchify_time_raises_on_invalid_inputs():
    z = torch.randn(BATCH, ASSETS, TIME, D_MODEL)
    with pytest.raises(ValueError):
        patchify_time(z, scale=0)
    with pytest.raises(ValueError):
        patchify_time(z[:, :, :-1, :], scale=2)
    with pytest.raises(ValueError):
        patchify_time(torch.randn(BATCH, TIME, D_MODEL), scale=2)


def test_causal_temporal_mixer_full_sequence_shape_finite_and_backward():
    z = torch.randn(BATCH, ASSETS, TIME, D_MODEL, requires_grad=True)
    module = CausalTemporalMixer(D_MODEL, patch_len=3, dropout=0.0)
    out = module(z)
    assert out.shape == (BATCH, ASSETS, D_MODEL)
    assert torch.isfinite(out).all()
    _assert_backward(out, z, module)


def test_causal_temporal_mixer_patched_sequence_shape():
    z = torch.randn(BATCH, ASSETS, TIME, D_MODEL)
    patched = patchify_time(z, scale=2)
    module = CausalTemporalMixer(D_MODEL, patch_len=3, dropout=0.1)
    out = module(patched)
    assert out.shape == (BATCH, ASSETS, D_MODEL)
    assert torch.isfinite(out).all()


def test_prediction_head_shape_finite_and_backward():
    h = torch.randn(BATCH, ASSETS, D_MODEL, requires_grad=True)
    module = PredictionHead(D_MODEL, dropout=0.0)
    out = module(h)
    assert out.shape == (BATCH, ASSETS)
    assert torch.isfinite(out).all()
    _assert_backward(out, h, module)


def test_modules_validate_feature_dimensions():
    with pytest.raises(ValueError):
        IndicatorEncoder(FEATURES, D_MODEL)(torch.randn(BATCH, ASSETS, TIME, FEATURES + 1))
    with pytest.raises(ValueError):
        MarketContextEncoder(7, D_MODEL)(torch.randn(BATCH, 6))
    with pytest.raises(ValueError):
        CausalTemporalMixer(D_MODEL, patch_len=3)(torch.randn(BATCH, ASSETS, TIME, D_MODEL + 1))
    with pytest.raises(ValueError):
        PredictionHead(D_MODEL)(torch.randn(BATCH, ASSETS, D_MODEL + 1))


def test_regime_gated_scale_fusion_shape_weights_and_backward():
    h1 = torch.randn(BATCH, ASSETS, D_MODEL, requires_grad=True)
    h2 = torch.randn(BATCH, ASSETS, D_MODEL, requires_grad=True)
    context = torch.randn(BATCH, D_MODEL, requires_grad=True)
    module = RegimeGatedScaleFusion(D_MODEL, num_scales=2)
    fused, weights = module([h1, h2], context)
    assert fused.shape == (BATCH, ASSETS, D_MODEL)
    assert weights.shape == (BATCH, 2)
    assert torch.isfinite(fused).all()
    assert torch.allclose(weights.sum(dim=-1), torch.ones(BATCH), atol=1e-6)
    (fused.sum() + weights.sum()).backward()
    assert h1.grad is not None
    assert h2.grad is not None
    assert context.grad is not None
    assert all(p.grad is not None for p in module.parameters())


def test_film_modulation_zero_init_is_identity_and_backward():
    h = torch.randn(BATCH, ASSETS, D_MODEL, requires_grad=True)
    context = torch.randn(BATCH, D_MODEL, requires_grad=True)
    module = FiLMModulation(D_MODEL)
    out = module(h, context)
    assert out.shape == h.shape
    assert torch.allclose(out, h, atol=1e-6)
    out.sum().backward()
    assert h.grad is not None
    assert context.grad is not None
    assert all(p.grad is not None for p in module.parameters())


def test_low_rank_experts_shape_router_no_full_asset_matrix_and_backward():
    h = torch.randn(BATCH, ASSETS, D_MODEL, requires_grad=True)
    context = torch.randn(BATCH, D_MODEL, requires_grad=True)
    module = RegimeConditionedLowRankExperts(
        num_assets=ASSETS,
        d_model=D_MODEL,
        market_dim=3,
        num_experts=4,
        dropout=0.0,
    )
    out, router = module(h, context)
    assert out.shape == (BATCH, ASSETS, D_MODEL)
    assert router.shape == (BATCH, 4)
    assert torch.isfinite(out).all()
    assert torch.allclose(router.sum(dim=-1), torch.ones(BATCH), atol=1e-6)
    for name, param in module.named_parameters():
        assert tuple(param.shape[:2]) != (ASSETS, ASSETS), name
    (out.sum() + router.sum()).backward()
    assert h.grad is not None
    assert context.grad is not None
    assert all(p.grad is not None for p in module.parameters())


def test_context_gated_residual_shape_gate_range_and_backward():
    h_base = torch.randn(BATCH, ASSETS, D_MODEL, requires_grad=True)
    h_inter = torch.randn(BATCH, ASSETS, D_MODEL, requires_grad=True)
    context = torch.randn(BATCH, D_MODEL, requires_grad=True)
    module = ContextGatedResidual(D_MODEL)
    out, gate = module(h_base, h_inter, context)
    assert out.shape == h_base.shape
    assert gate.shape == (BATCH, D_MODEL)
    assert torch.isfinite(out).all()
    assert torch.all(gate >= 0.0)
    assert torch.all(gate <= 1.0)
    (out.sum() + gate.sum()).backward()
    assert h_base.grad is not None
    assert h_inter.grad is not None
    assert context.grad is not None
    assert all(p.grad is not None for p in module.parameters())


def test_regime_modules_validate_shapes():
    h = torch.randn(BATCH, ASSETS, D_MODEL)
    context = torch.randn(BATCH, D_MODEL)
    with pytest.raises(ValueError):
        RegimeGatedScaleFusion(D_MODEL, 2)([h], context)
    with pytest.raises(ValueError):
        RegimeGatedScaleFusion(D_MODEL, 1)([h], torch.randn(BATCH, D_MODEL + 1))
    with pytest.raises(ValueError):
        FiLMModulation(D_MODEL)(h, torch.randn(BATCH, D_MODEL + 1))
    with pytest.raises(ValueError):
        RegimeConditionedLowRankExperts(ASSETS, D_MODEL)(torch.randn(BATCH, ASSETS + 1, D_MODEL), context)
    with pytest.raises(ValueError):
        ContextGatedResidual(D_MODEL)(h, torch.randn(BATCH, ASSETS + 1, D_MODEL), context)
