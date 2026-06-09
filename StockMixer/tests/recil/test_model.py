import pytest
import torch

from src.recil.model import ReCILMixer


BATCH = 2
ASSETS = 20
TIME = 16
FEATURES = 5
D_MODEL = 64
CONTEXT_DIM = 7
VARIANTS = ("static", "context_only", "single_gate", "moe", "full")


def _inputs():
    x = torch.randn(BATCH, ASSETS, TIME, FEATURES)
    context = torch.randn(BATCH, CONTEXT_DIM)
    return x, context


def _model(variant):
    return ReCILMixer(
        num_assets=ASSETS,
        num_features=FEATURES,
        d_model=D_MODEL,
        context_dim=CONTEXT_DIM,
        market_dim=8,
        num_experts=4,
        scales=(1, 2, 4),
        dropout=0.0,
        variant=variant,
    )


@pytest.mark.parametrize("variant", VARIANTS)
def test_recil_mixer_variant_forward_aux_shapes_and_backward(variant):
    x, context = _inputs()
    x.requires_grad_(True)
    model = _model(variant)
    pred, aux = model(x, context)

    assert pred.shape == (BATCH, ASSETS)
    assert torch.isfinite(pred).all()
    assert set(aux) == {"scale_weights", "router_weights", "context_gate"}

    if variant == "full":
        assert aux["scale_weights"].shape == (BATCH, 3)
        assert aux["router_weights"].shape == (BATCH, 4)
        assert aux["context_gate"].shape == (BATCH, D_MODEL)
    elif variant == "moe":
        assert aux["scale_weights"] is None
        assert aux["router_weights"].shape == (BATCH, 4)
        assert aux["context_gate"] is None
    elif variant == "single_gate":
        assert aux["scale_weights"] is None
        assert aux["router_weights"] is None
        assert aux["context_gate"].shape == (BATCH, D_MODEL)
    else:
        assert aux == {"scale_weights": None, "router_weights": None, "context_gate": None}

    pred.sum().backward()
    assert x.grad is not None
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads


def test_static_accepts_missing_context():
    x, _ = _inputs()
    pred, aux = _model("static")(x, context=None)
    assert pred.shape == (BATCH, ASSETS)
    assert aux == {"scale_weights": None, "router_weights": None, "context_gate": None}


@pytest.mark.parametrize("variant", ("context_only", "single_gate", "moe", "full"))
def test_context_conditioned_variants_require_context(variant):
    x, _ = _inputs()
    with pytest.raises(ValueError):
        _model(variant)(x, context=None)


def test_invalid_variant_raises():
    with pytest.raises(ValueError):
        ReCILMixer(num_assets=ASSETS, num_features=FEATURES, variant="bad_variant")


def test_bad_input_shapes_raise():
    x, context = _inputs()
    model = _model("full")
    with pytest.raises(ValueError):
        model(torch.randn(BATCH, TIME, FEATURES), context)
    with pytest.raises(ValueError):
        model(torch.randn(BATCH, ASSETS + 1, TIME, FEATURES), context)
    with pytest.raises(ValueError):
        model(torch.randn(BATCH, ASSETS, TIME, FEATURES + 1), context)
    with pytest.raises(ValueError):
        model(torch.randn(BATCH, ASSETS, TIME - 1, FEATURES), context)
    with pytest.raises(ValueError):
        model(x, torch.randn(BATCH, CONTEXT_DIM + 1))


def test_mask_is_accepted_but_does_not_change_interface():
    x, context = _inputs()
    mask = torch.ones(BATCH, ASSETS)
    pred, aux = _model("full")(x, context, mask=mask)
    assert pred.shape == (BATCH, ASSETS)
    assert set(aux) == {"scale_weights", "router_weights", "context_gate"}
    with pytest.raises(ValueError):
        _model("full")(x, context, mask=torch.ones(BATCH, ASSETS + 1))


def test_forward_signature_outputs_return_scores_without_base_price():
    x, context = _inputs()
    pred, _ = _model("context_only")(x, context=context)
    assert pred.shape == (BATCH, ASSETS)
    assert pred.ndim == 2


def test_router_controls_and_interaction_strength_are_reported_and_applied():
    x, context = _inputs()
    model = ReCILMixer(
        num_assets=ASSETS,
        num_features=FEATURES,
        d_model=D_MODEL,
        context_dim=CONTEXT_DIM,
        market_dim=8,
        num_experts=4,
        scales=(1, 2, 4),
        dropout=0.0,
        variant="moe",
        router_init="small_normal",
        router_temperature=0.5,
    )
    report = model.parameter_report()
    assert report["router_init"] == "small_normal"
    assert report["router_temperature"] == 0.5

    model.set_interaction_strength(0.0)
    pred_base, aux_base = model(x, context)
    model.set_interaction_strength(1.0)
    pred_full, aux_full = model(x, context)
    assert aux_base["router_weights"].shape == (BATCH, 4)
    assert aux_full["router_weights"].shape == (BATCH, 4)
    assert not torch.allclose(pred_base, pred_full)

    with pytest.raises(ValueError):
        model.set_interaction_strength(1.5)
