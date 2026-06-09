"""Regression tests for the ReCIL optimization patch."""

from __future__ import annotations

import numpy as np
import torch

from src.recil.metrics import evaluate_predictions
from src.recil.model import ReCILMixer


def _make_batch(batch: int = 2, assets: int = 6, lookback: int = 16, features: int = 5):
    torch.manual_seed(7)
    x = torch.randn(batch, assets, lookback, features)
    context = torch.randn(batch, 7)
    mask = torch.ones(batch, assets)
    mask[:, -2:] = 0.0
    return x, context, mask


def test_variant_specific_parameter_counts_are_not_identical():
    static = ReCILMixer(num_assets=10, num_features=5, variant="static", dropout=0.0)
    full = ReCILMixer(num_assets=10, num_features=5, variant="full", dropout=0.0)
    assert static.active_parameter_count != full.active_parameter_count
    assert static.parameter_report()["module_params"]["static_expert"] > 0
    assert "moe_experts" in full.parameter_report()["module_params"]


def test_invalid_assets_do_not_contaminate_valid_predictions():
    model = ReCILMixer(
        num_assets=6,
        num_features=5,
        variant="full",
        dropout=0.0,
        d_model=16,
        market_dim=8,
        num_experts=3,
        scales=(1, 2, 4),
    )
    model.eval()
    x, context, mask = _make_batch()
    x_corrupted = x.clone()
    x_corrupted[:, -2:, :, :] = 1e6
    with torch.no_grad():
        pred_clean, _ = model(x, context, mask=mask)
        pred_corrupt, _ = model(x_corrupted, context, mask=mask)
    assert torch.allclose(pred_clean[:, :-2], pred_corrupt[:, :-2], atol=1e-5, rtol=1e-5)
    assert torch.all(pred_corrupt[:, -2:] == 0)


def test_metrics_include_rankicir_turnover_and_cost_sharpe():
    rng = np.random.default_rng(3)
    pred = rng.normal(size=(5, 12)).astype(np.float32)
    target = rng.normal(size=(5, 12)).astype(np.float32)
    mask = np.ones((5, 12), dtype=np.float32)
    mask[2, :2] = 0.0
    metrics = evaluate_predictions(pred, target, mask, k=5, transaction_cost_bps=10.0)
    assert "RankICIR" in metrics
    assert "Turnover@5" in metrics
    assert "CostSharpe@5" in metrics
    assert metrics["NumDays"] == 5
