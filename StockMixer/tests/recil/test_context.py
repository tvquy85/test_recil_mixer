import numpy as np
import pytest
import torch

from src.recil.context import (
    CONTEXT_FEATURE_NAMES,
    TrainOnlyStandardizer,
    compute_market_context_raw,
)


def test_compute_market_context_raw_shape_and_bounds():
    close = np.array(
        [
            [10.0, 11.0, 12.0, 13.0, 14.0],
            [20.0, 19.0, 18.0, 17.0, 16.0],
            [30.0, 30.5, 31.0, 31.5, 32.0],
            [40.0, 39.0, 41.0, 40.0, 42.0],
        ],
        dtype=np.float32,
    )
    ctx = compute_market_context_raw(close)
    assert ctx.shape == (7,)
    assert len(CONTEXT_FEATURE_NAMES) == 7
    assert np.all(np.isfinite(ctx))
    assert ctx[2] >= 0.0
    assert ctx[4] >= 0.0 and ctx[4] <= 1.0
    assert ctx[5] >= 0.0 and ctx[5] <= 1.0
    assert ctx[6] >= 0.0


def test_compute_market_context_raw_constant_prices_are_safe():
    close = np.ones((5, 6), dtype=np.float32) * 100.0
    ctx = compute_market_context_raw(close)
    assert np.all(np.isfinite(ctx))
    assert ctx[2] == pytest.approx(0.0)
    assert ctx[3] == pytest.approx(0.0)
    assert ctx[4] == pytest.approx(0.0)


def test_compute_market_context_raw_excludes_invalid_assets():
    close = np.array(
        [
            [10.0, 11.0, 12.0, 13.0],
            [10.0, 9.0, 8.0, 7.0],
            [5.0, 5.5, 6.0, 6.5],
            [100.0, 0.0, 100.0, 100.0],
        ],
        dtype=np.float32,
    )
    mask = np.ones_like(close, dtype=np.float32)
    ctx_with_bad = compute_market_context_raw(close, mask)
    ctx_without_bad = compute_market_context_raw(close[:3], mask[:3])
    assert np.allclose(ctx_with_bad, ctx_without_bad)


def test_compute_market_context_raw_torch_roundtrip():
    close = torch.tensor(
        [
            [10.0, 11.0, 12.0, 13.0],
            [10.0, 9.0, 8.0, 7.0],
            [5.0, 5.5, 6.0, 6.5],
        ]
    )
    ctx = compute_market_context_raw(close)
    assert isinstance(ctx, torch.Tensor)
    assert ctx.shape == (7,)
    assert torch.isfinite(ctx).all()


def test_train_only_standardizer_behaviour():
    train = np.array(
        [
            [1.0, 2.0, 5.0],
            [2.0, 4.0, 5.0],
            [3.0, 6.0, 5.0],
            [4.0, 8.0, 5.0],
        ],
        dtype=np.float32,
    )
    val = np.array([[10.0, 20.0, 5.0]], dtype=np.float32)
    scaler = TrainOnlyStandardizer()
    with pytest.raises(RuntimeError):
        scaler.transform(train)
    train_z = scaler.fit_transform(train)
    assert np.allclose(train_z[:, :2].mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(train_z[:, :2].std(axis=0), 1.0, atol=1e-6)
    assert np.allclose(train_z[:, 2], 0.0)
    val_z = scaler.transform(val)
    assert np.isfinite(val_z).all()
    assert val_z[0, 0] > train_z[:, 0].max()


def test_train_only_standardizer_state_dict_roundtrip():
    train = np.arange(20, dtype=np.float32).reshape(5, 4)
    scaler = TrainOnlyStandardizer().fit(train)
    restored = TrainOnlyStandardizer.from_state_dict(scaler.state_dict())
    assert np.allclose(restored.transform(train), scaler.transform(train))
