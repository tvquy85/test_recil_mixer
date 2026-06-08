from pathlib import Path

import numpy as np
import pytest
import torch

from src.recil.context import CONTEXT_FEATURE_NAMES
from src.recil.data import (
    ReCILDataset,
    build_context_cache,
    default_split_indices,
    load_stockmixer_dataset,
)


def _synthetic_arrays(n=5, d=40, f=5):
    base = np.linspace(10.0, 30.0, d, dtype=np.float32)
    eod = np.zeros((n, d, f), dtype=np.float32)
    for i in range(n):
        for j in range(f):
            eod[i, :, j] = base + i * 0.7 + j * 0.1 + np.sin(np.arange(d) * (0.03 + i * 0.005))
    mask = np.ones((n, d), dtype=np.float32)
    gt = np.zeros((n, d), dtype=np.float32)
    price = eod[:, :, -1].copy()
    gt[:, 1:] = (price[:, 1:] - price[:, :-1]) / price[:, :-1]
    return eod, mask, gt, price


def test_build_context_cache_train_only_and_metadata():
    eod, mask, _, _ = _synthetic_arrays()
    cache = build_context_cache(
        eod,
        mask,
        train_range=(0, 20),
        val_range=(20, 30),
        test_range=(30, 36),
        lookback=4,
    )
    contexts = cache["contexts"]
    metadata = cache["metadata"]
    assert contexts.shape == (37, 7)
    assert cache["raw_contexts"].shape == (37, 7)
    assert np.isfinite(contexts).all()
    assert metadata["lookback"] == 4
    assert metadata["feature_names"] == list(CONTEXT_FEATURE_NAMES)
    assert "same_input_window" in metadata["alignment"]
    assert len(metadata["context_mean"]) == 7
    assert len(metadata["context_std"]) == 7

    train_z = contexts[:20]
    non_constant = cache["raw_contexts"][:20].std(axis=0) > 1e-6
    assert np.allclose(train_z[:, non_constant].mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(train_z[:, non_constant].std(axis=0), 1.0, atol=1e-5)
    assert np.isfinite(contexts[20:]).all()


def test_recil_dataset_item_alignment_and_dtypes():
    eod, mask, gt, price = _synthetic_arrays()
    mask[2, 0:5] = 0.0
    lookback = 4
    steps = 1
    cache = build_context_cache(eod, mask, (0, 20), (20, 30), (30, 36), lookback)
    dataset = ReCILDataset(eod, mask, gt, price, cache, offsets=[0, 1, 2], lookback=lookback, steps=steps)
    item = dataset[0]
    assert set(item) == {"x", "y", "mask", "context", "date_index", "base_price"}
    assert item["x"].shape == (5, lookback, 5)
    assert item["y"].shape == (5,)
    assert item["mask"].shape == (5,)
    assert item["context"].shape == (7,)
    assert item["base_price"].shape == (5,)
    assert item["date_index"] == lookback + steps - 1
    assert item["x"].dtype == torch.float32
    assert item["y"].dtype == torch.float32
    assert item["mask"].dtype == torch.float32
    assert item["context"].dtype == torch.float32
    assert item["base_price"].dtype == torch.float32
    assert item["mask"][2].item() == 0.0
    assert torch.allclose(item["x"], torch.as_tensor(eod[:, :lookback, :]))
    assert torch.allclose(item["y"], torch.as_tensor(gt[:, lookback + steps - 1]))
    assert torch.allclose(item["base_price"], torch.as_tensor(price[:, lookback - 1]))


def test_recil_dataset_rejects_offsets_without_target():
    eod, mask, gt, price = _synthetic_arrays(d=8)
    cache = build_context_cache(eod, mask, (0, 2), (2, 3), (3, 4), lookback=4)
    with pytest.raises(ValueError):
        ReCILDataset(eod, mask, gt, price, cache, offsets=[4], lookback=4, steps=1)


def test_default_split_indices():
    assert default_split_indices("nasdaq", 1245) == (756, 1008)
    assert default_split_indices("sp500", 1611) == (1006, 1259)
    assert default_split_indices("crypto", 1035) == (621, 828)


def test_real_loaders_smoke_shapes():
    root = Path(__file__).resolve().parents[2] / "dataset"
    nasdaq = load_stockmixer_dataset(root, "nasdaq")
    assert nasdaq["eod_data"].shape == (1026, 1245, 5)
    assert nasdaq["mask_data"].shape == (1026, 1245)
    assert nasdaq["gt_data"].shape == (1026, 1245)
    assert nasdaq["price_data"].shape == (1026, 1245)

    sp500 = load_stockmixer_dataset(root, "sp500")
    assert sp500["eod_data"].shape == (474, 1611, 5)
    assert sp500["mask_data"].shape == (474, 1611)
    assert sp500["gt_data"].shape == (474, 1611)
    assert sp500["price_data"].shape == (474, 1611)

    with pytest.raises(ValueError):
        load_stockmixer_dataset(root, "nyse")
