"""Dataset loading and alignment helpers for ReCIL."""

from __future__ import annotations

from pathlib import Path
import pickle
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from .context import (
    CONTEXT_FEATURE_NAMES,
    TrainOnlyStandardizer,
    compute_market_context_raw,
)


def _normalize_dataset_name(dataset: str) -> str:
    name = dataset.lower()
    aliases = {"nasdaq": "NASDAQ", "sp500": "SP500", "crypto": "crypto", "nyse": "NYSE"}
    if name not in aliases:
        raise ValueError(f"unsupported dataset: {dataset}")
    return aliases[name]


def load_stockmixer_dataset(data_root, dataset: str, steps: int = 1):
    """Load a StockMixer-format dataset.

    Returns a dictionary containing ``eod_data``, ``mask_data``, ``gt_data``,
    ``price_data``, ``dataset``, and ``steps``.
    """

    root = Path(data_root)
    dataset_name = _normalize_dataset_name(dataset)

    if dataset_name == "NYSE":
        raise ValueError("NYSE files are zero bytes in this workspace; do not use until repaired")

    if dataset_name == "SP500":
        data = np.load(root / "SP500" / "SP500.npy")
        data = data[:, 915:, :]
        price_data = data[:, :, -1].astype(np.float32)
        mask_data = np.ones((data.shape[0], data.shape[1]), dtype=np.float32)
        eod_data = data.astype(np.float32)
        gt_data = np.zeros((data.shape[0], data.shape[1]), dtype=np.float32)
        for row in range(steps, data.shape[1]):
            prev = data[:, row - steps, -1]
            cur = data[:, row, -1]
            gt_data[:, row] = ((cur - prev) / np.maximum(prev, 1e-8)).astype(np.float32)
        return {
            "dataset": "sp500",
            "eod_data": eod_data,
            "mask_data": mask_data,
            "gt_data": gt_data,
            "price_data": price_data,
            "steps": steps,
        }

    folder = root / dataset_name
    required = ("eod_data.pkl", "mask_data.pkl", "gt_data.pkl", "price_data.pkl")
    arrays = {}
    for filename in required:
        path = folder / filename
        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(f"missing or empty dataset file: {path}")
        with path.open("rb") as f:
            arrays[filename[:-4]] = pickle.load(f)

    return {
        "dataset": dataset_name.lower(),
        "eod_data": np.asarray(arrays["eod_data"], dtype=np.float32),
        "mask_data": np.asarray(arrays["mask_data"], dtype=np.float32),
        "gt_data": np.asarray(arrays["gt_data"], dtype=np.float32),
        "price_data": np.asarray(arrays["price_data"], dtype=np.float32),
        "steps": steps,
    }


def default_split_indices(dataset: str, trade_dates: int):
    """Return chronological split indices as ``(valid_index, test_index)``."""

    name = dataset.lower()
    if name == "nasdaq":
        if trade_dates <= 1008:
            raise ValueError("NASDAQ trade_dates must exceed test_index=1008")
        return 756, 1008
    if name == "sp500":
        if trade_dates > 1259:
            return 1006, 1259
    valid_index = int(trade_dates * 0.6)
    test_index = int(trade_dates * 0.8)
    if valid_index <= 0 or test_index <= valid_index or test_index >= trade_dates:
        raise ValueError(f"cannot derive chronological splits for trade_dates={trade_dates}")
    return valid_index, test_index


def _range_to_offsets(range_like) -> np.ndarray:
    if range_like is None:
        return np.array([], dtype=np.int64)
    if isinstance(range_like, range):
        return np.asarray(list(range_like), dtype=np.int64)
    if isinstance(range_like, tuple) and len(range_like) == 2:
        return np.arange(int(range_like[0]), int(range_like[1]), dtype=np.int64)
    arr = np.asarray(list(range_like) if not isinstance(range_like, np.ndarray) else range_like, dtype=np.int64)
    return arr


def build_context_cache(
    eod_data,
    masks,
    train_range,
    val_range,
    test_range,
    lookback: int,
    close_col: int = -1,
):
    """Build normalized market-context cache keyed by sample offset."""

    eod = np.asarray(eod_data)
    mask_arr = np.asarray(masks)
    if eod.ndim != 3:
        raise ValueError("eod_data must have shape [N, D, F]")
    if mask_arr.shape != eod.shape[:2]:
        raise ValueError("masks must have shape [N, D]")
    if lookback < 3:
        raise ValueError("lookback must be at least 3")

    _, trade_dates, _ = eod.shape
    max_context_offset = trade_dates - lookback
    if max_context_offset < 0:
        raise ValueError("lookback exceeds available trade dates")

    raw = np.zeros((max_context_offset + 1, len(CONTEXT_FEATURE_NAMES)), dtype=np.float32)
    for offset in range(max_context_offset + 1):
        close_window = eod[:, offset : offset + lookback, close_col]
        mask_window = mask_arr[:, offset : offset + lookback]
        raw[offset] = compute_market_context_raw(close_window, mask_window)

    train_offsets = _range_to_offsets(train_range)
    val_offsets = _range_to_offsets(val_range)
    test_offsets = _range_to_offsets(test_range)
    valid_train = train_offsets[(train_offsets >= 0) & (train_offsets <= max_context_offset)]
    if valid_train.size == 0:
        raise ValueError("train_range contains no valid context offsets")

    scaler = TrainOnlyStandardizer().fit(raw[valid_train])
    contexts = scaler.transform(raw)
    metadata = {
        "context_mean": scaler.state_dict()["mean"],
        "context_std": scaler.state_dict()["std"],
        "lookback": int(lookback),
        "train_range": train_offsets.tolist(),
        "val_range": val_offsets.tolist(),
        "test_range": test_offsets.tolist(),
        "feature_names": list(CONTEXT_FEATURE_NAMES),
        "alignment": "same_input_window_offset_to_offset_plus_lookback_exclusive",
        "close_col": int(close_col),
    }
    return {"contexts": contexts, "raw_contexts": raw, "metadata": metadata}


class ReCILDataset(Dataset):
    """One-market-day-per-item dataset preserving StockMixer alignment."""

    def __init__(
        self,
        eod_data,
        mask_data,
        gt_data,
        price_data,
        context_cache,
        offsets: Iterable[int],
        lookback: int,
        steps: int = 1,
    ):
        self.eod_data = np.asarray(eod_data, dtype=np.float32)
        self.mask_data = np.asarray(mask_data, dtype=np.float32)
        self.gt_data = np.asarray(gt_data, dtype=np.float32)
        self.price_data = np.asarray(price_data, dtype=np.float32)
        self.contexts = (
            np.asarray(context_cache["contexts"], dtype=np.float32)
            if isinstance(context_cache, dict)
            else np.asarray(context_cache, dtype=np.float32)
        )
        self.offsets = np.asarray(list(offsets), dtype=np.int64)
        self.lookback = int(lookback)
        self.steps = int(steps)

        if self.eod_data.ndim != 3:
            raise ValueError("eod_data must have shape [N, D, F]")
        if self.mask_data.shape != self.eod_data.shape[:2]:
            raise ValueError("mask_data must have shape [N, D]")
        if self.gt_data.shape != self.eod_data.shape[:2]:
            raise ValueError("gt_data must have shape [N, D]")
        if self.price_data.shape != self.eod_data.shape[:2]:
            raise ValueError("price_data must have shape [N, D]")

        max_offset = self.eod_data.shape[1] - self.lookback - self.steps
        if np.any(self.offsets < 0) or np.any(self.offsets > max_offset):
            raise ValueError("offsets include samples without full input+horizon")
        if self.contexts.shape[0] <= int(self.offsets.max(initial=0)):
            raise ValueError("context_cache does not cover all offsets")

    def __len__(self) -> int:
        return int(self.offsets.shape[0])

    def __getitem__(self, idx: int):
        offset = int(self.offsets[idx])
        end = offset + self.lookback
        target_idx = offset + self.lookback + self.steps - 1
        mask = np.min(self.mask_data[:, offset : end + self.steps], axis=1).astype(np.float32)
        item = {
            "x": torch.as_tensor(self.eod_data[:, offset:end, :], dtype=torch.float32),
            "y": torch.as_tensor(self.gt_data[:, target_idx], dtype=torch.float32),
            "mask": torch.as_tensor(mask, dtype=torch.float32),
            "context": torch.as_tensor(self.contexts[offset], dtype=torch.float32),
            "date_index": target_idx,
            "base_price": torch.as_tensor(self.price_data[:, end - 1], dtype=torch.float32),
        }
        return item
