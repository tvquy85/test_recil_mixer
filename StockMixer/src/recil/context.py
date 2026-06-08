"""Causal market-context utilities for ReCIL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:  # Torch is available in the project environment, but keep import optional.
    import torch
except Exception:  # pragma: no cover - exercised only in minimal environments.
    torch = None  # type: ignore[assignment]


CONTEXT_FEATURE_NAMES = (
    "market_return",
    "market_trend",
    "market_volatility",
    "cross_sectional_dispersion",
    "pca_ratio",
    "market_breadth",
    "downside_volatility",
)


def _is_torch_tensor(x: Any) -> bool:
    return torch is not None and isinstance(x, torch.Tensor)


def _to_numpy(x: Any) -> np.ndarray:
    if _is_torch_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _restore_type(values: np.ndarray, reference: Any):
    values = values.astype(np.float32, copy=False)
    if _is_torch_tensor(reference):
        return torch.as_tensor(values, dtype=reference.dtype, device=reference.device)
    return values


def compute_market_context_raw(close_window, valid_mask=None, eps: float = 1e-8):
    """Compute raw causal market context from one historical close window.

    Parameters
    ----------
    close_window:
        Close prices with shape ``[N, T]``. This must already be restricted to
        historical input days and must not include the prediction target day.
    valid_mask:
        Optional validity mask with shape ``[N]`` or ``[N, T]``.
    eps:
        Numerical stability constant.

    Returns
    -------
    np.ndarray or torch.Tensor
        Seven raw, unnormalized context features in ``CONTEXT_FEATURE_NAMES``
        order. The output type follows ``close_window``.
    """

    close_np = _to_numpy(close_window).astype(np.float64, copy=False)
    if close_np.ndim != 2:
        raise ValueError(f"close_window must have shape [N, T], got {close_np.shape}")

    n_assets, n_steps = close_np.shape
    if n_steps < 3 or n_assets < 3:
        return _restore_type(np.zeros(len(CONTEXT_FEATURE_NAMES)), close_window)

    valid_assets = np.all(np.isfinite(close_np) & (close_np > eps), axis=1)
    if valid_mask is not None:
        mask_np = _to_numpy(valid_mask)
        if mask_np.ndim == 1:
            if mask_np.shape[0] != n_assets:
                raise ValueError("1-D valid_mask must have length N")
            valid_assets &= mask_np > 0.5
        elif mask_np.ndim == 2:
            if mask_np.shape != close_np.shape:
                raise ValueError("2-D valid_mask must have shape [N, T]")
            valid_assets &= np.all(mask_np > 0.5, axis=1)
        else:
            raise ValueError("valid_mask must have shape [N] or [N, T]")

    close_valid = close_np[valid_assets]
    if close_valid.shape[0] < 3:
        return _restore_type(np.zeros(len(CONTEXT_FEATURE_NAMES)), close_window)

    returns = close_valid[:, 1:] / np.maximum(close_valid[:, :-1], eps) - 1.0
    returns = returns[np.all(np.isfinite(returns), axis=1)]
    if returns.shape[0] < 3 or returns.shape[1] < 2:
        return _restore_type(np.zeros(len(CONTEXT_FEATURE_NAMES)), close_window)

    market_series = returns.mean(axis=0)
    market_return = float(returns.mean())

    x = np.arange(market_series.shape[0], dtype=np.float64)
    x_centered = x - x.mean()
    y_centered = market_series - market_series.mean()
    denom = float(np.mean(x_centered * x_centered))
    market_trend = 0.0 if denom <= eps else float(np.mean(x_centered * y_centered) / denom)

    market_volatility = float(np.std(market_series))
    latest_returns = returns[:, -1]
    cross_sectional_dispersion = float(np.std(latest_returns))

    centered = returns - returns.mean(axis=1, keepdims=True)
    total_var = float(np.sum(centered * centered))
    if total_var <= eps:
        pca_ratio = 0.0
    else:
        # Window length is small, so this T x T covariance is cheaper than N x N.
        cov_time = centered.T @ centered
        eigvals = np.linalg.eigvalsh(cov_time)
        pca_ratio = float(np.clip(eigvals[-1] / max(float(eigvals.sum()), eps), 0.0, 1.0))

    market_breadth = float(np.mean(latest_returns > 0.0))
    downside = market_series[market_series < 0.0]
    downside_volatility = float(np.std(downside)) if downside.size > 0 else 0.0

    values = np.array(
        [
            market_return,
            market_trend,
            market_volatility,
            cross_sectional_dispersion,
            pca_ratio,
            market_breadth,
            downside_volatility,
        ],
        dtype=np.float64,
    )
    values = np.where(np.isfinite(values), values, 0.0)
    return _restore_type(values, close_window)


@dataclass
class TrainOnlyStandardizer:
    """Standardize context features using train statistics only."""

    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None

    def fit(self, x):
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError("x must have shape [num_samples, num_features]")
        finite_rows = np.all(np.isfinite(arr), axis=1)
        arr = arr[finite_rows]
        if arr.size == 0:
            raise ValueError("cannot fit TrainOnlyStandardizer with no finite rows")
        self.mean_ = arr.mean(axis=0)
        std = arr.std(axis=0)
        self.std_ = np.where(std > 1e-12, std, 1.0)
        return self

    def transform(self, x):
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("TrainOnlyStandardizer.transform called before fit")
        arr = np.asarray(x, dtype=np.float64)
        transformed = (arr - self.mean_) / self.std_
        transformed = np.where(np.isfinite(transformed), transformed, 0.0)
        return transformed.astype(np.float32)

    def fit_transform(self, x):
        return self.fit(x).transform(x)

    def state_dict(self):
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("TrainOnlyStandardizer.state_dict called before fit")
        return {
            "mean": self.mean_.astype(np.float32).copy(),
            "std": self.std_.astype(np.float32).copy(),
        }

    @classmethod
    def from_state_dict(cls, state):
        mean = np.asarray(state["mean"], dtype=np.float64)
        std = np.asarray(state["std"], dtype=np.float64)
        if mean.shape != std.shape:
            raise ValueError("mean and std must have the same shape")
        return cls(mean_=mean, std_=np.where(std > 1e-12, std, 1.0))
