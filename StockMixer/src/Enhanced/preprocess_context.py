# preprocess_context.py
"""
preprocess_context.py
======================

This module provides functions to compute a small set of market context
features from raw closing price data.  The context features are intended
to capture broad market characteristics such as trend, volatility,
dispersion and systemic correlation.  Two implementations are provided:

``market_state_from_closes``
    A NumPy implementation mirroring the original function in ``preprocess.py``.

``market_state_from_closes_torch``
    A PyTorch implementation that can run on CPU or GPU.  When supplied with a
    ``torch.Tensor`` this version carries out all operations using torch
    primitives.  It is useful when you wish to compute context on the fly
    inside a training loop without converting between NumPy and torch.

Both functions return a normalised matrix of shape ``(num_windows, 5)``
where each row corresponds to a sliding window of returns and each column
corresponds to one of the five context features.
"""

from __future__ import annotations

import numpy as np
import torch


def market_state_from_closes(closes: np.ndarray, close_col: int = -1, window: int = 16) -> np.ndarray:
    """Compute context metrics from closing prices using NumPy.

    Parameters
    ----------
    closes : np.ndarray
        Array of shape ``(N, T, F)`` containing price histories for N assets.
    close_col : int, default -1
        Index of the closing price within the feature dimension.
    window : int, default 16
        Length of the sliding window over which to compute metrics.

    Returns
    -------
    metrics_norm : np.ndarray
        Normalised context metrics of shape ``(T - window, 5)`` where
        each row corresponds to a window and the columns are [mean_ret,
        slope, real_vol, dispersion, pca_ratio].
    """
    close_px = closes[:, :, close_col]  # (N, T)
    # Compute log returns
    rets = np.log(close_px[:, 1:] / (close_px[:, :-1] + 1e-8))  # add small epsilon for stability

    N, Tm1 = rets.shape
    W = window
    num_win = Tm1 - W + 1
    metrics = np.zeros((num_win, 5), dtype=np.float32)
    # Precompute time axis for slope
    x = np.arange(W, dtype=np.float32)
    var_x = x.var()

    for k in range(num_win):
        win = rets[:, k:k + W]  # (N, W)
        idx_series = win.mean(axis=0)  # equally weighted market index (W,)
        # 1) mean return
        mean_ret = win.mean()
        # 2) slope (OLS on time vs index)
        cov_x = ((x - x.mean()) * (idx_series - idx_series.mean())).mean()
        slope = cov_x / (var_x + 1e-8)
        # 3) realised volatility
        real_vol = idx_series.std()
        # 4) dispersion (average cross sectional std)
        disp = win.std(axis=0).mean()
        # 5) PCA ratio (variance explained by first principal component)
        cov_stk = np.cov(win, bias=True)
        eigvals = np.linalg.eigvalsh(cov_stk)
        total_var = eigvals.sum()
        pca_ratio = 0.0 if total_var == 0.0 else eigvals[-1] / total_var
        metrics[k] = [mean_ret, slope, real_vol, disp, pca_ratio]
    # Normalise each column to [0,1]
    min_vals = metrics.min(axis=0)
    max_vals = metrics.max(axis=0)
    # Avoid division by zero by replacing zeros with ones
    denom = np.where(max_vals > min_vals, max_vals - min_vals, 1.0)
    metrics_norm = (metrics - min_vals) / denom
    return metrics_norm


def market_state_from_closes_torch(closes: torch.Tensor, close_col: int = -1, window: int = 16) -> torch.Tensor:
    """Compute context metrics from closing prices using PyTorch.

    This function parallels ``market_state_from_closes`` but operates on
    ``torch.Tensor`` inputs and returns a tensor.  It can execute on the
    GPU when ``closes`` resides on a CUDA device.  Eigenvalue computations
    are performed with ``torch.linalg.eigvalsh`` which is supported on
    GPU.

    Parameters
    ----------
    closes : torch.Tensor
        Tensor of shape ``(N, T, F)`` containing price histories.  The
        tensor may reside on any device.
    close_col : int, default -1
        Index of the closing price within the feature dimension.
    window : int, default 16
        Length of the sliding window.

    Returns
    -------
    metrics_norm : torch.Tensor
        A tensor of shape ``(T - window, 5)`` containing normalised
        context metrics.  Each column is individually scaled to [0,1].
    """
    # Extract closing prices and compute log returns
    close_px = closes[:, :, close_col]  # (N, T)
    eps = 1e-8
    rets = torch.log(close_px[:, 1:] / (close_px[:, :-1] + eps))
    N, Tm1 = rets.shape
    W = window
    num_win = Tm1 - W + 1
    # Precompute time axis for slope
    x = torch.arange(W, dtype=rets.dtype, device=rets.device)
    var_x = x.var()
    # Prepare tensor to accumulate metrics
    metrics = torch.zeros((num_win, 5), dtype=rets.dtype, device=rets.device)
    for k in range(num_win):
        win = rets[:, k:k + W]  # (N, W)
        idx_series = win.mean(dim=0)  # (W,)
        # 1) mean return
        mean_ret = win.mean()
        # 2) slope
        cov_x = ((x - x.mean()) * (idx_series - idx_series.mean())).mean()
        slope = cov_x / (var_x + 1e-8)
        # 3) realised volatility
        real_vol = idx_series.std()
        # 4) dispersion
        disp = win.std(dim=0).mean()
        # 5) PCA ratio
        # Compute covariance matrix across stocks
        cov_stk = torch.cov(win, correction=0)
        # eigvalsh returns eigenvalues in ascending order
        eigvals = torch.linalg.eigvalsh(cov_stk)
        total_var = eigvals.sum()
        pca_ratio = torch.tensor(0.0, dtype=rets.dtype, device=rets.device)
        if total_var > 0:
            pca_ratio = eigvals[-1] / total_var
        metrics[k] = torch.stack([mean_ret, slope, real_vol, disp, pca_ratio])
    # Normalise each column to [0,1]
    min_vals = metrics.min(dim=0).values
    max_vals = metrics.max(dim=0).values
    denom = torch.where(max_vals > min_vals, max_vals - min_vals, torch.ones_like(max_vals))
    metrics_norm = (metrics - min_vals) / denom
    return metrics_norm
