"""Mask-aware financial metrics for ReCIL."""

from __future__ import annotations

import numpy as np


def _valid_vectors(pred, target, mask):
    pred_arr = np.asarray(pred, dtype=np.float64).reshape(-1)
    target_arr = np.asarray(target, dtype=np.float64).reshape(-1)
    mask_arr = np.asarray(mask, dtype=np.float64).reshape(-1)
    if pred_arr.shape != target_arr.shape or pred_arr.shape != mask_arr.shape:
        raise ValueError("pred, target, and mask must have the same flattened shape")
    valid = np.isfinite(pred_arr) & np.isfinite(target_arr) & np.isfinite(mask_arr) & (mask_arr > 0.5)
    return pred_arr[valid], target_arr[valid]


def pearson_corr_masked(pred, target, mask, eps: float = 1e-8, min_valid: int = 3) -> float:
    """Compute single-day Pearson IC using only valid masked assets."""

    pred_valid, target_valid = _valid_vectors(pred, target, mask)
    if pred_valid.size < min_valid:
        return float("nan")

    pred_centered = pred_valid - pred_valid.mean()
    target_centered = target_valid - target_valid.mean()
    pred_std = float(np.sqrt(np.mean(pred_centered * pred_centered)))
    target_std = float(np.sqrt(np.mean(target_centered * target_centered)))
    if pred_std <= eps or target_std <= eps:
        return float("nan")

    corr = float(np.mean(pred_centered * target_centered) / (pred_std * target_std))
    return float(np.clip(corr, -1.0, 1.0))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return deterministic average ranks for a 1-D vector."""

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.shape[0], dtype=np.float64)

    start = 0
    while start < sorted_values.shape[0]:
        end = start + 1
        while end < sorted_values.shape[0] and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = 0.5 * (start + end - 1)
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def spearman_corr_masked(pred, target, mask, eps: float = 1e-8, min_valid: int = 3) -> float:
    """Compute single-day Spearman RankIC using average ranks on valid assets."""

    pred_valid, target_valid = _valid_vectors(pred, target, mask)
    if pred_valid.size < min_valid:
        return float("nan")
    pred_rank = _average_ranks(pred_valid)
    target_rank = _average_ranks(target_valid)
    return pearson_corr_masked(pred_rank, target_rank, np.ones_like(pred_rank), eps=eps, min_valid=min_valid)


def _as_day_major(preds, targets, masks, asset_major: bool):
    pred_arr = np.asarray(preds, dtype=np.float64)
    target_arr = np.asarray(targets, dtype=np.float64)
    mask_arr = np.asarray(masks, dtype=np.float64)
    if pred_arr.shape != target_arr.shape or pred_arr.shape != mask_arr.shape:
        raise ValueError("preds, targets, and masks must have the same shape")
    if pred_arr.ndim == 1:
        pred_arr = pred_arr.reshape(1, -1)
        target_arr = target_arr.reshape(1, -1)
        mask_arr = mask_arr.reshape(1, -1)
    elif pred_arr.ndim == 2:
        if asset_major:
            pred_arr = pred_arr.T
            target_arr = target_arr.T
            mask_arr = mask_arr.T
    else:
        raise ValueError("preds, targets, and masks must have shape [N], [D, N], or [N, D]")
    return pred_arr, target_arr, mask_arr


def _valid_indices(pred, target, mask) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred_arr = np.asarray(pred, dtype=np.float64).reshape(-1)
    target_arr = np.asarray(target, dtype=np.float64).reshape(-1)
    mask_arr = np.asarray(mask, dtype=np.float64).reshape(-1)
    if pred_arr.shape != target_arr.shape or pred_arr.shape != mask_arr.shape:
        raise ValueError("pred, target, and mask must have the same flattened shape")
    valid = np.flatnonzero(
        np.isfinite(pred_arr) & np.isfinite(target_arr) & np.isfinite(mask_arr) & (mask_arr > 0.5)
    )
    return pred_arr, target_arr, valid


def _top_k_valid_indices(values: np.ndarray, valid: np.ndarray, k_eff: int) -> np.ndarray:
    order = np.argsort(values[valid], kind="mergesort")
    return valid[order[-k_eff:]]


def precision_at_k(pred, target, mask, k: int = 10) -> float:
    """Compute top-K overlap precision among valid assets only."""

    pred_arr, target_arr, valid = _valid_indices(pred, target, mask)
    if valid.size == 0:
        return float("nan")
    k_eff = min(int(k), int(valid.size))
    if k_eff <= 0:
        raise ValueError("k must be positive")
    pred_top = set(_top_k_valid_indices(pred_arr, valid, k_eff).tolist())
    target_top = set(_top_k_valid_indices(target_arr, valid, k_eff).tolist())
    return float(len(pred_top & target_top) / k_eff)


def long_only_daily_return(pred, target_return, mask, k: int = 10) -> float:
    """Average realized return of predicted top-K valid assets."""

    pred_arr, target_arr, valid = _valid_indices(pred, target_return, mask)
    if valid.size == 0:
        return float("nan")
    k_eff = min(int(k), int(valid.size))
    if k_eff <= 0:
        raise ValueError("k must be positive")
    pred_top = _top_k_valid_indices(pred_arr, valid, k_eff)
    return float(target_arr[pred_top].mean())


def sharpe_ratio(daily_returns, annualization: int = 252, eps: float = 1e-8) -> float:
    """Compute annualized Sharpe ratio for finite daily returns."""

    returns = np.asarray(daily_returns, dtype=np.float64).reshape(-1)
    returns = returns[np.isfinite(returns)]
    if returns.size == 0:
        return float("nan")
    std = max(float(returns.std()), eps)
    return float(returns.mean() / std * np.sqrt(float(annualization)))


def compute_ic_series(preds, targets, masks, asset_major: bool = False, eps: float = 1e-8, min_valid: int = 3):
    """Compute per-day IC and RankIC series.

    The default multi-day layout is day-major ``[D, N]``. Set
    ``asset_major=True`` only for old StockMixer evaluator arrays shaped
    ``[N, D]``.
    """

    pred_arr, target_arr, mask_arr = _as_day_major(preds, targets, masks, asset_major)
    ic = np.empty(pred_arr.shape[0], dtype=np.float32)
    rankic = np.empty(pred_arr.shape[0], dtype=np.float32)
    for day in range(pred_arr.shape[0]):
        ic[day] = pearson_corr_masked(pred_arr[day], target_arr[day], mask_arr[day], eps=eps, min_valid=min_valid)
        rankic[day] = spearman_corr_masked(pred_arr[day], target_arr[day], mask_arr[day], eps=eps, min_valid=min_valid)
    return {"IC": ic, "RankIC": rankic}


def _nanmean(values: np.ndarray) -> float:
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return float("nan")
    return float(valid.mean())


def _icir(values: np.ndarray, eps: float) -> float:
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return float("nan")
    std = float(valid.std())
    if std <= eps:
        return float("nan")
    return float(valid.mean() / std)


def summarize_ic(ic_series, rankic_series, eps: float = 1e-8):
    """Summarize IC/RankIC series with NaN-aware means and information ratios."""

    ic = np.asarray(ic_series, dtype=np.float64).reshape(-1)
    rankic = np.asarray(rankic_series, dtype=np.float64).reshape(-1)
    return {
        "IC": _nanmean(ic),
        "RankIC": _nanmean(rankic),
        "ICIR": _icir(ic, eps),
        "RankICIR": _icir(rankic, eps),
        "num_valid_days": int(np.isfinite(ic).sum()),
        "num_rankic_valid_days": int(np.isfinite(rankic).sum()),
    }


def _masked_mse(preds: np.ndarray, targets: np.ndarray, masks: np.ndarray) -> float:
    valid = np.isfinite(preds) & np.isfinite(targets) & np.isfinite(masks) & (masks > 0.5)
    if not np.any(valid):
        return float("nan")
    diff = preds[valid] - targets[valid]
    return float(np.mean(diff * diff))


def evaluate_predictions(
    preds,
    targets,
    masks,
    k: int = 10,
    asset_major: bool = False,
    eps: float = 1e-8,
    min_valid: int = 3,
    annualization: int = 252,
):
    """Evaluate predictions with clean mask-aware ReCIL metric names."""

    pred_arr, target_arr, mask_arr = _as_day_major(preds, targets, masks, asset_major)
    ic_series = compute_ic_series(
        pred_arr,
        target_arr,
        mask_arr,
        asset_major=False,
        eps=eps,
        min_valid=min_valid,
    )
    ic_summary = summarize_ic(ic_series["IC"], ic_series["RankIC"], eps=eps)

    precision_values = np.empty(pred_arr.shape[0], dtype=np.float32)
    daily_returns = np.empty(pred_arr.shape[0], dtype=np.float32)
    for day in range(pred_arr.shape[0]):
        precision_values[day] = precision_at_k(pred_arr[day], target_arr[day], mask_arr[day], k=k)
        daily_returns[day] = long_only_daily_return(pred_arr[day], target_arr[day], mask_arr[day], k=k)

    return {
        "mse": _masked_mse(pred_arr, target_arr, mask_arr),
        "IC": ic_summary["IC"],
        "RankIC": ic_summary["RankIC"],
        "ICIR": ic_summary["ICIR"],
        f"Precision@{int(k)}": _nanmean(precision_values),
        "Sharpe": sharpe_ratio(daily_returns, annualization=annualization, eps=eps),
        "num_valid_days": ic_summary["num_valid_days"],
        "num_days": int(pred_arr.shape[0]),
    }
