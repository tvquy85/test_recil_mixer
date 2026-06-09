"""Mask-aware financial metrics for ReCIL.

The default API remains backward-compatible with the pre-patch evaluator while
optionally exposing RankICIR, turnover, and transaction-cost diagnostics.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _to_numpy(array) -> np.ndarray:
    if hasattr(array, "detach"):
        array = array.detach().cpu().numpy()
    return np.asarray(array)


def _valid_vectors(pred, target, mask):
    pred_arr = _to_numpy(pred).astype(np.float64).reshape(-1)
    target_arr = _to_numpy(target).astype(np.float64).reshape(-1)
    mask_arr = _to_numpy(mask).astype(np.float64).reshape(-1)
    if pred_arr.shape != target_arr.shape or pred_arr.shape != mask_arr.shape:
        raise ValueError("pred, target, and mask must have the same flattened shape")
    valid = np.isfinite(pred_arr) & np.isfinite(target_arr) & np.isfinite(mask_arr) & (mask_arr > 0.5)
    return pred_arr[valid], target_arr[valid]


def pearson_corr_masked(pred, target, mask, eps: float = 1e-8, min_valid: int = 3) -> float:
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
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.shape[0], dtype=np.float64)
    start = 0
    while start < sorted_values.shape[0]:
        end = start + 1
        while end < sorted_values.shape[0] and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def spearman_corr_masked(pred, target, mask, eps: float = 1e-8, min_valid: int = 3) -> float:
    pred_valid, target_valid = _valid_vectors(pred, target, mask)
    if pred_valid.size < min_valid:
        return float("nan")
    pred_rank = _average_ranks(pred_valid)
    target_rank = _average_ranks(target_valid)
    return pearson_corr_masked(pred_rank, target_rank, np.ones_like(pred_rank), eps=eps, min_valid=min_valid)


def _as_day_major(preds, targets, masks, asset_major: bool = False):
    pred_arr = _to_numpy(preds).astype(np.float64)
    target_arr = _to_numpy(targets).astype(np.float64)
    mask_arr = _to_numpy(masks).astype(np.float64)
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
        pred_arr = pred_arr.reshape(-1, pred_arr.shape[-1])
        target_arr = target_arr.reshape(-1, target_arr.shape[-1])
        mask_arr = mask_arr.reshape(-1, mask_arr.shape[-1])
    return pred_arr, target_arr, mask_arr


def _valid_indices(pred, target, mask):
    pred_arr = _to_numpy(pred).astype(np.float64).reshape(-1)
    target_arr = _to_numpy(target).astype(np.float64).reshape(-1)
    mask_arr = _to_numpy(mask).astype(np.float64).reshape(-1)
    if pred_arr.shape != target_arr.shape or pred_arr.shape != mask_arr.shape:
        raise ValueError("pred, target, and mask must have the same flattened shape")
    valid = np.flatnonzero(np.isfinite(pred_arr) & np.isfinite(target_arr) & np.isfinite(mask_arr) & (mask_arr > 0.5))
    return pred_arr, target_arr, valid


def _top_k_valid_indices(values: np.ndarray, valid: np.ndarray, k_eff: int) -> np.ndarray:
    order = np.argsort(values[valid], kind="mergesort")
    return valid[order[-k_eff:]]


def precision_at_k(pred, target, mask, k: int = 10) -> float:
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
    pred_arr, target_arr, valid = _valid_indices(pred, target_return, mask)
    if valid.size == 0:
        return float("nan")
    k_eff = min(int(k), int(valid.size))
    if k_eff <= 0:
        raise ValueError("k must be positive")
    pred_top = _top_k_valid_indices(pred_arr, valid, k_eff)
    return float(target_arr[pred_top].mean())


def _top_k_sets(pred_arr: np.ndarray, target_arr: np.ndarray, mask_arr: np.ndarray, k: int) -> list[set[int]]:
    selections: list[set[int]] = []
    for day in range(pred_arr.shape[0]):
        _, _, valid = _valid_indices(pred_arr[day], target_arr[day], mask_arr[day])
        if valid.size == 0:
            selections.append(set())
            continue
        k_eff = min(int(k), int(valid.size))
        selections.append(set(_top_k_valid_indices(pred_arr[day], valid, k_eff).tolist()))
    return selections


def topk_turnover(preds, targets, masks, k: int = 10) -> np.ndarray:
    pred_arr, target_arr, mask_arr = _as_day_major(preds, targets, masks)
    selections = _top_k_sets(pred_arr, target_arr, mask_arr, k)
    values = np.empty(len(selections), dtype=np.float32)
    previous: set[int] | None = None
    for idx, current in enumerate(selections):
        if not current:
            values[idx] = np.nan
        elif previous is None or not previous:
            values[idx] = 0.0
        else:
            values[idx] = 1.0 - len(current & previous) / max(len(current), 1)
        if current:
            previous = current
    return values


def sharpe_ratio(daily_returns: Sequence[float], annualization: int = 252, eps: float = 1e-8) -> float:
    returns = np.asarray(daily_returns, dtype=np.float64).reshape(-1)
    returns = returns[np.isfinite(returns)]
    if returns.size == 0:
        return float("nan")
    std = max(float(returns.std()), eps)
    return float(returns.mean() / std * np.sqrt(float(annualization)))


def compute_ic_series(preds, targets, masks, asset_major: bool = False, eps: float = 1e-8, min_valid: int = 3):
    pred_arr, target_arr, mask_arr = _as_day_major(preds, targets, masks, asset_major=asset_major)
    ic = np.empty(pred_arr.shape[0], dtype=np.float32)
    rankic = np.empty(pred_arr.shape[0], dtype=np.float32)
    for day in range(pred_arr.shape[0]):
        ic[day] = pearson_corr_masked(pred_arr[day], target_arr[day], mask_arr[day], eps=eps, min_valid=min_valid)
        rankic[day] = spearman_corr_masked(pred_arr[day], target_arr[day], mask_arr[day], eps=eps, min_valid=min_valid)
    return {"IC": ic, "RankIC": rankic}


def _nanmean(values) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return float("nan")
    return float(valid.mean())


def _icir(values, eps: float) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return float("nan")
    std = float(valid.std())
    if std <= eps:
        return float("nan")
    return float(valid.mean() / std)


def summarize_ic(ic_series, rankic_series, eps: float = 1e-8):
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


def _original_zero_fill_corr(pred, target, mask, eps: float = 1e-8) -> float:
    pred_arr = _to_numpy(pred).astype(np.float64).reshape(-1)
    target_arr = _to_numpy(target).astype(np.float64).reshape(-1)
    mask_arr = _to_numpy(mask).astype(np.float64).reshape(-1)
    if pred_arr.shape != target_arr.shape or pred_arr.shape != mask_arr.shape:
        raise ValueError("pred, target, and mask must have the same flattened shape")
    finite = np.isfinite(pred_arr) & np.isfinite(target_arr) & np.isfinite(mask_arr)
    pred_filled = np.where(finite, pred_arr * mask_arr, 0.0)
    target_filled = np.where(finite, target_arr * mask_arr, 0.0)
    pred_centered = pred_filled - pred_filled.mean()
    target_centered = target_filled - target_filled.mean()
    pred_std = float(np.sqrt(np.mean(pred_centered * pred_centered)))
    target_std = float(np.sqrt(np.mean(target_centered * target_centered)))
    if pred_std <= eps or target_std <= eps:
        return float("nan")
    return float(np.clip(np.mean(pred_centered * target_centered) / (pred_std * target_std), -1.0, 1.0))


def compute_original_ic_series(preds, targets, masks, asset_major: bool = False, eps: float = 1e-8) -> np.ndarray:
    """StockMixer Original-compatible daily IC.

    Original/evaluator.py computes pandas correlation after multiplying both
    prediction and ground-truth arrays by mask. That zero-fills invalid assets
    instead of removing them, so this helper intentionally preserves that
    historical behavior for fair baseline comparison.
    """

    pred_arr, target_arr, mask_arr = _as_day_major(preds, targets, masks, asset_major=asset_major)
    ic = np.empty(pred_arr.shape[0], dtype=np.float32)
    for day in range(pred_arr.shape[0]):
        ic[day] = _original_zero_fill_corr(pred_arr[day], target_arr[day], mask_arr[day], eps=eps)
    return ic


def original_positive_precision_at_k(pred, target, mask, k: int = 10) -> float:
    """Original prec_10: positive realized-return rate inside predicted top-k.

    The Original evaluator divides by the requested k, not by the effective
    number of valid assets. This mirrors that behavior.
    """

    if int(k) <= 0:
        raise ValueError("k must be positive")
    pred_arr, target_arr, valid = _valid_indices(pred, target, mask)
    if valid.size == 0:
        return float("nan")
    k_eff = min(int(k), int(valid.size))
    pred_top = _top_k_valid_indices(pred_arr, valid, k_eff)
    return float(np.sum(target_arr[pred_top] >= 0.0) / int(k))


def original_topk_daily_return(pred, target_return, mask, k: int = 5) -> float:
    """Original top-k long return used by sharpe5.

    Original/evaluator.py divides the sum of selected realized returns by fixed
    k. Datasets normally have at least k valid assets; when they do not, the
    fixed denominator is kept for exact compatibility.
    """

    if int(k) <= 0:
        raise ValueError("k must be positive")
    pred_arr, target_arr, valid = _valid_indices(pred, target_return, mask)
    if valid.size == 0:
        return float("nan")
    k_eff = min(int(k), int(valid.size))
    pred_top = _top_k_valid_indices(pred_arr, valid, k_eff)
    return float(np.sum(target_arr[pred_top]) / int(k))


def evaluate_original_stockmixer_metrics(
    preds,
    targets,
    masks,
    asset_major: bool = False,
    eps: float = 1e-8,
):
    """Return metrics compatible with StockMixer/src/Original/evaluator.py.

    Names use an ``Original`` prefix because the legacy key ``RIC`` is ICIR,
    not Spearman RankIC.
    """

    pred_arr, target_arr, mask_arr = _as_day_major(preds, targets, masks, asset_major=asset_major)
    valid_sum = float(np.sum(np.where(np.isfinite(mask_arr), mask_arr, 0.0)))
    if valid_sum <= eps:
        original_mse = float("nan")
    else:
        diff = np.where(np.isfinite(pred_arr) & np.isfinite(target_arr) & np.isfinite(mask_arr), (pred_arr - target_arr) * mask_arr, 0.0)
        original_mse = float(np.sum(diff * diff) / valid_sum)

    original_ic = compute_original_ic_series(pred_arr, target_arr, mask_arr, eps=eps)
    precision_values = np.empty(pred_arr.shape[0], dtype=np.float32)
    sharpe_returns = np.empty(pred_arr.shape[0], dtype=np.float32)
    for day in range(pred_arr.shape[0]):
        precision_values[day] = original_positive_precision_at_k(pred_arr[day], target_arr[day], mask_arr[day], k=10)
        sharpe_returns[day] = original_topk_daily_return(pred_arr[day], target_arr[day], mask_arr[day], k=5)

    sharpe_valid = sharpe_returns[np.isfinite(sharpe_returns)]
    if sharpe_valid.size == 0 or float(sharpe_valid.std()) <= eps:
        original_sharpe5 = float("nan")
    else:
        # Original uses the literal 15.87 multiplier, approximately sqrt(252).
        original_sharpe5 = float(sharpe_valid.mean() / sharpe_valid.std() * 15.87)

    return {
        "OriginalMSE": original_mse,
        "OriginalIC": _nanmean(original_ic),
        "OriginalICIR": _icir(original_ic, eps),
        "OriginalPositivePrecision@10": _nanmean(precision_values),
        "OriginalSharpe@5": original_sharpe5,
    }


def evaluate_predictions(
    preds,
    targets,
    masks,
    k: int = 10,
    asset_major: bool = False,
    eps: float = 1e-8,
    min_valid: int = 3,
    annualization: int = 252,
    transaction_cost_bps: float = 0.0,
    include_diagnostics: bool = False,
    include_original_metrics: bool = False,
):
    pred_arr, target_arr, mask_arr = _as_day_major(preds, targets, masks, asset_major=asset_major)
    ic_series = compute_ic_series(pred_arr, target_arr, mask_arr, eps=eps, min_valid=min_valid)
    ic_summary = summarize_ic(ic_series["IC"], ic_series["RankIC"], eps=eps)
    precision_values = np.empty(pred_arr.shape[0], dtype=np.float32)
    daily_returns = np.empty(pred_arr.shape[0], dtype=np.float32)
    for day in range(pred_arr.shape[0]):
        precision_values[day] = precision_at_k(pred_arr[day], target_arr[day], mask_arr[day], k=k)
        daily_returns[day] = long_only_daily_return(pred_arr[day], target_arr[day], mask_arr[day], k=k)

    result = {
        "mse": _masked_mse(pred_arr, target_arr, mask_arr),
        "IC": ic_summary["IC"],
        "RankIC": ic_summary["RankIC"],
        "ICIR": ic_summary["ICIR"],
        f"Precision@{int(k)}": _nanmean(precision_values),
        "Sharpe": sharpe_ratio(daily_returns, annualization=annualization, eps=eps),
        "num_valid_days": ic_summary["num_valid_days"],
        "num_days": int(pred_arr.shape[0]),
    }

    if include_diagnostics or float(transaction_cost_bps) != 0.0:
        turnover = topk_turnover(pred_arr, target_arr, mask_arr, k=k)
        cost_returns = daily_returns.astype(np.float64) - float(transaction_cost_bps) * 1e-4 * turnover
        result.update(
            {
                "RankICIR": ic_summary["RankICIR"],
                f"Return@{int(k)}": _nanmean(daily_returns),
                f"Sharpe@{int(k)}": result["Sharpe"],
                f"Turnover@{int(k)}": _nanmean(turnover),
                f"CostReturn@{int(k)}": _nanmean(cost_returns),
                f"CostSharpe@{int(k)}": sharpe_ratio(cost_returns, annualization=annualization, eps=eps),
                "NumDays": int(pred_arr.shape[0]),
                "MeanValidAssets": float(np.mean(np.sum(mask_arr > 0.5, axis=1))),
            }
        )
    if include_original_metrics:
        result.update(evaluate_original_stockmixer_metrics(pred_arr, target_arr, mask_arr, eps=eps))
    return result
