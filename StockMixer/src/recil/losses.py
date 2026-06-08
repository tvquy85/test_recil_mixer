"""Training losses for ReCIL."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _as_batch_tensor(pred, target, mask):
    pred_t = torch.as_tensor(pred) if not isinstance(pred, torch.Tensor) else pred
    target_t = torch.as_tensor(target, device=pred_t.device) if not isinstance(target, torch.Tensor) else target.to(pred_t.device)
    mask_t = torch.as_tensor(mask, device=pred_t.device) if not isinstance(mask, torch.Tensor) else mask.to(pred_t.device)

    if pred_t.ndim == 1:
        pred_t = pred_t.unsqueeze(0)
    if target_t.ndim == 1:
        target_t = target_t.unsqueeze(0)
    if mask_t.ndim == 1:
        mask_t = mask_t.unsqueeze(0)
    if pred_t.ndim != 2 or target_t.ndim != 2 or mask_t.ndim != 2:
        raise ValueError("pred, target, and mask must have shape [N] or [B, N]")
    if pred_t.shape != target_t.shape or pred_t.shape != mask_t.shape:
        raise ValueError("pred, target, and mask must have matching shapes")

    dtype = pred_t.dtype if pred_t.is_floating_point() else torch.float32
    return pred_t.to(dtype=dtype), target_t.to(dtype=dtype), mask_t.to(dtype=dtype)


def _graph_zero(pred: torch.Tensor) -> torch.Tensor:
    return pred.sum() * 0.0


def masked_mse_loss(pred, target, mask, eps: float = 1e-8) -> torch.Tensor:
    """Mean squared error over finite valid assets only."""

    del eps
    pred_t, target_t, mask_t = _as_batch_tensor(pred, target, mask)
    valid = torch.isfinite(pred_t) & torch.isfinite(target_t) & torch.isfinite(mask_t) & (mask_t > 0.5)
    if not torch.any(valid):
        return _graph_zero(pred_t)
    diff = pred_t[valid] - target_t[valid]
    return torch.mean(diff * diff)


def _subsample_pair_indices(num_pairs: int, max_pairs_per_day: int, device) -> torch.Tensor:
    if num_pairs <= max_pairs_per_day:
        return torch.arange(num_pairs, device=device)
    positions = torch.linspace(0, num_pairs - 1, steps=max_pairs_per_day, device=device)
    return positions.round().to(dtype=torch.long)


def pairwise_rank_loss(pred, target, mask, max_pairs_per_day: int = 4096, eps: float = 1e-8) -> torch.Tensor:
    """RankNet logistic pairwise loss within each day."""

    if max_pairs_per_day <= 0:
        raise ValueError("max_pairs_per_day must be positive")

    pred_t, target_t, mask_t = _as_batch_tensor(pred, target, mask)
    losses = []
    for day in range(pred_t.shape[0]):
        valid = torch.isfinite(pred_t[day]) & torch.isfinite(target_t[day]) & torch.isfinite(mask_t[day]) & (mask_t[day] > 0.5)
        valid_idx = torch.nonzero(valid, as_tuple=False).flatten()
        if valid_idx.numel() < 2:
            continue

        pair_idx = torch.triu_indices(valid_idx.numel(), valid_idx.numel(), offset=1, device=pred_t.device)
        left = valid_idx[pair_idx[0]]
        right = valid_idx[pair_idx[1]]
        target_diff = target_t[day, left] - target_t[day, right]
        non_tie = torch.abs(target_diff) > eps
        if not torch.any(non_tie):
            continue

        left = left[non_tie]
        right = right[non_tie]
        target_diff = target_diff[non_tie]
        selected = _subsample_pair_indices(int(left.numel()), int(max_pairs_per_day), pred_t.device)
        left = left[selected]
        right = right[selected]
        target_sign = torch.sign(target_diff[selected])
        pred_diff = pred_t[day, left] - pred_t[day, right]
        losses.append(F.softplus(-target_sign * pred_diff).mean())

    if not losses:
        return _graph_zero(pred_t)
    return torch.stack(losses).mean()


def recil_loss(
    pred,
    target,
    mask,
    aux=None,
    alpha_rank: float = 0.1,
    lambda_entropy: float = 0.0,
) -> torch.Tensor:
    """Combined ReCIL regression, ranking, and optional router-entropy loss."""

    mse = masked_mse_loss(pred, target, mask)
    rank = pairwise_rank_loss(pred, target, mask)
    total = mse + float(alpha_rank) * rank

    if lambda_entropy > 0.0 and aux is not None and "router_weights" in aux:
        weights = aux["router_weights"]
        weights_t = weights if isinstance(weights, torch.Tensor) else torch.as_tensor(weights, device=mse.device)
        weights_t = weights_t.to(device=mse.device, dtype=mse.dtype)
        entropy = -(weights_t * torch.log(weights_t.clamp_min(1e-8))).sum(dim=-1).mean()
        total = total - float(lambda_entropy) * entropy
    return total
