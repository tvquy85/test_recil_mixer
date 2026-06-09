"""Training losses for ReCIL."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def _as_batch_tensor(pred, target, mask) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred_t = pred if isinstance(pred, torch.Tensor) else torch.as_tensor(pred)
    target_t = target if isinstance(target, torch.Tensor) else torch.as_tensor(target, device=pred_t.device)
    mask_t = mask if isinstance(mask, torch.Tensor) else torch.as_tensor(mask, device=pred_t.device)
    target_t = target_t.to(device=pred_t.device)
    mask_t = mask_t.to(device=pred_t.device)
    if pred_t.ndim == 1:
        pred_t = pred_t.unsqueeze(0)
    if target_t.ndim == 1:
        target_t = target_t.unsqueeze(0)
    if mask_t.ndim == 1:
        mask_t = mask_t.unsqueeze(0)
    if pred_t.ndim != 2 or target_t.ndim != 2 or mask_t.ndim != 2:
        raise ValueError("pred, target, and mask must have shape [N] or [B, N]")
    dtype = pred_t.dtype if pred_t.is_floating_point() else torch.float32
    return pred_t.to(dtype=dtype), target_t.to(dtype=dtype), mask_t.to(dtype=dtype)


def _graph_zero(pred: torch.Tensor) -> torch.Tensor:
    return pred.sum() * 0.0


def _valid_pair(pred, target, mask) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred_t, target_t, mask_t = _as_batch_tensor(pred, target, mask)
    if pred_t.shape != target_t.shape or pred_t.shape != mask_t.shape:
        raise ValueError(
            f"pred, target and mask must share shape, got {tuple(pred_t.shape)}, {tuple(target_t.shape)}, {tuple(mask_t.shape)}"
        )
    valid = (mask_t > 0.5) & torch.isfinite(pred_t) & torch.isfinite(target_t)
    return pred_t, target_t, valid


def masked_mse_loss(pred, target, mask, eps: float = 1e-8) -> torch.Tensor:
    del eps
    pred, target, valid = _valid_pair(pred, target, mask)
    if not torch.any(valid):
        return _graph_zero(pred)
    diff = pred[valid] - target[valid]
    return (diff * diff).mean()


def pairwise_rank_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    max_pairs_per_day: int = 4096,
    eps: float = 1e-8,
    margin: float = 0.0,
) -> torch.Tensor:
    """Pairwise logistic ranking loss within each day.

    The deterministic sub-sampling keeps the loss reproducible while preventing
    O(N^2) memory blow-ups for large universes.
    """
    if max_pairs_per_day <= 0:
        raise ValueError("max_pairs_per_day must be positive")
    pred, target, valid = _valid_pair(pred, target, mask)
    losses = []
    for day in range(pred.shape[0]):
        idx = torch.nonzero(valid[day], as_tuple=False).flatten()
        if idx.numel() < 2:
            continue
        p = pred[day, idx]
        y = target[day, idx]
        pair_idx = torch.triu_indices(idx.numel(), idx.numel(), offset=1, device=pred.device)
        pred_diff = p[pair_idx[0]] - p[pair_idx[1]]
        target_diff = y[pair_idx[0]] - y[pair_idx[1]]
        non_tie = target_diff.abs() > eps
        if not torch.any(non_tie):
            continue
        pred_diff = pred_diff[non_tie]
        target_sign = target_diff[non_tie].sign()
        if pred_diff.numel() > max_pairs_per_day:
            select = torch.linspace(
                0,
                pred_diff.numel() - 1,
                steps=max_pairs_per_day,
                device=pred_diff.device,
            ).long()
            pred_diff = pred_diff[select]
            target_sign = target_sign[select]
        losses.append(F.softplus(margin - target_sign * pred_diff).mean())
    if not losses:
        return _graph_zero(pred)
    return torch.stack(losses).mean()


def router_entropy_loss(router_weights: torch.Tensor) -> torch.Tensor:
    """Negative entropy regularizer for router weights.

    Adding ``lambda_entropy * router_entropy_loss`` to the total loss encourages
    higher router entropy when ``lambda_entropy`` is positive.
    """
    if router_weights.ndim != 2:
        raise ValueError(f"router_weights must have shape [B, K], got {tuple(router_weights.shape)}")
    eps = torch.finfo(router_weights.dtype).eps
    entropy = -(router_weights.clamp_min(eps) * router_weights.clamp_min(eps).log()).sum(dim=-1)
    return -entropy.mean()


def recil_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    aux: Optional[Dict[str, torch.Tensor]] = None,
    alpha_rank: float = 0.1,
    lambda_entropy: float = 0.0,
    max_pairs_per_day: int = 4096,
) -> torch.Tensor:
    """Combined masked MSE, ranking and optional router-entropy loss."""
    mse = masked_mse_loss(pred, target, mask)
    if alpha_rank > 0:
        rank = pairwise_rank_loss(pred, target, mask, max_pairs_per_day=max_pairs_per_day)
    else:
        rank = pred.new_tensor(0.0)
    total = mse + float(alpha_rank) * rank
    if aux is not None and lambda_entropy > 0 and "router_weights" in aux:
        weights = aux["router_weights"]
        if weights is not None and weights.ndim == 2 and weights.shape[-1] > 1:
            total = total + float(lambda_entropy) * router_entropy_loss(weights)
    return total
