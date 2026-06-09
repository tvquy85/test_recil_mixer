"""Reusable encoder, temporal and interaction modules for ReCIL-Mixer.

This drop-in replacement follows the repository's original tensor convention:
``x`` has shape ``[B, N, T, F]`` where ``N`` is the asset dimension. The main
safety improvement is that asset masks are propagated through stock-mixing
layers, preventing invalid/padded securities from contaminating valid ones.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


def apply_asset_mask(tensor: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Zero invalid asset rows in a ``[B, N, ...]`` tensor."""
    if mask is None:
        return tensor
    if mask.ndim != 2:
        raise ValueError(f"mask must have shape [B, N], got {tuple(mask.shape)}")
    if tensor.ndim < 2:
        raise ValueError("tensor must have at least batch and asset dimensions")
    if tensor.shape[0] != mask.shape[0] or tensor.shape[1] != mask.shape[1]:
        raise ValueError(
            "mask shape must match tensor batch/asset dimensions: "
            f"tensor={tuple(tensor.shape)}, mask={tuple(mask.shape)}"
        )
    expand_shape = list(mask.shape) + [1] * (tensor.ndim - 2)
    return tensor * mask.to(device=tensor.device, dtype=tensor.dtype).reshape(expand_shape)


class IndicatorEncoder(nn.Module):
    """Encode per-asset technical indicators at each historical time step."""

    def __init__(self, num_features: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_features = int(num_features)
        self.d_model = int(d_model)
        self.net = nn.Sequential(
            nn.Linear(self.num_features, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"x must have shape [B, N, T, F], got {tuple(x.shape)}")
        if x.shape[-1] != self.num_features:
            raise ValueError(f"expected F={self.num_features}, got F={x.shape[-1]}")
        return self.net(x)


class MarketContextEncoder(nn.Module):
    """Encode train-only-scaled market context features."""

    def __init__(
        self,
        context_dim: int = 7,
        market_dim: int = 32,
        d_model: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model is not None:
            market_dim = d_model
        self.context_dim = int(context_dim)
        self.market_dim = int(market_dim)
        hidden = int(hidden_dim or max(market_dim, context_dim * 2))
        self.net = nn.Sequential(
            nn.LayerNorm(self.context_dim),
            nn.Linear(self.context_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.market_dim),
            nn.LayerNorm(self.market_dim),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        if context.ndim != 2:
            raise ValueError(f"context must have shape [B, C], got {tuple(context.shape)}")
        if context.shape[-1] != self.context_dim:
            raise ValueError(f"expected context_dim={self.context_dim}, got {context.shape[-1]}")
        return self.net(context)


def patchify_time(z: torch.Tensor, scale: int) -> torch.Tensor:
    """Average-pool historical time steps into non-overlapping patches.

    Args:
        z: Hidden tensor with shape ``[B, N, T, D]``.
        scale: Number of adjacent time steps per patch. ``scale=1`` leaves the
            sequence unchanged.
    """
    if z.ndim != 4:
        raise ValueError(f"z must have shape [B, N, T, D], got {tuple(z.shape)}")
    scale = int(scale)
    if scale <= 0:
        raise ValueError("scale must be positive")
    if scale == 1:
        return z
    time_steps = z.shape[2]
    if time_steps % scale != 0:
        raise ValueError(f"T={time_steps} must be divisible by scale={scale}")
    return z.reshape(z.shape[0], z.shape[1], time_steps // scale, scale, z.shape[3]).mean(dim=3)


class CausalTemporalMixer(nn.Module):
    """Lightweight causal temporal mixer over historical patch tokens."""

    def __init__(
        self,
        d_model: int,
        patch_len: Optional[int] = None,
        kernel_size: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.patch_len = int(patch_len if patch_len is not None else (kernel_size or 3))
        if self.patch_len <= 0:
            raise ValueError("patch_len/kernel_size must be positive")
        self.norm_in = nn.LayerNorm(self.d_model)
        self.depthwise = nn.Conv1d(
            in_channels=self.d_model,
            out_channels=self.d_model,
            kernel_size=self.patch_len,
            groups=self.d_model,
        )
        self.pointwise = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, self.d_model),
            nn.Dropout(dropout),
        )
        self.norm_out = nn.LayerNorm(self.d_model)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 4:
            raise ValueError(f"z must have shape [B, N, P, D], got {tuple(z.shape)}")
        if z.shape[-1] != self.d_model:
            raise ValueError(f"expected D={self.d_model}, got D={z.shape[-1]}")
        batch, assets, patches, d_model = z.shape
        if patches <= 0:
            raise ValueError("P must be positive")
        z_norm = self.norm_in(z)
        flat = z_norm.reshape(batch * assets, patches, d_model).transpose(1, 2)
        pad_left = max(self.patch_len - 1, 0)
        padded = F.pad(flat, (pad_left, 0))
        mixed = self.depthwise(padded).transpose(1, 2).reshape(batch, assets, patches, d_model)
        mixed = self.pointwise(mixed)
        return self.norm_out(z + mixed)[:, :, -1, :]


class PredictionHead(nn.Module):
    """Predict one scalar return/score per asset."""

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.net = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if h.ndim != 3:
            raise ValueError(f"h must have shape [B, N, D], got {tuple(h.shape)}")
        if h.shape[-1] != self.d_model:
            raise ValueError(f"expected D={self.d_model}, got D={h.shape[-1]}")
        return self.net(h).squeeze(-1)


class RegimeGatedScaleFusion(nn.Module):
    """Fuse multi-scale temporal features with context-conditioned weights."""

    def __init__(
        self,
        market_dim: int,
        num_scales: int,
        temperature: float = 1.0,
        dropout: float = 0.1,
        init_uniform: bool = True,
    ) -> None:
        super().__init__()
        if num_scales <= 0:
            raise ValueError("num_scales must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.market_dim = int(market_dim)
        self.num_scales = int(num_scales)
        self.temperature = float(temperature)
        self.router = nn.Sequential(
            nn.LayerNorm(self.market_dim),
            nn.Linear(self.market_dim, self.market_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.market_dim, self.num_scales),
        )
        if init_uniform:
            last = self.router[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)

    def forward(self, scale_outputs: torch.Tensor | list[torch.Tensor], context_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(scale_outputs, list):
            if len(scale_outputs) != self.num_scales:
                raise ValueError(f"expected {self.num_scales} scale tensors, got {len(scale_outputs)}")
            first_shape = scale_outputs[0].shape
            if any(t.shape != first_shape for t in scale_outputs):
                raise ValueError("all scale tensors must have the same shape")
            scale_outputs = torch.stack(scale_outputs, dim=1)
        if scale_outputs.ndim != 4:
            raise ValueError(
                "scale_outputs must have shape [B, S, N, D], "
                f"got {tuple(scale_outputs.shape)}"
            )
        if context_emb.ndim != 2 or context_emb.shape[-1] != self.market_dim:
            raise ValueError(f"context_emb must have shape [B, {self.market_dim}]")
        if scale_outputs.shape[0] != context_emb.shape[0]:
            raise ValueError("scale_outputs batch size must match context_emb")
        logits = self.router(context_emb) / self.temperature
        weights = torch.softmax(logits, dim=-1)
        fused = (scale_outputs * weights[:, :, None, None]).sum(dim=1)
        return fused, weights


class FiLMModulation(nn.Module):
    """Feature-wise affine modulation conditioned on market context.

    The final projection is zero-initialized, so the module starts as identity.
    """

    def __init__(self, d_model: int, market_dim: Optional[int] = None, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.market_dim = int(market_dim or d_model)
        self.net = nn.Sequential(
            nn.LayerNorm(self.market_dim),
            nn.Linear(self.market_dim, self.market_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.market_dim, self.d_model * 2),
        )
        final = self.net[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def forward(self, h: torch.Tensor, context_emb: torch.Tensor, return_params: bool = False):
        if h.ndim != 3 or h.shape[-1] != self.d_model:
            raise ValueError(f"h must have shape [B, N, {self.d_model}]")
        if context_emb.ndim != 2 or context_emb.shape != (h.shape[0], self.market_dim):
            raise ValueError(f"context_emb must have shape [B, {self.market_dim}]")
        gamma, beta = self.net(context_emb).chunk(2, dim=-1)
        out = h * (1.0 + gamma[:, None, :]) + beta[:, None, :]
        if return_params:
            return out, gamma, beta
        return out


class RegimeConditionedLowRankExperts(nn.Module):
    """Context-routed low-rank experts for asset-axis interaction."""

    ROUTER_INITS = {"zero", "small_normal"}

    def __init__(
        self,
        num_assets: int,
        d_model: int,
        market_dim: int = 32,
        rank: Optional[int] = None,
        num_experts: int = 4,
        dropout: float = 0.1,
        context_routed: bool = False,
        zero_init_output: bool = False,
        router_init: str = "zero",
        router_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_assets = int(num_assets)
        self.d_model = int(d_model)
        self.market_dim = int(market_dim)
        self.num_experts = int(num_experts)
        self.rank = int(rank or min(32, max(4, self.num_assets // 4)))
        self.context_routed = bool(context_routed and self.num_experts > 1)
        self.router_init = str(router_init)
        self.router_temperature = float(router_temperature)
        if self.num_assets <= 0:
            raise ValueError("num_assets must be positive")
        if self.market_dim <= 0:
            raise ValueError("market_dim must be positive")
        if self.num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if self.router_init not in self.ROUTER_INITS:
            raise ValueError(f"router_init must be one of {sorted(self.ROUTER_INITS)}")
        if self.router_temperature <= 0:
            raise ValueError("router_temperature must be positive")

        self.experts = nn.ModuleList()
        for _ in range(self.num_experts):
            expert = nn.Sequential(
                nn.LayerNorm(self.num_assets),
                nn.Linear(self.num_assets, self.rank),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.rank, self.num_assets),
            )
            if zero_init_output:
                final = expert[-1]
                if isinstance(final, nn.Linear):
                    nn.init.zeros_(final.weight)
            self.experts.append(expert)

        if self.context_routed:
            self.router = nn.Sequential(
                nn.LayerNorm(self.market_dim),
                nn.Linear(self.market_dim, self.market_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.market_dim, self.num_experts),
            )
            final = self.router[-1]
            if isinstance(final, nn.Linear):
                if self.router_init == "zero":
                    nn.init.zeros_(final.weight)
                    nn.init.zeros_(final.bias)
                else:
                    nn.init.normal_(final.weight, mean=0.0, std=0.02)
                    nn.init.zeros_(final.bias)
        else:
            self.router = None
        self.norm = nn.LayerNorm(self.d_model)

    def forward(
        self,
        h: torch.Tensor,
        context_emb: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if h.ndim != 3 or h.shape[1] != self.num_assets or h.shape[-1] != self.d_model:
            raise ValueError(f"h must have shape [B, {self.num_assets}, {self.d_model}]")
        h_masked = apply_asset_mask(h, mask)
        h_assets = h_masked.transpose(1, 2)  # [B, D, N]
        expert_outputs = [apply_asset_mask(expert(h_assets).transpose(1, 2), mask) for expert in self.experts]
        stacked = torch.stack(expert_outputs, dim=1)  # [B, K, N, D]

        if self.router is None:
            router_weights = h.new_full((h.shape[0], self.num_experts), 1.0 / float(self.num_experts))
            if context_emb is not None:
                router_weights = router_weights + context_emb.sum() * 0.0
        else:
            if context_emb is None:
                raise ValueError("context_emb is required when context_routed=True")
            if context_emb.ndim != 2 or context_emb.shape != (h.shape[0], self.market_dim):
                raise ValueError(f"context_emb must have shape [B, {self.market_dim}]")
            router_weights = torch.softmax(self.router(context_emb) / self.router_temperature, dim=-1)
        h_inter = (stacked * router_weights[:, :, None, None]).sum(dim=1)
        return apply_asset_mask(self.norm(h_masked + h_inter), mask), router_weights


class ContextGatedResidual(nn.Module):
    """Context-conditioned interpolation between base and interaction features."""

    def __init__(
        self,
        d_model: int,
        market_dim: Optional[int] = None,
        dropout: float = 0.1,
        gate_init: float = -2.0,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.market_dim = int(market_dim or d_model)
        self.gate_net = nn.Sequential(
            nn.LayerNorm(self.market_dim),
            nn.Linear(self.market_dim, self.market_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.market_dim, self.d_model),
        )
        final = self.gate_net[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.constant_(final.bias, float(gate_init))

    def forward(
        self,
        base: torch.Tensor,
        interacted: torch.Tensor,
        context_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if base.shape != interacted.shape:
            raise ValueError(
                f"base and interacted must have same shape, got {tuple(base.shape)} vs {tuple(interacted.shape)}"
            )
        if context_emb.ndim != 2 or context_emb.shape != (base.shape[0], self.market_dim):
            raise ValueError(f"context_emb must have shape [B, {self.market_dim}]")
        gate = torch.sigmoid(self.gate_net(context_emb)).unsqueeze(1)
        return base + gate * (interacted - base), gate.squeeze(1)
