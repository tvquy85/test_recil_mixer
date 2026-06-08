"""Reusable encoder and temporal modules for ReCIL-Mixer."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class IndicatorEncoder(nn.Module):
    """Encode per-asset technical indicators at each historical time step."""

    def __init__(self, num_features: int, d_model: int, dropout: float = 0.1):
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

    def __init__(self, context_dim: int = 7, d_model: int = 64, dropout: float = 0.1):
        super().__init__()
        self.context_dim = int(context_dim)
        self.d_model = int(d_model)
        self.net = nn.Sequential(
            nn.Linear(self.context_dim, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        if context.ndim != 2:
            raise ValueError(f"context must have shape [B, C], got {tuple(context.shape)}")
        if context.shape[-1] != self.context_dim:
            raise ValueError(f"expected context_dim={self.context_dim}, got {context.shape[-1]}")
        return self.net(context)


def patchify_time(z: torch.Tensor, scale: int) -> torch.Tensor:
    """Average-pool historical time steps into non-overlapping patches."""

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

    def __init__(self, d_model: int, patch_len: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = int(d_model)
        self.patch_len = int(patch_len)
        if self.patch_len <= 0:
            raise ValueError("patch_len must be positive")
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
        padded = F.pad(flat, (self.patch_len - 1, 0))
        mixed = self.depthwise(padded).transpose(1, 2).reshape(batch, assets, patches, d_model)
        mixed = self.pointwise(mixed)
        return self.norm_out(z + mixed)[:, :, -1, :]


class PredictionHead(nn.Module):
    """Predict one scalar return/score per asset."""

    def __init__(self, d_model: int, dropout: float = 0.1):
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

    def __init__(self, d_model: int, num_scales: int):
        super().__init__()
        self.d_model = int(d_model)
        self.num_scales = int(num_scales)
        if self.num_scales <= 0:
            raise ValueError("num_scales must be positive")
        self.router = nn.Linear(self.d_model, self.num_scales)

    def forward(self, h_scales: list[torch.Tensor], context_emb: torch.Tensor):
        if len(h_scales) == 0:
            raise ValueError("h_scales must be non-empty")
        if len(h_scales) != self.num_scales:
            raise ValueError(f"expected {self.num_scales} scales, got {len(h_scales)}")
        if context_emb.ndim != 2 or context_emb.shape[-1] != self.d_model:
            raise ValueError(f"context_emb must have shape [B, {self.d_model}]")

        ref_shape = h_scales[0].shape
        if len(ref_shape) != 3 or ref_shape[-1] != self.d_model:
            raise ValueError(f"each scale must have shape [B, N, {self.d_model}]")
        if ref_shape[0] != context_emb.shape[0]:
            raise ValueError("h_scales batch size must match context_emb")
        for h in h_scales:
            if h.shape != ref_shape:
                raise ValueError("all h_scales must have the same shape")

        scale_weights = torch.softmax(self.router(context_emb), dim=-1)
        stacked = torch.stack(h_scales, dim=1)
        fused = (stacked * scale_weights[:, :, None, None]).sum(dim=1)
        return fused, scale_weights


class FiLMModulation(nn.Module):
    """Feature-wise linear modulation conditioned on market context."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = int(d_model)
        self.net = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model * 2),
        )
        final = self.net[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, h: torch.Tensor, context_emb: torch.Tensor) -> torch.Tensor:
        if h.ndim != 3 or h.shape[-1] != self.d_model:
            raise ValueError(f"h must have shape [B, N, {self.d_model}]")
        if context_emb.ndim != 2 or context_emb.shape != (h.shape[0], self.d_model):
            raise ValueError(f"context_emb must have shape [B, {self.d_model}]")
        gamma, beta = self.net(context_emb).chunk(2, dim=-1)
        return h * (1.0 + gamma[:, None, :]) + beta[:, None, :]


class RegimeConditionedLowRankExperts(nn.Module):
    """Context-routed low-rank experts for asset-axis interaction."""

    def __init__(
        self,
        num_assets: int,
        d_model: int,
        market_dim: int = 32,
        num_experts: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_assets = int(num_assets)
        self.d_model = int(d_model)
        self.market_dim = int(market_dim)
        self.num_experts = int(num_experts)
        if self.num_assets <= 0:
            raise ValueError("num_assets must be positive")
        if self.market_dim <= 0:
            raise ValueError("market_dim must be positive")
        if self.num_experts <= 0:
            raise ValueError("num_experts must be positive")

        self.router = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.num_experts),
        )
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.num_assets, self.market_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(self.market_dim, self.num_assets),
                )
                for _ in range(self.num_experts)
            ]
        )
        self.norm = nn.LayerNorm(self.d_model)

    def forward(self, h: torch.Tensor, context_emb: torch.Tensor):
        if h.ndim != 3 or h.shape[1] != self.num_assets or h.shape[-1] != self.d_model:
            raise ValueError(f"h must have shape [B, {self.num_assets}, {self.d_model}]")
        if context_emb.ndim != 2 or context_emb.shape != (h.shape[0], self.d_model):
            raise ValueError(f"context_emb must have shape [B, {self.d_model}]")

        router_weights = torch.softmax(self.router(context_emb), dim=-1)
        h_assets = h.transpose(1, 2)
        expert_outputs = [expert(h_assets).transpose(1, 2) for expert in self.experts]
        stacked = torch.stack(expert_outputs, dim=1)
        h_inter = (stacked * router_weights[:, :, None, None]).sum(dim=1)
        return self.norm(h + h_inter), router_weights


class ContextGatedResidual(nn.Module):
    """Context-conditioned interpolation between base and interaction features."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = int(d_model)
        self.gate_net = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )

    def forward(self, h_base: torch.Tensor, h_inter: torch.Tensor, context_emb: torch.Tensor):
        if h_base.ndim != 3 or h_base.shape[-1] != self.d_model:
            raise ValueError(f"h_base must have shape [B, N, {self.d_model}]")
        if h_inter.shape != h_base.shape:
            raise ValueError("h_inter must have the same shape as h_base")
        if context_emb.ndim != 2 or context_emb.shape != (h_base.shape[0], self.d_model):
            raise ValueError(f"context_emb must have shape [B, {self.d_model}]")
        gate = torch.sigmoid(self.gate_net(context_emb))
        h_out = h_base + gate[:, None, :] * (h_inter - h_base)
        return h_out, gate
