"""Configurable ReCIL-Mixer model variants."""

from __future__ import annotations

import torch
from torch import nn

from .modules import (
    CausalTemporalMixer,
    ContextGatedResidual,
    FiLMModulation,
    IndicatorEncoder,
    MarketContextEncoder,
    PredictionHead,
    RegimeConditionedLowRankExperts,
    RegimeGatedScaleFusion,
    patchify_time,
)


VALID_VARIANTS = {"static", "context_only", "single_gate", "moe", "full"}


class ReCILMixer(nn.Module):
    """One configurable ReCIL-Mixer implementation for M0-M4 variants."""

    def __init__(
        self,
        num_assets: int,
        num_features: int,
        d_model: int = 64,
        context_dim: int = 7,
        market_dim: int = 32,
        num_experts: int = 4,
        scales=(1, 2, 4),
        dropout: float = 0.1,
        variant: str = "full",
    ):
        super().__init__()
        self.num_assets = int(num_assets)
        self.num_features = int(num_features)
        self.d_model = int(d_model)
        self.context_dim = int(context_dim)
        self.market_dim = int(market_dim)
        self.num_experts = int(num_experts)
        self.scales = tuple(int(scale) for scale in scales)
        self.variant = str(variant)

        if self.variant not in VALID_VARIANTS:
            raise ValueError(f"unsupported ReCILMixer variant: {variant}")
        if self.num_assets <= 0:
            raise ValueError("num_assets must be positive")
        if self.num_features <= 0:
            raise ValueError("num_features must be positive")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.context_dim <= 0:
            raise ValueError("context_dim must be positive")
        if self.market_dim <= 0:
            raise ValueError("market_dim must be positive")
        if self.num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if len(self.scales) == 0 or any(scale <= 0 for scale in self.scales):
            raise ValueError("scales must be a non-empty sequence of positive integers")

        self.indicator_encoder = IndicatorEncoder(self.num_features, self.d_model, dropout=dropout)
        self.context_encoder = MarketContextEncoder(self.context_dim, self.d_model, dropout=dropout)
        self.temporal_mixers = nn.ModuleList(
            [CausalTemporalMixer(self.d_model, patch_len=min(3, scale), dropout=dropout) for scale in self.scales]
        )
        self.scale_fusion = RegimeGatedScaleFusion(self.d_model, len(self.scales))
        self.film = FiLMModulation(self.d_model)
        self.static_expert = RegimeConditionedLowRankExperts(
            self.num_assets,
            self.d_model,
            market_dim=self.market_dim,
            num_experts=1,
            dropout=dropout,
        )
        self.single_expert = RegimeConditionedLowRankExperts(
            self.num_assets,
            self.d_model,
            market_dim=self.market_dim,
            num_experts=1,
            dropout=dropout,
        )
        self.moe_experts = RegimeConditionedLowRankExperts(
            self.num_assets,
            self.d_model,
            market_dim=self.market_dim,
            num_experts=self.num_experts,
            dropout=dropout,
        )
        self.context_gate = ContextGatedResidual(self.d_model)
        self.prediction_head = PredictionHead(self.d_model, dropout=dropout)

    def _validate_x(self, x: torch.Tensor) -> None:
        if x.ndim != 4:
            raise ValueError(f"x must have shape [B, N, T, F], got {tuple(x.shape)}")
        if x.shape[1] != self.num_assets:
            raise ValueError(f"expected N={self.num_assets}, got N={x.shape[1]}")
        if x.shape[-1] != self.num_features:
            raise ValueError(f"expected F={self.num_features}, got F={x.shape[-1]}")
        for scale in self.scales:
            if x.shape[2] % scale != 0:
                raise ValueError(f"T={x.shape[2]} must be divisible by scale={scale}")

    def _validate_mask(self, mask: torch.Tensor, x: torch.Tensor) -> None:
        if mask.ndim != 2:
            raise ValueError(f"mask must have shape [B, N], got {tuple(mask.shape)}")
        if mask.shape != x.shape[:2]:
            raise ValueError(f"mask must have shape {tuple(x.shape[:2])}, got {tuple(mask.shape)}")

    def _zero_context_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], self.d_model, dtype=x.dtype, device=x.device)

    def _context_embedding(self, x: torch.Tensor, context: torch.Tensor | None, required: bool) -> torch.Tensor:
        if context is None:
            if required:
                raise ValueError(f"variant '{self.variant}' requires context with shape [B, {self.context_dim}]")
            return self._zero_context_embedding(x)
        if context.ndim != 2 or context.shape != (x.shape[0], self.context_dim):
            raise ValueError(f"context must have shape [B, {self.context_dim}]")
        return self.context_encoder(context.to(device=x.device, dtype=x.dtype))

    @staticmethod
    def _uniform_scale_fusion(h_scales: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(h_scales, dim=0).mean(dim=0)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None, mask: torch.Tensor | None = None):
        self._validate_x(x)
        if mask is not None:
            self._validate_mask(mask, x)

        aux = {"scale_weights": None, "router_weights": None, "context_gate": None}
        context_required = self.variant != "static"
        context_emb = self._context_embedding(x, context, required=context_required)

        z = self.indicator_encoder(x)
        h_scales = []
        for scale, temporal_mixer in zip(self.scales, self.temporal_mixers):
            h_scales.append(temporal_mixer(patchify_time(z, scale)))

        if self.variant == "full":
            h, scale_weights = self.scale_fusion(h_scales, context_emb)
            aux["scale_weights"] = scale_weights
            h = self.film(h, context_emb)
            h_inter, router_weights = self.moe_experts(h, context_emb)
            h, context_gate = self.context_gate(h, h_inter, context_emb)
            aux["router_weights"] = router_weights
            aux["context_gate"] = context_gate
        else:
            h = self._uniform_scale_fusion(h_scales)
            if self.variant == "static":
                h, _ = self.static_expert(h, context_emb)
            elif self.variant == "context_only":
                h = self.film(h, context_emb)
            elif self.variant == "single_gate":
                h_inter, _ = self.single_expert(h, context_emb)
                h, context_gate = self.context_gate(h, h_inter, context_emb)
                aux["context_gate"] = context_gate
            elif self.variant == "moe":
                h, router_weights = self.moe_experts(h, context_emb)
                aux["router_weights"] = router_weights
            else:  # pragma: no cover - constructor validation keeps this unreachable.
                raise ValueError(f"unsupported ReCILMixer variant: {self.variant}")

        pred = self.prediction_head(h)
        return pred, aux
