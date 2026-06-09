"""Configurable ReCIL-Mixer model variants.

Compared with the repository snapshot reviewed on GitHub, this implementation
is stricter in three ways:

* only modules used by the selected variant are instantiated;
* asset masks are applied before/after asset-axis mixing;
* conditioning gates start near static/identity behaviour for stable training.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

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
    apply_asset_mask,
    patchify_time,
)

VALID_VARIANTS = {"static", "context_only", "single_gate", "moe", "full"}
CONTEXT_VARIANTS = {"context_only", "single_gate", "moe", "full"}


def _count_trainable_parameters(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


class ReCILMixer(nn.Module):
    """One configurable ReCIL-Mixer implementation for M0-M4 variants.

    The public signature remains compatible with the original code path that
    calls ``ReCILMixer(num_assets=..., num_features=...)``.
    """

    def __init__(
        self,
        num_assets: Optional[int] = None,
        num_features: Optional[int] = None,
        input_dim: Optional[int] = None,
        d_model: int = 64,
        context_dim: int = 7,
        market_dim: int = 32,
        num_experts: int = 4,
        rank: Optional[int] = None,
        scales: Sequence[int] = (1, 2, 4),
        dropout: float = 0.1,
        variant: str = "full",
        mask_invalid_assets: bool = True,
        scale_gate_temperature: float = 1.0,
        router_init: str = "zero",
        router_temperature: float = 1.0,
        gate_init: float = -2.0,
        lookback: Optional[int] = None,  # accepted for config compatibility
        steps: Optional[int] = None,  # accepted for config compatibility
    ) -> None:
        super().__init__()
        del lookback, steps
        if input_dim is None:
            input_dim = num_features
        if input_dim is None:
            raise ValueError("num_features/input_dim is required")
        if num_assets is None:
            raise ValueError("num_assets is required")
        self.num_assets = int(num_assets)
        self.num_features = int(input_dim)
        self.input_dim = self.num_features
        self.d_model = int(d_model)
        self.context_dim = int(context_dim)
        self.market_dim = int(market_dim)
        self.num_experts = int(num_experts)
        self.rank = int(rank or min(32, max(4, self.num_assets // 4)))
        self.scales = tuple(int(scale) for scale in scales)
        self.variant = str(variant)
        self.mask_invalid_assets = bool(mask_invalid_assets)
        self.router_init = str(router_init)
        self.router_temperature = float(router_temperature)
        self.interaction_strength = 1.0

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
        if self.router_temperature <= 0:
            raise ValueError("router_temperature must be positive")
        if len(self.scales) == 0 or any(scale <= 0 for scale in self.scales):
            raise ValueError("scales must be a non-empty sequence of positive integers")

        self.indicator_encoder = IndicatorEncoder(self.num_features, self.d_model, dropout=dropout)
        self.temporal_mixers = nn.ModuleList(
            [
                CausalTemporalMixer(self.d_model, patch_len=min(3, scale), dropout=dropout)
                for scale in self.scales
            ]
        )
        self.prediction_head = PredictionHead(self.d_model, dropout=dropout)

        self.context_encoder: Optional[MarketContextEncoder]
        if self.variant in CONTEXT_VARIANTS:
            self.context_encoder = MarketContextEncoder(self.context_dim, self.market_dim, dropout=dropout)
        else:
            self.context_encoder = None

        self.scale_fusion: Optional[RegimeGatedScaleFusion] = None
        self.film: Optional[FiLMModulation] = None
        self.static_expert: Optional[RegimeConditionedLowRankExperts] = None
        self.single_expert: Optional[RegimeConditionedLowRankExperts] = None
        self.moe_experts: Optional[RegimeConditionedLowRankExperts] = None
        self.context_gate: Optional[ContextGatedResidual] = None

        if self.variant == "static":
            self.static_expert = RegimeConditionedLowRankExperts(
                self.num_assets,
                self.d_model,
                market_dim=self.market_dim,
                rank=self.rank,
                num_experts=1,
                dropout=dropout,
                context_routed=False,
            )
        elif self.variant == "context_only":
            self.film = FiLMModulation(self.d_model, market_dim=self.market_dim, dropout=dropout)
        elif self.variant == "single_gate":
            self.single_expert = RegimeConditionedLowRankExperts(
                self.num_assets,
                self.d_model,
                market_dim=self.market_dim,
                rank=self.rank,
                num_experts=1,
                dropout=dropout,
                context_routed=False,
            )
            self.context_gate = ContextGatedResidual(
                self.d_model,
                market_dim=self.market_dim,
                dropout=dropout,
                gate_init=gate_init,
            )
        elif self.variant == "moe":
            self.moe_experts = RegimeConditionedLowRankExperts(
                self.num_assets,
                self.d_model,
                market_dim=self.market_dim,
                rank=self.rank,
                num_experts=self.num_experts,
                dropout=dropout,
                context_routed=True,
                router_init=self.router_init,
                router_temperature=self.router_temperature,
            )
        else:  # full
            self.scale_fusion = RegimeGatedScaleFusion(
                market_dim=self.market_dim,
                num_scales=len(self.scales),
                temperature=scale_gate_temperature,
                dropout=dropout,
                init_uniform=True,
            )
            self.film = FiLMModulation(self.d_model, market_dim=self.market_dim, dropout=dropout)
            self.moe_experts = RegimeConditionedLowRankExperts(
                self.num_assets,
                self.d_model,
                market_dim=self.market_dim,
                rank=self.rank,
                num_experts=self.num_experts,
                dropout=dropout,
                context_routed=True,
                router_init=self.router_init,
                router_temperature=self.router_temperature,
            )
            self.context_gate = ContextGatedResidual(
                self.d_model,
                market_dim=self.market_dim,
                dropout=dropout,
                gate_init=gate_init,
            )

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

    def _context_embedding(self, x: torch.Tensor, context: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if self.context_encoder is None:
            return None
        if context is None:
            raise ValueError(f"variant {self.variant!r} requires context with shape [B, {self.context_dim}]")
        if context.ndim != 2 or context.shape != (x.shape[0], self.context_dim):
            raise ValueError(f"context must have shape [B, {self.context_dim}]")
        return self.context_encoder(context.to(device=x.device, dtype=x.dtype))

    @staticmethod
    def _uniform_scale_fusion(scale_outputs: torch.Tensor) -> torch.Tensor:
        return scale_outputs.mean(dim=1)

    def set_interaction_strength(self, value: float) -> None:
        value = float(value)
        if value < 0.0 or value > 1.0:
            raise ValueError("interaction_strength must be in [0, 1]")
        self.interaction_strength = value

    def _blend_interaction(self, base: torch.Tensor, interacted: torch.Tensor) -> torch.Tensor:
        strength = float(self.interaction_strength)
        if strength >= 1.0:
            return interacted
        if strength <= 0.0:
            return base
        return base + strength * (interacted - base)

    def _temporal_encode(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        z = self.indicator_encoder(x)
        if self.mask_invalid_assets:
            z = apply_asset_mask(z, mask)
        h_scales = []
        for scale, temporal_mixer in zip(self.scales, self.temporal_mixers):
            h_scale = temporal_mixer(patchify_time(z, scale))
            h_scales.append(apply_asset_mask(h_scale, mask) if self.mask_invalid_assets else h_scale)
        return torch.stack(h_scales, dim=1)  # [B, S, N, D]

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Optional[torch.Tensor]]]:
        self._validate_x(x)
        if mask is not None:
            self._validate_mask(mask, x)
            if self.mask_invalid_assets:
                mask = mask.to(device=x.device, dtype=x.dtype)
            else:
                mask = None

        aux: Dict[str, Optional[torch.Tensor]] = {
            "scale_weights": None,
            "router_weights": None,
            "context_gate": None,
        }
        context_emb = self._context_embedding(x, context)

        scale_outputs = self._temporal_encode(x, mask)

        if self.scale_fusion is not None:
            assert context_emb is not None
            h, scale_weights = self.scale_fusion(scale_outputs, context_emb)
            aux["scale_weights"] = scale_weights.detach()
        else:
            h = self._uniform_scale_fusion(scale_outputs)
        h = apply_asset_mask(h, mask) if self.mask_invalid_assets else h
        h_base = h

        if self.variant == "static":
            assert self.static_expert is not None
            h, _ = self.static_expert(h, context_emb=None, mask=mask)
        elif self.variant == "context_only":
            assert self.film is not None and context_emb is not None
            h = self.film(h, context_emb)
            h = apply_asset_mask(h, mask) if self.mask_invalid_assets else h
        elif self.variant == "single_gate":
            assert self.single_expert is not None and self.context_gate is not None and context_emb is not None
            h_inter, _ = self.single_expert(h, context_emb=None, mask=mask)
            h, gate = self.context_gate(h_base, h_inter, context_emb)
            h = self._blend_interaction(h_base, h)
            h = apply_asset_mask(h, mask) if self.mask_invalid_assets else h
            aux["context_gate"] = gate.detach()
        elif self.variant == "moe":
            assert self.moe_experts is not None and context_emb is not None
            h, router_weights = self.moe_experts(h, context_emb=context_emb, mask=mask)
            h = self._blend_interaction(h_base, h)
            aux["router_weights"] = router_weights.detach()
        elif self.variant == "full":
            assert self.film is not None and self.moe_experts is not None
            assert self.context_gate is not None and context_emb is not None
            h_mod = self.film(h, context_emb)
            h_mod = apply_asset_mask(h_mod, mask) if self.mask_invalid_assets else h_mod
            h_inter, router_weights = self.moe_experts(h_mod, context_emb=context_emb, mask=mask)
            h, gate = self.context_gate(h_base, h_inter, context_emb)
            h = self._blend_interaction(h_base, h)
            h = apply_asset_mask(h, mask) if self.mask_invalid_assets else h
            aux["router_weights"] = router_weights.detach()
            aux["context_gate"] = gate.detach()
        else:  # pragma: no cover
            raise ValueError(f"unsupported ReCILMixer variant: {self.variant}")

        pred = self.prediction_head(h)
        if mask is not None and self.mask_invalid_assets:
            pred = pred.masked_fill(mask <= 0, 0.0)
        return pred, aux

    @property
    def active_parameter_count(self) -> int:
        return _count_trainable_parameters(self)

    @property
    def all_parameter_count(self) -> int:
        return _count_trainable_parameters(self)

    def parameter_report(self) -> Dict[str, object]:
        return {
            "variant": self.variant,
            "router_init": self.router_init,
            "router_temperature": self.router_temperature,
            "interaction_strength": self.interaction_strength,
            "active_params": self.active_parameter_count,
            "all_params": self.all_parameter_count,
            "module_params": {
                name: _count_trainable_parameters(module)
                for name, module in self.named_children()
            },
        }
