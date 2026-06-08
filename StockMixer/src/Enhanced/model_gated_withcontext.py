# model_gated_withcontext.py
"""
model_gated_withcontext.py
==========================

This module defines a StockMixer variant that uses a context‑aware gated
MLP for stock mixing.  It subclasses ``StockMixerBase`` and replaces the
stock mixing step with a gMLP whose gates are modulated by a 5‑dimensional
context vector computed from market statistics (see ``preprocess.py`` for
details).  The context vector is broadcast across time and added to the
gate branch of every block.
"""

from __future__ import annotations

import torch.nn as nn

import torch
import torch.nn.functional as F

from model_original_refactor import StockMixerBase, get_loss
from gmlp_core import gMLPWithContext


class GatedMLPWithContext(StockMixerBase):
    """StockMixer variant using context aware gated MLP for stock mixing.

    Parameters
    ----------
    stocks : int
        Number of assets/stocks.
    time_steps : int
        Lookback window length.
    channels : int
        Number of input feature channels.
    market_hidden : int
        Hidden dimension used within the gMLP blocks.
    depth : int
        Number of gMLP blocks stacked.
    ctx_dim : int, default 5
        Dimensionality of the external context vector; this should match
        whatever is returned by ``preprocess.market_state_from_closes``.
    dropout : float, default 0.0
        Dropout probability applied in gMLP blocks.
    """

    def __init__(self, stocks: int, time_steps: int, channels: int, market_hidden: int, depth: int = 2, ctx_dim: int = 5, dropout: float = 0.0) -> None:
        super().__init__(stocks, time_steps, channels, market_hidden, depth, dropout)
        self.ctx_dim = ctx_dim
        # gMLP conditioned on context.  The sequence length equals the mixed
        # time dimension (self.tsum) and input dimension is number of stocks.
        self.stock_mixer = gMLPWithContext(seq_len=self.tsum, input_dim=stocks, hidden_dim=market_hidden, depth=depth, ctx_dim=ctx_dim, dropout_rate=dropout)

    def forward_stock_mixing(self, y, ctx=None):
        """Mix across stocks using a context aware gMLP.

        Parameters
        ----------
        y : Tensor
            Mixed time/indicator features of shape ``(N, T_sum)``.
        ctx : Tensor or sequence
            External context vector(s).  If a sequence is passed, it will
            be converted to a 1D tensor.  If a tensor has multiple leading
            dimensions (e.g. a batch), the mean over those dimensions is used
            so that one context vector modulates all time steps.
        """
        if ctx is None:
            raise ValueError("Context vector must be provided for the context aware variant")
        # Ensure we have a torch tensor
        if isinstance(ctx, (list, tuple)):
            ctx_tensor = torch.tensor(ctx, dtype=torch.float32)
        else:
            ctx_tensor = ctx
        # Flatten any batch dimensions by averaging over leading dims
        if ctx_tensor.dim() > 1:
            ctx_tensor = ctx_tensor.mean(dim=0)
        # If the context dimension is smaller than expected, pad with zeros
        if ctx_tensor.numel() < self.ctx_dim:
            pad_len = self.ctx_dim - ctx_tensor.numel()
            ctx_tensor = F.pad(ctx_tensor, (0, pad_len))
        # Permute to (T_sum, N) so gMLP mixes along the stock dimension
        x = y.permute(1, 0)
        x = self.stock_mixer(x, ctx_tensor)
        x = x.permute(1, 0)
        return x
