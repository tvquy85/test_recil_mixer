# model_gated_nocontext.py
"""
model_gated_nocontext.py
========================

This module defines a StockMixer variant that replaces the original
stock‑mixing MLP with a stack of gated MLP (gMLP) blocks.  It inherits
common time/indicator mixing and head logic from ``StockMixerBase`` defined
in ``model_original_refactor``.  The gated MLP here does **not** use any
external context; the gates are purely a function of the input activations.
"""

from __future__ import annotations

import torch.nn as nn

from model_original_refactor import StockMixerBase, get_loss
from gmlp_core import gMLPNoContext


class GatedMLPNoContext(StockMixerBase):
    """StockMixer variant using context‑free gated MLP for stock mixing.

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
    dropout : float, default 0.0
        Dropout probability applied in gMLP blocks.
    """

    def __init__(self, stocks: int, time_steps: int, channels: int, market_hidden: int, depth: int = 2, dropout: float = 0.0) -> None:
        super().__init__(stocks, time_steps, channels, market_hidden, depth, dropout)
        # Initialise a context‑free gMLP to perform stock mixing.  The sequence
        # length for gMLP is the number of time positions after mixing (self.tsum),
        # and the input dimension is the number of stocks.
        self.stock_mixer = gMLPNoContext(seq_len=self.tsum, input_dim=stocks, hidden_dim=market_hidden, depth=depth, dropout_rate=dropout)

    def forward_stock_mixing(self, y, ctx=None):
        # y shape: (N, T_sum)
        # Permute to (T_sum, N) so gMLP mixes along the stock dimension (columns)
        x = y.permute(1, 0)
        x = self.stock_mixer(x)
        x = x.permute(1, 0)
        return x
