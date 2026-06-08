import torch
import torch.nn as nn
import torch.nn.functional as F

"""
model_gatedMLP.py
===================

This module defines a variant of the original StockMixer architecture
that replaces the stock‑mixing MLP with a gated MLP (gMLP) without
external context.  The remainder of the model – time/indicator
mixing, channel reduction and time aggregation – is kept identical
to the original implementation in ``model.py``.  The gMLP is
implemented via the reusable primitives provided in ``gmlp_core``.

Usage
-----
Instantiate ``StockMixerGatedNoContext`` with the same signature as
the original ``StockMixer`` class.  The optional ``depth`` and
``dropout`` arguments control the number of gMLP blocks and the
dropout probability inside the gMLP, respectively.  These defaults
mirror the behaviour of the original ``NoGraphMixer`` which
effectively uses a single hidden layer and no dropout.
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Original.model import MultTime2dMixer  # reuse multiscale time/indicator mixer
from Enhanced.gmlp_core import gMLPNoContext  # context‑free gated MLP


class StockMixerGatedNoContext(nn.Module):
    """StockMixer variant using a context‑free gated MLP for stock mixing.

    Parameters
    ----------
    stocks : int
        Number of assets (stocks) in the universe.
    time_steps : int
        Length of the historical lookback window.
    channels : int
        Number of feature channels per asset (e.g. OHLCV, indicators).
    market : int
        Hidden dimension of the gMLP; corresponds to the number of
        latent market factors.  This replaces the ``market`` argument
        of the original model's ``NoGraphMixer``.
    scale : int
        Unused in this implementation but kept for API compatibility
        with the original ``StockMixer``.  The original code hardcodes
        a scale dimension of 8, so this argument has no effect here.
    depth : int, optional
        Number of gMLP blocks to stack in the stock mixing network.
        Defaults to 1 to roughly mimic the capacity of the original
        ``NoGraphMixer``, which has a single hidden layer.
    dropout : float, optional
        Dropout probability applied inside the gMLP blocks.  Defaults
        to 0.0, matching the original implementation.
    """

    def __init__(self, stocks: int, time_steps: int, channels: int, market: int,
                 scale: int = 1, depth: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        # The original StockMixer uses a fixed scale dimension of 8 for
        # the auxiliary (downsampled) branch.  We mirror that here.
        scale_dim = 8
        self.stocks = stocks
        self.time_steps = time_steps
        self.channels = channels
        self.hidden_dim = market
        self.depth = depth
        self.dropout = dropout
        # The total number of time positions after multiscale mixing.
        # The main branch contributes ``time_steps`` positions, the
        # triangular time mixing adds another ``time_steps`` and the
        # downsampled branch contributes ``scale_dim`` positions.
        self.tsum = time_steps * 2 + scale_dim
        # Multiscale time/indicator mixer reused from the original model.
        self.mixer = MultTime2dMixer(time_steps, channels, scale_dim=scale_dim)
        # Channel reduction: reduce feature channels to a scalar per time
        # position.
        self.channel_fc = nn.Linear(channels, 1)
        # Downsampling convolution to halve the time dimension before
        # multiscale mixing.
        self.conv = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=2, stride=2)
        # gMLP for stock mixing.  Operates on a sequence of length
        # ``tsum`` with input dimension ``stocks`` and hidden dimension
        # ``market``.  The depth and dropout are configurable.
        self.stock_mixer = gMLPNoContext(
            seq_len=self.tsum,
            input_dim=stocks,
            hidden_dim=market,
            depth=depth,
            dropout_rate=dropout
        )
        # Two separate linear heads to aggregate over the time dimension
        # after time/indicator mixing and after stock mixing.  Their
        # outputs are summed to produce the final return ratio.
        self.time_fc = nn.Linear(self.tsum, 1)
        self.time_fc_ = nn.Linear(self.tsum, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass for StockMixerGatedNoContext.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape ``(N, T, F)`` where ``N`` is the
            number of stocks, ``T`` is the time dimension and ``F`` is
            the feature dimension.

        Returns
        -------
        torch.Tensor
            Output predictions of shape ``(N, 1)`` representing the
            predicted return ratios for each stock.
        """
        # Downsample the time dimension by a factor of 2 using the
        # convolution defined in the original model.  We first permute
        # the input to (N, F, T) so the convolution operates over time.
        x = inputs.permute(0, 2, 1)  # shape (N, F, T)
        x = self.conv(x)             # shape (N, F, T/2)
        x = x.permute(0, 2, 1)       # shape (N, T/2, F)
        # Multiscale mixing combines the original sequence with the
        # downsampled sequence.  The output has shape (N, T_sum, F).
        y = self.mixer(inputs, x)
        # Reduce feature channels to a single scalar per time position.
        y = self.channel_fc(y).squeeze(-1)  # shape (N, T_sum)
        # Permute to (T_sum, N) so that gMLP mixes along the stock
        # dimension.  The gMLP returns a tensor of the same shape.
        z = y.permute(1, 0)
        z = self.stock_mixer(z)
        z = z.permute(1, 0)  # shape (N, T_sum)
        # Aggregate along the time dimension via two linear heads.  The
        # outputs are then summed to produce the final predictions.
        y_out = self.time_fc(y)  # shape (N, 1)
        z_out = self.time_fc_(z)  # shape (N, 1)
        return y_out + z_out