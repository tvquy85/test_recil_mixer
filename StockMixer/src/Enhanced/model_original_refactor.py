# model_original_refactor.py
"""
model_original_refactor.py
===========================

This module factors out the common components of the original StockMixer
architecture.  It retains the original indicator/time mixing pipeline, the
channel reduction, and the two time heads used for forecasting.  What it
omits is the stock‑mixing block itself: child classes are expected to
provide their own implementation via ``forward_stock_mixing``.

``StockMixerBase`` serves as an abstract base class that defines a uniform
forward pass for all variants.  Given inputs of shape ``(N, T, F)`` (stocks
× time × features), it first performs a convolutional downsample on the
time axis, feeds the original and downsampled sequences through a
multiscale mixer, and then reduces the channel dimension to obtain a
sequence ``(N, T_sum)``.  The child implementation mixes across the stock
dimension to produce a second sequence of the same shape.  Finally the two
sequences are separately aggregated over the time axis and summed to yield
the prediction ``(N, 1)``.

The module also exposes ``get_loss`` which reproduces the original loss
function combining mean squared error with a pairwise ranking term.

To implement a concrete model, subclass ``StockMixerBase`` and override
``forward_stock_mixing`` to perform stock mixing either with a simple MLP
or with a gMLP.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======== Loss Function ========
def get_loss(prediction: torch.Tensor, ground_truth: torch.Tensor, base_price: torch.Tensor,
             mask: torch.Tensor, batch_size: int, alpha: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the combined loss used for training.

    The loss consists of a regression term (mean squared error on return ratios)
    and a pairwise ranking term that encourages correct ordering of returns.

    Parameters
    ----------
    prediction : Tensor
        Raw predictions of shape ``(batch_size, 1)``.
    ground_truth : Tensor
        True return ratios of shape ``(batch_size, 1)``.
    base_price : Tensor
        The baseline prices used to compute return ratios, shape ``(batch_size, 1)``.
    mask : Tensor
        A binary mask indicating valid stocks, shape ``(batch_size, 1)``.
    batch_size : int
        The number of stocks.
    alpha : float
        Weighting factor for the ranking loss.

    Returns
    -------
    loss : Tensor
        Total loss combining regression and ranking.
    reg_loss : Tensor
        Mean squared error term.
    rank_loss : Tensor
        Pairwise ranking loss term.
    return_ratio : Tensor
        The predicted return ratios used for downstream evaluation.
    """
    device = prediction.device
    all_one = torch.ones(batch_size, 1, dtype=torch.float32, device=device)
    return_ratio = (prediction - base_price) / base_price
    reg_loss = F.mse_loss(return_ratio * mask, ground_truth * mask)
    pre_pw_dif = (return_ratio @ all_one.T) - (all_one @ return_ratio.T)
    gt_pw_dif = (all_one @ ground_truth.T) - (ground_truth @ all_one.T)
    mask_pw = mask @ mask.T
    rank_loss = torch.mean(F.relu(pre_pw_dif * gt_pw_dif * mask_pw))
    loss = reg_loss + alpha * rank_loss
    return loss, reg_loss, rank_loss, return_ratio


# ======== Mixer Components ========
class TriU(nn.Module):
    """Triangular upper (causal) convolution along the time axis.

    For an input of shape ``(N, T, C)``, this layer applies an increasing
    receptive field along the time dimension such that the output at time
    ``t`` depends only on inputs up to ``t``.  It is parameterised by a
    separate linear layer for each time step.
    """

    def __init__(self, time_step: int) -> None:
        super().__init__()
        self.time_step = time_step
        self.triU = nn.ParameterList([
            nn.Linear(i + 1, 1) for i in range(time_step)
        ])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (N, C, T)
        # Output: (N, C, T) where each time position is a linear
        # combination of all previous inputs
        # We'll treat channels as batch and time as sequence
        x = self.triU[0](inputs[:, :, 0].unsqueeze(-1))
        for i in range(1, self.time_step):
            x = torch.cat([x, self.triU[i](inputs[:, :, : i + 1])], dim=-1)
        return x


class MixerBlock(nn.Module):
    """A simple MLP block used inside the mixer.

    It applies a linear projection, activation and another projection.  This
    implementation uses a Hardswish activation and optional dropout.
    """

    def __init__(self, mlp_dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.dense_1 = nn.Linear(mlp_dim, hidden_dim)
        self.act = nn.Hardswish()
        self.dense_2 = nn.Linear(hidden_dim, mlp_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense_1(x)
        x = self.act(x)
        if self.dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.dense_2(x)
        if self.dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class Mixer2dTriU(nn.Module):
    """Applies triangular mixing along the time axis and channel mixing.

    This layer implements a two‑step procedure: first, apply a triangular
    convolution (``TriU``) along the time dimension after layer normalisation,
    then apply a channel wise MLP (``MixerBlock``) to mix information across
    feature channels.  Residual connections are used between steps.
    """

    def __init__(self, time_steps: int, channels: int) -> None:
        super().__init__()
        self.ln_time = nn.LayerNorm([time_steps, channels])
        self.ln_channel = nn.LayerNorm([time_steps, channels])
        self.time_mixer = TriU(time_steps)
        self.channel_mixer = MixerBlock(channels, channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (N, T, C)
        x = self.ln_time(inputs)
        # transpose to (N, C, T) for TriU
        x_t = x.permute(0, 2, 1)
        x_t = self.time_mixer(x_t)
        # transpose back to (N, T, C)
        x_t = x_t.permute(0, 2, 1)
        # residual and channel mix
        x = self.ln_channel(x_t + inputs)
        y = self.channel_mixer(x)
        return x + y


class MultTime2dMixer(nn.Module):
    """Multiscale mixing along the time axis.

    Combines the outputs of two triangular mixers: one operating on the
    original sequence length and one on a downsampled sequence.  The outputs
    are concatenated along the time dimension together with the original
    inputs.
    """

    def __init__(self, time_steps: int, channels: int) -> None:
        super().__init__()
        self.mixer_full = Mixer2dTriU(time_steps, channels)
        self.mixer_half = Mixer2dTriU(time_steps // 2, channels)

    def forward(self, inputs: torch.Tensor, downsampled: torch.Tensor) -> torch.Tensor:
        # inputs: (N, T, C)
        # downsampled: (N, T/2, C)
        x_full = self.mixer_full(inputs)
        x_half = self.mixer_half(downsampled)
        # Concatenate: original inputs, full resolution mixing and half resolution mixing
        return torch.cat([inputs, x_full, x_half], dim=1)


# ======== Base Model ========
class StockMixerBase(nn.Module):
    """Abstract base class encapsulating the common logic of StockMixer.

    Parameters
    ----------
    stocks : int
        Number of assets/stocks.
    time_steps : int
        Length of the historical lookback window.
    channels : int
        Number of feature channels per asset (e.g. technical indicators).
    market_hidden : int
        Dimension of the hidden representation used during stock mixing.
    depth : int
        Number of gMLP blocks (for derived classes that use gMLP).  For the
        original MLP mixer this parameter has no effect.
    dropout : float, default 0.0
        Dropout probability passed to the gMLP blocks.

    Notes
    -----
    The actual stock mixing must be implemented in ``forward_stock_mixing``
    by subclasses.  This allows different variants (e.g. simple MLP, gMLP
    without context, gMLP with context) to reuse the same time/indicator
    mixing pipeline.
    """

    def __init__(self, stocks: int, time_steps: int, channels: int, market_hidden: int, depth: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        self.stocks = stocks
        self.time_steps = time_steps
        self.channels = channels
        self.market_hidden = market_hidden
        self.depth = depth
        self.dropout = dropout
        # Convolution to downsample the time dimension by a factor of 2
        self.conv_down = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=2, stride=2)
        # Multiscale mixer to combine original and downsampled sequences
        self.mixer = MultTime2dMixer(time_steps, channels)
        # Channel reduction: map feature channels to a single scalar per time step
        self.channel_fc = nn.Linear(channels, 1)
        # Compute Tsum: 2*T + T/2
        self.tsum = time_steps * 2 + time_steps // 2
        # Two separate linear heads for the mixed and stock mixed sequences
        self.time_fc_y = nn.Linear(self.tsum, 1)
        self.time_fc_z = nn.Linear(self.tsum, 1)

    def forward_stock_mixing(self, y: torch.Tensor, ctx: torch.Tensor | None = None) -> torch.Tensor:
        """Perform mixing across the stock dimension.

        Subclasses must override this method.  The input ``y`` has shape
        ``(N, T_sum)`` and represents the time/indicator mixed features per
        asset.  The optional context ``ctx`` is only used by context aware
        variants.
        """
        raise NotImplementedError

    def forward(self, inputs: torch.Tensor, ctx: torch.Tensor | None = None) -> torch.Tensor:
        # inputs: (N, T, F)
        # Downsample time dimension by a factor of 2 via convolution
        x1 = inputs.permute(0, 2, 1)  # (N, F, T)
        x1 = self.conv_down(x1)       # (N, F, T/2)
        x1 = x1.permute(0, 2, 1)     # (N, T/2, F)
        # Mix the original and downsampled sequences
        y = self.mixer(inputs, x1)    # (N, T_sum, F)
        # Reduce channels to scalar per time position
        y = self.channel_fc(y).squeeze(-1)  # (N, T_sum)
        # Perform stock mixing (implemented by subclass)
        z = self.forward_stock_mixing(y, ctx)
        # Aggregate across time for both paths
        y_out = self.time_fc_y(y)  # (N, 1)
        z_out = self.time_fc_z(z)  # (N, 1)
        return y_out + z_out
