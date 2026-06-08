import torch
import torch.nn as nn
import torch.nn.functional as F


def get_loss(prediction, ground_truth, base_price, mask, batch_size, alpha):
    """
    Compute the combined regression and ranking loss used in the original paper.

    Parameters
    ----------
    prediction : torch.Tensor
        Raw model outputs with shape ``(batch_size, 1)``.
    ground_truth : torch.Tensor
        Target return ratios with shape ``(batch_size, 1)``.
    base_price : torch.Tensor
        Last closing price for each stock in the lookback window used to compute actual
        return ratios, shape ``(batch_size, 1)``.
    mask : torch.Tensor
        Binary indicator specifying whether a given stock has valid ground truth. A value
        of 0 means that particular example should be ignored. Shape ``(batch_size, 1)``.
    batch_size : int
        Number of stocks in the batch.
    alpha : float
        Weight applied to the pairwise ranking loss.

    Returns
    -------
    tuple(float, float, float, torch.Tensor)
        A tuple containing the total loss, the regression loss, the ranking loss and
        the predicted return ratios (after dividing by base price).
    """
    device = prediction.device
    all_one = torch.ones(batch_size, 1, dtype=torch.float32).to(device)
    # convert raw output to return ratio by dividing by base price
    return_ratio = (prediction - base_price) / base_price
    reg_loss = F.mse_loss(return_ratio * mask, ground_truth * mask)
    # compute pairwise differences for ranking loss
    pre_pw_dif = return_ratio @ all_one.t() - all_one @ return_ratio.t()
    gt_pw_dif = all_one @ ground_truth.t() - ground_truth @ all_one.t()
    mask_pw = mask @ mask.t()
    # hinge loss encouraging the ordering of predictions to match the ordering of ground truth
    rank_loss = torch.mean(F.relu(pre_pw_dif * gt_pw_dif * mask_pw))
    loss = reg_loss + alpha * rank_loss
    return loss, reg_loss, rank_loss, return_ratio


class SpatialGatingUnit(nn.Module):
    """
    The spatial gating unit used by gMLP.  It splits the input along the feature
    dimension, applies a linear projection along the sequence dimension to one half,
    and multiplies it elementwise with the other half.  This encourages each time
    step to interact with every other through the learned weights.
    """
    def __init__(self, hidden_dim: int, seq_len: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        # project along the sequence (time) dimension
        self.spatial_proj = nn.Linear(seq_len, seq_len)
        # initialize the projection to the identity so training starts similar to
        # an MLP and gradually learns long range interactions
        nn.init.eye_(self.spatial_proj.weight)
        nn.init.zeros_(self.spatial_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (batch, seq_len, hidden_dim)
        u, v = x.chunk(2, dim=-1)  # split channels into two halves
        # apply projection along time dimension on v
        v = v.permute(0, 2, 1)  # (batch, hidden_dim/2, seq_len)
        v = self.spatial_proj(v)
        v = v.permute(0, 2, 1)  # (batch, seq_len, hidden_dim/2)
        return u * v  # elementwise product


class GatedMLPBlock(nn.Module):
    """
    A single gMLP block consisting of a feed-forward network followed by a spatial
    gating unit.  This is adapted from the gMLP architecture by Liu et al. and
    encourages interactions across the sequence (time) dimension without using
    explicit attention.
    """
    def __init__(self, input_dim: int, hidden_dim: int, seq_len: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        # expand to twice the hidden dimension for gating
        self.fc1 = nn.Linear(input_dim, 2 * hidden_dim)
        self.act = nn.GELU()
        self.sgu = SpatialGatingUnit(hidden_dim * 2, seq_len)
        # project back to original dimension
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        # feed-forward expansion
        x = self.fc1(x)
        x = self.act(x)
        # spatial gating unit mixes information across time
        x = self.sgu(x)
        # project back
        x = self.fc2(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x + residual


class IndicatorMLP(nn.Module):
    """
    Simple per-timestep feed-forward network to mix features (indicators) at
    each time step independently.  This plays a similar role to the indicator
    mixing in StockMixer.
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.Hardswish()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, in_dim)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class MarketContextGating(nn.Module):
    """
    Inject global market context into stock-specific features via gating.

    After computing per-stock embeddings, a global context vector is obtained by
    averaging the embeddings across all stocks.  A small gating network then
    computes a set of coefficients in ``[0, 1]`` which are used to rescale each
    stock embedding.  This allows the model to adaptively emphasize or de-
    emphasize certain dimensions based on the overall market condition.
    """
    def __init__(self, feature_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.Hardswish(),
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (stocks, feature_dim)
        # compute global context as mean across stocks
        global_ctx = x.mean(dim=0, keepdim=True)  # (1, feature_dim)
        global_ctx = self.norm(global_ctx)
        gating = self.fc(global_ctx)  # (1, feature_dim)
        return x * gating  # broadcast across stocks


class CrossStockAttention(nn.Module):
    """
    Multi-head attention module to enable each stock to attend to other stocks.

    The attention is performed on stock-level embeddings after temporal mixing.
    It encourages information flow across stocks without requiring explicit
    relation graphs.  The implementation uses PyTorch's built-in
    MultiheadAttention in batch-first mode for convenience.
    """
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads,
                                         dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (stocks, embed_dim)
        residual = x
        x = x.unsqueeze(1)  # (stocks, 1, embed_dim)
        attn_out, _ = self.mha(query=x, key=x, value=x)  # (stocks, 1, embed_dim)
        attn_out = attn_out.squeeze(1)  # (stocks, embed_dim)
        x = residual + self.dropout(attn_out)
        return self.norm(x)


class GatedMLPStockModel(nn.Module):
    """
    A simple yet expressive model that replaces the time mixing in StockMixer
    with a gMLP-based sequence mixer and integrates a market context gating
    mechanism.  The model processes per-stock sequences of financial indicators
    and produces a single scalar prediction for each stock corresponding to the
    expected return ratio.

    The high-level pipeline is as follows:

    1. **Indicator Mixing**: Mix features within each time step using an MLP.
    2. **Time Mixing (gMLP)**: Propagate information along the temporal axis using
       a gated MLP block.  This acts similarly to the multi-scale time mixing in
       StockMixer but is simpler and leverages the spatial gating unit for
       long-range dependencies.
    3. **Pooling**: Aggregate the sequence dimension by average pooling to obtain
       a fixed-size representation per stock.
    4. **Market Context Gating**: Re-weight each feature dimension based on the
       global market representation.
    5. **Cross-Stock Attention (optional)**: Allow stocks to attend to each
       other via a self-attention mechanism to capture stock correlations.
    6. **Regression Head**: Predict the one-day-ahead return ratio via a linear
       layer.

    This design is intentionally modular: the cross-stock attention can be
    enabled or disabled by setting ``use_cross_attn``.  When disabled, the
    model remains lightweight.
    """
    def __init__(self, stocks: int, time_steps: int, channels: int,
                 hidden_dim: int = 32, use_cross_attn: bool = True,
                 num_heads: int = 4):
        super().__init__()
        self.stocks = stocks
        self.time_steps = time_steps
        self.channels = channels
        self.hidden_dim = hidden_dim
        # indicator mixing
        self.indicator_mlp = IndicatorMLP(channels, hidden_dim, hidden_dim)
        # gated time mixing
        self.time_mixer = GatedMLPBlock(input_dim=hidden_dim,
                                        hidden_dim=hidden_dim,
                                        seq_len=time_steps)
        # market context gating
        self.market_gate = MarketContextGating(feature_dim=hidden_dim)
        # optional cross-stock attention
        self.use_cross_attn = use_cross_attn
        if self.use_cross_attn:
            self.cross_attn = CrossStockAttention(embed_dim=hidden_dim,
                                                  num_heads=num_heads)
        # regression head
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the GatedMLPStockModel.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape ``(stocks, time_steps, channels)`` representing the
            lookback window for each stock.

        Returns
        -------
        torch.Tensor
            Predictions of shape ``(stocks, 1)`` corresponding to the expected
            return ratio for each stock.
        """
        # mix indicators independently at each time step
        x = self.indicator_mlp(inputs)  # (stocks, time_steps, hidden_dim)
        # gated time mixing
        x = self.time_mixer(x)  # (stocks, time_steps, hidden_dim)
        # average pool over time dimension
        x = x.mean(dim=1)  # (stocks, hidden_dim)
        # inject market context via gating
        x = self.market_gate(x)  # (stocks, hidden_dim)
        # optional cross-stock attention
        if self.use_cross_attn:
            x = self.cross_attn(x)  # (stocks, hidden_dim)
        # final regression head
        out = self.head(x)  # (stocks, 1)
        return out


class CrossAttentionStockModel(nn.Module):
    """
    An alternative architecture that leverages transformer-style cross-attention to
    model both temporal dynamics and inter-stock relationships.  The sequence
    information is first compressed into a stock-level embedding via an LSTM, and
    the stock embeddings then interact through a multi-head attention layer.

    Compared to StockMixer, this model is slightly more complex but benefits
    from the expressive power of attention while remaining fully differentiable
    and end-to-end trainable.
    """
    def __init__(self, stocks: int, time_steps: int, channels: int,
                 hidden_dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.stocks = stocks
        self.time_steps = time_steps
        self.channels = channels
        self.hidden_dim = hidden_dim
        # temporal encoder: use a simple LSTM to summarize the sequence
        self.temporal_encoder = nn.LSTM(input_size=channels,
                                        hidden_size=hidden_dim // 2,
                                        num_layers=1,
                                        batch_first=True,
                                        bidirectional=True)
        # cross-stock attention
        self.stock_attn = nn.MultiheadAttention(embed_dim=hidden_dim,
                                                num_heads=num_heads,
                                                batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        # regression head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Hardswish(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CrossAttentionStockModel.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape ``(stocks, time_steps, channels)`` representing the
            lookback window for each stock.

        Returns
        -------
        torch.Tensor
            Predictions of shape ``(stocks, 1)`` corresponding to the expected
            return ratio for each stock.
        """
        # Encode temporal information using LSTM
        # The LSTM expects (batch, seq_len, input_dim), so our inputs are already in that shape
        lstm_out, _ = self.temporal_encoder(inputs)  # (stocks, time_steps, hidden_dim)
        # Average pool over time to obtain a single embedding per stock
        stock_emb = lstm_out.mean(dim=1, keepdim=True)  # (stocks, 1, hidden_dim)
        # Apply multi-head attention; each stock attends to all others
        attn_out, _ = self.stock_attn(query=stock_emb,
                                      key=stock_emb,
                                      value=stock_emb)  # (stocks, 1, hidden_dim)
        attn_out = attn_out.squeeze(1)  # (stocks, hidden_dim)
        attn_out = self.norm(attn_out)
        # Final regression head
        out = self.head(attn_out)  # (stocks, 1)
        return out


__all__ = [
    "get_loss",
    "SpatialGatingUnit",
    "GatedMLPBlock",
    "IndicatorMLP",
    "MarketContextGating",
    "CrossStockAttention",
    "GatedMLPStockModel",
    "CrossAttentionStockModel",
]