"""Components of the static two-dimensional OFormer encoder-decoder.

Adapted from ``BaratiLab/OFormer`` at commit
``f1111b3c634893329e14e47e8903e421dd726a0c`` (``uniform_grids/nn_module``).
The state-dictionary names and forward operations match the authors' Darcy
``SpatialEncoder2D`` and ``PointWiseDecoder2DSimple`` implementations.

Copyright (c) 2023 BaratiLab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class _GeGELU(nn.Module):
    """Author's gated activation: GELU(first half) times second half."""

    def forward(self, x):
        a, b = x.chunk(2, dim=-1)
        return F.gelu(a) * b


class _FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim * 2),
            _GeGELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class _RotaryEmbedding(nn.Module):
    def __init__(self, dim, min_freq, scale):
        super().__init__()
        self.min_freq = min_freq
        self.scale = scale
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, coordinates):
        coordinates = coordinates.to(self.inv_freq)
        coordinates = coordinates * (self.scale / self.min_freq)
        freqs = torch.einsum("...i,j->...ij", coordinates, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)


def _rotate_half(x):
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _rotary_2d(x, pos, embedding):
    """Rotate independent halves of each attention head with the two axes."""
    first, second = x.chunk(2, dim=-1)
    fx = embedding(pos[..., 0]).unsqueeze(1)
    fy = embedding(pos[..., 1]).unsqueeze(1)
    return torch.cat(
        (
            first * fx.cos() + _rotate_half(first) * fx.sin(),
            second * fy.cos() + _rotate_half(second) * fy.sin(),
        ),
        dim=-1,
    )


def _split_heads(x, heads):
    b, n, channels = x.shape
    return x.reshape(b, n, heads, channels // heads).permute(0, 2, 1, 3)


def _join_heads(x):
    b, heads, n, channels = x.shape
    return x.permute(0, 2, 1, 3).reshape(b, n, heads * channels)


def _author_normalize(x, norm):
    """Preserve the published checkpoint's feature-axis normalization.

    The original code passes ``(batch * heads, points, head_channels)`` to
    InstanceNorm1d. With affine=False, this normalizes the last axis rather
    than the point axis, despite the original ``norm_wrt_domain`` name.
    Calling the functional operation preserves that calculation without the
    module's misleading num_features warning. Transposing would change the
    learned operator.
    """
    b, heads, n, channels = x.shape
    normalized = F.instance_norm(
        x.reshape(b * heads, n, channels),
        running_mean=None,
        running_var=None,
        weight=None,
        bias=None,
        use_input_stats=True,
        momentum=norm.momentum,
        eps=norm.eps,
    )
    return normalized.reshape(b, heads, n, channels)


def _initialize_heads(weight, heads, dim_head, offset=0):
    """Orthogonal head initialization plus the author's diagonal addition."""
    gain = 1.0 / dim_head
    with torch.no_grad():
        diagonal = torch.eye(
            dim_head, weight.shape[-1], dtype=weight.dtype, device=weight.device
        )
        for h in range(heads):
            block = weight[(offset + h) * dim_head : (offset + h + 1) * dim_head]
            nn.init.orthogonal_(block, gain=gain)
            block.add_(diagonal * gain)


class _LinearAttention(nn.Module):
    """Galerkin self-attention Q(K^T V)/N with author-compatible weights."""

    def __init__(self, dim, heads, dropout, scale, min_freq):
        super().__init__()
        self.heads = heads
        self.dim_head = dim
        inner_dim = heads * dim
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.k_norm = nn.InstanceNorm1d(dim)
        self.v_norm = nn.InstanceNorm1d(dim)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if heads != 1
            else nn.Identity()
        )
        # In the original Galerkin self-attention, only Q is reinitialized.
        _initialize_heads(self.to_qkv.weight, heads, dim)
        self.emb_module = _RotaryEmbedding(dim // 2, min_freq, scale)

    def forward(self, x, pos):
        q, k, v = (_split_heads(t, self.heads) for t in self.to_qkv(x).chunk(3, dim=-1))
        k = _author_normalize(k, self.k_norm)
        v = _author_normalize(v, self.v_norm)
        q = _rotary_2d(q, pos, self.emb_module)
        k = _rotary_2d(k, pos, self.emb_module)
        dots = torch.matmul(k.transpose(-1, -2), v)
        out = torch.matmul(q, dots) * (1.0 / q.shape[2])
        return self.to_out(_join_heads(out))


class _CrossLinearAttention(nn.Module):
    """Linear cross-attention from query points to encoded input points."""

    def __init__(self, dim, heads, min_freq):
        super().__init__()
        self.heads = heads
        self.dim_head = dim
        inner_dim = heads * dim
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.k_norm = nn.InstanceNorm1d(dim)
        self.v_norm = nn.InstanceNorm1d(dim)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(0.0))
            if heads != 1
            else nn.Identity()
        )
        # Match the source initialization order: K_h, V_h for each head, then Q.
        gain = 1.0 / dim
        with torch.no_grad():
            diagonal = torch.eye(
                dim, dtype=self.to_kv.weight.dtype, device=self.to_kv.weight.device
            )
            for h in range(heads):
                for index in (h, heads + h):
                    block = self.to_kv.weight[index * dim : (index + 1) * dim]
                    nn.init.orthogonal_(block, gain=gain)
                    block.add_(gain * diagonal)
        _initialize_heads(self.to_q.weight, heads, dim)
        self.emb_module = _RotaryEmbedding(dim // 2, min_freq, 16)

    def forward(self, x, z, x_pos, z_pos):
        q = _split_heads(self.to_q(x), self.heads)
        k, v = (_split_heads(t, self.heads) for t in self.to_kv(z).chunk(2, dim=-1))
        k = _author_normalize(k, self.k_norm)
        v = _author_normalize(v, self.v_norm)
        q = _rotary_2d(q, x_pos, self.emb_module)
        k = _rotary_2d(k, z_pos, self.emb_module)
        dots = torch.matmul(k.transpose(-1, -2), v)
        out = torch.matmul(q, dots) * (1.0 / z.shape[1])
        return self.to_out(_join_heads(out))


class _Transformer(nn.Module):
    def __init__(self, dim, depth, heads, res, use_ln, dropout):
        super().__init__()
        self.layers = nn.ModuleList()
        self.use_ln = use_ln
        # The author model assumes depth >= 2. Truncation also supports depth=1.
        scales = ([res, res // 4] + [1] * max(depth - 2, 0))[:depth]
        for scale in scales:
            attention = _LinearAttention(dim, heads, dropout, scale, 1.0 / res)
            feedforward = _FeedForward(dim, dim, dropout)
            if use_ln:
                self.layers.append(
                    nn.ModuleList(
                        [nn.LayerNorm(dim), attention, nn.LayerNorm(dim), feedforward]
                    )
                )
            else:
                self.layers.append(nn.ModuleList([attention, feedforward]))

    def forward(self, x, pos):
        for layer in self.layers:
            if self.use_ln:
                norm1, attention, norm2, feedforward = layer
                # The residuals use normalized x, exactly as in the author code.
                x = norm1(x)
                x = attention(x, pos) + x
                x = norm2(x)
                x = feedforward(x) + x
            else:
                attention, feedforward = layer
                x = attention(x, pos) + x
                x = feedforward(x) + x
        return x


class _GaussianFourierFeatures(nn.Module):
    def __init__(self, mapping_size, scale):
        super().__init__()
        # The author's non-trainable Parameter is a buffer here; its key is kept.
        self.register_buffer("_B", torch.randn(2, mapping_size) * scale)

    def forward(self, x):
        b, n, channels = x.shape
        projected = (x.reshape(b * n, channels) @ self._B).reshape(b, n, -1)
        projected = 2 * math.pi * projected
        return torch.cat((torch.sin(projected), torch.cos(projected)), dim=-1)


class _CrossFormer(nn.Module):
    def __init__(self, dim, heads, res):
        super().__init__()
        self.cross_attn_module = _CrossLinearAttention(dim, heads, 1.0 / res)
        self.ffn = _FeedForward(dim, dim)

    def forward(self, x, z, x_pos, z_pos):
        x = self.cross_attn_module(x, z, x_pos, z_pos) + x
        return self.ffn(x) + x


class OFormerEncoder(nn.Module):
    """Static 2D OFormer encoder, compatible with SpatialEncoder2D state dicts.

    Parameters
    ----------
    input_channels : int
        Number of input channels, including the two appended coordinates.
    in_emb_dim : int
        Token width and width of each attention head; must be divisible by four.
    out_seq_emb_dim : int
        Width of the encoded sequence passed to the decoder.
    heads : int
        Number of encoder attention heads.
    depth : int
        Number of encoder attention blocks.
    res : int
        Reference resolution controlling rotary frequencies. It is a fixed
        architecture setting, independent of the number of runtime points.
    use_ln : bool, optional
        Whether to apply the author's layer-normalized residual blocks.
    emb_dropout : float, optional
        Dropout probability after input projection, by default 0.05.
    attention_dropout : float, optional
        Dropout probability in attention projections and feedforward blocks,
        by default 0.03, matching the author model.
    """

    def __init__(
        self,
        input_channels,
        in_emb_dim,
        out_seq_emb_dim,
        heads,
        depth,
        res,
        use_ln=True,
        emb_dropout=0.05,
        attention_dropout=0.03,
    ):
        super().__init__()
        if in_emb_dim <= 0 or in_emb_dim % 4:
            raise ValueError("in_emb_dim must be positive and divisible by four.")
        if min(input_channels, out_seq_emb_dim, heads, depth, res) <= 0:
            raise ValueError("Channel counts, heads, depth and res must be positive.")
        self.to_embedding = nn.Sequential(
            nn.Linear(input_channels, in_emb_dim, bias=False)
        )
        self.dropout = nn.Dropout(emb_dropout)
        self.s_transformer = _Transformer(
            in_emb_dim, depth, heads, res, use_ln, attention_dropout
        )
        self.to_out = nn.Sequential(nn.Linear(in_emb_dim, out_seq_emb_dim, bias=False))

    def forward(self, x, input_pos):
        """Encode ``x`` (B, N, C) using ``input_pos`` (B, N, 2)."""
        x = self.dropout(self.to_embedding(x))
        return self.to_out(self.s_transformer(x, input_pos))


class OFormerDecoder(nn.Module):
    """Static 2D OFormer decoder matching PointWiseDecoder2DSimple state dicts.

    Parameters
    ----------
    latent_channels : int
        Encoded width and width of each attention head; divisible by four.
    out_channels : int
        Number of output field channels.
    res : int, optional
        Fixed reference resolution for rotary frequencies, by default 211.
    scale : float, optional
        Scale of the frozen Gaussian Fourier feature matrix, by default 0.5.
    heads : int, optional
        Number of cross-attention heads, by default 4, as in the author model.
    """

    def __init__(self, latent_channels, out_channels, res=211, scale=0.5, heads=4):
        super().__init__()
        if latent_channels <= 0 or latent_channels % 4:
            raise ValueError("latent_channels must be positive and divisible by four.")
        if min(out_channels, res, heads) <= 0:
            raise ValueError("out_channels, res and heads must be positive.")
        self.latent_channels = latent_channels
        self.out_channels = out_channels
        self.coordinate_projection = nn.Sequential(
            _GaussianFourierFeatures(latent_channels // 2, scale),
            nn.Linear(latent_channels, latent_channels, bias=False),
            nn.GELU(),
            nn.Linear(latent_channels, latent_channels, bias=False),
        )
        self.decoding_transformer = _CrossFormer(latent_channels, heads, res)
        self.to_out = nn.Sequential(
            nn.Linear(latent_channels + 2, latent_channels, bias=False),
            nn.GELU(),
            nn.Linear(latent_channels, latent_channels // 2, bias=False),
            nn.GELU(),
            nn.Linear(latent_channels // 2, out_channels, bias=True),
        )

    def decode(self, z):
        return self.to_out(z)

    def forward(self, z, propagate_pos, input_pos):
        """Decode (B, N, C) latents at (B, M, 2) query coordinates."""
        x = self.coordinate_projection(propagate_pos)
        z = self.decoding_transformer(x, z, propagate_pos, input_pos)
        return self.decode(torch.cat((z, propagate_pos), dim=-1))
