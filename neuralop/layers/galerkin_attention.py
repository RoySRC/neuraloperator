"""Galerkin attention and encoder blocks with author-compatible state names.

Adapted from ``scaomath/galerkin-transformer``, revision
``f9e7d6abc24e958b1165f48b22625c5bab688043`` (``libs/layers.py`` and
``libs/model.py``). This module retains the static Galerkin path, including
headwise key/value normalization and coordinates prepended to every head.

The historical attention helper applied functional dropout with its default
probability even during evaluation. Here dropout uses its configured
probability and respects ``train()``/``eval()``. Evaluation therefore matches
the original code with its attention dropout explicitly disabled.

MIT License

Copyright (c) 2021 Shuhao Cao

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

import copy
import math
from numbers import Integral, Real

import torch
from torch import nn


def _positive_integer(name, value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _dropout_probability(name, value):
    if isinstance(value, bool) or not isinstance(value, Real) or not 0 <= value < 1:
        raise ValueError(f"{name} must lie in [0, 1).")


class GalerkinAttention(nn.Module):
    r"""Linear Galerkin self-attention with coordinate-enriched heads.

    Computes :math:`Q(K^T V)/N` without a softmax. Each head normalizes its key
    and value features independently, then prepends the input coordinates to
    Q, K and V. The resulting head features are concatenated and projected
    back to ``d_model`` when coordinates are supplied.

    Parameters
    ----------
    n_head, d_model : int
        Number of heads and total feature width. ``d_model`` must be divisible
        by ``n_head``; each head has width ``d_model // n_head``.
    pos_dim : int, optional
        Number of coordinates prepended to each head, by default 2.
    dropout : float, optional
        Dropout on the feature interaction matrix, by default 0.0. Disabled
        during evaluation, including when the configured probability is nonzero.
    xavier_init, diagonal_weight : float, optional
        Xavier initialization gain and diagonal addition for the Q/K/V
        projections. Their defaults match the author's attention constructor.
    symmetric_init : bool, optional
        Add the transpose to each initialized projection, by default False.
    norm : bool, optional
        Normalize keys and values independently per head, by default True.
    norm_type : {'layer', 'instance'}, optional
        Layer normalization over head features, or instance normalization over
        source points, by default 'layer'.
    eps : float, optional
        Normalization epsilon, by default 1e-5.
    attention_type : str, optional
        Must be 'galerkin'. Other author attention variants are not implemented.
    debug : bool, optional
        Retained for compatibility with author constructor settings.
    """

    def __init__(
        self,
        n_head,
        d_model,
        pos_dim=2,
        attention_type="galerkin",
        dropout=0.0,
        xavier_init=1e-4,
        diagonal_weight=1e-2,
        symmetric_init=False,
        norm=True,
        norm_type="layer",
        eps=1e-5,
        debug=False,
    ):
        super().__init__()
        _positive_integer("n_head", n_head)
        _positive_integer("d_model", d_model)
        if d_model % n_head:
            raise ValueError("d_model must be divisible by n_head.")
        if (
            isinstance(pos_dim, bool)
            or not isinstance(pos_dim, Integral)
            or pos_dim < 0
        ):
            raise ValueError("pos_dim must be a nonnegative integer.")
        if attention_type != "galerkin":
            raise ValueError("Only Galerkin attention is supported.")
        if norm_type not in ("layer", "instance"):
            raise ValueError("norm_type must be 'layer' or 'instance'.")
        if not isinstance(eps, Real) or not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive.")
        _dropout_probability("dropout", dropout)
        for name, value in (
            ("xavier_init", xavier_init),
            ("diagonal_weight", diagonal_weight),
        ):
            if not isinstance(value, Real) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")

        self.attention_type = attention_type
        self.d_model = d_model
        self.d_k = d_model // n_head
        self.n_head = n_head
        self.pos_dim = pos_dim
        # Keep original names and initialization order for checkpoint compatibility.
        self.linears = nn.ModuleList(
            [copy.deepcopy(nn.Linear(d_model, d_model)) for _ in range(3)]
        )
        self.xavier_init = xavier_init
        self.diagonal_weight = diagonal_weight
        self.symmetric_init = symmetric_init
        if xavier_init > 0:
            self._reset_parameters()
        self.add_norm = norm
        self.norm_type = norm_type
        self.norm_eps = eps
        if norm:
            norm_class = nn.LayerNorm if norm_type == "layer" else nn.InstanceNorm1d
            options = {"eps": eps}
            if norm_type == "instance":
                options["affine"] = True
            self.norm_K = nn.ModuleList(
                [copy.deepcopy(norm_class(self.d_k, **options)) for _ in range(n_head)]
            )
            self.norm_V = nn.ModuleList(
                [copy.deepcopy(norm_class(self.d_k, **options)) for _ in range(n_head)]
            )
        if pos_dim > 0:
            self.fc = nn.Linear(d_model + n_head * pos_dim, d_model)
        self.attn_weight = None
        self.dropout = nn.Dropout(dropout)
        self.debug = debug

    def _reset_parameters(self):
        with torch.no_grad():
            for param in self.linears.parameters():
                if param.ndim > 1:
                    nn.init.xavier_uniform_(param, gain=self.xavier_init)
                    if self.diagonal_weight > 0:
                        param.add_(
                            self.diagonal_weight
                            * torch.eye(
                                param.size(-1), dtype=param.dtype, device=param.device
                            )
                        )
                    if self.symmetric_init:
                        param.add_(param.T.clone())
                else:
                    nn.init.constant_(param, 0)

    def forward(self, query, key, value, pos=None, mask=None, weight=None):
        """Return transformed features and the headwise interaction matrix.

        Q/K/V must have the same shape ``(B, N, d_model)``. Coordinates, when
        supplied, have shape ``(B, N, pos_dim)``. ``weight`` optionally multiplies
        Q and K before projection, as in the original implementation; it is not
        a general quadrature-weight API. Causal masks are unsupported.
        """
        if mask is not None:
            raise ValueError(
                "Galerkin linear attention does not support a causal mask."
            )
        if query.ndim != 3 or query.shape[-1] != self.d_model or 0 in query.shape:
            raise ValueError("query must have nonempty shape (B, N, d_model).")
        if key.shape != query.shape or value.shape != query.shape:
            raise ValueError("Galerkin self-attention requires equal Q/K/V shapes.")
        if pos is not None and self.pos_dim > 0:
            if pos.shape != (*query.shape[:2], self.pos_dim):
                raise ValueError("pos must have shape (B, N, pos_dim).")
            if pos.dtype != query.dtype or pos.device != query.device:
                raise ValueError("pos must use the same dtype and device as query.")
        bsz, n_points, _ = query.shape
        if weight is not None:
            query, key = weight * query, weight * key
        query, key, value = [
            layer(x).view(bsz, -1, self.n_head, self.d_k).transpose(1, 2)
            for layer, x in zip(self.linears, (query, key, value))
        ]
        if self.add_norm:
            if self.norm_type == "instance":
                key, value = key.transpose(-2, -1), value.transpose(-2, -1)
            key = torch.stack(
                [norm(key[:, i]) for i, norm in enumerate(self.norm_K)], dim=1
            )
            value = torch.stack(
                [norm(value[:, i]) for i, norm in enumerate(self.norm_V)], dim=1
            )
            if self.norm_type == "instance":
                key, value = key.transpose(-2, -1), value.transpose(-2, -1)
        if pos is not None and self.pos_dim > 0:
            coordinates = pos.unsqueeze(1).repeat(1, self.n_head, 1, 1)
            query, key, value = [
                torch.cat((coordinates, x), dim=-1) for x in (query, key, value)
            ]

        interaction = torch.matmul(key.transpose(-2, -1), value) / n_points
        interaction = self.dropout(interaction)
        output = torch.matmul(query, interaction)
        self.attn_weight = interaction
        output = output.transpose(1, 2).contiguous().view(bsz, n_points, -1)
        if pos is not None and self.pos_dim > 0:
            output = self.fc(output)
        return output, interaction


class _FeedForward(nn.Module):
    def __init__(self, in_dim, dim_feedforward, batch_norm, activation, dropout):
        super().__init__()
        self.lr1 = nn.Linear(in_dim, dim_feedforward)
        activations = {"relu": nn.ReLU, "silu": nn.SiLU, "gelu": nn.GELU}
        if activation not in activations:
            raise ValueError("activation_type must be 'relu', 'silu' or 'gelu'.")
        self.activation = activations[activation]()
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn = nn.BatchNorm1d(dim_feedforward)
        self.lr2 = nn.Linear(dim_feedforward, in_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(self.activation(self.lr1(x)))
        if self.batch_norm:
            x = self.bn(x.permute(0, 2, 1)).permute(0, 2, 1)
        return self.lr2(x)


class GalerkinTransformerEncoderLayer(nn.Module):
    """Author-compatible Galerkin encoder with residual attention and MLP.

    Argument and state-dictionary names follow the original
    ``SimpleTransformerEncoderLayer``. This module implements only the
    Galerkin path with spatial coordinates. Optional post-residual LayerNorm,
    per-head attention normalization, and feedforward BatchNorm are retained.
    Positional sequence encoding and other attention architectures are excluded.

    ``dropout`` controls attention interactions and both residual branches;
    ``ffn_dropout`` controls the MLP's hidden activations. All dropout honors
    the module's training/evaluation mode.
    """

    def __init__(
        self,
        d_model=128,
        pos_dim=2,
        n_head=4,
        dim_feedforward=None,
        attention_type="galerkin",
        pos_emb=False,
        layer_norm=False,
        attn_norm=None,
        norm_type="layer",
        norm_eps=None,
        batch_norm=False,
        attn_weight=False,
        xavier_init=1e-2,
        diagonal_weight=1e-2,
        symmetric_init=False,
        residual_type="add",
        activation_type="relu",
        dropout=0.1,
        ffn_dropout=None,
        debug=False,
    ):
        super().__init__()
        if pos_emb:
            raise ValueError(
                "Sequence positional encoding is not part of this static Galerkin layer."
            )
        if residual_type not in ("add", "plus", "subtract", "minus", None):
            raise ValueError(
                "residual_type must be 'add'/'plus' or 'subtract'/'minus'."
            )
        dropout = 0.05 if dropout is None else dropout
        ffn_dropout = dropout if ffn_dropout is None else ffn_dropout
        norm_eps = 1e-5 if norm_eps is None else norm_eps
        norm_type = "layer" if norm_type is None else norm_type
        attn_norm = not layer_norm if attn_norm is None else attn_norm
        if not layer_norm and not attn_norm:
            attn_norm = True
        _dropout_probability("dropout", dropout)
        _dropout_probability("ffn_dropout", ffn_dropout)
        dim_feedforward = 2 * d_model if dim_feedforward is None else dim_feedforward
        _positive_integer("dim_feedforward", dim_feedforward)

        self.attn = GalerkinAttention(
            n_head=n_head,
            d_model=d_model,
            attention_type=attention_type,
            diagonal_weight=diagonal_weight,
            xavier_init=xavier_init,
            symmetric_init=symmetric_init,
            pos_dim=pos_dim,
            norm=attn_norm,
            norm_type=norm_type,
            eps=norm_eps,
            dropout=dropout,
        )
        self.d_model = d_model
        self.n_head = n_head
        self.pos_dim = pos_dim
        self.norm_eps = norm_eps
        self.add_layer_norm = layer_norm
        if layer_norm:
            self.layer_norm1 = nn.LayerNorm(d_model, eps=norm_eps)
            self.layer_norm2 = nn.LayerNorm(d_model, eps=norm_eps)
        self.ff = _FeedForward(
            d_model, dim_feedforward, batch_norm, activation_type, ffn_dropout
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.residual_type = residual_type
        self.add_pos_emb = False
        self.debug = debug
        self.attn_weight = attn_weight

    def forward(self, x, pos=None, weight=None):
        """Transform ``x`` (B, N, C), optionally returning interaction weights."""
        att_output, attn_weight = self.attn(x, x, x, pos=pos, weight=weight)
        if self.residual_type in ("add", "plus", None):
            x = x + self.dropout1(att_output)
        else:
            x = x - self.dropout1(att_output)
        if self.add_layer_norm:
            x = self.layer_norm1(x)
        x = x + self.dropout2(self.ff(x))
        if self.add_layer_norm:
            x = self.layer_norm2(x)
        return (x, attn_weight) if self.attn_weight else x
