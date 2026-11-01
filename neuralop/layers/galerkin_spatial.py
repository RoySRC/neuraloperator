"""Spatial and spectral layers of the static 2D Galerkin Transformer.

Adapted from scaomath/galerkin-transformer at revision
f9e7d6abc24e958b1165f48b22625c5bab688043. State-dictionary names and the
float32 square-grid operations match the original Darcy architecture.
Interpolation wrappers additionally accept rectangular grids and optional
per-call size overrides; no saved architecture setting is mutated.

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

import math
from functools import partial
from numbers import Integral, Real

import torch
from torch import nn
from torch.nn import functional as F


def default(value, fallback):
    return fallback if value is None else value


def _activation(name):
    name = default(name, "silu")
    if name not in ("silu", "relu"):
        raise ValueError("activation must be 'silu' or 'relu'.")
    return nn.SiLU() if name == "silu" else nn.ReLU()


def _stages(sizes, allow_factors=False):
    """Validate the two sequential interpolation operations."""
    if not isinstance(sizes, (tuple, list)) or len(sizes) != 2:
        raise ValueError("interp_size must describe exactly two interpolation stages.")
    factors = all(isinstance(s, Real) and not isinstance(s, Integral) for s in sizes)
    if factors:
        if not allow_factors or any(not math.isfinite(s) or s <= 0 for s in sizes):
            raise ValueError(
                "Scale factors must be finite, positive and used by the encoder."
            )
        return tuple(float(s) for s in sizes), True
    result = []
    for size in sizes:
        if isinstance(size, Integral) and not isinstance(size, bool):
            size = (size, size)
        if (
            not isinstance(size, (tuple, list))
            or len(size) != 2
            or any(
                isinstance(v, bool) or not isinstance(v, Integral) or v < 1
                for v in size
            )
        ):
            raise ValueError(
                "Explicit interpolation stages must be positive (height, width) pairs."
            )
        result.append(tuple(int(v) for v in size))
    return tuple(result), False


class Shortcut2d(nn.Module):
    """
    (-1, in, S, S) -> (-1, out, S, S)
    Used in SimpleResBlock
    """

    def __init__(
        self,
        in_features=None,
        out_features=None,
    ):
        super(Shortcut2d, self).__init__()
        self.shortcut = nn.Linear(in_features, out_features)

    def forward(self, x, edge=None, grid=None):
        x = x.permute(0, 2, 3, 1)
        x = self.shortcut(x)
        x = x.permute(0, 3, 1, 2)
        return x


class Conv2dResBlock(nn.Module):
    """
    Conv2d + a residual block
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    Modified from ResNet's basic block, one conv less, no batchnorm
    No batchnorm
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size=3,
        padding=1,
        dilation=1,
        dropout=0.1,
        stride=1,
        bias=False,
        residual=False,
        basic_block=False,
        activation_type="silu",
    ):
        super(Conv2dResBlock, self).__init__()

        activation_type = default(activation_type, "silu")
        self.activation = _activation(activation_type)
        self.add_res = residual
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_dim,
                out_dim,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                stride=stride,
                bias=bias,
            ),
            nn.Dropout(dropout),
        )
        self.basic_block = basic_block
        if self.basic_block:
            self.conv1 = nn.Sequential(
                self.activation,
                nn.Conv2d(
                    out_dim,
                    out_dim,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=bias,
                ),
                nn.Dropout(dropout),
            )
        self.apply_shortcut = in_dim != out_dim

        if self.add_res:
            if self.apply_shortcut:
                self.res = Shortcut2d(in_dim, out_dim)
            else:
                self.res = nn.Identity()

    def forward(self, x):
        if self.add_res:
            h = self.res(x)

        x = self.conv(x)

        if self.basic_block:
            x = self.conv1(x)

        if self.add_res:
            return self.activation(x + h)
        else:
            return self.activation(x)


class Interp2dEncoder(nn.Module):
    r"""
    Using Interpolate instead of avg pool
    interp dim hard coded or using a factor
    old code uses lambda and cannot be pickled
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        interp_size=None,
        residual=False,
        activation_type="silu",
        dropout=0.1,
        debug=False,
    ):
        super(Interp2dEncoder, self).__init__()

        if out_dim < 3:
            raise ValueError(
                "out_dim must be at least three for the interpolation encoder."
            )
        conv_dim0 = out_dim // 3
        conv_dim1 = out_dim // 3
        conv_dim2 = int(out_dim - conv_dim0 - conv_dim1)
        padding1 = padding // 2 if padding // 2 >= 1 else 1
        padding2 = padding // 4 if padding // 4 >= 1 else 1
        activation_type = default(activation_type, "silu")
        self.interp_size, self.is_scale_factor = _stages(
            interp_size, allow_factors=True
        )
        self.conv0 = Conv2dResBlock(
            in_dim,
            out_dim,
            kernel_size=kernel_size,
            padding=padding,
            activation_type=activation_type,
            dropout=dropout,
            residual=residual,
        )
        self.conv1 = Conv2dResBlock(
            out_dim,
            conv_dim0,
            kernel_size=kernel_size,
            padding=padding1,
            stride=stride,
            residual=residual,
            dropout=dropout,
            activation_type=activation_type,
        )
        self.conv2 = Conv2dResBlock(
            conv_dim0,
            conv_dim1,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding2,
            residual=residual,
            dropout=dropout,
            activation_type=activation_type,
        )
        self.conv3 = Conv2dResBlock(
            conv_dim1,
            conv_dim2,
            kernel_size=kernel_size,
            residual=residual,
            dropout=dropout,
            activation_type=activation_type,
        )
        self.activation = _activation(activation_type)
        self.add_res = residual
        self.debug = debug

    def forward(self, x, interp_size=None):
        sizes, factors = _stages(
            self.interp_size if interp_size is None else interp_size, allow_factors=True
        )
        x = self.conv0(x)
        if factors:
            x = F.interpolate(
                x,
                scale_factor=sizes[0],
                mode="bilinear",
                recompute_scale_factor=True,
                align_corners=True,
            )
        else:
            x = F.interpolate(x, size=sizes[0], mode="bilinear", align_corners=True)
        x = self.activation(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        out = torch.cat([x1, x2, x3], dim=1)
        if self.add_res:
            out += x

        if factors:
            out = F.interpolate(
                out,
                scale_factor=sizes[1],
                mode="bilinear",
                recompute_scale_factor=True,
                align_corners=True,
            )
        else:
            out = F.interpolate(out, size=sizes[1], mode="bilinear", align_corners=True)
        out = self.activation(out)
        return out


class Interp2dUpsample(nn.Module):
    """
    interpolate then Conv2dResBlock
    old code uses lambda and cannot be pickled
    temp hard-coded dimensions
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size: int = 3,
        padding: int = 1,
        residual=False,
        conv_block=True,
        interp_mode="bilinear",
        interp_size=None,
        activation_type="silu",
        dropout=0.1,
        debug=False,
    ):
        super(Interp2dUpsample, self).__init__()
        activation_type = default(activation_type, "silu")
        self.activation = _activation(activation_type)
        self.dropout = nn.Dropout(dropout)
        if conv_block:
            self.conv = nn.Sequential(
                Conv2dResBlock(
                    in_dim,
                    out_dim,
                    kernel_size=kernel_size,
                    padding=padding,
                    residual=residual,
                    dropout=dropout,
                    activation_type=activation_type,
                ),
                self.dropout,
                self.activation,
            )
        self.conv_block = conv_block
        self.interp_size, _ = _stages(interp_size)
        if interp_mode not in ("bilinear", "bicubic"):
            raise ValueError("interp_mode must be bilinear or bicubic.")
        self.interp_mode = interp_mode
        self.debug = debug

    def forward(self, x, interp_size=None):
        sizes, _ = _stages(self.interp_size if interp_size is None else interp_size)
        x = F.interpolate(x, size=sizes[0], mode=self.interp_mode, align_corners=True)
        if self.conv_block:
            x = self.conv(x)
        x = F.interpolate(x, size=sizes[1], mode=self.interp_mode, align_corners=True)
        return x


class DownScaler(nn.Module):
    """Author-compatible interpolation encoder for channels-last 2D grids.

    ``interp_size`` describes two sequential resizes. Scalar floating-point
    factors preserve the released checkpoint's historical 0.54, 0.54 path.
    Explicit ``((H1, W1), (H2, W2))`` sizes are also supported.
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        dropout=0.1,
        padding=5,
        downsample_mode="interp",
        activation_type="silu",
        interp_size=None,
        debug=False,
    ):
        super().__init__()
        if downsample_mode != "interp":
            raise ValueError("Only interpolation downsampling is supported.")
        self.downsample = Interp2dEncoder(
            in_dim=in_dim,
            out_dim=out_dim,
            interp_size=interp_size,
            activation_type=activation_type,
            dropout=dropout,
            debug=debug,
        )
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x, interp_size=None):
        if x.ndim != 4 or x.shape[-1] != self.in_dim:
            raise ValueError("DownScaler expects (B, H, W, in_dim).")
        return self.downsample(x.permute(0, 3, 1, 2), interp_size=interp_size).permute(
            0, 2, 3, 1
        )


class UpScaler(nn.Module):
    """Author-compatible interpolation decoder for channels-last 2D grids.

    A per-call ``interp_size`` overrides both interpolation stages. Alternatively
    ``output_shape`` overrides only the final output grid. Neither changes the
    constructor settings, which remain part of checkpoint metadata.
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_dim=None,
        padding=2,
        output_padding=0,
        dropout=0.1,
        upsample_mode="interp",
        activation_type="silu",
        interp_mode="bilinear",
        interp_size=None,
        debug=False,
    ):
        super().__init__()
        if upsample_mode != "interp":
            raise ValueError("Only interpolation upsampling is supported.")
        self.upsample = Interp2dUpsample(
            in_dim=in_dim,
            out_dim=out_dim,
            interp_mode=interp_mode,
            interp_size=interp_size,
            dropout=dropout,
            activation_type=activation_type,
            debug=debug,
        )
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x, interp_size=None, output_shape=None):
        if x.ndim != 4 or x.shape[-1] != self.in_dim:
            raise ValueError("UpScaler expects (B, H, W, in_dim).")
        if interp_size is not None and output_shape is not None:
            raise ValueError("Specify interp_size or output_shape, not both.")
        if output_shape is not None:
            interp_size = (self.upsample.interp_size[0], output_shape)
        return self.upsample(x.permute(0, 3, 1, 2), interp_size=interp_size).permute(
            0, 2, 3, 1
        )


class SpectralConv2d(nn.Module):
    """Two-band real-parameter spectral convolution used by the author model.

    The original real/imaginary contractions and ``fourier_weight`` state names
    are retained for pretrained compatibility. Rectangular grids and float64
    tensors use an FFT workspace matching the input precision. Modes are never
    silently truncated when a grid is too small.
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        modes,
        n_grid=None,
        dropout=0.1,
        norm="ortho",
        activation="silu",
        return_freq=False,
        debug=False,
    ):
        super().__init__()
        if isinstance(modes, bool) or not isinstance(modes, Integral) or modes < 1:
            raise ValueError("modes must be a positive integer.")
        if in_dim < 1 or out_dim < 1:
            raise ValueError("in_dim and out_dim must be positive.")
        if norm not in ("ortho", "forward", "backward"):
            raise ValueError("norm must be ortho, forward or backward.")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim)
        self.modes = modes
        self.activation = _activation(activation)
        self.n_grid = n_grid
        self.fourier_weight = nn.ParameterList(
            [
                nn.Parameter(torch.empty(in_dim, out_dim, modes, modes, 2))
                for _ in range(2)
            ]
        )
        for parameter in self.fourier_weight:
            nn.init.xavier_normal_(
                parameter, gain=1 / (in_dim * out_dim) * math.sqrt(in_dim + out_dim)
            )
        self.dropout = nn.Dropout(dropout)
        self.norm = norm
        self.return_freq = return_freq
        self.debug = debug

    @staticmethod
    def complex_matmul_2d(a, b):
        op = partial(torch.einsum, "bixy,ioxy->boxy")
        return torch.stack(
            [
                op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
                op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1]),
            ],
            dim=-1,
        )

    def forward(self, x):
        """Map (B,H,W,C) grids or flattened square (B,N,C) fields."""
        flattened = x.ndim == 3
        if flattened:
            size = math.isqrt(x.shape[1])
            if size * size != x.shape[1]:
                raise ValueError(
                    "Flattened spectral input must describe a square grid."
                )
            x = x.reshape(x.shape[0], size, size, x.shape[-1])
        if x.ndim != 4 or x.shape[-1] != self.in_dim:
            raise ValueError("SpectralConv2d expects (B,H,W,in_dim) or (B,N,in_dim).")
        batch, height, width, _ = x.shape
        modes = self.modes
        if height < 2 * modes or width // 2 + 1 < modes:
            raise ValueError(
                "Grid is too small for the configured spectral modes: require "
                "height >= 2*modes and width//2+1 >= modes."
            )
        residual = self.linear(x)
        x = self.dropout(x).permute(0, 3, 1, 2)
        spectrum = torch.fft.rfft2(x, s=(height, width), norm=self.norm)
        spectrum = torch.stack((spectrum.real, spectrum.imag), dim=-1)
        output_spectrum = spectrum.new_zeros(
            batch, self.out_dim, height, width // 2 + 1, 2
        )
        output_spectrum[:, :, :modes, :modes] = self.complex_matmul_2d(
            spectrum[:, :, :modes, :modes], self.fourier_weight[0]
        )
        output_spectrum[:, :, -modes:, :modes] = self.complex_matmul_2d(
            spectrum[:, :, -modes:, :modes], self.fourier_weight[1]
        )
        output_spectrum = torch.complex(
            output_spectrum[..., 0], output_spectrum[..., 1]
        )
        result = torch.fft.irfft2(output_spectrum, s=(height, width), norm=self.norm)
        result = self.activation(result.permute(0, 2, 3, 1) + residual)
        if flattened:
            result = result.reshape(batch, height * width, self.out_dim)
        if self.return_freq:
            return result, output_spectrum
        return result


class SpectralRegressor(nn.Module):
    """Author-compatible coordinate lifting and 2D spectral output decoder."""

    def __init__(
        self,
        in_dim,
        n_hidden,
        freq_dim,
        out_dim,
        modes,
        num_spectral_layers=2,
        n_grid=None,
        dim_feedforward=None,
        spacial_fc=False,
        spacial_dim=2,
        return_freq=False,
        return_latent=False,
        normalizer=None,
        activation="silu",
        last_activation=True,
        dropout=0.1,
        debug=False,
    ):
        super().__init__()
        if spacial_dim != 2:
            raise ValueError(
                "This regressor implements only the static 2D architecture."
            )
        if num_spectral_layers < 1:
            raise ValueError("At least one spectral layer is required.")
        self.activation = _activation(activation)
        dropout = default(dropout, 0.1)
        self.spacial_fc = spacial_fc
        if spacial_fc:
            self.fc = nn.Linear(in_dim + spacial_dim, n_hidden)
        elif in_dim != n_hidden:
            raise ValueError("Without coordinate lifting, in_dim must equal n_hidden.")
        self.spectral_conv = nn.ModuleList()
        for index in range(num_spectral_layers):
            self.spectral_conv.append(
                SpectralConv2d(
                    in_dim=n_hidden if index == 0 else freq_dim,
                    out_dim=freq_dim,
                    n_grid=n_grid,
                    modes=modes,
                    dropout=dropout,
                    activation=activation,
                    return_freq=return_freq,
                    debug=debug,
                )
            )
        if not last_activation:
            self.spectral_conv[-1].activation = nn.Identity()
        self.n_grid = n_grid
        self.dim_feedforward = default(dim_feedforward, 2 * spacial_dim * freq_dim)
        self.regressor = nn.Sequential(
            nn.Linear(freq_dim, self.dim_feedforward),
            self.activation,
            nn.Linear(self.dim_feedforward, out_dim),
        )
        self.normalizer = normalizer
        self.return_freq = return_freq
        self.return_latent = return_latent
        self.debug = debug

    def forward(self, x, edge=None, pos=None, grid=None):
        latents, spectra = [], []
        if self.spacial_fc:
            if grid is None or grid.shape[:-1] != x.shape[:-1] or grid.shape[-1] != 2:
                raise ValueError(
                    "Coordinate lifting requires a matching (B,H,W,2) grid."
                )
            x = self.fc(torch.cat((x, grid), dim=-1))
        for layer in self.spectral_conv:
            if self.return_freq:
                x, spectrum = layer(x)
                spectra.append(spectrum.contiguous())
            else:
                x = layer(x)
            if self.return_latent:
                latents.append(x.contiguous())
        x = self.regressor(x)
        if self.normalizer is not None:
            x = self.normalizer.inverse_transform(x)
        if self.return_freq or self.return_latent:
            return x, {"preds_freq": spectra, "preds_latent": latents}
        return x
