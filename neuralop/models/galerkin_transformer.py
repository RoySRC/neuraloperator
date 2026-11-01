"""Static 2D Galerkin Transformer with the author's Darcy parameter layout."""

import copy
import math
from collections.abc import Mapping
from numbers import Integral, Real

import torch
from torch import nn

from .base_model import BaseModel
from ..layers.galerkin_attention import GalerkinTransformerEncoderLayer
from ..layers.galerkin_spatial import DownScaler, UpScaler, SpectralRegressor


class GalerkinTransformer(BaseModel):
    """Two-dimensional Galerkin Transformer with a spectral decoder [1]_.

    Convolution and interpolation reduce a regular input grid, Galerkin linear
    attention mixes the latent samples, and a convolution/interpolation
    upscaler and Fourier decoder produce the output grid. This is the static
    2D architecture used by the authors for Darcy flow, with state-dictionary
    names compatible with their released checkpoint. Graph feature extractors,
    1D models and temporal rollout variants are outside this implementation.

    Parameters
    ----------
    in_channels, out_channels : int
        Number of input and output field channels.
    hidden_channels : int, optional
        Latent width, at least three and divisible by ``n_heads``, by default 128.
    n_heads, n_layers : int, optional
        Number of attention heads and encoder layers, by default 4 and 6.
    dim_feedforward : int, optional
        Encoder feedforward width, by default twice ``hidden_channels``.
    n_modes : int, optional
        Retained Fourier modes in each signed first-axis band and in the
        nonnegative second-axis band, by default 12. Output grids require at
        least ``2 * n_modes`` rows and ``n_modes`` real-FFT columns.
    spectral_channels, n_spectral_layers : int, optional
        Spectral decoder width and layer count, by default 32 and 2.
    downsample_factors : pair of float, optional
        Two successive bilinear downsampling factors in (0, 1], by default
        (0.54, 0.54), which map 211 to 113 to 61 points per axis. Upsampling uses
        an intermediate grid of floor(output size * first factor), followed by
        the requested output grid. Rectangular grids follow this rule per axis.
    layer_norm, attention_norm : bool, optional
        Outer encoder LayerNorm and per-head K/V LayerNorm, by default False
        and True. At least one must be enabled.
    norm_eps : float, optional
        LayerNorm epsilon, by default 1e-5, matching the released checkpoint.
    dropout, encoder_dropout, ffn_dropout : float, optional
        Overall, encoder, and feedforward dropout probabilities, by default
        0.0, 0.1, and 0.05. All dropout respects ``train()`` / ``eval()``.
    downscaler_dropout, upscaler_dropout, decoder_dropout : float, optional
        Spatial encoder, upscaler, and spectral dropout probabilities, by
        default 0.05, 0.0, and 0.0.

    Notes
    -----
    Coordinates use the author's endpoint-inclusive unit-square ``xy``
    convention: coordinate 0 varies across width, coordinate 1 across height.
    Explicit attention positions and decoder grids may override this default.

    Outputs are raw model fields. Physical input/output normalization and
    boundary conditions are external. To reproduce the Darcy pipeline, apply
    the original output inverse normalization *before* setting Dirichlet
    boundary values. The old author attention applies functional dropout even
    in evaluation; this implementation disables it in ``eval()``, matching the
    deterministic frozen-teacher policy used in the comparison.

    The model is randomly initialized unless weights are explicitly loaded.
    Supporting a different grid size does not establish pretrained accuracy
    at that resolution.

    References
    ----------
    .. [1] Cao, S. "Choose a Transformer: Fourier or Galerkin." NeurIPS (2021).
       https://arxiv.org/abs/2105.14995
       Author implementation: https://github.com/scaomath/galerkin-transformer
    """

    _version = "0.1.0"

    def __init__(
        self,
        *,
        in_channels,
        out_channels,
        hidden_channels=128,
        n_heads=4,
        n_layers=6,
        dim_feedforward=None,
        n_modes=12,
        spectral_channels=32,
        n_spectral_layers=2,
        downsample_factors=(0.54, 0.54),
        layer_norm=False,
        attention_norm=True,
        norm_eps=1e-5,
        dropout=0.0,
        encoder_dropout=0.1,
        ffn_dropout=0.05,
        downscaler_dropout=0.05,
        upscaler_dropout=0.0,
        decoder_dropout=0.0,
    ):
        super().__init__()
        for name, value in (
            ("in_channels", in_channels),
            ("out_channels", out_channels),
            ("hidden_channels", hidden_channels),
            ("n_heads", n_heads),
            ("n_layers", n_layers),
            ("n_modes", n_modes),
            ("spectral_channels", spectral_channels),
            ("n_spectral_layers", n_spectral_layers),
        ):
            self._positive_int(name, value)
        if dim_feedforward is None:
            dim_feedforward = 2 * hidden_channels
        self._positive_int("dim_feedforward", dim_feedforward)
        if hidden_channels < 3 or hidden_channels % n_heads:
            raise ValueError(
                "hidden_channels must be at least 3 and divisible by n_heads."
            )
        if not isinstance(layer_norm, bool) or not isinstance(attention_norm, bool):
            raise ValueError("layer_norm and attention_norm must be bools.")
        if not layer_norm and not attention_norm:
            raise ValueError(
                "At least one of layer_norm or attention_norm must be enabled."
            )
        if not self._finite_real(norm_eps) or norm_eps <= 0:
            raise ValueError("norm_eps must be finite and positive.")
        if (
            not isinstance(downsample_factors, (tuple, list))
            or len(downsample_factors) != 2
            or any(
                not self._finite_real(v) or not 0 < v <= 1 for v in downsample_factors
            )
        ):
            raise ValueError(
                "downsample_factors must contain two finite factors in (0, 1]."
            )
        for name, value in (
            ("dropout", dropout),
            ("encoder_dropout", encoder_dropout),
            ("ffn_dropout", ffn_dropout),
            ("downscaler_dropout", downscaler_dropout),
            ("upscaler_dropout", upscaler_dropout),
            ("decoder_dropout", decoder_dropout),
        ):
            if not self._finite_real(value) or not 0 <= value < 1:
                raise ValueError(f"{name} must be in [0, 1).")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.n_modes = n_modes
        self.downsample_factors = tuple(float(v) for v in downsample_factors)

        self.downscaler = DownScaler(
            in_dim=in_channels,
            out_dim=hidden_channels,
            downsample_mode="interp",
            interp_size=self.downsample_factors,
            dropout=downscaler_dropout,
            activation_type="relu",
        )
        self.feat_extract = nn.Identity()
        self.dpo = nn.Dropout(dropout)
        encoder = GalerkinTransformerEncoderLayer(
            d_model=hidden_channels,
            pos_dim=2,
            n_head=n_heads,
            attention_type="galerkin",
            dim_feedforward=dim_feedforward,
            layer_norm=layer_norm,
            attn_norm=attention_norm,
            norm_type="layer",
            norm_eps=norm_eps,
            batch_norm=False,
            attn_weight=False,
            xavier_init=0.01,
            diagonal_weight=0.01,
            symmetric_init=False,
            residual_type="add",
            activation_type="relu",
            dropout=encoder_dropout,
            ffn_dropout=ffn_dropout,
        )
        self.encoder_layers = nn.ModuleList(
            [copy.deepcopy(encoder) for _ in range(n_layers)]
        )
        # Runtime sizes are supplied on each forward; no mutable grid-size cache.
        self.upscaler = UpScaler(
            in_dim=hidden_channels,
            out_dim=hidden_channels,
            upsample_mode="interp",
            interp_size=((1, 1), (1, 1)),
            dropout=upscaler_dropout,
            activation_type="silu",
        )
        self.regressor = SpectralRegressor(
            in_dim=hidden_channels,
            n_hidden=spectral_channels,
            freq_dim=spectral_channels,
            out_dim=out_channels,
            modes=n_modes,
            num_spectral_layers=n_spectral_layers,
            spacial_dim=2,
            spacial_fc=True,
            activation="silu",
            last_activation=True,
            dropout=decoder_dropout,
            return_latent=False,
            return_freq=False,
        )

    @staticmethod
    def _positive_int(name, value):
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")

    @staticmethod
    def _finite_real(value):
        return (
            not isinstance(value, bool)
            and isinstance(value, Real)
            and math.isfinite(value)
        )

    @classmethod
    def _shape(cls, shape, name):
        if not isinstance(shape, (tuple, list, torch.Size)) or len(shape) != 2:
            raise ValueError(f"{name} must contain two positive integers.")
        for value in shape:
            cls._positive_int(name, value)
        return tuple(shape)

    @staticmethod
    def _grid(shape, x):
        height, width = shape
        y = torch.linspace(0, 1, height, dtype=torch.float64, device="cpu").to(x)
        x_axis = torch.linspace(0, 1, width, dtype=torch.float64, device="cpu").to(x)
        yy, xx = torch.meshgrid(y, x_axis, indexing="ij")
        return torch.stack((xx, yy), dim=-1)

    @staticmethod
    def _batch_coordinates(value, x, shape, name):
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            raise ValueError(f"{name} must be a floating-point tensor.")
        if value.ndim == len(shape):
            value = value.unsqueeze(0)
        if (
            value.ndim != len(shape) + 1
            or tuple(value.shape[1:]) != tuple(shape)
            or value.shape[0] not in (1, x.shape[0])
        ):
            raise ValueError(
                f"{name} must have shape {shape}, (1, {shape}) or (B, {shape})."
            )
        return value.to(x).expand(x.shape[0], *shape)

    def forward(self, x, pos=None, grid=None, output_shape=None, weight=None, **kwargs):
        """Evaluate a regular 2D field, optionally on a different output grid.

        Parameters
        ----------
        x : torch.Tensor
            Input fields ``(B, in_channels, H, W)``.
        pos : torch.Tensor, optional
            Latent attention coordinates ``(N_latent, 2)``, ``(1, N_latent, 2)``
            or ``(B, N_latent, 2)``. Their number must match the two downsampling
            stages. Defaults to an endpoint-inclusive ``xy`` unit-square grid.
        grid : torch.Tensor, optional
            Spectral-decoder coordinates ``(H_out, W_out, 2)``,
            ``(1, H_out, W_out, 2)`` or ``(B, H_out, W_out, 2)``. Its spatial
            shape determines the output shape unless ``output_shape`` is also
            supplied, in which case they must agree. Converted to x's dtype/device.
        output_shape : pair of int, optional
            Output grid shape, by default the input spatial shape. Generated
            coordinates cover the endpoint-inclusive unit square.
        weight : torch.Tensor, optional
            Author-style multiplicative Q/K weights ``(N_latent, 1)``,
            ``(1, N_latent, 1)`` or ``(B, N_latent, 1)``. These are not a
            general mass-matrix quadrature interface.
        **kwargs
            Additional sample metadata, such as Trainer's ``y``, is ignored.

        Returns
        -------
        torch.Tensor
            Raw output fields ``(B, out_channels, H_out, W_out)``. Apply any
            physical output normalization and boundary enforcement externally.
        """
        if not isinstance(x, torch.Tensor) or not x.is_floating_point():
            raise ValueError("x must be a floating-point tensor.")
        if x.ndim != 4 or any(v == 0 for v in x.shape):
            raise ValueError("x must have nonempty shape (B, C, H, W).")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels.")
        target = (
            tuple(x.shape[-2:])
            if output_shape is None
            else self._shape(output_shape, "output_shape")
        )
        if grid is not None:
            if (
                not isinstance(grid, torch.Tensor)
                or grid.ndim not in (3, 4)
                or grid.shape[-1] != 2
            ):
                raise ValueError(
                    "grid must have shape (H, W, 2), (1, H, W, 2) or (B, H, W, 2)."
                )
            grid_shape = self._shape(tuple(grid.shape[-3:-1]), "grid shape")
            if output_shape is not None and grid_shape != target:
                raise ValueError("grid spatial shape does not match output_shape.")
            target = grid_shape
        if target[0] < 2 * self.n_modes or target[1] // 2 + 1 < self.n_modes:
            raise ValueError(
                "Output grid is too small for n_modes; modes are not silently truncated."
            )
        up_mid = tuple(math.floor(v * self.downsample_factors[0]) for v in target)
        latent_shape = tuple(x.shape[-2:])
        for factor in self.downsample_factors:
            latent_shape = tuple(math.floor(v * factor) for v in latent_shape)
        if min(*latent_shape, *up_mid) < 1:
            raise ValueError("Grid is too small for the downsample_factors.")
        n_latent = math.prod(latent_shape)
        if pos is None:
            pos = self._grid(latent_shape, x).reshape(n_latent, 2)
        pos = self._batch_coordinates(pos, x, (n_latent, 2), "pos")
        if grid is None:
            grid = self._grid(target, x)
        grid = self._batch_coordinates(grid, x, (*target, 2), "grid")
        if weight is not None:
            weight = self._batch_coordinates(weight, x, (n_latent, 1), "weight")

        z = self.downscaler(x.permute(0, 2, 3, 1))
        if tuple(z.shape[1:3]) != latent_shape:
            raise RuntimeError("Unexpected spatial encoder shape.")
        z = self.dpo(
            self.feat_extract(z.reshape(x.shape[0], n_latent, self.hidden_channels))
        )
        for encoder in self.encoder_layers:
            z = encoder(z, pos=pos, weight=weight)
        z = z.reshape(x.shape[0], *latent_shape, self.hidden_channels)
        z = self.upscaler(z, interp_size=(up_mid, target))
        z = self.regressor(self.dpo(z), grid=grid)
        return z.permute(0, 3, 1, 2).contiguous()

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Retain saved convolution memory layouts, including nested loads.

        The author's checkpoint has a channels-last first convolution, including
        a nonstandard stride on its singleton input-channel axis. Copying only
        its values into a default contiguous weight selects different CUDA
        kernels and changes floating-point results. Restore supported dense
        convolution strides before the normal copy, retaining each Parameter's
        identity, device and dtype. PyTorch's assign mode retains its own
        assignment semantics instead.
        """
        if not local_metadata.get("assign_to_params_buffers", False):
            with torch.no_grad():
                for name, module in self.named_modules():
                    if not isinstance(module, nn.Conv2d):
                        continue
                    source = state_dict.get(f"{prefix}{name}.weight")
                    target = module.weight
                    if (
                        not isinstance(source, torch.Tensor)
                        or source.shape != target.shape
                        or source.is_meta
                        or source.layout != torch.strided
                        or not source.is_floating_point()
                        or source.stride() == target.stride()
                    ):
                        continue
                    if not (
                        source.is_contiguous()
                        or source.is_contiguous(memory_format=torch.channels_last)
                    ):
                        continue
                    restored = torch.empty_strided(
                        target.shape,
                        source.stride(),
                        dtype=target.dtype,
                        device=target.device,
                    )
                    restored.copy_(target)
                    target.set_(restored)
        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def load_author_state_dict(self, state_dict):
        """Strictly load tensor weights from the author's static 2D model.

        The mapping contains the original ``downscaler``, ``encoder_layers``,
        ``upscaler`` and ``regressor`` keys. Shapes and dense, real tensor
        formats are validated before loading. This method performs no download
        or unpickling. Constructor settings, including interpolation factors
        and normalization epsilon, must match the checkpoint; tensor shapes
        cannot verify those non-tensor settings. Saved contiguous/channels-last
        convolution memory layouts are retained. Physical normalization and
        boundary values are not part of this mapping.
        """
        expected = nn.Module.state_dict(self)
        if not isinstance(state_dict, Mapping) or set(state_dict) != set(expected):
            raise ValueError("Author state dictionary keys do not match the model.")
        for key, value in state_dict.items():
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != expected[key].shape
            ):
                raise ValueError(f"{key} has an incompatible tensor shape.")
            if (
                value.layout != torch.strided
                or value.is_meta
                or not value.is_floating_point()
            ):
                raise ValueError(f"{key} must be a real, dense floating-point tensor.")
        return self.load_state_dict(dict(state_dict), strict=True)
