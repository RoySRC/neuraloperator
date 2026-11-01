from collections.abc import Mapping
import math
from numbers import Integral, Real

import torch

from .base_model import BaseModel
from ..layers.oformer import OFormerDecoder, OFormerEncoder


class OFormer(BaseModel):
    """Static two-dimensional Operator Transformer (OFormer).

    A Galerkin linear-attention encoder embeds the input samples. A coordinate
    decoder uses Gaussian Fourier features, rotary cross-attention and a
    pointwise projection to evaluate the output at independently chosen points.
    This implements the authors' static 2D (Darcy) architecture [1]_, including
    its normalization and residual conventions. Temporal rollout variants are
    not included.

    Parameters
    ----------
    in_channels, out_channels : int
        Input and output field channels. Coordinates are appended internally
        and must not be counted in ``in_channels``.
    hidden_channels : int, optional
        Encoder width, by default 96. Must be a positive multiple of four.
        Each attention head has this full width, rather than width / heads.
    latent_channels : int, optional
        Decoder width, by default 256. Must be a positive multiple of four.
    n_layers : int, optional
        Number of encoder attention blocks, at least two, by default 6.
    n_heads : int, optional
        Number of heads in both encoder and decoder, by default 4.
    resolution : int, optional
        Fixed reference resolution for the rotary frequency scales, by default
        211. Must be at least four. It is an architecture parameter, not a
        restriction on the number of runtime input or query points. Preserve
        the training value when loading pretrained weights.
    coordinate_scale : float, optional
        Gaussian Fourier feature scale, by default 0.5.
    use_layer_norm : bool, optional
        Whether to apply the authors' encoder layer normalization, by default
        True. Decoder attention retains the authors' instance normalization.
    embedding_dropout, attention_dropout : float, optional
        Encoder embedding and attention-block dropout probabilities, by default
        0.05 and 0.03. Call ``eval()`` for deterministic inference.

    Notes
    -----
    Default grids include both endpoints of [0, 1] on each axis, matching the
    authors' Darcy convention. Explicit geometries may describe other domains,
    but their coordinate units must agree with the trained model.

    The architecture does not normalize physical fields or impose boundary
    values. The published Darcy checkpoint requires its original input/output
    normalization and zero-Dirichlet postprocessing outside this model. A model
    constructed here has random weights unless weights are explicitly loaded.

    References
    ----------
    .. [1] Li, Z., Meidani, K., and Farimani, A. B. "Transformer for Partial
       Differential Equations' Operator Learning." TMLR (2023).
       https://arxiv.org/abs/2205.13671
       Author implementation: https://github.com/BaratiLab/OFormer

    Examples
    --------
    >>> model = OFormer(in_channels=1, out_channels=1)
    >>> model.eval()
    >>> with torch.no_grad():
    ...     y = model(torch.randn(1, 1, 16, 16))
    >>> y.shape
    torch.Size([1, 1, 16, 16])
    """

    _version = "0.1.0"

    def __init__(
        self,
        *,
        in_channels,
        out_channels,
        hidden_channels=96,
        latent_channels=256,
        n_layers=6,
        n_heads=4,
        resolution=211,
        coordinate_scale=0.5,
        use_layer_norm=True,
        embedding_dropout=0.05,
        attention_dropout=0.03,
    ):
        super().__init__()
        for name, value in (
            ("in_channels", in_channels),
            ("out_channels", out_channels),
            ("hidden_channels", hidden_channels),
            ("latent_channels", latent_channels),
            ("n_layers", n_layers),
            ("n_heads", n_heads),
            ("resolution", resolution),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}.")
        if hidden_channels % 4 or latent_channels % 4:
            raise ValueError(
                "hidden_channels and latent_channels must be divisible by 4."
            )
        if n_layers < 2:
            raise ValueError("n_layers must be at least 2.")
        if resolution < 4:
            raise ValueError("resolution must be at least 4.")
        if not isinstance(use_layer_norm, bool):
            raise ValueError("use_layer_norm must be a bool.")
        if (
            isinstance(coordinate_scale, bool)
            or not isinstance(coordinate_scale, Real)
            or not math.isfinite(coordinate_scale)
            or coordinate_scale <= 0
        ):
            raise ValueError("coordinate_scale must be finite and positive.")
        for name, value in (
            ("embedding_dropout", embedding_dropout),
            ("attention_dropout", attention_dropout),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not 0 <= value < 1
            ):
                raise ValueError(f"{name} must be in [0, 1).")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.latent_channels = latent_channels
        self.resolution = resolution
        self.encoder = OFormerEncoder(
            input_channels=in_channels + 2,
            in_emb_dim=hidden_channels,
            out_seq_emb_dim=latent_channels,
            heads=n_heads,
            depth=n_layers,
            res=resolution,
            use_ln=use_layer_norm,
            emb_dropout=embedding_dropout,
            attention_dropout=attention_dropout,
        )
        self.decoder = OFormerDecoder(
            latent_channels=latent_channels,
            out_channels=out_channels,
            res=resolution,
            scale=coordinate_scale,
            heads=n_heads,
        )

    @staticmethod
    def _grid(shape, x):
        # The author code uses np.linspace in float64, followed by a dtype cast.
        # No coordinate cache: device/dtype changes cannot leave stale grids.
        axes = [
            torch.linspace(0, 1, size, device="cpu", dtype=torch.float64).to(
                device=x.device, dtype=x.dtype
            )
            for size in shape
        ]
        return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 2)

    @staticmethod
    def _coordinates(pos, x, name, n_points=None):
        if not isinstance(pos, torch.Tensor) or not pos.is_floating_point():
            raise ValueError(f"{name} must be a floating-point tensor.")
        if pos.ndim == 2:
            pos = pos.unsqueeze(0)
        if pos.ndim != 3 or pos.shape[-1] != 2:
            raise ValueError(f"{name} must have shape (N, 2), (1, N, 2) or (B, N, 2).")
        if pos.shape[0] not in (1, x.shape[0]) or pos.shape[1] < 1:
            raise ValueError(f"{name} has an invalid batch size or no points.")
        if n_points is not None and pos.shape[1] != n_points:
            raise ValueError(f"{name} must contain {n_points} input points.")
        return pos.to(device=x.device, dtype=x.dtype).expand(x.shape[0], -1, -1)

    def forward(
        self, x, input_geom=None, output_queries=None, output_shape=None, **kwargs
    ):
        """Evaluate an input field at the requested coordinates.

        Parameters
        ----------
        x : torch.Tensor
            Grid fields ``(B, in_channels, H, W)`` or point fields
            ``(B, N, in_channels)``. Must be a nonempty floating-point tensor.
        input_geom : torch.Tensor, optional
            Input coordinates ``(N, 2)``, ``(1, N, 2)`` or ``(B, N, 2)``.
            Required for point fields; for grids N = H * W in row-major order.
            If omitted for grids, use an endpoint-inclusive unit-square grid.
            Coordinates are converted to the input field's device and dtype.
        output_queries : torch.Tensor, optional
            Query coordinates ``(M, 2)``, ``(1, M, 2)`` or ``(B, M, 2)``.
            Returns point fields ``(B, M, out_channels)`` when provided.
        output_shape : tuple of two ints, optional
            Generate an endpoint-inclusive unit-square query grid with this
            shape and return ``(B, out_channels, H_out, W_out)``. Mutually
            exclusive with ``output_queries``.
        **kwargs
            Additional sample metadata (for example ``y`` from Trainer), ignored.

        Returns
        -------
        torch.Tensor
            Without explicit output queries or shape, returns values at the
            input coordinates and preserves the input's grid/point layout.
        """
        if not isinstance(x, torch.Tensor) or not x.is_floating_point():
            raise ValueError("x must be a floating-point tensor.")
        if x.ndim not in (3, 4) or any(size == 0 for size in x.shape):
            raise ValueError("x must have nonempty shape (B, C, H, W) or (B, N, C).")
        grid_shape = tuple(x.shape[-2:]) if x.ndim == 4 else None
        fields = x.flatten(2).transpose(1, 2) if grid_shape is not None else x
        if fields.shape[-1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels.")
        if input_geom is None:
            if grid_shape is None:
                raise ValueError("input_geom is required for point inputs.")
            input_geom = self._grid(grid_shape, x)
        pos = self._coordinates(input_geom, fields, "input_geom", fields.shape[1])

        if output_queries is not None and output_shape is not None:
            raise ValueError("output_queries and output_shape are mutually exclusive.")
        if output_shape is not None:
            if (
                not isinstance(output_shape, (tuple, list, torch.Size))
                or len(output_shape) != 2
                or any(
                    isinstance(v, bool) or not isinstance(v, Integral) or v < 1
                    for v in output_shape
                )
            ):
                raise ValueError("output_shape must contain two positive integers.")
            grid_shape = tuple(output_shape)
            query_pos = self._coordinates(
                self._grid(grid_shape, x), fields, "output_shape grid"
            )
        elif output_queries is not None:
            grid_shape = None
            query_pos = self._coordinates(output_queries, fields, "output_queries")
        else:
            query_pos = pos

        z = self.encoder(torch.cat((fields, pos), dim=-1), pos)
        output = self.decoder(z, query_pos, pos)
        if grid_shape is not None:
            return output.transpose(1, 2).reshape(
                x.shape[0], self.out_channels, *grid_shape
            )
        return output

    def load_author_state_dict(self, encoder_state_dict, decoder_state_dict):
        """Strictly load the original authors' separate encoder/decoder weights.

        Pass the tensor mappings saved under ``encoder`` and ``decoder`` in an
        author checkpoint. Both dictionaries' keys, shapes and tensor formats
        are validated before either component is updated. No file is downloaded
        or unpickled by this method.

        The constructor must match the checkpoint architecture. The released
        Darcy model uses the constructor defaults with one input and one output
        field channel. Coordinates are appended by this wrapper. Normalization
        statistics and boundary postprocessing remain the caller's responsibility.
        """
        components = (
            ("encoder", self.encoder, encoder_state_dict),
            ("decoder", self.decoder, decoder_state_dict),
        )
        for name, component, state in components:
            expected = component.state_dict()
            if not isinstance(state, Mapping) or set(state) != set(expected):
                raise ValueError(
                    f"{name} state dictionary keys do not match the model."
                )
            for key, value in state.items():
                if (
                    not isinstance(value, torch.Tensor)
                    or value.shape != expected[key].shape
                ):
                    raise ValueError(f"{name}.{key} has an incompatible tensor shape.")
                if (
                    value.layout != torch.strided
                    or value.is_meta
                    or not value.is_floating_point()
                ):
                    raise ValueError(
                        f"{name}.{key} must be a real, dense floating-point tensor."
                    )
        for _, component, state in components:
            component.load_state_dict(state, strict=True)
