"""Static two-dimensional Wavelet Neural Operator with a fixed sampling grid."""

from collections.abc import Mapping
from numbers import Integral

import torch
from torch import nn
from torch.nn import functional as F

from .base_model import BaseModel
from ..layers.wavelet_convolution import WaveletConv2d, wavelet_coefficient_shape


def _positive_pair(value, name):
    if (
        not isinstance(value, (tuple, list, torch.Size))
        or len(value) != 2
        or any(
            isinstance(v, bool) or not isinstance(v, Integral) or v < 1 for v in value
        )
    ):
        raise ValueError(f"{name} must contain two positive integers.")
    return tuple(int(v) for v in value)


class WNO(BaseModel):
    """Two-dimensional Wavelet Neural Operator, following the Version 1 design.

    Each block combines a wavelet transform operator with a pointwise linear
    map. The wavelet operator learns spatially indexed channel mixing on the
    coarsest lowpass coefficients and the first coarsest detail orientation;
    all other detail coefficients pass through unchanged. This is the static
    rectangular-domain formulation used in the authors' Darcy experiments [1]_,
    rather than their later variants that learn additional detail orientations.

    Parameters
    ----------
    in_channels, out_channels : int
        Input and output field channels. Two coordinate channels are appended
        internally and are not included in ``in_channels``.
    hidden_channels : int, optional
        Width of the lifted representation, by default 64.
    n_layers : int, optional
        Number of wavelet-plus-pointwise blocks, by default 4. GELU follows all
        blocks except the last.
    input_shape : tuple of two ints, optional
        Required input grid shape, by default (85, 85). Learned wavelet weights
        are tied to its coefficient grid. Runtime resolution changes are
        rejected rather than interpolating fields or weights.
    n_modes : tuple of two ints or None, optional
        Coarsest wavelet coefficient shape. If None, derive it from the padded
        input shape and the wavelet settings. An explicit value must agree
        exactly with that shape. The default Darcy configuration gives (11, 11).
    levels : int, optional
        Number of discrete wavelet decomposition levels, by default 4.
    wavelet : str, optional
        Wavelet name understood by PyWavelets, by default "db4".
    mode : str, optional
        Wavelet boundary extension mode, by default "symmetric".
    padding : int, optional
        Zero padding on the right and bottom before the operator blocks,
        removed after the final block, by default 1.
    projection_channels : int, optional
        Width of the output projection, by default 192, matching the released
        Version 1 Darcy checkpoint. Some author source constructors use 128;
        set this argument to the width actually stored in the checkpoint.

    Notes
    -----
    Install the optional dependencies with ``pip install 'neuraloperator[wavelets]'``
    (or ``pip install -e '.[wavelets]'`` from a source checkout).

    This implementation evaluates static 2D fields, with channel-first input
    and output. It does not include the authors' 1D, time-dependent or Version 2
    architectures. Default coordinates include both endpoints of [0, 1] on each
    axis. Input/output normalization remains external to the architecture;
    pretrained use must retain the original training-only normalization and
    sampling grid. Constructing a model initializes random parameters.

    References
    ----------
    .. [1] Tripura, T. and Chakraborty, S. "Wavelet Neural Operator for solving
       parametric partial differential equations in computational mechanics
       problems." Computer Methods in Applied Mechanics and Engineering 404
       (2023), 115783. https://doi.org/10.1016/j.cma.2022.115783
       Reference implementation: https://github.com/TapasTripura/WNO

    Examples
    --------
    >>> model = WNO(in_channels=1, out_channels=1).eval()
    >>> with torch.no_grad():
    ...     y = model(torch.randn(1, 1, 85, 85))
    >>> y.shape
    torch.Size([1, 1, 85, 85])
    """

    _version = "0.1.0"

    def __init__(
        self,
        *,
        in_channels,
        out_channels,
        hidden_channels=64,
        n_layers=4,
        input_shape=(85, 85),
        n_modes=None,
        levels=4,
        wavelet="db4",
        mode="symmetric",
        padding=1,
        projection_channels=192,
    ):
        super().__init__()
        for name, value in (
            ("in_channels", in_channels),
            ("out_channels", out_channels),
            ("hidden_channels", hidden_channels),
            ("n_layers", n_layers),
            ("levels", levels),
            ("projection_channels", projection_channels),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if (
            isinstance(padding, bool)
            or not isinstance(padding, Integral)
            or padding < 0
        ):
            raise ValueError("padding must be a nonnegative integer.")
        self.input_shape = _positive_pair(input_shape, "input_shape")
        if not isinstance(wavelet, str) or not isinstance(mode, str):
            raise ValueError("wavelet and mode must be strings.")
        padded_shape = tuple(size + padding for size in self.input_shape)
        derived_modes = wavelet_coefficient_shape(padded_shape, levels, wavelet, mode)
        if n_modes is not None and _positive_pair(n_modes, "n_modes") != derived_modes:
            raise ValueError(
                f"n_modes must equal {derived_modes} for input_shape={self.input_shape}, "
                f"padding={padding} and the specified wavelet settings."
            )
        self.n_modes = derived_modes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.levels = levels
        self.wavelet = wavelet
        self.mode = mode
        self.padding = padding
        self.projection_channels = projection_channels

        # Preserve the established parameter names for explicit author-weight
        # interchange, while exposing the library's channel-first field API.
        self.fc0 = nn.Linear(in_channels + 2, hidden_channels)
        for index in range(n_layers):
            self.add_module(
                f"conv{index}",
                WaveletConv2d(hidden_channels, derived_modes, levels, wavelet, mode),
            )
        for index in range(n_layers):
            self.add_module(f"w{index}", nn.Conv2d(hidden_channels, hidden_channels, 1))
        self.fc1 = nn.Linear(hidden_channels, projection_channels)
        self.fc2 = nn.Linear(projection_channels, out_channels)

    @staticmethod
    def _grid(x):
        # Match the author's float64 linspace followed by a field-dtype cast.
        # Construct on CPU so float32 models also work on devices without fp64.
        axes = [
            torch.linspace(0, 1, size, device="cpu", dtype=torch.float64).to(
                device=x.device, dtype=x.dtype
            )
            for size in x.shape[-2:]
        ]
        return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).expand(
            x.shape[0], -1, -1, -1
        )

    def forward(self, x, **kwargs):
        """Map ``(B, in_channels, H, W)`` to ``(B, out_channels, H, W)``.

        H and W must match ``input_shape``. Extra sample metadata such as ``y``
        from Trainer is ignored. An ``output_shape`` request is rejected because
        this model does not implement output resampling.
        """
        if not isinstance(x, torch.Tensor) or not x.is_floating_point():
            raise ValueError("x must be a floating-point tensor.")
        if x.ndim != 4 or x.shape[0] < 1:
            raise ValueError("x must have nonempty shape (B, C, H, W).")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels.")
        if tuple(x.shape[-2:]) != self.input_shape:
            raise ValueError(
                f"Expected input_shape={self.input_shape}, got {tuple(x.shape[-2:])}. "
                "WNO does not silently interpolate to a different resolution."
            )
        if kwargs.get("output_shape") is not None:
            raise ValueError("WNO does not support output_shape resampling.")
        height, width = self.input_shape
        values = torch.cat((x.permute(0, 2, 3, 1), self._grid(x)), dim=-1)
        values = self.fc0(values).permute(0, 3, 1, 2)
        if self.padding:
            values = F.pad(values, (0, self.padding, 0, self.padding))
        for index in range(self.n_layers):
            values = getattr(self, f"conv{index}")(values) + getattr(self, f"w{index}")(
                values
            )
            if index < self.n_layers - 1:
                values = F.gelu(values)
        values = values[..., :height, :width].permute(0, 2, 3, 1)
        values = self.fc2(F.gelu(self.fc1(values)))
        return values.permute(0, 3, 1, 2)

    def load_author_state_dict(self, state_dict):
        """Strictly load a Version 1 author model's tensor state dictionary.

        Tensor keys, shapes and formats are checked before updating the model.
        The caller must also match non-tensor settings (wavelet, boundary mode,
        decomposition levels, padding and input grid). Tensor shapes alone do
        not establish that these settings agree. No checkpoint file is opened,
        unpickled or downloaded here; pass the extracted tensor mapping only.
        """
        expected = nn.Module.state_dict(self)
        if not isinstance(state_dict, Mapping) or set(state_dict) != set(expected):
            raise ValueError("Author state dictionary keys do not match the WNO model.")
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
        nn.Module.load_state_dict(self, dict(state_dict), strict=True)
