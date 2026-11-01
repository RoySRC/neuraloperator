"""Two-dimensional wavelet integral layers for the Version 1 WNO architecture.

This implementation expresses the coefficient-domain operator described by
Tripura and Chakraborty, "Wavelet neural operator for solving parametric partial
differential equations in computational mechanics problems" (2023),
https://doi.org/10.1016/j.cma.2022.115783.

The compatibility target is the static Darcy model in TapasTripura/WNO,
revision 8133e13f36d66f04895571d1c5507b31972805cb, Version 1.0.0. Its two learned
coefficient fields act on the approximation band and orientation zero of the
coarsest detail band. The other orientations and finer details pass through.
This is not the later WNO variant that learns all three detail orientations.
"""

from numbers import Integral

import torch
from torch import nn

_BOUNDARY_MODES = {
    "zero",
    "symmetric",
    "reflect",
    "periodization",
    "periodic",
}


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _shape_pair(value, name):
    if not isinstance(value, (tuple, list, torch.Size)) or len(value) != 2:
        raise ValueError(f"{name} must contain two positive integers.")
    return tuple(_positive_integer(v, name) for v in value)


def _wavelet_spec(levels, wavelet, mode):
    levels = _positive_integer(levels, "levels")
    if not isinstance(wavelet, str) or not wavelet:
        raise ValueError("wavelet must be a nonempty PyWavelets wavelet name.")
    if not isinstance(mode, str) or mode not in _BOUNDARY_MODES:
        raise ValueError(f"mode must be one of {sorted(_BOUNDARY_MODES)}.")
    try:
        import pywt
    except ImportError as exc:
        raise ImportError(
            "WNO requires the optional wavelet dependencies. "
            "Install them with `pip install 'neuraloperator[wavelets]'`."
        ) from exc
    return levels, pywt.Wavelet(wavelet)


def wavelet_coefficient_shape(input_shape, levels=4, wavelet="db4", mode="symmetric"):
    """Return the coarsest coefficient-grid shape without a tensor allocation.

    ``input_shape`` is the spatial shape entering the wavelet transform, after
    any model-level padding. Both axes undergo ``levels`` one-dimensional
    transforms with the specified wavelet and boundary extension.
    """
    spatial_shape = _shape_pair(input_shape, "input_shape")
    levels, specification = _wavelet_spec(levels, wavelet, mode)
    import pywt

    coefficient_shape = list(spatial_shape)
    for _ in range(levels):
        coefficient_shape = [
            pywt.dwt_coeff_len(size, specification.dec_len, mode)
            for size in coefficient_shape
        ]
    return tuple(coefficient_shape)


class WaveletConv2d(nn.Module):
    """Learn two coarsest-scale coefficient maps and reconstruct a 2D field.

    Parameters
    ----------
    channels : int
        Number of input and output channels. They must agree because the
        unmodified wavelet-detail bands bypass the learned maps.
    n_modes : tuple of two ints
        Exact coarsest coefficient-grid shape. This is not a truncation size;
        inputs yielding a different coefficient shape are rejected.
    levels : int, optional
        Number of discrete-wavelet decomposition levels, by default 4.
    wavelet : str, optional
        PyWavelets wavelet name, by default ``"db4"``.
    mode : str, optional
        Boundary extension used by both transforms, by default ``"symmetric"``.

    Notes
    -----
    The Version 1 Darcy checkpoint uses four db4 decomposition levels with
    symmetric extension. ``weights1`` acts on the approximation coefficients;
    ``weights2`` acts only on detail orientation zero at the coarsest level.
    Every other detail coefficient is preserved. Inverse-transform extension
    for an odd-sized input is cropped back to that input's original shape.

    Wavelet filters follow module device/dtype changes as nonpersistent buffers.
    Only the learned ``weights1`` and ``weights2`` appear in the state dictionary,
    matching the original architecture. Wavelet name, levels, boundary mode and
    intended physical grid remain architecture metadata, not learned weights.
    """

    def __init__(self, channels, n_modes, levels=4, wavelet="db4", mode="symmetric"):
        super().__init__()
        self.channels = _positive_integer(channels, "channels")
        self.n_modes = _shape_pair(n_modes, "n_modes")
        self.levels, _ = _wavelet_spec(levels, wavelet, mode)
        self.wavelet = wavelet
        self.mode = mode
        try:
            from pytorch_wavelets import DWTForward, DWTInverse
        except ImportError as exc:
            raise ImportError(
                "WNO requires the optional wavelet dependencies. "
                "Install them with `pip install 'neuraloperator[wavelets]'`."
            ) from exc

        shape = (self.channels, self.channels, *self.n_modes)
        # Match the Version 1 initialization distribution: U[0, 1) / channels^2.
        scale = 1.0 / self.channels**2
        self.weights1 = nn.Parameter(torch.rand(shape) * scale)
        self.weights2 = nn.Parameter(torch.rand(shape) * scale)
        self.analysis = DWTForward(J=self.levels, wave=self.wavelet, mode=self.mode)
        self.synthesis = DWTInverse(wave=self.wavelet, mode=self.mode)
        for transform in (self.analysis, self.synthesis):
            for module in transform.modules():
                module._non_persistent_buffers_set.update(module._buffers)

    def forward(self, x):
        """Map ``(batch, channels, height, width)`` to the same layout and shape."""
        if not isinstance(x, torch.Tensor) or not x.is_floating_point():
            raise ValueError("x must be a real floating-point tensor.")
        if x.ndim != 4 or any(size == 0 for size in x.shape):
            raise ValueError(
                "x must have nonempty shape (batch, channels, height, width)."
            )
        if x.shape[1] != self.channels:
            raise ValueError(f"Expected {self.channels} input channels.")
        spatial_shape = tuple(x.shape[-2:])
        actual_modes = wavelet_coefficient_shape(
            spatial_shape, self.levels, self.wavelet, self.mode
        )
        if actual_modes != self.n_modes:
            raise ValueError(
                f"Input shape {spatial_shape} yields wavelet coefficients {actual_modes}, "
                f"but this layer requires {self.n_modes}; no resampling is performed."
            )

        approximation, detail_bands = self.analysis(x)
        approximation = torch.einsum("bihw,iohw->bohw", approximation, self.weights1)
        coarsest = detail_bands[-1]
        orientation_zero = torch.einsum(
            "bihw,iohw->bohw", coarsest[:, :, 0], self.weights2
        )
        # Rebuild the band instead of modifying the transform's output in place.
        transformed_coarsest = torch.cat(
            (orientation_zero.unsqueeze(2), coarsest[:, :, 1:]), dim=2
        )
        reconstructed = self.synthesis(
            (approximation, [*detail_bands[:-1], transformed_coarsest])
        )
        return reconstructed[..., : spatial_shape[0], : spatial_shape[1]]
