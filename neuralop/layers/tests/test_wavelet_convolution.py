"""Wavelet reconstruction, precision and coefficient-grid contracts."""

import pytest
import torch

pytest.importorskip("pywt", reason="Optional wavelet dependencies are unavailable")
pytest.importorskip(
    "pytorch_wavelets", reason="Optional wavelet dependencies are unavailable"
)

from neuralop.layers.wavelet_convolution import (
    WaveletConv2d,
    wavelet_coefficient_shape,
)


@pytest.mark.parametrize(
    "mode", ["zero", "symmetric", "reflect", "periodic", "periodization"]
)
@pytest.mark.parametrize("wavelet", ["db2", "db4"])
@pytest.mark.parametrize("shape", [(32, 40), (31, 39)])
def test_unit_coefficient_maps_reconstruct_input(mode, wavelet, shape):
    """Unit maps plus untouched detail bands must realize the identity operator."""
    modes = wavelet_coefficient_shape(shape, levels=2, wavelet=wavelet, mode=mode)
    layer = WaveletConv2d(1, modes, levels=2, wavelet=wavelet, mode=mode)
    with torch.no_grad():
        layer.weights1.fill_(1)
        layer.weights2.fill_(1)
    generator = torch.Generator().manual_seed(20261101)
    x = torch.randn(2, 1, *shape, generator=generator)
    with torch.no_grad():
        reconstructed = layer(x)
    # For odd shapes, inverse-transform extension must not leak into the output.
    assert reconstructed.shape == x.shape
    torch.testing.assert_close(reconstructed, x, rtol=3e-6, atol=1e-6)


def test_double_precision_after_forward_and_filter_serialization():
    shape = (32, 40)
    modes = wavelet_coefficient_shape(shape, levels=2, wavelet="db2")
    layer = WaveletConv2d(2, modes, levels=2, wavelet="db2")
    with torch.no_grad():
        layer(torch.randn(1, 2, *shape))
    layer.double()
    buffers = list(layer.named_buffers())
    assert buffers, "Wavelet filters should follow module device/dtype changes."
    assert all(buffer.dtype == torch.float64 for _, buffer in buffers)
    # Fixed filters are architecture data and must not alter author weight keys.
    assert set(layer.state_dict()) == {"weights1", "weights2"}

    x = torch.randn(2, 2, *shape, dtype=torch.float64, requires_grad=True)
    output = layer(x)
    assert output.dtype == torch.float64
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0
    for name, parameter in layer.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


def test_incompatible_coefficient_grid_is_rejected():
    shape = (32, 40)
    modes = wavelet_coefficient_shape(shape, levels=2, wavelet="db2")
    layer = WaveletConv2d(1, (modes[0] + 1, modes[1]), levels=2, wavelet="db2")
    with pytest.raises(ValueError, match="no resampling"):
        layer(torch.randn(1, 1, *shape))
