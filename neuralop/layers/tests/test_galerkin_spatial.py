"""Numerical contracts for the static Galerkin spatial/spectral components."""

import pytest
import torch

from neuralop.layers.galerkin_spatial import DownScaler, SpectralConv2d, UpScaler


def test_historical_downsampling_floors_each_stage():
    model = DownScaler(1, 6, interp_size=(0.54, 0.54), dropout=0).eval()
    x = torch.randn(1, 211, 211, 1)
    with torch.no_grad():
        output = model(x)
    assert output.shape == (1, 61, 61, 6)


def test_rectangular_scaler_override_does_not_mutate_settings():
    down = DownScaler(2, 6, interp_size=(0.54, 0.54), dropout=0).eval()
    up = UpScaler(6, 3, interp_size=((8, 8), (16, 16)), dropout=0).eval()
    x = torch.randn(2, 17, 21, 2, requires_grad=True)
    latent = down(x)
    assert latent.shape == (2, 4, 5, 6)
    y = up(latent, interp_size=((9, 13), (17, 25)))
    assert y.shape == (2, 17, 25, 3)
    y[0].square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert x.grad[0].abs().sum() > 0
    assert torch.count_nonzero(x.grad[1]) == 0
    assert up.upsample.interp_size == ((8, 8), (16, 16))
    with torch.no_grad():
        original = up(latent.detach())
    assert original.shape == (2, 16, 16, 3)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_spectral_layer_matches_complex_fourier_reference(dtype):
    model = SpectralConv2d(2, 3, modes=2, dropout=0).to(dtype).eval()
    x = torch.randn(2, 7, 10, 2, dtype=dtype, requires_grad=True)
    spectrum = torch.fft.rfft2(x.permute(0, 3, 1, 2), norm="ortho")
    # An independent complex contraction checks the saved real/imaginary layout.
    expected_spectrum = spectrum.new_zeros(2, 3, 7, 6)
    for band, parameter in zip((slice(0, 2), slice(-2, None)), model.fourier_weight):
        expected_spectrum[:, :, band, :2] = torch.einsum(
            "bixy,ioxy->boxy",
            spectrum[:, :, band, :2],
            torch.view_as_complex(parameter),
        )
    spatial = torch.fft.irfft2(expected_spectrum, s=(7, 10), norm="ortho")
    expected = torch.nn.functional.silu(spatial.permute(0, 2, 3, 1) + model.linear(x))
    actual = model(x)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


def test_spectral_flattened_square_equivalence():
    model = SpectralConv2d(2, 3, modes=2, dropout=0).eval()
    x = torch.randn(2, 8, 8, 2)
    with torch.no_grad():
        expected = model(x)
        actual = model(x.reshape(2, 64, 2))
    torch.testing.assert_close(actual.reshape_as(expected), expected)


@pytest.mark.parametrize("shape", [(1, 3, 8, 2), (1, 8, 2, 2), (1, 17, 2)])
def test_spectral_rejects_unsupported_grid(shape):
    model = SpectralConv2d(2, 3, modes=3)
    with pytest.raises(ValueError):
        model(torch.randn(*shape))


@pytest.mark.parametrize("sizes", [None, (0.0, 0.5), (0.5, -1.0), ((2, 0), (4, 4))])
def test_invalid_interpolation_spec(sizes):
    with pytest.raises(ValueError):
        DownScaler(1, 6, interp_size=sizes)
