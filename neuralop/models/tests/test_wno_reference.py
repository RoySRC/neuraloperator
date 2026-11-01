"""Regression against independent original-author WNO Version 1 outputs.

The fixture was generated with the author's hash-pinned ``WaveConv2d`` and
``WNO2d`` classes at revision 8133e13f36d66f04895571d1c5507b31972805cb. It contains
small random synthetic models, not the trained Darcy checkpoint. The cases
exercise rectangular inputs and both equal and unequal coarse wavelet mode
dimensions. Reproduction instructions and source hashes are in
``data/generate_wno_reference.py`` and the fixture metadata.
"""

from pathlib import Path

import pytest
import torch

pytest.importorskip("pytorch_wavelets")

from neuralop.models import WNO


@pytest.mark.parametrize("case_index", [0, 1], ids=["equal_modes", "unequal_modes"])
def test_matches_independent_author_reference(case_index):
    fixture = torch.load(
        Path(__file__).with_name("data") / "wno_reference.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert fixture["source_revision"] == "8133e13f36d66f04895571d1c5507b31972805cb"
    assert fixture["source_sha256"] == (
        "9bbff5fdf67026d3fa30349bc31b2f6d908206b9fa103ad372978df3bc01fbc9"
    )
    case = fixture["cases"][case_index]
    model = WNO(**case["model_kwargs"]).eval()
    model.load_author_state_dict(case["state_dict"])
    x = case["x"].clone().requires_grad_(True)
    output = model(x)
    gradient = torch.autograd.grad(output.square().sum(), x)[0]
    torch.testing.assert_close(output, case["output"], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(gradient, case["input_gradient"], rtol=1e-5, atol=1e-6)
    # Check the wavelet transform before pointwise/residual mixing can mask it.
    with torch.no_grad():
        layer_output = model.conv0(case["layer_input"])
    torch.testing.assert_close(layer_output, case["layer_output"], rtol=1e-5, atol=1e-6)
