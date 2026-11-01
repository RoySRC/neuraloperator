"""Regression against independently generated, original-author OFormer outputs.

The fixture contains small random-weight models from BaratiLab/OFormer revision
f1111b3c634893329e14e47e8903e421dd726a0c, not trained publisher weights. Both
layer-normalization settings were evaluated by the original SpatialEncoder2D
and PointWiseDecoder2DSimple classes on synthetic data with 11 source points
and 7 query points. Source hashes and generation metadata are embedded in the
fixture; see data/generate_oformer_reference.py to reproduce it. Tests require
neither the upstream checkout nor its einops dependency.
"""

from pathlib import Path

import pytest
import torch

from neuralop.models import OFormer


@pytest.mark.parametrize(
    "case_index", [0, 1], ids=["without_layer_norm", "with_layer_norm"]
)
def test_matches_independent_author_reference(case_index):
    fixture = torch.load(
        Path(__file__).with_name("data") / "oformer_reference.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert fixture["source_revision"] == "f1111b3c634893329e14e47e8903e421dd726a0c"
    case = fixture["cases"][case_index]
    model = OFormer(**case["model_kwargs"]).eval()
    model.load_author_state_dict(case["encoder_state_dict"], case["decoder_state_dict"])
    x = case["x"].clone().requires_grad_(True)
    output = model(
        x,
        input_geom=case["input_geom"],
        output_queries=case["output_queries"],
    )
    gradient = torch.autograd.grad(output.square().sum(), x)[0]
    torch.testing.assert_close(output, case["output"], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(gradient, case["input_gradient"], rtol=1e-6, atol=1e-6)
    # Also expose encoder changes that a coincidental decoder cancellation hides.
    with torch.no_grad():
        encoded = model.encoder(
            torch.cat((case["x"], case["input_geom"]), dim=-1), case["input_geom"]
        )
    torch.testing.assert_close(encoded, case["encoder_output"], rtol=1e-6, atol=1e-6)
