"""Generate an independent, synthetic Galerkin Transformer reference fixture.

Run with PyTorch and NumPy installed::

    python generate_galerkin_reference.py --upstream-root /path/to/galerkin-transformer

Only the pinned authors' class definitions are evaluated. This script does not
import NeuralOperator, execute a training program, load pretrained weights, or
read PDE data. The original attention's functional dropout is disabled in eval,
matching the established deterministic frozen-teacher policy.
"""

import argparse
import ast
from collections import defaultdict
import copy
import hashlib
import importlib.util
from pathlib import Path
import subprocess

import numpy as np
import torch

REVISION = "f9e7d6abc24e958b1165f48b22625c5bab688043"
SOURCE_URL = "https://github.com/scaomath/galerkin-transformer"
SOURCE_HASHES = {
    "libs/layers.py": "3c8176f3a72b1f2d2fd999fc8d0c123d2487e37cc5dad6f4701c21df7242d670",
    "libs/model.py": "f480498b028b128dcfc296405474390bddf13ac18e106bc240dbd70a156dceef",
}


def original_classes(upstream_root):
    revision = subprocess.check_output(
        ["git", "-C", str(upstream_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REVISION:
        raise RuntimeError(f"Expected author revision {REVISION}, got {revision}")
    for name, digest in SOURCE_HASHES.items():
        if hashlib.sha256((upstream_root / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Author source hash mismatch: {name}")
    spec = importlib.util.spec_from_file_location(
        "_galerkin_author_layers", upstream_root / "libs/layers.py"
    )
    layers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(layers)
    names = {
        "SimpleTransformerEncoderLayer",
        "SpectralRegressor",
        "DownScaler",
        "UpScaler",
        "FourierTransformer2D",
    }
    source = upstream_root / "libs/model.py"
    selected = []
    for node in ast.parse(source.read_text()).body:
        if isinstance(node, ast.ClassDef) and node.name in names:
            selected.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "ADDITIONAL_ATTR"
            for target in node.targets
        ):
            selected.append(node)
    if {node.name for node in selected if isinstance(node, ast.ClassDef)} != names:
        raise RuntimeError("Required author classes are missing.")
    namespace = dict(vars(layers), copy=copy, defaultdict=defaultdict)
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return namespace


def xy_grid(size):
    axis = np.linspace(0, 1, size)
    xx, yy = np.meshgrid(axis, axis)
    return torch.tensor(np.stack((xx, yy), axis=-1), dtype=torch.float32)


def generate(upstream_root, output):
    classes = original_classes(upstream_root)
    torch.set_num_threads(1)
    cases = []
    for index, (size, layer_norm) in enumerate(((16, False), (20, True))):
        seed = 20261101 + index
        torch.manual_seed(seed)
        midpoint = int(size * 0.54)
        latent_size = int(midpoint * 0.54)
        config = dict(
            node_feats=1,
            pos_dim=2,
            n_targets=2,
            n_hidden=8,
            num_feat_layers=0,
            num_encoder_layers=2,
            n_head=2,
            dim_feedforward=16,
            feat_extract_type=None,
            attention_type="galerkin",
            xavier_init=0.01,
            diagonal_weight=0.01,
            symmetric_init=False,
            layer_norm=layer_norm,
            attn_norm=True,
            norm_eps=1e-5,
            batch_norm=False,
            return_attn_weight=False,
            return_latent=False,
            decoder_type="ifft2",
            spacial_dim=2,
            spacial_fc=True,
            upsample_mode="interp",
            downsample_mode="interp",
            downscaler_size=(0.54, 0.54),
            upscaler_size=((midpoint, midpoint), (size, size)),
            freq_dim=4,
            boundary_condition="free",
            num_regressor_layers=2,
            fourier_modes=2,
            regressor_activation="silu",
            downscaler_activation="relu",
            upscaler_activation="silu",
            last_activation=True,
            dropout=0.0,
            downscaler_dropout=0.05,
            upscaler_dropout=0.0,
            ffn_dropout=0.05,
            encoder_dropout=0.1,
            decoder_dropout=0.0,
            normalizer=None,
            debug=False,
        )
        model = classes["FourierTransformer2D"](**config).eval()
        for layer in model.encoder_layers:
            layer.attn.dropout = None
        x = torch.randn(2, 1, size, size, requires_grad=True)
        pos = xy_grid(latent_size).reshape(1, -1, 2).expand(2, -1, -1)
        grid = xy_grid(size).unsqueeze(0).expand(2, -1, -1, -1)
        result = model(x.permute(0, 2, 3, 1), None, pos, grid)["preds"].permute(
            0, 3, 1, 2
        )
        gradient = torch.autograd.grad(result.square().sum(), x)[0]
        cases.append(
            {
                "seed": seed,
                "model_kwargs": dict(
                    in_channels=1,
                    out_channels=2,
                    hidden_channels=8,
                    n_heads=2,
                    n_layers=2,
                    dim_feedforward=16,
                    n_modes=2,
                    spectral_channels=4,
                    n_spectral_layers=2,
                    downsample_factors=(0.54, 0.54),
                    layer_norm=layer_norm,
                    attention_norm=True,
                    norm_eps=1e-5,
                    dropout=0.0,
                    encoder_dropout=0.1,
                    ffn_dropout=0.05,
                    downscaler_dropout=0.05,
                    upscaler_dropout=0.0,
                    decoder_dropout=0.0,
                ),
                "state_dict": {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                },
                "x": x.detach(),
                "pos": pos,
                "grid": grid,
                "output": result.detach(),
                "input_gradient": gradient.detach(),
            }
        )
    fixture = {
        "source_url": SOURCE_URL,
        "source_revision": REVISION,
        "source_sha256": SOURCE_HASHES,
        "torch_version": str(torch.__version__),
        "scope": "Synthetic random author weights and inputs; no pretrained weights or PDE data.",
        "attention_policy": "Legacy functional attention dropout disabled in eval.",
        "normalization": "None; raw network output; boundary_condition=free.",
        "coordinates": "Endpoint-inclusive numpy.meshgrid xy ordering.",
        "mode": "eval",
        "dtype": "float32",
        "gradient_loss": "output.square().sum()",
        "cases": cases,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fixture, output)
    print(
        f"Saved {len(cases)} author-reference cases to {output} ({output.stat().st_size} bytes)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_name("galerkin_reference.pt")
    )
    args = parser.parse_args()
    generate(args.upstream_root, args.output)
