"""Generate the independent, synthetic OFormer regression fixture.

Run with torch, numpy and einops installed::

    python generate_oformer_reference.py --upstream-root /path/to/OFormer

The source must be the unmodified BaratiLab/OFormer revision pinned below.
Only original class definitions are executed; training programs, real datasets
and trained checkpoints are never loaded. This generator does not import
NeuralOperator or the new OFormer implementation.
"""

import argparse
import ast
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import warnings

import torch

REVISION = "f1111b3c634893329e14e47e8903e421dd726a0c"
SOURCE_URL = "https://github.com/BaratiLab/OFormer"
SOURCE_FILES = (
    "uniform_grids/nn_module/attention_module.py",
    "uniform_grids/nn_module/encoder_module.py",
    "uniform_grids/nn_module/decoder_module.py",
)


def load_classes(path, names, namespace):
    """Evaluate only the requested, unmodified author class definitions."""
    tree = ast.parse(path.read_text())
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name in names
    ]
    if {node.name for node in selected} != set(names):
        raise RuntimeError(f"Missing author classes in {path}")
    namespace = dict(namespace)
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace


def tensor_state(module):
    return {
        key: value.detach().cpu().clone() for key, value in module.state_dict().items()
    }


def generate(upstream_root, output):
    revision = subprocess.check_output(
        ["git", "-C", str(upstream_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REVISION:
        raise RuntimeError(f"Expected upstream revision {REVISION}, got {revision}")
    changes = subprocess.check_output(
        ["git", "-C", str(upstream_root), "diff", "HEAD", "--", *SOURCE_FILES],
        text=True,
    )
    if changes:
        raise RuntimeError("The original author source files must be unmodified.")

    attention_path = upstream_root / SOURCE_FILES[0]
    spec = importlib.util.spec_from_file_location(
        "_oformer_author_attention", attention_path
    )
    attention = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(attention)
    encoder_namespace = load_classes(
        upstream_root / SOURCE_FILES[1],
        {"TransformerCatNoCls", "SpatialEncoder2D"},
        vars(attention),
    )
    decoder_namespace = load_classes(
        upstream_root / SOURCE_FILES[2],
        {"GaussianFourierFeatureTransform", "CrossFormer", "PointWiseDecoder2DSimple"},
        vars(attention),
    )
    torch.set_num_threads(1)
    cases = []
    for use_layer_norm in (False, True):
        seed = 20261101 + int(use_layer_norm)
        torch.manual_seed(seed)
        encoder = encoder_namespace["SpatialEncoder2D"](
            3, 8, 16, 4, 2, res=7, use_ln=use_layer_norm
        ).eval()
        decoder = decoder_namespace["PointWiseDecoder2DSimple"](
            16, 2, res=7, scale=0.5
        ).eval()
        x = torch.randn(2, 11, 1, requires_grad=True)
        source = torch.rand(2, 11, 2)
        queries = torch.rand(2, 7, 2)
        with warnings.catch_warnings():
            # The original InstanceNorm1d call has a known num_features warning.
            warnings.filterwarnings("ignore", category=UserWarning)
            encoded = encoder(torch.cat((x, source), dim=-1), source)
            result = decoder(encoded, queries, source)
            gradient = torch.autograd.grad(result.square().sum(), x)[0]
        cases.append(
            {
                "seed": seed,
                "model_kwargs": {
                    "in_channels": 1,
                    "out_channels": 2,
                    "hidden_channels": 8,
                    "latent_channels": 16,
                    "n_layers": 2,
                    "n_heads": 4,
                    "resolution": 7,
                    "coordinate_scale": 0.5,
                    "use_layer_norm": use_layer_norm,
                    "embedding_dropout": 0.05,
                    "attention_dropout": 0.03,
                },
                "encoder_state_dict": tensor_state(encoder),
                "decoder_state_dict": tensor_state(decoder),
                "x": x.detach(),
                "input_geom": source,
                "output_queries": queries,
                "encoder_output": encoded.detach(),
                "output": result.detach(),
                "input_gradient": gradient.detach(),
            }
        )
    fixture = {
        "source_url": SOURCE_URL,
        "source_revision": REVISION,
        "source_sha256": {
            name: hashlib.sha256((upstream_root / name).read_bytes()).hexdigest()
            for name in SOURCE_FILES
        },
        "torch_version": str(torch.__version__),
        "scope": "Synthetic random-weight author reference; no trained checkpoint or PDE data.",
        "mode": "eval",
        "dtype": "float32",
        "gradient_loss": "output.square().sum()",
        "cases": cases,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fixture, output)
    print(
        f"Saved {len(cases)} synthetic author-reference cases to {output} ({output.stat().st_size} bytes)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_name("oformer_reference.pt")
    )
    args = parser.parse_args()
    generate(args.upstream_root, args.output)
