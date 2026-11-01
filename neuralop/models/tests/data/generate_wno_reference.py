"""Generate small, independent WNO Version 1 reference cases.

Run with PyTorch, NumPy, PyWavelets and pytorch-wavelets installed::

    python generate_wno_reference.py --upstream-root /path/to/WNO

This executes only the original ``WaveConv2d`` and ``WNO2d`` class definitions
from the separately supplied, hash-pinned author checkout. It does not import
NeuralOperator, load a trained checkpoint, execute the author's top-level
training/evaluation program, or access PDE data. The fixture contains random
synthetic weights and tensors only; the author source is not redistributed.
"""

import argparse
import ast
import hashlib
import importlib.metadata
from pathlib import Path
import subprocess

import numpy as np
import torch
from pytorch_wavelets import DWT, IDWT

REVISION = "8133e13f36d66f04895571d1c5507b31972805cb"
SOURCE_URL = "https://github.com/TapasTripura/WNO"
SOURCE_FILE = "Version 1.0.0/WNO_testing_2d_Darcy_r.py"
SOURCE_SHA256 = "9bbff5fdf67026d3fa30349bc31b2f6d908206b9fa103ad372978df3bc01fbc9"


def original_classes(upstream_root):
    revision = subprocess.check_output(
        ["git", "-C", str(upstream_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REVISION:
        raise RuntimeError(f"Expected original revision {REVISION}, got {revision}")
    source = upstream_root / SOURCE_FILE
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise RuntimeError("The original author source does not match its pinned hash.")
    tree = ast.parse(source.read_text())
    names = {"WaveConv2d", "WNO2d"}
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name in names
    ]
    if {node.name for node in classes} != names:
        raise RuntimeError("Required original model classes are missing.")
    namespace = {
        "torch": torch,
        "np": np,
        "nn": torch.nn,
        "F": torch.nn.functional,
        "DWT": DWT,
        "IDWT": IDWT,
        "device": torch.device("cpu"),
    }
    exec(
        compile(ast.Module(body=classes, type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return namespace


def generate(upstream_root, output):
    classes = original_classes(upstream_root)
    torch.set_num_threads(1)
    cases = []
    for index, shape in enumerate(((17, 19), (17, 33))):
        seed = 20261101 + index
        torch.manual_seed(seed)
        np.random.seed(seed)
        width, projection = 4, 8
        padded_shape = tuple(size + 1 for size in shape)
        lowpass, _ = DWT(J=4, mode="symmetric", wave="db4")(
            torch.zeros(1, width, *padded_shape)
        )
        modes = tuple(lowpass.shape[-2:])
        model = classes["WNO2d"](*modes, width).eval()
        # Replace only the projection widths to keep the synthetic fixture small.
        # The released checkpoint's projection width is 192, not this value.
        model.fc1 = torch.nn.Linear(width, projection)
        model.fc2 = torch.nn.Linear(projection, 1)
        x = torch.randn(2, 1, *shape, requires_grad=True)
        result = model(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        gradient = torch.autograd.grad(result.square().sum(), x)[0]
        layer_input = torch.randn(2, width, *padded_shape)
        with torch.no_grad():
            layer_output = model.conv0(layer_input)
        cases.append(
            {
                "seed": seed,
                "model_kwargs": {
                    "in_channels": 1,
                    "out_channels": 1,
                    "hidden_channels": width,
                    "n_layers": 4,
                    "input_shape": list(shape),
                    "n_modes": list(modes),
                    "levels": 4,
                    "wavelet": "db4",
                    "mode": "symmetric",
                    "padding": 1,
                    "projection_channels": projection,
                },
                "state_dict": {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                },
                "x": x.detach(),
                "output": result.detach(),
                "input_gradient": gradient.detach(),
                "layer_input": layer_input,
                "layer_output": layer_output,
            }
        )
    fixture = {
        "source_url": SOURCE_URL,
        "source_revision": REVISION,
        "source_file": SOURCE_FILE,
        "source_sha256": SOURCE_SHA256,
        "source_license_metadata": "No license file found in the audited author revision; source is supplied separately and not copied into this fixture.",
        "torch_version": str(torch.__version__),
        "pytorch_wavelets_version": importlib.metadata.version("pytorch-wavelets"),
        "pywavelets_version": importlib.metadata.version("PyWavelets"),
        "scope": "Random synthetic author-model weights and inputs, not pretrained weights or PDE data.",
        "mode": "eval",
        "dtype": "float32",
        "gradient_loss": "output.square().sum()",
        "cases": cases,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fixture, output)
    print(
        f"Saved {len(cases)} synthetic author-reference cases to {output} "
        f"({output.stat().st_size} bytes)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_name("wno_reference.pt")
    )
    args = parser.parse_args()
    generate(args.upstream_root, args.output)
