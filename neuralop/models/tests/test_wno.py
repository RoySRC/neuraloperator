"""Interface, geometry and checkpoint contracts of the static 2D WNO."""

import copy
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from neuralop.models import WNO, get_model
from neuralop.models.base_model import available_models


def require_wavelets():
    pytest.importorskip("pytorch_wavelets")
    pytest.importorskip("pywt")


def small_model(**kwargs):
    require_wavelets()
    options = dict(
        in_channels=2,
        out_channels=3,
        hidden_channels=4,
        n_layers=2,
        input_shape=(17, 19),
        levels=2,
        padding=1,
        projection_channels=8,
    )
    options.update(kwargs)
    return WNO(**options).eval()


def tensor_state(model):
    return {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if torch.is_tensor(value)
    }


@pytest.mark.parametrize("shape", [(17, 19), (18, 20), (17, 20), (18, 19)])
@pytest.mark.parametrize("padding", [0, 1])
def test_output_shape_and_finiteness(shape, padding):
    model = small_model(input_shape=shape, padding=padding)
    x = torch.randn(2, 2, *shape)
    with torch.no_grad():
        result = model(x, y=torch.randn(2, 3, *shape))
    assert result.shape == (2, 3, *shape)
    assert torch.isfinite(result).all()


def test_coarse_modes_match_actual_decomposition():
    model = small_model()
    from pytorch_wavelets import DWTForward

    padded = torch.zeros(1, 1, 18, 20)
    coarse, _ = DWTForward(J=2, wave="db4", mode="symmetric")(padded)
    assert tuple(model.n_modes) == tuple(coarse.shape[-2:])
    explicit = small_model(n_modes=tuple(coarse.shape[-2:]))
    explicit.load_state_dict(copy.deepcopy(model.state_dict()))
    x = torch.randn(1, 2, 17, 19)
    with torch.no_grad():
        torch.testing.assert_close(explicit(x), model(x), rtol=0, atol=0)


def test_gradients_and_batch_isolation():
    model = small_model()
    x = torch.randn(2, 2, 17, 19, requires_grad=True)
    result = model(x)
    result[0].square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert x.grad[0].abs().sum() > 0
    assert torch.count_nonzero(x.grad[1]) == 0
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_filters_follow_dtype_after_first_forward():
    model = small_model()
    x = torch.randn(1, 2, 17, 19)
    with torch.no_grad():
        model(x)
        result = model.double()(x.double())
    assert result.dtype == torch.float64
    assert torch.isfinite(result).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
def test_cuda_after_cpu_forward(monkeypatch):
    model = small_model()
    x = torch.randn(1, 2, 17, 19)
    # Compare full float32 arithmetic; TF32 convolution has a different error
    # budget and is an application-level numerical policy, not a device bug.
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    with torch.no_grad(), torch.backends.cudnn.flags(allow_tf32=False):
        expected = model(x)
        actual = model.cuda()(x.cuda()).cpu()
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def test_checkpoint_round_trip(tmp_path):
    model = small_model()
    x = torch.randn(1, 2, 17, 19)
    with torch.no_grad():
        expected = model(x)
    model.save_checkpoint(tmp_path, "wno")
    restored = WNO.from_checkpoint(tmp_path, "wno", map_location="cpu").eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(x), expected, rtol=0, atol=0)


def test_author_state_dict_bridge():
    model = small_model()
    restored = small_model()
    state = tensor_state(model)
    restored.load_author_state_dict(state)
    x = torch.randn(1, 2, 17, 19)
    with torch.no_grad():
        torch.testing.assert_close(restored(x), model(x), rtol=0, atol=0)
    # Fixed wavelet filters are reconstructed from architecture settings.
    assert not any("h0" in name or "g0" in name for name in state)
    assert "conv0.weights1" in state
    assert "conv0.weights2" in state
    assert "w0.weight" in state
    assert "fc0.weight" in state


@pytest.mark.parametrize(
    "error", ["missing", "unexpected", "shape", "integer", "complex", "sparse", "meta"]
)
def test_author_state_validation_is_atomic(error):
    model = small_model()
    before = tensor_state(model)
    incoming = copy.deepcopy(before)
    names = list(incoming)
    # Ensure an early, valid modification cannot leak into a failed load.
    incoming[names[0]] = incoming[names[0]] + 1
    name = names[-1]
    original = incoming[name]
    if error == "missing":
        incoming.pop(name)
    elif error == "unexpected":
        incoming["unexpected_parameter"] = torch.zeros(1)
    elif error == "shape":
        incoming[name] = torch.zeros(original.numel() + 1)
    elif error == "integer":
        incoming[name] = original.to(torch.int64)
    elif error == "complex":
        incoming[name] = original.to(torch.complex64)
    elif error == "sparse":
        incoming[name] = original.to_sparse()
    else:
        incoming[name] = torch.empty_like(original, device="meta")
    with pytest.raises((ValueError, RuntimeError)):
        model.load_author_state_dict(incoming)
    torch.testing.assert_close(tensor_state(model), before, rtol=0, atol=0)


def test_registry_and_typed_config():
    require_wavelets()
    from config.models import WNOConfig

    class Config(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError as error:
                raise AttributeError(name) from error

    settings = WNOConfig(
        data_channels=2,
        out_channels=3,
        hidden_channels=4,
        n_layers=2,
        input_shape=[17, 19],
        levels=2,
        padding=1,
        projection_channels=8,
    ).to_dict()
    config = Config(model=settings)
    before = copy.deepcopy(config)
    model = get_model(config).eval()
    assert isinstance(model, WNO)
    assert "wno" in available_models()
    assert config == before
    with torch.no_grad():
        assert model(torch.randn(1, 2, 17, 19)).shape == (1, 3, 17, 19)


@pytest.mark.parametrize(
    "options",
    [
        {"in_channels": 0},
        {"out_channels": 0},
        {"hidden_channels": 0},
        {"n_layers": 0},
        {"input_shape": (0, 19)},
        {"input_shape": (17,)},
        {"levels": 0},
        {"padding": -1},
        {"projection_channels": 0},
        {"n_modes": (1, 1)},
        {"wavelet": "not-a-wavelet"},
        {"mode": "not-a-boundary-mode"},
    ],
)
def test_invalid_model_settings(options):
    with pytest.raises((ValueError, TypeError)):
        small_model(**options)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 2, 18, 19),
        (1, 2, 17, 20),
        (1, 1, 17, 19),
        (1, 17, 19, 2),
        (1, 2, 17),
        (0, 2, 17, 19),
    ],
)
def test_invalid_input_geometry(shape):
    model = small_model()
    with pytest.raises(ValueError):
        model(torch.randn(*shape))


def test_integer_input_is_rejected():
    model = small_model()
    with pytest.raises(ValueError):
        model(torch.ones(1, 2, 17, 19, dtype=torch.int64))


def test_missing_optional_dependencies_do_not_break_core_imports():
    # A fresh interpreter avoids an already-imported wavelet dependency masking
    # an accidental eager import from neuralop.models.
    script = """
import builtins
real_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'pytorch_wavelets', 'pywt'}:
        raise ModuleNotFoundError('Deliberately unavailable optional dependency: ' + name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import neuralop
from neuralop.models import FNO, WNO
assert FNO is not None
try:
    WNO(in_channels=1, out_channels=1)
except ImportError as error:
    assert 'neuraloperator[wavelets]' in str(error), str(error)
else:
    raise AssertionError('WNO should require its optional wavelet dependency')
"""
    root = Path(__file__).resolve().parents[3]
    environment = dict(os.environ, PYTHONPATH=str(root))
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
