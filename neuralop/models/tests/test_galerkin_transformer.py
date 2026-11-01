"""Numerical and public API contracts of the static 2D Galerkin Transformer."""

import copy
from types import SimpleNamespace

import pytest
import torch

from neuralop.models import GalerkinTransformer, get_model
from neuralop.models.base_model import available_models


def small_model(**kwargs):
    settings = dict(
        in_channels=1,
        out_channels=2,
        hidden_channels=8,
        n_heads=2,
        n_layers=2,
        dim_feedforward=16,
        n_modes=2,
        spectral_channels=4,
        n_spectral_layers=2,
    )
    settings.update(kwargs)
    return GalerkinTransformer(**settings).eval()


def xy_grid(height, width, dtype=torch.float32):
    y = torch.linspace(0, 1, height, dtype=torch.float64).to(dtype)
    x = torch.linspace(0, 1, width, dtype=torch.float64).to(dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), dim=-1)


def tensor_state(model):
    return {
        k: v.detach().clone()
        for k, v in model.state_dict().items()
        if torch.is_tensor(v)
    }


@pytest.mark.parametrize("shape", [(16, 20), (20, 16)])
def test_rectangular_fields_and_explicit_coordinates(shape):
    model = small_model()
    x = torch.randn(2, 1, *shape)
    latent = tuple(int(int(size * 0.54) * 0.54) for size in shape)
    pos = xy_grid(*latent).reshape(-1, 2)
    grid = xy_grid(*shape)
    with torch.no_grad():
        default = model(x)
        explicit = model(x, pos=pos, grid=grid)
    assert default.shape == (2, 2, *shape)
    assert torch.isfinite(default).all()
    torch.testing.assert_close(default, explicit, rtol=0, atol=0)


@pytest.mark.parametrize("coordinate_batch", [None, 1, 2])
def test_shared_and_batched_coordinates(coordinate_batch):
    model = small_model()
    x = torch.randn(2, 1, 16, 20)
    pos = xy_grid(4, 5).reshape(-1, 2)
    grid = xy_grid(18, 22)
    with torch.no_grad():
        expected = model(x, pos=pos, grid=grid)
        if coordinate_batch is not None:
            pos = pos[None].expand(coordinate_batch, -1, -1)
            grid = grid[None].expand(coordinate_batch, -1, -1, -1)
        actual = model(x, pos=pos, grid=grid)
    assert actual.shape == (2, 2, 18, 22)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_requested_output_shape_matches_explicit_grid():
    model = small_model()
    x = torch.randn(2, 1, 16, 20)
    with torch.no_grad():
        requested = model(x, output_shape=(22, 18))
        explicit = model(x, grid=xy_grid(22, 18))
        both = model(x, grid=xy_grid(22, 18), output_shape=(22, 18))
    assert requested.shape == (2, 2, 22, 18)
    torch.testing.assert_close(requested, explicit, rtol=0, atol=0)
    torch.testing.assert_close(requested, both, rtol=0, atol=0)


def test_eval_is_deterministic_with_nonzero_dropout():
    model = small_model(
        dropout=0.2,
        encoder_dropout=0.3,
        ffn_dropout=0.2,
        downscaler_dropout=0.2,
        upscaler_dropout=0.2,
        decoder_dropout=0.2,
    )
    x = torch.randn(1, 1, 16, 20)
    with torch.no_grad():
        first = model(x)
        second = model(x)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_training_enables_dropout():
    model = small_model(
        dropout=0.3,
        encoder_dropout=0.4,
        ffn_dropout=0.3,
        downscaler_dropout=0.3,
        upscaler_dropout=0.3,
        decoder_dropout=0.3,
    ).train()
    x = torch.randn(1, 1, 16, 20)
    with torch.no_grad():
        first = model(x)
        second = model(x)
    assert not torch.equal(first, second)


@pytest.mark.parametrize(
    "layer_norm,attention_norm", [(False, True), (True, False), (True, True)]
)
def test_gradients_and_batch_isolation(layer_norm, attention_norm):
    model = small_model(layer_norm=layer_norm, attention_norm=attention_norm)
    x = torch.randn(2, 1, 16, 20, requires_grad=True)
    output = model(x, y=torch.randn(2, 2, 16, 20))
    output[0].square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert x.grad[0].abs().sum() > 0
    assert torch.count_nonzero(x.grad[1]) == 0
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_coordinate_gradients_survive_conversion():
    model = small_model()
    x = torch.randn(1, 1, 16, 20)
    pos = xy_grid(4, 5, torch.float64).reshape(-1, 2).requires_grad_(True)
    grid = xy_grid(16, 20, torch.float64).requires_grad_(True)
    model(x, pos=pos, grid=grid).square().mean().backward()
    for geometry in (pos, grid):
        assert geometry.grad is not None
        assert torch.isfinite(geometry.grad).all()
        assert geometry.grad.abs().sum() > 0


@pytest.mark.parametrize("weight_shape", [(20, 1), (1, 20, 1), (2, 20, 1)])
def test_unit_attention_weights_preserve_output(weight_shape):
    model = small_model()
    x = torch.randn(2, 1, 16, 20)
    with torch.no_grad():
        expected = model(x)
        actual = model(x, weight=torch.ones(weight_shape))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_double_precision_after_float_forward():
    model = small_model()
    x = torch.randn(1, 1, 16, 20)
    with torch.no_grad():
        model(x)
        output = model.double()(x.double())
    assert output.dtype == torch.float64
    assert torch.isfinite(output).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
def test_cuda_after_cpu_forward():
    model = small_model()
    x = torch.randn(1, 1, 16, 20)
    with torch.no_grad():
        expected = model(x)
        actual = model.cuda()(x.cuda()).cpu()
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def test_state_and_checkpoint_round_trips(tmp_path):
    model = small_model(layer_norm=True)
    x = torch.randn(1, 1, 16, 20)
    with torch.no_grad():
        expected = model(x)
    same = small_model(layer_norm=True)
    same.load_state_dict(copy.deepcopy(model.state_dict()))
    model.save_checkpoint(tmp_path, "galerkin")
    restored = GalerkinTransformer.from_checkpoint(
        tmp_path, "galerkin", map_location="cpu"
    ).eval()
    with torch.no_grad():
        torch.testing.assert_close(same(x), expected, rtol=0, atol=0)
        torch.testing.assert_close(restored(x), expected, rtol=0, atol=0)


def test_checkpoint_preserves_convolution_layout_and_parameter_identity(tmp_path):
    source = small_model()
    first_conv = next(m for m in source.modules() if isinstance(m, torch.nn.Conv2d))
    # Reproduce the released checkpoint's singleton-channel stride (9,1,3,1).
    with torch.no_grad():
        first_conv.weight.set_(
            first_conv.weight.clone(memory_format=torch.channels_last)
        )
    weights = {k: v for k, v in source.state_dict().items() if torch.is_tensor(v)}
    target = small_model().double()
    before_ids = {name: id(p) for name, p in target.named_parameters()}
    target.load_author_state_dict(weights)
    for name, parameter in target.named_parameters():
        assert id(parameter) == before_ids[name]
        assert parameter.dtype == torch.float64
        assert parameter.device.type == "cpu"
        assert parameter.stride() == weights[name].stride()
        torch.testing.assert_close(parameter, weights[name].double(), rtol=0, atol=0)
    source.save_checkpoint(tmp_path, "layout")
    restored = GalerkinTransformer.from_checkpoint(
        tmp_path, "layout", map_location="cpu"
    ).eval()
    for name, parameter in restored.named_parameters():
        assert parameter.stride() == weights[name].stride()
    x = torch.randn(1, 1, 16, 20)
    with torch.no_grad():
        torch.testing.assert_close(restored(x), source(x), rtol=0, atol=0)

    # An enclosing module loads children recursively, without calling their
    # public load_state_dict methods. The memory-layout contract still holds.
    nested = torch.nn.ModuleDict({"operator": small_model()})
    nested.load_state_dict(
        {f"operator.{name}": value for name, value in weights.items()}
    )
    for name, parameter in nested["operator"].named_parameters():
        assert parameter.stride() == weights[name].stride()
        torch.testing.assert_close(parameter, weights[name], rtol=0, atol=0)


def test_config_dispatch_and_typed_config():
    from config.models import GalerkinTransformerConfig

    class Config(SimpleNamespace):
        def __getitem__(self, name):
            if name not in vars(self):
                raise KeyError(name)
            return vars(self)[name]

    settings = GalerkinTransformerConfig(
        data_channels=1,
        out_channels=2,
        hidden_channels=8,
        n_heads=2,
        n_layers=2,
        n_modes=2,
        spectral_channels=4,
    ).to_dict()
    original = copy.deepcopy(settings)
    model = get_model(Config(model=settings)).eval()
    assert "galerkintransformer" in available_models()
    assert isinstance(model, GalerkinTransformer)
    assert settings == original
    with torch.no_grad():
        assert model(torch.randn(1, 1, 16, 20)).shape == (1, 2, 16, 20)


def test_author_state_loading_is_strict():
    model = small_model()
    restored = small_model()
    state = tensor_state(model)
    restored.load_author_state_dict(state)
    x = torch.randn(1, 1, 16, 20)
    with torch.no_grad():
        torch.testing.assert_close(restored(x), model(x), rtol=0, atol=0)
    missing = dict(state)
    missing.pop(next(iter(missing)))
    with pytest.raises(ValueError):
        restored.load_author_state_dict(missing)
    unexpected = dict(state, invalid_tensor=torch.ones(1))
    with pytest.raises(ValueError):
        restored.load_author_state_dict(unexpected)


@pytest.mark.parametrize("kind", ["shape", "integer", "complex", "sparse", "meta"])
def test_invalid_author_state_fails_before_any_weight_change(kind):
    model = small_model()
    before = tensor_state(model)
    state = copy.deepcopy(before)
    keys = list(state)
    state[keys[0]] += 1
    value = state[keys[-1]]
    if kind == "shape":
        replacement = value.new_zeros(99)
    elif kind == "integer":
        replacement = value.to(torch.int64)
    elif kind == "complex":
        replacement = value.to(torch.complex64)
    elif kind == "sparse":
        replacement = value.to_sparse()
    else:
        replacement = torch.empty_like(value, device="meta")
    state[keys[-1]] = replacement
    with pytest.raises(ValueError):
        model.load_author_state_dict(state)
    torch.testing.assert_close(tensor_state(model), before, rtol=0, atol=0)


@pytest.mark.parametrize(
    "options",
    [
        {"in_channels": 0},
        {"out_channels": 0},
        {"hidden_channels": 2},
        {"hidden_channels": 7},
        {"n_heads": 0},
        {"n_layers": 0},
        {"dim_feedforward": 0},
        {"n_modes": 0},
        {"spectral_channels": 0},
        {"n_spectral_layers": 0},
        {"downsample_factors": (0.0, 0.5)},
        {"downsample_factors": (0.5, 1.1)},
        {"downsample_factors": (0.5,)},
        {"downsample_factors": (float("nan"), 0.5)},
        {"norm_eps": 0.0},
        {"dropout": -0.1},
        {"encoder_dropout": 1.1},
        {"layer_norm": False, "attention_norm": False},
    ],
)
def test_invalid_constructor_options(options):
    with pytest.raises(ValueError):
        small_model(**options)


@pytest.mark.parametrize(
    "shape,kwargs",
    [
        ((1, 16, 1), {}),
        ((1, 2, 16, 20), {}),
        ((1, 1, 0, 20), {}),
        ((1, 1, 3, 20), {}),
        ((2, 1, 16, 20), {"pos": torch.rand(19, 2)}),
        ((2, 1, 16, 20), {"pos": torch.rand(20, 3)}),
        ((2, 1, 16, 20), {"pos": torch.rand(3, 20, 2)}),
        ((2, 1, 16, 20), {"grid": torch.rand(16, 20, 3)}),
        ((2, 1, 16, 20), {"grid": torch.rand(3, 16, 20, 2)}),
        ((2, 1, 16, 20), {"output_shape": (0, 20)}),
        ((2, 1, 16, 20), {"output_shape": (3, 20)}),
        ((2, 1, 16, 20), {"output_shape": (16, 1)}),
        ((2, 1, 16, 20), {"grid": torch.rand(16, 20, 2), "output_shape": (20, 16)}),
        ((2, 1, 16, 20), {"weight": torch.ones(20, 20)}),
        ((2, 1, 16, 20), {"weight": torch.ones(19, 1)}),
    ],
)
def test_invalid_inputs(shape, kwargs):
    with pytest.raises(ValueError):
        small_model()(torch.randn(shape), **kwargs)


def test_input_requires_real_floating_point():
    model = small_model()
    for x in (
        torch.ones(1, 1, 16, 20, dtype=torch.int64),
        torch.ones(1, 1, 16, 20, dtype=torch.complex64),
    ):
        with pytest.raises(ValueError):
            model(x)
