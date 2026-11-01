"""Interface and numerical contracts of the static two-dimensional OFormer."""

import copy

import pytest
import torch

from neuralop.models import OFormer, get_model
from neuralop.models.base_model import available_models


def small_model(**kwargs):
    options = dict(
        in_channels=2,
        out_channels=3,
        hidden_channels=8,
        latent_channels=16,
        n_layers=2,
        n_heads=2,
        resolution=16,
        embedding_dropout=0.0,
        attention_dropout=0.0,
    )
    options.update(kwargs)
    return OFormer(**options).eval()


def grid(height, width, *, dtype=torch.float32, device="cpu"):
    return torch.stack(
        torch.meshgrid(
            torch.linspace(0, 1, height, dtype=torch.float64).to(
                device=device, dtype=dtype
            ),
            torch.linspace(0, 1, width, dtype=torch.float64).to(
                device=device, dtype=dtype
            ),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 2)


@pytest.mark.parametrize("height,width", [(3, 5), (6, 4)])
def test_grid_and_explicit_points_agree(height, width):
    model = small_model()
    x = torch.randn(2, 2, height, width)
    points = x.flatten(2).transpose(1, 2)
    coordinates = grid(height, width)
    with torch.no_grad():
        on_grid = model(x)
        on_points = model(points, input_geom=coordinates)
        explicit_grid = model(x, input_geom=coordinates)
    assert on_grid.shape == (2, 3, height, width)
    assert on_points.shape == (2, height * width, 3)
    assert torch.isfinite(on_grid).all()
    torch.testing.assert_close(on_grid.flatten(2).transpose(1, 2), on_points)
    torch.testing.assert_close(on_grid, explicit_grid)


@pytest.mark.parametrize("coordinate_batch", [None, 1, 2])
def test_shared_and_batched_coordinates(coordinate_batch):
    model = small_model()
    x = torch.randn(2, 7, 2)
    source = torch.rand(7, 2)
    queries = torch.rand(11, 2)
    with torch.no_grad():
        expected = model(x, input_geom=source, output_queries=queries)
        if coordinate_batch is not None:
            source = source.unsqueeze(0).expand(coordinate_batch, -1, -1)
            queries = queries.unsqueeze(0).expand(coordinate_batch, -1, -1)
        actual = model(x, input_geom=source, output_queries=queries)
    assert actual.shape == (2, 11, 3)
    torch.testing.assert_close(actual, expected)


def test_output_grid_matches_explicit_queries():
    model = small_model()
    x = torch.randn(2, 2, 3, 4)
    queries = grid(5, 7)
    with torch.no_grad():
        expected = model(x, output_queries=queries)
        actual = model(x, output_shape=(5, 7))
        from_points = model(
            x.flatten(2).transpose(1, 2),
            input_geom=grid(3, 4),
            output_shape=(5, 7),
        )
    assert actual.shape == (2, 3, 5, 7)
    torch.testing.assert_close(actual.flatten(2).transpose(1, 2), expected)
    torch.testing.assert_close(actual, from_points)


def test_point_permutations_preserve_the_operator():
    model = small_model()
    x = torch.randn(2, 9, 2)
    source = torch.rand(2, 9, 2)
    queries = torch.rand(2, 6, 2)
    source_order = torch.randperm(9)
    query_order = torch.randperm(6)
    with torch.no_grad():
        expected = model(x, input_geom=source, output_queries=queries)
        actual = model(
            x[:, source_order],
            input_geom=source[:, source_order],
            output_queries=queries[:, query_order],
        )
    torch.testing.assert_close(actual, expected[:, query_order], rtol=1e-4, atol=1e-5)


def test_query_chunk_and_subset_consistency():
    model = small_model()
    x = torch.randn(2, 7, 2)
    source = torch.rand(2, 7, 2)
    queries = torch.rand(2, 11, 2)
    subset = torch.tensor([7, 0, 4])
    with torch.no_grad():
        full = model(x, input_geom=source, output_queries=queries)
        chunks = torch.cat(
            [
                model(x, input_geom=source, output_queries=chunk)
                for chunk in queries.split(3, dim=1)
            ],
            dim=1,
        )
        selected = model(x, input_geom=source, output_queries=queries[:, subset])
    torch.testing.assert_close(chunks, full, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(selected, full[:, subset], rtol=1e-4, atol=1e-5)


def test_coordinate_gradients_survive_dtype_conversion():
    model = small_model()
    x = torch.randn(1, 5, 2)
    source = torch.rand(1, 5, 2, dtype=torch.float64, requires_grad=True)
    queries = torch.rand(1, 3, 2, dtype=torch.float64, requires_grad=True)
    model(x, input_geom=source, output_queries=queries).square().sum().backward()
    for coordinates in (source, queries):
        assert coordinates.grad is not None
        assert torch.isfinite(coordinates.grad).all()
        assert coordinates.grad.abs().sum() > 0


@pytest.mark.parametrize("use_layer_norm", [False, True])
def test_gradients_and_batch_isolation(use_layer_norm):
    model = small_model(use_layer_norm=use_layer_norm)
    x = torch.randn(2, 2, 3, 4, requires_grad=True)
    output = model(x, y=torch.randn(2, 3, 3, 4))
    output[0].square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert x.grad[0].abs().sum() > 0
    assert torch.count_nonzero(x.grad[1]) == 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name


def test_single_source_point():
    model = small_model()
    x = torch.randn(2, 1, 2)
    with torch.no_grad():
        output = model(x, input_geom=torch.rand(1, 2))
    assert output.shape == (2, 1, 3)
    assert torch.isfinite(output).all()


def test_double_precision_after_first_forward():
    model = small_model()
    x = torch.randn(1, 2, 3, 4)
    with torch.no_grad():
        model(x)
        output = model.double()(x.double())
    assert output.dtype == torch.float64
    assert torch.isfinite(output).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
def test_cuda_after_cpu_forward():
    model = small_model()
    x = torch.randn(1, 2, 3, 4)
    with torch.no_grad():
        expected = model(x)
        actual = model.cuda()(x.cuda()).cpu()
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS device unavailable"
)
def test_mps_float32_grid_after_cpu_forward():
    model = small_model()
    x = torch.randn(1, 2, 3, 4)
    with torch.no_grad():
        expected = model(x, output_shape=(4, 5))
        actual = model.to("mps")(x.to("mps"), output_shape=(4, 5)).cpu()
    assert actual.dtype == torch.float32
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-4)


def test_registry_and_config_dispatch():
    class Config(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

    config = Config(
        model=dict(
            model_arch="oformer",
            data_channels=2,
            out_channels=3,
            hidden_channels=8,
            latent_channels=16,
            n_layers=2,
            n_heads=2,
            resolution=16,
        )
    )
    original = copy.deepcopy(config)
    model = get_model(config).eval()
    assert "oformer" in available_models()
    assert isinstance(model, OFormer)
    assert config == original
    with torch.no_grad():
        assert model(torch.randn(1, 2, 3, 4)).shape == (1, 3, 3, 4)


def test_typed_oformer_config():
    from types import SimpleNamespace

    from config.models import OFormerConfig

    class Config(SimpleNamespace):
        def __getitem__(self, name):
            if name not in vars(self):
                raise KeyError(name)
            return vars(self)[name]

    settings = OFormerConfig(
        data_channels=2,
        out_channels=3,
        hidden_channels=8,
        latent_channels=16,
        n_layers=2,
        n_heads=2,
        resolution=16,
    ).to_dict()
    model = get_model(Config(model=settings))
    assert isinstance(model, OFormer)
    assert model.in_channels == 2


def test_checkpoint_round_trip(tmp_path):
    model = small_model()
    x = torch.randn(1, 2, 3, 4)
    with torch.no_grad():
        expected = model(x)
    model.save_checkpoint(tmp_path, "oformer")
    restored = OFormer.from_checkpoint(tmp_path, "oformer", map_location="cpu").eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(x), expected, rtol=0, atol=0)


def test_author_state_dict_bridge_is_strict():
    model = small_model()
    restored = small_model()
    encoder = copy.deepcopy(model.encoder.state_dict())
    decoder = copy.deepcopy(model.decoder.state_dict())
    restored.load_author_state_dict(encoder, decoder)
    x = torch.randn(1, 2, 3, 4)
    with torch.no_grad():
        torch.testing.assert_close(restored(x), model(x), rtol=0, atol=0)
    # A missing tensor must fail rather than leaving random parameters in place.
    incomplete = copy.deepcopy(encoder)
    incomplete.pop(next(iter(incomplete)))
    with pytest.raises((ValueError, RuntimeError)):
        restored.load_author_state_dict(incomplete, decoder)
    unexpected = copy.deepcopy(decoder)
    unexpected["not_a_decoder_parameter"] = torch.ones(1)
    with pytest.raises((ValueError, RuntimeError)):
        restored.load_author_state_dict(encoder, unexpected)


def test_author_shape_validation_precedes_any_weight_update():
    model = small_model()
    encoder = copy.deepcopy(model.encoder.state_dict())
    decoder = copy.deepcopy(model.decoder.state_dict())
    # A valid encoder update must not happen if the decoder is incompatible.
    encoder_key = next(iter(encoder))
    encoder[encoder_key] = encoder[encoder_key] + 1
    decoder_key = next(iter(decoder))
    decoder[decoder_key] = decoder[decoder_key].new_zeros(1)
    before = {
        name: value.clone()
        for name, value in model.state_dict().items()
        if torch.is_tensor(value)
    }
    with pytest.raises(ValueError):
        model.load_author_state_dict(encoder, decoder)
    after = {
        name: value
        for name, value in model.state_dict().items()
        if torch.is_tensor(value)
    }
    torch.testing.assert_close(after, before, rtol=0, atol=0)


@pytest.mark.parametrize("tensor_format", ["integer", "complex", "sparse", "meta"])
def test_author_tensor_format_validation_is_atomic(tensor_format):
    model = small_model()
    encoder = copy.deepcopy(model.encoder.state_dict())
    decoder = copy.deepcopy(model.decoder.state_dict())
    encoder_key = next(iter(encoder))
    encoder[encoder_key] = encoder[encoder_key] + 1
    decoder_key = next(iter(decoder))
    value = decoder[decoder_key]
    if tensor_format == "integer":
        decoder[decoder_key] = value.to(torch.int64)
    elif tensor_format == "complex":
        decoder[decoder_key] = value.to(torch.complex64)
    elif tensor_format == "sparse":
        decoder[decoder_key] = value.to_sparse()
    else:
        decoder[decoder_key] = torch.empty_like(value, device="meta")
    before = {
        name: value.clone()
        for name, value in model.state_dict().items()
        if torch.is_tensor(value)
    }
    with pytest.raises(ValueError):
        model.load_author_state_dict(encoder, decoder)
    after = {
        name: value
        for name, value in model.state_dict().items()
        if torch.is_tensor(value)
    }
    torch.testing.assert_close(after, before, rtol=0, atol=0)


@pytest.mark.parametrize(
    "options",
    [
        {"in_channels": 0},
        {"out_channels": 0},
        {"hidden_channels": 6},
        {"latent_channels": 10},
        {"n_layers": 1},
        {"n_heads": 0},
        {"resolution": 3},
        {"coordinate_scale": 0},
        {"embedding_dropout": -0.1},
        {"attention_dropout": 1.1},
    ],
)
def test_invalid_model_settings(options):
    with pytest.raises(ValueError):
        small_model(**options)


@pytest.mark.parametrize(
    "x_shape,kwargs",
    [
        ((2, 7, 2), {}),  # Point inputs need their coordinates.
        ((2, 2, 3, 4), {"input_geom": torch.rand(11, 2)}),
        ((2, 2, 3, 4), {"input_geom": torch.rand(12, 3)}),
        ((2, 2, 3, 4), {"input_geom": torch.rand(3, 12, 2)}),
        ((2, 2, 3, 4), {"output_queries": torch.rand(5, 3)}),
        ((2, 2, 3, 4), {"output_queries": torch.rand(0, 2)}),
        ((2, 2, 3, 4), {"output_shape": (0, 4)}),
        (
            (2, 2, 3, 4),
            {"output_shape": (3, 4), "output_queries": torch.rand(5, 2)},
        ),
        ((2, 3, 3, 4), {}),  # Wrong field channel count.
        ((2, 2, 3, 4, 5), {}),  # Only static two-dimensional grids are supported.
    ],
)
def test_invalid_field_geometry(x_shape, kwargs):
    with pytest.raises(ValueError):
        small_model()(torch.randn(*x_shape), **kwargs)
