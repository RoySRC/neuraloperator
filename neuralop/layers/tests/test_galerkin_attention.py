import pytest
import torch

from neuralop.layers.galerkin_attention import (
    GalerkinAttention,
    GalerkinTransformerEncoderLayer,
)


@pytest.mark.parametrize("norm_type", ["layer", "instance"])
@pytest.mark.parametrize("pos_dim", [0, 2])
def test_galerkin_matches_explicit_pointwise_kernel(norm_type, pos_dim):
    """Compare efficient Q(K^T V) with the independent, quadratic (QK^T)V."""
    model = (
        GalerkinAttention(
            n_head=2, d_model=8, pos_dim=pos_dim, norm_type=norm_type, dropout=0.4
        )
        .double()
        .eval()
    )
    x = torch.randn(2, 5, 8, dtype=torch.float64)
    pos = torch.rand(2, 5, pos_dim, dtype=x.dtype) if pos_dim else None
    actual, _ = model(x, x, x, pos=pos)

    q, k, v = [
        layer(x).reshape(2, 5, 2, 4).permute(0, 2, 1, 3) for layer in model.linears
    ]
    for h in range(2):
        if norm_type == "instance":
            k[:, h] = model.norm_K[h](k[:, h].transpose(1, 2)).transpose(1, 2)
            v[:, h] = model.norm_V[h](v[:, h].transpose(1, 2)).transpose(1, 2)
        else:
            k[:, h] = model.norm_K[h](k[:, h])
            v[:, h] = model.norm_V[h](v[:, h])
    if pos_dim:
        p = pos[:, None].expand(-1, 2, -1, -1)
        q, k, v = [torch.cat((p, a), dim=-1) for a in (q, k, v)]
    kernel = torch.einsum("bhic,bhjc->bhij", q, k) / x.shape[1]
    expected = torch.einsum("bhij,bhjd->bhid", kernel, v)
    expected = expected.transpose(1, 2).reshape(2, 5, -1)
    if pos_dim:
        expected = model.fc(expected)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_eval_disables_nonzero_attention_and_residual_dropout():
    layer = GalerkinTransformerEncoderLayer(
        d_model=8, n_head=2, dropout=0.6, ffn_dropout=0.4
    ).eval()
    x, pos = torch.randn(2, 7, 8), torch.rand(2, 7, 2)
    with torch.no_grad():
        first = layer(x, pos)
        second = layer(x, pos)
        layer.attn.dropout.p = 0
        layer.dropout1.p = layer.dropout2.p = layer.ff.dropout.p = 0
        zero_dropout = layer(x, pos)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(first, zero_dropout, rtol=0, atol=0)


def test_training_attention_uses_configured_probability():
    attention = GalerkinAttention(n_head=2, d_model=8, dropout=0.25).train()
    x, pos = torch.randn(2, 7, 8), torch.rand(2, 7, 2)
    attention.eval()
    _, full = attention(x, x, x, pos=pos)
    attention.train()
    torch.manual_seed(71)
    _, dropped = attention(x, x, x, pos=pos)
    torch.manual_seed(71)
    expected = torch.nn.functional.dropout(full, p=0.25, training=True)
    torch.testing.assert_close(dropped, expected, rtol=0, atol=0)
    assert torch.count_nonzero(dropped) < dropped.numel()


@pytest.mark.parametrize("layer_norm", [False, True])
def test_encoder_gradients_and_batch_isolation(layer_norm):
    layer = GalerkinTransformerEncoderLayer(
        d_model=8,
        n_head=2,
        layer_norm=layer_norm,
        attn_norm=True,
        dropout=0.0,
        ffn_dropout=0.0,
    ).eval()
    x = torch.randn(2, 7, 8, requires_grad=True)
    pos = torch.rand(2, 7, 2, requires_grad=True)
    layer(x, pos)[0].square().sum().backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(pos.grad).all()
    assert torch.count_nonzero(x.grad[1]) == 0
    assert torch.count_nonzero(pos.grad[1]) == 0
    for name, parameter in layer.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


@pytest.mark.parametrize(
    "kwargs",
    [
        {"d_model": 7, "n_head": 2},
        {"norm_type": "unknown"},
        {"dropout": -0.1},
        {"attention_type": "softmax"},
        {"pos_dim": -1},
    ],
)
def test_invalid_attention_configuration(kwargs):
    options = dict(n_head=2, d_model=8)
    options.update(kwargs)
    with pytest.raises(ValueError):
        GalerkinAttention(**options)


def test_attention_rejects_mask_and_mismatched_geometry():
    attention = GalerkinAttention(n_head=2, d_model=8)
    x = torch.randn(2, 7, 8)
    with pytest.raises(ValueError, match="mask"):
        attention(x, x, x, mask=torch.ones(7, 7))
    with pytest.raises(ValueError, match="pos"):
        attention(x, x, x, pos=torch.rand(2, 6, 2))
