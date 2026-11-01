"""
.. _galerkin_synthetic_example:

Galerkin Transformer on rectangular grids
=========================================

This example uses synthetic fields and random weights to demonstrate the static
two-dimensional model's tensor interface. It does not train, download a dataset,
load a checkpoint, or measure PDE accuracy.
"""

import matplotlib.pyplot as plt
import torch

from neuralop.models import GalerkinTransformer

# %%
# Build a small model. The default model constructor instead matches the width,
# depth and spectral settings of the author's static Darcy checkpoint.

torch.manual_seed(0)
model = GalerkinTransformer(
    in_channels=1,
    out_channels=1,
    hidden_channels=16,
    n_heads=4,
    n_layers=2,
    n_modes=4,
    spectral_channels=8,
    n_spectral_layers=2,
).eval()
y_axis = torch.linspace(0, 1, 24)
x_axis = torch.linspace(0, 1, 32)
yy, xx = torch.meshgrid(y_axis, x_axis, indexing="ij")
x = (torch.sin(torch.pi * xx) * torch.cos(2 * torch.pi * yy))[None, None]
with torch.inference_mode():
    output = model(x)
    larger_output = model(x, output_shape=(32, 40))
print("Input:", tuple(x.shape))
print("Same-grid output:", tuple(output.shape))
print("Larger-grid output:", tuple(larger_output.shape))

# %%
# Explicit output coordinates are optional. Coordinate channel zero denotes x
# across width; channel one denotes y across height. Shared and batched grids
# are supported, and the model converts them to the field's dtype and device.

grid = torch.stack((xx, yy), dim=-1)
with torch.inference_mode():
    explicit_output = model(x, grid=grid)
print("Explicit-grid output:", tuple(explicit_output.shape))

# %%
# Display random model outputs only to inspect the layouts.

figure, axes = plt.subplots(1, 3, figsize=(10, 3), constrained_layout=True)
for ax, field, title in zip(
    axes,
    (x[0, 0], output[0, 0], larger_output[0, 0]),
    ("Synthetic input", "Random model: 24 x 32", "Random model: 32 x 40"),
):
    image = ax.imshow(field.numpy(), origin="lower", extent=(0, 1, 0, 1))
    ax.set_title(title)
    figure.colorbar(image, ax=ax, shrink=0.8)
plt.show()

# %%
# For a matching author checkpoint, use ``model.load_author_state_dict(state)``
# with its tensor dictionary. Original input/output normalization and boundary
# postprocessing must be applied outside the model. In particular, enforce zero
# physical Dirichlet values after inverse normalization, not before it.
