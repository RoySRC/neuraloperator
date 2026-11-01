"""
.. _oformer_synthetic_example:

OFormer on grids and query points
================================

This example demonstrates the static two-dimensional OFormer interface with
synthetic tensors and random weights. The outputs illustrate tensor layouts,
not PDE solutions or pretrained accuracy. It performs no training, checkpoint
download, or dataset download.
"""

# %%
# Construct a small static model
# -----------------------------
# ``in_channels`` counts field values only: the model adds spatial coordinates.
# Set the device before inference. The same model can query different output
# grids while keeping its rotary-frequency calibration (``resolution``) fixed.

import matplotlib.pyplot as plt
import torch

from neuralop.models import OFormer

torch.manual_seed(0)
model = OFormer(
    in_channels=1,
    out_channels=1,
    hidden_channels=16,
    latent_channels=32,
    n_layers=2,
    n_heads=2,
    resolution=16,
).eval()

axis = torch.linspace(0, 1, 16)
xx, yy = torch.meshgrid(axis, axis, indexing="ij")
x = (torch.sin(2 * torch.pi * xx) * torch.cos(2 * torch.pi * yy))[None, None]

with torch.inference_mode():
    same_grid = model(x)
    finer_grid = model(x, output_shape=(24, 24))

print("Input field:", tuple(x.shape))
print("Same-grid output:", tuple(same_grid.shape))
print("Finer-grid output:", tuple(finer_grid.shape))

# %%
# Query points independently of the input grid
# -------------------------------------------
# Point fields use a channel-last layout. Shared coordinates have shape
# ``(points, 2)``; batched coordinates may also be supplied. Queries can have a
# different number of points from the input sensors.

input_geom = torch.stack((xx, yy), dim=-1).reshape(-1, 2)
point_values = x.flatten(2).transpose(1, 2)
queries = torch.tensor([[0.1, 0.2], [0.5, 0.5], [0.9, 0.8]])
with torch.inference_mode():
    at_points = model(point_values, input_geom=input_geom, output_queries=queries)
print("Point-query output:", tuple(at_points.shape))

# %%
# Inspect shapes visually
# -----------------------
# Both output fields below come from an untrained model.

figure, axes = plt.subplots(1, 3, figsize=(10, 3), constrained_layout=True)
for ax, values, title in zip(
    axes,
    (x[0, 0], same_grid[0, 0], finer_grid[0, 0]),
    ("Synthetic input (16 x 16)", "Random model (16 x 16)", "Random model (24 x 24)"),
):
    image = ax.imshow(values.numpy(), origin="lower", extent=(0, 1, 0, 1))
    ax.set_title(title)
    figure.colorbar(image, ax=ax, shrink=0.8)
plt.show()

# %%
# Using an existing author checkpoint
# ----------------------------------
# The model's ``load_author_state_dict(encoder_state_dict, decoder_state_dict)``
# method accepts the separate parameter dictionaries from the static Darcy
# implementation. Match the original architecture and preserve its input/output
# normalization and boundary processing. This example deliberately does not
# load a checkpoint or claim pretrained inference readiness.
