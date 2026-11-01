"""
.. _wno_synthetic_example:

WNO on a fixed two-dimensional grid
==================================

This example evaluates the static Version 1 Wavelet Neural Operator on a
synthetic field. The model has random weights: its output is not a numerical
PDE solution or evidence of pretrained accuracy. There is no training or data
download.

The optional dependencies can be installed from this checkout with
``pip install -e '.[wavelets]'``.
"""

# %%
# Define the model and its grid
# ----------------------------
# The input shape is an architecture setting. Coarse wavelet coefficient sizes
# are derived from this shape; resolution changes require a separately declared
# compatible model rather than implicit interpolation of the input.

import matplotlib.pyplot as plt
import torch

from neuralop.models import WNO

torch.manual_seed(0)
height, width = 17, 19
model = WNO(
    in_channels=1,
    out_channels=1,
    hidden_channels=8,
    n_layers=2,
    input_shape=(height, width),
    levels=2,
    padding=1,
    projection_channels=16,
).eval()

axis_x = torch.linspace(0, 1, height)
axis_y = torch.linspace(0, 1, width)
xx, yy = torch.meshgrid(axis_x, axis_y, indexing="ij")
field = torch.sin(2 * torch.pi * xx) * torch.cos(2 * torch.pi * yy)
x = field[None, None]

with torch.inference_mode():
    output = model(x)

print("Input:", tuple(x.shape))
print("Output:", tuple(output.shape))
print("Coarse coefficient shape:", model.n_modes)

# %%
# Inspect the synthetic input and random-weight output
# ---------------------------------------------------

figure, axes = plt.subplots(1, 2, figsize=(7, 3), constrained_layout=True)
for ax, values, title in zip(
    axes, (field, output[0, 0]), ("Synthetic input", "Random WNO output")
):
    image = ax.imshow(values.numpy(), origin="lower", extent=(0, 1, 0, 1))
    ax.set_title(title)
    figure.colorbar(image, ax=ax, shrink=0.8)
plt.show()

# %%
# Existing author weights
# -----------------------
# ``model.load_author_state_dict(state_dict)`` accepts the original tensor-only
# parameter mapping. For the released Darcy checkpoint, preserve its native
# 85 x 85 input grid, projection width 192, four db4 levels and symmetric mode.
# Original physical normalization is applied outside the model. This example
# deliberately does not load a checkpoint.
