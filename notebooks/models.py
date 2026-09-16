"""A minimal UNet, built from scratch.

This mirrors the `BasicUNet` from the original notebook: three
down-convolutions and three up-convolutions with skip connections between
matching resolutions, `MaxPool2d` for downsampling and `nn.Upsample` for
upsampling (i.e. no learnable up/downsampling, unlike `diffusers`'
`UNet2DModel` used in Phase 4). It takes a noisy image and predicts the
denoised image directly -- no timestep conditioning, no attention, no
normalization layers. It's intentionally the simplest thing that could
plausibly work, so the more sophisticated pieces added in Phase 4 stand
out clearly by contrast.
"""

import torch
from torch import nn


class BasicUNet(nn.Module):
    """A minimal UNet implementation.

    Args:
        in_channels: Number of channels in the input image (1 for
            greyscale MNIST/FashionMNIST).
        out_channels: Number of channels in the output prediction.
            Normally equal to ``in_channels``.

    Shape:
        Input: ``(batch, in_channels, H, W)``
        Output: ``(batch, out_channels, H, W)`` -- same spatial size as the
        input (``H`` and ``W`` must be divisible by 4, since we downsample
        twice by a factor of 2).
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super().__init__()
        self.down_layers = nn.ModuleList(
            [
                nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),
                nn.Conv2d(32, 64, kernel_size=5, padding=2),
                nn.Conv2d(64, 64, kernel_size=5, padding=2),
            ]
        )
        self.up_layers = nn.ModuleList(
            [
                nn.Conv2d(64, 64, kernel_size=5, padding=2),
                nn.Conv2d(64, 32, kernel_size=5, padding=2),
                nn.Conv2d(32, out_channels, kernel_size=5, padding=2),
            ]
        )
        self.act = nn.SiLU()
        self.downscale = nn.MaxPool2d(2)
        self.upscale = nn.Upsample(scale_factor=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for i, layer in enumerate(self.down_layers):
            x = self.act(layer(x))
            if i < 2:  # All but the last down layer: stash for a skip connection
                skips.append(x)
                x = self.downscale(x)

        for i, layer in enumerate(self.up_layers):
            if i > 0:  # All but the first up layer: upscale and merge the skip
                x = self.upscale(x)
                x = x + skips.pop()
            x = self.act(layer(x))

        return x


def count_parameters(model: nn.Module) -> int:
    """Total number of parameters in a model (trainable or not)."""
    return sum(p.numel() for p in model.parameters())
