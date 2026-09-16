"""A DDPM built from `diffusers` primitives, for direct comparison with
`BasicUNet` + `corrupt()` + the naive sampler from Phases 2-3.

The three concrete differences this module makes visible in code:

1. **Noise type.** ``corrupt()`` mixes in *uniform* noise (``torch.rand``);
   real DDPM training adds *Gaussian* noise (``torch.randn``), matching the
   Gaussian forward-diffusion process the theory assumes.
2. **Training objective.** ``BasicUNet`` was trained to predict the clean
   image directly. Here the model predicts the *noise that was added*
   (the standard DDPM parameterization) -- the loss is against ``noise``,
   not against ``x``.
3. **Timestep conditioning.** ``BasicUNet`` only ever sees the corrupted
   image and has to guess how corrupted it is. ``UNet2DModel`` also
   receives the timestep as an explicit input, so it knows exactly how
   much noise to expect and can specialize its prediction accordingly.

Sampling also changes: instead of the ad hoc "move `1/(n_steps-i)` of the
way toward the prediction" rule, ``DDPMScheduler.step`` implements the
actual reverse-process update derived from the forward noise schedule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F

from diffusers import DDPMScheduler, UNet2DModel


def build_ddpm_unet(
    sample_size: int = 28,
    in_channels: int = 1,
    out_channels: int = 1,
    layers_per_block: int = 2,
    block_out_channels: tuple[int, ...] = (32, 64, 64),
    down_block_types: tuple[str, ...] = ("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
    up_block_types: tuple[str, ...] = ("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
) -> UNet2DModel:
    """Build the `UNet2DModel` used for the DDPM comparison.

    Defaults match the original notebook's configuration: at
    ``sample_size=28`` this has ~1.7M parameters, versus ``BasicUNet``'s
    ~309k -- most of the extra capacity goes toward the attention blocks
    and the timestep-conditioning pathway that ``BasicUNet`` doesn't have
    at all.
    """
    return UNet2DModel(
        sample_size=sample_size,
        in_channels=in_channels,
        out_channels=out_channels,
        layers_per_block=layers_per_block,
        block_out_channels=block_out_channels,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
    )


def build_ddpm_scheduler(num_train_timesteps: int = 1000) -> DDPMScheduler:
    """Build the noise scheduler defining the forward diffusion process."""
    return DDPMScheduler(num_train_timesteps=num_train_timesteps)


@dataclass
class DDPMTrainConfig:
    epochs: int = 1
    lr: float = 4e-4
    device: str = "cpu"
    log_every: int = 100


@dataclass
class DDPMTrainResult:
    losses: list[float]
    epoch_mean_losses: list[float]


def train_ddpm(
    model: UNet2DModel,
    scheduler: DDPMScheduler,
    dataloader: Iterable,
    config: DDPMTrainConfig | None = None,
) -> DDPMTrainResult:
    """Train `model` to predict the Gaussian noise added at a random timestep.

    Contrast with `dms.train.train_model`: here we draw Gaussian noise and
    an integer timestep per sample, corrupt via the scheduler's actual
    forward process (`scheduler.add_noise`), pass the timestep into the
    model alongside the image, and regress onto the *noise*, not the
    clean image.
    """
    config = config or DDPMTrainConfig()
    model.to(config.device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    num_train_timesteps = scheduler.config.num_train_timesteps

    losses: list[float] = []
    epoch_mean_losses: list[float] = []

    for epoch in range(config.epochs):
        epoch_losses: list[float] = []
        for step, (x, _labels) in enumerate(dataloader):
            x = x.to(config.device)
            noise = torch.randn_like(x)
            timesteps = torch.randint(
                0, num_train_timesteps, (x.shape[0],), device=config.device
            ).long()
            noisy_x = scheduler.add_noise(x, noise, timesteps)

            pred = model(noisy_x, timesteps).sample
            loss = F.mse_loss(pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_value = loss.item()
            losses.append(loss_value)
            epoch_losses.append(loss_value)

            if config.log_every and step % config.log_every == 0:
                print(f"epoch {epoch} step {step}: loss={loss_value:.4f}")

        mean_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float("nan")
        epoch_mean_losses.append(mean_loss)
        print(f"epoch {epoch}: mean loss={mean_loss:.4f}")

    model.eval()
    return DDPMTrainResult(losses=losses, epoch_mean_losses=epoch_mean_losses)


@torch.no_grad()
def sample_ddpm(
    model: UNet2DModel,
    scheduler: DDPMScheduler,
    shape: tuple[int, int, int, int] = (8, 1, 28, 28),
    device: str = "cpu",
    num_inference_steps: int | None = None,
    return_intermediates: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
    """Sample by running the scheduler's actual reverse diffusion process.

    Contrast with `dms.sample.sample`: instead of an ad hoc mixing
    fraction, `scheduler.step` applies the reverse-process update implied
    by the forward noise schedule the model was trained against.
    """
    was_training = model.training
    model.eval()
    model.to(device)

    scheduler.set_timesteps(num_inference_steps or scheduler.config.num_train_timesteps)
    x = torch.randn(shape, device=device)
    step_history = [x.detach().cpu()]

    for t in scheduler.timesteps:
        residual = model(x, t).sample
        x = scheduler.step(residual, t, x).prev_sample
        step_history.append(x.detach().cpu())

    if was_training:
        model.train()

    if return_intermediates:
        return x, step_history
    return x
