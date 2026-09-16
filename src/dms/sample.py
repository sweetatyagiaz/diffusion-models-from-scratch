"""The naive iterative sampler.

BasicUNet was trained to jump straight from a corrupted image to its
guess at the *fully* clean image -- it was never taught to take small
steps. Sampling in one jump from pure noise therefore gives a blurry,
low-quality result. The fix used in the original notebook (and here) is
purely a sampling-time trick: take the model's prediction, but only move
a fraction of the way towards it, then repeat with the (slightly less
noisy) result. This has no learned or principled schedule behind it --
contrast with the DDPM sampler in Phase 4, where the step sizes and
noise schedule come directly from the forward diffusion process the
model was actually trained on.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class SampleResult:
    """Output of ``sample`` when intermediate steps are requested."""

    samples: torch.Tensor
    step_history: list[torch.Tensor]  # x after each step, including the initial noise
    pred_history: list[torch.Tensor]  # the model's raw prediction at each step


@torch.no_grad()
def sample(
    model: nn.Module,
    n_steps: int = 5,
    shape: tuple[int, int, int, int] = (8, 1, 28, 28),
    device: str = "cpu",
    x: torch.Tensor | None = None,
    return_intermediates: bool = False,
) -> torch.Tensor | SampleResult:
    """Generate samples from ``model`` by iteratively denoising noise.

    At each of ``n_steps`` steps, the model predicts a fully denoised
    image from the current ``x``, and we move ``1 / (n_steps - i)`` of
    the way from ``x`` towards that prediction -- so the first step is a
    small nudge and the last step goes all the way to the model's final
    prediction.

    Args:
        model: A trained model such as ``BasicUNet``, mapping a
            (partially) corrupted image to a predicted clean image.
            Temporarily switched to ``eval()`` for sampling and restored
            to ``train()`` afterwards.
        n_steps: Number of denoising steps. More steps generally trade
            compute for quality, up to a point.
        shape: Shape of the noise to start from, used only if ``x`` is
            not provided.
        device: Device to run sampling on.
        x: Optional starting point (e.g. pure noise, or a partially
            corrupted real image). If omitted, starts from
            ``torch.rand(shape)``, matching the corruption process's
            ``amount=1`` case.
        return_intermediates: If ``True``, also return every
            intermediate ``x`` and every raw model prediction, useful
            for visualizing the denoising trajectory.

    Returns:
        The final denoised batch, shape ``(batch, channels, H, W)``. If
        ``return_intermediates`` is ``True``, a ``SampleResult`` instead.
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")

    was_training = model.training
    model.eval()
    model.to(device)

    if x is None:
        x = torch.rand(shape, device=device)
    else:
        x = x.to(device)

    step_history = [x.detach().cpu()]
    pred_history: list[torch.Tensor] = []

    for i in range(n_steps):
        pred = model(x)
        pred_history.append(pred.detach().cpu())

        mix_factor = 1 / (n_steps - i)
        x = x * (1 - mix_factor) + pred * mix_factor
        step_history.append(x.detach().cpu())

    if was_training:
        model.train()

    if return_intermediates:
        return SampleResult(samples=x, step_history=step_history, pred_history=pred_history)
    return x
