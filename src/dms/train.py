"""Training loop for BasicUNet.

Mirrors the notebook's training loop: for each batch, draw a random
per-sample corruption `amount`, corrupt the batch, ask the model to
predict the *clean* image from the corrupted one, and take the MSE loss
against the real clean image. There's no timestep conditioning here --
the model only ever sees the corrupted image, not how corrupted it is --
which is one of the key simplifications Phase 4's DDPM comparison will
remove.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader

from dms.corrupt import corrupt


@dataclass
class TrainConfig:
    """Hyperparameters for a training run."""

    epochs: int = 1
    lr: float = 1e-3
    device: str = "cpu"
    log_every: int = 100
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = field(
        default_factory=lambda: nn.MSELoss()
    )


@dataclass
class TrainResult:
    """What a training run hands back."""

    losses: list[float]
    epoch_mean_losses: list[float]


def train_model(
    model: nn.Module,
    dataloader: Iterable,
    config: TrainConfig | None = None,
) -> TrainResult:
    """Train ``model`` to denoise corrupted batches from ``dataloader``.

    Args:
        model: A model mapping a corrupted image to a predicted clean
            image, e.g. ``BasicUNet``. Trained and left in ``eval()``
            mode when this returns.
        dataloader: Yields ``(images, labels)`` batches; only ``images``
            is used. ``images`` should already be in ``[0, 1]``.
        config: Training hyperparameters. Defaults to
            ``TrainConfig()`` if omitted.

    Returns:
        A ``TrainResult`` with the per-step loss history and the mean
        loss per epoch (useful for a quick "is it learning" check
        without plotting every step).
    """
    config = config or TrainConfig()
    model.to(config.device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    losses: list[float] = []
    epoch_mean_losses: list[float] = []

    for epoch in range(config.epochs):
        epoch_losses: list[float] = []
        for step, (x, _labels) in enumerate(dataloader):
            x = x.to(config.device)
            amount = torch.rand(x.shape[0], device=config.device)
            noisy_x = corrupt(x, amount)

            pred = model(noisy_x)
            loss = config.loss_fn(pred, x)

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
    return TrainResult(losses=losses, epoch_mean_losses=epoch_mean_losses)


def save_checkpoint(model: nn.Module, path: str | Path) -> None:
    """Save just the model weights (not optimizer state) to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model: nn.Module, path: str | Path, device: str = "cpu") -> nn.Module:
    """Load weights from ``path`` into ``model`` in place and return it."""
    state_dict = torch.load(Path(path), map_location=device)
    model.load_state_dict(state_dict)
    return model
