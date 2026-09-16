"""The toy corruption (noising) process.

This is deliberately the simplest thing that could work: linearly mix the
clean input with uniform noise, controlled by a per-sample `amount` in
[0, 1]. At `amount=0` we get the clean input back; at `amount=1` we get
pure noise. Because both `x` and the noise live in [0, 1], the mixture
stays in [0, 1] too.

This is *not* the corruption process used by DDPM (see Phase 4), which
instead scales `x` by `sqrt(alpha_bar_t)` and adds Gaussian noise scaled by
`sqrt(1 - alpha_bar_t)`. Keeping this toy version separate and simple makes
it easier to see exactly what changes when we adopt the DDPM formulation
later.
"""

import torch


def corrupt(x: torch.Tensor, amount: torch.Tensor) -> torch.Tensor:
    """Corrupt a batch of inputs by linearly mixing them with uniform noise.

    Args:
        x: Input tensor of shape ``(batch, channels, height, width)`` with
            values expected to be in ``[0, 1]`` (e.g. an MNIST batch).
        amount: 1D tensor of shape ``(batch,)`` with values in ``[0, 1]``,
            one corruption amount per sample in the batch. ``0`` means
            "no corruption", ``1`` means "pure noise".

    Returns:
        A tensor with the same shape as ``x``, linearly interpolated
        between ``x`` and random noise according to ``amount``.

    Example:
        >>> x = torch.rand(8, 1, 28, 28)
        >>> amount = torch.linspace(0, 1, 8)
        >>> noisy_x = corrupt(x, amount)
        >>> noisy_x.shape
        torch.Size([8, 1, 28, 28])
    """
    if amount.ndim != 1:
        raise ValueError(f"`amount` must be 1D (one value per sample), got shape {tuple(amount.shape)}")
    if amount.shape[0] != x.shape[0]:
        raise ValueError(
            f"`amount` must have one entry per sample in the batch: "
            f"got {amount.shape[0]} amounts for a batch of size {x.shape[0]}"
        )

    noise = torch.rand_like(x)
    amount = amount.view(-1, *([1] * (x.ndim - 1)))  # Reshape for broadcasting, e.g. (B, 1, 1, 1)
    return x * (1 - amount) + noise * amount
