"""dms: Diffusion Models from Scratch.

A small, from-scratch educational implementation of a toy diffusion model,
built up in phases. See PROJECT_PLAN.md in the repo root for the roadmap.
"""

from dms.corrupt import corrupt
from dms.data import get_dataloader, get_dataset
from dms.models import BasicUNet, count_parameters

__all__ = [
    "corrupt",
    "get_dataloader",
    "get_dataset",
    "BasicUNet",
    "count_parameters",
]

__version__ = "0.2.0"
