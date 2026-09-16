"""dms: Diffusion Models from Scratch.

A small, from-scratch educational implementation of a toy diffusion model,
built up in phases. See PROJECT_PLAN.md in the repo root for the roadmap.
"""

from dms.corrupt import corrupt
from dms.data import get_dataloader, get_dataset
from dms.ddpm import (
    DDPMTrainConfig,
    DDPMTrainResult,
    build_ddpm_scheduler,
    build_ddpm_unet,
    sample_ddpm,
    train_ddpm,
)
from dms.models import BasicUNet, count_parameters
from dms.sample import SampleResult, sample
from dms.train import TrainConfig, TrainResult, load_checkpoint, save_checkpoint, train_model

__all__ = [
    "corrupt",
    "get_dataloader",
    "get_dataset",
    "BasicUNet",
    "count_parameters",
    "sample",
    "SampleResult",
    "train_model",
    "TrainConfig",
    "TrainResult",
    "save_checkpoint",
    "load_checkpoint",
    "build_ddpm_unet",
    "build_ddpm_scheduler",
    "train_ddpm",
    "sample_ddpm",
    "DDPMTrainConfig",
    "DDPMTrainResult",
]

__version__ = "1.0.0"
