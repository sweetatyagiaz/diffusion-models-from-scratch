# Diffusion Models from Scratch

A from-scratch, educational reimplementation of the [Hugging Face Diffusion Course — "Diffusion Models from Scratch"](https://huggingface.co/learn/diffusion-course/en/unit1/3) notebook, restructured as a proper Python project instead of a single notebook.

The goal is to build a **toy diffusion model** end-to-end (corruption process → minimal UNet → training loop → sampling), then compare it against a **DDPM-style implementation** using `diffusers` (`UNet2DModel`, `DDPMScheduler`), so the differences in design choices are visible in code, not just in theory.

## Why this project exists

The original material is a single Colab notebook. This repo breaks it into:
- reusable, testable Python modules instead of notebook cells,
- a package that can be installed and imported (`import dms`),
- a clear phase-by-phase build order so each concept (corruption, model, training, sampling, DDPM comparison) can be reviewed and understood independently,
- a place to extend the toy model later (FashionMNIST, class conditioning, longer training, alternative samplers).

## Project status

This repository is being built in phases. **We are currently in Phase 3: training loop and naive iterative sampler.**

| Phase | Scope | Status |
|-------|-------|--------|
| 1 | Repo scaffolding & project documents | ✅ Done |
| 2 | Data pipeline + corruption process (`corrupt`) + BasicUNet | ✅ Done |
| 3 | Training loop + naive iterative sampler (this phase) | ✅ Done |
| 4 | `diffusers`-based DDPM comparison (`UNet2DModel`, `DDPMScheduler`, noise/timestep conditioning, samplers) | ⏳ Planned |

> **Note on sample quality:** the environment this repo was built in can't reach the real MNIST mirrors (network restrictions), so the committed notebook outputs were produced against `torchvision.datasets.FakeData` (random noise) rather than real digits. The pipeline is fully verified end-to-end -- data loading, corruption, training loop, checkpointing, and the sampler all run and the loss provably decreases (see `tests/test_train.py::test_overfitting_a_single_batch_reduces_loss`) -- but the sample images in `docs/assets/` only show noise converging to a blurred average, not digit shapes, because there's no digit structure in FakeData to learn. Re-run `notebooks/02_training_and_sampling.ipynb` somewhere with normal internet access to get real MNIST-trained samples.

See [`PROJECT_PLAN.md`](PROJECT_PLAN.md) for the detailed breakdown of each phase, deliverables, and acceptance criteria.

## Planned project structure

```
diffusion-models-from-scratch/
├── README.md                # You are here
├── PROJECT_PLAN.md           # Phase-by-phase roadmap
├── CONTRIBUTING.md           # How to work on this repo
├── LICENSE                   # MIT
├── requirements.txt          # Pinned-ish dependencies
├── .gitignore
├── src/
│   └── dms/                  # "diffusion models from scratch" package (added in Phase 2+)
│       ├── data.py           # dataset + corruption process
│       ├── models.py         # BasicUNet
│       ├── train.py          # training loop
│       └── sample.py         # naive iterative sampler
├── notebooks/                # Exploratory notebooks mirroring the original course notebook
├── tests/                    # Unit tests for corruption, model shapes, training step
└── docs/
    └── assets/                # Generated plots / diagrams for README and docs
```

Folders for future phases (`src/`, `notebooks/`, `tests/`, `docs/assets/`) already exist as empty placeholders (`.gitkeep`) so the structure is visible from Phase 1 onward, even before code is added.

## Getting started

```bash
git clone <your-fork-url>
cd diffusion-models-from-scratch
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .          # installs the `dms` package (src/dms) in editable mode
pytest                     # run the unit tests
jupyter notebook notebooks/01_corruption_and_model.ipynb
```

```python
import torch
from dms.corrupt import corrupt
from dms.models import BasicUNet, count_parameters

net = BasicUNet()
x = torch.rand(8, 1, 28, 28)
amount = torch.linspace(0, 1, 8)

noisy_x = corrupt(x, amount)      # toy linear noising process
pred = net(noisy_x)               # BasicUNet's denoising prediction
print(count_parameters(net))      # ~309,000, matching the original notebook
```

**Note on the dataset:** `dms.data.get_dataloader` downloads MNIST/FashionMNIST via `torchvision`, which needs outbound network access to the dataset mirrors. If that's unavailable (e.g. a sandboxed environment), fall back to `torchvision.datasets.FakeData` for correctly-shaped random data — see `notebooks/01_corruption_and_model.ipynb` for a working example of this fallback.

```python
from dms.data import get_dataloader
from dms.train import TrainConfig, save_checkpoint, train_model
from dms.sample import sample

dataloader = get_dataloader(name="mnist", root="data", batch_size=64)
net = BasicUNet()
result = train_model(net, dataloader, TrainConfig(epochs=3, lr=1e-3))
save_checkpoint(net, "checkpoints/basic_unet.pt")

generated = sample(net, n_steps=40, shape=(8, 1, 28, 28))  # naive iterative sampler
```

See `notebooks/02_training_and_sampling.ipynb` for the full training + sampling walkthrough, including a visualization of the denoising trajectory from pure noise to a final sample.

## Reference

- Original notebook: [Diffusion Models from Scratch — Hugging Face Diffusion Course](https://huggingface.co/learn/diffusion-course/en/unit1/3)
- Written by Jonathan Whitaker for the Hugging Face Diffusion Course; overlaps with his own course, ["The Generative Landscape"](https://johnowhitaker.github.io/tglcourse/dm1.html).
- Background reading: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597), [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364).

## License

MIT — see [`LICENSE`](LICENSE).
