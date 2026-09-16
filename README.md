# Diffusion Models from Scratch

"""
A from-scratch, educational reimplementation of the [Hugging Face Diffusion Course — "Diffusion Models from Scratch"](https://huggingface.co/learn/diffusion-course/en/unit1/3) notebook, restructured as a proper Python project instead of a single notebook.
"""

The goal is to build a **toy diffusion model** end-to-end (corruption process → minimal UNet → training loop → sampling), then compare it against a **DDPM-style implementation** using `diffusers` (`UNet2DModel`, `DDPMScheduler`), so the differences in design choices are visible in code, not just in theory.

## Why this project exists

The original material is a single Colab notebook. This repo breaks it into:
- reusable, testable Python modules instead of notebook cells,
- a package that can be installed and imported (`import dms`),
- a clear phase-by-phase build order so each concept (corruption, model, training, sampling, DDPM comparison) can be reviewed and understood independently,
- a place to extend the toy model later (FashionMNIST, class conditioning, longer training, alternative samplers).

## Project status

**All four phases are complete.** This project fully reproduces the [original notebook](https://huggingface.co/learn/diffusion-course/en/unit1/3) as a tested, importable package: the toy corruption/BasicUNet/naive-sampler pipeline, and a `diffusers`-based DDPM (`UNet2DModel` + `DDPMScheduler`) for direct comparison.

| Phase | Scope | Status |
|-------|-------|--------|
| 1 | Repo scaffolding & project documents | ✅ Done |
| 2 | Data pipeline + corruption process (`corrupt`) + BasicUNet | ✅ Done |
| 3 | Training loop + naive iterative sampler | ✅ Done |
| 4 | `diffusers`-based DDPM comparison (`UNet2DModel`, `DDPMScheduler`) | ✅ Done |

33 unit tests pass (`pytest`), covering the corruption process, `BasicUNet`, the training loop, the naive sampler, and the full DDPM wrapper (`build_ddpm_unet`, `build_ddpm_scheduler`, `train_ddpm`, `sample_ddpm`).

> **Note on sample quality:** the environment this repo was built in can't reach the real MNIST mirrors (network restrictions), so the committed notebook outputs were produced against `torchvision.datasets.FakeData` (random noise) rather than real digits. The pipeline is fully verified end-to-end for both models -- data loading, corruption/noising, training loop, checkpointing, and both samplers all run, and training loss provably decreases for both (see `tests/test_train.py::test_overfitting_a_single_batch_reduces_loss`) -- but the sample images in `docs/assets/` only show noise converging toward a blurred average, not digit shapes, because there's no digit structure in FakeData to learn. Re-run the notebooks somewhere with normal internet access to get real MNIST-trained samples that actually look like digits.

See [`PROJECT_PLAN.md`](PROJECT_PLAN.md) for the detailed breakdown of each phase, deliverables, and acceptance criteria.

## Project structure

```
diffusion-models-from-scratch/
├── README.md                # You are here
├── PROJECT_PLAN.md           # Phase-by-phase build history
├── CONTRIBUTING.md           # How to work on this repo
├── LICENSE                   # MIT
├── requirements.txt          # Pinned-ish dependencies
├── pyproject.toml            # Installable `dms` package (src layout)
├── .gitignore
├── src/
│   └── dms/                  # "diffusion models from scratch" package
│       ├── data.py           # dataset + dataloader helpers (MNIST/FashionMNIST)
│       ├── corrupt.py        # the toy uniform-noise corruption process
│       ├── models.py         # BasicUNet
│       ├── train.py          # training loop for BasicUNet
│       ├── sample.py         # naive iterative sampler
│       └── ddpm.py           # diffusers-based UNet2DModel + DDPMScheduler wrapper
├── notebooks/
│   ├── 01_corruption_and_model.ipynb    # Phase 2: corruption sweep, model shape checks
│   ├── 02_training_and_sampling.ipynb   # Phase 3: train BasicUNet, sample, visualize trajectory
│   └── 03_ddpm_comparison.ipynb         # Phase 4: train the DDPM, sample, visualize trajectory
├── tests/                     # 33 unit tests across all four phases
└── docs/
    ├── COMPARISON.md          # Phase 4: toy model vs. DDPM, point by point
    └── assets/                # Generated plots referenced from this README
```

`checkpoints/` is created by the notebooks at run time (git-ignored) to hold trained model weights.

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

## The DDPM comparison (Phase 4)

```python
from dms.data import get_dataloader
from dms.ddpm import DDPMTrainConfig, build_ddpm_scheduler, build_ddpm_unet, sample_ddpm, train_ddpm

dataloader = get_dataloader(name="mnist", root="data", batch_size=64)
ddpm_net = build_ddpm_unet()                  # ~1.7M params, vs. BasicUNet's ~309k
scheduler = build_ddpm_scheduler(num_train_timesteps=1000)

result = train_ddpm(ddpm_net, scheduler, dataloader, DDPMTrainConfig(epochs=3))
generated = sample_ddpm(ddpm_net, scheduler, shape=(8, 1, 28, 28), num_inference_steps=200)
```

See `notebooks/03_ddpm_comparison.ipynb` for the full walkthrough, and [`docs/COMPARISON.md`](docs/COMPARISON.md) for a point-by-point comparison of every difference between this and the toy model above: noise type, training objective, timestep conditioning, sampler, and model capacity.

## Reference

- Original notebook: [Diffusion Models from Scratch — Hugging Face Diffusion Course](https://huggingface.co/learn/diffusion-course/en/unit1/3)
- Written by Jonathan Whitaker for the Hugging Face Diffusion Course; overlaps with his own course, ["The Generative Landscape"](https://johnowhitaker.github.io/tglcourse/dm1.html).
- Background reading: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597), [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364).

## License

MIT — see [`LICENSE`](LICENSE).
