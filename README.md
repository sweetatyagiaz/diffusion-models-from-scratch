# Diffusion Models Course


In this free course, you will:
- 👩‍🎓 Study the theory behind diffusion models
- 🧨 Learn how to generate images and audio with the popular 🤗 Diffusers library
- 🏋️‍♂️ Train your own diffusion models from scratch
- 📻 Fine-tune existing diffusion models on new datasets
- 🗺 Explore conditional generation and guidance
- 🧑‍🔬 Create your own custom diffusion model pipelines

A from-scratch, educational reimplementation of the [Hugging Face Diffusion Course — "Diffusion Models from Scratch"](https://huggingface.co/learn/diffusion-course/en/unit1/3) notebook, restructured as a proper Python project instead of a single notebook.

The goal is to build a **toy diffusion model** end-to-end (corruption process → minimal UNet → training loop → sampling), then compare it against a **DDPM-style implementation** using `diffusers` (`UNet2DModel`, `DDPMScheduler`), so the differences in design choices are visible in code, not just in theory.

## Why this project exists

The original material is a single Colab notebook. This repo breaks it into:
- reusable, testable Python modules instead of notebook cells,
- a package that can be installed and imported (`import dms`),
- a clear phase-by-phase build order so each concept (corruption, model, training, sampling, DDPM comparison) can be reviewed and understood independently,
- a place to extend the toy model later (FashionMNIST, class conditioning, longer training, alternative samplers).

## Project status

This repository is being built in phases. **We are currently in Phase 1: project setup and documentation.**

| Phase | Scope | Status |
|-------|-------|--------|
| 1 | Repo scaffolding & project documents (this phase) | ✅ In progress |
| 2 | Data pipeline + corruption process (`corrupt`) + BasicUNet | ⏳ Planned |
| 3 | Training loop + naive iterative sampler + experiment tracking | ⏳ Planned |
| 4 | `diffusers`-based DDPM comparison (`UNet2DModel`, `DDPMScheduler`, noise/timestep conditioning, samplers) | ⏳ Planned |

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

## Getting started (once Phase 2 lands)

```bash
git clone <your-fork-url>
cd diffusion-models-from-scratch
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Usage instructions will be filled in as each phase adds real code.

## Reference

"""
- Original notebook: [Diffusion Models from Scratch — Hugging Face Diffusion Course](https://huggingface.co/learn/diffusion-course/en/unit1/3)
- Written by Jonathan Whitaker for the Hugging Face Diffusion Course; overlaps with his own course, ["The Generative Landscape"](https://johnowhitaker.github.io/tglcourse/dm1.html). 
"""
- Background reading: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597), [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364).

## License

MIT — see [`LICENSE`](LICENSE).
