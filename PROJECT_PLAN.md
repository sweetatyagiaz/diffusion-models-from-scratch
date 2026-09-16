# Project Plan

This document records the phased build-out of the project. **All four phases are complete.** Each phase was delivered as its own reviewable step, in order, with tests passing before moving to the next.

---

## Phase 1 — Project documents & scaffolding ✅

**Goal:** have a repo that clearly communicates what it is, how it's organized, and how it will grow, before any model code is written.

**Delivered:**
- `README.md`, `PROJECT_PLAN.md`, `CONTRIBUTING.md`, `LICENSE` (MIT), `requirements.txt`, `.gitignore`
- Empty `src/`, `notebooks/`, `tests/`, `docs/assets/` directories, later populated in Phases 2–4

---

## Phase 2 — Data pipeline, corruption process, and BasicUNet ✅

**Goal:** reproduce the "toy" half of the notebook as importable, tested modules.

**Delivered:**
- `src/dms/data.py` — MNIST/FashionMNIST dataset + `DataLoader` helpers, with a `FakeData` fallback pattern for network-restricted environments (documented in the notebook)
- `src/dms/corrupt.py` — the linear `corrupt(x, amount)` uniform-noise mixing function
- `src/dms/models.py` — `BasicUNet` (3 down-conv / 3 up-conv, skip connections, `MaxPool2d` + `Upsample`); **309,057 parameters**, matching the original notebook's "~300k" claim
- `notebooks/01_corruption_and_model.ipynb` — corruption sweep + model shape checks, executed end-to-end
- 12 unit tests (edge cases, broadcasting, shapes, parameter count)

---

## Phase 3 — Training loop and naive iterative sampler ✅

**Goal:** train `BasicUNet` end-to-end and generate samples using the simple iterative "move partway toward the prediction" sampler.

**Delivered:**
- `src/dms/train.py` — `train_model()` with a `TrainConfig` dataclass, `save_checkpoint`/`load_checkpoint`
- `src/dms/sample.py` — the naive `n_steps` iterative sampler, with `return_intermediates` for trajectory visualization
- `notebooks/02_training_and_sampling.ipynb` — trains `BasicUNet`, plots the loss curve, samples at several step counts, visualizes the denoising trajectory
- 12 new unit tests (24 total), including an overfit-a-single-batch test proving the loop actually reduces loss (not just runs)

---

## Phase 4 — DDPM comparison via `diffusers` ✅

**Goal:** reproduce the second half of the notebook — swapping in `UNet2DModel` + `DDPMScheduler` — and document the conceptual differences from Phases 2–3 directly in code and docs.

**Delivered:**
- `src/dms/ddpm.py` — `build_ddpm_unet`, `build_ddpm_scheduler`, `train_ddpm`, `sample_ddpm`, mirroring the Phase 2–3 interface for direct comparison
- Parameter count check: **1,707,009** (`UNet2DModel`, default config) vs. **309,057** (`BasicUNet`) — matches the notebook's "~1.7M vs. ~300k" almost exactly
- `notebooks/03_ddpm_comparison.ipynb` — trains the DDPM, plots the noise-prediction loss curve, samples at several inference-step counts, visualizes the reverse-diffusion trajectory
- `docs/COMPARISON.md` — point-by-point written comparison: noise type (uniform vs. Gaussian), training objective (predict image vs. predict noise), timestep conditioning (absent vs. present), sampler (ad hoc mixing vs. `scheduler.step`), and model capacity
- 9 new unit tests (33 total across the whole suite)

---

## Known limitation

The environment this repo was built in cannot reach the real MNIST mirrors (both `ossci-datasets.s3.amazonaws.com` and `yann.lecun.com` return HTTP 403 under its network restrictions), so every notebook's committed outputs were produced against `torchvision.datasets.FakeData` rather than real digits. This is called out explicitly in `README.md` and `docs/COMPARISON.md`. The code itself does not have this limitation: `get_dataloader(name="mnist", ...)` will download and use real MNIST automatically in any environment with normal internet access, and every module is unit-tested independently of which dataset is used.

## Suggested next steps beyond this project

Not part of the four planned phases, but natural follow-ups if extending this further:
- Re-run all three notebooks with real MNIST/FashionMNIST for genuine digit samples
- Add class conditioning (generate a specific digit on request)
- Try alternative samplers (DDIM, ancestral sampling with fewer steps) via `diffusers`' other schedulers
- Longer training runs with a learning-rate schedule and a held-out validation split
