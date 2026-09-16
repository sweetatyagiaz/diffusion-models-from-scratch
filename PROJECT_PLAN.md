# Project Plan

This document defines the phased build-out of the project. Each phase is scoped so it can be a self-contained pull request with its own review point before moving on. Phase 1 is fully defined now; Phases 2–4 are defined at a summary level here and will each get a detailed spec (acceptance criteria, file list, test list) written immediately before that phase starts, once Phase 1 is confirmed.

---

## Phase 1 — Project documents & scaffolding (current phase)

**Goal:** have a repo that clearly communicates what it is, how it's organized, and how it will grow, before any model code is written.

**Deliverables:**
- `README.md` — project overview, structure, roadmap, usage stub
- `PROJECT_PLAN.md` — this file
- `CONTRIBUTING.md` — workflow, branch/commit conventions, coding style
- `LICENSE` — MIT
- `requirements.txt` — pinned core dependencies (`torch`, `torchvision`, `diffusers`, `matplotlib`, etc.)
- `.gitignore` — Python, Jupyter, OS, and dataset artifacts
- Empty `src/`, `notebooks/`, `tests/`, `docs/assets/` directories (with `.gitkeep`) so the intended structure is visible immediately

**Acceptance criteria:**
- A newcomer can read `README.md` and `PROJECT_PLAN.md` and understand the goal, the four phases, and where future code will live, without reading any implementation.
- `git init` + first commit produces a clean initial repo history ("Phase 1: project scaffolding and docs").

---

## Phase 2 — Data pipeline, corruption process, and BasicUNet

**Goal:** reproduce the "toy" half of the notebook as importable, tested modules.

**Planned scope:**
- `src/dms/data.py`: MNIST/FashionMNIST dataset loading + `DataLoader` helpers
- `src/dms/corrupt.py`: the linear `corrupt(x, amount)` mixing function, with unit tests for the `amount=0` and `amount=1` edge cases and shape/broadcasting correctness
- `src/dms/models.py`: `BasicUNet` (3 down-conv / 3 up-conv layers, skip connections, `MaxPool2d` + `Upsample`), with a test asserting output shape matches input shape and parameter count is in the expected ballpark (~300k)
- `notebooks/01_corruption_and_model.ipynb`: exploratory notebook visualizing corrupted digits at increasing `amount`, mirroring the original notebook's plots

**Will be detailed further** (file-by-file spec, exact function signatures, test cases) at the start of Phase 2.

---

## Phase 3 — Training loop, naive sampler, and experiment tracking

**Goal:** train `BasicUNet` end-to-end and generate samples using the simple iterative "move partway toward the prediction" sampler.

**Planned scope:**
- `src/dms/train.py`: configurable training loop (epochs, batch size, LR, optimizer), loss logging
- `src/dms/sample.py`: the naive `n_steps` iterative sampler from the notebook, parameterized by number of steps
- `docs/assets/`: saved loss curves and sample grids for a baseline run, referenced from the README
- `tests/`: a fast smoke test that trains for a handful of steps on a tiny subset and checks the loss is finite and decreasing on average
- Optional: a `configs/` folder if runs become parameterized via YAML/CLI args

**Will be detailed further** at the start of Phase 3.

---

## Phase 4 — DDPM comparison via `diffusers`

**Goal:** reproduce the second half of the notebook — swapping in `UNet2DModel` + `DDPMScheduler` — and document the conceptual differences from Phases 2–3 directly in code and docs.

**Planned scope:**
- `src/dms/ddpm.py`: wraps `diffusers.UNet2DModel` and `DDPMScheduler`, exposing the same train/sample interface as Phases 2–3 for direct comparison
- Side-by-side comparison covering: Gaussian (`randn`) vs. uniform (`rand`) noise, noise-prediction vs. denoised-image-prediction objectives, timestep conditioning, and parameter count differences (~1.7M vs. ~300k)
- `notebooks/02_ddpm_comparison.ipynb`: recreates the notebook's side-by-side plots (loss curves, generated digit grids)
- `docs/COMPARISON.md`: a written summary of BasicUNet/naive-sampler vs. `UNet2DModel`/`DDPMScheduler`, with pointers to further reading (e.g., "Elucidating the Design Space of Diffusion-Based Generative Models")

**Will be detailed further** at the start of Phase 4.

---

## How phases will be delivered

Each phase (2, 3, 4) will be proposed as its own step in this conversation once the previous phase is accepted: a short spec first, then the actual files. This keeps changes reviewable instead of dumping the whole implementation at once.
