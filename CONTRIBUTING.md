# Contributing

This is a learning/reference project built in phases (see `PROJECT_PLAN.md`). These conventions keep each phase reviewable on its own.

## Workflow

1. Work phase-by-phase. Don't mix code from Phase *n+1* into a Phase *n* branch/PR.
2. Branch naming: `phase-<n>-<short-description>`, e.g. `phase-2-corruption-and-unet`.
3. Commit messages: prefix with the phase, e.g. `Phase 2: add corrupt() with broadcasting tests`.
4. Open one pull request per phase. The PR description should link back to the relevant section of `PROJECT_PLAN.md`.

## Code style

- Python 3.10+.
- Format with `black`, lint with `ruff` (both listed in `requirements.txt` once Phase 2 adds code).
- Type hints on public functions.
- Every module under `src/dms/` gets at least a smoke test under `tests/` before a phase is considered done.
- Keep notebooks under `notebooks/` for exploration/visualization only — real logic belongs in `src/dms/` so it's importable and testable.

## Adding a new phase

When starting a phase:
1. Re-read its section in `PROJECT_PLAN.md`.
2. If anything is underspecified, expand that section with concrete file names, function signatures, and acceptance criteria before writing code.
3. Implement, test, update `README.md`'s status table.

## Reporting issues

Open a GitHub issue describing what you expected vs. what happened. If it's about the underlying math/theory rather than this repo's code, the [original Hugging Face notebook](https://huggingface.co/learn/diffusion-course/en/unit1/3) and its Discord are the better venue.
