# Toy Model vs. DDPM: A Comparison

This compares the toy diffusion model built in Phases 2–3 (`BasicUNet` +
`corrupt()` + the naive sampler) against the DDPM built in Phase 4
(`diffusers`' `UNet2DModel` + `DDPMScheduler`). Both are trained on the same
kind of data with the same general shape (corrupt an image, ask a model to
undo it, sample by iterating), but nearly every specific choice differs.

## 1. The noise

- **Toy model:** `corrupt(x, amount)` linearly interpolates each pixel
  toward **uniform** random noise (`torch.rand`), by a per-sample `amount`
  in `[0, 1]`.
- **DDPM:** `scheduler.add_noise` mixes in **Gaussian** noise (`torch.randn`),
  scaled and combined according to a fixed noise schedule (`beta_schedule`)
  and an integer timestep in `[0, num_train_timesteps)`.

This isn't cosmetic. The Gaussian case is what the DDPM math (and the closed
forms for the forward/reverse process) is actually derived for. Uniform
noise is a simplification that makes the toy version easy to implement and
reason about, at the cost of not being backed by the same theory.

## 2. What the model predicts

- **Toy model:** the model's output is compared directly against the
  **clean image** `x` (`loss = mse(model(noisy_x), x)`). The model is
  asked to jump straight to its best guess at the fully denoised image.
- **DDPM:** the model's output is compared against the **noise that was
  added** (`loss = mse(model(noisy_x, t), noise)`). This is the standard
  "epsilon-prediction" parameterization used in the original DDPM paper.

Predicting the noise rather than the image is generally easier to learn and
is what makes the reverse-process math work out cleanly, but it means the
loss values between the two models aren't directly comparable -- they're
predicting different targets.

## 3. Timestep conditioning

- **Toy model:** `BasicUNet(noisy_x)` -- the model sees only the corrupted
  image and has to *infer* how corrupted it is from the image itself.
- **DDPM:** `UNet2DModel(noisy_x, timestep)` -- the model is told exactly
  how far along the noise schedule this sample is, via a timestep embedding
  added inside the network.

Explicit timestep conditioning lets the model specialize: its behavior when
asked to remove a small amount of noise can be very different from its
behavior at t=999, without having to infer the noise level from a possibly
ambiguous noisy image.

## 4. Sampling

- **Toy model:** `dms.sample.sample` starts from `torch.rand` noise and, at
  each of `n_steps` steps, moves `1 / (n_steps - i)` of the way from the
  current `x` toward the model's prediction. This rule isn't derived from
  anything -- it's a reasonable-looking heuristic that happens to produce
  progressively better samples.
- **DDPM:** `dms.ddpm.sample_ddpm` starts from `torch.randn` noise and, at
  each scheduled timestep, calls `scheduler.step`, which applies the actual
  reverse-process update implied by the forward noise schedule (predicted
  noise, current timestep, and the schedule's precomputed coefficients all
  feed into a closed-form update, optionally with added noise for
  stochasticity).

## 5. Model capacity

| | `BasicUNet` | `UNet2DModel` (this repo's config) |
|---|---|---|
| Parameters | 309,057 | 1,707,009 |
| Down/up blocks | 3 conv-only stages | 3 stages, attention in the deeper two |
| Skip connections | Yes (simple concatenation) | Yes (ResNet-style blocks) |
| Timestep input | No | Yes |

The DDPM model is about 5.5x larger, mostly due to the attention blocks and
the timestep-embedding pathway that `BasicUNet` doesn't have.

## What stayed the same

Both models are trained on 28x28 grayscale images, both use `Adam`, both are
trained with the `dms.train`/`dms.ddpm` module's near-identical training
loop structure (draw a batch, corrupt it, predict, backprop), and both are
sampled by iterating a denoising step some number of times starting from
noise. The point of this comparison isn't that one approach is "toy" and the
other "real" in every respect -- it's to make the *specific* design
decisions of a real DDPM legible by contrasting each one against a
minimal alternative.

## Caveat on the samples in this repo

The environment this repo was built in can't reach the real MNIST mirrors
(see the note in the top-level `README.md`), so both models here were
trained on `torchvision.datasets.FakeData` (uncorrelated random pixels).
Both training losses decrease as expected, proving the training loops work,
but neither model's *samples* look like digits, because there's no digit
structure in the training data to learn. Re-run
`notebooks/02_training_and_sampling.ipynb` and
`notebooks/03_ddpm_comparison.ipynb` with real MNIST (any environment with
normal internet access) to see the qualitative difference in sample quality
that the DDPM's Gaussian noise, noise-prediction objective, timestep
conditioning, and principled sampler are actually meant to produce.

## Further reading

- Ho et al., [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) -- the original DDPM paper.
- Karras et al., [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364) -- a systematic look at which of these design choices actually matter.
- [`diffusers` documentation](https://huggingface.co/docs/diffusers) for `UNet2DModel` and `DDPMScheduler`.
