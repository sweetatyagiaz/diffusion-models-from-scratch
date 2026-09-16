# Notebook: Diffusion Models Course


In this free course, you will:
- 👩‍🎓 Study the theory behind diffusion models
- 🧨 Learn how to generate images and audio with the popular 🤗 Diffusers library
- 🏋️‍♂️ Train your own diffusion models from scratch
- 📻 Fine-tune existing diffusion models on new datasets
- 🗺 Explore conditional generation and guidance
- 🧑‍🔬 Create your own custom diffusion model pipelines

A from-scratch, educational reimplementation of the [Hugging Face Diffusion Course — "Diffusion Models from Scratch"](https://huggingface.co/learn/diffusion-course/en/unit1/3) notebook, restructured as a proper Python project instead of a single notebook.

The goal is to build a **toy diffusion model** end-to-end (corruption process → minimal UNet → training loop → sampling), then compare it against a **DDPM-style implementation** using `diffusers` (`UNet2DModel`, `DDPMScheduler`), so the differences in design choices are visible in code, not just in theory.