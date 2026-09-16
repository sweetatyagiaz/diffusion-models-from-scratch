import torch
from torch.utils.data import DataLoader, TensorDataset

from dms.ddpm import (
    DDPMTrainConfig,
    DDPMTrainResult,
    build_ddpm_scheduler,
    build_ddpm_unet,
    sample_ddpm,
    train_ddpm,
)


def _tiny_unet(sample_size=16):
    """A much smaller UNet2DModel than the default, so tests run fast.
    Channel counts must stay divisible by GroupNorm's default 32 groups."""
    return build_ddpm_unet(
        sample_size=sample_size,
        in_channels=1,
        out_channels=1,
        layers_per_block=1,
        block_out_channels=(32, 64),
        down_block_types=("DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D"),
    )


def _tiny_dataloader(n=16, size=16, batch_size=8):
    images = torch.rand(n, 1, size, size)
    labels = torch.zeros(n, dtype=torch.long)
    return DataLoader(TensorDataset(images, labels), batch_size=batch_size, shuffle=True)


def test_build_ddpm_unet_default_param_count_matches_notebook_ballpark():
    net = build_ddpm_unet()  # default sample_size=28, matches the notebook's config
    n_params = sum(p.numel() for p in net.parameters())
    # The original notebook reports ~1.7M for this exact configuration.
    assert 1_000_000 < n_params < 3_000_000


def test_ddpm_unet_forward_pass_preserves_shape():
    net = _tiny_unet(sample_size=16)
    x = torch.rand(2, 1, 16, 16)
    timesteps = torch.randint(0, 1000, (2,)).long()
    out = net(x, timesteps).sample
    assert out.shape == x.shape


def test_build_ddpm_scheduler_defaults():
    scheduler = build_ddpm_scheduler()
    assert scheduler.config.num_train_timesteps == 1000

    scheduler_50 = build_ddpm_scheduler(num_train_timesteps=50)
    assert scheduler_50.config.num_train_timesteps == 50


def test_scheduler_add_noise_preserves_shape():
    scheduler = build_ddpm_scheduler(num_train_timesteps=1000)
    x = torch.rand(4, 1, 16, 16)
    noise = torch.randn_like(x)
    timesteps = torch.randint(0, 1000, (4,)).long()
    noisy_x = scheduler.add_noise(x, noise, timesteps)
    assert noisy_x.shape == x.shape


def test_train_ddpm_runs_and_returns_expected_number_of_losses():
    net = _tiny_unet(sample_size=16)
    scheduler = build_ddpm_scheduler(num_train_timesteps=100)
    dataloader = _tiny_dataloader(n=16, size=16, batch_size=8)

    result = train_ddpm(net, scheduler, dataloader, DDPMTrainConfig(epochs=2, log_every=0))

    assert isinstance(result, DDPMTrainResult)
    assert len(result.losses) == 4  # 2 batches/epoch * 2 epochs
    assert len(result.epoch_mean_losses) == 2
    assert all(torch.isfinite(torch.tensor(loss)) for loss in result.losses)


def test_train_ddpm_leaves_model_in_eval_mode():
    net = _tiny_unet(sample_size=16)
    scheduler = build_ddpm_scheduler(num_train_timesteps=100)
    dataloader = _tiny_dataloader(n=8, size=16, batch_size=4)
    train_ddpm(net, scheduler, dataloader, DDPMTrainConfig(epochs=1, log_every=0))
    assert net.training is False


def test_sample_ddpm_output_shape():
    net = _tiny_unet(sample_size=16)
    scheduler = build_ddpm_scheduler(num_train_timesteps=100)
    shape = (2, 1, 16, 16)
    out = sample_ddpm(net, scheduler, shape=shape, num_inference_steps=5)
    assert out.shape == shape
    assert torch.isfinite(out).all()


def test_sample_ddpm_return_intermediates_has_expected_length():
    net = _tiny_unet(sample_size=16)
    scheduler = build_ddpm_scheduler(num_train_timesteps=100)
    n_steps = 5
    samples, step_history = sample_ddpm(
        net, scheduler, shape=(1, 1, 16, 16), num_inference_steps=n_steps, return_intermediates=True
    )
    assert samples.shape == (1, 1, 16, 16)
    # initial noise + one entry per inference step
    assert len(step_history) == n_steps + 1
    assert torch.allclose(step_history[-1], samples)


def test_sample_ddpm_restores_model_training_mode():
    net = _tiny_unet(sample_size=16)
    scheduler = build_ddpm_scheduler(num_train_timesteps=100)
    net.train()
    sample_ddpm(net, scheduler, shape=(1, 1, 16, 16), num_inference_steps=3)
    assert net.training is True
