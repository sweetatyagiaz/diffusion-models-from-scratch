import torch
from torch.utils.data import DataLoader, TensorDataset

from dms.models import BasicUNet
from dms.train import TrainConfig, TrainResult, load_checkpoint, save_checkpoint, train_model


def _tiny_dataloader(n=32, size=16, batch_size=8):
    """A small synthetic (images, labels) dataloader, shaped like MNIST but
    16x16 and random, so tests run fast without needing a real dataset."""
    images = torch.rand(n, 1, size, size)
    labels = torch.zeros(n, dtype=torch.long)
    return DataLoader(TensorDataset(images, labels), batch_size=batch_size, shuffle=True)


def test_train_model_runs_and_returns_expected_number_of_losses():
    model = BasicUNet()
    dataloader = _tiny_dataloader(n=32, batch_size=8)
    config = TrainConfig(epochs=2, lr=1e-3, log_every=0)

    result = train_model(model, dataloader, config)

    assert isinstance(result, TrainResult)
    # 32 samples / batch_size 8 = 4 steps per epoch, 2 epochs = 8 losses
    assert len(result.losses) == 8
    assert len(result.epoch_mean_losses) == 2


def test_train_model_losses_are_finite():
    model = BasicUNet()
    dataloader = _tiny_dataloader(n=16, batch_size=4)
    result = train_model(model, dataloader, TrainConfig(epochs=1, log_every=0))

    assert all(torch.isfinite(torch.tensor(loss)) for loss in result.losses)


def test_train_model_leaves_model_in_eval_mode():
    model = BasicUNet()
    dataloader = _tiny_dataloader(n=8, batch_size=4)
    train_model(model, dataloader, TrainConfig(epochs=1, log_every=0))

    assert model.training is False


def test_overfitting_a_single_batch_reduces_loss():
    """Not a guarantee for every random seed, but with enough epochs on a
    single small batch, average loss over the second half of training
    should be lower than over the first half."""
    torch.manual_seed(0)
    model = BasicUNet()
    dataloader = _tiny_dataloader(n=8, size=16, batch_size=8)  # a single batch, repeated
    result = train_model(model, dataloader, TrainConfig(epochs=40, lr=1e-3, log_every=0))

    half = len(result.losses) // 2
    first_half_mean = sum(result.losses[:half]) / half
    second_half_mean = sum(result.losses[half:]) / (len(result.losses) - half)
    assert second_half_mean < first_half_mean


def test_save_and_load_checkpoint_roundtrip(tmp_path):
    model = BasicUNet()
    path = tmp_path / "checkpoints" / "model.pt"
    save_checkpoint(model, path)
    assert path.exists()

    new_model = BasicUNet()
    load_checkpoint(new_model, path)

    x = torch.rand(2, 1, 28, 28)
    with torch.no_grad():
        original_out = model(x)
        loaded_out = new_model(x)
    assert torch.allclose(original_out, loaded_out)
