import pytest
import torch

from dms.corrupt import corrupt


def test_amount_zero_returns_input_unchanged():
    x = torch.rand(4, 1, 28, 28)
    amount = torch.zeros(4)
    out = corrupt(x, amount)
    assert torch.allclose(out, x)


def test_amount_one_returns_pure_noise_not_input():
    torch.manual_seed(0)
    x = torch.zeros(4, 1, 28, 28)  # distinguishable from noise
    amount = torch.ones(4)
    out = corrupt(x, amount)
    # With amount=1, output should equal the (random) noise, not x.
    assert not torch.allclose(out, x)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_output_shape_matches_input_shape():
    x = torch.rand(8, 1, 28, 28)
    amount = torch.rand(8)
    out = corrupt(x, amount)
    assert out.shape == x.shape


def test_output_values_stay_in_unit_interval():
    x = torch.rand(16, 1, 28, 28)
    amount = torch.rand(16)
    out = corrupt(x, amount)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_broadcasting_matches_manual_computation_with_fixed_noise(monkeypatch):
    x = torch.tensor([[[[0.2, 0.8], [0.4, 0.6]]]])  # shape (1, 1, 2, 2)
    fixed_noise = torch.tensor([[[[1.0, 0.0], [0.5, 0.5]]]])
    amount = torch.tensor([0.25])

    monkeypatch.setattr(torch, "rand_like", lambda t: fixed_noise)

    out = corrupt(x, amount)
    expected = x * 0.75 + fixed_noise * 0.25
    assert torch.allclose(out, expected)


def test_raises_on_mismatched_batch_sizes():
    x = torch.rand(4, 1, 28, 28)
    amount = torch.rand(3)  # wrong batch size
    with pytest.raises(ValueError):
        corrupt(x, amount)


def test_raises_on_non_1d_amount():
    x = torch.rand(4, 1, 28, 28)
    amount = torch.rand(4, 1)  # wrong number of dims
    with pytest.raises(ValueError):
        corrupt(x, amount)
