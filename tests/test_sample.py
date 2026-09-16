import torch

from dms.models import BasicUNet
from dms.sample import SampleResult, sample


def test_sample_output_shape_matches_requested_shape():
    model = BasicUNet()
    shape = (4, 1, 16, 16)
    out = sample(model, n_steps=3, shape=shape)
    assert out.shape == shape


def test_sample_output_is_finite():
    model = BasicUNet()
    out = sample(model, n_steps=5, shape=(2, 1, 16, 16))
    assert torch.isfinite(out).all()


def test_sample_works_with_a_single_step():
    model = BasicUNet()
    out = sample(model, n_steps=1, shape=(2, 1, 16, 16))
    assert out.shape == (2, 1, 16, 16)
    assert torch.isfinite(out).all()


def test_sample_rejects_zero_or_negative_steps():
    model = BasicUNet()
    for bad_n_steps in (0, -1):
        try:
            sample(model, n_steps=bad_n_steps, shape=(2, 1, 16, 16))
            assert False, f"expected ValueError for n_steps={bad_n_steps}"
        except ValueError:
            pass


def test_sample_accepts_a_custom_starting_point():
    model = BasicUNet()
    x0 = torch.rand(3, 1, 16, 16)
    out = sample(model, n_steps=4, x=x0)
    assert out.shape == x0.shape
    # The output should have moved away from the exact starting point.
    assert not torch.allclose(out, x0)


def test_sample_return_intermediates_has_expected_lengths():
    model = BasicUNet()
    n_steps = 5
    result = sample(model, n_steps=n_steps, shape=(2, 1, 16, 16), return_intermediates=True)

    assert isinstance(result, SampleResult)
    assert result.samples.shape == (2, 1, 16, 16)
    # step_history includes the initial x plus one entry per step
    assert len(result.step_history) == n_steps + 1
    assert len(result.pred_history) == n_steps
    # The last step_history entry should be exactly the returned samples.
    assert torch.allclose(result.step_history[-1], result.samples)


def test_sample_restores_model_training_mode():
    model = BasicUNet()
    model.train()
    sample(model, n_steps=2, shape=(1, 1, 16, 16))
    assert model.training is True

    model.eval()
    sample(model, n_steps=2, shape=(1, 1, 16, 16))
    assert model.training is False
