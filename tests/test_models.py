import torch

from dms.models import BasicUNet, count_parameters


def test_output_shape_matches_input_shape():
    net = BasicUNet()
    x = torch.rand(8, 1, 28, 28)
    out = net(x)
    assert out.shape == x.shape


def test_output_shape_with_different_channel_config():
    net = BasicUNet(in_channels=3, out_channels=3)
    x = torch.rand(2, 3, 28, 28)
    out = net(x)
    assert out.shape == x.shape


def test_parameter_count_is_in_expected_ballpark():
    net = BasicUNet()
    n_params = count_parameters(net)
    # The original notebook reports "just over 300,000 parameters".
    assert 280_000 <= n_params <= 330_000


def test_forward_does_not_produce_nans():
    net = BasicUNet()
    x = torch.rand(4, 1, 28, 28)
    out = net(x)
    assert torch.isfinite(out).all()


def test_gradients_flow_to_all_parameters():
    net = BasicUNet()
    x = torch.rand(4, 1, 28, 28, requires_grad=False)
    target = torch.rand(4, 1, 28, 28)
    out = net(x)
    loss = torch.nn.functional.mse_loss(out, target)
    loss.backward()
    for name, param in net.named_parameters():
        assert param.grad is not None, f"No gradient reached parameter {name!r}"
