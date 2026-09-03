import pytest
import torch

from diffusion_policy_3d.common.pytorch_util import masked_mean


def test_masked_mean_normalizes_only_unconditioned_dimensions():
    values = torch.tensor(
        [
            [[2.0, 4.0], [100.0, 100.0]],
            [[1.0, 3.0], [5.0, 7.0]],
        ]
    )
    loss_mask = torch.tensor(
        [
            [[True, True], [False, False]],
            [[False, True], [False, True]],
        ]
    )

    result = masked_mean(values, loss_mask)
    assert torch.allclose(result, torch.tensor([3.0, 5.0]))


def test_masked_mean_is_finite_when_everything_is_conditioned():
    values = torch.randn(2, 3, 4, requires_grad=True)
    mask = torch.zeros_like(values, dtype=torch.bool)

    result = masked_mean(values, mask)
    assert torch.equal(result, torch.zeros(2))
    result.sum().backward()
    assert torch.equal(values.grad, torch.zeros_like(values))


def test_masked_mean_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="identical shapes"):
        masked_mean(torch.zeros(2, 3, 4), torch.ones(2, 3, dtype=torch.bool))
