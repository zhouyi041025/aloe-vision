"""BiFPN_Fusion 单元测试：加权融合数学正确性、错误处理与梯度。"""
from __future__ import annotations

import pytest
import torch

from aloe_vision.models.neck import BiFPN_Fusion


def test_equal_weights_behave_like_mean():
    module = BiFPN_Fusion(8, 4)
    a = torch.randn(2, 4, 6, 6)
    b = torch.randn(2, 4, 6, 6)
    out = module(torch.cat([a, b], dim=1))
    assert torch.allclose(out, (a + b) / 2, atol=1e-3)


def test_manual_weights_math():
    module = BiFPN_Fusion(8, 4)
    with torch.no_grad():
        module.weights.copy_(torch.tensor([1.0, 3.0]))
    a = torch.ones(1, 4, 2, 2)
    b = torch.full((1, 4, 2, 2), 5.0)
    out = module(torch.cat([a, b], dim=1))
    expected = (1 * a + 3 * b) / 4
    assert torch.allclose(out, expected, atol=1e-3)


def test_three_branches():
    module = BiFPN_Fusion(12, 4, n_branches=3)
    x = torch.randn(1, 12, 4, 4)
    assert module(x).shape == (1, 4, 4, 4)


@pytest.mark.parametrize(
    ("c1", "c2", "n_branches"),
    [(10, 4, 2), (8, 4, 1), (8, 4, 3)],
)
def test_invalid_arguments(c1, c2, n_branches):
    with pytest.raises(ValueError):
        BiFPN_Fusion(c1, c2, n_branches=n_branches)


def test_missing_c2():
    with pytest.raises(ValueError, match="c2"):
        BiFPN_Fusion(8, None)


def test_backward():
    module = BiFPN_Fusion(8, 4)
    x = torch.randn(1, 8, 4, 4, requires_grad=True)
    module(x).mean().backward()
    assert module.weights.grad is not None
    assert torch.count_nonzero(module.weights.grad) > 0
