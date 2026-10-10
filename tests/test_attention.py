"""MCA / DEAttention 单元测试：形状、梯度、边界与错误处理。"""
from __future__ import annotations

import pytest
import torch

from aloe_vision.models.attention import MCA, DEAttention


@pytest.mark.parametrize("shape", [(2, 32, 16, 16), (1, 64, 24, 40)])
def test_mca_shape(shape):
    module = MCA(shape[1])
    out = module(torch.randn(*shape))
    assert out.shape == shape


def test_mca_no_spatial_variant():
    module = MCA(32, no_spatial=True)
    assert not hasattr(module, "c_hw")
    assert module(torch.randn(2, 32, 8, 8)).shape == (2, 32, 8, 8)


def test_mca_channel_mismatch():
    with pytest.raises(ValueError, match="通道保持"):
        MCA(32, 64)


def test_mca_backward():
    module = MCA(16)
    x = torch.randn(2, 16, 8, 8, requires_grad=True)
    module(x).mean().backward()
    assert x.grad is not None
    assert all(p.grad is not None for p in module.parameters())


def test_mca_deterministic():
    module = MCA(16).eval()
    x = torch.randn(1, 16, 8, 8)
    with torch.no_grad():
        assert torch.equal(module(x), module(x))


@pytest.mark.parametrize("channels", [3, 8, 16, 48])
def test_de_shape(channels):
    module = DEAttention(channels)
    out = module(torch.randn(2, channels, 16, 16))
    assert out.shape == (2, channels, 16, 16)


def test_de_small_input():
    module = DEAttention(8, contrast_kernel=3, kernel_size=3)
    assert module(torch.randn(1, 8, 4, 4)).shape == (1, 8, 4, 4)


def test_de_channel_mismatch():
    with pytest.raises(ValueError, match="通道保持"):
        DEAttention(16, 32)


def test_de_backward():
    module = DEAttention(24)
    x = torch.randn(2, 24, 12, 12, requires_grad=True)
    module(x).mean().backward()
    assert x.grad is not None
    assert all(p.grad is not None for p in module.parameters())
