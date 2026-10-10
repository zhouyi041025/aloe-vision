"""轻量化卷积模块测试：形状、梯度与参数量对比。"""
from __future__ import annotations

import pytest
import torch
from torch import nn

from aloe_vision.models.blocks import Bottleneck_DCN, Bottleneck_Faster, C3k2_DCN, C3k2_Faster, PConv


def test_pconv_channel_preserving_and_stride():
    assert PConv(16)(torch.randn(1, 16, 8, 8)).shape == (1, 16, 8, 8)
    assert PConv(16, s=2)(torch.randn(1, 16, 8, 8)).shape == (1, 16, 4, 4)


@pytest.mark.parametrize("channels", [3, 17])
def test_pconv_odd_channels(channels):
    assert PConv(channels)(torch.randn(1, channels, 6, 6)).shape == (1, channels, 6, 6)


def test_pconv_channel_mismatch():
    with pytest.raises(ValueError, match="通道保持"):
        PConv(16, 32)


def test_bottleneck_faster_backward():
    module = Bottleneck_Faster(16, 16)
    x = torch.randn(2, 16, 8, 8, requires_grad=True)
    module(x).mean().backward()
    assert x.grad is not None
    assert all(p.grad is not None for p in module.parameters())


def test_c3k2_faster_shapes():
    x = torch.randn(1, 32, 16, 16)
    assert C3k2_Faster(32, 64, n=1).eval()(x).shape == (1, 64, 16, 16)
    assert C3k2_Faster(32, 64, n=2).eval()(x).shape == (1, 64, 16, 16)
    assert C3k2_Faster(32, 64, n=2, c3k=True).eval()(x).shape == (1, 64, 16, 16)


def test_c3k2_dcn_forward_and_backward():
    module = C3k2_DCN(32, 64, n=2)
    x = torch.randn(1, 32, 16, 16, requires_grad=True)
    module(x).mean().backward()
    assert x.grad is not None


class _PlainBottleneck(nn.Module):
    """测试用对照：标准 3×3 卷积瓶颈。"""

    def __init__(self, c1, c2, e=0.5):
        super().__init__()
        c_ = max(1, int(c2 * e))
        self.cv1 = nn.Conv2d(c1, c_, 1)
        self.cv2 = nn.Conv2d(c_, c_, 3, padding=1)

    def forward(self, x):
        return self.cv2(self.cv1(x))


class _PlainC3k2(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        c = int(c2 * e)
        self.cv1 = nn.Conv2d(c1, 2 * c, 1)
        self.cv2 = nn.Conv2d((2 + n) * c, c2, 1)
        self.m = nn.ModuleList(_PlainBottleneck(c, c) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


def test_c3k2_faster_has_fewer_params_than_plain_conv_version():
    fast = C3k2_Faster(32, 64, n=2)
    plain = _PlainC3k2(32, 64, n=2)
    fast_params = sum(p.numel() for p in fast.parameters())
    plain_params = sum(p.numel() for p in plain.parameters())
    assert fast_params < plain_params


def test_bottleneck_dcn_channel_change():
    module = Bottleneck_DCN(16, 16, shortcut=False)
    out = module(torch.randn(1, 16, 8, 8))
    assert out.shape == (1, 16, 8, 8)
