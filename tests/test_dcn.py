"""DeformConv2d 单元测试：形状、梯度、零偏移与标准卷积的一致性、降级路径。"""
from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from aloe_vision.models.dcn import DeformConv2d


def test_shape_and_backward():
    module = DeformConv2d(16, 24, k=3)
    x = torch.randn(2, 16, 32, 32, requires_grad=True)
    out = module(x)
    assert out.shape == (2, 24, 32, 32)
    out.mean().backward()
    assert x.grad is not None
    assert module.weight.grad is not None
    assert module.offset_conv.weight.grad is not None


def test_stride_and_padding():
    module = DeformConv2d(4, 8, k=3, s=2, p=1, v2=False)
    assert module(torch.randn(1, 4, 15, 15)).shape == (1, 8, 8, 8)


def test_zero_offset_equals_standard_conv():
    """v2=False 且 offset=0 时，可变形卷积退化为标准卷积（显式走降级实现）。"""
    torch.manual_seed(0)
    module = DeformConv2d(6, 6, k=3, v2=False, bias=True)
    module._tv_ok = False  # 强制走 grid_sample 降级路径
    x = torch.randn(2, 6, 12, 12)
    reference = F.conv2d(x, module.weight, module.bias, stride=1, padding=1)
    out = module(x)
    assert torch.allclose(out, reference, atol=1e-4)


def test_zero_offset_equals_standard_conv_torchvision_path():
    """同上一测试，但走 torchvision 快速路径（本机不支持时跳过）。"""
    torch.manual_seed(0)
    module = DeformConv2d(6, 6, k=3, v2=False, bias=True)
    x = torch.randn(2, 6, 12, 12)
    out = module(x)
    if not module._tv_ok:
        pytest.skip("torchvision deform_conv2d 在本机不可用")
    reference = F.conv2d(x, module.weight, module.bias, stride=1, padding=1)
    assert torch.allclose(out, reference, atol=1e-4)


def test_v2_modulation_forward():
    module = DeformConv2d(4, 4, k=3, v2=True)
    out = module(torch.randn(1, 4, 8, 8))
    assert out.shape == (1, 4, 8, 8)
    assert module.mask_conv is not None


def test_torchvision_vs_fallback_when_supported():
    """若本机 torchvision 支持可变形卷积，则两条路径结果应一致；否则跳过。"""
    module = DeformConv2d(4, 4, k=3)
    x = torch.randn(1, 4, 10, 10)
    with torch.no_grad():
        module.offset_conv.weight.normal_(0, 0.05)
        module.offset_conv.bias.normal_(0, 0.05)
    y_default = module(x).detach()
    if not module._tv_ok:
        pytest.skip("torchvision deform_conv2d 在本机不可用，已走 grid_sample 降级实现")

    fallback = DeformConv2d(4, 4, k=3)
    fallback.load_state_dict(module.state_dict())
    fallback._tv_ok = False
    with torch.no_grad():
        y_fallback = fallback(x)
    assert torch.allclose(y_default, y_fallback, atol=1e-3)
