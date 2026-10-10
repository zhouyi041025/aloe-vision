"""DCNv2 可变形卷积（含 CPU 可用降级实现）。

优先调用 torchvision 的 deform_conv2d；当当前设备/后端不支持时（例如 CPU 版
torchvision 未编译可变形卷积），自动降级为基于 grid_sample 的等价参考实现，
保证任何环境都能前向/反向。降级实现只支持 groups=1。
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

try:  # torchvision 在 CPU 平台上可能没有可变形卷积后端
    from torchvision.ops import deform_conv2d as _tv_deform_conv2d
except Exception:  # pragma: no cover
    _tv_deform_conv2d = None

__all__ = ["DeformConv2d"]


def _pair(value):
    return (value, value) if isinstance(value, int) else tuple(value)


class DeformConv2d(nn.Module):
    """DCNv2 可变形卷积层。

    Args:
        c1: 输入通道；c2: 输出通道；k: 卷积核尺寸；s: 步长；p: 填充（默认 same）；
        d: 空洞；groups: 分组（降级实现仅支持 1）；bias: 是否使用偏置；
        v2: 是否启用 modulation（DCNv2 的调制机制）。
    """

    def __init__(self, c1, c2, k=3, s=1, p=None, d=1, groups=1, bias=False, v2=True):
        super().__init__()
        self.k = _pair(k)
        self.stride = _pair(s)
        self.dilation = _pair(d)
        self.groups = int(groups)
        self.v2 = bool(v2)
        kh, kw = self.k
        dh, dw = self.dilation
        if p is None:
            p = (dh * (kh - 1) // 2, dw * (kw - 1) // 2)
        self.padding = _pair(p)

        self.weight = nn.Parameter(torch.empty(c2, c1 // self.groups, kh, kw))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.zeros(c2))
        else:
            self.register_parameter("bias", None)

        # offset / modulation 分支零初始化，保证训练初期行为接近标准卷积
        self.offset_conv = nn.Conv2d(c1, 2 * kh * kw * self.groups, self.k, self.stride, self.padding, self.dilation, bias=True)
        self.mask_conv = (
            nn.Conv2d(c1, kh * kw * self.groups, self.k, self.stride, self.padding, self.dilation, bias=True)
            if self.v2
            else None
        )
        for conv in (self.offset_conv, self.mask_conv):
            if conv is not None:
                nn.init.zeros_(conv.weight)
                nn.init.zeros_(conv.bias)

        self._tv_ok = _tv_deform_conv2d is not None

    def forward(self, x):
        offset = self.offset_conv(x)
        # 注意：torchvision 的 deform_conv2d 直接乘 mask（内部不做 sigmoid），
        # 因此这里统一先对 modulation 分支做 sigmoid，再交给两条实现路径。
        mask = torch.sigmoid(self.mask_conv(x)) if self.mask_conv is not None else None
        if self._tv_ok:
            try:
                return _tv_deform_conv2d(
                    x, offset, self.weight, self.bias, self.stride, self.padding, self.dilation, mask
                )
            except (RuntimeError, NotImplementedError):
                # 当前后端不支持（如 CPU 版 torchvision），切换降级实现
                self._tv_ok = False
        if self.groups > 1:
            raise NotImplementedError("grid_sample 降级实现仅支持 groups=1；请使用支持可变形卷积的 torchvision 后端")
        return self._fallback(x, offset, mask)

    def _fallback(self, x, offset, mask):
        """基于 grid_sample 的参考实现（逐卷积核采样点累加）。"""
        b, _, h, w = x.shape
        kh, kw = self.k
        sh, sw = self.stride
        ph, pw = self.padding
        dh, dw = self.dilation
        oh = (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1
        ow = (w + 2 * pw - dw * (kw - 1) - 1) // sw + 1

        ys = torch.arange(oh, device=x.device, dtype=x.dtype) * sh - ph
        xs = torch.arange(ow, device=x.device, dtype=x.dtype) * sw - pw
        out = torch.zeros(b, self.weight.shape[0], oh, ow, device=x.device, dtype=x.dtype)

        denom_w = max(w - 1, 1)
        denom_h = max(h - 1, 1)
        for i in range(kh):
            for j in range(kw):
                tap = i * kw + j
                # torchvision 约定：offset 通道逐采样点交错排列（2*tap 为 y、2*tap+1 为 x）
                off_y = offset[:, 2 * tap]
                off_x = offset[:, 2 * tap + 1]
                yy = ys.view(1, oh, 1) + i * dh + off_y
                xx = xs.view(1, 1, ow) + j * dw + off_x
                grid = torch.stack([xx / denom_w * 2 - 1, yy / denom_h * 2 - 1], dim=-1)
                sampled = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
                if mask is not None:
                    sampled = sampled * mask[:, tap].unsqueeze(1)
                out = out + torch.einsum("oc,bchw->bohw", self.weight[:, :, i, j], sampled)
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)
        return out

    def extra_repr(self) -> str:
        return (
            f"c1={self.weight.shape[1] * self.groups}, c2={self.weight.shape[0]}, k={self.k}, "
            f"stride={self.stride}, padding={self.padding}, groups={self.groups}, v2={self.v2}"
        )
