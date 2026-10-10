"""基础卷积模块（与 ultralytics 的 Conv 语义一致，但保持对 ultralytics 无依赖）。"""
from __future__ import annotations

from math import gcd

import torch.nn as nn


def autopad(k: int, p: int | None = None, d: int = 1) -> int:
    """Same padding 计算（与 ultralytics 保持一致）。"""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    return k // 2 if p is None else p


class Conv(nn.Module):
    """标准卷积：Conv2d + BatchNorm2d + SiLU。"""

    default_act = nn.SiLU()  # 默认激活函数

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """推理期融合 BN 后的前向（用于部署导出）。"""
        return self.act(self.conv(x))


class DWConv(Conv):
    """深度可分离卷积（groups=gcd(c1, c2)）。"""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        super().__init__(c1, c2, k, s, g=gcd(c1, c2), d=d, act=act)
