"""轻量化卷积模块与可变形卷积瓶颈（论文 5.3.2 / 5.3.3 的改进点）。

- PConv / Bottleneck_Faster / C3k2_Faster：FasterNet 风格部分卷积 + 1×1 组合，
  替换传统 3×3 卷积，降低参数量与 FLOPs（检测模型骨干/颈部）。
- Bottleneck_DCN / C3k2_DCN：在颈部瓶颈内引入 DCNv2 可变形卷积，动态贴合
  不规则病斑形状（分割模型颈部）。

结构习惯与 ultralytics 的 C2f/C3k2 保持一致（外层 cv1/cv2 + n 个内部瓶颈），
便于在 YOLO 系列 YAML 中直接替换使用。
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from aloe_vision.models.conv import Conv
from aloe_vision.models.dcn import DeformConv2d

__all__ = ["PConv", "Bottleneck_Faster", "C3k2_Faster", "Bottleneck_DCN", "C3k2_DCN"]


class PConv(nn.Module):
    """FasterNet 风格部分卷积（通道保持）。

    仅前 1/n_div 通道参与 k×k 卷积，其余通道恒等直通（步长 >1 时做平均池化对齐），
    显著减少 FLOPs 与内存访问。
    """

    def __init__(self, c1: int, c2: int | None = None, k: int = 3, s: int = 1, n_div: int = 4, act=True):
        super().__init__()
        if c2 is not None and c2 != c1:
            raise ValueError(f"PConv 为通道保持模块，要求 c2 == c1，收到 c1={c1}, c2={c2}")
        if n_div < 1:
            raise ValueError(f"n_div 必须 >= 1，收到 {n_div}")
        self.partial = max(1, c1 // n_div)
        self.stride = int(s)
        self.conv = Conv(self.partial, self.partial, k, s, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.partial
        x1, x2 = x[:, :p], x[:, p:]
        y1 = self.conv(x1)
        if x2.shape[1] == 0:
            return y1
        if self.stride > 1:
            x2 = F.avg_pool2d(x2, self.stride, self.stride)
        return torch.cat([y1, x2], dim=1)


class Bottleneck_Faster(nn.Module):
    """轻量化瓶颈：Conv1×1 → (PConv + Conv1×1)，替代传统 3×3 卷积瓶颈。"""

    def __init__(self, c1: int, c2: int, shortcut: bool = True, e: float = 0.5, n_div: int = 4):
        super().__init__()
        c_ = max(1, int(c2 * e))
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Sequential(PConv(c_, k=3, n_div=n_div), Conv(c_, c2, 1))
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C3k2_Faster(nn.Module):
    """C3k2 结构的轻量化版本（内部瓶颈替换为 Bottleneck_Faster）。

    参数布局与 ultralytics C3k2 一致：c1, c2, n, c3k=False, e=0.5, g=1, shortcut=True。
    c3k=True 时用两个串联的 Bottleneck_Faster 近似 C3k 的双瓶颈语义。
    """

    def __init__(self, c1: int, c2: int, n: int = 1, c3k: bool = False, e: float = 0.5, g: int = 1, shortcut: bool = True):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            nn.Sequential(
                Bottleneck_Faster(self.c, self.c, shortcut),
                Bottleneck_Faster(self.c, self.c, shortcut),
            )
            if c3k
            else Bottleneck_Faster(self.c, self.c, shortcut)
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class Bottleneck_DCN(nn.Module):
    """含 DCNv2 可变形卷积的瓶颈：Conv1×1 → DCNv2(3×3) → BN → SiLU。"""

    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, e: float = 0.5, v2: bool = True, n_div: int = 4):
        super().__init__()
        c_ = max(1, int(c2 * e))
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Sequential(
            DeformConv2d(c_, c2, k=3, s=1, groups=g, v2=v2),
            nn.BatchNorm2d(c2),
            nn.SiLU(),
        )
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C3k2_DCN(nn.Module):
    """C3k2 结构 + DCNv2 瓶颈（分割模型颈部，论文 5.3.3 第 2 点）。参数布局同 C3k2。"""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        g: int = 1,
        shortcut: bool = True,
        v2: bool = True,
    ):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            nn.Sequential(
                Bottleneck_DCN(self.c, self.c, shortcut, g, v2=v2),
                Bottleneck_DCN(self.c, self.c, shortcut, g, v2=v2),
            )
            if c3k
            else Bottleneck_DCN(self.c, self.c, shortcut, g, v2=v2)
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
