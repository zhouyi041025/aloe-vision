"""注意力模块：MCA（多维协作注意力）与 DE（Disease-Enhanced）注意力。

- MCA：依据论文 *MCA: Multidimensional Collaborative Attention in Deep
  Convolutional Neural Networks for Image Recognition* 的实现逻辑重写
  （与官方实现 ndsclark/MCANet 的 ImageNet/mca_module.py 数学等价）：
  在通道 / 高度 / 宽度三个维度上并行构建注意力门，每个门内使用 avg + std
  两种池化统计，并通过可学习权重进行跨维度特征交互。
  论文用途：插入 YOLOv11n 骨干末端，提升密集小目标植株的特征提取能力。

- DEAttention：毕业论文自研模块的等价实现：双路径通道注意力（全局平均池化 +
  全局最大池化共享 MLP）+ 局部对比度空间注意力（中心-邻域对比突出黄褐色病斑
  边缘），通道与空间协同相乘。论文用途：骨干末端与各分割检测头前端。
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["StdPool", "MCAGate", "MCA", "DEAttention"]


class StdPool(nn.Module):
    """标准差池化：返回每个通道的全局标准差，形状 (B, C, 1, 1)。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        std = x.reshape(b, c, -1).std(dim=2, keepdim=True)
        return std.reshape(b, c, 1, 1)


class MCAGate(nn.Module):
    """MCA 的单个注意力门：avg/std 池化 + 可学习权重 + 1×k 条带卷积。

    Args:
        k_size: 条带卷积核尺寸（沿描述子维度）。
        pool_types: 池化类型组合，支持 "avg" / "max" / "std"。
    """

    def __init__(self, k_size: int, pool_types=("avg", "std")):
        super().__init__()
        pools = []
        for pool_type in pool_types:
            if pool_type == "avg":
                pools.append(nn.AdaptiveAvgPool2d(1))
            elif pool_type == "max":
                pools.append(nn.AdaptiveMaxPool2d(1))
            elif pool_type == "std":
                pools.append(StdPool())
            else:
                raise NotImplementedError(f"不支持的池化类型: {pool_type}")
        self.pools = nn.ModuleList(pools)
        self.conv = nn.Conv2d(1, 1, kernel_size=(1, k_size), stride=1, padding=(0, (k_size - 1) // 2), bias=False)
        self.sigmoid = nn.Sigmoid()
        self.weight = nn.Parameter(torch.rand(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [pool(x) for pool in self.pools]
        if len(feats) == 1:
            out = feats[0]
        elif len(feats) == 2:
            weight = torch.sigmoid(self.weight)
            out = 1 / 2 * (feats[0] + feats[1]) + weight[0] * feats[0] + weight[1] * feats[1]
        else:
            raise ValueError("MCAGate 目前支持 1~2 路池化")
        out = out.permute(0, 3, 2, 1).contiguous()
        out = self.conv(out)
        out = out.permute(0, 3, 2, 1).contiguous()
        out = self.sigmoid(out)
        return x * out.expand_as(x)


class MCA(nn.Module):
    """MCA 多维协作注意力（通道保持）。

    在 ultralytics YAML 中的用法（参数为设计通道数，与输入通道一致）：
        - [-1, 1, MCA, [1024]]
    """

    def __init__(self, c1: int, c2: int | None = None, no_spatial: bool = False):
        super().__init__()
        if c2 is not None and c2 != c1:
            raise ValueError(f"MCA 是通道保持模块，要求 c2 == c1，收到 c1={c1}, c2={c2}")
        if c1 < 1:
            raise ValueError(f"通道数必须为正，收到 c1={c1}")

        # 通道维门的条带卷积核尺寸（与论文公式一致，小通道时兜底为 1）
        temp = round(abs((math.log2(c1) - 1) / 1.5))
        kernel = temp if temp % 2 else temp - 1
        self.kernel_size = max(1, kernel)

        self.h_cw = MCAGate(3)
        self.w_hc = MCAGate(3)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.c_hw = MCAGate(self.kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_h = self.h_cw(x.permute(0, 2, 1, 3).contiguous()).permute(0, 2, 1, 3).contiguous()
        x_w = self.w_hc(x.permute(0, 3, 2, 1).contiguous()).permute(0, 3, 2, 1).contiguous()
        if self.no_spatial:
            return 0.5 * (x_h + x_w)
        x_c = self.c_hw(x)
        return 1 / 3 * (x_c + x_h + x_w)


class DEAttention(nn.Module):
    """DE（Disease-Enhanced）注意力：双路径通道注意力 + 局部对比度空间注意力。

    在 ultralytics YAML 中的用法（参数为设计通道数，与输入通道一致）：
        - [-1, 1, DEAttention, [256]]
    """

    def __init__(self, c1: int, c2: int | None = None, reduction: int = 16, contrast_kernel: int = 5, kernel_size: int = 7):
        super().__init__()
        if c2 is not None and c2 != c1:
            raise ValueError(f"DEAttention 是通道保持模块，要求 c2 == c1，收到 c1={c1}, c2={c2}")
        if c1 < 1:
            raise ValueError(f"通道数必须为正，收到 c1={c1}")
        hidden = max(c1 // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Conv2d(c1, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, c1, 1, bias=False),
        )
        self.contrast_kernel = int(contrast_kernel)
        self.spatial_conv = nn.Conv2d(1, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 双路径通道注意力：全局平均池化 + 全局最大池化，共享 MLP
        avg = self.mlp(F.adaptive_avg_pool2d(x, 1))
        peak = self.mlp(F.adaptive_max_pool2d(x, 1))
        channel_att = self.sigmoid(avg + peak)

        # 局部对比度空间注意力：中心-邻域差分，突出病斑边缘
        local_mean = F.avg_pool2d(x, self.contrast_kernel, stride=1, padding=self.contrast_kernel // 2)
        contrast = (x - local_mean).abs().mean(dim=1, keepdim=True)
        spatial_att = self.sigmoid(self.spatial_conv(contrast))

        return x * channel_att * spatial_att
