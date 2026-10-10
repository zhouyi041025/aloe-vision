"""BiFPN 加权融合模块（论文 5.3.2 第 2 点）。

ultralytics 的 YAML 是线性模块表，无法直接表达 BiFPN 的多输入加权求和，
因此这里采用"标准 Concat + 加权融合"的组合实现 fast normalized fusion：

    out = Σ_i w'_i · x_i,   w'_i = relu(w_i) / (Σ_j relu(w_j) + ε)

放在 Concat 之后按通道分组求和，数学上等价于各分支的加权融合（EfficientDet）。
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["BiFPN_Fusion"]


class BiFPN_Fusion(nn.Module):
    """BiFPN fast normalized fusion（置于 Concat 之后）。

    Args:
        c1: 输入通道（= 拼接后通道数，应等于 c2 × n_branches）。
        c2: 输出通道（= 单分支通道数）。
        n_branches: 参与融合的分支数。
        eps: 归一化数值稳定项。
    """

    def __init__(self, c1: int, c2: int | None = None, n_branches: int = 2, eps: float = 1e-4):
        super().__init__()
        if c2 is None:
            raise ValueError("BiFPN_Fusion 需要显式指定输出通道 c2")
        if n_branches < 2:
            raise ValueError(f"n_branches 必须 >= 2，收到 {n_branches}")
        if c1 != c2 * n_branches:
            raise ValueError(
                f"BiFPN_Fusion 要求输入通道 c1 == c2 × n_branches，收到 c1={c1}, c2={c2}, "
                f"n_branches={n_branches}；请检查上游 Concat 的分支通道是否对齐。"
            )
        self.c1, self.c2 = c1, c2
        self.n_branches = n_branches
        self.eps = eps
        self.weights = nn.Parameter(torch.ones(n_branches))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunks = x.chunk(self.n_branches, dim=1)
        weight = F.relu(self.weights)
        weight = weight / (weight.sum() + self.eps)
        out = chunks[0] * weight[0]
        for i in range(1, len(chunks)):
            out = out + chunks[i] * weight[i]
        return out

    def extra_repr(self) -> str:
        return f"c1={self.c1}, c2={self.c2}, n_branches={self.n_branches}"
