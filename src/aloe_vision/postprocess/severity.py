"""病害五级分级：按病斑面积占比 p（病斑像素 / 叶片面积像素）划分等级。

分级标准来自毕业论文《基于物联网与深度学习的芦荟智慧农业监测系统》：

    健康 healthy   p < 0.1%
    轻度 mild      0.1% <= p < 1%
    中度 moderate  1%   <= p < 5%
    重度 severe    5%   <= p < 15%
    坏死 necrosis  p >= 15%

比例阈值为 0.001 / 0.01 / 0.05 / 0.15；边界点归入更严重的一级
（恰好 0.001 判为轻度、0.01 判为中度、0.05 判为重度、0.15 判为坏死）。
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

try:  # torch 仅用于接受张量输入，缺失时仅支持 numpy 数组
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

#: 五级分级（按严重程度升序）：英文 key、中文标签与比例区间 [min_ratio, max_ratio)。
#: 最高级 necrosis 的 max_ratio=1.0 视为闭区间上界（p 恰好为 1 时归入坏死）。
SEVERITY_LEVELS: tuple[dict[str, Any], ...] = (
    {"key": "healthy", "label": "健康", "min_ratio": 0.0, "max_ratio": 0.001},
    {"key": "mild", "label": "轻度感染", "min_ratio": 0.001, "max_ratio": 0.01},
    {"key": "moderate", "label": "中度感染", "min_ratio": 0.01, "max_ratio": 0.05},
    {"key": "severe", "label": "重度感染", "min_ratio": 0.05, "max_ratio": 0.15},
    {"key": "necrosis", "label": "坏死", "min_ratio": 0.15, "max_ratio": 1.0},
)

#: 分级比例阈值（0.1% / 1% / 5% / 15%），供报告、论文对照等外部引用。
SEVERITY_THRESHOLDS: tuple[float, ...] = (0.001, 0.01, 0.05, 0.15)

_LABELS: dict[str, str] = {level["key"]: level["label"] for level in SEVERITY_LEVELS}


def _check_ratio(ratio: float) -> float:
    """校验并归一化病斑占比（必须为 [0, 1] 内的有限数值）。"""
    value = float(ratio)
    if not math.isfinite(value):
        raise ValueError(f"病斑占比必须为有限数值，收到 {ratio!r}")
    if value < 0.0 or value > 1.0:
        raise ValueError(f"病斑占比必须在 [0, 1] 范围内，收到 {ratio!r}")
    return value


def severity_from_ratio(ratio: float) -> str:
    """按病斑面积占比返回英文等级 key。

    返回 healthy / mild / moderate / severe / necrosis 之一；边界点
    （0.001 / 0.01 / 0.05 / 0.15）归入更严重的一级。非法输入（负数、NaN、
    无穷、大于 1）抛出 ValueError。
    """
    value = _check_ratio(ratio)
    for level in SEVERITY_LEVELS:
        if level["min_ratio"] <= value < level["max_ratio"]:
            return level["key"]
    return SEVERITY_LEVELS[-1]["key"]


def severity_label(ratio: float) -> str:
    """按病斑面积占比返回中文等级标签（健康/轻度感染/中度感染/重度感染/坏死）。"""
    return _LABELS[severity_from_ratio(ratio)]


def _mask_to_numpy(mask: Any) -> np.ndarray:
    """将 numpy 数组 / torch 张量统一为二维 numpy 数组。"""
    if torch is not None and isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    array = np.asarray(mask)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    elif array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"掩码应为二维 (H, W)，实际形状 {array.shape}")
    if array.size == 0:
        raise ValueError("掩码为空（size=0），无法计算病斑占比")
    return array


def lesion_ratio_from_mask(mask: Any, leaf_mask: Any | None = None, leaf_area: float | None = None) -> float:
    """计算病斑像素占比：阳性像素数 / 参考面积像素数。

    参数：
        mask: 病斑掩码，支持 numpy 数组或 torch 张量，取值 0/1 或 0~1 概率图
            （按 > 0.5 二值化）；
        leaf_mask: 可选，叶片区域掩码（与 mask 同尺寸）；提供时以其阳性像素数
            作为分母（论文口径：病斑面积 / 叶片面积）；
        leaf_area: 可选，叶片面积像素数；没有叶片掩码时可用它兜底。

    默认（两者均不提供）以掩码总像素数（整图面积）为分母——该退化口径仅
    适用于拿不到叶片区域、只需要一个粗占比的场景，不等价于论文口径。
    """
    array = _mask_to_numpy(mask)
    positive = int((array > 0.5).sum())
    if leaf_mask is not None:
        leaf = _mask_to_numpy(leaf_mask)
        if leaf.shape != array.shape:
            raise ValueError(f"叶片掩码形状 {leaf.shape} 与病斑掩码 {array.shape} 不一致")
        denominator = float((leaf > 0.5).sum())
    elif leaf_area is not None:
        denominator = float(leaf_area)
    else:
        denominator = float(array.size)
    if denominator <= 0.0:
        return 0.0
    return positive / denominator


def grade_ratio(ratio: float) -> dict:
    """按病斑占比直接分级，返回 {"ratio", "level", "label"}。

    与 :func:`severity_from_ratio` 的严格校验不同，这里把超出 [0, 1] 的
    占比（例如掩码重叠导致的略大于 1）截断到边界后再分级，便于业务侧
    直接使用模型输出。
    """
    value = min(1.0, max(0.0, float(ratio)))
    return {
        "ratio": value,
        "level": severity_from_ratio(value),
        "label": severity_label(value),
    }


def grade_mask(mask: Any) -> dict:
    """对分割掩码做五级分级，返回 {"ratio": float, "level": str, "label": str}。"""
    return grade_ratio(lesion_ratio_from_mask(mask))


def _demo() -> None:
    """自检演示：打印五级阈值表与一组示例占比的分级结果。"""
    print("病害五级分级阈值（病斑面积占比 p，来自毕业论文）：")
    for level in SEVERITY_LEVELS:
        lo, hi = level["min_ratio"], level["max_ratio"]
        if level["key"] == "healthy":
            rule = f"p < {hi * 100:g}%"
        elif level["key"] == "necrosis":
            rule = f"p >= {lo * 100:g}%"
        else:
            rule = f"{lo * 100:g}% <= p < {hi * 100:g}%"
        print(f"  {level['key']:<9} {level['label']}  {rule}")

    print("\n示例占比分级：")
    for ratio in (0.0005, 0.001, 0.005, 0.01, 0.03, 0.05, 0.10, 0.15, 0.30):
        print(f"{ratio:g} -> {severity_from_ratio(ratio)}（{severity_label(ratio)}）")


if __name__ == "__main__":
    _demo()
