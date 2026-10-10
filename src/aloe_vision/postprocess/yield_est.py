"""减产率预估：基于各病害等级植株数的多元线性回归（论文 5.3.4 节）。

论文基于 50 个田间样方实测数据拟合（R^2 = 0.983）：

    y = x1 + 0.818*x2 + 0.692*x3 + 0.307*x4

x1~x4 分别为健康、轻度、中度、重度等级植株数量；坏死等级项系数不显著、
按 0 处理（不贡献产量）。理想产量按全部植株健康计（系数 1.0，即植株总数）。

减产率 = (理想产量 - 预测产量) / 理想产量（取值 [0, 1]；样方为空时返回 0）。
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

#: 各病害等级的产量系数（论文 5.3.4 节回归公式；坏死项不显著按 0 处理）。
YIELD_COEFS: dict[str, float] = {
    "healthy": 1.0,
    "mild": 0.818,
    "moderate": 0.692,
    "severe": 0.307,
    "necrosis": 0.0,
}

_LEVELS: tuple[str, ...] = tuple(YIELD_COEFS)


def _normalize(counts: Mapping[str, Any]) -> dict[str, float]:
    """校验并补齐五个等级的数量（缺省 0，未知等级报错）。"""
    if not isinstance(counts, Mapping):
        raise TypeError(f"counts 应为 dict，收到 {type(counts).__name__}")
    unknown = sorted(set(counts) - set(YIELD_COEFS))
    if unknown:
        raise ValueError(f"未知的病害等级 {unknown}；可用等级：{list(YIELD_COEFS)}")
    normalized: dict[str, float] = {}
    for level in _LEVELS:
        value = float(counts.get(level, 0))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{level} 的数量必须为非负有限数值，收到 {counts.get(level)!r}")
        normalized[level] = value
    return normalized


def predict_yield(counts: Mapping[str, Any]) -> float:
    """按回归公式预测相对产量（单位：健康株当量）。"""
    normalized = _normalize(counts)
    return float(sum(YIELD_COEFS[level] * normalized[level] for level in _LEVELS))


def ideal_yield(counts: Mapping[str, Any]) -> float:
    """理想产量：全部植株健康（系数 1.0），等于植株总数。"""
    return float(sum(_normalize(counts).values()))


def loss_rate(counts: Mapping[str, Any]) -> float:
    """减产率 = (理想产量 - 预测产量) / 理想产量；理想产量为 0 时返回 0.0。"""
    normalized = _normalize(counts)
    ideal = sum(normalized.values())
    if ideal == 0.0:
        return 0.0
    predicted = sum(YIELD_COEFS[level] * normalized[level] for level in _LEVELS)
    return float((ideal - predicted) / ideal)


def report(counts: Mapping[str, Any]) -> dict:
    """汇总各等级数量、预测产量、理想产量与减产率。"""
    normalized = _normalize(counts)
    predicted = sum(YIELD_COEFS[level] * normalized[level] for level in _LEVELS)
    ideal = sum(normalized.values())
    rate = 0.0 if ideal == 0.0 else (ideal - predicted) / ideal
    return {
        "counts": normalized,
        "predicted_yield": float(predicted),
        "ideal_yield": float(ideal),
        "loss_rate": float(rate),
    }
