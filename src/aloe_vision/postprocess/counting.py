"""植株计数：对通用格式的检测结果过滤低置信度后按类别统计。

输入不依赖 ultralytics 的 Results 对象，支持两种形式：

- 映射（dict）：{"boxes": (N, 4), "scores": (N,), "classes": (N,)}，
  数组可为 numpy 数组或 torch 张量（也接受 conf/confs、cls/labels 等别名）；
- 检测列表：[{"box": [...], "score": 0.9, "class": 0}, ...]（键允许别名）。

统计规则（num_classes=2，对应论文的“正常/倒伏”两类）：

- class 0 -> "normal"，class 1 -> "lodged"；
- 类别 id 超出 [0, num_classes) -> "other"；
- 置信度非有限值（NaN/Inf）或低于 conf_thres 的检测被过滤；
- "total" 为所有保留检测数之和。
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

try:  # torch 仅用于接受张量输入，缺失时仅支持 numpy 数组
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

_CLASS_NAMES: tuple[str, ...] = ("normal", "lodged")
_SCORE_KEYS: tuple[str, ...] = ("scores", "score", "confs", "conf")
_CLASS_KEYS: tuple[str, ...] = ("classes", "class", "cls", "labels", "label")
_BOX_KEYS: tuple[str, ...] = ("boxes", "box")


def _first_present(mapping: Mapping[str, Any], keys: tuple[str, ...], field: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    raise KeyError(f"检测结果缺少{field}字段（可用键：{'/'.join(keys)}）")


def _to_numpy(value: Any) -> np.ndarray:
    if torch is not None and isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _arrays_from_mapping(detections: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    scores = _to_numpy(_first_present(detections, _SCORE_KEYS, "置信度 scores")).astype(np.float64).reshape(-1)
    classes = _to_numpy(_first_present(detections, _CLASS_KEYS, "类别 classes")).reshape(-1)
    if scores.size != classes.size:
        raise ValueError(f"scores 与 classes 数量不一致：{scores.size} != {classes.size}")
    for key in _BOX_KEYS:
        if key in detections:
            boxes = _to_numpy(detections[key])
            if boxes.shape[0] != scores.size:
                raise ValueError(f"boxes 与 scores 数量不一致：{boxes.shape[0]} != {scores.size}")
            break
    return scores, classes


def _arrays_from_list(detections: list[Any] | tuple[Any, ...]) -> tuple[np.ndarray, np.ndarray]:
    scores: list[float] = []
    classes: list[int] = []
    for index, item in enumerate(detections):
        if not isinstance(item, Mapping):
            raise TypeError(f"检测列表第 {index} 项应为 dict，收到 {type(item).__name__}")
        score = _to_numpy(_first_present(item, _SCORE_KEYS, "置信度 score")).reshape(-1)
        cls = _to_numpy(_first_present(item, _CLASS_KEYS, "类别 class")).reshape(-1)
        if score.size != 1 or cls.size != 1:
            raise ValueError(f"检测列表第 {index} 项的 score/class 应为标量")
        scores.append(float(score[0]))
        classes.append(int(cls[0]))
    return np.asarray(scores, dtype=np.float64), np.asarray(classes, dtype=np.int64)


def count_plants(detections: Any, num_classes: int = 2, conf_thres: float = 0.25) -> dict:
    """统计各类别植株数量。

    参数：
        detections: dict 或 list[dict]，见模块 docstring；
        num_classes: 类别总数（默认 2：0=正常株 normal，1=倒伏株 lodged）；
        conf_thres: 置信度阈值（保留 score >= conf_thres 的检测）。

    返回：
        {"normal": int, "lodged": int, "other": int, "total": int}
        （num_classes > 2 时，0/1 之外的合法类别会追加 "class_<id>" 键。）
    """
    if num_classes < 1:
        raise ValueError(f"num_classes 必须 >= 1，收到 {num_classes}")
    threshold = float(conf_thres)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"conf_thres 必须在 [0, 1] 内，收到 {conf_thres}")

    if isinstance(detections, Mapping):
        scores, classes = _arrays_from_mapping(detections)
    elif isinstance(detections, (list, tuple)):
        scores, classes = _arrays_from_list(detections)
    else:
        raise TypeError(f"detections 应为 dict 或 list[dict]，收到 {type(detections).__name__}")

    keep = np.isfinite(scores) & (scores >= threshold)
    counts: dict[str, int] = {name: 0 for name in _CLASS_NAMES}
    counts["other"] = 0
    for cls in classes[keep]:
        cid = int(cls)
        if 0 <= cid < num_classes:
            if cid < len(_CLASS_NAMES):
                counts[_CLASS_NAMES[cid]] += 1
            else:
                key = f"class_{cid}"
                counts[key] = counts.get(key, 0) + 1
        else:
            counts["other"] += 1
    counts["total"] = sum(counts.values())
    return counts
