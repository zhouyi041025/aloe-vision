"""postprocess 模块单元测试：五级分级、植株计数与减产率预估。"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from aloe_vision.postprocess.counting import count_plants
from aloe_vision.postprocess.severity import (
    SEVERITY_LEVELS,
    grade_mask,
    lesion_ratio_from_mask,
    severity_from_ratio,
    severity_label,
)
from aloe_vision.postprocess.yield_est import (
    YIELD_COEFS,
    ideal_yield,
    loss_rate,
    predict_yield,
    report,
)

# ------------------------------ 五级分级 ------------------------------


@pytest.mark.parametrize(
    ("ratio", "expected"),
    [
        (0.0, "healthy"),
        (0.0009, "healthy"),
        (0.001, "mild"),
        (0.009, "mild"),
        (0.01, "moderate"),
        (0.049, "moderate"),
        (0.05, "severe"),
        (0.149, "severe"),
        (0.15, "necrosis"),
        (0.5, "necrosis"),
        (1.0, "necrosis"),
    ],
)
def test_severity_from_ratio_boundaries(ratio: float, expected: str) -> None:
    assert severity_from_ratio(ratio) == expected


def test_severity_levels_are_ordered() -> None:
    assert [level["key"] for level in SEVERITY_LEVELS] == ["healthy", "mild", "moderate", "severe", "necrosis"]
    mins = [level["min_ratio"] for level in SEVERITY_LEVELS]
    assert mins == sorted(mins)
    assert [level["label"] for level in SEVERITY_LEVELS] == ["健康", "轻度感染", "中度感染", "重度感染", "坏死"]


def test_severity_label_chinese() -> None:
    assert severity_label(0.0005) == "健康"
    assert severity_label(0.005) == "轻度感染"
    assert severity_label(0.03) == "中度感染"
    assert severity_label(0.1) == "重度感染"
    assert severity_label(0.3) == "坏死"


@pytest.mark.parametrize("bad", [-0.1, -1.0, 1.01, float("nan"), float("inf")])
def test_severity_from_ratio_invalid(bad: float) -> None:
    with pytest.raises(ValueError):
        severity_from_ratio(bad)


# ------------------------------ 掩码占比与分级 ------------------------------


def test_lesion_ratio_numpy_binary() -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[:1, :5] = 1
    assert lesion_ratio_from_mask(mask) == pytest.approx(0.05)


def test_lesion_ratio_torch_probability_binarizes() -> None:
    probs = torch.full((4, 5), 0.6)
    probs[:, :3] = 0.4  # 12 个低于 0.5 -> 8 个阳性
    assert lesion_ratio_from_mask(probs) == pytest.approx(8 / 20)


def test_lesion_ratio_singleton_channel() -> None:
    mask = torch.zeros(1, 8, 8)
    mask[0, :4, :4] = 1.0
    assert lesion_ratio_from_mask(mask) == pytest.approx(16 / 64)
    stacked = np.zeros((8, 8, 1), dtype=np.float32)
    stacked[:2, :2, 0] = 1.0
    assert lesion_ratio_from_mask(stacked) == pytest.approx(4 / 64)


def test_lesion_ratio_invalid() -> None:
    with pytest.raises(ValueError):
        lesion_ratio_from_mask(np.zeros((2, 2, 2)))
    with pytest.raises(ValueError):
        lesion_ratio_from_mask(np.zeros((0, 0)))


def test_grade_mask_returns_dict() -> None:
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask.reshape(-1)[:50] = 1  # 0.5% -> 轻度
    result = grade_mask(mask)
    assert set(result) == {"ratio", "level", "label"}
    assert result["ratio"] == pytest.approx(0.005)
    assert result["level"] == "mild"
    assert result["label"] == "轻度感染"


# ------------------------------ 植株计数 ------------------------------


def test_count_plants_numpy_filters_low_confidence() -> None:
    detections = {
        "boxes": np.array([[0, 0, 10, 10], [1, 1, 12, 12], [2, 2, 14, 14]], dtype=np.float64),
        "scores": np.array([0.90, 0.24, 0.80]),
        "classes": np.array([0, 1, 1]),
    }
    result = count_plants(detections)
    assert result["normal"] == 1
    assert result["lodged"] == 1
    assert result["other"] == 0
    assert result["total"] == 2


def test_count_plants_torch_tensor_and_other_class() -> None:
    detections = {
        "boxes": torch.zeros(3, 4),
        "scores": torch.tensor([0.9, 0.5, 0.7]),
        "classes": torch.tensor([0, 5, 1]),
    }
    result = count_plants(detections, conf_thres=0.5)
    assert result == {"normal": 1, "lodged": 1, "other": 1, "total": 3}


def test_count_plants_list_of_dicts_with_aliases() -> None:
    detections = [
        {"box": [0, 0, 5, 5], "score": 0.9, "class": 0},
        {"box": [1, 1, 6, 6], "conf": 0.1, "cls": 1},
        {"box": [2, 2, 7, 7], "score": 0.6, "class": -1},
    ]
    result = count_plants(detections)
    assert result["normal"] == 1
    assert result["lodged"] == 0
    assert result["other"] == 1
    assert result["total"] == 2


def test_count_plants_drops_non_finite_scores() -> None:
    detections = {"scores": np.array([np.nan, 0.9]), "classes": np.array([0, 1])}
    result = count_plants(detections)
    assert result["normal"] == 0
    assert result["lodged"] == 1
    assert result["total"] == 1


def test_count_plants_empty_and_mismatch() -> None:
    empty = {"boxes": np.zeros((0, 4)), "scores": np.zeros(0), "classes": np.zeros(0)}
    assert count_plants(empty)["total"] == 0
    with pytest.raises(ValueError):
        count_plants({"scores": np.array([0.9, 0.8]), "classes": np.array([0])})
    with pytest.raises(KeyError):
        count_plants({"scores": np.array([0.9])})


# ------------------------------ 减产率预估 ------------------------------


def test_predict_yield_hand_calculation() -> None:
    counts = {"healthy": 10, "mild": 10, "moderate": 10, "severe": 10}
    assert predict_yield(counts) == pytest.approx(28.17)
    assert ideal_yield(counts) == pytest.approx(40.0)
    assert loss_rate(counts) == pytest.approx((40.0 - 28.17) / 40.0)


def test_loss_rate_boundaries() -> None:
    assert loss_rate({"healthy": 7}) == 0.0
    assert loss_rate({}) == 0.0
    assert loss_rate({"necrosis": 5}) == pytest.approx(1.0)


def test_yield_coefs_and_report() -> None:
    assert set(YIELD_COEFS) == {"healthy", "mild", "moderate", "severe", "necrosis"}
    assert YIELD_COEFS["necrosis"] == 0.0
    result = report({"healthy": 1, "mild": 2})
    assert result["predicted_yield"] == pytest.approx(1.0 + 2 * 0.818)
    assert result["ideal_yield"] == pytest.approx(3.0)
    assert result["loss_rate"] == pytest.approx((3.0 - (1.0 + 2 * 0.818)) / 3.0)
    assert set(result["counts"]) == set(YIELD_COEFS)


def test_yield_rejects_unknown_level_or_negative_count() -> None:
    with pytest.raises(ValueError):
        predict_yield({"normal": 1})
    with pytest.raises(ValueError):
        predict_yield({"healthy": -1})
