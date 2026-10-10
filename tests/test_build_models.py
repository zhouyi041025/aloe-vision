"""模型构建与结构冒烟测试：验证自定义模块能被 ultralytics YAML 正确解析。

覆盖：
1. 补丁回归——注册自定义模块后，官方结构参数量仍与 ultralytics 官方一致；
2. AloeDet / AloeSeg 的自定义模块数量、参数量、推理路径可运行；
3. 轻量化分割检测头只作用于 aloe-seg 配置。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("ultralytics")

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"

# ultralytics 官方 n 尺度参数量（作为回归基准）
OFFICIAL_PARAMS = {"det": 2_624_080, "seg": 2_876_848}


def _build(name: str):
    from aloe_vision.registry import register

    register()
    from ultralytics import YOLO

    return YOLO(str(CONFIGS / name))


def _iter_impls(model, cls):
    for module in model.model.modules():
        impl = getattr(module, "_aloe_impl", None)
        if isinstance(impl, cls):
            yield impl


def _count_params(model) -> int:
    return sum(p.numel() for p in model.model.parameters())


def _predict_dummy(model):
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    results = model.predict(source=image, imgsz=256, conf=0.1, verbose=False)
    assert len(results) == 1
    return results[0]


def test_baseline_matches_official_params():
    """补丁不得改变官方结构：参数量与 ultralytics 官方一致。"""
    det = _build("yolo11n-baseline.yaml")
    assert _count_params(det) == OFFICIAL_PARAMS["det"]
    seg = _build("yolo11n-seg-baseline.yaml")
    assert _count_params(seg) == OFFICIAL_PARAMS["seg"]


def test_aloe_det_structure_and_forward():
    from aloe_vision.models.attention import MCA
    from aloe_vision.models.blocks import C3k2_Faster
    from aloe_vision.models.neck import BiFPN_Fusion

    model = _build("aloe-det-v11n.yaml")
    assert len(list(_iter_impls(model, MCA))) == 1, "骨干末端应有 1 个 MCA"
    assert len(list(_iter_impls(model, C3k2_Faster))) == 8, "骨干 4 + 颈部 4 个 C3k2-Faster"
    assert len(list(_iter_impls(model, BiFPN_Fusion))) == 4, "BiFPN 应有 4 个加权融合节点"

    params = _count_params(model)
    print(f"[aloe-vision] AloeDet params = {params:,}")
    # 重建版实测 2.32M（相对同类别数基线 −10.4%；论文原始工程口径为 1.89M，见 docs/EXPERIMENTS.md）
    assert 1_500_000 < params < 2_450_000

    result = _predict_dummy(model)
    assert result.boxes is not None


def test_aloe_seg_structure_and_forward():
    from torch import nn

    from aloe_vision.models.attention import DEAttention
    from aloe_vision.models.blocks import C3k2_DCN
    from aloe_vision.models.neck import BiFPN_Fusion

    model = _build("aloe-seg-v11n.yaml")
    assert len(list(_iter_impls(model, DEAttention))) == 5, "骨干末端 1 + 检测头前端 4 个 DE"
    assert len(list(_iter_impls(model, C3k2_DCN))) == 6, "颈部 6 个 C3k2-DCN"
    assert len(list(_iter_impls(model, BiFPN_Fusion))) == 6, "BiFPN 应有 6 个加权融合节点"

    head = model.model.model[-1]
    assert isinstance(head.proto.cv1, nn.Sequential), "分割头 proto 的 3×3 卷积应已被轻量化替换"
    assert any(type(module).__name__ == "PConv" for module in head.modules()), "轻量化分割头应包含 PConv"

    params = _count_params(model)
    print(f"[aloe-vision] AloeSeg params = {params:,}")
    assert 1_800_000 < params < 3_300_000

    # 端到端推理冒烟（随机初始化下可能没有超过阈值的检测，这里只要求推理链路可运行）
    _predict_dummy(model)

    # 直接前向一次，确认分割头（含 proto/mask 分支）参与计算
    model.model.eval()
    with torch.no_grad():
        out = model.model(torch.zeros(1, 3, 256, 256))
    assert out is not None


def test_light_head_only_for_aloe_cfg():
    """官方分割模型不应被轻量化替换。"""
    from torch import nn
    from ultralytics.nn.modules import Conv

    baseline = _build("yolo11n-seg-baseline.yaml")
    assert isinstance(baseline.model.model[-1].proto.cv1, Conv)
    assert not isinstance(baseline.model.model[-1].proto.cv1, nn.Sequential)
