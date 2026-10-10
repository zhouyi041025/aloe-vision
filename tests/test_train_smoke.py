"""端到端训练冒烟（slow）：合成数据 1 epoch 训练 + 验证。

默认排除于 CI；本机验证命令：
    python -m pytest tests/test_train_smoke.py -q -m slow
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("ultralytics")
pytestmark = pytest.mark.slow

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


def test_det_smoke_train(tmp_path):
    from aloe_vision.data.make_demo_dataset import make_det_dataset
    from aloe_vision.registry import register

    register()
    from ultralytics import YOLO

    data_dir = tmp_path / "det"
    make_det_dataset(data_dir, n_train=8, n_val=4, imgsz=64, seed=0)

    model = YOLO(str(CONFIGS / "aloe-det-v11n.yaml"))
    model.train(
        data=str(data_dir / "data.yaml"),
        epochs=1,
        imgsz=64,
        batch=4,
        workers=0,
        project=str(tmp_path / "runs"),
        name="smoke",
        exist_ok=True,
        verbose=False,
        plots=False,
        deterministic=True,
    )
    metrics = model.val(
        data=str(data_dir / "data.yaml"),
        imgsz=64,
        batch=4,
        project=str(tmp_path / "runs"),
        name="smoke-val",
        verbose=False,
    )
    assert metrics is not None


def test_seg_smoke_train(tmp_path):
    from aloe_vision.data.make_demo_dataset import make_seg_dataset
    from aloe_vision.registry import register

    register()
    from ultralytics import YOLO

    data_dir = tmp_path / "seg"
    make_seg_dataset(data_dir, n_train=8, n_val=4, imgsz=64, seed=0)

    model = YOLO(str(CONFIGS / "aloe-seg-v11n.yaml"))
    model.train(
        data=str(data_dir / "data.yaml"),
        epochs=1,
        imgsz=64,
        batch=4,
        workers=0,
        project=str(tmp_path / "runs"),
        name="smoke",
        exist_ok=True,
        verbose=False,
        plots=False,
        deterministic=True,
    )
    metrics = model.val(
        data=str(data_dir / "data.yaml"),
        imgsz=64,
        batch=4,
        project=str(tmp_path / "runs"),
        name="smoke-val",
        verbose=False,
    )
    assert metrics is not None
