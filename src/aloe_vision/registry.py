"""模型注册入口：把 aloe-vision 的自定义模块接入 ultralytics 的 YAML 解析流程。

用法（任何按 YAML 构建模型的入口都需要先注册）：

    from aloe_vision.registry import register
    register()
    from ultralytics import YOLO
    model = YOLO("configs/aloe-det-v11n.yaml")
"""
from __future__ import annotations

_REGISTERED = False


def register() -> None:
    """注册自定义模块（幂等）。"""
    global _REGISTERED
    if _REGISTERED:
        return
    from aloe_vision import compat

    compat.install()
    _REGISTERED = True
