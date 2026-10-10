"""模型基准测试：参数量、FLOPs（可选）与推理时延。

- count_params / count_flops / measure_latency：通用函数，接受任意 torch 模型
  （也接受 ultralytics 的包装对象，会自动取内部 module）；
- summarize_yaml：通过 YAML 配置构建模型（先注册 aloe_vision 自定义模块），
  汇总参数量、FLOPs 与 CPU 单图时延；
- thop 与 ultralytics 均为可选依赖：thop 缺失时 FLOPs 返回 None，
  ultralytics / registry 缺失时抛出带说明的错误。

命令行：

    python -m aloe_vision.engine.benchmark --config configs/aloe-det-v11n.yaml --imgsz 640
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch


def _unwrap(model: Any) -> Any:
    """取 ultralytics 包装对象内部的 torch 模块。

    - YOLO 等包装对象（非 nn.Module）→ 返回其内部 `model`（DetectionModel/SegmentationModel）；
    - 已经是 nn.Module 的对象（含 DetectionModel 本身）→ 原样返回，避免二次下钻到内部分层。
    """
    if isinstance(model, torch.nn.Module):
        return model
    return getattr(model, "model", model)


def count_params(model: Any) -> int:
    """统计模型参数量（int）。"""
    module = _unwrap(model)
    return int(sum(parameter.numel() for parameter in module.parameters()))


def count_flops(model: Any, imgsz: int = 640) -> int | None:
    """用 thop 估算 FLOPs；thop 未安装或估算失败时返回 None（非硬依赖）。"""
    try:
        from thop import profile  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        module = _unwrap(model)
        module.eval()
        dummy = torch.randn(1, 3, imgsz, imgsz)
        flops, _ = profile(module, inputs=(dummy,), verbose=False)
        return int(flops)
    except Exception:  # pragma: no cover - thop 对个别结构可能不支持
        return None


def measure_latency(model: Any, imgsz: int = 640, runs: int = 20, warmup: int = 5, device: str = "cpu") -> float:
    """测量单图平均推理时延（毫秒/图，torch.no_grad + perf_counter）。"""
    if runs < 1:
        raise ValueError(f"runs 必须 >= 1，收到 {runs}")
    if warmup < 0:
        raise ValueError(f"warmup 不能为负数，收到 {warmup}")
    module = _unwrap(model)
    module.eval()
    module.to(device)
    dummy = torch.randn(1, 3, imgsz, imgsz, device=device)
    cuda = device.startswith("cuda") and torch.cuda.is_available()
    with torch.no_grad():
        for _ in range(warmup):
            module(dummy)
        if cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(runs):
            module(dummy)
        if cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    return elapsed * 1000.0 / runs


def summarize_yaml(yaml_path: str | Path, imgsz: int = 640) -> dict:
    """构建 YAML 配置的模型并汇总参数量 / FLOPs / CPU 时延。

    依赖 aloe_vision.registry（注册自定义模块）与 ultralytics；
    两者不可用时抛出带说明的错误。
    """
    try:
        from aloe_vision.registry import register
    except ImportError as exc:
        raise RuntimeError(
            "未找到 aloe_vision.registry：无法注册自定义模块（MCA/C3k2-Faster 等），请确认仓库版本完整"
        ) from exc
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("需要安装 ultralytics 才能通过 YAML 构建模型：pip install ultralytics") from exc

    register()
    model = YOLO(str(yaml_path))
    module = _unwrap(model)
    params = count_params(module)
    return {
        "yaml": str(yaml_path),
        "params": params,
        "params_m": round(params / 1e6, 2),
        "flops": count_flops(module, imgsz=imgsz),
        "latency_ms": round(measure_latency(module, imgsz=imgsz), 2),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="aloe-vision 模型基准测试（参数量 / FLOPs / 时延）")
    parser.add_argument("--config", required=True, help="模型 YAML 配置文件，如 configs/aloe-det-v11n.yaml")
    parser.add_argument("--imgsz", type=int, default=640, help="推理输入尺寸（默认 640）")
    args = parser.parse_args(argv)
    summary = summarize_yaml(args.config, imgsz=args.imgsz)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
