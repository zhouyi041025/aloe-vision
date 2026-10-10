"""训练入口（CLI）。

示例：
    python -m aloe_vision.engine.train --config configs/aloe-det-v11n.yaml \
        --data demo_data/det/data.yaml --epochs 1 --imgsz 128 --batch 4
"""
from __future__ import annotations

import argparse


def _parse_overrides(pairs):
    out = {}
    for item in pairs or []:
        if "=" not in item:
            raise SystemExit(f"--set 需要 k=v 形式，收到: {item}")
        key, value = item.split("=", 1)
        for cast in (int, float):
            try:
                value = cast(value)
                break
            except ValueError:
                continue
        out[key] = value
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="aloe-vision 训练入口（基于 ultralytics）")
    parser.add_argument("--config", required=True, help="模型 YAML，如 configs/aloe-det-v11n.yaml")
    parser.add_argument("--data", required=True, help="数据集 data.yaml")
    parser.add_argument("--epochs", type=int, default=300, help="训练轮数（论文设置 300）")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16, help="检测默认 32 / 分割常用 16")
    parser.add_argument("--device", default=None, help="如 0 / cpu，默认自动选择")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--project", default="runs/aloe")
    parser.add_argument("--name", default=None)
    parser.add_argument("--weights", default=None, help="已有权重 .pt（提供时结构以权重为准）")
    parser.add_argument("--set", nargs="*", default=[], help="透传 ultralytics 参数，如 lr0=0.01 momentum=0.937")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    from aloe_vision.registry import register

    register()
    from ultralytics import YOLO

    if args.weights:
        print("[aloe-vision] 使用已有权重构建模型（--config 仅作记录）。")
        model = YOLO(args.weights)
    else:
        model = YOLO(args.config)

    train_kwargs = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "seed": args.seed,
        "project": args.project,
        "deterministic": True,
    }
    if args.device is not None:
        train_kwargs["device"] = args.device
    if args.name:
        train_kwargs["name"] = args.name
    train_kwargs.update(_parse_overrides(args.set))

    results = model.train(**train_kwargs)
    save_dir = getattr(results, "save_dir", None)
    if save_dir:
        print(f"[aloe-vision] 训练完成，输出目录: {save_dir}")


if __name__ == "__main__":
    main()
