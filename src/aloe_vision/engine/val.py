"""验证/评估入口（CLI）。"""
from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="aloe-vision 验证入口")
    parser.add_argument("--config", default=None, help="模型 YAML（无 --weights 时使用）")
    parser.add_argument("--weights", default=None, help="权重 .pt")
    parser.add_argument("--data", required=True, help="数据集 data.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--project", default="runs/aloe")
    parser.add_argument("--name", default=None)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if not args.weights and not args.config:
        raise SystemExit("需要提供 --weights 或 --config 之一")
    from aloe_vision.registry import register

    register()
    from ultralytics import YOLO

    model = YOLO(args.weights or args.config)
    kwargs = {
        "data": args.data,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "split": args.split,
        "project": args.project,
    }
    if args.device is not None:
        kwargs["device"] = args.device
    if args.name:
        kwargs["name"] = args.name
    metrics = model.val(**kwargs)
    print(f"[aloe-vision] 验证完成: {metrics}")


if __name__ == "__main__":
    main()
