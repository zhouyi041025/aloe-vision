"""模型导出入口（CLI）：ONNX / TensorRT / TorchScript 等（对应论文端侧部署链路）。"""
from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="aloe-vision 模型导出")
    parser.add_argument("--weights", required=True, help="训练好的权重 .pt")
    parser.add_argument("--format", default="onnx", help="onnx / engine / torchscript / openvino ...")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--half", action="store_true", help="FP16（TensorRT 场景常用）")
    parser.add_argument("--dynamic", action="store_true", help="动态尺寸（ONNX）")
    parser.add_argument("--device", default=None)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    from aloe_vision.registry import register

    register()
    from ultralytics import YOLO

    model = YOLO(args.weights)
    kwargs = {"format": args.format, "imgsz": args.imgsz, "half": args.half, "dynamic": args.dynamic}
    if args.device is not None:
        kwargs["device"] = args.device
    path = model.export(**kwargs)
    print(f"[aloe-vision] 导出完成: {path}")


if __name__ == "__main__":
    main()
