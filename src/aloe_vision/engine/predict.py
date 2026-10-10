"""推理入口（CLI）：支持植株计数与病斑分级汇总。"""
from __future__ import annotations

import argparse
import json

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="aloe-vision 推理入口")
    parser.add_argument("--config", default=None, help="模型 YAML（可选，用于注册自定义模块）")
    parser.add_argument("--weights", required=True, help="权重 .pt")
    parser.add_argument("--source", required=True, help="图片/目录/视频/摄像头")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save", action="store_true", help="保存可视化结果")
    parser.add_argument("--count", action="store_true", help="按检测结果统计正常/倒伏/总植株数")
    parser.add_argument("--grade", action="store_true", help="按分割掩码统计病斑面积占比与五级分级")
    parser.add_argument("--leaf-area", type=int, default=None, help="叶片面积像素数（无叶片类掩码时的分母兜底）")
    return parser


def _to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def _grade_from_result(result, names, leaf_area=None) -> dict:
    """把一张图的分割结果汇总为病斑面积占比与五级分级。

    - 类别名含 leaf/叶片 的掩码并集作为叶片面积；
    - 类别名含 lesion/病斑 的掩码并集作为病斑面积；模型没有病斑类时，
      把结果中出现过的全部类别视为病斑（单类别分割模型的常见情形）；
    - 拿不到叶片区域且未提供 ``--leaf-area`` 时返回 skipped 提示。
    """
    from aloe_vision.postprocess.severity import grade_ratio, lesion_ratio_from_mask

    masks = _to_numpy(result.masks.data) > 0.5
    classes = _to_numpy(result.boxes.cls).astype(int) if result.boxes is not None else np.zeros(masks.shape[0], dtype=int)

    def _match(key: str) -> list[int]:
        matched = []
        for cid, name in (names or {}).items():
            low = str(name).lower()
            if key in low or (key == "leaf" and "叶" in low) or (key == "lesion" and "病斑" in low):
                matched.append(int(cid))
        return matched

    def _union(ids: list[int]):
        picked = [masks[classes == cid] for cid in ids if bool((classes == cid).any())]
        if not picked:
            return None
        return np.any(np.concatenate(picked, axis=0), axis=0)

    leaf_ids = _match("leaf")
    lesion_ids = _match("lesion") or sorted({int(c) for c in classes})
    if not lesion_ids:
        return {"skipped": True, "reason": "结果中没有可用的分割掩码"}

    lesion_mask = _union(lesion_ids)
    leaf_mask = _union(leaf_ids) if leaf_ids else None
    lesion_pixels = 0 if lesion_mask is None else int(lesion_mask.sum())
    if leaf_mask is None and leaf_area is None:
        return {
            "skipped": True,
            "reason": "模型无叶片类掩码且未提供 --leaf-area，无法按病斑/叶片面积比分級",
            "lesion_pixels": lesion_pixels,
        }
    ratio = 0.0 if lesion_mask is None else lesion_ratio_from_mask(lesion_mask, leaf_mask=leaf_mask, leaf_area=leaf_area)
    graded = grade_ratio(ratio)
    return {"ratio": graded["ratio"], "level": graded["level"], "label": graded["label"]}


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    from aloe_vision.registry import register

    register()
    from ultralytics import YOLO

    model = YOLO(args.weights)
    kwargs = {"source": args.source, "imgsz": args.imgsz, "conf": args.conf, "save": args.save}
    if args.device is not None:
        kwargs["device"] = args.device
    results = model.predict(**kwargs)

    summary = []
    for idx, result in enumerate(results):
        item = {"index": idx, "source": str(getattr(result, "path", idx))}
        if args.count and getattr(result, "boxes", None) is not None:
            from aloe_vision.postprocess.counting import count_plants

            boxes = _to_numpy(result.boxes.xyxy)
            scores = _to_numpy(result.boxes.conf)
            classes = _to_numpy(result.boxes.cls)
            item["plants"] = count_plants(
                {"boxes": boxes, "scores": scores, "classes": classes},
                num_classes=2,
                conf_thres=args.conf,
            )
        if args.grade and getattr(result, "masks", None) is not None:
            item["grade"] = _grade_from_result(result, getattr(model, "names", {}), args.leaf_area)
        summary.append(item)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
