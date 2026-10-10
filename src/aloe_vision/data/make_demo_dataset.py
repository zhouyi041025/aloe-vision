"""生成“合成演示数据集”（synthetic demo only，非真实数据）。

用途：在没有论文原始数据的情况下，跑通训练/评估/推理流程与 CI 冒烟测试。
全部图像与标注由 numpy + PIL 程序合成（固定 seed 可复现）：

- 检测（det）：绿色椭圆 = 正常植株（class 0 "normal"），倾斜深绿椭圆 = 倒伏植株（class 1 "lodged"）；
- 分割（seg）：叶片大椭圆 + 若干黄褐色病斑小椭圆，标注为病斑多边形（class 0 "lesion"）。

输出目录结构（YOLO 格式）：

    <out>/images/{train,val}/*.png
    <out>/labels/{train,val}/*.txt
    <out>/data.yaml

命令行：

    python -m aloe_vision.data.make_demo_dataset --kind det --out demo_data/det
    python -m aloe_vision.data.make_demo_dataset --kind seg --out demo_data/seg

注意：本模块生成的是程序合成的演示数据，仅用于冒烟测试与 CI，
与毕业论文的真实芦荟数据集无关，也不能用于任何精度对比。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

BACKGROUND = (205, 214, 184)     # 浅土色背景
NORMAL_GREEN = (64, 128, 55)     # 正常植株绿色
LODGED_GREEN = (36, 80, 38)      # 倒伏植株深绿
LEAF_GREEN = (82, 145, 66)       # 叶片绿色
LESION_BROWN = (170, 128, 56)    # 病斑黄褐色

DET_NAMES = {0: "normal", 1: "lodged"}
SEG_NAMES = {0: "lesion"}


def _ellipse_points(cx: float, cy: float, rx: float, ry: float, angle_deg: float, n: int) -> np.ndarray:
    """采样旋转椭圆的边界点，返回 (n, 2) 像素坐标数组。"""
    theta = np.deg2rad(angle_deg)
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    xs = rx * np.cos(t)
    ys = ry * np.sin(t)
    x = cx + xs * np.cos(theta) - ys * np.sin(theta)
    y = cy + xs * np.sin(theta) + ys * np.cos(theta)
    return np.stack([x, y], axis=1)


def _draw_points(draw: ImageDraw.ImageDraw, points: np.ndarray, fill: tuple[int, int, int]) -> None:
    draw.polygon([(int(round(x)), int(round(y))) for x, y in points], fill=fill)


def _normalized_bbox(points: np.ndarray, imgsz: int) -> tuple[float, float, float, float]:
    """由像素多边形得到 YOLO 归一化 (cx, cy, w, h)。"""
    clipped = np.clip(points, 0.0, float(imgsz))
    x0, y0 = clipped.min(axis=0)
    x1, y1 = clipped.max(axis=0)
    cx = (x0 + x1) / 2.0 / imgsz
    cy = (y0 + y1) / 2.0 / imgsz
    w = (x1 - x0) / imgsz
    h = (y1 - y0) / imgsz
    return float(cx), float(cy), float(w), float(h)


def _write_dataset_yaml(out_dir: Path, names: dict[int, str], comment: str | None = None) -> None:
    data = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": names,
    }
    text = yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
    if comment:
        text = f"# {comment}\n{text}"
    (out_dir / "data.yaml").write_text(text, encoding="utf-8")


def _prepare_dirs(out_dir: Path) -> None:
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def _write_label(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_det_dataset(out_dir: str | Path, n_train: int = 8, n_val: int = 4, imgsz: int = 128, seed: int = 0) -> Path:
    """生成合成检测数据集（正常/倒伏两类椭圆植株），返回输出目录。"""
    out_dir = Path(out_dir)
    _prepare_dirs(out_dir)
    rng = np.random.default_rng(seed)
    counts = {"train": int(n_train), "val": int(n_val)}
    for split, count in counts.items():
        for index in range(count):
            image = Image.new("RGB", (imgsz, imgsz), BACKGROUND)
            draw = ImageDraw.Draw(image)
            lines: list[str] = []
            for _ in range(int(rng.integers(2, 5))):  # 每图 2~4 株
                rx = float(rng.uniform(0.07, 0.13)) * imgsz
                ry = float(rng.uniform(0.06, 0.11)) * imgsz
                margin = max(rx, ry) + 3.0
                cx = float(rng.uniform(margin, imgsz - margin))
                cy = float(rng.uniform(margin, imgsz - margin))
                lodged = bool(rng.random() < 0.35)
                angle = float(rng.uniform(-65.0, 65.0)) if lodged else float(rng.uniform(-12.0, 12.0))
                color = LODGED_GREEN if lodged else NORMAL_GREEN
                points = _ellipse_points(cx, cy, rx, ry, angle, 32)
                _draw_points(draw, points, color)
                cx_n, cy_n, w, h = _normalized_bbox(points, imgsz)
                cls = 1 if lodged else 0
                lines.append(f"{cls} {cx_n:.4f} {cy_n:.4f} {w:.4f} {h:.4f}")
            name = f"img_{index:03d}"
            image.save(out_dir / "images" / split / f"{name}.png")
            _write_label(out_dir / "labels" / split / f"{name}.txt", lines)
    _write_dataset_yaml(out_dir, DET_NAMES, comment="synthetic demo dataset for smoke tests / CI only")
    print(f"[demo] 已生成【合成演示】检测数据集（非真实数据，仅供冒烟测试/CI）：{out_dir}")
    print(f"[demo] split=train 图像 {counts['train']} 张；split=val 图像 {counts['val']} 张；imgsz={imgsz}，seed={seed}")
    return out_dir


def make_seg_dataset(out_dir: str | Path, n_train: int = 8, n_val: int = 4, imgsz: int = 128, seed: int = 0) -> Path:
    """生成合成分割数据集（叶片 + 黄褐色病斑多边形），返回输出目录。"""
    out_dir = Path(out_dir)
    _prepare_dirs(out_dir)
    rng = np.random.default_rng(seed)
    counts = {"train": int(n_train), "val": int(n_val)}
    for split, count in counts.items():
        for index in range(count):
            image = Image.new("RGB", (imgsz, imgsz), BACKGROUND)
            draw = ImageDraw.Draw(image)

            leaf_rx = float(rng.uniform(0.24, 0.32)) * imgsz
            leaf_ry = float(rng.uniform(0.18, 0.26)) * imgsz
            leaf_angle = float(rng.uniform(-25.0, 25.0))
            leaf_cx = imgsz / 2.0 + float(rng.uniform(-0.04, 0.04)) * imgsz
            leaf_cy = imgsz / 2.0 + float(rng.uniform(-0.04, 0.04)) * imgsz
            _draw_points(draw, _ellipse_points(leaf_cx, leaf_cy, leaf_rx, leaf_ry, leaf_angle, 64), LEAF_GREEN)

            lines: list[str] = []
            for _ in range(int(rng.integers(1, 5))):  # 每图 1~4 个病斑
                lesion_rx = float(rng.uniform(0.03, 0.08)) * imgsz
                lesion_ry = float(rng.uniform(0.03, 0.08)) * imgsz
                lesion_cx = leaf_cx + float(rng.uniform(-0.5, 0.5)) * leaf_rx * 0.8
                lesion_cy = leaf_cy + float(rng.uniform(-0.5, 0.5)) * leaf_ry * 0.8
                lesion_cx = float(np.clip(lesion_cx, lesion_rx + 2.0, imgsz - lesion_rx - 2.0))
                lesion_cy = float(np.clip(lesion_cy, lesion_ry + 2.0, imgsz - lesion_ry - 2.0))
                lesion_angle = float(rng.uniform(0.0, 180.0))
                n_points = int(rng.integers(8, 17))  # 8~16 个多边形点
                base = _ellipse_points(lesion_cx, lesion_cy, lesion_rx, lesion_ry, lesion_angle, n_points)
                jitter = 1.0 + rng.uniform(-0.12, 0.12, size=(n_points, 1))
                center = np.array([lesion_cx, lesion_cy])
                points = center + (base - center) * jitter
                _draw_points(draw, points, LESION_BROWN)

                normalized = np.clip(points, 0.0, float(imgsz)) / imgsz
                coords = " ".join(f"{value:.4f}" for value in normalized.reshape(-1))
                lines.append(f"0 {coords}")

            name = f"img_{index:03d}"
            image.save(out_dir / "images" / split / f"{name}.png")
            _write_label(out_dir / "labels" / split / f"{name}.txt", lines)

    _write_dataset_yaml(
        out_dir,
        SEG_NAMES,
        comment="synthetic demo dataset; task: segment (YOLO-seg polygons); smoke tests / CI only",
    )
    print(f"[demo] 已生成【合成演示】分割数据集（非真实数据，仅供冒烟测试/CI）：{out_dir}")
    print(f"[demo] split=train 图像 {counts['train']} 张；split=val 图像 {counts['val']} 张；imgsz={imgsz}，seed={seed}")
    return out_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="生成合成演示数据集（synthetic demo only，非真实数据；用于冒烟测试与 CI）"
    )
    parser.add_argument("--kind", choices=("det", "seg"), required=True, help="det=检测数据集，seg=分割数据集")
    parser.add_argument("--out", required=True, help="输出目录（例如 demo_data/det）")
    parser.add_argument("--n-train", type=int, default=8)
    parser.add_argument("--n-val", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    factory = make_det_dataset if args.kind == "det" else make_seg_dataset
    factory(args.out, n_train=args.n_train, n_val=args.n_val, imgsz=args.imgsz, seed=args.seed)


if __name__ == "__main__":
    main()
