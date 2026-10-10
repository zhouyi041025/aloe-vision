"""合成演示数据集生成器单元测试（仅依赖 numpy/PIL/pyyaml）。"""
from __future__ import annotations

from pathlib import Path

import yaml

from aloe_vision.data.make_demo_dataset import make_det_dataset, make_seg_dataset


def _read_label_rows(path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(line.split())
    return rows


def test_make_det_dataset_structure_labels_and_yaml(tmp_path: Path) -> None:
    out = make_det_dataset(tmp_path / "det", n_train=3, n_val=2, imgsz=96, seed=1)

    for split, expected in (("train", 3), ("val", 2)):
        image_stems = {p.stem for p in (out / "images" / split).glob("*.png")}
        label_stems = {p.stem for p in (out / "labels" / split).glob("*.txt")}
        assert len(image_stems) == expected
        assert image_stems == label_stems

    rows = _read_label_rows(out / "labels" / "train" / "img_000.txt")
    assert rows, "每张演示图至少应有一个检测标注"
    for row in rows:
        assert len(row) == 5
        assert int(row[0]) in (0, 1)
        cx, cy, w, h = (float(value) for value in row[1:])
        assert 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0
        assert 0.0 < w <= 1.0 and 0.0 < h <= 1.0

    data = yaml.safe_load((out / "data.yaml").read_text(encoding="utf-8"))
    assert data["names"] == {0: "normal", 1: "lodged"}
    assert data["train"] == "images/train"
    assert data["val"] == "images/val"
    assert Path(data["path"]) == out.resolve()


def test_make_seg_dataset_polygon_labels(tmp_path: Path) -> None:
    out = make_seg_dataset(tmp_path / "seg", n_train=2, n_val=1, imgsz=96, seed=2)

    assert len(list((out / "images" / "train").glob("*.png"))) == 2
    assert len(list((out / "images" / "val").glob("*.png"))) == 1

    rows = _read_label_rows(out / "labels" / "train" / "img_000.txt")
    assert rows, "每张演示图至少应有一个病斑标注"
    for row in rows:
        assert row[0] == "0"  # 唯一类别：lesion
        coords = [float(value) for value in row[1:]]
        assert len(coords) % 2 == 0
        assert 8 <= len(coords) // 2 <= 16
        assert all(0.0 <= value <= 1.0 for value in coords)

    raw = (out / "data.yaml").read_text(encoding="utf-8")
    assert "segment" in raw
    data = yaml.safe_load(raw)
    assert data["names"] == {0: "lesion"}


def test_demo_dataset_is_reproducible_with_same_seed(tmp_path: Path) -> None:
    first = make_det_dataset(tmp_path / "a", n_train=1, n_val=1, imgsz=64, seed=7)
    second = make_det_dataset(tmp_path / "b", n_train=1, n_val=1, imgsz=64, seed=7)
    label_first = (first / "labels" / "train" / "img_000.txt").read_text(encoding="utf-8")
    label_second = (second / "labels" / "train" / "img_000.txt").read_text(encoding="utf-8")
    assert label_first == label_second
