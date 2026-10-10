# 复现与使用指南

## 1. 环境

```bash
python >= 3.9
pip install -e ".[dev]"        # 或 pip install -r requirements-dev.txt
```

CPU 环境可完成：单元测试、模型构建/前向、合成数据冒烟训练；
真实训练建议 NVIDIA GPU（论文使用 RTX 3090）。

## 2. 跑通流程（合成数据，1 分钟级）

```bash
python -m aloe_vision.data.make_demo_dataset --kind det --out demo_data/det
python -m aloe_vision.engine.train --config configs/aloe-det-v11n.yaml \
    --data demo_data/det/data.yaml --epochs 1 --imgsz 128 --batch 4
```

> 合成数据仅用于验证代码通路，不代表任何真实精度。

## 3. 用自己的数据训练

数据集采用标准 YOLO 格式（检测/分割通用）：

```
datasets/aloe_det/
  images/{train,val}/xxx.jpg
  labels/{train,val}/xxx.txt      # class cx cy w h（归一化）
  data.yaml                       # path / train / val / names
```

```bash
python -m aloe_vision.engine.train --config configs/aloe-det-v11n.yaml --data datasets/aloe_det/data.yaml
python -m aloe_vision.engine.train --config configs/aloe-seg-v11n.yaml --data datasets/aloe_seg/data.yaml
```

训练配置（与论文一致）：300 epochs、SGD(lr0=0.01, momentum=0.937, wd=5e-4)、
检测 batch 32 / 分割 batch 16、Mosaic(+MixUp)。命令未指定的参数可用 `--set k=v` 透传，例如：

```bash
python -m aloe_vision.engine.train --config configs/aloe-det-v11n.yaml \
    --data datasets/aloe_det/data.yaml --epochs 300 --batch 32 --set lr0=0.01 momentum=0.937 weight_decay=0.0005
```

## 4. 评估与推理

```bash
python -m aloe_vision.engine.val --config configs/aloe-det-v11n.yaml --data datasets/aloe_det/data.yaml --weights runs/.../best.pt
python -m aloe_vision.engine.predict --config configs/aloe-seg-v11n.yaml --weights runs/.../best.pt --source path/to/images --count --grade
```

`--count` 输出正常/倒伏/总植株数；`--grade` 对分割结果计算病斑面积占比并给出五级分级。

## 5. 导出部署（对应论文端侧方案）

```bash
python -m aloe_vision.engine.export --weights runs/.../best.pt --format onnx        # → NCNN/TensorRT 的前置
python -m aloe_vision.engine.export --weights runs/.../best.pt --format engine --half  # TensorRT FP16（需 GPU）
```

论文端侧口径：Android（NCNN）450ms/张、Jetson Nano（TensorRT FP16）580ms/张。

## 6. 常见问题

- **CPU 上 DCNv2 报错**：`models/dcn.py` 会自动降级到 grid_sample 参考实现；
  若 torchvision 已实现 CPU deform_conv，则优先走官方算子。
- **自定义模块未注册**：任何按 YAML 构建模型的入口都会先调用 `aloe_vision.registry.register()`，
  自己写脚本时记得先注册再 `YOLO(yaml)`。
- **显存不足**：调小 `--batch` 或 `--imgsz`；P2 检测头会增加显存占用。
