# aloe-vision · 芦荟植株检测与炭疽病斑分割

> 本科毕业设计《基于物联网与深度学习的芦荟智慧农业监测系统》核心算法部分的开源整理版：
> 面向芦荟大田场景的 **植株检测计数**（改进 YOLOv11n）与 **炭疽病斑分割分级**（改进 YOLOv11n-seg），
> 包含模型结构、训练/评估/推理管线与业务后处理（计数、五级分级、减产率预估）。

[![ci](https://github.com/zhouyi041025/aloe-vision/actions/workflows/ci.yml/badge.svg)](https://github.com/zhouyi041025/aloe-vision/actions/workflows/ci.yml)

## 它解决什么问题

芦荟大田种植中，人工巡检效率低、病害发现晚、产量损失难以量化。本仓库给出论文中两个专用轻量化模型的
**可复现代码实现**：

- **AloeDet（改进 YOLOv11n）**：检测正常 / 倒伏植株并计数，为产量预估提供基础数据；
- **AloeSeg（改进 YOLOv11n-seg）**：像素级分割炭疽病斑，并按病斑面积占比做五级病害分级。

## 结果速览（论文原始实验数据）

| 模型 | 关键指标 | 基线 | 改进后 | 参数量/体积 |
| --- | --- | --- | --- | --- |
| AloeDet | mAP@50 | 94.0% | **94.5%**（精确率 +0.6%，召回率 +1.7%） | 2.58M → **1.89M**（−26.7%），FLOPs −4.8% |
| AloeSeg | mask mAP / 病斑分割精确率 | 74.1% / 77.2% | **76.4% / 82.5%**（+3.4 / +5.3 个百分点） | 参数 −27.9%，体积 6.0MB → **4.6MB** |
| 病害分级 | 五级分级准确率 | — | **92.3%** | 病斑面积占比阈值法 |
| 减产率预估 | 预测相对误差 | — | **3.2%**（R² = 0.983） | 多元线性回归 |

> 以上数字来自毕业论文与答辩材料（自建数据集、RTX 3090 训练），**未在本仓库重新训练验证**；
> 参数量/计算量等与数据无关的指标已在本仓库实测核对（见 `docs/EXPERIMENTS.md`）。

### 本仓库重建版实测（CPU，imgsz=640）

| 模型 | 参数量 | GFLOPs | CPU 时延 (ms/图) |
| --- | --- | --- | --- |
| AloeDet（重建） | 2.32M | 5.7 | 51.0 |
| └ 对照：YOLO11n 基线（nc=80） | 2.62M | 6.7 | 33.6 |
| AloeSeg（重建） | 2.56M | 8.9 | 134.8 |
| └ 对照：YOLO11n-seg 基线（nc=80） | 2.88M | 10.0 | 42.7 |

复测：`python -m aloe_vision.engine.benchmark --config configs/aloe-det-v11n.yaml`。
重建版相对基线：检测参数量 −11.5%、计算量 −14.9%；分割参数量 −11.0%、计算量 −11.0%。
与论文原始工程口径（−26.7% / −27.9%）的差异及原因见 `docs/EXPERIMENTS.md` 第 5 节
（原始网络结构图未保留，本仓库按论文文字描述重建）。

## 模型改进一览

**AloeDet（植株检测与计数）**

| 改进 | 位置 | 作用 |
| --- | --- | --- |
| MCA 多维协作注意力 | 骨干末端 | 通道/高度/宽度三维并行注意力，增强密集小目标植株特征 |
| BiFPN 加权双向特征融合 | 颈部（替换 PANet） | fast normalized fusion，强化多尺度融合、保留浅层细节 |
| C3k2-Faster 轻量化模块 | 骨干（替换 C3k2） | PConv + 1×1 组合替代 3×3 卷积，大幅降参数量与 FLOPs |

**AloeSeg（病斑分割与分级）**

| 改进 | 位置 | 作用 |
| --- | --- | --- |
| DE 注意力（自研） | 骨干末端 + 各分割检测头前端 | 双路径通道注意力 + 局部对比度空间注意力，突出微小病斑边缘 |
| DCNv2 可变形卷积 | 颈部各 C3k2 模块内 | 动态采样点，适配不规则病斑形状 |
| P2 小目标检测头 | BiFPN 颈部新增 | 四检测头（P2–P5），解决毫米级病斑漏检 |
| 轻量化分割检测头 | 分割头 | 部分卷积替换 3×3 卷积，抵消 P2 带来的参数增长 |

## 快速开始

```bash
pip install -e ".[dev]"          # 安装依赖（CPU 环境即可跑通冒烟测试）

# 1) 生成合成演示数据（仅用于跑通流程，非真实数据）
python -m aloe_vision.data.make_demo_dataset --kind det --out demo_data/det
python -m aloe_vision.data.make_demo_dataset --kind seg --out demo_data/seg

# 2) 冒烟训练（1 epoch，小图）
python -m aloe_vision.engine.train --config configs/aloe-det-v11n.yaml --data demo_data/det/data.yaml --epochs 1 --imgsz 128

# 3) 推理 + 病害分级演示
python -m aloe_vision.engine.predict --config configs/aloe-det-v11n.yaml --source demo_data/det/images/val --weights runs/detect/train/weights/best.pt
python -m aloe_vision.postprocess.severity            # 分级规则自检
```

真实数据训练见 [docs/REPRODUCE.md](docs/REPRODUCE.md)。

## 目录结构

```
configs/                  # 两个改进模型的 YAML（ultralytics 兼容）
src/aloe_vision/
  models/                 # MCA / DE 注意力、C3k2-Faster、DCNv2、BiFPN 融合、轻量化分割头
  engine/                 # train / val / predict / benchmark
  data/                   # 合成演示数据集与数据规范
  postprocess/            # 植株计数、五级分级、减产率预估
tests/                    # 单元测试 + 模型构建/前向冒烟测试
docs/                     # 设计说明、实验记录、复现指南
```

其中 [docs/SYSTEM.md](docs/SYSTEM.md) 是毕业论文中**物联网系统部分**（硬件选型、
LoRa/4G 组网、平台层、Web / Android / 边缘端应用与系统测试）的整理文档。

## 关于本仓库的说明（重要）

- 原始毕设工程代码未保留，本仓库是**按论文文字描述与公开文献重建**的实现：
  MCA 参考论文官方实现重写（见 `docs/DESIGN.md` 引用），DE 注意力为论文自研模块的等价重构。
- 训练数据（自建芦荟数据集）与训练权重未开源；论文精度数字无法在本仓库直接复现，
  但流程可通过合成数据完整跑通，替换真实数据即可训练。
- 训练框架依赖 [ultralytics](https://github.com/ultralytics/ultralytics)（AGPL-3.0）；
  本仓库的模型模块本身为纯 PyTorch 实现，协议为 MIT。

## 局限与后续

- 数据集单一（广州增城、库拉索芦荟），跨地域/品种泛化未验证；
- 病害种类仅覆盖炭疽病，后续可扩展至根腐病、叶枯病等；
- 论文中的物联网链路与 Web/Android/边缘端工程代码未开源，已按论文整理为
  [docs/SYSTEM.md](docs/SYSTEM.md) 文档；本仓库聚焦视觉算法核心。

## 致谢

- 论文指导：华南农业大学 岳学军 教授；数据采集场地：广州增城健桥芦荟种植基地；
- MCA 模块参考：*MCA: Multidimensional Collaborative Attention*（ndsclark/MCANet）；
  BiFPN 参考 EfficientDet；DCNv2 参考 Deformable ConvNets v2。

MIT License · © 2026 周意（Zhou Yi）
