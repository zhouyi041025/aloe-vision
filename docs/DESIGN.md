# 设计说明：AloeDet / AloeSeg 模型与后处理

> 本文档对应毕业论文第 5 章"芦荟专用深度学习模型设计与训练"。
> 原始工程代码未保留，以下为按论文描述重建的实现说明；模块行为以本仓库代码为准。

## 1. 基线与数据集

- 基线：YOLOv11n（检测，2.58M 参数）与 YOLOv11n-seg（实例分割），Ultralytics 框架。
- 植株检测数据集：无人机航拍 1050 张原始图像，LabelImg 标注正常/倒伏两类，7 种增强后扩充到 3150 张，
  按 8:1.6:0.4 划分为 2520/504/126，统一 640×640。
- 炭疽病斑分割数据集：604 张叶片/植株图像，Labelme 多边形精细标注，增强 + Mosaic 扩充到 2416 张，
  划分 1932/387/97，统一 640×640。
- 训练：预训练权重迁移学习；检测 300 epochs / batch 32；分割 300 epochs / batch 16；
  SGD（lr0=0.01，momentum=0.937，weight_decay=5e-4）；Mosaic（+MixUp）增强。

## 2. AloeDet：检测与计数

### 2.1 MCA 多维协作注意力（骨干末端）

动机：高空俯拍下植株小、密集遮挡，单一维度注意力（仅通道或仅空间）对"是什么/在哪里"的建模不足。

MCA 在三个维度上并行构建注意力门（gate），每个门内部使用平均池化与标准差池化两条统计路径，
并通过可学习权重做跨维度特征交互；三个门输出取平均后与原特征相乘：

```
h 门：沿宽度方向卷绕 → 对高度维建模（input 转置后做 1×k 条带卷积）
w 门：沿高度方向卷绕 → 对宽度维建模
c 门：通道描述子（avg+std）→ 1×k 条带卷积（k 由通道数按论文公式取奇数）
out = x * mean(gate_h(x), gate_w(x), gate_c(x))
```

本仓库实现为对论文官方代码（ndsclark/MCANet `ImageNet/mca_module.py`）的等价重写，保持相同数学形式。

### 2.2 BiFPN 加权双向特征融合（替换 PANet）

- 去掉 PANet 中只有单输入的冗余节点，降低计算量；
- 同层输入与输出直接相连，减少特征传递的信息损失；
- 融合采用 fast normalized fusion：`out = Σ w'_i · x_i`，`w'_i = relu(w_i) / (Σ relu(w_j) + ε)`。

在 ultralytics 的模块列表式 YAML 中，本仓库用"标准 Concat + `BiFPN_Fusion`（按通道分组的可学习加权和）"
实现上述加权融合语义，拓扑按 BiFPN 结构组织（见 `configs/aloe-det-v11n.yaml` 颈部）。
跨尺度分支在融合前先通过 1×1 对齐卷积统一通道数（EfficientDet 中的标准做法，
否则加权融合无法对各分支做同尺度归一化）；同层输入输出以捷径直连以减少信息损失。

### 2.3 C3k2-Faster 轻量化卷积模块

用 PConv（部分卷积）+ 1×1 卷积的组合替代传统 3×3 卷积：

- PConv 只让 1/n 通道（默认 1/4）参与 3×3 卷积，其余通道直通（stride>1 时对直通部分做平均池化），
  显著降低 FLOPs 与内存访问开销；
- 瓶颈结构为 `Conv1×1 → PConv → Conv1×1`，残差相连；
- 骨干中所有 C3k2 替换为 C3k2-Faster，输出通道数不变，参数量与计算量整体下降。

## 3. AloeSeg：病斑分割与分级

### 3.1 DE 注意力（自研 Disease-Enhanced，骨干末端 + 各分割头前端）

面向"病斑尺寸微小、边缘模糊、呈黄褐色"的特点设计，两条并行支路：

1. **双路径通道注意力**：全局平均池化与全局最大池化两路描述子共享 MLP，逐元素相加后 sigmoid；
2. **局部对比度空间注意力**：`contrast = x - avgpool(x, k=5)`（中心-邻域对比，突出病斑边缘），
   沿通道平均后经 7×7 卷积 + sigmoid 得到空间图。

融合：`out = x * A_channel * A_spatial`，通道与空间协同增强，小目标漏检率下降。

### 3.2 DCNv2 可变形卷积（颈部各 C3k2 内）

在 C3k2 的瓶颈卷积中引入 DCNv2：offset 分支预测每个采样点的偏移，modulation 分支学习逐点权重，
动态贴合不规则病斑边缘；offset/modulation 卷积零初始化，保证训练初期行为接近标准卷积。
实现上有两个细节：① torchvision 的 `deform_conv2d` 直接乘 modulation 系数（内部不做
sigmoid），因此本仓库统一先做 sigmoid 再传入两条路径；② offset 通道按采样点交错排列
（每个采样点依次为 y、x 偏移），降级实现遵循同一约定。若当前后端不支持可变形卷积
（例如部分 CPU 版 torchvision），自动降级到 grid_sample 参考实现（`models/dcn.py`）。

### 3.3 P2 小目标检测头 + 轻量化分割检测头

- 在 BiFPN 颈部新增 P2（160×160 @640 输入）检测头，共 P2–P5 四个尺度，毫米级病斑不再因下采样丢失；
- P2 带来的参数增长由**轻量化分割头**抵消：用部分卷积替换 3×3 卷积、简化冗余层，参数净减少 27.9%。

**掩码原型（Proto）分辨率对齐**：ultralytics 的损失/验证约定掩码分辨率为 imgsz/4
（官方实现用 stride-8 的 P3 特征 + 2× 上采样）。新增 P2 后 `Segment` 的第一路输入是
stride-4 特征，Proto 若仍做 2× 上采样会输出 imgsz/2、与 GT 掩码不匹配。本仓库因此把
Proto 的上采样替换为恒等映射（输出仍为 imgsz/4），检测分支保留 P2 小目标头不变。

### 3.4 五级病害分级标准

基于分割结果计算病斑面积占叶片面积比例 p：

| 等级 | 英文 | 阈值 |
| --- | --- | --- |
| 健康 | healthy | p < 0.1% |
| 轻度感染 | mild | 0.1% ≤ p < 1% |
| 中度感染 | moderate | 1% ≤ p < 5% |
| 重度感染 | severe | 5% ≤ p < 15% |
| 坏死 | necrosis | p ≥ 15% |

实现见 `src/aloe_vision/postprocess/severity.py`。

## 4. 产量与减产率预估

基于 50 个田间样方的实测数据（自变量：各病害等级植株数，因变量：产量）建立多元线性回归：

```
y = x1 + 0.818·x2 + 0.692·x3 + 0.307·x4      （R² = 0.983）
减产率 = (理想产量 − 预测产量) / 理想产量
```

其中 x1–x4 为健康/轻度/中度/重度植株数量（坏死等级影响不显著）；理想产量按全部植株健康（系数 1.0）计算。
实现见 `src/aloe_vision/postprocess/yield_est.py`。

## 5. 重建边界（诚实声明）

- 论文中 图 5-2 / 图 5-3（网络结构图）未随文件保留，本仓库依据论文文字描述重建，
  模块插入位置与超参数以文字描述为准；与原始工程可能存在实现细节差异；
- MCA 采用论文官方逻辑重写；DE 注意力为"双路径通道 + 局部对比度空间"描述的等价实现；
- 训练数据与权重未开源，精度指标不可在本仓库直接复现（流程可由合成数据验证）。
- 工程集成：自定义模块通过 `aloe_vision.registry.register()` 注入 ultralytics 的 YAML
  解析流程（YAML 行预处理 + 同名分发类，见 `src/aloe_vision/compat.py`），
  未注册时官方模型结构不受影响（有回归测试保证）。

## 参考文献

1. MCA: Multidimensional collaborative attention in deep convolutional neural networks for image recognition.
2. Tan M, et al. EfficientDet: Scalable and Efficient Object Detection. CVPR 2020.（BiFPN）
3. Zhu X, et al. Deformable ConvNets v2. CVPR 2019.（DCNv2）
4. Chen J, et al. Run, Don't Walk: Chasing Higher FLOPS for Faster Neural Networks. CVPR 2023.（PConv/FasterNet）
5. Jocher G, et al. Ultralytics YOLO11. 2024.
