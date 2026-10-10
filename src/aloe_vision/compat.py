"""ultralytics 集成层：让 aloe-vision 的自定义模块通过标准 YAML 构建模型。

为什么需要这一层：
ultralytics 的 `parse_model` 用局部 frozenset（base_modules / repeat_modules）判断模块的
通道与重复语义，外部无法直接扩展这些集合。本模块采用"YAML 行预处理 + 同名分发类"方案：

1. 我们的 YAML 使用可读的自定义模块名（MCA / DEAttention / C3k2_Faster / C3k2_DCN / BiFPN_Fusion）；
2. `install()` 把 `ultralytics.nn.tasks` 里的官方类名 `Bottleneck` / `C3k2` 替换为"分发类"，
   并把 `parse_model` 换成带预处理的包装器；官方集合在每次解析时从模块全局名取类，因此补丁即时生效；
3. 预处理会把自定义模块名改写成官方名并在 args 末尾附加字符串标记；分发类看到标记就构建自定义模块，
   没有标记时行为与官方实现完全一致（直接走父类逻辑）——因此官方模型结构不受影响。

注意：轻量化分割头（论文 5.3.3 第 4 点）不走此机制，而是在 `SegmentationModel` 构建完成后
对检测头做子模块级轻量化替换（见 `lighten_segment_head`），避免破坏官方对 Segment 的身份判断。
"""
from __future__ import annotations

from copy import deepcopy

import torch.nn as nn
import ultralytics.nn.tasks as _tasks
from ultralytics.nn.modules import Conv as UltraConv
from ultralytics.nn.modules import DWConv

from aloe_vision.models.attention import MCA, DEAttention
from aloe_vision.models.blocks import C3k2_DCN, C3k2_Faster, PConv
from aloe_vision.models.conv import Conv
from aloe_vision.models.neck import BiFPN_Fusion

# 自定义模块名 -> (官方替代名, 标记字符串)
REWRITE: dict[str, tuple[str, str]] = {
    "MCA": ("Bottleneck", "mca"),
    "DEAttention": ("Bottleneck", "de"),
    "C3k2_Faster": ("C3k2", "faster"),
    "C3k2_DCN": ("C3k2", "dcn"),
    "BiFPN_Fusion": ("Bottleneck", "bifpn"),
}
_MARKERS = {mark for _, mark in REWRITE.values()}

_ORIG_BOTTLENECK = _tasks.Bottleneck
_ORIG_C3K2 = _tasks.C3k2
_ORIG_PARSE_MODEL = _tasks.parse_model


def has_custom_modules(cfg: dict) -> bool:
    """判断模型配置里是否使用了 aloe-vision 自定义模块。"""
    for section in ("backbone", "head"):
        for row in cfg.get(section, []) or []:
            if len(row) > 2 and row[2] in REWRITE:
                return True
    return False


def preprocess_yaml_dict(cfg: dict) -> dict:
    """把自定义模块名改写成官方名并在 args 末尾附加标记（返回新 dict，不修改入参）。"""
    cfg = deepcopy(cfg)
    for section in ("backbone", "head"):
        for row in cfg.get(section, []) or []:
            name = row[2]
            if name in REWRITE:
                new_name, mark = REWRITE[name]
                row[2] = new_name
                row[3] = [*list(row[3]), mark]
    return cfg


def _split_marker(args: tuple, kwargs: dict) -> tuple[str | None, tuple, dict]:
    """从参数里摘出自定义模块标记字符串。"""
    args = list(args)
    for i, value in enumerate(args):
        if isinstance(value, str) and value in _MARKERS:
            return args.pop(i), tuple(args), kwargs
    for key in list(kwargs):
        value = kwargs[key]
        if isinstance(value, str) and value in _MARKERS:
            del kwargs[key]
            return value, tuple(args), kwargs
    return None, tuple(args), kwargs


class _BottleneckDispatch(_ORIG_BOTTLENECK):
    """Bottleneck 名称的分发：无标记 = 官方实现；标记 mca/de/bifpn = 自定义模块。"""

    def __init__(self, c1, c2, *args, **kwargs):
        marker, args, kwargs = _split_marker(args, kwargs)
        if marker is None:
            super().__init__(c1, c2, *args, **kwargs)
            self._aloe_impl = None
            return
        nn.Module.__init__(self)  # 跳过父类的卷积构建，仅初始化 Module 机制
        if marker in {"mca", "de"}:
            if c2 != c1:
                raise ValueError(
                    f"{marker.upper()} 是通道保持模块，要求 c1 == c2，收到 c1={c1}, c2={c2}；"
                    "请在 YAML 里把该行的参数写成与输入通道一致的设计通道数。"
                )
            self._aloe_impl = (MCA if marker == "mca" else DEAttention)(c1)
        elif marker == "bifpn":
            self._aloe_impl = BiFPN_Fusion(c1, c2, *args, **kwargs)
        else:  # pragma: no cover - 理论上不可达
            raise ValueError(f"未知标记: {marker}")

    def forward(self, x):
        impl = getattr(self, "_aloe_impl", None)
        if impl is not None:
            return impl(x)
        return super().forward(x)


class _C3k2Dispatch(_ORIG_C3K2):
    """C3k2 名称的分发：无标记 = 官方实现；标记 faster/dcn = 自定义模块。"""

    def __init__(self, c1, c2, *args, **kwargs):
        marker, args, kwargs = _split_marker(args, kwargs)
        if marker is None:
            super().__init__(c1, c2, *args, **kwargs)
            self._aloe_impl = None
            return
        nn.Module.__init__(self)  # 跳过父类的卷积构建，仅初始化 Module 机制
        cls = {"faster": C3k2_Faster, "dcn": C3k2_DCN}[marker]
        self._aloe_impl = cls(c1, c2, *args, **kwargs)

    def forward(self, x):
        impl = getattr(self, "_aloe_impl", None)
        if impl is not None:
            return impl(x)
        return super().forward(x)


def _parse_model_patched(cfg, ch, verbose=True):
    """parse_model 包装器：检测到自定义模块时先做行预处理。"""
    if has_custom_modules(cfg):
        cfg = preprocess_yaml_dict(cfg)
    return _ORIG_PARSE_MODEL(cfg, ch, verbose=verbose)


def _light_conv3x3(conv: nn.Module) -> nn.Module | None:
    """把 ultralytics 的 3×3 Conv 替换为 Conv1×1 → PConv → Conv1×1（论文轻量化分割头）。

    深度可分离卷积（DWConv）与 1×1 卷积返回 None（保持原样）。
    """
    if type(conv) is not UltraConv or isinstance(conv, DWConv):
        return None
    inner = getattr(conv, "conv", None)
    if not isinstance(inner, nn.Conv2d) or inner.kernel_size[0] != 3 or inner.groups != 1:
        return None
    c1, c2, stride = inner.in_channels, inner.out_channels, inner.stride[0]
    return nn.Sequential(Conv(c1, c2, 1, 1), PConv(c2, c2, k=3, s=stride), Conv(c2, c2, 1, 1))


def _lighten_sequential(seq: nn.Sequential) -> int:
    """替换 Sequential 中的 3×3 卷积，返回替换数量。"""
    replaced = 0
    for i, child in enumerate(seq):
        repl = _light_conv3x3(child)
        if repl is not None:
            seq[i] = repl
            replaced += 1
    return replaced


def lighten_segment_head(model) -> bool:
    """把分割头（box/cls/mask 分支与 proto）的 3×3 卷积替换为 PConv + 1×1 组合。

    返回是否执行了替换。仅对包含 proto/cv4 的 Segment 头生效（对应论文 5.3.3
    "轻量化分割检测头"：部分卷积替换 3×3 卷积，用来抵消新增 P2 检测头的参数增长）。
    """
    head = model.model[-1] if hasattr(model, "model") else None
    if head is None or not hasattr(head, "proto") or not hasattr(head, "cv4"):
        return False

    replaced = 0
    for attr in ("cv2", "cv3", "cv4", "one2one_cv2", "one2one_cv3", "one2one_cv4"):
        branch = getattr(head, attr, None)
        if isinstance(branch, nn.ModuleList):
            for seq in branch:
                if isinstance(seq, nn.Sequential):
                    replaced += _lighten_sequential(seq)
    proto = getattr(head, "proto", None)
    for name in ("cv1", "cv2"):
        child = getattr(proto, name, None)
        repl = _light_conv3x3(child) if child is not None else None
        if repl is not None:
            setattr(proto, name, repl)
            replaced += 1
    return replaced > 0


def _is_aloe_seg_cfg(cfg) -> bool:
    """判断配置是否为 aloe-vision 的分割模型（或直接是含自定义模块的 dict）。"""
    if isinstance(cfg, dict):
        if cfg.get("aloe_light_head"):
            return True
        return has_custom_modules(cfg) and any(row[2] == "Segment" for row in cfg.get("head", []) or [])
    if isinstance(cfg, str):
        try:
            from pathlib import Path

            import yaml

            if Path(cfg).exists():
                data = yaml.safe_load(Path(cfg).read_text(encoding="utf-8"))
                return bool(data.get("aloe_light_head"))
        except Exception:
            return False
    return False


def _align_proto_resolution(model) -> bool:
    """对齐掩码原型（Proto）输出分辨率：P2 输入时去掉 2× 上采样。

    背景：ultralytics 的损失/验证约定掩码分辨率为 imgsz/4（官方实现用 stride-8
    的 P3 特征 + 2× 上采样）。本仓库的分割模型新增 P2 检测头后，Segment 的
    `x[0]` 是 stride-4 的 P2 特征，若 Proto 仍做 2× 上采样，输出会变成 imgsz/2，
    与 GT 掩码分辨率不匹配（验证阶段 mask IoU 会直接报维度错误）。
    因此这里把 Proto 的上采样替换为恒等映射：P2 已是 imgsz/4，输出与官方一致。
    返回是否执行了替换。
    """
    head = model.model[-1] if hasattr(model, "model") else None
    if head is None or not hasattr(head, "proto") or not hasattr(head, "cv2"):
        return False
    try:
        stride0 = float(head.stride[0])
    except Exception:  # pragma: no cover - 结构不匹配时不动
        return False
    if stride0 > 4.5:  # 没有 P2 尺度，保持官方结构（P3 输入 + 2× 上采样）
        return False
    head.proto.upsample = nn.Identity()
    return True


_INSTALLED = False


def install() -> None:
    """打补丁：替换官方类名 + 包装 parse_model + 包装 SegmentationModel 构建后处理。"""
    global _INSTALLED
    if _INSTALLED:
        return

    _tasks.Bottleneck = _BottleneckDispatch
    _tasks.C3k2 = _C3k2Dispatch
    _tasks.parse_model = _parse_model_patched

    _orig_seg_init = _tasks.SegmentationModel.__init__

    def _seg_init(self, cfg="yolo11-seg.yaml", ch=3, nc=None, verbose=True):
        _orig_seg_init(self, cfg, ch, nc, verbose)
        if _is_aloe_seg_cfg(cfg):
            _align_proto_resolution(self)
            lighten_segment_head(self)

    _tasks.SegmentationModel.__init__ = _seg_init
    _INSTALLED = True
