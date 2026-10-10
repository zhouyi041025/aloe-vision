"""训练引擎 CLI 冒烟测试：五个入口的参数解析与帮助信息（不跑真实训练）。"""
from __future__ import annotations

import pytest

pytest.importorskip("ultralytics")

from aloe_vision.engine import benchmark, export, predict, train, val


@pytest.mark.parametrize("module", [train, val, predict, export, benchmark], ids=lambda m: m.__name__.split(".")[-1])
def test_cli_help(module, capsys):
    """每个入口 --help 都能正常打印并退出（exit code 0）。"""
    with pytest.raises(SystemExit) as exc:
        module.main(["--help"])
    assert exc.value.code == 0
    text = capsys.readouterr()
    assert "usage" in (text.out + text.err).lower()


def test_parse_overrides_types():
    """--set k=v 的类型解析：int / float / 字符串。"""
    parsed = train._parse_overrides(["lr0=0.01", "epochs=5", "mosaic=1.0", "project=myrun"])
    assert parsed == {"lr0": 0.01, "epochs": 5, "mosaic": 1.0, "project": "myrun"}


def test_parse_overrides_invalid():
    """缺少 '=' 的透传参数应当直接报错，而不是被静默忽略。"""
    with pytest.raises(SystemExit):
        train._parse_overrides(["lr0"])


def test_train_requires_data():
    """--data 为必填参数。"""
    with pytest.raises(SystemExit) as exc:
        train.main(["--config", "configs/aloe-det-v11n.yaml"])
    assert exc.value.code == 2
