"""Web 控制台页面冒烟测试。

用 Streamlit 自带的 `AppTest` 无头跑每个页面：它会把渲染过程中抛出的
异常收进 `at.exception`，所以「页面能不能打开」这件事可以自动验证，
不必每次手动点一遍浏览器。

这些测试跑在仓库真实的工件目录上（vault/ research/ models/）。工件不
存在时页面应当走空状态分支而不是崩——那本身就是要测的行为之一。
"""

from __future__ import annotations

import pytest

from streamlit.testing.v1 import AppTest

SCREENS = [
    "copilot_page",
    "overview_page",
    "data_page",
    "quality_page",
    "research_page",
    "results_page",
    "models_page",
    "report_page",
    "settings_page",
]


def _run(screen_name: str) -> AppTest:
    """在 AppTest 的脚本上下文里导入并调用页面函数。

    用 `from_string` 而不是 `from_function`：后者没法传参，只能靠改闭包
    全局变量注入页面名，绕得毫无必要。
    """
    source = (f"import thermoforge_webui.screens as screens\n"
              f"screens.{screen_name}()\n")
    return AppTest.from_string(source, default_timeout=90).run()


@pytest.mark.parametrize("screen_name", SCREENS)
def test_screen_renders_without_exception(screen_name: str) -> None:
    app = _run(screen_name)
    assert not app.exception, (
        f"{screen_name} 渲染抛异常："
        + "\n".join(str(item.value) for item in app.exception))


def test_app_entry_runs() -> None:
    """入口脚本本身（导航 + 侧栏）能跑通。"""
    app = AppTest.from_file("src/thermoforge_webui/app.py",
                            default_timeout=90).run()
    assert not app.exception, "\n".join(str(i.value) for i in app.exception)


def test_frames_tolerate_truncation_marker():
    """信封截断会在列表尾部追加字符串标记，表格渲染不能因此抛异常。"""
    from thermoforge_webui.services import catalog

    marker = "... (truncated, total=241)"

    schema = {"variables": [
        {"variable_id": "chiller_01.power", "source_kind": "measured"},
        marker,
    ]}
    frame = catalog.variables_frame(schema)
    assert len(frame) == 1
    assert frame.iloc[0]["变量"] == "chiller_01.power"

    profile = {"variables": [
        {"variable_id": "chiller_01.power", "count": 33109},
        marker,
    ]}
    frame = catalog.profile_frame(profile)
    assert len(frame) == 1
    assert frame.iloc[0]["有效样本"] == 33109
