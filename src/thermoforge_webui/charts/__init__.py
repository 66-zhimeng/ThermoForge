"""图表层：一份绘图数据，两种画法。

`series.py` 做**所有**取数与统计（抽稀、残差、分位、区间切段），
`interactive.py`（plotly，界面与 HTML 报告）和 `static.py`（matplotlib，
Markdown 与 PDF 用的 PNG）只负责把同一份准备好的数据画出来。

之所以要两个渲染器：plotly 静态化需要 kaleido，而 kaleido 0.2.1 在
Windows 上 `to_image` 会无限阻塞，1.x 又要额外下载 Chrome——两条路都
和「双击 .bat 就能用」冲突。把真正有逻辑的部分收进 series.py 之后，
两个渲染器都只剩描点，重复是可控的。
"""

from __future__ import annotations

# 统一配色：两种渲染器共用，报告和界面看起来才是同一套图。
PALETTE = {
    "actual": "#2f6feb",     # 实测
    "predicted": "#e8710a",  # 预测
    "residual": "#6f42c1",
    "train": "#94a3b8",
    "validate": "#38bdf8",
    "test": "#22c55e",
    "gap": "#f59e0b",        # purge / embargo 间隙
    "reference": "#64748b",  # 45° 线、零线
    "grid": "#e2e8f0",
    "bad": "#b3261e",
    "good": "#0b7a48",
}

SURFACE_COLORS = {
    "train": PALETTE["train"],
    "validate": PALETTE["validate"],
    "test": PALETTE["test"],  # 切分图的测试段；漏掉会 fallback 成训练灰
    "A": PALETTE["test"],
    "B": "#16a34a",
    "C": "#15803d",
}
