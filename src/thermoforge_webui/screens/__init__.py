"""页面模块：一个文件一个页面，只做取数 + 渲染，不放业务逻辑。"""

from __future__ import annotations

from .copilot import copilot_page
from .data import data_page
from .models import models_page
from .overview import overview_page
from .quality import quality_page
from .report import report_page
from .research import research_page
from .results import results_page
from .settings import settings_page

__all__ = [
    "copilot_page",
    "data_page",
    "models_page",
    "overview_page",
    "quality_page",
    "report_page",
    "research_page",
    "results_page",
    "settings_page",
]
