"""报告导出：同一份文档模型 → HTML / Markdown / PDF。

- **HTML**：自包含单文件，图是 plotly 交互版（plotly.js 内联），
  发给别人双击就能看，不需要起任何服务。
- **Markdown**：图导出成 PNG 一起打包成 zip，可进 git 做版本化记录。
- **PDF**：reportlab 排版，中文用内置 STSong-Light CID 字体——不依赖
  浏览器、也不依赖系统装了哪种中文字体。
"""

from __future__ import annotations

from .html import render_html
from .markdown import render_markdown_bundle
from .pdf import render_pdf

__all__ = ["render_html", "render_markdown_bundle", "render_pdf"]
