"""Markdown 报告：正文 .md + 图片 PNG，打包成 zip。

打包而不是单文件，是因为 Markdown 没有内联图片的通行写法（base64 data
URI 在 GitHub 和多数编辑器里不渲染）。zip 解开就是一个能直接进 git 的
目录，风格与 `examples/chiller_power/report.md` 一致。
"""

from __future__ import annotations

import io
import zipfile

import pandas as pd

from ..services.reports import ReportDocument, Section

IMAGE_DIR = "images"


def _cell(value: object) -> str:
    """单元格文本。手写而不用 `DataFrame.to_markdown`——那个要 tabulate，
    为一张表加一个依赖不划算。"""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    if isinstance(value, float):
        return f"{value:.4g}"
    text = str(value)
    return text.replace("|", "\\|") if "|" in text else text


def _table_markdown(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join(_cell(value) for value in row) + " |")
    return "\n".join(lines)


def _section_markdown(section: Section, images: dict[str, bytes]) -> list[str]:
    lines = [f"## {section.title}", ""]
    if section.key_values:
        lines.append("| 项 | 值 |")
        lines.append("|---|---|")
        for key, value in section.key_values.items():
            lines.append(f"| {key} | {value} |")
        lines.append("")
    for paragraph in section.paragraphs:
        lines.extend([paragraph, ""])
    for block in section.math_blocks:
        # 一行一个 $$ 块：多行塞同一个 $$ 里 KaTeX 会忽略换行、串成一行
        for line in block.lines:
            lines.extend(["$$", line, "$$", ""])
        if block.caption:
            lines.extend([f"*{block.caption}*", ""])
    if section.table is not None and not section.table.empty:
        lines.extend([_table_markdown(section.table), ""])
        if section.table_caption:
            lines.extend([f"*{section.table_caption}*", ""])
    for caption, frame in section.extra_tables:
        if frame is None or frame.empty:
            continue
        lines.extend([_table_markdown(frame), ""])
        if caption:
            lines.extend([f"*{caption}*", ""])
    for figure in section.figures:
        name = f"{figure.key}.png"
        images[name] = figure.png()
        lines.extend([f"### {figure.title}", "",
                      f"![{figure.title}]({IMAGE_DIR}/{name})", ""])
        if figure.caption:
            lines.extend([f"*{figure.caption}*", ""])
    return lines


def render_markdown(document: ReportDocument,
                    images: dict[str, bytes]) -> str:
    lines = [f"# {document.title}", ""]
    if document.subtitle:
        lines.extend([document.subtitle, ""])
    lines.extend([f"生成时间：{document.generated_at}", ""])
    for section in document.sections:
        lines.extend(_section_markdown(section, images))
    lines.extend(["---", "", document.footer, ""])
    return "\n".join(lines)


def render_markdown_bundle(document: ReportDocument,
                           basename: str = "report") -> bytes:
    """返回 zip 字节：`report.md` + `images/*.png`。"""
    images: dict[str, bytes] = {}
    text = render_markdown(document, images)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        # 显式 UTF-8 + \n：跨平台哈希稳定性，与项目其它文本写入同调
        archive.writestr(f"{basename}.md", text.encode("utf-8"))
        for name, payload in images.items():
            archive.writestr(f"{IMAGE_DIR}/{name}", payload)
    return buffer.getvalue()
