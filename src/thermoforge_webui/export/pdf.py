"""PDF 报告：reportlab 排版，中文用内置 CID 字体。

`STSong-Light` 是 reportlab 自带的 Adobe CJK 字体资源，不需要外部 ttf、
不需要浏览器、也不管目标机器装没装中文字体——这是选它而不是 HTML 转
PDF 的唯一原因（那条路要拖一个 Chromium 或一套 GTK 库）。
"""

from __future__ import annotations

import io

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from ..services.reports import ReportDocument, Section

CJK_FONT = "STSong-Light"
PAGE_WIDTH = A4[0] - 36 * mm  # 左右各 18mm 页边距
MAX_TABLE_COLUMNS = 9  # 超过就转成竖排，横排会挤成一团


def _register_font() -> None:
    if CJK_FONT not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(UnicodeCIDFont(CJK_FONT))


def _styles() -> dict[str, ParagraphStyle]:
    _register_font()
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TFTitle", parent=base["Title"], fontName=CJK_FONT, fontSize=20,
            leading=26, spaceAfter=4),
        "subtitle": ParagraphStyle(
            "TFSubtitle", parent=base["Normal"], fontName=CJK_FONT,
            fontSize=10, leading=14, textColor=colors.HexColor("#5d6773")),
        "h2": ParagraphStyle(
            "TFH2", parent=base["Heading2"], fontName=CJK_FONT, fontSize=14,
            leading=19, spaceBefore=16, spaceAfter=6,
            textColor=colors.HexColor("#16305c")),
        "h3": ParagraphStyle(
            "TFH3", parent=base["Heading3"], fontName=CJK_FONT, fontSize=11,
            leading=15, spaceBefore=10, spaceAfter=4,
            textColor=colors.HexColor("#5d6773")),
        "body": ParagraphStyle(
            "TFBody", parent=base["Normal"], fontName=CJK_FONT, fontSize=9.5,
            leading=15, alignment=TA_LEFT, spaceAfter=6),
        "caption": ParagraphStyle(
            "TFCaption", parent=base["Normal"], fontName=CJK_FONT, fontSize=8,
            leading=11, textColor=colors.HexColor("#5d6773"), spaceAfter=8),
        "cell": ParagraphStyle(
            "TFCell", parent=base["Normal"], fontName=CJK_FONT, fontSize=7.6,
            leading=10),
    }


def _inline(text: str) -> str:
    """把正文里的 `**粗体**` / `` `代码` `` 转成 reportlab 的行内标签。"""
    escaped = (text.replace("&", "&amp;").replace("<", "&lt;")
               .replace(">", "&gt;"))
    if escaped.startswith("&gt; "):
        escaped = escaped[5:]
    for marker, open_tag, close_tag in (
            ("**", "<b>", "</b>"),
            ("`", "<font face='Courier'>", "</font>")):
        parts = escaped.split(marker)
        escaped = "".join(
            part if index % 2 == 0 else f"{open_tag}{part}{close_tag}"
            for index, part in enumerate(parts))
    return escaped


def _cell_text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _table(frame: pd.DataFrame, styles: dict) -> Table:
    columns = list(frame.columns)[:MAX_TABLE_COLUMNS]
    data = [[Paragraph(f"<b>{column}</b>", styles["cell"])
             for column in columns]]
    for row in frame[columns].itertuples(index=False):
        data.append([Paragraph(_cell_text(value), styles["cell"])
                     for value in row])
    table = Table(data, colWidths=[PAGE_WIDTH / len(columns)] * len(columns),
                  repeatRows=1)
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d7dce3")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f2f5f9")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return table


def _key_value_table(pairs: dict[str, str], styles: dict) -> Table:
    rows = [[Paragraph(f"<b>{key}</b>", styles["cell"]),
             Paragraph(str(value), styles["cell"])]
            for key, value in pairs.items()]
    table = Table(rows, colWidths=[PAGE_WIDTH * 0.28, PAGE_WIDTH * 0.72])
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e3e7ec")),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f7f9fb")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return table


def _image(payload: bytes) -> Image:
    """按页宽等比缩放。PNG 是 130dpi 出的，直接放会溢出版心。"""
    from PIL import Image as PILImage

    with PILImage.open(io.BytesIO(payload)) as handle:
        width, height = handle.size
    scale = PAGE_WIDTH / width
    return Image(io.BytesIO(payload), width=PAGE_WIDTH, height=height * scale)


def _section_flowables(section: Section, styles: dict) -> list:
    flowables: list = [Paragraph(section.title, styles["h2"])]
    if section.key_values:
        flowables.extend([_key_value_table(section.key_values, styles),
                          Spacer(1, 8)])
    for paragraph in section.paragraphs:
        if paragraph.strip():
            flowables.append(Paragraph(_inline(paragraph), styles["body"]))
    if section.table is not None and not section.table.empty:
        flowables.extend([_table(section.table, styles), Spacer(1, 4)])
        if section.table_caption:
            flowables.append(Paragraph(section.table_caption, styles["caption"]))
        if len(section.table.columns) > MAX_TABLE_COLUMNS:
            flowables.append(Paragraph(
                f"（表格列数超过 {MAX_TABLE_COLUMNS}，PDF 里只排前 "
                f"{MAX_TABLE_COLUMNS} 列；完整数据见 HTML 或 Markdown 版）",
                styles["caption"]))
    for figure in section.figures:
        flowables.append(Paragraph(figure.title, styles["h3"]))
        flowables.append(_image(figure.png()))
        if figure.caption:
            flowables.append(Paragraph(figure.caption, styles["caption"]))
    return flowables


def render_pdf(document: ReportDocument) -> bytes:
    styles = _styles()
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=16 * mm, bottomMargin=16 * mm, title=document.title)

    flowables: list = [Paragraph(document.title, styles["title"])]
    if document.subtitle:
        flowables.append(Paragraph(document.subtitle, styles["subtitle"]))
    flowables.extend([
        Paragraph(f"生成时间 {document.generated_at}", styles["subtitle"]),
        Spacer(1, 14),
    ])
    for index, section in enumerate(document.sections):
        if index and section.title.startswith("实验 "):
            flowables.append(PageBreak())  # 每个实验单独起页，便于打印分发
        flowables.extend(_section_flowables(section, styles))
    flowables.extend([Spacer(1, 18),
                      Paragraph(document.footer, styles["caption"])])
    doc.build(flowables)
    return buffer.getvalue()
