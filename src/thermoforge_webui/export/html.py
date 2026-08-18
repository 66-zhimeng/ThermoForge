"""自包含单文件 HTML 报告。

plotly.js 用 `include_plotlyjs="inline"` 只在**第一张图**内联一次（约 3MB），
后面的图复用同一份运行时——每张图都内联会让文件按图数线性膨胀。
"""

from __future__ import annotations

import html as html_escape

import pandas as pd
from jinja2 import Environment
from markupsafe import Markup

from ..services.reports import ReportDocument, Section

_TEMPLATE = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{{ doc.title }}</title>
<style>
:root{--bg:#ffffff;--fg:#16191d;--muted:#5d6773;--line:#e3e7ec;
      --accent:#2f6feb;--sunk:#f6f8fa;--radius:10px}
@media(prefers-color-scheme:dark){:root{--bg:#12161b;--fg:#e6e9ee;
      --muted:#9aa4b1;--line:#262d36;--sunk:#171d24}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
     font:16px/1.65 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:40px 24px 72px}
header{border-bottom:2px solid var(--accent);padding-bottom:18px;
       margin-bottom:32px}
h1{font-size:28px;margin:0 0 6px}
h2{font-size:21px;margin:44px 0 14px;padding-top:14px;
   border-top:1px solid var(--line)}
h3{font-size:16px;margin:26px 0 8px;color:var(--muted)}
.sub{color:var(--muted);font-size:14px}
.kv{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));
    gap:12px;margin:16px 0}
.kv div{background:var(--sunk);border:1px solid var(--line);
        border-radius:var(--radius);padding:10px 14px}
.kv dt{font-size:12px;color:var(--muted);margin:0 0 4px}
.kv dd{margin:0;font-size:15px;font-weight:560;word-break:break-all}
.tablewrap{overflow-x:auto;margin:14px 0}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{border:1px solid var(--line);padding:7px 10px;text-align:right;
      white-space:nowrap}
th{background:var(--sunk);font-weight:600;text-align:center}
td:first-child,th:first-child{text-align:left}
.cap{color:var(--muted);font-size:13px;margin:6px 0 0}
.fig{margin:22px 0}
blockquote{margin:14px 0;padding:10px 16px;border-left:3px solid var(--accent);
           background:var(--sunk);color:var(--muted)}
code{background:var(--sunk);padding:1px 5px;border-radius:4px;font-size:.92em}
pre{background:var(--sunk);border:1px solid var(--line);padding:10px 14px;
    border-radius:var(--radius);overflow-x:auto;font-size:13px;line-height:1.55}
footer{margin-top:56px;padding-top:16px;border-top:1px solid var(--line);
       color:var(--muted);font-size:13px}
</style></head><body><div class="wrap">
<header>
  <h1>{{ doc.title }}</h1>
  {% if doc.subtitle %}<p class="sub">{{ doc.subtitle }}</p>{% endif %}
  <p class="sub">生成时间 {{ doc.generated_at }}</p>
</header>
{% for section in doc.sections %}
<section>
  <h2>{{ section.title }}</h2>
  {% if section.key_values %}
  <dl class="kv">
    {% for key, value in section.key_values.items() %}
    <div><dt>{{ key }}</dt><dd>{{ value }}</dd></div>
    {% endfor %}
  </dl>
  {% endif %}
  {% for paragraph in section.paragraphs %}{{ paragraph | md }}{% endfor %}
  {% if section.table_html %}
  <div class="tablewrap">{{ section.table_html | safe }}</div>
  {% if section.table_caption %}<p class="cap">{{ section.table_caption }}</p>{% endif %}
  {% endif %}
  {% for caption, table_html in section.extra_tables_html %}
  <div class="tablewrap">{{ table_html | safe }}</div>
  {% if caption %}<p class="cap">{{ caption }}</p>{% endif %}
  {% endfor %}
  {% for figure in section.figures_html %}
  <div class="fig">
    <h3>{{ figure.title }}</h3>
    {{ figure.body | safe }}
    {% if figure.caption %}<p class="cap">{{ figure.caption }}</p>{% endif %}
  </div>
  {% endfor %}
</section>
{% endfor %}
<footer>{{ doc.footer }}</footer>
</div></body></html>
"""


def _mini_markdown(text: str) -> str:
    """段落里只支持 **粗体**、`代码`、开头的 `> 引用` 和 ``` 围栏代码块
    （模型公式用）——报告正文用不到更多。"""
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.split("\n")
        body = "\n".join(lines[1:-1])
        return f"<pre>{html_escape.escape(body)}</pre>"
    escaped = html_escape.escape(text)
    quote = escaped.startswith("&gt; ")
    if quote:
        escaped = escaped[5:]
    parts = escaped.split("**")
    escaped = "".join(part if index % 2 == 0 else f"<strong>{part}</strong>"
                      for index, part in enumerate(parts))
    parts = escaped.split("`")
    escaped = "".join(part if index % 2 == 0 else f"<code>{part}</code>"
                      for index, part in enumerate(parts))
    return f"<blockquote>{escaped}</blockquote>" if quote else f"<p>{escaped}</p>"


def _table_html(frame: pd.DataFrame) -> str:
    return frame.to_html(index=False, border=0, na_rep="—",
                         float_format=lambda v: f"{v:.4g}")


def _section_view(section: Section, first_figure: list[bool]) -> dict:
    figures = []
    for figure in section.figures:
        # plotly.js 只内联一次，后续图复用同一份运行时
        include = "inline" if first_figure[0] else False
        first_figure[0] = False
        body = figure.plotly().to_html(
            full_html=False, include_plotlyjs=include,
            config={"displaylogo": False, "responsive": True})
        figures.append({"title": figure.title, "caption": figure.caption,
                        "body": body})
    return {
        "title": section.title,
        "key_values": section.key_values,
        "paragraphs": section.paragraphs,
        "table_html": (_table_html(section.table)
                       if section.table is not None and not section.table.empty
                       else ""),
        "table_caption": section.table_caption,
        "extra_tables_html": [(caption, _table_html(frame))
                              for caption, frame in section.extra_tables
                              if frame is not None and not frame.empty],
        "figures_html": figures,
    }


def render_html(document: ReportDocument) -> str:
    environment = Environment(autoescape=True)
    # _mini_markdown 自己做了转义，标记为安全，否则 autoescape 会二次转义
    environment.filters["md"] = lambda text: Markup(_mini_markdown(text))
    template = environment.from_string(_TEMPLATE)
    first_figure = [True]
    view = {
        "title": document.title,
        "subtitle": document.subtitle,
        "generated_at": document.generated_at,
        "footer": document.footer,
        "sections": [_section_view(section, first_figure)
                     for section in document.sections],
    }
    return template.render(doc=view)
