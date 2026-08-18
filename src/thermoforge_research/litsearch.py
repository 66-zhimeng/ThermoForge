"""文献检索（CrossRef / arXiv / Semantic Scholar 免费 API，无需 key）。

架构位置：网络 + 解析层，不依赖 ToolContext；`tools.py` 的
`tf_literature_search` 是薄封装。三个源均无鉴权、零配置：

- CrossRef：HVAC 建模主刊（Energy and Buildings / Applied Energy 等）覆盖
  最全；abstract 常带 JATS 标签，解析时剥掉。
- arXiv：方法类新工作；Atom XML，无引用数。
- Semantic Scholar：摘要与引用数质量最好；无 key 限流（429 按源失败处理）。

错误码：文献域使用 `TFL-` 前缀。conventions.md §7 错误码表当前无文献域
小节，沿 thermoforge_data.preprocess 的 TFPP 先例，码以模块级常量维护于此。

HTTP 层 `http_get` 是独立模块级函数：测试 monkeypatch 替换，全程不触网。
用标准库 urllib 而非 httpx——pyproject 无 httpx 直接依赖，且 urllib 默认
读取系统/环境代理，适配受限网络（client.py 的本地端点直连问题不涉及这里，
三个 API 都是外网地址）。
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any, Callable, Mapping, Sequence

TFL_SOURCE_FAILED = "TFL-001"        # 单源失败（网络/限流/解析）
TFL_ALL_SOURCES_FAILED = "TFL-002"   # 所有源均不可用
TFL_SOURCE_UNKNOWN = "TFL-003"       # sources 参数含未登记的源

DEFAULT_SOURCES = ("crossref", "arxiv", "semantic_scholar")
SOURCES = DEFAULT_SOURCES  # 允许值清单（工具层校验用）

MAX_LIMIT = 20               # 单源请求行数与最终返回数的硬上限
ABSTRACT_MAX_CHARS = 800     # 配合 MAX_LIMIT 控制在 32KB 信封内
TIMEOUT_SECONDS = 15.0
USER_AGENT = "ThermoForge/0.1 (literature-search; urllib)"


class AllSourcesFailed(RuntimeError):
    """所有源都抛错（区别于「检索成功但零命中」——后者是正常结果）。"""

    def __init__(self, failures: Sequence[Mapping[str, Any]]):
        self.failures = [dict(f) for f in failures]
        detail = "; ".join(f"{f['source']}: {f['error']}" for f in failures)
        super().__init__(f"[{TFL_ALL_SOURCES_FAILED}] 所有文献源均不可用: {detail}")


def http_get(url: str, *, timeout: float = TIMEOUT_SECONDS) -> bytes:
    """GET 并返回原始字节。CrossRef/arXiv 对默认 UA 不友好，必须自带。"""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


# ---------------------------------------------------------------- 解析（纯函数）

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _clean_abstract(text: Any) -> str | None:
    """剥 JATS/HTML 标签、压空白、截断；空内容归一为 None。"""
    if not isinstance(text, str) or not text.strip():
        return None
    cleaned = _WS_RE.sub(" ", _TAG_RE.sub(" ", text)).strip()
    if not cleaned:
        return None
    if len(cleaned) > ABSTRACT_MAX_CHARS:
        return cleaned[:ABSTRACT_MAX_CHARS] + "…(截断)"
    return cleaned


def _parse_crossref(payload: bytes, limit: int) -> list[dict[str, Any]]:
    doc = json.loads(payload)
    items = (doc.get("message") or {}).get("items") or []
    out: list[dict[str, Any]] = []
    for item in items[:limit]:
        title = _WS_RE.sub(" ", " ".join(item.get("title") or [])).strip()
        if not title:
            continue
        authors = [
            " ".join(p for p in (a.get("given"), a.get("family")) if p)
            for a in (item.get("author") or [])
        ][:5]
        year = None
        for key in ("published", "published-print", "published-online"):
            parts = (item.get(key) or {}).get("date-parts") or []
            if parts and parts[0] and isinstance(parts[0][0], int):
                year = parts[0][0]
                break
        venue = " ".join(item.get("container-title") or []).strip() or None
        doi = item.get("DOI")
        out.append({
            "title": title,
            "authors": authors,
            "year": year,
            "venue": venue,
            "doi": doi,
            "url": item.get("URL")
                or (f"https://doi.org/{doi}" if doi else None),
            "abstract": _clean_abstract(item.get("abstract")),
            "citation_count": item.get("is-referenced-by-count"),
            "source": "crossref",
        })
    return out


_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom"}


def _parse_arxiv(payload: bytes, limit: int,
                 year_from: int | None = None) -> list[dict[str, Any]]:
    root = ET.fromstring(payload)
    out: list[dict[str, Any]] = []
    for entry in root.findall("atom:entry", _ATOM_NS):
        title = _WS_RE.sub(" ", entry.findtext("atom:title", "", _ATOM_NS)
                           or "").strip()
        if not title:
            continue
        published = entry.findtext("atom:published", "", _ATOM_NS) or ""
        year = int(published[:4]) if published[:4].isdigit() else None
        if year_from is not None and year is not None and year < year_from:
            continue  # arXiv 的 API 日期区间语法对模糊查询不友好，后置过滤
        authors = [a.findtext("atom:name", "", _ATOM_NS) or ""
                   for a in entry.findall("atom:author", _ATOM_NS)][:5]
        doi = entry.findtext("arxiv:doi", None, _ATOM_NS) or None
        link = (entry.findtext("atom:id", "", _ATOM_NS) or "").strip() or None
        out.append({
            "title": title,
            "authors": authors,
            "year": year,
            "venue": entry.findtext("arxiv:journal_ref", None, _ATOM_NS),
            "doi": doi,
            "url": f"https://doi.org/{doi}" if doi else link,
            "abstract": _clean_abstract(
                entry.findtext("atom:summary", "", _ATOM_NS)),
            "citation_count": None,
            "source": "arxiv",
        })
        if len(out) >= limit:
            break
    return out


def _parse_semantic_scholar(payload: bytes,
                            limit: int) -> list[dict[str, Any]]:
    doc = json.loads(payload)
    data = doc.get("data") or []
    out: list[dict[str, Any]] = []
    for item in data[:limit]:
        title = _WS_RE.sub(" ", item.get("title") or "").strip()
        if not title:
            continue
        ext = item.get("externalIds") or {}
        doi = ext.get("DOI")
        arxiv_id = ext.get("ArXiv")
        url = item.get("url") \
            or (f"https://doi.org/{doi}" if doi else None) \
            or (f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else None)
        out.append({
            "title": title,
            "authors": [a.get("name") or ""
                        for a in (item.get("authors") or [])][:5],
            "year": item.get("year"),
            "venue": item.get("venue") or None,
            "doi": doi,
            "url": url,
            "abstract": _clean_abstract(item.get("abstract")),
            "citation_count": item.get("citationCount"),
            "source": "semantic_scholar",
        })
    return out


# ---------------------------------------------------------------- 检索编排


def _crossref_url(query: str, limit: int, year_from: int | None) -> str:
    params: dict[str, Any] = {
        "query.bibliographic": query,
        "rows": limit,
        "select": "DOI,title,author,published,container-title,"
                  "abstract,is-referenced-by-count,URL",
    }
    if year_from is not None:
        params["filter"] = f"from-pub-date:{year_from}-01-01"
    return "https://api.crossref.org/works?" + urllib.parse.urlencode(params)


def _arxiv_url(query: str, limit: int) -> str:
    # 多词查询显式 AND：裸空格在 arXiv 语法里是解析错误而非隐式 AND
    terms = " AND ".join(f"all:{t}" for t in query.split()) or "all:"
    params = {"search_query": terms, "start": 0, "max_results": limit,
              "sortBy": "relevance", "sortOrder": "descending"}
    return "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(params)


def _s2_url(query: str, limit: int, year_from: int | None) -> str:
    params: dict[str, Any] = {
        "query": query,
        "limit": limit,
        "fields": "title,abstract,year,venue,authors,externalIds,"
                  "citationCount,url",
    }
    if year_from is not None:
        params["year"] = f"{year_from}-"
    return ("https://api.semanticscholar.org/graph/v1/paper/search?"
            + urllib.parse.urlencode(params))


def _search_crossref(query: str, limit: int, year_from: int | None,
                     timeout: float) -> list[dict[str, Any]]:
    return _parse_crossref(
        http_get(_crossref_url(query, limit, year_from), timeout=timeout),
        limit)


def _search_arxiv(query: str, limit: int, year_from: int | None,
                  timeout: float) -> list[dict[str, Any]]:
    return _parse_arxiv(
        http_get(_arxiv_url(query, limit), timeout=timeout), limit, year_from)


def _search_semantic_scholar(query: str, limit: int, year_from: int | None,
                             timeout: float) -> list[dict[str, Any]]:
    return _parse_semantic_scholar(
        http_get(_s2_url(query, limit, year_from), timeout=timeout), limit)


_SOURCE_FNS: dict[str, Callable[[str, int, int | None, float],
                                list[dict[str, Any]]]] = {
    "crossref": _search_crossref,
    "arxiv": _search_arxiv,
    "semantic_scholar": _search_semantic_scholar,
}


def _dedup(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """按 DOI（无 DOI 按规范化标题）去重；先到的记录为骨架，缺字段用后到的补。"""
    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for item in results:
        doi = (item.get("doi") or "").lower()
        key = f"doi:{doi}" if doi else \
            "title:" + _WS_RE.sub(" ", item.get("title") or "").lower().strip()
        if key not in merged:
            merged[key] = dict(item)
            order.append(key)
            continue
        base = merged[key]
        for field in ("abstract", "citation_count", "venue", "url", "year"):
            if base.get(field) is None and item.get(field) is not None:
                base[field] = item[field]
    return [merged[key] for key in order]


def search(
    query: str,
    *,
    sources: Sequence[str] = DEFAULT_SOURCES,
    limit: int = 10,
    year_from: int | None = None,
    timeout: float = TIMEOUT_SECONDS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """逐源检索 → 归一化 → 去重。返回 (results, failures)。

    单源失败不拖垮整体；全部源抛错才抛 `AllSourcesFailed`。
    `sources`/`query` 的合法性由工具层校验，这里假定已过滤。
    """
    limit = max(1, min(int(limit), MAX_LIMIT))
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for source in sources:
        try:
            results.extend(_SOURCE_FNS[source](query, limit, year_from, timeout))
        except Exception as exc:
            failures.append({"source": source,
                             "error": f"{type(exc).__name__}: {exc}"})
    if sources and not results and len(failures) == len(list(sources)):
        raise AllSourcesFailed(failures)
    return _dedup(results)[:limit], failures
