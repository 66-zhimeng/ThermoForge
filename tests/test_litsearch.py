"""tf_literature_search / litsearch 测试——HTTP 层全程 monkeypatch，不触网。

- 三个源的解析器各吃一段 fixture 报文（含缺字段/空摘要健壮性）；
- search() 的合并、DOI 去重、单源失败与全源失败；
- 工具层信封形状（ok/status/inputs/summary/diagnostics）与参数校验。
"""

from __future__ import annotations

import json

import pytest

from thermoforge_research import litsearch
from thermoforge_research.tools import TOOL_REGISTRY, ToolContext


CROSSREF_PAYLOAD = json.dumps({
    "message": {"items": [
        {"DOI": "10.1016/j.enbuild.2021.001",
         "title": ["Static  chiller\nmodel"],
         "author": [{"given": "A.", "family": "Zhang"},
                    {"given": "B", "family": "Li"}],
         "published": {"date-parts": [[2021, 5, 1]]},
         "container-title": ["Energy and Buildings"],
         "is-referenced-by-count": 42,
         "URL": "https://doi.org/10.1016/j.enbuild.2021.001"},
        {"DOI": "10.1000/x2",
         "title": ["Minimal record"],  # 无 author/published/abstract/URL
         "published": {"date-parts": [[2019]]}},
        {"title": []},  # 无标题 → 丢弃
    ]},
}).encode()

ARXIV_PAYLOAD = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2301.00001v1</id>
    <title>  Deep chiller\n networks </title>
    <published>2023-01-01T00:00:00Z</published>
    <summary> We  propose\n a network. </summary>
    <author><name>Alice</name></author>
    <arxiv:doi>10.48550/arXiv.2301.00001</arxiv:doi>
    <arxiv:journal_ref>J. Test 2023</arxiv:journal_ref>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/1999.00001v1</id>
    <title>Old paper</title>
    <published>1999-01-01T00:00:00Z</published>
    <summary>old</summary>
    <author><name>Bob</name></author>
  </entry>
</feed>
""".encode()

S2_PAYLOAD = json.dumps({"data": [
    {"title": "Static chiller model",
     "abstract": "Same paper, with abstract.",
     "year": 2021, "venue": "Energy Build.",
     "authors": [{"name": "A. Zhang"}],
     "externalIds": {"DOI": "10.1016/j.enbuild.2021.001"},
     "citationCount": 42,
     "url": "https://www.semanticscholar.org/paper/xxx"},
    {"title": "Sparse record", "abstract": None, "year": None,
     "venue": "", "authors": [], "externalIds": {},
     "citationCount": 0, "url": None},
]}).encode()


def _fake_get(url: str, *, timeout: float = 0) -> bytes:
    if "crossref" in url:
        return CROSSREF_PAYLOAD
    if "arxiv" in url:
        return ARXIV_PAYLOAD
    if "semanticscholar" in url:
        return S2_PAYLOAD
    raise AssertionError(f"未预期的 URL: {url}")


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    monkeypatch.setattr(litsearch, "http_get", _fake_get)


# ---------------------------------------------------------------- 解析器


def test_parse_crossref():
    records = litsearch._parse_crossref(CROSSREF_PAYLOAD, 10)
    assert len(records) == 2  # 无标题条目被丢弃
    first = records[0]
    assert first["title"] == "Static chiller model"  # 空白已压平
    assert first["authors"] == ["A. Zhang", "B Li"]
    assert first["year"] == 2021
    assert first["venue"] == "Energy and Buildings"
    assert first["citation_count"] == 42
    assert first["source"] == "crossref"
    minimal = records[1]
    assert minimal["abstract"] is None
    # 缺 URL 时用 DOI 回退拼链接
    assert minimal["url"] == "https://doi.org/10.1000/x2"


def test_parse_crossref_strips_jats():
    payload = json.dumps({"message": {"items": [
        {"title": ["T"], "abstract": "<jats:p>正文 <jats:italic>斜体</jats:italic></jats:p>"},
    ]}}).encode()
    assert litsearch._parse_crossref(payload, 10)[0]["abstract"] == \
        "正文 斜体"


def test_parse_arxiv():
    records = litsearch._parse_arxiv(ARXIV_PAYLOAD, 10)
    assert len(records) == 2
    first = records[0]
    assert first["title"] == "Deep chiller networks"
    assert first["year"] == 2023
    assert first["venue"] == "J. Test 2023"
    assert first["doi"] == "10.48550/arXiv.2301.00001"
    assert first["url"] == "https://doi.org/10.48550/arXiv.2301.00001"
    assert first["abstract"] == "We propose a network."
    assert first["citation_count"] is None
    # year_from 后置过滤（arXiv 日期区间语法对模糊查询不友好）
    recent = litsearch._parse_arxiv(ARXIV_PAYLOAD, 10, year_from=2000)
    assert [r["title"] for r in recent] == ["Deep chiller networks"]


def test_parse_semantic_scholar():
    records = litsearch._parse_semantic_scholar(S2_PAYLOAD, 10)
    assert len(records) == 2
    sparse = records[1]
    assert sparse["venue"] is None  # 空串归一为 None
    assert sparse["doi"] is None and sparse["url"] is None
    assert sparse["abstract"] is None


# ---------------------------------------------------------------- 检索编排


def test_search_merges_and_dedups():
    results, failures = litsearch.search("chiller power", limit=10)
    assert failures == []
    titles = [r["title"] for r in results]
    # CrossRef 的 "Static chiller model" 与 S2 的同 DOI 记录被去重
    assert titles.count("Static chiller model") == 1
    merged = results[titles.index("Static chiller model")]
    assert merged["source"] == "crossref"          # 先到的为骨架
    assert merged["abstract"] == "Same paper, with abstract."  # 缺字段由后者补
    assert len(results) == 5  # 2 + 2 + 2 − 1 次 DOI 去重


def test_search_partial_failure(monkeypatch):
    def flaky(url: str, *, timeout: float = 0) -> bytes:
        if "semanticscholar" in url:
            raise TimeoutError("timed out")
        return _fake_get(url)

    monkeypatch.setattr(litsearch, "http_get", flaky)
    results, failures = litsearch.search("chiller", limit=10)
    assert [f["source"] for f in failures] == ["semantic_scholar"]
    assert "TimeoutError" in failures[0]["error"]
    assert results  # 其余源照常返回


def test_search_all_sources_failed(monkeypatch):
    def dead(url: str, *, timeout: float = 0) -> bytes:
        raise OSError("network down")

    monkeypatch.setattr(litsearch, "http_get", dead)
    with pytest.raises(litsearch.AllSourcesFailed) as exc_info:
        litsearch.search("chiller", limit=5)
    assert len(exc_info.value.failures) == 3


def test_search_subset_sources():
    results, _ = litsearch.search("chiller", sources=("arxiv",), limit=10)
    assert {r["source"] for r in results} == {"arxiv"}


# ---------------------------------------------------------------- 工具层


def _ctx(tmp_path):
    return ToolContext(vault_root=tmp_path / "vault",
                       research_root=tmp_path / "research")


def test_tool_envelope_ok(tmp_path):
    env = TOOL_REGISTRY["tf_literature_search"](
        _ctx(tmp_path), "chiller power prediction", limit=5)
    assert env["ok"] is True and env["status"] == "OK"
    assert env["tool"] == "tf_literature_search"
    assert env["inputs"] == {"query": "chiller power prediction",
                             "sources": list(litsearch.DEFAULT_SOURCES),
                             "limit": 5, "year_from": None}
    summary = env["summary"]
    assert summary["count"] == len(summary["results"]) > 0
    assert summary["sources_failed"] == []
    first = summary["results"][0]
    assert {"title", "authors", "year", "doi", "url", "source"} <= set(first)


def test_tool_partial_failure(tmp_path, monkeypatch):
    def flaky(url: str, *, timeout: float = 0) -> bytes:
        if "arxiv" in url:
            raise TimeoutError("timed out")
        return _fake_get(url)

    monkeypatch.setattr(litsearch, "http_get", flaky)
    env = TOOL_REGISTRY["tf_literature_search"](_ctx(tmp_path), "chiller")
    assert env["ok"] is True and env["status"] == "PARTIAL"
    assert env["summary"]["sources_failed"] == ["arxiv"]
    diag = env["diagnostics"][0]
    assert diag["code"] == litsearch.TFL_SOURCE_FAILED
    assert diag["level"] == "WARN" and diag["location"] == "arxiv"


def test_tool_all_sources_failed(tmp_path, monkeypatch):
    def dead(url: str, *, timeout: float = 0) -> bytes:
        raise OSError("network down")

    monkeypatch.setattr(litsearch, "http_get", dead)
    env = TOOL_REGISTRY["tf_literature_search"](_ctx(tmp_path), "chiller")
    assert env["ok"] is False and env["status"] == "FAILED"
    assert env["diagnostics"][0]["code"] == litsearch.TFL_ALL_SOURCES_FAILED
    assert len(env["summary"]["failures"]) == 3


def test_tool_unknown_source(tmp_path):
    env = TOOL_REGISTRY["tf_literature_search"](
        _ctx(tmp_path), "chiller", sources=("cnki",))
    assert env["ok"] is False
    assert env["diagnostics"][0]["code"] == litsearch.TFL_SOURCE_UNKNOWN


def test_tool_empty_query(tmp_path):
    env = TOOL_REGISTRY["tf_literature_search"](_ctx(tmp_path), "  ")
    assert env["ok"] is False and env["status"] == "FAILED"


def test_tool_limit_clamped(tmp_path):
    env = TOOL_REGISTRY["tf_literature_search"](
        _ctx(tmp_path), "chiller", limit=999)
    assert env["inputs"]["limit"] == litsearch.MAX_LIMIT
    assert len(env["summary"]["results"]) <= litsearch.MAX_LIMIT
