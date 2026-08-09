"""错误码注册表与 conventions.md §7 的一致性测试。

直接解析 docs/conventions.md 的错误码表格，与 ERROR_REGISTRY 逐条比对，
防止文档与实现漂移。同类问题聚合计数等结构要求见 §7.1。
"""

import re
from pathlib import Path

import pytest

from thermoforge_core.errors import ERROR_REGISTRY, Diagnostic, Level

ROW_RE = re.compile(r"^\| (TF[A-Z]+-\d+) \| `([A-Z_]+)` \| (.+?) \| (.+?) \|$")
LEVEL_TOKEN_RE = re.compile(r"ERROR|REJECT|WARN")


def _parse_doc_table() -> dict[str, tuple[str, str]]:
    """从 conventions.md §7 解析 {code: (name, level_first_token)}。"""
    text = (Path(__file__).resolve().parents[1] / "docs" / "conventions.md").read_text(
        encoding="utf-8"
    )
    rows: dict[str, tuple[str, str]] = {}
    for line in text.splitlines():
        m = ROW_RE.match(line.strip())
        if not m:
            continue
        code, name, _trigger, level_cell = m.groups()
        token = LEVEL_TOKEN_RE.search(level_cell)
        assert token, f"{code} 的级别列无法解析: {level_cell!r}"
        rows[code] = (name, token.group(0))
    return rows


def test_registry_matches_doc_full_set():
    doc_rows = _parse_doc_table()
    assert len(doc_rows) >= 50, "解析到的错误码行数异常，文档格式可能已变化"
    assert set(doc_rows) == set(ERROR_REGISTRY)
    assert len(ERROR_REGISTRY) == 61


def test_registry_names_match_doc():
    for code, (name, _level) in _parse_doc_table().items():
        assert ERROR_REGISTRY[code].name == name, f"{code} 名称与文档不一致"


def test_registry_levels_match_doc():
    for code, (_name, level) in _parse_doc_table().items():
        entry_level = ERROR_REGISTRY[code].level
        first_token = LEVEL_TOKEN_RE.search(entry_level)
        assert first_token and first_token.group(0) == level, f"{code} 级别与文档不一致"


def test_diagnostic_structure():
    d = Diagnostic(
        code="TFDC-401",
        level=Level.ERROR,
        message="未登记的单位",
        location="variables:row=3",
        count=5,
    )
    assert d.code == "TFDC-401"
    assert d.level is Level.ERROR
    assert d.count == 5


def test_diagnostic_rejects_unknown_code():
    with pytest.raises(ValueError):
        Diagnostic(code="TFDC-999", level=Level.ERROR, message="x")


def test_diagnostic_rejects_bad_count():
    with pytest.raises(ValueError):
        Diagnostic(code="TFDC-401", level=Level.ERROR, message="x", count=0)
