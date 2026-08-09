"""工具信封约定（implementation-notes.md §11）。

- 32 KB 响应上限：超限截断 + 完整内容落 artifact；
- 诊断聚合计数；
- 常量：sample 200 行上限、profile 分位数固定点位。
"""

from __future__ import annotations

import json

from thermoforge_core.errors import Diagnostic, Level
from thermoforge_research.envelope import (
    PROFILE_QUANTILE_LABELS,
    SAMPLE_MAX_ROWS,
    TOOL_RESPONSE_MAX_BYTES,
    diagnostic_dicts,
    finalize_envelope,
    make_envelope,
)


def _size(env) -> int:
    return len(json.dumps(env, ensure_ascii=False).encode("utf-8"))


def test_envelope_fields_and_small_response_passthrough(tmp_path):
    env = make_envelope(
        "tf_dataset_import", ok=True, id="DS@rev_0001", status="IMPORTED",
        inputs={"a": 1}, summary={"rows": 10},
        diagnostics=[{"code": "TFDC-505", "level": "WARN", "count": 3}],
    )
    out = finalize_envelope(env, tmp_path)
    assert out["truncated"] is False
    assert set(out) == {"ok", "tool", "id", "status", "inputs", "summary",
                        "diagnostics", "artifacts", "truncated"}
    assert _size(out) <= TOOL_RESPONSE_MAX_BYTES


def test_envelope_truncates_over_32kb_and_drops_full_artifact(tmp_path):
    big_rows = [{"ts": f"2026-08-08T00:{i % 60:02d}:00Z",
                 "value": "x" * 100, "idx": i} for i in range(2000)]
    env = make_envelope("tf_dataset_sample", summary={"rows": big_rows})
    assert _size(env) > TOOL_RESPONSE_MAX_BYTES

    out = finalize_envelope(env, tmp_path)
    assert out["truncated"] is True
    assert _size(out) <= TOOL_RESPONSE_MAX_BYTES
    full = [a for a in out["artifacts"] if a["kind"] == "full_response"]
    assert len(full) == 1
    with open(full[0]["path"], encoding="utf-8") as fp:
        complete = json.load(fp)
    assert len(complete["summary"]["rows"]) == 2000  # 完整内容在 artifact


def test_envelope_truncation_is_progressive(tmp_path):
    """列表逐级收紧后仍超限：诊断仅留聚合计数，summary 指向 artifact。"""
    env = make_envelope(
        "tf_dataset_profile",
        summary={"blob": "y" * (TOOL_RESPONSE_MAX_BYTES * 2)},
        diagnostics=[{"code": "TFDC-505", "level": "WARN", "count": 7,
                      "message": "m" * 5000}],
    )
    out = finalize_envelope(env, tmp_path)
    assert out["truncated"] is True
    assert _size(out) <= TOOL_RESPONSE_MAX_BYTES
    assert out["diagnostics"][0]["count"] == 7  # 聚合计数保留
    assert "message" not in out["diagnostics"][0]


def test_diagnostic_dicts_keep_aggregated_counts():
    diags = [
        Diagnostic(code="TFDC-505", level=Level.WARN, message="m",
                   location="data", count=3),
        Diagnostic(code="TFDC-601", level=Level.REJECT, message="m2", count=12),
    ]
    out = diagnostic_dicts(diags)
    assert out[0]["count"] == 3 and out[1]["count"] == 12
    assert out[0]["level"] == "WARN" and out[1]["level"] == "REJECT"


def test_envelope_constants():
    assert TOOL_RESPONSE_MAX_BYTES == 32 * 1024
    assert SAMPLE_MAX_ROWS == 200
    assert PROFILE_QUANTILE_LABELS == ("min", "p01", "p25", "p50", "p75",
                                       "p99", "max")
