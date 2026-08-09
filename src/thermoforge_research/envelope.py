"""工具返回信封（architecture.md §6、implementation-notes.md §11）。

统一结构::

    {
      "ok": true,
      "tool": "tf_dataset_import",
      "id": "DC01_2026_CHILLER@rev_0001",      # 有副作用的工具必须返回稳定 ID
      "status": "IMPORTED_WITH_WARNINGS",
      "inputs": {...},                          # 输入版本/指纹摘要
      "summary": {...},                         # 结构化摘要，非自由文本
      "diagnostics": [{"code", "level", "count", ...}],   # 聚合计数
      "artifacts": [{"kind", "path", "bytes"}],
      "truncated": false,
    }

硬约束（§11）：

- **响应体 32 KB 上限**：超限先把完整内容落 artifact（
  `<artifacts_root>/envelopes/<tool>-<timestamp>-<hash8>.json`），
  再逐步截断 `summary` 中的长列表并置 ``truncated: true``。这是
  「Agent 不直接读取海量原始数据」在接口层的强制手段。
- `tf_dataset_sample` 行数硬上限 **200**（`SAMPLE_MAX_ROWS`）。
- `tf_dataset_profile` 分位数点位固定（`PROFILE_QUANTILE_LABELS`），
  避免 Agent 通过自定义点位反复调用变相拉取全量数据。
- 诊断聚合计数：同类 (code, level, location) 合并为一条带 `count`。
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from thermoforge_core.errors import Diagnostic

TOOL_RESPONSE_MAX_BYTES = 32 * 1024  # §11 [草案：32 KB]
SAMPLE_MAX_ROWS = 200  # §11 [草案：200 行]
PROFILE_QUANTILE_LABELS = ("min", "p01", "p25", "p50", "p75", "p99", "max")

# 截断时列表保留的条目数（逐级收紧）
_TRIM_STEPS = (20, 5, 0)


def diagnostic_dicts(diagnostics: Iterable[Diagnostic]) -> list[dict[str, Any]]:
    """Diagnostic → 信封诊断条目（保持聚合计数语义，§7.1）。"""
    return [
        {
            "code": d.code,
            "level": d.level.value,
            "location": d.location,
            "count": d.count,
            "message": d.message,
        }
        for d in diagnostics
    ]


def make_envelope(
    tool: str,
    *,
    ok: bool = True,
    id: str | None = None,  # noqa: A002 - 信封字段名即契约
    status: str | None = None,
    inputs: Mapping[str, Any] | None = None,
    summary: Mapping[str, Any] | None = None,
    diagnostics: Sequence[Mapping[str, Any]] = (),
    artifacts: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """构造信封（尚未做 32 KB 截断，见 `finalize_envelope`）。"""
    return {
        "ok": bool(ok),
        "tool": tool,
        "id": id,
        "status": status,
        "inputs": dict(inputs or {}),
        "summary": dict(summary or {}),
        "diagnostics": [dict(d) for d in diagnostics],
        "artifacts": [dict(a) for a in artifacts],
        "truncated": False,
    }


def _size_bytes(env: Mapping[str, Any]) -> int:
    return len(json.dumps(env, ensure_ascii=False).encode("utf-8"))


def _write_full_envelope(artifacts_root: Path, env: Mapping[str, Any]) -> Path:
    """完整信封落 artifact（临时文件 + os.replace，§10.1）。"""
    out_dir = artifacts_root / "envelopes"
    out_dir.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(env, ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    path = out_dir / f"{env['tool']}-{stamp}-{digest}.json"
    fd, tmp = tempfile.mkstemp(dir=out_dir, prefix=".tmp_", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fp:
        json.dump(env, fp, ensure_ascii=False, sort_keys=True, indent=2)
        fp.write("\n")
    os.replace(tmp, path)
    return path


def _trim_summary_lists(summary: dict[str, Any], keep: int) -> dict[str, Any]:
    """把 summary 中的长列表截断为前 `keep` 条并附总数标记。"""

    def trim(value: Any) -> Any:
        if isinstance(value, list):
            if len(value) > keep:
                trimmed = [trim(v) for v in value[:keep]]
                trimmed.append(f"... (truncated, total={len(value)})")
                return trimmed
            return [trim(v) for v in value]
        if isinstance(value, dict):
            return {k: trim(v) for k, v in value.items()}
        return value

    return {k: trim(v) for k, v in summary.items()}


def finalize_envelope(
    env: dict[str, Any],
    artifacts_root: str | Path,
) -> dict[str, Any]:
    """执行 32 KB 响应上限（§11）。

    未超限原样返回；超限则：完整信封落 artifact → 逐级截断
    `summary` 中的列表 → 仍超限再聚合 `diagnostics` → 置
    ``truncated: true`` 并在 artifacts 中给出完整内容位置。
    """
    if _size_bytes(env) <= TOOL_RESPONSE_MAX_BYTES:
        return env

    full_path = _write_full_envelope(Path(artifacts_root), env)
    env["artifacts"] = list(env["artifacts"]) + [{
        "kind": "full_response",
        "path": str(full_path),
        "bytes": full_path.stat().st_size,
    }]
    env["truncated"] = True

    for keep in _TRIM_STEPS:
        env["summary"] = _trim_summary_lists(dict(env["summary"]), keep)
        if _size_bytes(env) <= TOOL_RESPONSE_MAX_BYTES:
            return env

    # 诊断兜底：仅保留聚合计数（code/level/count），丢弃长 message
    env["diagnostics"] = [
        {"code": d.get("code"), "level": d.get("level"),
         "count": d.get("count", 1)}
        for d in env["diagnostics"]
    ]
    env["summary"] = {
        "_note": "summary 已整体移除，完整内容见 artifacts 中的 full_response",
    }
    return env
