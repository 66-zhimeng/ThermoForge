"""真实 Codex 六进程/两轮动态工具验收，会消耗当前账号的 Codex 用量。

运行：.venv/Scripts/python scripts/verify_v2_codex.py
默认保持本机 Codex 模型设置，再将首实例报告的模型固定给其余实例。
仅交换随机协议探针，不读取或训练用户数据，也不是研究质量/效果基准。
证据写入 gitignored research/v2_validation/；每轮最多 180 秒。
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import time
import uuid

from thermoforge_v2.codex import CodexConfig, CodexSession, VALIDATED_CODEX_VERSION


TOOL_SPEC = {
    "name": "tf_validation_probe",
    "description": "ThermoForge 协议验收探针。返回独立回执，第二轮校验同一会话的回执记忆。",
    "inputSchema": {
        "type": "object", "additionalProperties": False,
        "properties": {
            "nonce": {"type": "string"}, "phase": {"type": "integer", "enum": [1, 2]},
            "previous_receipt": {"type": "string"},
        },
        "required": ["nonce", "phase"],
    },
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def error_text(exc) -> str:
    value = str(exc)
    value = re.sub(r"(?i)(bearer\s+)[^\s,;]+", r"\1[REDACTED]", value)
    value = re.sub(r"\bsk-[A-Za-z0-9_-]{12,}", "[REDACTED]", value)
    value = re.sub(r"(?i)(https?://)[^\s/@:]+:[^\s/@]+@", r"\1[REDACTED]@", value)
    return value[:2000]


def write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                         encoding="utf-8", newline="\n")
    os.replace(temporary, path)


async def verify(args) -> tuple[dict, Path]:
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    folder = Path(args.output_root).resolve() / run_stamp
    folder.mkdir(parents=True)
    evidence_path = folder / "codex-smoke.json"
    evidence = {
        "kind": "real_codex_protocol_smoke", "started_at": now(),
        "scientific_quality_benchmark": False,
        "scope": "六个独立隐藏 Codex 进程，各自两轮真实动态工具回传；不训练用户数据。",
        "codex_protocol_version": args.expected_version,
        "requested_model": args.model, "model_selection": "explicit" if args.model else "local_codex_default",
        "turn_timeout_seconds": args.turn_timeout, "peak_concurrent_turns": 0,
        "instances": [], "errors": [], "passed": False,
    }
    instances = []
    live_turns = 0

    def instance(role: str, model: str | None):
        nonce, receipt = uuid.uuid4().hex, uuid.uuid4().hex
        row = {"role": role, "pid": None, "thread_id": None, "model": None,
               "turns": [], "probe_calls": [], "events": {}, "errors": [],
               "model_reroutes": [], "closed": False, "phase": 0}
        event_counts = Counter()
        evidence["instances"].append(row)

        async def handle(name, inputs):
            valid = (name == TOOL_SPEC["name"] and inputs.get("nonce") == nonce
                     and inputs.get("phase") == row["phase"])
            if row["phase"] == 2:
                valid = valid and inputs.get("previous_receipt") == receipt
            row["probe_calls"].append({"phase": row["phase"], "valid": valid, "at": now()})
            if not valid:
                return {"ok": False, "error": "探针参数不匹配；请使用当前会话中已有的 nonce 和回执。"}
            return {"ok": True, "role": role, "phase": row["phase"], "receipt": receipt,
                    "message": "协议探针通过；用一句话报告，不调用其他工具。"}

        async def events(event):
            method = event.get("method", "")
            event_counts[method] += 1
            row["events"] = dict(event_counts)
            if method == "model/rerouted":
                p = event.get("params") or {}
                row["model_reroutes"].append({k: p.get(k) for k in ("fromModel", "toModel", "reason")})

        session = CodexSession(CodexConfig(
            cwd=folder / "workspaces" / role, model=model,
            expected_version=args.expected_version,
            turn_timeout=args.turn_timeout, tool_timeout=10,
            developer_instructions=(
                "你是 ThermoForge 的有界协议验收智能体。只调用 tf_validation_probe，"
                "每轮按用户指令调用一次，之后用不超过40个汉字报告。"
                "保留会话中的 nonce 和工具回执供下一轮验证。"
                "无需查资料、研究代码、创建文件或调用其他工具。"
                "这不是科学实验，不要做设备模型训练。")), [TOOL_SPEC], handle, events)
        instances.append((session, row, nonce))
        return session, row, nonce

    async def start(item):
        session, row, _ = item
        try:
            info = await session.start()
            row.update({k: info.get(k) for k in ("pid", "thread_id", "model", "model_provider")})
            row["permission_profile"] = info.get("permission_profile")
            print(json.dumps({"event": "instance_ready", "role": row["role"],
                              "pid": row["pid"], "thread_id": row["thread_id"], "model": row["model"]}), flush=True)
        except Exception as exc:
            row["errors"].append(error_text(exc))
            raise

    async def turn(item, phase):
        nonlocal live_turns
        session, row, nonce = item
        row["phase"] = phase
        started = time.monotonic()
        live_turns += 1
        evidence["peak_concurrent_turns"] = max(evidence["peak_concurrent_turns"], live_turns)
        prompt = (f"你是 {row['role']}。这是第一轮协议验收。请调用 tf_validation_probe，"
                  f"nonce={nonce}，phase=1。记住工具返回的 receipt，之后一句话报告。" if phase == 1 else
                  "这是同一会话的第二轮验收。调用 tf_validation_probe，phase=2，"
                  "nonce 使用首轮用户提供的值，previous_receipt 使用首轮工具返回的 receipt。"
                  "不要读取文件或猜测，调用成功后一句话报告。")
        try:
            result = await session.turn(prompt)
            record = {"phase": phase, "status": result.status, "thread_id": result.thread_id,
                      "turn_id": result.turn_id, "text": result.text[:2000], "usage": result.usage,
                      "error": error_text(result.error) if result.error else None,
                      "duration_seconds": round(time.monotonic() - started, 3)}
            row["turns"].append(record)
            if result.status != "completed":
                row["errors"].append(record["error"] or f"turn status: {result.status}")
            print(json.dumps({"event": "turn_finished", "role": row["role"], "phase": phase,
                              "status": result.status, "duration_seconds": record["duration_seconds"],
                              "probe_valid": any(p["phase"] == phase and p["valid"] for p in row["probe_calls"])}), flush=True)
        except Exception as exc:
            row["errors"].append(error_text(exc))
            print(json.dumps({"event": "turn_failed", "role": row["role"], "phase": phase,
                              "error": error_text(exc)}), flush=True)
        finally:
            live_turns -= 1
            write_json(evidence_path, evidence)

    try:
        # 不替用户选模型：读取主实例实际设置，再将同一模型固定给其余实例。
        main = instance("main", args.model)
        await start(main)
        fixed_model = main[1]["model"]
        if not fixed_model:
            raise RuntimeError("主实例未返回模型标识，无法确认六实例使用同一模型。")
        evidence["fixed_model"] = fixed_model
        candidates = [instance(f"candidate-{i}", fixed_model) for i in range(1, 6)]
        # Windows 权限 helper 的初始化先顺序建立；研究 turn 仍由下方六路并发。
        for item in candidates:
            await start(item)
        if any(row["model"] != fixed_model for _, row, _ in instances):
            raise RuntimeError("六实例模型标识不一致，拒绝继续验收。")
        for phase in (1, 2):
            await asyncio.gather(*(turn(item, phase) for item in instances))
            if any(row["errors"] or not any(p["phase"] == phase and p["valid"] for p in row["probe_calls"])
                   for _, row, _ in instances):
                raise RuntimeError(f"第 {phase} 轮未全部通过；停止后续轮次，不自动改换模型或重试。")
        rows = evidence["instances"]
        evidence["passed"] = (
            len({r["pid"] for r in rows}) == 6 and len({r["thread_id"] for r in rows}) == 6
            and evidence["peak_concurrent_turns"] == 6
            and all(len(r["turns"]) == 2 and not r["errors"] and not r["model_reroutes"]
                    and all(t["status"] == "completed" and t["thread_id"] == r["thread_id"] for t in r["turns"])
                    and all(any(c["phase"] == phase and c["valid"] for c in r["probe_calls"]) for phase in (1, 2))
                    for r in rows)
        )
    except Exception as exc:
        evidence["errors"].append(error_text(exc))
    finally:
        closed = await asyncio.gather(*(s.close() for s, _, _ in instances), return_exceptions=True)
        for (_, row, _), result in zip(instances, closed):
            row["closed"] = not isinstance(result, BaseException)
            if isinstance(result, BaseException):
                row["errors"].append("关闭失败：" + error_text(result))
                evidence["passed"] = False
        # 用量为各会话末次累计 total 的合计；缺失不是零，不混加多个累计事件。
        totals = [(row["turns"][-1].get("usage") or {}).get("total", {}).get("totalTokens")
                  if row["turns"] else None for _, row, _ in instances]
        evidence["tokens_used"] = sum(totals) if totals and all(isinstance(t, int) for t in totals) else None
        evidence["usage_available_instances"] = sum(isinstance(t, int) for t in totals)
        evidence["finished_at"] = now()
        write_json(evidence_path, evidence)
    return evidence, evidence_path


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="仅在明确指定时覆盖本机 Codex 模型；不会自动选择替代模型")
    parser.add_argument("--turn-timeout", type=float, default=180)
    parser.add_argument("--expected-version", default=VALIDATED_CODEX_VERSION,
                        help="明确指定待验证的 CLI 版本；不修改全局安装，也不跳过权限检查")
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parents[1] / "research" / "v2_validation")
    args = parser.parse_args()
    if not 1 <= args.turn_timeout <= 180:
        parser.error("--turn-timeout 必须介于 1 和 180 秒")
    result, path = asyncio.run(verify(args))
    print(json.dumps({"passed": result["passed"], "evidence": str(path),
                      "tokens_used": result["tokens_used"], "errors": result["errors"],
                      "scientific_quality_benchmark": False}, ensure_ascii=False), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
