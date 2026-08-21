"""用量归账查询工具（tf_usage_overview / tf_goal_usage / tf_experiment_usage）。

外部 Agent 经 MCP 查到的数字与网页「用量与留痕」页签同源：这里用
tmp_path 造一份最小研究现场（目标+假设+实验+规划留痕+一条会话），
验证三个工具的信封、归因总量与思维链字段。
"""

from __future__ import annotations

import json

from thermoforge_research.tools import (
    tf_experiment_usage,
    tf_goal_create,
    tf_goal_usage,
    tf_hypothesis_create,
    tf_usage_overview,
)

from phase2_helpers import FEATURES, TARGET
from phase34_helpers import make_ctx


def _goal(ctx):
    env = tf_goal_create(ctx, {
        "name": "冷水机输入功率模型",
        "object_model": "chiller.v1",
        "purpose": "optimization",
        "target": TARGET,
        "candidate_inputs": list(FEATURES),
        "acceptance": {"cvrmse_max": 0.5},
    })
    assert env["ok"], env["summary"]
    return env["id"]


def _seed_research(ctx):
    """目标 + 假设 + 实验目录（report/planner_trace）+ 一条会话 jsonl。"""
    goal_id = _goal(ctx)
    hyp = tf_hypothesis_create(ctx, goal_id, "ridge 基线足够好")
    assert hyp["ok"]
    exp_id = "EXP-0001"
    exp_dir = ctx.research_root / "experiments" / exp_id
    exp_dir.mkdir(parents=True)
    (exp_dir / "report.json").write_text(json.dumps({
        "experiment_id": exp_id,
        "goal_id": goal_id,
        "hypothesis_id": hyp["id"],
    }, ensure_ascii=False), encoding="utf-8")
    (exp_dir / "planner_trace.json").write_text(json.dumps({
        "round_index": 0,
        "reasoning": "先跑 ridge 基线，便宜且可解释",
        "cot": "端点思维链：特征共线性高，正则化更稳",
        "raw_reply": '{"statement": "ridge 基线足够好"}',
        "usage": {"prompt_tokens": 13149, "completion_tokens": 3203,
                  "total_tokens": 16352},
        "cost": 0.12,
    }, ensure_ascii=False), encoding="utf-8")
    session = ctx.research_root / "agent_sessions" / "s.jsonl"
    session.parent.mkdir(parents=True, exist_ok=True)
    session.write_text("".join(
        json.dumps(event, ensure_ascii=False) + "\n" for event in [
            {"at": "2026-08-19T03:00:00+00:00", "type": "message",
             "role": "user", "content": "把 EXP-0001 跑起来"},
            {"at": "2026-08-19T03:00:05+00:00", "type": "message",
             "role": "assistant", "content": "",
             "usage": {"prompt_tokens": 100, "completion_tokens": 20,
                       "total_tokens": 120},
             "cost": 0.01, "reasoning": "先确认实验登记没毛病",
             "tool_calls": [{"id": "c1", "name": "tf_experiment_run",
                             "arguments": '{"experiment_id": "EXP-0001"}'}]},
            {"at": "2026-08-19T03:00:06+00:00", "type": "tool_result",
             "tool": "tf_experiment_run", "ok": True, "id": exp_id},
        ]), encoding="utf-8")
    return goal_id, hyp["id"], exp_id


def test_usage_overview_envelope_and_totals(tmp_path):
    ctx, _ref = make_ctx(tmp_path)
    goal_id, _hyp_id, exp_id = _seed_research(ctx)

    env = tf_usage_overview(ctx)

    assert env["ok"] is True and env["status"] == "OK"
    summary = env["summary"]
    assert summary["sessions_scanned"] == 1
    assert summary["planner_traces"] == 1
    assert summary["orphan_planner_traces"] == 0
    # 规划留痕 16352 + 会话 120，总量在两块各自守恒
    assert summary["planner_tokens"] == {"prompt": 13149, "completion": 3203}
    assert summary["session_tokens"] == {"prompt": 100, "completion": 20,
                                         "turns": 1}
    row = next(g for g in summary["goals"] if g["goal_id"] == goal_id)
    assert row["prompt_tokens"] == 13149 + 100
    assert row["completion_tokens"] == 3203 + 20
    assert row["experiments"] == 1
    assert summary["unassigned"]["prompt_tokens"] == 0
    assert summary["attribution"]


def test_goal_usage_entries_with_reasoning_labels(tmp_path):
    ctx, _ref = make_ctx(tmp_path)
    goal_id, hyp_id, exp_id = _seed_research(ctx)

    env = tf_goal_usage(ctx, goal_id)

    assert env["ok"] is True
    summary = env["summary"]
    assert summary["goal"] == {"goal_id": goal_id,
                               "name": "冷水机输入功率模型",
                               "status": summary["goal"]["status"]}
    assert summary["total"]["prompt_tokens"] == 13149 + 100
    hyp_row = next(h for h in summary["hypotheses"] if h["id"] == hyp_id)
    assert hyp_row["statement"].startswith("ridge 基线")
    exp_row = next(e for e in summary["experiments"] if e["id"] == exp_id)
    assert exp_row["prompt_tokens"] == 13149 + 100
    # 逐条留痕：规划条目带两段标签化思维链；目标级不带 raw_reply
    planner = next(e for e in summary["entries"] if e["source"] == "规划")
    labels = [r["label"] for r in planner["reasoning"]]
    assert labels == ["思维链（端点 CoT）", "规划理由"]
    assert "raw_reply" not in planner
    session_entry = next(e for e in summary["entries"] if e["source"] == "副驾")
    assert session_entry["reasoning"][0]["text"] == "先确认实验登记没毛病"


def test_experiment_usage_includes_raw_reply(tmp_path):
    ctx, _ref = make_ctx(tmp_path)
    goal_id, hyp_id, exp_id = _seed_research(ctx)

    env = tf_experiment_usage(ctx, exp_id)

    assert env["ok"] is True
    summary = env["summary"]
    assert summary["goal_id"] == goal_id
    assert summary["hypothesis_id"] == hyp_id
    assert summary["total"]["prompt_tokens"] == 13149 + 100
    planner = next(e for e in summary["entries"] if e["source"] == "规划")
    assert planner["raw_reply"] == '{"statement": "ridge 基线足够好"}'


def test_orphan_planner_round_rolls_up_to_goal(tmp_path):
    """没产出实验的规划轮次（planner_rounds/）归到目标级并在总览计数。"""
    ctx, _ref = make_ctx(tmp_path)
    goal_id = _goal(ctx)
    directory = ctx.research_root / "planner_rounds" / goal_id
    directory.mkdir(parents=True)
    (directory / "round_001_planner_stop_103000123456.json").write_text(
        json.dumps({
            "goal_id": goal_id, "round_number": 1,
            "orphan_kind": "planner_stop",
            "reasoning": "瓶颈是缺阀位测点",
            "cot": "端点思维链：再拟合也是浪费",
            "usage": {"prompt_tokens": 500, "completion_tokens": 60,
                      "total_tokens": 560},
        }, ensure_ascii=False), encoding="utf-8")

    env = tf_goal_usage(ctx, goal_id)

    assert env["ok"] is True
    summary = env["summary"]
    assert summary["total"]["prompt_tokens"] == 500
    entry = summary["entries"][0]
    assert "planner_stop" in entry["detail"] or "主动停止" in entry["detail"]

    overview = tf_usage_overview(ctx)["summary"]
    assert overview["orphan_planner_traces"] == 1
    assert overview["planner_tokens"]["prompt"] == 500


def test_usage_unknown_ids_return_error_envelope(tmp_path):
    ctx, _ref = make_ctx(tmp_path)

    goal_env = tf_goal_usage(ctx, "RG-9999")
    assert goal_env["ok"] is False

    exp_env = tf_experiment_usage(ctx, "EXP-9999")
    assert exp_env["ok"] is False
