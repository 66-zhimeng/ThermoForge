"""外部副驾的大响应仍可解释真实依据；不得依赖读取本机文件的能力。"""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

from thermoforge_v2.mcp import MAX_RESPONSE_BYTES, _invoke
from thermoforge_v2.reports import build_report


def large_report():
    snapshot = {"run": {"run_id": "RUN-large", "status": "completed",
                         "protocol": {"fingerprint": "frozen-protocol"}},
                "tracks": [], "sources": [], "ideas": [], "jobs": [], "reports": []}
    for index in range(6):
        tid = "main" if index == 0 else f"candidate-{index}"
        snapshot["tracks"].append({"track_id": tid, "role": "main" if index == 0 else "candidate",
                                    "status": "completed", "usage": None})
        snapshot["sources"].append({"id": f"S-{index}", "track_id": tid, "title": f"来源论文{index}",
            "url": f"https://example.org/papers/{index}", "read_scope": "abstract", "kind": "paper",
            "content": "未进入紧凑摘要的论文全文。" * 3000})
        snapshot["ideas"].append({"id": f"I-{index}", "track_id": tid, "statement": f"路线{index}",
            "reason": f"从来源论文{index}的残差观察提出", "origin": "literature", "source_ids": [f"S-{index}"]})
        snapshot["jobs"].append({"id": f"J-{index}", "track_id": tid, "idea_id": f"I-{index}",
            "status": "completed", "result": {"experiment_id": f"EXP-{index}", "protocol_fingerprint": "frozen-protocol",
                "feedback_surface": "validate", "metrics": {"CVRMSE": .1 + index / 100}}})
        for phase in ("旧", "最新"):
            snapshot["reports"].append({"id": f"R-{index}-{phase}", "track_id": tid, "kind": "track",
                "title": f"轨迹{index}报告", "summary": f"{phase}摘要：该路线仍需消融验证",
                "limitations": "只验证了一个时间窗口", "idea_ids": [f"I-{index}"], "job_ids": [f"J-{index}"],
                "body": "未进入紧凑摘要的报告正文。" * 3000})
    snapshot["jobs"].append({"id": "J-failed", "track_id": "main", "status": "failed",
                              "result": {"error": "模型接口不兼容", "failure_category": "implementation"}})
    report = build_report(snapshot)
    report["artifacts"] = [{"format": "html", "path": "research/runs/RUN-large/reports/team.html"}]
    return report


def invoke(tmp_path, data, name="tf_v2_report"):
    body = _invoke(name, lambda _: data, SimpleNamespace(root=tmp_path))
    assert len(body.encode("utf-8")) <= MAX_RESPONSE_BYTES
    return body, json.loads(body)


def test_large_real_report_retains_all_six_latest_summaries_sources_and_comparison(tmp_path):
    report = large_report()
    before = deepcopy(report)
    body, result = invoke(tmp_path, report)
    assert result["ok"] and result["truncated"]
    summary = result["summary"]
    assert summary["run_id"] == "RUN-large" and summary["status"] == "completed"
    assert summary["title"] == report["title"]
    assert summary["summary"] == {"tracks": 6, "ideas": 6, "jobs": 7, "sources": 6}
    assert len(summary["latest_reports"]) == 6
    assert all(r["summary"].startswith("最新摘要") for r in summary["latest_reports"])
    assert summary["latest_reports"][0]["limitations"] == "只验证了一个时间窗口"
    assert summary["latest_reports"][0]["job_ids"] == ["J-0"]
    assert summary["sources"][0]["url"] == "https://example.org/papers/0"
    assert summary["sources"][0]["read_scope"] == "abstract"
    assert summary["ideas"][0]["origin"] == "literature"
    assert summary["ideas"][0]["source_ids"] == ["S-0"]
    assert "来源论文0" in summary["ideas"][0]["reason"]
    assert summary["comparison"]["groups"][0]["best_per_metric"]["CVRMSE"] == {
        "direction": "min", "value": .1, "job_ids": ["J-0"]}
    assert summary["comparison"]["excluded_count"] == 1
    assert summary["usage_totals"]["cost"] is None
    assert summary["usage_totals"]["total_tokens"] is None
    assert "论文全文" not in body and "报告正文" not in body
    assert result["artifacts"][1] == report["artifacts"][0]
    complete = json.loads(Path(result["artifacts"][0]["path"]).read_text(encoding="utf-8"))
    assert complete == {"ok": True, "tool": "tf_v2_report", "summary": report}
    assert report == before
    assert not list((tmp_path / "tool_artifacts").glob("*.tmp"))


def test_status_retains_budget_progress_and_recent_failures_without_invented_usage(tmp_path):
    snapshot = {"run": {"run_id": "RUN-status", "status": "paused", "version": 9,
        "config": {"token_budget": 1000000, "max_experiments": 20},
        "tokens_used": 250123, "experiments_reserved": 3, "experiments_settled": 2},
        "tracks": [{"track_id": "main", "status": "failed", "phase": "experiment",
                    "turns": 2, "usage": None, "error": "连接已断开"}],
        "jobs": [{"id": "old", "status": "failed", "result": {"error": "旧错误"}},
                 {"id": "new", "status": "failed", "result": {"error": "最新接口错误"}}],
        "messages": [{"body": "巨大历史消息" * 20000}]}
    _, result = invoke(tmp_path, snapshot, "tf_v2_status")
    summary = result["summary"]
    assert summary["budget"]["token_budget"] == 1000000
    assert summary["budget"]["max_turns"] is None
    assert summary["progress"]["tokens_used"] == 250123
    assert summary["progress"]["experiments_settled"] == 2
    assert summary["progress"]["jobs_by_status"] == {"failed": 2}
    assert summary["tracks"][0]["error"] == "连接已断开"
    assert summary["tracks"][0]["usage"] is None
    assert summary["recent_errors"][0]["error"] == "最新接口错误"
    assert summary["summary"]["sources"] is None
    assert summary["comparison"]["excluded_count"] is None
    assert summary["sampling"]["sources"]["available"] is None


def test_multibyte_long_fields_and_many_tracks_stay_within_byte_limit(tmp_path):
    report = large_report()
    report["reports"] = [{"id": f"R-{i}", "track_id": f"t-{i}", "summary": "中文😊" * 2000,
                           "limitations": "限制条件" * 2000, "job_ids": [f"J-{j}" for j in range(100)]}
                          for i in range(80)]
    report["sources"] *= 30
    report["ideas"] *= 30
    body, result = invoke(tmp_path, report)
    assert "截断" in body
    sampling = result["summary"]["sampling"]
    assert sampling["latest_reports"]["available"] == 80
    assert 0 < sampling["latest_reports"]["returned"] < 80
    assert sampling["sources"]["available"] == 180
    assert sampling["sources"]["returned"] < 180
    assert result["summary"]["comparison"]["groups"]


def test_nested_large_error_still_has_hard_byte_ceiling(tmp_path):
    large = {f"k-{n}": {f"i-{i}": ["错误信息😊" * 1000] * 10 for i in range(10)} for n in range(10)}
    _, result = invoke(tmp_path, {"run": {"run_id": "RUN-nested", "error": large}, "padding": "大" * 40000}, "tf_v2_status")
    assert result["ok"] and result["truncated"]
    assert result["summary"]["run_id"] == "RUN-nested"


def test_small_response_is_unchanged_and_does_not_write_artifact(tmp_path):
    data = {"run_id": "RUN-small", "status": "queued", "unknown_cost": None}
    _, result = invoke(tmp_path, data, "tf_v2_start")
    assert result == {"ok": True, "tool": "tf_v2_start", "summary": data}
    assert not (tmp_path / "tool_artifacts").exists()


def test_error_response_also_obeys_multibyte_limit(tmp_path):
    def fail(_):
        raise RuntimeError("错误😊" * 20000)
    body = _invoke("tf_v2_status", fail, SimpleNamespace(root=tmp_path))
    assert len(body.encode("utf-8")) <= MAX_RESPONSE_BYTES
    result = json.loads(body)
    assert not result["ok"] and "截断" in result["error"]
