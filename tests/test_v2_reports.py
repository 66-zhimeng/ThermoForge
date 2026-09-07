"""V2 报告必须保留证据边界、负结果和真实用量，不能用排名伪造结论。"""

from copy import deepcopy
import json
import re

import pytest

from thermoforge_v2.reports import build_report, render_html, render_markdown, to_document


@pytest.fixture()
def snapshot():
    return {
        "run": {"run_id": "RUN-001", "status": "running", "config": {"goal_id": "RG-0001"},
                "protocol": {"fingerprint": "protocol-a", "dataset_ref": "test@rev_0001",
                             "metrics": {"primary": "CVRMSE"}, "validation": {"method": "time"}}},
        "tracks": [{"track_id": "main", "role": "main", "status": "running", "turns": 2,
                    "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}},
                   {"track_id": "candidate-1", "role": "candidate", "status": "waiting_experiment",
                    "usage": {"input_tokens": 15, "output_tokens": 25, "total_tokens": 40}},
                   {"track_id": "candidate-2", "role": "candidate", "status": "failed",
                    "error": "连接中断"}],
        "sources": [{"id": "SRC-1", "track_id": "candidate-1", "kind": "paper",
                     "title": "真实参考论文", "read_scope": "abstract", "verification": "retrieved",
                     "url": "https://arxiv.org/abs/2402.05120"}],
        "ideas": [{"id": "IDEA-1", "track_id": "candidate-1", "origin": "literature",
                   "statement": "约束残差结构", "reason": "由论文和先前残差观察提出",
                   "prediction": "验证误差降低", "falsification": "误差未降低", "source_ids": ["SRC-1"]},
                  {"id": "IDEA-2", "track_id": "main", "origin": "conjecture",
                   "statement": "自主提出分段模型", "reason": "检查未解释的工况边界",
                   "prediction": "边界段误差降低", "source_ids": []}],
        "jobs": [{"id": "JOB-1", "track_id": "candidate-1", "idea_id": "IDEA-1", "status": "completed",
                  "result": {"experiment_id": "EXP-1", "protocol_fingerprint": "protocol-a",
                             "feedback_surface": "validate", "metrics": {"CVRMSE": 0.1, "R2": 0.9}}},
                 {"id": "JOB-2", "track_id": "candidate-1", "idea_id": "IDEA-1", "status": "failed",
                  "result": {"error": "接口校验失败", "failure_category": "implementation"}}],
        "findings": [{"id": "F-1", "track_id": "candidate-1", "statement": "完成基线验证",
                      "interpretation": "还需消融", "limitations": "一个验证窗口",
                      "idea_ids": ["IDEA-1"], "job_ids": ["JOB-1"]}],
        "decisions": [{"id": "D-1", "track_id": "main", "action": "retain",
                       "reason": "保留用于消融", "finding_ids": ["F-1"], "job_ids": ["JOB-1"]}],
        "reports": [{"id": "REPORT-1", "track_id": "candidate-1", "kind": "track",
                     "title": "候选研究总结", "summary": "有待重复", "body": "下一步消融。",
                     "idea_ids": ["IDEA-1"], "job_ids": ["JOB-1"], "finding_ids": ["F-1"],
                     "limitations": "尚未复现"}],
    }


def test_report_preserves_lineage_and_pre_experiment_hypothesis(snapshot):
    before = deepcopy(snapshot)
    report = build_report(snapshot)
    assert snapshot == before
    assert report["summary"]["tracks"] == 3
    assert not report["lineage"]["missing_references"]
    assert {"from": "SRC-1", "to": "IDEA-1", "relation": "source"} in report["lineage"]["edges"]
    assert {"from": "JOB-1", "to": "F-1", "relation": "experiment"} in report["lineage"]["edges"]
    markdown = render_markdown(report)
    for expected in ["由论文和先前残差观察提出", "实验前预测", "反证条件", "abstract",
                     "自主假说", "保留用于消融", "尚未复现", "连接中断"]:
        assert expected in markdown


def test_failures_do_not_become_zero_score_or_disproved_hypotheses(snapshot):
    report = build_report(snapshot)
    group = report["comparability"]["groups"][0]
    assert [r["job_id"] for r in group["rows"]] == ["JOB-1"]
    assert group["best_per_metric"]["CVRMSE"]["job_ids"] == ["JOB-1"]
    negative = report["negative_results"][0]
    assert negative["kind"] == "implementation"
    assert "不能据此认定假说被证伪" in negative["interpretation"]
    assert report["comparability"]["excluded"][0]["job_id"] == "JOB-2"


@pytest.mark.parametrize("change", [
    {"protocol_fingerprint": "other"}, {"protocol_fingerprint": None},
    {"feedback_surface": "A"}, {"metrics": {"R2": float("nan")}},
])
def test_noncomparable_or_unmeasured_results_are_excluded(snapshot, change):
    snapshot["jobs"][0]["result"].update(change)
    report = build_report(snapshot)
    assert not report["comparability"]["groups"]
    assert len(report["comparability"]["excluded"]) == 2


def test_partial_unknown_usage_never_reports_zero(snapshot):
    report = build_report(snapshot)
    assert report["usage"]["totals"]["total_tokens"] is None
    assert report["usage"]["totals"]["cost"] is None
    snapshot["tracks"].pop()
    assert build_report(snapshot)["usage"]["totals"]["total_tokens"] == 70


def test_codex_backend_usage_and_team_evidence_are_preserved(snapshot):
    snapshot["tracks"] = snapshot["tracks"][:1]
    snapshot["tracks"][0]["usage"] = {"total": {"inputTokens": 21, "outputTokens": 34, "totalTokens": 55},
                                         "last": {"totalTokens": 20}}
    snapshot["decisions"][0]["evidence_ids"] = ["F-1"]
    report = build_report(snapshot)
    assert report["usage"]["totals"]["total_tokens"] == 55
    assert report["usage"]["totals"]["cost"] is None
    assert {"from": "F-1", "to": "D-1", "relation": "evidence"} in report["lineage"]["edges"]


def test_normal_finding_is_not_classified_as_negative(snapshot):
    snapshot["findings"][0]["failure_category"] = "none"
    assert len(build_report(snapshot)["negative_results"]) == 1


def test_missing_run_protocol_prevents_comparison(snapshot):
    snapshot["run"]["protocol"].pop("fingerprint")
    assert not build_report(snapshot)["comparability"]["groups"]


def test_track_report_keeps_own_sources_and_hides_other_tracks(snapshot):
    report = build_report(snapshot, track_id="candidate-1")
    assert report["summary"]["tracks"] == 1
    assert [i["id"] for i in report["ideas"]] == ["IDEA-1"]
    assert [s["id"] for s in report["sources"]] == ["SRC-1"]
    with pytest.raises(ValueError, match="没有这条研究轨迹"):
        build_report(snapshot, track_id="missing")


def test_broken_source_reference_is_explicit(snapshot):
    snapshot["sources"] = []
    report = build_report(snapshot)
    assert {"record_id": "IDEA-1", "reference_id": "SRC-1"} in report["lineage"]["missing_references"]
    assert "来源记录缺失" in render_markdown(report)


def test_html_escapes_untrusted_research_text(snapshot):
    snapshot["ideas"][0]["statement"] = '<script>alert("source")</script>'
    snapshot["sources"][0]["url"] = "javascript:alert(1)"
    html = render_html(build_report(snapshot))
    assert "<script>" not in html
    assert "javascript:" not in html
    assert "&lt;script&gt;" in html


def test_flow_payload_cannot_terminate_script_or_run_source_markup(snapshot):
    text = '</script><img src=x onerror="alert(1)"><script>'
    snapshot["ideas"][0]["statement"] = text
    snapshot["sources"][0]["url"] = "javascript:alert(1)"
    html = render_html(build_report(snapshot))
    payload = re.search(r'<script type="application/json" id="tf-flow-data">(.*?)</script>',
                        html, re.DOTALL).group(1)
    assert "<" not in payload
    decoded = json.loads(payload)
    idea = next(node for node in decoded["nodes"] if node["id"] == "IDEA-1")
    assert idea["detail"]["statement"] == text
    assert "javascript:" not in html
    assert '<img src=x onerror=' not in html


def test_html_flow_preserves_text_report_and_legacy_reports_still_render(snapshot):
    report = build_report(snapshot)
    html = render_html(report)
    assert 'aria-label="研究流程追踪"' in html
    assert 'aria-label="直接查看登记记录"' in html
    assert "完整文字报告与证据记录" in html
    assert "下一步消融。" in html
    report.pop("flow")
    legacy = render_html(report)
    assert "下一步消融。" in legacy
    assert "tf-flow-data" not in legacy


def test_existing_document_export_adapter_uses_same_facts(snapshot):
    report = build_report(snapshot)
    document = to_document(report)
    assert len(document.sections) == len(report["sections"])
    assert document.title == report["title"]
    assert any(section.title == "负结果与未完成工作" for section in document.sections)
    assert 'href="https://arxiv.org/abs/2402.05120"' in render_html(report)


def test_external_holdout_is_reported_separately_from_search_metrics(snapshot):
    snapshot["final_evaluation"] = {
        "status": "evaluated", "job_id": "JOB-1", "experiment_id": "EXP-1",
        "track_id_selected": "candidate-1", "selection_reason": "由验证结果先选定",
        "report_sha256": "fixed-report-hash", "feedback_to_agents": False,
        "surfaces": {"A": {"metrics": {"CVRMSE": 0.24}}}}
    report = build_report(snapshot)
    assert report["comparability"]["groups"][0]["best_per_metric"]["CVRMSE"]["value"] == 0.1
    assert report["final_evaluation"]["surfaces"]["A"]["metrics"]["CVRMSE"] == 0.24
    assert "最终留出评价（仅供外部查看）" in render_markdown(report)
    other = build_report(snapshot, track_id="main")
    assert other["final_evaluation"]["status"] == "not_selected"
    assert "0.24" not in render_markdown(other)


def test_empty_holdout_does_not_imply_acceptance(snapshot):
    snapshot["final_evaluation"] = {"status": "evaluated", "surfaces": {}, "feedback_to_agents": False}
    assert "未提供可用留出面结果，不能判断最终留出是否达标" in render_markdown(build_report(snapshot))
