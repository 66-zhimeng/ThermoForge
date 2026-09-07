"""Research survives model interruption without inventing an agent conclusion."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import threading

import pytest

from thermoforge_v2 import checkpoints
from thermoforge_v2.checkpoints import build_checkpoint, export_checkpoint
from thermoforge_v2.objectives import build_objective_contract
from thermoforge_v2.reports import build_report, render_html, render_markdown


@pytest.fixture()
def facts():
    protocol = {"fingerprint": "frozen-protocol", "goal_definition": {
        "acceptance": {"cvrmse_max": 0.05, "evaluated_on": "validate"}}}
    config = {"research_mode": "autonomous", "candidates": 1, "review_required": True}
    return {
        "run": {"run_id": "RUN-001", "status": "budget_exhausted", "protocol": protocol, "config": config,
                "objective_contract": build_objective_contract(config, protocol)},
        "tracks": [{"track_id": "main", "role": "main"}, {"track_id": "candidate-1", "role": "candidate"}],
        "sources": [{"id": "SRC-001", "track_id": "candidate-1", "title": "Abstract read", "read_scope": "abstract",
                     "kind": "paper", "url": "https://example.test/paper", "text": "Unneeded entire source body"}],
        "ideas": [{"id": "IDEA-001", "track_id": "candidate-1", "origin": "literature", "source_ids": ["SRC-001"],
                   "statement": "Regularization may help", "reason": "Source observation", "prediction": "Lower error"}],
        "proposals": [{"id": "PROPOSAL-001", "track_id": "candidate-1", "idea_id": "IDEA-001"},
                      {"id": "PROPOSAL-002", "track_id": "candidate-1", "idea_id": "IDEA-001", "parent_job_ids": ["JOB-001"]}],
        "jobs": [{"id": "JOB-001", "track_id": "candidate-1", "idea_id": "IDEA-001", "proposal_id": "PROPOSAL-001",
                  "status": "completed", "executed": True, "finished_at": 10,
                  "result": {"experiment_id": "EXP-0001", "protocol_fingerprint": "frozen-protocol",
                             "feedback_surface": "validate", "metrics": {"CVRMSE": .04, "R2": .9},
                             "model": {"spec": {"category": "data", "estimator": "ridge"}, "content_hash": "artifact-sha"}}},
                 {"id": "JOB-002", "track_id": "candidate-1", "idea_id": "IDEA-001", "proposal_id": "PROPOSAL-002",
                  "status": "completed", "executed": True, "finished_at": 20,
                  "result": {"experiment_id": "EXP-0002", "protocol_fingerprint": "frozen-protocol",
                             "feedback_surface": "validate", "metrics": {"CVRMSE": .06, "R2": .8}}},
                 {"id": "JOB-003", "track_id": "candidate-1", "idea_id": "IDEA-001", "status": "failed",
                  "finished_at": 30, "result": {"failure_category": "implementation", "error": "Training failed"}}],
        "findings": [], "reports": [], "stops": [], "events": [{"seq": 5, "payload": {"external": "must not forward"}}],
    }


class SnapshotStore:
    def __init__(self, root, facts):
        self.root, self.facts = root, facts

    def snapshot(self, run_id):
        assert run_id == "RUN-001"
        return deepcopy(self.facts)


def test_budget_exit_preserves_results_and_missing_agent_explanations(facts):
    before = deepcopy(facts)
    brief = build_checkpoint(facts)
    assert facts == before
    assert brief["status"] == "budget_exhausted"
    assert brief["generated_by"] == "system" and brief["is_agent_report"] is False
    assert brief["counts"]["settled_jobs"] == 3 and brief["counts"]["agent_reports"] == 0
    track = brief["tracks"][1]
    assert track["hypotheses"][0]["reason"] == "Source observation"
    assert track["sources"][0]["read_scope"] == "abstract"
    assert track["settled_jobs"][0]["artifact_ids"]["experiment_id"] == "EXP-0001"
    assert track["missing"]["finding_job_ids"] == ["JOB-001", "JOB-002", "JOB-003"]
    assert track["missing"]["agent_report"] and track["missing"]["stop_decision"]
    assert brief["objective"]["best_observed"]["job_id"] == "JOB-001"
    assert brief["objective"]["best_eligible"] is None
    assert {n["kind"] for n in brief["negative_results"]} == {"failed", "metric_regression"}
    assert facts["reports"] == [] and facts["stops"] == []


def test_registered_reviews_and_reports_roll_forward_without_hiding_old_results(facts):
    initial = build_checkpoint(facts)
    facts["findings"] = [{"id": "FINDING-001", "track_id": "candidate-1", "job_ids": ["JOB-001"],
        "protocol_fingerprint": "frozen-protocol", "statement": "Observed improvement", "interpretation": "Limited evidence",
        "limitations": "No replication"}]
    facts["reports"] = [{"id": "REPORT-001", "track_id": "candidate-1", "kind": "track", "summary": "First result",
                          "job_ids": ["JOB-001"], "created_at": 15}]
    partial = build_checkpoint(facts)
    assert partial["checkpoint_id"] != initial["checkpoint_id"]
    assert partial["objective"]["best_eligible"]["job_id"] == "JOB-001"
    candidate = partial["tracks"][1]
    assert candidate["latest_agent_report"]["id"] == "REPORT-001"
    assert candidate["missing"]["report_job_ids"] == ["JOB-002", "JOB-003"]
    assert candidate["missing"]["report_predates_latest_result"]
    assert candidate["missing"]["finding_job_ids"] == ["JOB-002", "JOB-003"]
    assert len(partial["negative_results"]) == 2


def test_main_without_candidates_requires_own_report_and_stop(facts):
    facts["run"]["config"]["candidates"] = 0
    facts["tracks"] = [{"track_id": "main", "role": "main"}]
    for group in ("sources", "ideas", "proposals", "jobs"):
        for row in facts[group]:
            row["track_id"] = "main"
    facts["reports"] = [{"id": "REPORT-001", "track_id": "main", "kind": "track", "created_at": 40,
                          "job_ids": ["JOB-001", "JOB-002", "JOB-003"]}]
    candidate = build_checkpoint(facts)["tracks"][0]
    assert candidate["job_scope"] == "track"
    assert not candidate["missing"]["agent_report"]
    assert candidate["missing"]["stop_decision"]
    assert not candidate["missing"]["final_team_report"]


def test_checkpoint_ignores_holdout_and_external_events_without_opening_artifacts(facts, tmp_path):
    facts["final_evaluation"] = {"surfaces": {"A": {"CVRMSE": "secret-holdout"}}}
    facts["jobs"][0]["result"].update(artifact=str(tmp_path / "private-report.json"),
        surfaces={"A": {"CVRMSE": "secret-holdout"}}, final_evaluation={"secret": "secret-holdout"})
    facts["jobs"][1]["result"] = {"feedback_surface": "A", "metrics": {"CVRMSE": 987.654321}}
    facts["events"][0]["payload"] = {"finalization": "secret-holdout"}
    before = deepcopy(facts)
    metadata = export_checkpoint(SnapshotStore(tmp_path, facts), "RUN-001", "budget_exit")
    assert facts == before
    for artifact in metadata["artifacts"]:
        content = Path(artifact["path"]).read_text(encoding="utf-8")
        assert "secret-holdout" not in content
        assert "987.654321" not in content
        assert "private-report.json" not in content
    brief = json.loads(Path(metadata["path"]).read_text(encoding="utf-8"))
    assert brief["event_cursor"] == 5
    assert brief["evidence_scope"] == "train_validate_only"
    assert brief["comparison_surface"] == "validate"


def test_exports_readable_artifacts_and_hashes_without_creating_agent_reports(facts, tmp_path):
    result = export_checkpoint(SnapshotStore(tmp_path, facts), "RUN-001", "job.settled")
    output = Path(result["path"]).parent
    assert (output / "team-flow.html").is_file() and (output / "team-flow.json").is_file()
    assert (output / "candidate-1.html").is_file() and (output / "jobs" / "JOB-003.json").is_file()
    report = json.loads((output / "team.json").read_text(encoding="utf-8"))
    assert report["reports"] == []
    assert report["checkpoint_id"] == result["checkpoint_id"]
    assert "系统自动汇总" in (output / "team.html").read_text(encoding="utf-8")
    for artifact in result["artifacts"]:
        assert Path(artifact["path"]).is_relative_to(output / "versions" / result["version_id"])
        assert hashlib.sha256(Path(artifact["path"]).read_bytes()).hexdigest() == artifact["sha256"]
    assert not list(output.rglob("*.tmp"))


def test_successful_concurrent_exports_publish_newest_locked_snapshot(facts, tmp_path):
    class SequencedStore(SnapshotStore):
        calls = 0
        active = False

        def snapshot(self, run_id):
            assert not self.active, "Snapshot acquisition must be serialized with the export"
            self.active = True
            self.calls += 1
            snapshot = deepcopy(self.facts)
            snapshot["events"][0]["seq"] = self.calls
            self.active = False
            return snapshot

    store = SequencedStore(tmp_path, facts)
    gate = threading.Barrier(4)

    def export(index):
        gate.wait(timeout=10)
        return export_checkpoint(store, "RUN-001", f"concurrent-{index}")

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(export, range(4)))
    assert sorted(r["event_cursor"] for r in results) == [1, 2, 3, 4]
    latest = json.loads(Path(results[0]["path"]).read_text(encoding="utf-8"))
    assert latest["event_cursor"] == 4
    for result in results:
        for artifact in result["artifacts"]:
            assert hashlib.sha256(Path(artifact["path"]).read_bytes()).hexdigest() == artifact["sha256"]


def test_failed_replace_keeps_previous_file_and_manifest_and_cleans_temporary(facts, tmp_path, monkeypatch):
    store = SnapshotStore(tmp_path, facts)
    first = export_checkpoint(store, "RUN-001", "first")
    output = Path(first["path"]).parent
    before_html = (output / "team.html").read_bytes()
    before_manifest = Path(first["path"]).read_bytes()
    facts["events"][0]["seq"] = 6
    original = checkpoints.os.replace

    def fail(source, destination):
        if Path(destination).name == "team.html":
            raise OSError("simulated disk write failure")
        return original(source, destination)

    monkeypatch.setattr(checkpoints.os, "replace", fail)
    with pytest.raises(OSError, match="simulated"):
        export_checkpoint(store, "RUN-001", "second")
    assert (output / "team.html").read_bytes() == before_html
    assert Path(first["path"]).read_bytes() == before_manifest
    for artifact in first["artifacts"]:
        assert hashlib.sha256(Path(artifact["path"]).read_bytes()).hexdigest() == artifact["sha256"]
    assert not list(output.rglob("*.tmp"))


@pytest.mark.parametrize("phase", ["version", "alias"])
def test_mid_export_failure_cannot_invalidate_previously_published_artifacts(facts, tmp_path, monkeypatch, phase):
    store = SnapshotStore(tmp_path, facts)
    first = export_checkpoint(store, "RUN-001", "first")
    latest_path = Path(first["path"])
    before_manifest = latest_path.read_bytes()
    facts["jobs"][0]["result"]["metrics"]["CVRMSE"] = .025
    facts["events"][0]["seq"] = 6
    original, written = checkpoints._atomic_write, []

    def fail_late(path, content):
        in_version = "versions" in path.parts
        if path.name == "candidate-1.html" and in_version == (phase == "version"):
            assert len(written) >= 5
            raise OSError("late multi-file failure")
        original(path, content)
        written.append(path)

    monkeypatch.setattr(checkpoints, "_atomic_write", fail_late)
    with pytest.raises(OSError, match="late multi-file"):
        export_checkpoint(store, "RUN-001", "second")
    assert latest_path.read_bytes() == before_manifest
    for artifact in first["artifacts"]:
        assert hashlib.sha256(Path(artifact["path"]).read_bytes()).hexdigest() == artifact["sha256"]


def test_readers_can_use_old_manifest_while_new_generation_is_written(facts, tmp_path, monkeypatch):
    store = SnapshotStore(tmp_path, facts)
    first = export_checkpoint(store, "RUN-001", "first")
    latest_path = Path(first["path"])
    initial = json.loads(latest_path.read_text(encoding="utf-8"))
    paused, resume = threading.Event(), threading.Event()
    original = checkpoints._atomic_write

    def pause_mid_write(path, content):
        original(path, content)
        if path.name == "team.json" and "versions" in path.parts:
            paused.set()
            assert resume.wait(timeout=10)

    monkeypatch.setattr(checkpoints, "_atomic_write", pause_mid_write)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(export_checkpoint, store, "RUN-001", "same_facts_new_export")
        try:
            assert paused.wait(timeout=10)
            observed = json.loads(latest_path.read_text(encoding="utf-8"))
            assert observed == initial
            for artifact in observed["artifacts"]:
                assert hashlib.sha256(Path(artifact["path"]).read_bytes()).hexdigest() == artifact["sha256"]
        finally:
            resume.set()
        second = future.result(timeout=10)
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    assert latest["version_id"] == second["version_id"] != first["version_id"]
    assert latest["checkpoint_id"] == first["checkpoint_id"]
    for manifest in (initial, latest):
        for artifact in manifest["artifacts"]:
            assert hashlib.sha256(Path(artifact["path"]).read_bytes()).hexdigest() == artifact["sha256"]


def test_deep_export_uses_windows_long_path_io_but_keeps_normal_manifest_paths(facts, tmp_path):
    deep_root = tmp_path.joinpath(*(["research-directory-with-a-long-name"] * 8))
    assert len(os.path.abspath(deep_root)) > 260
    result = export_checkpoint(SnapshotStore(deep_root, facts), "RUN-001", "deep_path")

    def read_bytes(path):
        # Read independently through the documented Win32 prefix so a normal
        # MAX_PATH-limited read cannot obscure whether export actually succeeded.
        native = "\\\\?\\" + path if os.name == "nt" else path
        return Path(native).read_bytes()

    assert not result["path"].startswith("\\\\?\\")
    manifest = json.loads(read_bytes(result["path"]).decode("utf-8"))
    for artifact in manifest["artifacts"]:
        assert not artifact["path"].startswith("\\\\?\\")
        assert hashlib.sha256(read_bytes(artifact["path"])).hexdigest() == artifact["sha256"]
    output = str(Path(result["path"]).parent)
    assert b"<!doctype html>" in read_bytes(str(Path(output) / "team.html"))
    repeated = export_checkpoint(SnapshotStore(deep_root, facts), "RUN-001", "deep_path_again")
    assert repeated["version_id"] != result["version_id"]
    for artifact in manifest["artifacts"]:
        assert hashlib.sha256(read_bytes(artifact["path"])).hexdigest() == artifact["sha256"]


@pytest.mark.parametrize("field,value", [("run_id", "../outside"), ("track_id", "../outside"), ("job_id", "../outside")])
def test_untrusted_ids_cannot_escape_checkpoint_directory(facts, tmp_path, field, value):
    if field == "track_id":
        facts["tracks"][1]["track_id"] = value
    if field == "job_id":
        facts["jobs"][0]["id"] = value
    with pytest.raises(ValueError, match="路径组件"):
        export_checkpoint(SnapshotStore(tmp_path, facts), value if field == "run_id" else "RUN-001", "test")
    assert not (tmp_path / "outside").exists()


def test_checkpoint_report_keeps_system_facts_distinct_and_html_safe(facts):
    facts["ideas"][0]["statement"] = "</script><script>alert(1)</script>"
    report = build_report(facts)
    text = render_markdown(report)
    assert "这不是智能体撰写的研究解释" in text
    assert "修订实验的指标退化" in text
    html = render_html(report)
    assert "<script>alert(1)</script>" not in html
    assert report["flow"]["checkpoint"]["is_agent_report"] is False
    assert not any(n["kind"] == "report" for n in report["flow"]["nodes"])


def test_unknown_historical_objective_and_nonfinite_values_do_not_become_success(facts):
    facts["run"].pop("objective_contract")
    facts["jobs"][0]["result"]["metrics"]["CVRMSE"] = float("nan")
    brief = build_checkpoint(facts)
    assert not brief["objective"]["contract_available"]
    assert brief["objective"]["acceptance_status"] == "unknown"
    json.dumps(brief, allow_nan=False)


def test_diagnostics_preserve_train_validate_gap_and_filter_holdout_everywhere(facts, tmp_path):
    facts["jobs"][0]["result"]["diagnostics"] = {
        "train": {"metrics": {"CVRMSE": .02}}, "validate": {"metrics": {"CVRMSE": .04}},
        "generalization_gap": {"validate_minus_train": {"CVRMSE": .02}},
        "A": {"metrics": {"CVRMSE": "secret-diagnostic"}}, "private_path": "secret-diagnostic"}
    report = build_report(facts)
    job = next(n for n in report["flow"]["nodes"] if n["id"] == "JOB-001")
    assert job["detail"]["result"]["diagnostics"]["train"]["metrics"]["CVRMSE"] == .02
    assert "secret-diagnostic" not in json.dumps(job)
    metadata = export_checkpoint(SnapshotStore(tmp_path, facts), "RUN-001", "diagnostics")
    for artifact in metadata["artifacts"]:
        assert "secret-diagnostic" not in Path(artifact["path"]).read_text(encoding="utf-8")


def test_shared_projection_preserves_only_trusted_constraint_evidence(facts):
    facts["run"]["protocol"]["goal_definition"]["acceptance"]["physics_violation_rate_max"] = .01
    facts["run"]["objective_contract"] = build_objective_contract(facts["run"]["config"], facts["run"]["protocol"])
    facts["jobs"][0]["result"]["constraint_evidence"] = {
        "physics_violation_rate": {"evaluated_on": "validate", "verified": True, "value": 0},
        "inference_latency_ms": {"evaluated_on": "A", "verified": True, "value": 4},
        "raw": "private-constraint"}
    before = build_checkpoint(facts)
    first = before["objective"]["per_job"][0]
    assert first["constraint_status"] == "pass"
    feedback = before["tracks"][1]["settled_jobs"][0]["result"]
    assert feedback["constraint_evidence"] == {
        "physics_violation_rate": {"evaluated_on": "validate", "verified": True, "value": 0}}
    assert "private-constraint" not in json.dumps(before)
