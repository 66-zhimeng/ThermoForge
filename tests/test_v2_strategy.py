"""V2 策略建议只消费同协议验证证据，不让留出或失败伪装成优胜。"""

from __future__ import annotations

from copy import deepcopy

import pytest

from thermoforge_v2.strategy import compare_jobs, strategy_advice


def job(identifier, score, *, status="completed", surface="validate", fingerprint="protocol"):
    return {"id": identifier, "track_id": f"track-{identifier}", "idea_id": f"idea-{identifier}",
            "status": status, "result": {"protocol_fingerprint": fingerprint,
                "feedback_surface": surface, "metrics": {"CVRMSE": score}}}


def test_comparison_uses_validation_order_and_top_k_with_stable_ties():
    jobs = [job("c", .2), job("b", .1), job("a", .1), job("d", .4)]
    result = compare_jobs(jobs, "protocol", top_k=2)
    assert [r["job_id"] for r in result["ranked"]] == ["a", "b", "c", "d"]
    assert [r["job_id"] for r in result["top_k"]] == ["a", "b"]
    assert result["metric"] == "CVRMSE" and result["surface"] == "validate"
    assert result["lower_is_better"] is True
    assert compare_jobs(jobs, "protocol", top_k=1)["top_k"][0]["idea_id"] == "idea-a"
    assert len(compare_jobs(jobs, "protocol", top_k=20)["top_k"]) == 4


@pytest.mark.parametrize("bad", [
    job("heldout-A", .00001, surface="A"),
    job("heldout-C", .00001, surface="C"),
    job("changed-protocol", .00001, fingerprint="different"),
    job("missing-protocol", .00001, fingerprint=None),
    job("failed", .00001, status="failed"),
    job("running", .00001, status="running"),
    job("nan", float("nan")),
    job("infinity", float("inf")),
    job("boolean", False),
    job("missing-value", None),
    job("string", "0.001"),
])
def test_incomparable_or_invalid_result_cannot_displace_validated_model(bad):
    result = compare_jobs([bad, job("valid", .3)], "protocol", top_k=1)
    assert [r["job_id"] for r in result["top_k"]] == ["valid"]
    assert result["excluded"] == [bad["id"]]


def test_empty_or_only_excluded_evidence_has_no_winner_or_parent():
    assert compare_jobs([], "protocol")["top_k"] == []
    advice = strategy_advice([job("failed", 0, status="failed")],
                              {"strategy": "adaptive", "top_k": 2, "stagnation_rounds": 2}, "protocol")
    assert advice["parents"] == []
    assert advice["stagnant"] is False
    assert advice["requires_scientific_decision"] is True


def test_improving_recent_validation_keeps_refinement_and_traces_parents():
    jobs = [job("old", .3), job("middle", .2), job("new", .1)]
    advice = strategy_advice(jobs, {"strategy": "adaptive", "top_k": 2,
                                    "stagnation_rounds": 2}, "protocol")
    assert not advice["stagnant"]
    assert advice["suggested_action"] == "refine"
    assert advice["parents"] == ["idea-new", "idea-middle"]


@pytest.mark.parametrize("jobs,expected", [
    ([job("best", .1), job("failed", None, status="failed"), job("worse", .3)], "explore"),
    ([job("best", .1), job("next", .2), job("failed", None, status="failed"), job("worse", .3)], "recombine"),
])
def test_adaptive_stagnation_changes_advice_without_claiming_success(jobs, expected):
    config = {"strategy": "adaptive", "top_k": 1, "stagnation_rounds": 2}
    before_jobs, before_config = deepcopy(jobs), deepcopy(config)
    advice = strategy_advice(jobs, config, "protocol")
    assert advice["stagnant"]
    assert advice["suggested_action"] == expected
    assert advice["parents"] == ["idea-best"]
    assert advice["requires_scientific_decision"] is True
    assert jobs == before_jobs and config == before_config


def test_independent_and_top_k_do_not_silently_become_adaptive():
    jobs = [job("best", .1), job("second", .3), job("third", .4)]
    independent = strategy_advice(jobs, {"strategy": "independent", "stagnation_rounds": 2}, "protocol")
    top_k = strategy_advice(jobs, {"strategy": "top_k", "stagnation_rounds": 2}, "protocol")
    assert independent["stagnant"] and independent["suggested_action"] == "independent"
    assert top_k["stagnant"] and top_k["suggested_action"] == "refine"


def test_holdout_improvement_never_appears_in_adaptive_parent_selection():
    jobs = [job("validation", .2), job("heldout", .001, surface="A"),
            job("validation-worse", .3)]
    advice = strategy_advice(jobs, {"strategy": "adaptive", "stagnation_rounds": 2}, "protocol")
    assert "idea-heldout" not in advice["parents"]
    assert advice["stagnant"]


def test_multiple_experiments_for_one_idea_do_not_fill_distinct_route_slots():
    first = job("first", .1)
    refined = job("refined", .09) | {"idea_id": first["idea_id"]}
    second = job("second", .2)
    third = job("third", .3)
    jobs = [first, refined, second, third]
    # 实验排行榜和最终选型语义保持不变，允许同一路线多个实验入榜。
    experiments = compare_jobs(jobs, "protocol", top_k=2)
    assert [row["job_id"] for row in experiments["top_k"]] == ["refined", "first"]
    advice = strategy_advice(jobs, {"strategy": "top_k", "top_k": 2,
                                    "stagnation_rounds": 2}, "protocol")
    assert advice["parents"] == [first["idea_id"], second["idea_id"]]
    assert [row["job_id"] for row in advice["selected_routes"]] == ["refined", "second"]


def test_last_finishing_improvement_is_recent_even_when_it_was_reserved_first():
    jobs = [job("slow-best", .1) | {"created_at": 1, "finished_at": 30},
            job("quick-old", .4) | {"created_at": 2, "finished_at": 10},
            job("middle", .3) | {"created_at": 3, "finished_at": 20}]
    advice = strategy_advice(jobs, {"strategy": "adaptive", "top_k": 2,
                                    "stagnation_rounds": 1}, "protocol")
    assert advice["recent_feedback_job_ids"] == ["slow-best"]
    assert advice["stagnant"] is False
    assert advice["suggested_action"] == "refine"


def test_last_finishing_failure_counts_as_no_progress_after_earlier_good_result():
    jobs = [job("slow-failure", None, status="failed") | {"created_at": 1, "finished_at": 30},
            job("early-good", .1) | {"created_at": 2, "finished_at": 10},
            job("later-worse", .3) | {"created_at": 3, "finished_at": 20}]
    advice = strategy_advice(jobs, {"strategy": "adaptive", "stagnation_rounds": 1}, "protocol")
    assert advice["recent_feedback_job_ids"] == ["slow-failure"]
    assert advice["stagnant"] is True
    assert advice["suggested_action"] == "explore"
    assert "idea-slow-failure" not in advice["parents"]
