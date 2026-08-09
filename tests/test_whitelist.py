"""DD-16 封闭白名单的机器校验（data-survey §F1/F4）。

负例断言：
- 派生量（load = current_percent × 96.72）加入 candidate_inputs → 拒绝；
- 白名单外变量（current_percent）进入实验特征 → 拒绝；
- 两层的 object.property 条目只在指定对象上放行。
"""

from __future__ import annotations

from thermoforge_research.runner import current_environment_lock
from thermoforge_research.tools import (
    tf_dataset_materialize,
    tf_experiment_plan,
    tf_goal_create,
    tf_hypothesis_create,
)
from thermoforge_research.whitelist import (
    check_candidate_inputs,
    check_view_within_whitelist,
)

from phase2_helpers import FEATURES, TARGET, view_definition
from phase34_helpers import make_ctx

# 模拟真实数据版本的变量元数据（chiller.v2：load 为 derived，§F1）
REAL_LIKE_VARIABLES = [
    {"variable_id": "chiller_01.power", "property_code": "power",
     "source_kind": "measured"},
    {"variable_id": "chiller_01.current_percent",
     "property_code": "current_percent", "source_kind": "measured"},
    {"variable_id": "chiller_01.load", "property_code": "load",
     "source_kind": "derived"},
    {"variable_id": "chw_A1.f", "property_code": "f",
     "source_kind": "measured"},
]


def test_derived_candidate_input_rejected():
    violations = check_candidate_inputs(
        ["f", "load"], target="power", variables=REAL_LIKE_VARIABLES)
    assert any("load" in v and "派生量" in v for v in violations)


def test_missing_candidate_input_rejected():
    violations = check_candidate_inputs(
        ["f", "nonexistent"], target="power", variables=REAL_LIKE_VARIABLES)
    assert any("nonexistent" in v for v in violations)


def test_two_layer_candidate_entry_allowed():
    violations = check_candidate_inputs(
        ["chw_A1.f"], target="power", variables=REAL_LIKE_VARIABLES)
    assert violations == []
    # object.property 形式但对象不存在
    violations = check_candidate_inputs(
        ["chw_A9.f"], target="power", variables=REAL_LIKE_VARIABLES)
    assert any("chw_A9.f" in v for v in violations)


def test_view_features_must_be_within_whitelist():
    violations = check_view_within_whitelist(
        features=["f", "current_percent"], view_target="power",
        view_objects=["PLANT"], candidate_inputs=["f"], goal_target="power")
    assert any("current_percent" in v for v in violations)
    # 白名单内则通过
    assert check_view_within_whitelist(
        features=["f"], view_target="power", view_objects=["PLANT"],
        candidate_inputs=["f"], goal_target="power") == []
    # 目标不一致拒绝
    assert check_view_within_whitelist(
        features=["f"], view_target="load", view_objects=["PLANT"],
        candidate_inputs=["f"], goal_target="power")


def test_two_layer_entry_scoped_to_object():
    # cooling_tower.supply_temp 只允许 cooling_tower 的 supply_temp
    assert check_view_within_whitelist(
        features=["supply_temp"], view_target="power",
        view_objects=["cooling_tower"],
        candidate_inputs=["cooling_tower.supply_temp"],
        goal_target="power") == []
    violations = check_view_within_whitelist(
        features=["supply_temp"], view_target="power",
        view_objects=["chiller"], candidate_inputs=["cooling_tower.supply_temp"],
        goal_target="power")
    assert any("supply_temp" in v for v in violations)


# ---------------------------------------------------------------- 工具层接线


def test_goal_create_with_dataset_ref_rejects_bad_candidate(tmp_path):
    ctx, ref = make_ctx(tmp_path)
    doc = {
        "name": "g", "object_model": "chiller.v1", "purpose": "optimization",
        "target": TARGET, "candidate_inputs": [*FEATURES, "no_such_prop"],
        "acceptance": {"cvrmse_max": 0.5},
    }
    env = tf_goal_create(ctx, doc, dataset_ref=ref)
    assert env["ok"] is False
    assert "no_such_prop" in env["summary"]["error"]
    ok = tf_goal_create(ctx, {**doc, "candidate_inputs": list(FEATURES)},
                        dataset_ref=ref)
    assert ok["ok"] is True


def test_experiment_plan_rejects_feature_outside_whitelist(tmp_path):
    ctx, ref = make_ctx(tmp_path)
    goal = tf_goal_create(ctx, {
        "name": "g", "object_model": "chiller.v1", "purpose": "optimization",
        "target": TARGET,
        "candidate_inputs": list(FEATURES[:3]),  # 故意排除 cw_supply_temp
        "acceptance": {"cvrmse_max": 0.5},
    }, dataset_ref=ref)
    assert goal["ok"]
    mat = tf_dataset_materialize(ctx, view_definition(ref))  # 含全部 4 特征
    hyp = tf_hypothesis_create(ctx, goal["id"], "h")
    exp_doc = {
        "goal_id": goal["id"], "hypothesis_id": hyp["id"],
        "dataset_view": mat["id"],
        "model": {"category": "data", "estimator": "ridge"},
        "target": TARGET,
        "validation": {"temporal_split": {"train": 0.7, "validate": 0.15,
                                          "test": 0.15}},
        "metrics": ["CVRMSE"],
        "runtime": {"environment_lock": current_environment_lock()[0],
                    "random_seed": 1},
    }
    env = tf_experiment_plan(ctx, exp_doc)
    assert env["ok"] is False
    assert "cw_supply_temp" in env["summary"]["error"]
    assert "DD-16" in env["summary"]["error"]
