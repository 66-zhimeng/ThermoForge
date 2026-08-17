"""Phase 3 工具层（architecture §6、implementation-notes §11）。

- 信封字段与稳定 ID（有副作用工具）；
- sample 200 行硬上限；profile 分位数固定点位；
- goal → hypothesis（basis 强制）→ view → experiment → run/get/compare/publish；
- harness/tools.json 与 TOOL_REGISTRY 一致性。
"""

from __future__ import annotations

import json

import pytest

from thermoforge_models.hybrid import ResidualHybrid
from thermoforge_models.identification import EffectivenessNTU
from thermoforge_models.physics import ChillerPhysicsV2
from thermoforge_research.envelope import SAMPLE_MAX_ROWS
from thermoforge_research.runner import current_environment_lock
from thermoforge_runtime.artifact import load_model_artifact
from thermoforge_research.tools import (
    TOOL_REGISTRY,
    tf_dataset_compare,
    tf_dataset_get,
    tf_dataset_list,
    tf_dataset_materialize,
    tf_dataset_profile,
    tf_dataset_query,
    tf_dataset_sample,
    tf_dataset_schema,
    tf_experiment_get,
    tf_experiment_plan,
    tf_experiment_run,
    tf_goal_create,
    tf_hypothesis_create,
    tf_model_compare,
    tf_model_publish,
    tf_research_status,
)

from phase2_helpers import (
    FEATURES,
    PHYSICS_INPUTS,
    RATED_CAPACITY_KW,
    RATED_POWER_KW,
    TARGET,
    view_definition,
)
from phase34_helpers import make_ctx


@pytest.fixture()
def ctx_ref(tmp_path):
    return make_ctx(tmp_path, n_steps=600)


def _goal_doc(**overrides):
    doc = {
        "name": "冷水机输入功率模型",
        "object_model": "chiller.v1",
        "purpose": "optimization",
        "target": TARGET,
        "candidate_inputs": list(FEATURES),
        "acceptance": {"cvrmse_max": 0.5},
    }
    doc.update(overrides)
    return doc


def _experiment_doc(goal_id, hypothesis_id, view_id):
    return {
        "goal_id": goal_id,
        "hypothesis_id": hypothesis_id,
        "dataset_view": view_id,
        "model": {"category": "data", "estimator": "ridge",
                  "hyperparameters": {"alpha": 0.5}},
        "target": TARGET,
        "validation": {"temporal_split": {"train": 0.70, "validate": 0.15,
                                          "test": 0.15}},
        "metrics": ["RMSE", "MAE", "MAPE", "CVRMSE", "NMBE"],
        "runtime": {"environment_lock": current_environment_lock()[0],
                    "random_seed": 20260808},
    }


def _research_setup(ctx, ref):
    goal = tf_goal_create(ctx, _goal_doc())
    mat = tf_dataset_materialize(ctx, view_definition(ref))
    hyp = tf_hypothesis_create(ctx, goal["id"], "线性基线足够好")
    plan = tf_experiment_plan(
        ctx, _experiment_doc(goal["id"], hyp["id"], mat["id"]))
    return goal, mat, hyp, plan


# ---------------------------------------------------------------- 数据工具


def test_dataset_list_get_schema(ctx_ref):
    ctx, ref = ctx_ref
    listing = tf_dataset_list(ctx)
    assert listing["ok"] and listing["summary"]["count"] == 1
    assert listing["summary"]["datasets"][0]["revisions"][0]["ref"] == ref

    got = tf_dataset_get(ctx, ref)
    assert got["ok"] and got["id"] == ref
    assert got["summary"]["manifest"]["dataset_id"] == "TEST02_RESEARCH"

    schema = tf_dataset_schema(ctx, ref)
    props = {v["property_code"] for v in schema["summary"]["variables"]}
    assert set(FEATURES) | {TARGET} <= props

    missing = tf_dataset_get(ctx, "TEST02_RESEARCH@rev_9999")
    assert missing["ok"] is False
    assert missing["diagnostics"][0]["code"] == "TFV-703"


def test_dataset_profile_fixed_quantile_points(ctx_ref):
    ctx, ref = ctx_ref
    env = tf_dataset_profile(ctx, ref)
    assert env["ok"]
    assert env["summary"]["quantile_points"] == ["min", "p01", "p25", "p50",
                                                 "p75", "p99", "max"]
    var = next(v for v in env["summary"]["variables"]
               if v["variable_id"] == "CH-01.input_power")
    dist = var["distribution"]
    assert list(dist) == ["min", "p01", "p25", "p50", "p75", "p99", "max"]
    assert dist["min"] <= dist["p01"] <= dist["p50"] <= dist["p99"] <= dist["max"]


def test_dataset_query_aggregates_only(ctx_ref):
    ctx, ref = ctx_ref
    env = tf_dataset_query(ctx, ref, variable_ids=["CH-01.input_power"])
    assert env["ok"]
    stats = env["summary"]["stats"]["CH-01.input_power"]
    assert stats["count"] == 600 and stats["missing"] == 0
    assert stats["min"] < stats["mean"] < stats["max"]
    bad = tf_dataset_query(ctx, ref, variable_ids=["CH-01.nope"])
    assert bad["ok"] is False and bad["diagnostics"][0]["code"] == "TFV-802"


def test_dataset_sample_hard_cap_200(ctx_ref):
    ctx, ref = ctx_ref
    env = tf_dataset_sample(ctx, ref, n=500,
                            variable_ids=["CH-01.input_power"])
    assert env["ok"] and env["truncated"] is True
    assert env["summary"]["returned"] == SAMPLE_MAX_ROWS
    assert env["summary"]["total_matched"] == 600
    small = tf_dataset_sample(ctx, ref, n=10,
                              variable_ids=["CH-01.input_power"])
    assert small["summary"]["returned"] == 10
    assert small["truncated"] is False


def test_dataset_materialize_stable_view_id(ctx_ref):
    ctx, ref = ctx_ref
    env = tf_dataset_materialize(ctx, view_definition(ref))
    assert env["ok"] and env["id"] == "VIEW-0001"
    assert env["summary"]["view_hash"]
    again = tf_dataset_materialize(ctx, view_definition(ref))
    assert again["summary"]["cache_reused"] is True  # 同定义命中缓存
    assert again["id"] == "VIEW-0002"  # 登记是新实体，物化缓存复用


def test_dataset_compare(ctx_ref):
    ctx, ref = ctx_ref
    same = tf_dataset_compare(ctx, ref, ref)
    assert same["summary"]["content_identical"] is True
    assert same["summary"]["variables_added"] == []


# ---------------------------------------------------------------- 研究工具


def test_goal_create_returns_stable_id_and_validates(ctx_ref):
    ctx, _ = ctx_ref
    env = tf_goal_create(ctx, _goal_doc())
    assert env["ok"] and env["id"] == "RG-0001"
    assert env["summary"]["target"] == TARGET
    bad = tf_goal_create(ctx, {"name": "缺字段"})
    assert bad["ok"] is False


def test_hypothesis_basis_enforced(ctx_ref):
    ctx, _ = ctx_ref
    goal = tf_goal_create(ctx, _goal_doc())
    first = tf_hypothesis_create(ctx, goal["id"], "首个假设可无 basis")
    assert first["ok"] and first["id"] == "H-0001"
    second = tf_hypothesis_create(ctx, goal["id"], "无证据的后续假设")
    assert second["ok"] is False  # research-loop §3：必须引用已有证据
    third = tf_hypothesis_create(ctx, goal["id"], "有证据", basis=["H-0001"])
    # basis 只接受 F-/EXP- 证据；H- 引用应被拒绝
    assert third["ok"] is False
    finding = ctx.ledger.create_finding("证据", actor="test")
    fourth = tf_hypothesis_create(ctx, goal["id"], "有证据",
                                  basis=[finding["id"]])
    assert fourth["ok"]


def test_experiment_plan_run_get_compare(ctx_ref):
    ctx, ref = ctx_ref
    goal, mat, hyp, plan = _research_setup(ctx, ref)
    assert plan["ok"] and plan["id"] == "EXP-0001"

    ran = tf_experiment_run(ctx, plan["id"])
    assert ran["ok"], ran["summary"].get("error_code")
    assert ran["status"] == "COMPLETED"
    surfaces = ran["summary"]["surfaces"]
    assert surfaces["A"]["metrics"]["CVRMSE"] is not None
    kinds = {a["kind"] for a in ran["artifacts"]}
    assert {"experiment_report", "metrics", "model_dir"} <= kinds

    got = tf_experiment_get(ctx, plan["id"])
    assert got["ok"] and got["summary"]["status"] == "completed"
    assert got["summary"]["surfaces"]["A"]["n_samples"] > 0

    cmp_env = tf_model_compare(ctx, [plan["id"], "EXP-9999"])
    assert cmp_env["ok"]
    assert cmp_env["summary"]["best"] == plan["id"]
    assert cmp_env["summary"]["skipped"] == ["EXP-9999"]


def test_experiment_plan_fills_environment_lock_but_not_seed(ctx_ref):
    """环境指纹缺省时由工具填真值；随机种子缺省时仍须报错。

    指纹是机器算的，调用方猜不到——以前必填导致 Agent 编一个假串，
    每次 run 都撞 TFX-901。种子性质相反：它是研究决策，不能替调用方决定。
    """
    ctx, ref = ctx_ref
    goal = tf_goal_create(ctx, _goal_doc())
    mat = tf_dataset_materialize(ctx, view_definition(ref))
    hyp = tf_hypothesis_create(ctx, goal["id"], "线性基线足够好")

    doc = _experiment_doc(goal["id"], hyp["id"], mat["id"])
    doc["runtime"] = {"environment_lock": "", "random_seed": 7}
    plan = tf_experiment_plan(ctx, doc)
    assert plan["ok"]
    assert plan["summary"]["environment_lock"] == current_environment_lock()[0]
    ran = tf_experiment_run(ctx, plan["id"])
    assert ran["ok"], ran["summary"].get("error_code")

    doc2 = _experiment_doc(goal["id"], hyp["id"], mat["id"])
    doc2["runtime"] = {"environment_lock": ""}
    assert tf_experiment_plan(ctx, doc2)["ok"] is False


def test_research_status_tracks_budget(ctx_ref):
    ctx, ref = ctx_ref
    goal, mat, hyp, plan = _research_setup(ctx, ref)
    status = tf_research_status(ctx, goal["id"])
    assert status["ok"]
    assert status["summary"]["progress"]["experiments"]["total"] == 1
    budget = status["summary"]["budget"]
    assert budget["experiments_used"] == 1
    assert budget["experiments_max"] is None


def test_model_publish_end_to_end(ctx_ref):
    """工具链闭环：实验 → 模型包 → 注册 → 门禁 → production。"""
    ctx, ref = ctx_ref
    goal, mat, hyp, plan = _research_setup(ctx, ref)
    ran = tf_experiment_run(ctx, plan["id"])
    assert ran["ok"]

    env = tf_model_publish(ctx, plan["id"], model_id="chiller-power",
                           version="1.0.0")
    assert env["ok"], env["summary"].get("error")
    assert env["id"] == "chiller-power@1.0.0"
    assert env["status"] == "PRODUCTION"
    gate_names = [g["name"] for g in env["summary"]["gates"]]
    assert "smoke" in gate_names  # 冷加载冒烟在新解释器子进程执行
    assert env["summary"]["golden_samples"] > 0
    # Ledger 谱系：模型实体 + 发布决策
    models = [m for m in ctx.ledger.list_goals() if m["id"] == goal["id"]]
    assert models  # goal 仍在
    progress = ctx.ledger.goal_progress(goal["id"])
    assert progress["models"]["total"] == 1


def test_model_publish_rejected_when_acceptance_unmet(ctx_ref):
    ctx, ref = ctx_ref
    goal = tf_goal_create(ctx, _goal_doc(
        acceptance={"cvrmse_max": 1e-12}))  # 不可能达标
    mat = tf_dataset_materialize(ctx, view_definition(ref))
    hyp = tf_hypothesis_create(ctx, goal["id"], "严苛验收")
    plan = tf_experiment_plan(
        ctx, _experiment_doc(goal["id"], hyp["id"], mat["id"]))
    ran = tf_experiment_run(ctx, plan["id"])
    assert ran["ok"]
    env = tf_model_publish(ctx, plan["id"], model_id="chiller-power",
                           version="1.0.0", run_smoke=False)
    assert env["ok"] is False
    assert env["diagnostics"][0]["code"] == "TFM-1003"
    assert env["summary"]["gates"][-1]["ok"] is False


ROLLING_CV = {"enabled": True, "mode": "expanding",
              "initial_train_fraction": 0.4, "horizon_seconds": 6 * 3600,
              "step_seconds": 6 * 3600, "max_folds": 5}


def test_acceptance_can_be_judged_on_rolling_cv(ctx_ref):
    """门槛判在滚动交叉验证聚合上，而不是尾部留出面。

    面 A 是「训练截止后隔一段再考」，衡量漂移；rolling_cv 每折用最近数据
    重训，衡量定期重训下的精度。同一模型两者能差数倍 —— 门槛从哪个口径的
    基线推出来，就必须判在哪个口径上，否则是拿甲的尺子量乙。
    """
    ctx, ref = ctx_ref
    goal = tf_goal_create(ctx, _goal_doc(
        acceptance={"cvrmse_max": 0.5, "evaluated_on": "rolling_cv"}))
    mat = tf_dataset_materialize(ctx, view_definition(ref))
    hyp = tf_hypothesis_create(ctx, goal["id"], "滚动口径足够好")
    doc = _experiment_doc(goal["id"], hyp["id"], mat["id"])
    doc["validation"]["rolling_cv"] = ROLLING_CV
    plan = tf_experiment_plan(ctx, doc)
    ran = tf_experiment_run(ctx, plan["id"])
    assert ran["ok"], ran["summary"].get("error_code")

    # 折间均值进了 metrics，模型包因此自带这个数
    report = json.loads(
        (ctx.research_root / "experiments" / plan["id"] / "metrics.json")
        .read_text(encoding="utf-8"))
    assert report["rolling_cv"]["n_folds"] >= 2
    assert report["rolling_cv"]["metrics"]["CVRMSE"] > 0

    # 比较工具按 Goal 声明的口径取数，与发布门禁同调
    cmp_env = tf_model_compare(ctx, [plan["id"]])
    assert cmp_env["summary"]["experiments"][0]["primary_surface"] == "rolling_cv"

    env = tf_model_publish(ctx, plan["id"], model_id="chiller-power",
                           version="1.0.0", run_smoke=False)
    assert env["ok"], env["summary"].get("error")
    detail = next(g for g in env["summary"]["gates"]
                  if g["name"] == "acceptance")["detail"]
    assert "rolling_cv" in detail


def test_publish_fails_when_declared_criterion_has_no_metrics(ctx_ref):
    """钉死的判据面拿不到数就该发布失败，不许悄悄回退到别的面。"""
    ctx, ref = ctx_ref
    goal = tf_goal_create(ctx, _goal_doc(
        acceptance={"cvrmse_max": 0.5, "evaluated_on": "rolling_cv"}))
    mat = tf_dataset_materialize(ctx, view_definition(ref))
    hyp = tf_hypothesis_create(ctx, goal["id"], "忘了开滚动切分")
    plan = tf_experiment_plan(          # 未配 rolling_cv
        ctx, _experiment_doc(goal["id"], hyp["id"], mat["id"]))
    assert tf_experiment_run(ctx, plan["id"])["ok"]
    env = tf_model_publish(ctx, plan["id"], model_id="chiller-power",
                           version="1.0.0", run_smoke=False)
    assert env["ok"] is False
    assert env["diagnostics"][0]["code"] == "TFM-1003"


def _publish_doc(goal_id, hypothesis_id, view_id, model):
    doc = _experiment_doc(goal_id, hypothesis_id, view_id)
    doc["model"] = model
    return doc


def test_model_publish_physics_v2_end_to_end(ctx_ref):
    """cooling_balance_v2 实验可发布（缺陷回归：_LOADERS 曾缺 v2 格式，
    发布构建模型包时被「未知模型格式」拒掉）。"""
    ctx, ref = ctx_ref
    goal = tf_goal_create(ctx, _goal_doc())
    mat = tf_dataset_materialize(ctx, view_definition(ref))
    hyp = tf_hypothesis_create(ctx, goal["id"], "DOE-2 三曲线足够好")
    plan = tf_experiment_plan(ctx, _publish_doc(
        goal["id"], hyp["id"], mat["id"],
        {"category": "physics", "physics": "cooling_balance_v2",
         "hyperparameters": {"rated_capacity_kw": RATED_CAPACITY_KW,
                             "rated_power_kw": RATED_POWER_KW,
                             "inputs": PHYSICS_INPUTS}},
    ))
    assert plan["ok"], plan.get("diagnostics")
    ran = tf_experiment_run(ctx, plan["id"])
    assert ran["ok"], ran["summary"].get("error_code")
    assert ran["status"] == "COMPLETED"

    env = tf_model_publish(ctx, plan["id"], model_id="chiller-v2",
                           version="1.0.0")
    assert env["ok"], env["summary"]
    assert env["status"] == "PRODUCTION"
    # 冷加载冒烟已过门禁；再显式验证制品可脱离实验目录重建
    artifact_dir = ctx.registry.package_dir("chiller-v2", "1.0.0") / "artifact"
    model = load_model_artifact(artifact_dir)
    assert isinstance(model, ChillerPhysicsV2)


def test_model_publish_hybrid_identification_base_end_to_end(ctx_ref):
    """hybrid + 系统辨识族主干可发布、可冷加载（缺陷回归：主干参数曾被
    hybrid 的 model.json 覆盖丢失，load 走 params.yaml 必崩）。"""
    ctx, ref = ctx_ref
    goal = tf_goal_create(ctx, _goal_doc())
    mat = tf_dataset_materialize(ctx, view_definition(ref))
    hyp = tf_hypothesis_create(ctx, goal["id"], "ε-NTU 主干 + 残差足够好")
    plan = tf_experiment_plan(ctx, _publish_doc(
        goal["id"], hyp["id"], mat["id"],
        {"category": "hybrid", "physics": "eps_ntu", "residual": "xgboost",
         "hyperparameters": {
             "inputs": ("t_hot_in=cw_supply_temp;"
                        "t_cold_in=evap_chw_supply_temp;"
                        "f_hot=evap_chw_flow;f_cold=evap_chw_flow"),
             "n_estimators": 50}},
    ))
    assert plan["ok"], plan.get("diagnostics")
    ran = tf_experiment_run(ctx, plan["id"])
    assert ran["ok"], ran["summary"].get("error_code")
    assert ran["status"] == "COMPLETED"

    env = tf_model_publish(ctx, plan["id"], model_id="hx-hybrid",
                           version="1.0.0")
    assert env["ok"], env["summary"]
    assert env["status"] == "PRODUCTION"
    artifact_dir = ctx.registry.package_dir("hx-hybrid", "1.0.0") / "artifact"
    assert (artifact_dir / "base_model.json").exists()  # 主干参数随包
    model = load_model_artifact(artifact_dir)
    assert isinstance(model, ResidualHybrid)
    assert isinstance(model.physics, EffectivenessNTU)


def test_harness_manifest_matches_registry(repo_root):
    with open(repo_root / "harness" / "tools.json", encoding="utf-8") as fp:
        manifest = json.load(fp)
    names = {t["name"] for t in manifest["tools"]}
    assert names == set(TOOL_REGISTRY)
    for entry in manifest["tools"]:
        module, _, func = entry["entry"].partition(":")
        assert module == "thermoforge_research.tools"
        assert callable(TOOL_REGISTRY[entry["name"]])
        assert func == entry["name"]
