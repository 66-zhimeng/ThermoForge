"""CLI（thermoforge_cli / `tf`）测试。

- 每个命令族至少一例：参数解析 → 信封 JSON 输出；
- exit code 语义：工具级失败 exit 0（ok=false）、CLI 错误 exit 2；
- tf status 空仓库 / 有数据仓库；
- stdout 纯净：默认输出可直接 json.loads。
"""

from __future__ import annotations

import json

import pytest

from thermoforge_cli.main import main
from thermoforge_research.runner import current_environment_lock
from thermoforge_research.tools import (
    tf_dataset_materialize,
    tf_experiment_plan,
    tf_goal_create,
    tf_hypothesis_create,
)

from phase2_helpers import FEATURES, TARGET, view_definition
from phase34_helpers import make_ctx
from test_preprocess import _ruleset_doc, _small_legacy_workbook


def run_cli(capsys, tmp_path, *argv):
    code = main(["--vault-root", str(tmp_path / "vault"),
                 "--research-root", str(tmp_path / "research"),
                 "--models-root", str(tmp_path / "models"),
                 *argv])
    out = capsys.readouterr()
    assert out.err == "" or code == 2  # 成功路径 stderr 无杂讯
    return code, out.out


def run_json(capsys, tmp_path, *argv):
    code, raw = run_cli(capsys, tmp_path, *argv)
    return code, json.loads(raw)  # stdout 必须可直接解析


@pytest.fixture()
def ctx_ref(tmp_path):
    return make_ctx(tmp_path, n_steps=300)


# ---------------------------------------------------------------- dataset 族


def test_dataset_list(ctx_ref, capsys, tmp_path):
    code, env = run_json(capsys, tmp_path, "dataset", "list")
    assert code == 0 and env["ok"] and env["tool"] == "tf_dataset_list"
    assert env["summary"]["count"] == 1


def test_dataset_get_tool_failure_exit_zero(ctx_ref, capsys, tmp_path):
    code, env = run_json(capsys, tmp_path, "dataset", "get", "NOPE@rev_9999")
    assert code == 0  # 工具级失败不靠退出码
    assert env["ok"] is False
    assert env["diagnostics"][0]["code"] == "TFV-703"


def test_dataset_schema(ctx_ref, capsys, tmp_path):
    ctx, ref = ctx_ref
    code, env = run_json(capsys, tmp_path, "dataset", "schema", ref)
    assert code == 0 and env["ok"]
    assert f"CH-01.{TARGET}" in env["summary"]["variable_ids"]
    assert env["summary"]["variables_in_artifact"] == len(
        env["summary"]["variable_ids"])


def test_dataset_profile_and_query(ctx_ref, capsys, tmp_path):
    ctx, ref = ctx_ref
    code, env = run_json(capsys, tmp_path, "dataset", "profile", ref)
    assert code == 0
    assert env["summary"]["quantile_points"] == [
        "min", "p01", "p25", "p50", "p75", "p99", "max"]
    code, env = run_json(capsys, tmp_path, "dataset", "query", ref,
                         "--variables", f"CH-01.{TARGET}")
    assert code == 0
    stats = env["summary"]["stats"][f"CH-01.{TARGET}"]
    assert stats["count"] == 300


def test_dataset_sample_cap(ctx_ref, capsys, tmp_path):
    ctx, ref = ctx_ref
    code, env = run_json(capsys, tmp_path, "dataset", "sample", ref,
                         "--n", "500", "--variables", f"CH-01.{TARGET}")
    assert code == 0
    assert env["summary"]["returned"] == 200 and env["truncated"] is True


def test_dataset_materialize_and_compare(ctx_ref, capsys, tmp_path):
    ctx, ref = ctx_ref
    doc = json.dumps(view_definition(ref))
    code, env = run_json(capsys, tmp_path, "dataset", "materialize",
                         "--definition-json", doc)
    assert code == 0 and env["ok"] and env["id"] == "VIEW-0001"
    code, env = run_json(capsys, tmp_path, "dataset", "compare", ref, ref)
    assert code == 0 and env["summary"]["content_identical"] is True


def test_dataset_modelability(ctx_ref, capsys, tmp_path):
    ctx, ref = ctx_ref
    goal = tf_goal_create(ctx, {
        "name": "g", "object_model": "chiller.v1", "purpose": "optimization",
        "target": TARGET, "candidate_inputs": list(FEATURES),
        "acceptance": {"cvrmse_max": 0.5}}, dataset_ref=ref)
    code, env = run_json(capsys, tmp_path, "dataset", "modelability", ref,
                         "--goal-id", goal["id"])
    assert code == 0 and env["status"] == "PASS"


# ---------------------------------------------------------------- 研究族


def test_goal_create_and_whitelist_reject(ctx_ref, capsys, tmp_path):
    ctx, ref = ctx_ref
    doc = {"name": "g", "object_model": "chiller.v1",
           "purpose": "optimization", "target": TARGET,
           "candidate_inputs": list(FEATURES),
           "acceptance": {"cvrmse_max": 0.5}}
    code, env = run_json(capsys, tmp_path, "goal", "create",
                         "--definition-json", json.dumps(doc),
                         "--dataset-ref", ref)
    assert code == 0 and env["ok"] and env["id"] == "RG-0001"
    bad = {**doc, "candidate_inputs": ["no_such_prop"]}
    code, env = run_json(capsys, tmp_path, "goal", "create",
                         "--definition-json", json.dumps(bad),
                         "--dataset-ref", ref)
    assert code == 0 and env["ok"] is False  # 白名单拒绝，exit 仍 0


def test_research_status(ctx_ref, capsys, tmp_path):
    ctx, ref = ctx_ref
    goal = tf_goal_create(ctx, {
        "name": "g", "object_model": "chiller.v1", "purpose": "optimization",
        "target": TARGET, "candidate_inputs": list(FEATURES),
        "acceptance": {"cvrmse_max": 0.5}})
    code, env = run_json(capsys, tmp_path, "research", "status",
                         "--goal-id", goal["id"])
    assert code == 0 and env["ok"]
    assert env["summary"]["progress"]["goal_id"] == goal["id"]


def test_hypothesis_basis_enforced(ctx_ref, capsys, tmp_path):
    ctx, ref = ctx_ref
    goal = tf_goal_create(ctx, {
        "name": "g", "object_model": "chiller.v1", "purpose": "optimization",
        "target": TARGET, "candidate_inputs": list(FEATURES),
        "acceptance": {"cvrmse_max": 0.5}})
    code, env = run_json(capsys, tmp_path, "hypothesis", "create",
                         goal["id"], "首个假设")
    assert code == 0 and env["ok"]
    code, env = run_json(capsys, tmp_path, "hypothesis", "create",
                         goal["id"], "无证据的后续假设")
    assert code == 0 and env["ok"] is False


def test_experiment_family(ctx_ref, capsys, tmp_path):
    """plan（文件入参）→ run（子进程）→ get → model compare。"""
    ctx, ref = ctx_ref
    goal = tf_goal_create(ctx, {
        "name": "g", "object_model": "chiller.v1", "purpose": "optimization",
        "target": TARGET, "candidate_inputs": list(FEATURES),
        "acceptance": {"cvrmse_max": 0.5}})
    mat = tf_dataset_materialize(ctx, view_definition(ref))
    hyp = tf_hypothesis_create(ctx, goal["id"], "h")
    exp_doc = {
        "goal_id": goal["id"], "hypothesis_id": hyp["id"],
        "dataset_view": mat["id"],
        "model": {"category": "data", "estimator": "ridge",
                  "hyperparameters": {"alpha": 0.5}},
        "target": TARGET,
        "validation": {"temporal_split": {"train": 0.7, "validate": 0.15,
                                          "test": 0.15}},
        "metrics": ["RMSE", "MAE", "MAPE", "CVRMSE", "NMBE"],
        "runtime": {"environment_lock": current_environment_lock()[0],
                    "random_seed": 20260808},
    }
    doc_path = tmp_path / "exp.json"
    doc_path.write_text(json.dumps(exp_doc), encoding="utf-8", newline="\n")
    code, env = run_json(capsys, tmp_path, "experiment", "plan",
                         "--definition-file", str(doc_path))
    assert code == 0 and env["ok"] and env["id"] == "EXP-0001"

    code, env = run_json(capsys, tmp_path, "experiment", "run", "EXP-0001")
    assert code == 0 and env["ok"], env["summary"].get("error_code")
    assert env["summary"]["surfaces"]["A"]["metrics"]["CVRMSE"] is not None

    code, env = run_json(capsys, tmp_path, "experiment", "get", "EXP-0001")
    assert code == 0 and env["summary"]["status"] == "completed"

    code, env = run_json(capsys, tmp_path, "model", "compare",
                         "EXP-0001", "EXP-9999")
    assert code == 0 and env["summary"]["best"] == "EXP-0001"
    assert env["summary"]["skipped"] == ["EXP-9999"]


def test_model_publish(ctx_ref, capsys, tmp_path):
    ctx, ref = ctx_ref
    goal = tf_goal_create(ctx, {
        "name": "g", "object_model": "chiller.v1", "purpose": "optimization",
        "target": TARGET, "candidate_inputs": list(FEATURES),
        "acceptance": {"cvrmse_max": 0.9}})
    mat = tf_dataset_materialize(ctx, view_definition(ref))
    hyp = tf_hypothesis_create(ctx, goal["id"], "h")
    plan = tf_experiment_plan(ctx, {
        "goal_id": goal["id"], "hypothesis_id": hyp["id"],
        "dataset_view": mat["id"],
        "model": {"category": "data", "estimator": "ridge"},
        "target": TARGET,
        "validation": {"temporal_split": {"train": 0.7, "validate": 0.15,
                                          "test": 0.15}},
        "metrics": ["CVRMSE", "NMBE"],
        "runtime": {"environment_lock": current_environment_lock()[0],
                    "random_seed": 1}})
    code, env = run_json(capsys, tmp_path, "experiment", "run", plan["id"])
    assert code == 0 and env["ok"]
    code, env = run_json(capsys, tmp_path, "model", "publish", plan["id"],
                         "--model-id", "chiller-power", "--version", "1.0.0")
    assert code == 0 and env["ok"], env["summary"].get("error")
    assert env["id"] == "chiller-power@1.0.0"
    gate_names = [g["name"] for g in env["summary"]["gates"]]
    assert "smoke" in gate_names  # 冷加载冒烟真实执行


# ---------------------------------------------------------------- preprocess 族


def test_preprocess_family(ctx_ref, capsys, tmp_path):
    src = _small_legacy_workbook(tmp_path / "legacy.xlsx")
    code, env = run_json(capsys, tmp_path, "preprocess", "propose",
                         "--ruleset-json", json.dumps(_ruleset_doc()))
    assert code == 0 and env["ok"] and env["id"] == "WX_FIX@v1"

    code, env = run_json(capsys, tmp_path, "preprocess", "list")
    assert code == 0 and env["summary"]["count"] == 1

    # 默认 actor=cli：审批被拒（TFPP-006）
    code, env = run_json(capsys, tmp_path, "preprocess", "approve", "WX_FIX")
    assert code == 0 and env["ok"] is False
    assert env["diagnostics"][0]["code"] == "TFPP-006"

    # --actor human：审批通过；apply 预览执行
    code, env = run_json(capsys, tmp_path, "--actor", "human",
                         "preprocess", "approve", "WX_FIX", "--note", "ok")
    assert code == 0 and env["status"] == "APPROVED"
    code, env = run_json(capsys, tmp_path, "preprocess", "apply",
                         str(src), "WX_FIX")
    assert code == 0 and env["ok"] and env["status"] == "APPLIED"


# ---------------------------------------------------------------- exit code 与 status


def test_parse_error_exit_2(tmp_path):
    with pytest.raises(SystemExit) as excinfo:  # argparse 语义
        main(["--vault-root", str(tmp_path / "v"), "dataset"])
    assert excinfo.value.code == 2


def test_cli_error_exit_2(capsys, tmp_path):
    code = main(["--vault-root", str(tmp_path / "vault"),
                 "--research-root", str(tmp_path / "research"),
                 "--models-root", str(tmp_path / "models"),
                 "dataset", "list", "--param", "not-a-kv"])
    assert code == 2
    assert "k=v" in capsys.readouterr().err


def test_status_empty_repo(capsys, tmp_path):
    code, raw = run_cli(capsys, tmp_path, "status")
    assert code == 0
    assert "（无）" in raw
    # 空仓库面板纯读：不创建 research/models/vault 目录
    assert not (tmp_path / "research").exists()
    code, status = run_json(capsys, tmp_path, "status", "--json")
    assert code == 0
    assert status["goals"] == [] and status["production_models"] == []


def test_status_with_data(ctx_ref, capsys, tmp_path):
    ctx, ref = ctx_ref
    goal = tf_goal_create(ctx, {
        "name": "g", "object_model": "chiller.v1", "purpose": "optimization",
        "target": TARGET, "candidate_inputs": list(FEATURES),
        "acceptance": {"cvrmse_max": 0.5}})
    code, raw = run_cli(capsys, tmp_path, "status")
    assert code == 0
    assert goal["id"] in raw and "Research Goals" in raw
    code, status = run_json(capsys, tmp_path, "status", "--json")
    assert status["goals"][0]["goal_id"] == goal["id"]


def test_pretty_is_indented_json(ctx_ref, capsys, tmp_path):
    code, raw = run_cli(capsys, tmp_path, "--pretty", "dataset", "list")
    assert code == 0
    assert "\n  " in raw  # 缩进格式
    assert json.loads(raw)["ok"] is True  # 仍可机读
