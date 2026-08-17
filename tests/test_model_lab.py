"""模型实验室（model_lab）：扫描 → 子进程校验 → 版本化入库 → 人工审批 →
category=lab 实验 → 发布自包含 的全链路测试。

- scan_source：import 白名单 / 危险调用 / dunder 逃生梯 / 体积上限；
- validate_module：五连检（interface/fit_predict/save_contract/roundtrip/
  determinism）在子进程执行，非确定性模块必须被揪出；
- LabStore：幂等重交、版本递增、篡改检测（TFML-004）、未批准门禁
  （TFML-006）；
- 工具层：审批只有 human 可行（TFML-005）；
- e2e：category=lab 实验跑通且源码快照随实验/发布包固化。
"""

from __future__ import annotations

import pytest

from thermoforge_core.contracts.experiment import Experiment
from thermoforge_research.model_lab import (
    TFML_APPROVAL_FORBIDDEN,
    TFML_NOT_APPROVED,
    TFML_SOURCE_VIOLATION,
    TFML_VERSION_CONFLICT,
    LabError,
    LabStore,
    parse_lab_ref,
    scan_source,
    validate_module,
)
from thermoforge_research.runner import current_environment_lock, run_experiment
from thermoforge_research.tools import (
    ToolContext,
    tf_dataset_materialize,
    tf_experiment_plan,
    tf_experiment_run,
    tf_goal_create,
    tf_hypothesis_create,
    tf_lab_approve,
    tf_lab_list,
    tf_lab_submit,
    tf_model_publish,
)
from thermoforge_runtime.artifact import load_model_artifact

from phase2_helpers import FEATURES, TARGET, build_chiller_vault, view_definition
from phase34_helpers import make_ctx

# ---------------------------------------------------------------- 候选模块源码

MINI_SOURCE = '''"""最小线性实验室模块：普通最小二乘 + 截距（测试用，确定性）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from thermoforge_models.lab import parse_inputs

MODEL_FORMAT = "thermoforge.lab.mini_view.v1"
INPUT_ROLES = [
    "evap_chw_flow",
    "evap_chw_supply_temp",
    "evap_chw_return_temp",
    "cw_supply_temp",
]


class MiniLinear:
    def __init__(self, columns):
        self._columns = list(columns)
        self._coef = None
        self._intercept = 0.0

    def _design(self, df):
        x = df[self._columns].to_numpy(np.float64)
        return np.column_stack([x, np.ones(len(x))])

    def fit(self, df, y):
        sol, *_ = np.linalg.lstsq(
            self._design(df), np.asarray(y, dtype=np.float64), rcond=None)
        self._coef = sol[:-1]
        self._intercept = float(sol[-1])
        return self

    def predict(self, df):
        return self._design(df) @ np.append(self._coef, self._intercept)

    def save(self, directory):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        doc = {
            "format": MODEL_FORMAT,
            "columns": self._columns,
            "coef": [float(c) for c in self._coef],
            "intercept": self._intercept,
        }
        with open(path / "model.json", "w", encoding="utf-8") as fp:
            json.dump(doc, fp)


def build_model(hyperparameters: Mapping[str, Any], seed: int):
    inputs = parse_inputs((hyperparameters or {}).get("inputs"))
    columns = ([inputs.get(role, role) for role in INPUT_ROLES]
               if inputs else list(INPUT_ROLES))
    return MiniLinear(columns)


def load_model(directory):
    with open(Path(directory) / "model.json", encoding="utf-8") as fp:
        doc = json.load(fp)
    model = MiniLinear(doc["columns"])
    model._coef = np.asarray(doc["coef"], dtype=np.float64)
    model._intercept = float(doc["intercept"])
    return model
'''

# 扫描干净但同种子重训不一致（无种子随机源）——determinism 检查必须拒绝
JITTER_SOURCE = '''"""非确定性模块：fit 里用无种子随机源。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

MODEL_FORMAT = "thermoforge.lab.jitter.v1"
INPUT_ROLES = []


class Jitter:
    def fit(self, df, y):
        rng = np.random.default_rng()
        self._bias = float(np.mean(y)) + float(rng.normal())
        return self

    def predict(self, df):
        return np.full(len(df), self._bias)

    def save(self, directory):
        with open(Path(directory) / "model.json", "w",
                  encoding="utf-8") as fp:
            json.dump({"format": MODEL_FORMAT, "bias": self._bias}, fp)


def build_model(hyperparameters, seed):
    return Jitter()


def load_model(directory):
    with open(Path(directory) / "model.json", encoding="utf-8") as fp:
        doc = json.load(fp)
    model = Jitter()
    model._bias = float(doc["bias"])
    return model
'''

NO_BUILD_SOURCE = '''"""缺 build_model 的模块。"""

MODEL_FORMAT = "thermoforge.lab.no_build.v1"
INPUT_ROLES = []


def load_model(directory):
    raise NotImplementedError
'''


# ---------------------------------------------------------------- 静态扫描

def test_scan_source_accepts_mini_module():
    assert scan_source(MINI_SOURCE) == []


@pytest.mark.parametrize("bad, needle", [
    ("import socket\n", "白名单"),
    ("from urllib import request\n", "白名单"),
    ("import os\nos.system('echo hi')\n", "system"),
    ("eval('1+1')\n", "eval"),
    ("def f():\n    return f.__globals__\n", "dunder"),
    ("x = __subclasses__\n", "dunder"),
    ("from . import sibling\n", "相对 import"),
])
def test_scan_source_rejects_dangerous_constructs(bad, needle):
    violations = scan_source(bad)
    assert violations, f"应检出违规: {bad!r}"
    assert any(needle in v for v in violations), violations


def test_scan_source_rejects_oversize():
    source = MINI_SOURCE + "\n# " + "x" * (64 * 1024)
    assert scan_source(source) == ["源码超过 65536 字节上限"]


def test_parse_lab_ref():
    assert parse_lab_ref("mini_view") == ("mini_view", None)
    assert parse_lab_ref("mini_view@v3") == ("mini_view", 3)
    with pytest.raises(LabError):
        parse_lab_ref("Mini_View")  # 大写非法
    with pytest.raises(LabError):
        parse_lab_ref("mini_view@v0")


# ---------------------------------------------------------------- 子进程结构校验

def test_validate_module_accepts_deterministic(tmp_path):
    src = tmp_path / "mini_view.py"
    src.write_text(MINI_SOURCE, encoding="utf-8")
    report = validate_module(src, tmp_path / "check_ok")
    assert report["ok"], report
    assert [c["name"] for c in report["checks"]] == [
        "interface", "fit_predict", "save_contract", "roundtrip",
        "determinism",
    ]
    assert all(c["ok"] for c in report["checks"])


def test_validate_module_rejects_nondeterministic(tmp_path):
    src = tmp_path / "jitter.py"
    src.write_text(JITTER_SOURCE, encoding="utf-8")
    report = validate_module(src, tmp_path / "check_jitter")
    assert report["ok"] is False
    checks = {c["name"]: c["ok"] for c in report["checks"]}
    assert checks["roundtrip"] is True   # 保存/重载一致
    assert checks["determinism"] is False  # 重训不一致 → 揪出


def test_validate_module_rejects_missing_interface(tmp_path):
    src = tmp_path / "no_build.py"
    src.write_text(NO_BUILD_SOURCE, encoding="utf-8")
    report = validate_module(src, tmp_path / "check_iface")
    assert report["ok"] is False
    assert report["checks"][0]["name"] == "interface"
    assert report["checks"][0]["ok"] is False


# ---------------------------------------------------------------- 版本化存储

def test_lab_store_submit_idempotent_and_versioned(tmp_path):
    store = LabStore(tmp_path)
    first = store.submit("mini_view", MINI_SOURCE)
    assert first["version"] == 1 and first["status"] == "proposed"
    again = store.submit("mini_view", MINI_SOURCE)  # 幂等重交
    assert again["version"] == 1
    bumped = store.submit("mini_view", MINI_SOURCE + "\n# v2\n")
    assert bumped["version"] == 2
    assert store.latest_version("mini_view") == 2
    assert [m["version"] for m in store.list()] == [1, 2]


def test_lab_store_detects_tampering(tmp_path):
    store = LabStore(tmp_path)
    store.submit("mini_view", MINI_SOURCE)
    path = store.dir / "mini_view.v1.py"
    path.write_text(MINI_SOURCE + "\n# 被篡改\n", encoding="utf-8")
    with pytest.raises(LabError) as excinfo:
        store.get("mini_view", 1)
    assert excinfo.value.code == TFML_VERSION_CONFLICT


def test_lab_store_require_approved_gate(tmp_path):
    store = LabStore(tmp_path)
    store.submit("mini_view", MINI_SOURCE)
    with pytest.raises(LabError) as excinfo:
        store.require_approved("mini_view")
    assert excinfo.value.code == TFML_NOT_APPROVED
    store.approve("mini_view", actor="human")
    record = store.require_approved("mini_view")
    assert record["status"] == "approved"
    assert record["approvals"][0]["actor"] == "human"


# ---------------------------------------------------------------- 工具层

def test_lab_tools_submit_approve_list(tmp_path):
    ctx, _ = make_ctx(tmp_path)
    env = tf_lab_submit(ctx, "mini_view", MINI_SOURCE)
    assert env["ok"], env["summary"]
    assert env["status"] == "PROPOSED"
    assert env["summary"]["validation"]["ok"]

    # 非 human 审批被拒（与预处理审批同纪律）
    denied = tf_lab_approve(ctx, "mini_view")
    assert denied["ok"] is False
    assert denied["diagnostics"][0]["code"] == TFML_APPROVAL_FORBIDDEN

    human_ctx = ToolContext(
        vault_root=ctx.vault_root, research_root=ctx.research_root,
        models_root=ctx.models_root, actor="human",
    )
    approved = tf_lab_approve(human_ctx, "mini_view", note="看过源码")
    assert approved["ok"], approved["summary"]

    listed = tf_lab_list(ctx)
    assert listed["ok"]
    modules = {m["ref"]: m["status"] for m in listed["summary"]["modules"]}
    assert modules == {"mini_view@v1": "approved"}


def test_lab_tools_submit_scan_violation(tmp_path):
    ctx, _ = make_ctx(tmp_path)
    env = tf_lab_submit(ctx, "evil", "import socket\n")
    assert env["ok"] is False
    assert env["diagnostics"][0]["code"] == TFML_SOURCE_VIOLATION


# ---------------------------------------------------------------- 契约

def test_lab_contract_rules():
    base = dict(
        experiment_id="EXP-9001", goal_id="RG-0001", hypothesis_id="H-0001",
        dataset_view="VIEW-0001", target=TARGET,
        validation={"temporal_split": {"train": 0.7, "validate": 0.15,
                                       "test": 0.15}},
        metrics=["RMSE"],
        runtime={"environment_lock": "test", "random_seed": 1},
    )
    with pytest.raises(ValueError, match="hyperparameters.lab"):
        Experiment(model={"category": "lab"}, **base)
    with pytest.raises(ValueError, match="不得声明"):
        Experiment(
            model={"category": "lab", "physics": "cooling_balance_v1",
                   "hyperparameters": {"lab": "mini_view"}},
            **base,
        )
    exp = Experiment(
        model={"category": "lab",
               "hyperparameters": {"lab": "mini_view@v1"}},
        **base,
    )
    assert exp.model.category == "lab"


# ---------------------------------------------------------------- runner e2e

def _lab_experiment(exp_id: str, lab_ref: str = "mini_view") -> Experiment:
    return Experiment(
        experiment_id=exp_id,
        goal_id="RG-0001",
        hypothesis_id="H-0001",
        dataset_view="VIEW-0001",
        model={
            "category": "lab",
            "hyperparameters": {"lab": lab_ref, "inputs": ";".join(FEATURES)},
        },
        target=TARGET,
        validation={
            "temporal_split": {"train": 0.70, "validate": 0.15, "test": 0.15},
            "equipment_holdout": {"enabled": False, "holdout_objects": []},
        },
        metrics=["RMSE", "MAE", "MAPE", "CVRMSE", "NMBE"],
        physics_tests={"enabled": True},
        runtime={"environment_lock": current_environment_lock()[0],
                 "random_seed": 20260808},
    )


def test_lab_experiment_end_to_end(tmp_path):
    _, ref = build_chiller_vault(tmp_path)
    research_root = tmp_path / "research"
    store = LabStore(research_root)
    store.submit("mini_view", MINI_SOURCE)
    store.approve("mini_view", actor="human")

    report = run_experiment(
        _lab_experiment("EXP-9101"),
        research_root=research_root,
        vault_root=tmp_path / "vault",
        view_definition=view_definition(ref),
    )
    assert report["status"] == "completed", report.get("error")
    exp_dir = research_root / "experiments" / "EXP-9101"
    # 源码快照固化在实验目录，复现不依赖实验室存储
    assert (exp_dir / "lab_module.py").exists()
    # 源码随模型制品打包（发布自包含的前提）
    assert list(exp_dir.rglob("lab_source.py"))
    surfaces = report["metrics"]["surfaces"]
    assert surfaces["A"]["metrics"]["CVRMSE"] is not None


def test_lab_experiment_rejected_when_not_approved(tmp_path):
    _, ref = build_chiller_vault(tmp_path)
    research_root = tmp_path / "research"
    LabStore(research_root).submit("mini_view", MINI_SOURCE)  # 不审批

    with pytest.raises(LabError) as excinfo:
        run_experiment(
            _lab_experiment("EXP-9102"),
            research_root=research_root,
            vault_root=tmp_path / "vault",
            view_definition=view_definition(ref),
        )
    assert excinfo.value.code == TFML_NOT_APPROVED


# ---------------------------------------------------------------- 发布 e2e

def test_lab_publish_produces_self_contained_package(tmp_path):
    ctx, ref = make_ctx(tmp_path, n_steps=600)
    submitted = tf_lab_submit(ctx, "mini_view", MINI_SOURCE)
    assert submitted["ok"]
    human_ctx = ToolContext(
        vault_root=ctx.vault_root, research_root=ctx.research_root,
        models_root=ctx.models_root, actor="human",
    )
    assert tf_lab_approve(human_ctx, "mini_view")["ok"]

    goal = tf_goal_create(ctx, {
        "name": "冷水机输入功率模型",
        "object_model": "chiller.v1",
        "purpose": "optimization",
        "target": TARGET,
        "candidate_inputs": list(FEATURES),
        "acceptance": {"cvrmse_max": 0.5},
    })
    mat = tf_dataset_materialize(ctx, view_definition(ref))
    hyp = tf_hypothesis_create(ctx, goal["id"], "实验室最小线性方案足够好")
    plan = tf_experiment_plan(ctx, {
        "goal_id": goal["id"],
        "hypothesis_id": hyp["id"],
        "dataset_view": mat["id"],
        "model": {"category": "lab",
                  "hyperparameters": {"lab": "mini_view"}},
        "target": TARGET,
        "validation": {"temporal_split": {"train": 0.70, "validate": 0.15,
                                          "test": 0.15}},
        "metrics": ["RMSE", "MAE", "MAPE", "CVRMSE", "NMBE"],
        "runtime": {"environment_lock": current_environment_lock()[0],
                    "random_seed": 20260808},
    })
    assert plan["ok"], plan.get("diagnostics")
    ran = tf_experiment_run(ctx, plan["id"])
    assert ran["ok"], ran["summary"].get("error_code")
    assert ran["status"] == "COMPLETED"

    env = tf_model_publish(ctx, plan["id"], model_id="lab-linear",
                           version="1.0.0")
    assert env["ok"], env["summary"]
    assert env["status"] == "PRODUCTION"

    package_dir = ctx.registry.package_dir("lab-linear", "1.0.0")
    sources = list(package_dir.rglob("lab_source.py"))
    assert sources, "发布包必须携带实验室模块源码"
    # 冷加载：不经过实验室存储，直接从发布包 artifact 重建模型
    artifact_dir = sources[0].parent
    model = load_model_artifact(artifact_dir)
    assert model is not None
