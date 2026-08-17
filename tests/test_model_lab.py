"""模型实验室（model_lab）：扫描 → 子进程校验 → 版本化入库 →
category=lab 实验 → 发布自包含 的全链路测试。

这条通路是**自治**的：过了结构校验就能跑真实数据，没有人工审批环节。
测试要盯住的是「机器判定的门禁真的挡得住东西」，而不是「人批了没有」。

- scan_source：import 白名单 / 禁用子模块 / 危险调用 / dunder 逃生梯 /
  体积上限；
- validate_module：五连检（interface/fit_predict/save_contract/roundtrip/
  determinism）在子进程执行，非确定性模块必须被揪出；
- LabStore：幂等重交、版本递增、篡改检测（TFML-004）、可运行门禁
  （TFML-006：未过校验或已停用）；
- e2e：提交后**不经任何审批**直接跑通 category=lab 实验并发布，
  源码快照随实验/发布包固化。
"""

from __future__ import annotations

import json

import pytest

from thermoforge_core.contracts.experiment import Experiment
from thermoforge_research.model_lab import (
    TFML_NOT_RUNNABLE,
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
    tf_dataset_materialize,
    tf_experiment_plan,
    tf_experiment_run,
    tf_goal_create,
    tf_hypothesis_create,
    tf_lab_deprecate,
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

# 直接用 LabStore 时省掉真跑一遍子进程五连检（工具层测试覆盖真校验）
OK_VALIDATION = {"ok": True, "checks": [], "model_format": "thermoforge.lab.x"}

# 「除记账列外都当特征」的贪心模块——最自然、也最容易把目标列训进去的写法。
# runner 必须让它拿不到目标列（否则 R² 直奔 0.99，是拿 y predict y）。
GREEDY_SOURCE = '''"""贪心模块：把拿到的每一列都当特征。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

MODEL_FORMAT = "thermoforge.lab.greedy.v1"
INPUT_ROLES = []

_SKIP = ("object_id", "timestamp")


class Greedy:
    def __init__(self):
        self._columns = None
        self._coef = None
        self._intercept = 0.0

    def _design(self, df):
        x = df[self._columns].to_numpy(np.float64)
        return np.column_stack([x, np.ones(len(x))])

    def fit(self, df, y):
        self._columns = [c for c in df.columns if c not in _SKIP]
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
        doc = {"format": MODEL_FORMAT, "columns": self._columns,
               "coef": [float(c) for c in self._coef],
               "intercept": self._intercept}
        with open(path / "model.json", "w", encoding="utf-8") as fp:
            json.dump(doc, fp)


def build_model(hyperparameters, seed):
    return Greedy()


def load_model(directory):
    with open(Path(directory) / "model.json", encoding="utf-8") as fp:
        doc = json.load(fp)
    model = Greedy()
    model._columns = list(doc["columns"])
    model._coef = np.asarray(doc["coef"], dtype=np.float64)
    model._intercept = float(doc["intercept"])
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
    # 白名单根模块下的已知逃逸口（扫描是执行前唯一的自动关卡）
    ("import pandas as pd\npd.read_pickle('http://x/y.pkl')\n", "read_pickle"),
    ("import numpy as np\nnp.ctypeslib.load_library('a', '.')\n",
     "load_library"),
    ("import sklearn.datasets as ds\n", "禁用子模块"),
    ("from sklearn import datasets\n", "禁用子模块"),
    ("from numpy import ctypeslib\n", "禁用子模块"),
])
def test_scan_source_rejects_dangerous_constructs(bad, needle):
    violations = scan_source(bad)
    assert violations, f"应检出违规: {bad!r}"
    assert any(needle in v for v in violations), violations


def test_scan_source_allows_json_and_project_imports():
    """收紧扫描时最容易误伤的两处：json.loads 与 thermoforge_models 子模块。"""
    ok = ("import json\n"
          "from thermoforge_models.lab import parse_inputs\n"
          "def load_model(directory):\n"
          "    return json.loads('{}')\n")
    assert scan_source(ok) == []


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
    first = store.submit("mini_view", MINI_SOURCE, validation=OK_VALIDATION)
    assert first["version"] == 1 and first["status"] == "validated"
    again = store.submit("mini_view", MINI_SOURCE,
                         validation=OK_VALIDATION)  # 幂等重交
    assert again["version"] == 1
    bumped = store.submit("mini_view", MINI_SOURCE + "\n# v2\n",
                          validation=OK_VALIDATION)
    assert bumped["version"] == 2
    assert store.latest_version("mini_view") == 2
    assert [m["version"] for m in store.list()] == [1, 2]


def test_lab_store_marks_unvalidated_when_no_report(tmp_path):
    """没有校验报告就不该是 validated——门禁只认机器判定过的模块。"""
    store = LabStore(tmp_path)
    record = store.submit("bare", MINI_SOURCE)
    assert record["status"] == "unvalidated"
    assert store.runnable_refs() == []


def test_lab_store_detects_tampering(tmp_path):
    store = LabStore(tmp_path)
    store.submit("mini_view", MINI_SOURCE, validation=OK_VALIDATION)
    path = store.dir / "mini_view.v1.py"
    path.write_text(MINI_SOURCE + "\n# 被篡改\n", encoding="utf-8")
    with pytest.raises(LabError) as excinfo:
        store.get("mini_view", 1)
    assert excinfo.value.code == TFML_VERSION_CONFLICT


def test_lab_store_runnable_right_after_submit(tmp_path):
    """过了校验就能跑——中间没有任何人工环节。"""
    store = LabStore(tmp_path)
    store.submit("mini_view", MINI_SOURCE, validation=OK_VALIDATION)
    record = store.require_runnable("mini_view")
    assert record["status"] == "validated"
    assert store.runnable_refs() == ["mini_view@v1"]


def test_lab_store_gate_blocks_unvalidated_and_deprecated(tmp_path):
    store = LabStore(tmp_path)
    store.submit("bare", MINI_SOURCE)  # 无校验报告
    with pytest.raises(LabError) as excinfo:
        store.require_runnable("bare")
    assert excinfo.value.code == TFML_NOT_RUNNABLE

    store.submit("mini_view", MINI_SOURCE, validation=OK_VALIDATION)
    store.deprecate("mini_view", actor="agent", note="被 v2 取代")
    with pytest.raises(LabError) as excinfo:
        store.require_runnable("mini_view")
    assert excinfo.value.code == TFML_NOT_RUNNABLE
    record = store.get("mini_view")
    assert record["status"] == "deprecated"
    assert record["audit"][0]["action"] == "deprecate"
    assert record["audit"][0]["actor"] == "agent"


def test_lab_store_normalizes_legacy_approval_statuses(tmp_path):
    """取消审批前入库的模块：能否运行改由 validation 重判，不看旧 status。"""
    store = LabStore(tmp_path)
    store.submit("legacy_ok", MINI_SOURCE, validation=OK_VALIDATION)
    meta_path = store.dir / "legacy_ok.v1.yaml"
    meta_path.write_text(
        meta_path.read_text(encoding="utf-8").replace(
            "status: validated", "status: proposed"),
        encoding="utf-8", newline="\n")
    assert store.require_runnable("legacy_ok")["status"] == "validated"
    assert store.list()[0]["runnable"] is True


# ---------------------------------------------------------------- 工具层

def test_lab_tools_submit_is_runnable_without_approval(tmp_path):
    """agent 身份提交 → 立刻 runnable，全程没有 human 参与。"""
    ctx, _ = make_ctx(tmp_path)
    assert ctx.actor != "human"
    env = tf_lab_submit(ctx, "mini_view", MINI_SOURCE)
    assert env["ok"], env["summary"]
    assert env["status"] == "VALIDATED"
    assert env["summary"]["status"] == "validated"
    assert env["summary"]["validation"]["ok"]

    listed = tf_lab_list(ctx)
    assert listed["ok"]
    assert listed["summary"]["runnable"] == ["mini_view@v1"]


def test_lab_tools_deprecate_removes_from_runnable(tmp_path):
    ctx, _ = make_ctx(tmp_path)
    assert tf_lab_submit(ctx, "mini_view", MINI_SOURCE)["ok"]
    env = tf_lab_deprecate(ctx, "mini_view", note="走不通")
    assert env["ok"], env["summary"]
    assert env["status"] == "DEPRECATED"
    assert env["summary"]["audit"][-1]["action"] == "deprecate"

    listed = tf_lab_list(ctx)
    assert listed["summary"]["runnable"] == []
    assert listed["summary"]["modules"][0]["status"] == "deprecated"


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
    store.submit("mini_view", MINI_SOURCE, validation=OK_VALIDATION)

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


def test_lab_iteration_loop_v1_then_v2(tmp_path):
    """自治闭环：跑完 v1 → 改代码 → v2 立刻能跑，中间没有审批。

    两轮各自冻结自己的源码快照，历史实验不被新版本改写。
    """
    _, ref = build_chiller_vault(tmp_path)
    research_root = tmp_path / "research"
    store = LabStore(research_root)
    store.submit("mini_view", MINI_SOURCE, validation=OK_VALIDATION)

    first = run_experiment(
        _lab_experiment("EXP-9111", lab_ref="mini_view@v1"),
        research_root=research_root, vault_root=tmp_path / "vault",
        view_definition=view_definition(ref),
    )
    assert first["status"] == "completed", first.get("error")

    # agent 读完指标改模型：源码一变就是新版本，无需任何人放行
    v2 = store.submit("mini_view", MINI_SOURCE + "\n# v2: 换了设计矩阵\n",
                      validation=OK_VALIDATION)
    assert v2["version"] == 2 and v2["status"] == "validated"

    second = run_experiment(
        _lab_experiment("EXP-9112", lab_ref="mini_view@v2"),
        research_root=research_root, vault_root=tmp_path / "vault",
        view_definition=view_definition(ref),
    )
    assert second["status"] == "completed", second.get("error")

    exps = research_root / "experiments"
    snap1 = (exps / "EXP-9111" / "lab_module.py").read_text(encoding="utf-8")
    snap2 = (exps / "EXP-9112" / "lab_module.py").read_text(encoding="utf-8")
    assert "# v2" not in snap1 and "# v2" in snap2


def test_lab_model_never_sees_target_column(tmp_path):
    """贪心模块拿不到目标列——否则 R² 是拿 y predict y 的假精度。

    DD-16 的白名单门禁管的是「视图特征 ⊆ 白名单」，管不到模块自己从
    DataFrame 里伸手取列；结构上堵死的位置在 `_child._model_frame`。
    """
    _, ref = build_chiller_vault(tmp_path)
    research_root = tmp_path / "research"
    LabStore(research_root).submit("greedy", GREEDY_SOURCE,
                                   validation=OK_VALIDATION)

    report = run_experiment(
        _lab_experiment("EXP-9121", lab_ref="greedy"),
        research_root=research_root, vault_root=tmp_path / "vault",
        view_definition=view_definition(ref),
    )
    assert report["status"] == "completed", report.get("error")

    saved = json.loads(
        (research_root / "experiments" / "EXP-9121" / "model" / "model.json")
        .read_text(encoding="utf-8"))
    assert TARGET not in saved["columns"], (
        f"目标列 {TARGET} 进了训练特征：{saved['columns']}")
    assert set(saved["columns"]) <= set(FEATURES), saved["columns"]


def test_model_frame_drops_target_and_keeps_bookkeeping():
    """投影函数本身的单元断言（三处 predict 与 fit 都过它）。"""
    import pandas as pd

    from thermoforge_research._child import _model_frame

    df = pd.DataFrame({
        "object_id": ["CH01"], "timestamp": [pd.Timestamp("2025-01-01")],
        "feat_a": [1.0], "feat_b": [2.0], "power": [99.0],
    })
    out = _model_frame(df, ["feat_a", "feat_b"])
    assert list(out.columns) == ["object_id", "timestamp", "feat_a", "feat_b"]
    assert "power" not in out.columns


def test_lab_experiment_rejected_when_deprecated(tmp_path):
    """停用是事后否决：停掉之后就不能再开新实验。"""
    _, ref = build_chiller_vault(tmp_path)
    research_root = tmp_path / "research"
    store = LabStore(research_root)
    store.submit("mini_view", MINI_SOURCE, validation=OK_VALIDATION)
    store.deprecate("mini_view", actor="agent")

    with pytest.raises(LabError) as excinfo:
        run_experiment(
            _lab_experiment("EXP-9102"),
            research_root=research_root,
            vault_root=tmp_path / "vault",
            view_definition=view_definition(ref),
        )
    assert excinfo.value.code == TFML_NOT_RUNNABLE


def test_lab_experiment_rejected_when_unvalidated(tmp_path):
    """没过五连检的模块连子进程都进不去（机器判定的硬门禁）。"""
    _, ref = build_chiller_vault(tmp_path)
    research_root = tmp_path / "research"
    LabStore(research_root).submit("mini_view", MINI_SOURCE)  # 无校验报告

    with pytest.raises(LabError) as excinfo:
        run_experiment(
            _lab_experiment("EXP-9103"),
            research_root=research_root,
            vault_root=tmp_path / "vault",
            view_definition=view_definition(ref),
        )
    assert excinfo.value.code == TFML_NOT_RUNNABLE


# ---------------------------------------------------------------- 发布 e2e

def test_lab_publish_produces_self_contained_package(tmp_path):
    """agent 一个身份走完：提交 → 开实验 → 跑真实数据 → 发布自包含模型包。"""
    ctx, ref = make_ctx(tmp_path, n_steps=600)
    submitted = tf_lab_submit(ctx, "mini_view", MINI_SOURCE)
    assert submitted["ok"]

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
