"""五份契约 Schema 的合法/非法样例校验，及导出文件的新鲜度检查。"""

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from thermoforge_core.contracts import (
    Experiment,
    ModelPackage,
    ObjectModel,
    ResearchGoal,
    TfdcDataset,
)
from thermoforge_core.contracts.export import SCHEMA_TARGETS, render_schema
from thermoforge_core.contracts.model_package import MODEL_STATUSES, can_transition

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACTS_DIR = REPO_ROOT / "contracts"


# ---------- TFOM ----------

def _load_chiller_example() -> dict:
    path = CONTRACTS_DIR / "tfom" / "examples" / "chiller.v1.yaml"
    with open(path, encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def test_tfom_chiller_example_valid():
    model = ObjectModel.model_validate(_load_chiller_example())
    assert model.object_model_id == "chiller.v1"
    roles = {p.role for p in model.properties.values()}
    assert {"state", "control", "target", "derived"} <= roles
    diff = model.properties["supply_return_temp_diff"]
    # 温差单位统一为 K（conventions.md §2.2），不得沿用 data-contract 示例的 Cel
    assert diff.unit == "K"
    assert diff.quantity_kind == "temperature_difference"
    assert diff.expression


def test_tfom_derived_requires_expression():
    doc = _load_chiller_example()
    doc["properties"]["supply_return_temp_diff"]["expression"] = None
    with pytest.raises(ValidationError):
        ObjectModel.model_validate(doc)


def test_tfom_rejects_bad_property_code_and_unit():
    doc = _load_chiller_example()
    doc["properties"]["InputPower"] = doc["properties"].pop("input_power")
    with pytest.raises(ValidationError):
        ObjectModel.model_validate(doc)
    doc = _load_chiller_example()
    doc["properties"]["input_power"]["unit"] = "BTU/h"
    with pytest.raises(ValidationError):
        ObjectModel.model_validate(doc)


# ---------- TFDC ----------

def _tfdc_valid_doc() -> dict:
    return {
        "manifest": {
            "contract": "TFDC",
            "contract_version": "1.0",
            "dataset_id": "DC01_2026_CHILLER",
            "dataset_version": 1,
            "site_id": "DC01",
            "timezone": "Asia/Shanghai",
            "time_resolution": "60s",
        },
        "objects": [
            {"object_id": "CH-01", "object_model_id": "chiller.v1"},
            {"object_id": "SYS-CHW", "object_model_id": "chilled_header.v1"},
        ],
        "variables": [
            {
                "variable_id": "CH-01.evap_chw_supply_temp",
                "object_id": "CH-01",
                "property_code": "evap_chw_supply_temp",
                "unit": "Cel",
                "dtype": "float",
                "role": "state",
                "source_kind": "measured",
            },
            {
                "variable_id": "CH-01.input_power",
                "object_id": "CH-01",
                "property_code": "input_power",
                "unit": "kW",
                "dtype": "float",
                "role": "target",
                "source_kind": "measured",
            },
        ],
        "parameters": [
            {"object_id": "CH-01", "parameter_code": "rated_power", "value": 620, "unit": "kW"}
        ],
        "relations": [
            {"from_object": "CH-01", "relation": "chilled_water_to", "to_object": "SYS-CHW"}
        ],
        "bindings": [
            {"variable_id": "CH-01.input_power", "adapter": "OPCUA", "source_ref": "ns=2;s=CH01.Power"}
        ],
    }


def test_tfdc_valid():
    ds = TfdcDataset.model_validate(_tfdc_valid_doc())
    assert ds.manifest.timezone == "Asia/Shanghai"
    assert len(ds.variables) == 2


@pytest.mark.parametrize(
    "mutate",
    [
        # TFDC-204：非 IANA 时区
        lambda d: d["manifest"].update(timezone="CST"),
        # TFDC-205：time_resolution 格式非法
        lambda d: d["manifest"].update(time_resolution="PT60S"),
        # TFDC-301：variables 引用不存在的对象
        lambda d: d["variables"][0].update(object_id="CH-99", variable_id="CH-99.evap_chw_supply_temp"),
        # TFDC-307：variable_id 重复声明
        lambda d: d["variables"].append(dict(d["variables"][0])),
        # TFDC-304：variable_id 非法
        lambda d: d["variables"][0].update(variable_id="CH-01.Evap_Temp"),
        # variable_id 与 object_id/property_code 不一致
        lambda d: d["variables"][0].update(property_code="evap_chw_return_temp"),
        # TFDC-401：单位未登记
        lambda d: d["variables"][0].update(unit="BTU/h"),
    ],
)
def test_tfdc_invalid(mutate):
    doc = _tfdc_valid_doc()
    mutate(doc)
    with pytest.raises(ValidationError):
        TfdcDataset.model_validate(doc)


# ---------- Research Goal ----------

def _goal_valid_doc() -> dict:
    # research-loop.md §1 示例 + DD-14 的 NMBE 验收
    return {
        "goal_id": "RG-0001",
        "name": "冷水机输入功率模型",
        "object_model": "chiller.v1",
        "purpose": "optimization",
        "target": "input_power",
        "candidate_inputs": [
            "evap_chw_supply_temp",
            "evap_chw_return_temp",
            "evap_chw_flow",
            "cw_supply_temp",
            "cw_return_temp",
            "cooling_capacity",
            "outdoor_wet_bulb_temp",
            "CT-01.supply_temp",  # 跨设备取数：object.property 两层命名
        ],
        "model_types": {"physics": True, "data": True, "hybrid": True},
        "acceptance": {
            "mape_max": 0.05,
            "cvrmse_max": 0.10,
            "nmbe_abs_max": 0.02,
            "physics_violation_rate_max": 0.001,
            "inference_latency_ms_max": 5,
            "extrapolation_required": True,
        },
        "max_experiments": 200,
    }


def test_research_goal_valid():
    goal = ResearchGoal.model_validate(_goal_valid_doc())
    assert goal.goal_id == "RG-0001"
    assert goal.acceptance.nmbe_abs_max == 0.02  # DD-14：NMBE 纳入验收


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(goal_id="RG1"),  # 顺序 ID 必须四位零填充
        lambda d: d.update(target="Input_Power"),
        lambda d: d["candidate_inputs"].append("not a code!"),
        lambda d: d.update(object_model="chiller"),  # 缺 .vN
    ],
)
def test_research_goal_invalid(mutate):
    doc = _goal_valid_doc()
    mutate(doc)
    with pytest.raises(ValidationError):
        ResearchGoal.model_validate(doc)


# ---------- Experiment ----------

def _experiment_valid_doc() -> dict:
    # research-loop.md §5 示例
    return {
        "experiment_id": "EXP-0042",
        "goal_id": "RG-0001",
        "hypothesis_id": "H-0011",
        "dataset_view": "VIEW-0021",
        "model": {"category": "hybrid", "physics": "cooling_balance_v2", "residual": "xgboost"},
        "target": "input_power",
        "validation": {
            "temporal_split": {"train": 0.70, "validate": 0.15, "test": 0.15},
            "equipment_holdout": {"enabled": True},
        },
        "metrics": ["RMSE", "MAE", "MAPE", "CVRMSE", "NMBE"],
        "physics_tests": {"enabled": True},
        "runtime": {"environment_lock": "env-sha256", "random_seed": 20260808},
    }


def test_experiment_valid():
    exp = Experiment.model_validate(_experiment_valid_doc())
    assert exp.validation.temporal_split.validate_ == 0.15
    assert "NMBE" in exp.metrics  # DD-14


@pytest.mark.parametrize(
    "mutate",
    [
        # 切分比例之和必须为 1
        lambda d: d["validation"]["temporal_split"].update(test=0.2),
        # 指标不在登记表
        lambda d: d["metrics"].append("R2"),
        # hybrid 必须声明 physics + residual
        lambda d: d.update(model={"category": "hybrid", "physics": "cooling_balance_v2"}),
        lambda d: d.update(experiment_id="EXP-42"),
    ],
)
def test_experiment_invalid(mutate):
    doc = _experiment_valid_doc()
    mutate(doc)
    with pytest.raises(ValidationError):
        Experiment.model_validate(doc)


# ---------- Model Package ----------

def _package_valid_doc() -> dict:
    # model-package.md §3 签名示例 + §5 状态
    return {
        "model_id": "chiller-power",
        "version": "1.7.0",
        "status": "candidate",
        "signature": {
            "model_id": "chiller-power",
            "version": "1.7.0",
            "object_model": "chiller.v1",
            "object_model_compat": ">=1.0,<2.0",
            "inputs": [
                {"property_code": "evap_chw_supply_temp", "unit": "Cel", "dtype": "float", "required": True},
                {"property_code": "evap_chw_return_temp", "unit": "Cel", "dtype": "float", "required": True},
                {"property_code": "evap_chw_flow", "unit": "m3/h", "dtype": "float", "required": True},
                {"property_code": "cw_supply_temp", "unit": "Cel", "dtype": "float", "required": True},
            ],
            "outputs": [
                {"property_code": "input_power", "unit": "kW", "dtype": "float"}
            ],
        },
        "constraints": {
            "inputs": [
                {
                    "property_code": "evap_chw_supply_temp",
                    "min_value": 2.0,
                    "max_value": 20.0,
                    "out_of_range": "reject",
                }
            ],
            "output_min_value": 0.0,
            "object_model_compat": ">=1.0,<2.0",
        },
        "goal_id": "RG-0001",
        "experiment_id": "EXP-0042",
    }


def test_model_package_valid():
    pkg = ModelPackage.model_validate(_package_valid_doc())
    assert pkg.signature.object_model == "chiller.v1"
    assert pkg.status == "candidate"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(version="1.7"),  # 必须语义化三段版本
        lambda d: d.update(status="released"),  # 状态机外的非法状态
        lambda d: d["signature"].update(object_model="chiller"),  # 缺 .vN
        lambda d: d["signature"]["inputs"][0].update(unit="BTU/h"),
        lambda d: d["constraints"]["inputs"][0].update(out_of_range="ignore"),
    ],
)
def test_model_package_invalid(mutate):
    doc = _package_valid_doc()
    mutate(doc)
    with pytest.raises(ValidationError):
        ModelPackage.model_validate(doc)


def test_status_state_machine():
    # candidate → validated → approved → production → deprecated → retired
    for i in range(len(MODEL_STATUSES) - 1):
        assert can_transition(MODEL_STATUSES[i], MODEL_STATUSES[i + 1])
    assert can_transition("candidate", "production")  # 允许沿链跨步前进
    assert not can_transition("production", "candidate")  # 不得回退
    assert not can_transition("retired", "candidate")
    assert not can_transition("candidate", "candidate")
    assert not can_transition("unknown", "candidate")


# ---------- 导出的 JSON Schema ----------

@pytest.mark.parametrize("name,model", sorted(SCHEMA_TARGETS.items()))
def test_exported_schema_up_to_date(name, model):
    path = CONTRACTS_DIR / name / "schema.json"
    assert path.exists(), f"缺少导出的 Schema: {path}，请运行 scripts/export_schemas.py"
    with open(path, encoding="utf-8", newline="") as fp:
        on_disk = fp.read()
    assert on_disk == render_schema(model), (
        f"{path} 与当前 pydantic 模型不一致，请重新运行 scripts/export_schemas.py"
    )


def test_schema_declares_set_semantics():
    # conventions.md §5.1 规则 6：集合语义字段必须写进 Schema
    research_goal = json.loads(render_schema(ResearchGoal))
    props = research_goal["properties"]
    assert props["candidate_inputs"].get("x-tf-set-semantics") is True
    experiment = json.loads(render_schema(Experiment))
    assert experiment["properties"]["metrics"].get("x-tf-set-semantics") is True
    tfdc = json.loads(render_schema(TfdcDataset))
    for field in ("objects", "variables", "parameters", "relations", "bindings"):
        assert tfdc["properties"][field].get("x-tf-set-semantics") is True
