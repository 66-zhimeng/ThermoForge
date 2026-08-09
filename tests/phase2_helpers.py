"""Phase 2 测试共享工具：合成冷水机数据集 → vault → View 定义 → Experiment。

数据由确定性的物理关系生成（Q = m·Cp·ΔT、P = Q/COP、
COP = c0 + c1·T_chws + c2·T_cws + c3·PLR），不使用随机源，
保证 Runner 复现性断言的 bit-exact 判定不受数据侧影响。
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any

from thermoforge_core.contracts.experiment import Experiment
from thermoforge_core.contracts.tfdc import (
    ObjectRecord,
    TfdcDataset,
    TfdcManifest,
    VariableRecord,
)
from thermoforge_data.importer import import_parsed
from thermoforge_data.vault import DataVault
from thermoforge_research.runner import current_environment_lock

UTC = timezone.utc
N_STEPS = 192  # 15 min × 192 = 2 天
OBJECTS = ("CH-01", "CH-02")
FEATURES = (
    "evap_chw_flow",
    "evap_chw_supply_temp",
    "evap_chw_return_temp",
    "cw_supply_temp",
)
TARGET = "input_power"
RATED_CAPACITY_KW = 6000.0
RATED_POWER_KW = 1200.0

# physics 模型逻辑输入 → 数据列名（chiller.v1 TFOM 属性）
PHYSICS_INPUTS = (
    "chw_flow=evap_chw_flow;chw_supply_temp=evap_chw_supply_temp;"
    "chw_return_temp=evap_chw_return_temp;cw_supply_temp=cw_supply_temp"
)


def _series(n: int, offset: float) -> dict[str, list[float]]:
    flow, t_s, t_r, t_cw, power = [], [], [], [], []
    for i in range(n):
        f = 350.0 + 80.0 * math.sin(i / 10.0) + offset
        s = 6.5 + 0.8 * math.sin(i / 25.0)
        r = s + 4.2 + 0.5 * math.sin(i / 8.0)
        c = 25.0 + 4.0 * math.sin(i / 40.0)
        q = f * 998.0 / 3600.0 * 4.186 * (r - s)
        cop = 4.0 + 0.05 * s - 0.06 * c + 1.0 * q / RATED_CAPACITY_KW
        flow.append(f)
        t_s.append(s)
        t_r.append(r)
        t_cw.append(c)
        power.append(q / cop)
    return {
        "evap_chw_flow": flow,
        "evap_chw_supply_temp": t_s,
        "evap_chw_return_temp": t_r,
        "cw_supply_temp": t_cw,
        "input_power": power,
    }


def build_chiller_vault(tmp_path, n_steps: int = N_STEPS) -> tuple[DataVault, str]:
    """导入两台冷水机的合成数据并落 vault，返回 (vault, ref)。"""
    manifest = TfdcManifest(
        contract="TFDC", contract_version="1.0", dataset_id="TEST02_RESEARCH",
        dataset_version=1, site_id="T02", timezone="Asia/Shanghai",
        time_resolution="15min",
    )
    objects = [
        ObjectRecord(object_id=o, object_model_id="chiller.v1") for o in OBJECTS
    ]
    variables = []
    units = {
        "evap_chw_flow": "m3/h",
        "evap_chw_supply_temp": "Cel",
        "evap_chw_return_temp": "Cel",
        "cw_supply_temp": "Cel",
        "input_power": "kW",
    }
    for obj in OBJECTS:
        for prop, unit in units.items():
            variables.append(VariableRecord(
                variable_id=f"{obj}.{prop}", object_id=obj, property_code=prop,
                unit=unit, dtype="float",
                role="target" if prop == TARGET else "state",
                source_kind="measured",
            ))
    dataset = TfdcDataset(manifest=manifest, objects=objects, variables=variables)
    base = datetime(2026, 3, 1, 0, 0, tzinfo=UTC)
    ts = [base + timedelta(minutes=15 * i) for i in range(n_steps)]
    columns: dict[str, list[float]] = {}
    for k, obj in enumerate(OBJECTS):
        for prop, values in _series(n_steps, offset=20.0 * k).items():
            columns[f"{obj}.{prop}"] = values
    result = import_parsed(dataset, ts, columns)
    assert result.ok, result.diagnostics
    vault = DataVault(tmp_path / "vault")
    ref = vault.store(result)
    return vault, ref


def view_definition(ref: str) -> dict[str, Any]:
    return {
        "dataset": ref,
        "scope": {"object_model": "chiller.v1"},
        "objects": list(OBJECTS),
        "features": list(FEATURES),
        "target": TARGET,
    }


def make_experiment(
    experiment_id: str,
    *,
    goal_id: str = "RG-0001",
    hypothesis_id: str = "H-0001",
    view_id: str = "VIEW-0001",
    category: str = "data",
    holdout: list[str] | None = None,
    rolling_cv: dict | None = None,
    environment_lock: str | None = None,
) -> Experiment:
    """构造 Experiment 契约对象；environment_lock 默认取当前环境指纹。"""
    if category == "data":
        model: dict[str, Any] = {"category": "data", "estimator": "ridge",
                                 "hyperparameters": {"alpha": 0.5}}
    elif category == "physics":
        model = {
            "category": "physics", "physics": "cooling_balance_v1",
            "hyperparameters": {
                "rated_capacity_kw": RATED_CAPACITY_KW,
                "rated_power_kw": RATED_POWER_KW,
                "inputs": PHYSICS_INPUTS,
            },
        }
    else:
        model = {
            "category": "hybrid", "physics": "cooling_balance_v1",
            "residual": "xgboost",
            "hyperparameters": {
                "rated_capacity_kw": RATED_CAPACITY_KW,
                "rated_power_kw": RATED_POWER_KW,
                "inputs": PHYSICS_INPUTS,
                "n_estimators": 20,
            },
        }
    lock = environment_lock or current_environment_lock()[0]
    validation: dict[str, Any] = {
        "temporal_split": {"train": 0.70, "validate": 0.15, "test": 0.15},
        "equipment_holdout": {
            "enabled": bool(holdout),
            "holdout_objects": holdout or [],
        },
    }
    if rolling_cv is not None:
        validation["rolling_cv"] = rolling_cv
    return Experiment(
        experiment_id=experiment_id,
        goal_id=goal_id,
        hypothesis_id=hypothesis_id,
        dataset_view=view_id,
        model=model,
        target=TARGET,
        validation=validation,
        metrics=["RMSE", "MAE", "MAPE", "CVRMSE", "NMBE"],
        physics_tests={"enabled": True},
        runtime={"environment_lock": lock, "random_seed": 20260808},
    )
