"""系统级派生数据集：DD-12 垂直切片的目标构造。

「运行冷机总功率 = sum(chiller.power where status_run=1)」是跨对象聚合，
单机物模型无法表达；Dataset View 的长表机制也不做跨对象聚合（
`object_id` 是分组键而非特征，implementation-notes §3.3）。

落点选择（三选一，见实验报告 §目标构造）：

1. 在 View 定义中扩展派生目标语法 —— 侵入 Phase 1 契约，放弃。
2. 在脚本里手算后塞回 —— 绕过契约校验与指纹，放弃。
3. **派生数据集**（本模块）：确定性纯函数把源 revision 映射为系统级
   对象 `PLANT`（plant.v1，见 tfom/），产出走 `import_parsed` 标准管线
   （全部 TFDC 校验 + 指纹），作为新数据集 `WX_2025_PLANT` 的不可变
   revision 落 vault，lineage 记录 `derived_from` 与派生规则。
   复用全部既有契约机制，不新增任何接口。

派生规则（DERIVATION_SPEC，写入 lineage 与实验报告）：

- `chw_flow` = chw_A1.f + chw_A2.f（任一缺失则缺失）；
- 温度类两总管逐点一致（data-survey §F3，本模块复核 max|A1−A2|），
  取 A1，A1 缺失回退 A2；
- `run_count` = status_run=true 的冷机台数（null 不计）；
- `any_running` = run_count >= 1（View 过滤用）；
- `total_power` = 运行冷机的 power 之和；run_count=0 时为 0.0；
  运行中冷机 power 缺失则整体缺失（不凑数）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from thermoforge_core.contracts.tfom import ObjectModel
from thermoforge_core.contracts.tfdc import (
    ObjectRecord,
    TfdcDataset,
    TfdcManifest,
    VariableRecord,
)
from thermoforge_data.importer import TfomRegistry, default_registry

PLANT_DATASET_ID = "WX_2025_PLANT"
PLANT_OBJECT_ID = "PLANT"
PLANT_MODEL_ID = "plant.v1"
DERIVATION_VERSION = "derive_plant v1"

CHILLERS = ("chiller_01", "chiller_02", "chiller_03", "chiller_04")

# property_code → (source_kind, 说明)
PLANT_PROPERTIES: tuple[tuple[str, str, str], ...] = (
    ("chw_flow", "measured", "chw_A1.f + chw_A2.f"),
    ("chw_supply_temp", "measured", "chw_A1.t_supply（A1 缺失回退 A2）"),
    ("chw_return_temp", "measured", "chw_A1.t_return（A1 缺失回退 A2）"),
    ("cw_supply_temp", "measured", "cw_A1.t_supply（A1 缺失回退 A2）"),
    ("cw_return_temp", "measured", "cw_A1.t_return（A1 缺失回退 A2）"),
    ("ambient_t", "measured", "environment_parameters.ambient_t"),
    ("ambient_h", "measured", "environment_parameters.ambient_h"),
    ("run_count", "estimated", "status_run=true 的冷机台数"),
    ("any_running", "estimated", "run_count >= 1"),
    ("total_power", "estimated", "sum(chiller.power where status_run=true)"),
)

FEATURE_PROPS = (
    "chw_flow", "chw_supply_temp", "chw_return_temp",
    "cw_supply_temp", "cw_return_temp",
    "ambient_t", "ambient_h", "run_count",
)
TARGET_PROP = "total_power"

_SOURCE = {
    "chw_flow": ("chw_A1.f", "chw_A2.f"),
    "chw_supply_temp": ("chw_A1.t_supply", "chw_A2.t_supply"),
    "chw_return_temp": ("chw_A1.t_return", "chw_A2.t_return"),
    "cw_supply_temp": ("cw_A1.t_supply", "cw_A2.t_supply"),
    "cw_return_temp": ("cw_A1.t_return", "cw_A2.t_return"),
    "ambient_t": ("environment_parameters.ambient_t",),
    "ambient_h": ("environment_parameters.ambient_h",),
}


def demo_tfom_registry() -> TfomRegistry:
    """默认注册表 + 本切片的 plant.v1（显式合并，不改仓库契约目录）。"""
    models: dict[str, ObjectModel] = dict(default_registry()._models)
    tfom_dir = Path(__file__).resolve().parent / "tfom"
    for path in sorted(tfom_dir.glob("*.yaml")):
        with open(path, encoding="utf-8") as fp:
            model = ObjectModel.model_validate(yaml.safe_load(fp))
        models[model.object_model_id] = model
    return TfomRegistry(models)


def derive_plant_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """源数据宽表 → PLANT 帧的确定性纯函数（无随机源）。

    返回 (plant_df, divergence_doc)：divergence 记录两总管一致性复核结果。
    """
    out = pd.DataFrame({"timestamp": df["timestamp"]})
    divergence: dict[str, Any] = {}

    for prop, sources in _SOURCE.items():
        cols = [pd.to_numeric(df[s], errors="coerce").to_numpy(np.float64)
                for s in sources]
        if len(cols) == 1:
            out[prop] = cols[0]
        elif prop == "chw_flow":
            stacked = np.vstack(cols)
            both = np.all(np.isfinite(stacked), axis=0)
            out[prop] = np.where(both, stacked.sum(axis=0), np.nan)
        else:
            a, b = cols
            diff = np.abs(a - b)
            finite = np.isfinite(diff)
            divergence[prop] = {
                "max_abs_diff": float(np.max(diff[finite])) if finite.any() else None,
                "compared_samples": int(finite.sum()),
            }
            out[prop] = np.where(np.isfinite(a), a, b)

    status = df[[f"{c}.status_run" for c in CHILLERS]].eq(True).to_numpy()
    power = df[[f"{c}.power" for c in CHILLERS]].to_numpy(dtype=np.float64)
    run_count = status.sum(axis=1).astype(np.int64)
    running_null = (status & ~np.isfinite(power)).any(axis=1)
    total = np.where(status, np.nan_to_num(power, nan=0.0), 0.0).sum(axis=1)
    total = total.astype(np.float64)
    total[(run_count >= 1) & running_null] = np.nan

    out["run_count"] = run_count
    out["any_running"] = run_count >= 1
    out["total_power"] = total
    return out, divergence


def build_plant_dataset(
    registry: TfomRegistry,
) -> TfdcDataset:
    """plant.v1 的单对象数据集元数据（variables 范围取自 TFOM）。"""
    model = registry.get(PLANT_MODEL_ID)
    if model is None:
        raise ValueError(f"注册表缺少 {PLANT_MODEL_ID}")
    manifest = TfdcManifest(
        contract="TFDC", contract_version="1.0", dataset_id=PLANT_DATASET_ID,
        dataset_version=1, site_id="WX", timezone="Asia/Shanghai",
        time_resolution="900s", source_system="derived",
        description="运行冷机总功率切片的系统级派生数据集（DD-12）",
    )
    objects = [ObjectRecord(object_id=PLANT_OBJECT_ID,
                            object_model_id=PLANT_MODEL_ID,
                            object_name="冷站（系统级派生对象）")]
    notes = {p: note for p, _kind, note in PLANT_PROPERTIES}
    variables = []
    for prop, source_kind, _note in PLANT_PROPERTIES:
        tfom_prop = model.properties[prop]
        variables.append(VariableRecord(
            variable_id=f"{PLANT_OBJECT_ID}.{prop}",
            object_id=PLANT_OBJECT_ID, property_code=prop,
            unit=tfom_prop.unit, dtype=tfom_prop.dtype, role=tfom_prop.role,
            source_kind=source_kind,  # type: ignore[arg-type]
            name_zh=tfom_prop.name_zh or notes[prop],
            nullable=True,
            min_value=tfom_prop.min_value, max_value=tfom_prop.max_value,
            sample_period="900s", aggregation="mean",
        ))
    return TfdcDataset(manifest=manifest, objects=objects, variables=variables)


def derivation_lineage(source_ref: str,
                       divergence: dict[str, Any]) -> dict[str, Any]:
    return {
        "adapter": "examples.chiller_power.derive_plant",
        "derivation_version": DERIVATION_VERSION,
        "derived_from": source_ref,
        "derivation_spec": {p: note for p, _k, note in PLANT_PROPERTIES},
        "header_divergence_check": divergence,
    }
