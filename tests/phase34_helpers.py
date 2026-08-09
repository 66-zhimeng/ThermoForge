"""Phase 3/4 测试共享工具：ToolContext、确定性训练帧、最小模型包构建。

数据与模型均为确定性生成（无随机源），与 phase2_helpers 同一风格。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from thermoforge_models.baseline import LinearBaseline
from thermoforge_research.tools import ToolContext
from thermoforge_runtime.package import boundary_rows, build_model_package

from phase2_helpers import FEATURES, TARGET, build_chiller_vault

UNITS = {
    "evap_chw_flow": "m3/h",
    "evap_chw_supply_temp": "Cel",
    "evap_chw_return_temp": "Cel",
    "cw_supply_temp": "Cel",
    "input_power": "kW",
}

# 训练帧的取值范围（inference 超范围测试以此外推）
RANGES = {
    "evap_chw_flow": (300.0, 500.0),
    "evap_chw_supply_temp": (5.7, 7.3),
    "evap_chw_return_temp": (9.9, 12.1),
    "cw_supply_temp": (21.0, 29.0),
}


def make_ctx(tmp_path, n_steps: int = 600, actor: str = "test"):
    """合成冷水机 vault + ToolContext，返回 (ctx, dataset_ref)。"""
    vault, ref = build_chiller_vault(tmp_path, n_steps=n_steps)
    assert vault is not None
    ctx = ToolContext(
        vault_root=tmp_path / "vault",
        research_root=tmp_path / "research",
        models_root=tmp_path / "models",
        actor=actor,
    )
    return ctx, ref


def training_frame(n: int = 120) -> pd.DataFrame:
    """确定性训练帧：目标为特征的线性组合，ridge 可近精确拟合。"""
    rows: dict[str, list[float]] = {f: [] for f in FEATURES}
    rows[TARGET] = []
    for i in range(n):
        f = 300.0 + 200.0 * (i % 12) / 11.0
        s = 5.7 + 1.6 * ((i // 12) % 10) / 9.0
        r = s + 4.2 + 0.5 * math.sin(i / 5.0)
        c = 21.0 + 8.0 * ((i // 7) % 9) / 8.0
        for name, v in zip(FEATURES, (f, s, r, c)):
            rows[name].append(v)
        rows[TARGET].append(0.9 * f + 12.0 * s + 3.0 * r + 1.5 * c + 7.0)
    return pd.DataFrame(rows)


def signature_doc(
    model_id: str = "chiller-power",
    version: str = "1.0.0",
    *,
    history_required: Mapping[str, Any] | None = None,
    cold_start: str | None = None,
) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "model_id": model_id,
        "version": version,
        "object_model": "chiller.v1",
        "inputs": [
            {"property_code": f, "unit": UNITS[f], "dtype": "float",
             "required": True}
            for f in FEATURES
        ],
        "outputs": [
            {"property_code": TARGET, "unit": UNITS[TARGET], "dtype": "float"},
        ],
    }
    if history_required is not None:
        doc["history_required"] = dict(history_required)
    if cold_start is not None:
        doc["cold_start"] = cold_start
    return doc


def constraints_doc(default_policy: str = "reject") -> dict[str, Any]:
    policies = {
        "evap_chw_flow": "reject",
        "evap_chw_supply_temp": "clamp",
        "evap_chw_return_temp": "passthrough_with_flag",
        "cw_supply_temp": default_policy,
    }
    return {
        "inputs": [
            {
                "property_code": f,
                "min_value": RANGES[f][0],
                "max_value": RANGES[f][1],
                "out_of_range": policies[f],
            }
            for f in FEATURES
        ],
        "output_min_value": 0.0,
    }


def good_metrics(cvrmse: float = 0.01) -> dict[str, Any]:
    return {
        "surfaces": {
            "validate": {"n_samples": 30, "metrics": {
                "RMSE": 1.0, "MAE": 0.8, "MAPE": 0.02,
                "CVRMSE": cvrmse, "NMBE": 0.001}},
            "A": {"n_samples": 30, "metrics": {
                "RMSE": 1.1, "MAE": 0.9, "MAPE": 0.021,
                "CVRMSE": cvrmse, "NMBE": 0.001}},
        },
    }


def build_test_package(
    dest: Path,
    *,
    model_id: str = "chiller-power",
    version: str = "1.0.0",
    history_required: Mapping[str, Any] | None = None,
    cold_start: str | None = None,
    constraints: Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
) -> Path:
    """训练 ridge 基线并构建最小模型包（deterministic）。"""
    df = training_frame()
    artifact_src = dest.parent / f"artifact_src_{version}"
    model = LinearBaseline(method="ridge", alpha=0.5)
    model.fit(df[list(FEATURES)], df[TARGET].to_numpy(),
              feature_order=list(FEATURES))
    model.save(artifact_src)
    return build_model_package(
        dest,
        model_id=model_id,
        version=version,
        signature=signature_doc(
            model_id, version,
            history_required=history_required, cold_start=cold_start),
        artifact_dir=artifact_src,
        metrics=dict(metrics or good_metrics()),
        validation={"physics": {"overall_rate": 0.0}},
        dataset_lineage={"dataset": "TEST02_RESEARCH@rev_0001",
                         "view_hash": "0" * 64},
        research_lineage={"goal_id": "RG-0001", "hypothesis_id": "H-0001",
                          "experiment_id": "EXP-0001"},
        environment={"environment_lock": "test"},
        golden_inputs=boundary_rows(df, list(FEATURES)),
        constraints=dict(constraints or constraints_doc()),
        goal_id="RG-0001",
        experiment_id="EXP-0001",
    )
