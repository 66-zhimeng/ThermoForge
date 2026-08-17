"""模型制品加载分派（thermoforge_runtime.artifact）。

缺陷回归：`_LOADERS` 曾只登记 baseline / chiller_physics.v1 /
residual_hybrid 三种格式，导致 cooling_balance_v2 与系统辨识族
（gordon_ng / eps_ntu）实验在发布构建模型包时被「未知模型格式」拒掉；
ResidualHybrid 的 save/load 也只认 params.yaml 路径，辨识族主干的
model.json 被 hybrid meta 覆盖后参数丢失、无法冷加载。

这里锁**分派完备性**：每种 MODEL_FORMAT 都能 save → load_model_artifact
→ predict 往返一致；hybrid 的三种主干（能量平衡 v1、Gordon-Ng、ε-NTU）
全部可重载。
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from thermoforge_models.baseline import LinearBaseline
from thermoforge_models.hybrid import ResidualHybrid
from thermoforge_models.identification import EffectivenessNTU, GordonNgChiller
from thermoforge_models.physics import ChillerPhysicsModel, ChillerPhysicsV2
from thermoforge_runtime.artifact import detect_format, load_model_artifact


def _chiller_frame(n: int = 240) -> tuple[pd.DataFrame, np.ndarray]:
    """确定性冷机帧：Q = m·Cp·ΔT，P = Q/COP（与 phase2_helpers 同关系）。"""
    flow, t_s, t_r, t_cw, power = [], [], [], [], []
    for i in range(n):
        f = 350.0 + 80.0 * math.sin(i / 10.0)
        s = 6.5 + 0.8 * math.sin(i / 25.0)
        r = s + 4.2 + 0.5 * math.sin(i / 8.0)
        c = 25.0 + 4.0 * math.sin(i / 40.0)
        q = f * 998.0 / 3600.0 * 4.186 * (r - s)
        cop = 4.0 + 0.05 * s - 0.06 * c + 1.0 * q / 6000.0
        flow.append(f)
        t_s.append(s)
        t_r.append(r)
        t_cw.append(c)
        power.append(q / cop)
    df = pd.DataFrame({
        "chw_flow": flow,
        "chw_supply_temp": t_s,
        "chw_return_temp": t_r,
        "cw_supply_temp": t_cw,
    })
    return df, np.asarray(power)


def _gn_frame(n: int = 240) -> tuple[pd.DataFrame, np.ndarray]:
    """按 GN 方程正向生成自洽样本（已知参数 r_e/r_c/ΔS）。"""
    rng = np.random.default_rng(0)
    qe = rng.uniform(3000.0, 7000.0, n)
    t_ei = rng.uniform(15.0, 19.0, n)
    t_ci = t_ei + rng.uniform(4.0, 11.0, n)
    r_e, r_c, ds = 2e-4, 1.5e-3, 0.5
    t_e = t_ei + 273.15 - qe * r_e
    k = qe / t_e + ds
    qc = (t_ci + 273.15) * k / (1.0 - r_c * k)
    df = pd.DataFrame({"cooling_load": qe, "t_evap_out": t_ei,
                       "t_cond_in": t_ci})
    return df, qc - qe


def _ntu_frame(n: int = 240) -> tuple[pd.DataFrame, np.ndarray]:
    """按已知 UA0/β 正向生成板换自洽样本。"""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "t_chw_in": rng.uniform(20.0, 24.0, n),
        "t_cw_in": rng.uniform(12.0, 18.0, n),
        "f_chw": rng.uniform(2400.0, 3000.0, n),
        "f_cw": rng.uniform(1800.0, 2400.0, n),
        "run_count": np.full(n, 3),
    })
    truth = EffectivenessNTU()
    truth.ua0_, truth.beta_ = 8.0, 0.6
    return df, truth.predict(df)


def _fitted_baseline():
    df, y = _chiller_frame()
    order = list(df.columns)
    return df, LinearBaseline(method="ridge", alpha=0.5).fit(
        df, y, feature_order=order)


def _fitted_physics_v1():
    df, y = _chiller_frame()
    model = ChillerPhysicsModel(rated_capacity_kw=6000.0,
                                rated_power_kw=1200.0)
    return df, model.fit(df, y)


def _fitted_physics_v2():
    df, y = _chiller_frame()
    model = ChillerPhysicsV2(rated_capacity_kw=6000.0, rated_power_kw=1200.0)
    return df, model.fit(df, y)


def _fitted_gordon_ng():
    df, y = _gn_frame()
    return df, GordonNgChiller().fit(df, y)


def _fitted_eps_ntu():
    df, y = _ntu_frame()
    return df, EffectivenessNTU().fit(df, y)


def _fitted_hybrid(base_factory, frame_factory):
    df, y = frame_factory()
    model = ResidualHybrid(base_factory(), seed=42, nthread=1,
                           xgb_params={"n_estimators": 30, "max_depth": 3})
    model.fit(df, y, feature_order=list(df.columns))
    return df, model


CASES = {
    "linear_baseline": _fitted_baseline,
    "chiller_physics_v1": _fitted_physics_v1,
    "chiller_physics_v2": _fitted_physics_v2,
    "gordon_ng": _fitted_gordon_ng,
    "eps_ntu": _fitted_eps_ntu,
    "hybrid_physics_v1": lambda: _fitted_hybrid(
        lambda: ChillerPhysicsModel(rated_capacity_kw=6000.0,
                                    rated_power_kw=1200.0),
        _chiller_frame),
    "hybrid_gordon_ng": lambda: _fitted_hybrid(GordonNgChiller, _gn_frame),
    "hybrid_eps_ntu": lambda: _fitted_hybrid(EffectivenessNTU, _ntu_frame),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_load_model_artifact_roundtrip(name, tmp_path):
    """每种格式：save → load_model_artifact 分派 → 预测与保存前一致。"""
    df, model = CASES[name]()
    pred = np.asarray(model.predict(df), dtype=np.float64)
    model.save(tmp_path)
    loaded = load_model_artifact(tmp_path)
    after = np.asarray(loaded.predict(df), dtype=np.float64)
    assert np.max(np.abs(after - pred)) < 1e-9, f"{name} 往返预测不一致"


def test_hybrid_identification_base_writes_base_model_json(tmp_path):
    """辨识族主干参数必须落 base_model.json——其 save 只写 model.json，
    会被 hybrid meta 覆盖（缺陷 2 的直接回归）。"""
    df, model = _fitted_hybrid(GordonNgChiller, _gn_frame)
    model.save(tmp_path)
    assert (tmp_path / "base_model.json").exists()
    with open(tmp_path / "model.json", encoding="utf-8") as fp:
        meta = json.load(fp)
    assert meta["base_format"] == "thermoforge.gordon_ng.v1"
    # 旧包兼容：删掉 base_format 字段后，能量平衡族仍走 params.yaml
    df2, model2 = _fitted_hybrid(
        lambda: ChillerPhysicsModel(rated_capacity_kw=6000.0,
                                    rated_power_kw=1200.0),
        _chiller_frame)
    model2.save(tmp_path / "legacy")
    with open(tmp_path / "legacy" / "model.json", encoding="utf-8") as fp:
        legacy_meta = json.load(fp)
    legacy_meta.pop("base_format")
    with open(tmp_path / "legacy" / "model.json", "w", encoding="utf-8") as fp:
        json.dump(legacy_meta, fp)
    loaded = load_model_artifact(tmp_path / "legacy")
    assert np.max(np.abs(loaded.predict(df2)
                         - model2.predict(df2))) < 1e-9


def test_detect_format_rejects_unknown(tmp_path):
    with open(tmp_path / "model.json", "w", encoding="utf-8") as fp:
        json.dump({"format": "thermoforge.bogus.v1"}, fp)
    with pytest.raises(ValueError, match="未知模型格式"):
        detect_format(tmp_path)
