"""模型模板测试（research-loop.md §4 三类路线、implementation-notes §8）。

三个模板在小号合成数据上 fit / predict / 序列化往返一致：
- 线性基线：系数 JSON（不用 pickle）。
- 物理模型：YAML 明文参数 + 参数范围/单位/方程版本；COP 参数可辨识。
- 残差混合：XGBoost 原生 .json；单调性约束在受控扰动下成立。
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest
import yaml

from thermoforge_models.baseline import LinearBaseline
from thermoforge_models.hybrid import ResidualHybrid
from thermoforge_models.physics import (
    ChillerPhysicsModel,
    ChillerPhysicsV2,
    load_physics_model,
)
from thermoforge_models.preprocessing import FeatureScaler
from thermoforge_research.physics_checks import check_monotonicity

N = 300
COP_TRUE = {"c0": 4.0, "c1": 0.05, "c2": -0.06, "c3": 1.0}


def _data(n: int = N, seed: int = 7) -> tuple[pd.DataFrame, np.ndarray]:
    """确定性物理关系 + 固定种子的微小扰动（测试数据本身可复现）。"""
    rng = np.random.default_rng(seed)
    flow = rng.uniform(200, 800, n)
    t_s = rng.uniform(5, 10, n)
    t_r = t_s + rng.uniform(3, 6, n)
    t_cw = rng.uniform(20, 35, n)
    df = pd.DataFrame({
        "chw_flow": flow, "chw_supply_temp": t_s,
        "chw_return_temp": t_r, "cw_supply_temp": t_cw,
    })
    truth = ChillerPhysicsModel(rated_capacity_kw=3000.0)
    truth.cop_coefs = dict(COP_TRUE)
    q = truth.cooling_capacity(df)
    y = q / truth.cop(df, q) + 0.02 * (q / 3000.0) ** 2 * 50  # 弱非线性残差
    return df, y


# ---------------------------------------------------------------- 线性基线


def test_linear_fit_predict_serialize_roundtrip(tmp_path):
    df, y = _data()
    model = LinearBaseline(method="ridge", alpha=0.1)
    model.fit(df, y, feature_order=["chw_flow", "cw_supply_temp"])
    before = model.predict(df)
    model.save(tmp_path)
    loaded = LinearBaseline.load(tmp_path)
    assert np.array_equal(before, loaded.predict(df))
    # 交付格式是系数 JSON（§8.1），不含 pickle
    doc = json.loads((tmp_path / "model.json").read_text(encoding="utf-8"))
    assert set(doc["coefficients"]) == {"chw_flow", "cw_supply_temp"}
    assert isinstance(doc["intercept"], float)
    assert doc["scaler"]["feature_order"] == ["chw_flow", "cw_supply_temp"]


def test_feature_order_is_explicit_not_column_order(tmp_path):
    df, y = _data()
    order = ["cw_supply_temp", "chw_flow"]
    model = LinearBaseline().fit(df, y, feature_order=order)
    shuffled = df[["chw_flow", "cw_supply_temp"]]  # 列序相反
    direct = df[["cw_supply_temp", "chw_flow"]]
    assert np.array_equal(model.predict(shuffled), model.predict(direct))


def test_scaler_fit_on_train_only_and_serialized():
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0], "z": [0.0, 0.0, 0.0]})
    scaler = FeatureScaler.fit(train, ["x", "z"])
    assert scaler.means["x"] == pytest.approx(2.0)
    assert scaler.stds["z"] == 1.0  # 常数特征防零除
    restored = FeatureScaler.from_dict(scaler.to_dict())
    assert restored.to_dict() == scaler.to_dict()


# ---------------------------------------------------------------- 物理模型


def test_physics_recovers_cop_coefficients(tmp_path):
    df, y = _data()
    model = ChillerPhysicsModel(rated_capacity_kw=3000.0,
                                rated_power_kw=600.0).fit(df, y)
    for name, expected in COP_TRUE.items():
        assert model.cop_coefs[name] == pytest.approx(expected, abs=0.15), name
    assert model.identification["method"] == "least_squares"
    assert model.identification["n_samples"] == N

    params_path = model.save(tmp_path)
    doc = yaml.safe_load(params_path.read_text(encoding="utf-8"))
    # 参数范围/单位/方程版本明文记录（research-loop §4）
    assert doc["equation_version"] == "cooling_balance_v1"
    assert doc["parameters"]["rho"]["unit"] == "kg/m3"
    assert "bounds" in doc["parameters"]["cop_coefficients"]["c0"]

    loaded = ChillerPhysicsModel.load(tmp_path)
    assert np.array_equal(model.predict(df), loaded.predict(df))


def test_physics_predict_satisfies_energy_balance():
    df, _ = _data(n=5)
    model = ChillerPhysicsModel(rated_capacity_kw=3000.0)
    model.cop_coefs = dict(COP_TRUE)
    q = model.cooling_capacity(df)
    p = model.predict(df)
    # P = Q / COP ⇒ Q / P = COP
    assert np.allclose(q / p, model.cop(df, q))


def test_physics_coefficient_bounds_clip():
    # 构造 COP 恒为 100 的数据（超出物理范围）→ c0 裁剪到上界 20
    df, _ = _data()
    q = ChillerPhysicsModel(rated_capacity_kw=3000.0).cooling_capacity(df)
    model = ChillerPhysicsModel(rated_capacity_kw=3000.0).fit(df, q / 100.0)
    assert model.cop_coefs["c0"] == 20.0
    assert "c0" in model.identification["clipped_coefficients"]


# ---------------------------------------------------------------- 残差混合


def test_hybrid_roundtrip_and_beats_physics_alone(tmp_path):
    df, y = _data()
    physics = ChillerPhysicsModel(rated_capacity_kw=3000.0,
                                  rated_power_kw=600.0)
    model = ResidualHybrid(physics, seed=42, nthread=1,
                           xgb_params={"n_estimators": 30, "max_depth": 3})
    model.fit(df, y, feature_order=["chw_flow", "chw_supply_temp",
                                    "chw_return_temp", "cw_supply_temp"])
    pred = model.predict(df)
    phys_pred = model.physics_only(df)
    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
    assert rmse(pred, y) < rmse(phys_pred, y)  # 混合优于物理主干
    assert model.residual_share(df) >= 0.0

    model.save(tmp_path)
    assert (tmp_path / "booster.json").exists()  # XGBoost 原生格式
    assert (tmp_path / "params.yaml").exists()  # 物理主干 YAML 参数
    loaded = ResidualHybrid.load(tmp_path)
    after = loaded.predict(df)
    assert np.max(np.abs(after - pred)) < 1e-9  # 往返一致（跨 CPU 容差内）


def test_hybrid_monotone_constraint_holds():
    df, y = _data()
    model = ResidualHybrid(
        ChillerPhysicsModel(rated_capacity_kw=3000.0), seed=1,
        xgb_params={"n_estimators": 30},
        monotone_constraints={"chw_flow": 1},
    )
    model.fit(df, y, feature_order=["chw_flow", "chw_supply_temp",
                                    "chw_return_temp", "cw_supply_temp"])
    base = {f: float(df[f].median()) for f in model.feature_order}
    grid = np.linspace(df["chw_flow"].min(), df["chw_flow"].max(), 25)
    result = check_monotonicity(model.predict, base, "chw_flow",
                                grid.tolist(), direction=1)
    assert result.violations == 0  # 训练时强制约束（§6.3）


def test_hybrid_rejects_unknown_monotone_feature():
    df, y = _data()
    model = ResidualHybrid(
        ChillerPhysicsModel(rated_capacity_kw=3000.0), seed=1,
        monotone_constraints={"no_such_feature": 1},
    )
    with pytest.raises(ValueError):
        model.fit(df, y, feature_order=["chw_flow"])


# ---------------------------------------------------------------- v2：DOE-2 三曲线

_V2_CAPFT = [1.0, 0.01, 0.0, -0.005, 0.0, 0.0]
_V2_EIRFT = [0.9, 0.005, 0.0, 0.01, 0.0001, 0.0]
_V2_EIRFPLR = [0.15, 0.7, 0.15]  # const/plr/plr_sq，Σ=1（DOE-2 归一）
_V2_QR, _V2_PR = 9672.0, 1100.0  # 单台额定


def _biquad(c, a, b):
    return c[0] + c[1] * a + c[2] * a * a + c[3] * b + c[4] * b * b + c[5] * a * b


def _data_v2(n: int = 600, seed: int = 11, constant_units: bool = False):
    """按已知三曲线生成数据；cw_supply_temp 是真冷凝侧，cw_return_temp 是含噪副本。"""
    rng = np.random.default_rng(seed)
    flow = rng.uniform(800, 2400, n)
    t1 = rng.uniform(15, 19, n)
    d_t = rng.uniform(4, 5.5, n)
    t_cw = rng.uniform(20, 36, n)
    t_cw_noisy = 0.9 * t_cw + rng.normal(0, 0.5, n)
    units = (np.ones(n) if constant_units
             else rng.integers(1, 4, n).astype(float))
    q = flow * 998.0 / 3600.0 * 4.186 * d_t
    capft = _biquad(_V2_CAPFT, t1, t_cw)
    eirft = _biquad(_V2_EIRFT, t1, t_cw)
    plr = q / (_V2_QR * units * capft)
    h = _V2_EIRFPLR[0] + _V2_EIRFPLR[1] * plr + _V2_EIRFPLR[2] * plr * plr
    p = _V2_PR * units * plr * eirft * h
    df = pd.DataFrame({
        "chw_flow": flow, "chw_supply_temp": t1, "chw_return_temp": t1 + d_t,
        "cw_supply_temp": t_cw, "cw_return_temp": t_cw_noisy,
        "run_count": units,
    })
    return df, p


def test_v2_identification_recovers_known_model():
    df, p = _data_v2()
    model = ChillerPhysicsV2(_V2_QR, _V2_PR).fit(df, p)
    # 冷凝侧代理：按辨识残差选出真冷凝侧（含噪副本残差更高）
    assert model.condenser_proxy == "cw_supply_temp"
    scores = model.identification["condenser_proxy"]["rel_rmse_by_candidate"]
    assert scores["cw_supply_temp"] < scores["cw_return_temp"]
    # 预测级恢复：无噪合成数据上残差应很小
    pred = model.predict(df)
    rel_rmse = float(np.sqrt(np.mean((pred - p) ** 2)) / np.mean(p))
    assert rel_rmse < 0.05
    # EIRFPLR 系数恢复（PLR 形状只被它承载，可辨识性强）
    for name, expected in zip(("const", "plr", "plr_sq"), _V2_EIRFPLR):
        assert model.eirfplr_coefs[name] == pytest.approx(expected, abs=0.05), name
    # Σc = 1 归一（DOE-2 额定工况约定）
    assert sum(model.eirfplr_coefs.values()) == pytest.approx(1.0)
    # 归一化参数随模型落盘（双二次在原始温度下病态，必须归一化）
    assert model.normalization["t_chws"]["scale"] > 0
    assert model.identification["method"] == "alternating_least_squares"


def test_v2_serialization_roundtrip(tmp_path):
    df, p = _data_v2()
    model = ChillerPhysicsV2(_V2_QR, _V2_PR).fit(df, p)
    before = model.predict(df)
    params_path = model.save(tmp_path)
    doc = yaml.safe_load(params_path.read_text(encoding="utf-8"))
    # YAML 明文：方程版本、曲线系数+范围、保护范围、归一化、代理选择
    assert doc["equation_version"] == "cooling_balance_v2"
    assert doc["format"] == "thermoforge.chiller_physics.v2"
    for curve in ("capft", "eirft", "eirfplr"):
        for term, entry in doc["parameters"]["curves"][curve].items():
            assert "bounds" in entry and "value" in entry, (curve, term)
    assert doc["parameters"]["curve_guards"]["plr"]
    assert doc["parameters"]["input_normalization"]["t_chws"]["center"] > 0
    assert doc["condenser_proxy"] == "cw_supply_temp"

    loaded = load_physics_model(tmp_path)
    assert isinstance(loaded, ChillerPhysicsV2)
    assert np.array_equal(before, loaded.predict(df))  # bit-exact 往返


def test_v2_coefficient_clipping_recorded(tmp_path):
    # y 放大 5 倍而额定不变 → 辨识出的 EIRFT 系统性超界 → 裁剪并记录（沿用 v1 做法）
    df, p = _data_v2()
    model = ChillerPhysicsV2(_V2_QR, _V2_PR).fit(df, p * 5.0)
    assert model.identification["clipped_coefficients"], "超界未被裁剪记录"
    from thermoforge_models.physics import CURVE_BOUNDS_V2
    for curve, coefs in (("capft", model.capft_coefs),
                         ("eirft", model.eirft_coefs),
                         ("eirfplr", model.eirfplr_coefs)):
        for term, value in coefs.items():
            lo, hi = CURVE_BOUNDS_V2[curve][term]
            assert lo <= value <= hi, (curve, term, value)


def test_v1_v2_coexist(tmp_path):
    df, p = _data_v2()
    # v1 接口与行为不变
    v1 = ChillerPhysicsModel(rated_capacity_kw=9672.0,
                             rated_power_kw=4400.0).fit(df, p)
    v1.save(tmp_path / "v1")
    assert isinstance(load_physics_model(tmp_path / "v1"), ChillerPhysicsModel)
    assert load_physics_model(tmp_path / "v1").condenser_col == "cw_supply_temp"
    # v2 无 run_count 输入时退化为系统级额定（units ≡ 1，数据按单台生成）
    df1, p1 = _data_v2(constant_units=True)
    inputs = {"chw_flow": "chw_flow", "chw_supply_temp": "chw_supply_temp",
              "chw_return_temp": "chw_return_temp",
              "cw_supply_temp": "cw_supply_temp",
              "cw_return_temp": "cw_return_temp"}
    v2 = ChillerPhysicsV2(9672.0, 1100.0, inputs=inputs).fit(df1, p1)
    pred = v2.predict(df1)
    rel = float(np.sqrt(np.mean((pred - p1) ** 2)) / np.mean(p1))
    assert rel < 0.08
    # v2 需要 rated_power_kw
    with pytest.raises(ValueError):
        ChillerPhysicsV2(9672.0).fit(df, p)
