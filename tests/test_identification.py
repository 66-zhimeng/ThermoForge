"""系统辨识模型族（thermoforge_models.identification）。

不追求拟合精度——那取决于数据。这里锁的是**结构性质**：
参数非负、单调性、只吃入口值、可存取往返。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thermoforge_models.identification import EffectivenessNTU, GordonNgChiller


def _chiller_frame(n: int = 400, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """按 GN 方程正向生成一组自洽样本（已知参数）。"""
    rng = np.random.default_rng(seed)
    qe = rng.uniform(3000.0, 7000.0, n)
    t_ei = rng.uniform(15.0, 19.0, n)
    t_ci = t_ei + rng.uniform(4.0, 11.0, n)
    r_e, r_c, ds = 2e-4, 1.5e-3, 0.5
    t_e = t_ei + 273.15 - qe * r_e
    k = qe / t_e + ds
    qc = (t_ci + 273.15) * k / (1.0 - r_c * k)
    df = pd.DataFrame({"cooling_load": qe, "t_evap_out": t_ei, "t_cond_in": t_ci})
    return df, qc - qe


def test_gordon_ng_recovers_power_on_self_consistent_data():
    df, power = _chiller_frame()
    model = GordonNgChiller().fit(df, power)
    pred = model.predict(df)
    ok = np.isfinite(pred)
    assert ok.mean() > 0.95
    cvrmse = np.sqrt(((power[ok] - pred[ok]) ** 2).mean()) / power[ok].mean()
    assert cvrmse < 0.02, f"自洽数据上应当拟合得很好，实际 CVRMSE={cvrmse:.3%}"


def test_gordon_ng_parameters_are_non_negative():
    """熵产、热阻都是非负物理量；负值说明拟合跑飞了。"""
    df, power = _chiller_frame(seed=1)
    params = GordonNgChiller().fit(df, power).parameters
    assert set(params) == {"r_evaporator", "r_condenser", "delta_s_internal"}
    for key, value in params.items():
        assert value >= 0.0, f"{key} 应非负，实际 {value}"


def test_gordon_ng_monotonicity_holds_by_construction():
    """方程结构保证 ∂P/∂Q>0、∂P/∂T_cond>0、∂P/∂T_evap<0，无需额外约束。"""
    df, power = _chiller_frame(seed=2)
    model = GordonNgChiller().fit(df, power)
    base = pd.DataFrame({"cooling_load": [5000.0], "t_evap_out": [17.0],
                         "t_cond_in": [24.0]})
    p0 = model.predict(base)[0]

    more_load = base.assign(cooling_load=[5500.0])
    hotter_cond = base.assign(t_cond_in=[26.0])
    warmer_evap = base.assign(t_evap_out=[18.0])

    assert model.predict(more_load)[0] > p0        # 负荷升 → 功耗升
    assert model.predict(hotter_cond)[0] > p0      # 冷凝温升 → 功耗升
    assert model.predict(warmer_evap)[0] < p0      # 蒸发温升 → 功耗降


def test_gordon_ng_roundtrip(tmp_path):
    df, power = _chiller_frame(seed=3)
    model = GordonNgChiller().fit(df, power)
    model.save(tmp_path)
    loaded = GordonNgChiller.load(tmp_path)
    np.testing.assert_allclose(model.predict(df), loaded.predict(df),
                               equal_nan=True)


def _hx_frame(n: int = 300, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
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


def test_eps_ntu_recovers_duty():
    df, q = _hx_frame()
    model = EffectivenessNTU().fit(df, q)
    pred = model.predict(df)
    cvrmse = np.sqrt(((q - pred) ** 2).mean()) / q.mean()
    assert cvrmse < 0.02, f"自洽数据上 CVRMSE={cvrmse:.3%}"


def test_eps_ntu_effectiveness_within_physical_bounds():
    """有效度必须落在 (0,1]；超过 1 违反热力学第二定律。"""
    df, q = _hx_frame(seed=4)
    model = EffectivenessNTU().fit(df, q)
    ch, cc = model._capacity_rates(df)
    eps = model._eps(model._ua(df, model.ua0_, model.beta_),
                     np.minimum(ch, cc), np.maximum(ch, cc))
    assert (eps > 0).all() and (eps <= 1.0).all()


def test_eps_ntu_uses_inlet_ports_only():
    """只吃入口值——出口温度不在输入里，结构上杜绝「出口×流量=标签」泄漏。"""
    assert set(EffectivenessNTU().inputs) == {
        "t_hot_in", "t_cold_in", "f_hot", "f_cold"}


def test_eps_ntu_duty_increases_with_driving_temperature_difference():
    df, q = _hx_frame(seed=5)
    model = EffectivenessNTU().fit(df, q)
    base = pd.DataFrame({"t_chw_in": [22.0], "t_cw_in": [15.0],
                         "f_chw": [2700.0], "f_cw": [2100.0],
                         "run_count": [3]})
    hotter = base.assign(t_chw_in=[24.0])
    assert model.predict(hotter)[0] > model.predict(base)[0]


def test_eps_ntu_roundtrip(tmp_path):
    df, q = _hx_frame(seed=6)
    model = EffectivenessNTU().fit(df, q)
    model.save(tmp_path)
    loaded = EffectivenessNTU.load(tmp_path)
    np.testing.assert_allclose(model.predict(df), loaded.predict(df))


def test_runner_can_build_identification_families():
    """两个模型族已接进实验 runner 的分派表。"""
    from thermoforge_research._child import _build_model

    gn = _build_model({"category": "physics", "physics": "gordon_ng",
                       "hyperparameters": {}}, seed=0)
    assert isinstance(gn, GordonNgChiller)

    ntu = _build_model({"category": "physics", "physics": "eps_ntu",
                        "hyperparameters": {}}, seed=0)
    assert isinstance(ntu, EffectivenessNTU)


def test_unknown_physics_family_still_rejected():
    from thermoforge_research._child import _build_model

    with pytest.raises(ValueError):
        _build_model({"category": "physics", "physics": "no_such_model",
                      "hyperparameters": {"rated_capacity_kw": 1000.0}}, seed=0)


def test_parse_inputs_accepts_bare_names_and_both_separators():
    """`inputs` 省略 `=` 时按同名映射；`,` 与 `;` 都作分隔符。

    强制写 `a=a;b=b` 只会制造无谓失败（agent 连续三次栽在这里）。
    """
    from thermoforge_research._child import _parse_inputs

    same = {"cooling_load": "cooling_load", "t_cond_in": "t_cond_in"}
    assert _parse_inputs("cooling_load,t_cond_in") == same
    assert _parse_inputs("cooling_load;t_cond_in") == same
    assert _parse_inputs("cooling_load=cooling_load;t_cond_in=t_cond_in") == same
    assert _parse_inputs("cooling_load=Q_e, t_cond_in=T_ci") == {
        "cooling_load": "Q_e", "t_cond_in": "T_ci"}
    assert _parse_inputs("") is None
    assert _parse_inputs(None) is None


def test_missing_input_error_names_what_the_model_needs():
    """报错要说清楚缺哪列、视图有哪些列、模型需要哪些 —— 否则无法自我纠正。"""
    from thermoforge_models.identification import MissingInputError

    train, power = _chiller_frame(seed=7)
    model = GordonNgChiller().fit(train, power)
    df = pd.DataFrame({"t_evap_out": [17.0], "t_cond_in": [24.0]})   # 少 cooling_load
    with pytest.raises(MissingInputError) as excinfo:
        model.predict(df)
    message = str(excinfo.value)
    assert "cooling_load" in message
    assert "t_evap_out" in message        # 列出了视图现有列


def test_gordon_ng_never_predicts_nan():
    """预测不得含 NaN —— 契约规定计算层产生 NaN 属于缺陷（TFX-905）。

    定义域外（零负荷、lift≤0、缺失输入）要如实外推并可被物理检查逮住，
    而不是把 NaN 丢给指标层。
    """
    train, power = _chiller_frame(seed=8)
    model = GordonNgChiller().fit(train, power)

    nasty = pd.DataFrame({
        "cooling_load": [0.0, -100.0, 5000.0, np.nan, 1e9],
        "t_evap_out": [17.0, 17.0, 25.0, 17.0, 17.0],
        "t_cond_in": [24.0, 24.0, 17.0, 24.0, 24.0],   # 第 3 行 lift 为负
    })
    pred = model.predict(nasty)
    assert np.isfinite(pred).all(), f"预测含非有限值: {pred}"
    assert (pred >= 0).all(), f"功耗不得为负: {pred}"


def test_experiment_get_surfaces_fitted_parameters(tmp_path):
    """已辨识的物理参数必须进信封 —— 否则「参数体检」这一步做不了。

    只看 CVRMSE 会放过两种病症：参数贴死在边界（该项不可辨识）、
    量级失控（模型被推到饱和区）。两者都不会让指标变差。
    """
    import json as _json

    from thermoforge_research.tools import _fitted_parameters

    exp_dir = tmp_path / "EXP-9999"
    (exp_dir / "model").mkdir(parents=True)
    GordonNgChiller().fit(*_chiller_frame(seed=9)).save(exp_dir / "model")

    got = _fitted_parameters(exp_dir)
    assert got is not None
    assert got["format"] == "thermoforge.gordon_ng.v1"
    assert set(got["values"]) == {"r_evaporator", "r_condenser",
                                  "delta_s_internal"}
    assert got["n_train"] > 0

    # 纯数据模型没有物理参数，不应捏造
    (tmp_path / "EXP-8888" / "model").mkdir(parents=True)
    (tmp_path / "EXP-8888" / "model" / "model.json").write_text(
        _json.dumps({"format": "thermoforge.linear_baseline.v1"}),
        encoding="utf-8")
    assert _fitted_parameters(tmp_path / "EXP-8888") is None
    assert _fitted_parameters(tmp_path / "EXP-7777") is None   # 目录不存在
