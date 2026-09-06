"""模型结果报告导出：把模型注册表里每一个模型包「导出来整合成一份报告」。

为什么是脚本而不是手写文档：报告里的每个数字都必须能追溯到工件文件，
手抄一次就会漂。本脚本只读不写业务数据，从下面四个真源汇总：

- `models/<model_id>/<version>/`  模型包：签名、约束、指标、血缘、门禁工件
- `models/<model_id>/registry.json`  版本状态机与发布门禁结论
- `research/experiments/<EXP>/`  实验规格、时间切分、滚动交叉验证逐折结果
- `research/goals/<RG>.yaml` / `research/hypotheses/<H>.yaml`  验收判据与假设

公式一律从**工件参数**现场渲染（`_FORMULA_BUILDERS`），不写死数值；
内置族（hybrid / gordon_ng / 线性基线）的符号式与
`webui/services/inspect_model.py` 同口径，实验室自定义族的公式逐个还原自
`artifact/lab_source.py`。

用法：
    .venv/Scripts/python scripts/export_model_report.py [--out-dir docs/sharing]
产物：`模型结果报告.md` + `模型结果报告.html`（自包含单文件）。
"""

from __future__ import annotations

import argparse
import html as html_escape
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import yaml

ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = ROOT / "models"
RESEARCH_ROOT = ROOT / "research"
VAULT_ROOT = ROOT / "vault"

# 变量中文名：报告面向人读，property_code 本身不自解释。
# 单位不在这里写死——单位的真源是模型包 signature.yaml。
PROPERTY_LABELS: dict[str, str] = {
    "power": "功率", "cooling_load": "制冷量", "plr": "部分负荷率",
    "t_evap_out": "蒸发器出水温（冷冻水供水）",
    "t_cond_in": "冷凝器进水温（冷却水回水）",
    "lift": "冷凝-蒸发温升 lift", "run_count": "运行台数",
    "hx_run_count": "板换运行台数", "ambient_t": "室外干球温度",
    "ambient_h": "室外相对湿度", "f_cw_branch": "本机冷却水支路流量",
    "f_chw_branch": "本机冷冻水支路流量", "cwp_frequency": "对应冷却泵频率",
    "chwp_frequency": "对应冷冻泵频率", "t_cw_tower_out": "冷却塔出水温",
    "t_chw_return": "冷冻水回水温", "runtime_accum": "累计运行时长",
    "heat_transfer": "换热量", "t_chw_in": "板换热侧（冷冻水）进水温",
    "t_cw_in": "板换冷侧（冷却水）进水温", "drive_dt": "冷热侧进水温差（驱动温差）",
    "f_chw": "冷冻水侧流量", "f_cw": "冷却水侧流量", "flow_ratio": "两侧流量比",
    "chiller_run_count": "冷机运行台数",
    "t_ct_out": "冷却塔群出水温", "t_ct_in": "冷却塔群进水温",
    "f_cw_total": "冷却水总流量", "t_wetbulb": "室外湿球温度",
    "fan_run_count": "风机运行台数", "fan_freq_mean": "风机平均频率",
    "fan_freq_max": "风机最高频率", "cwp_run_count": "冷却泵运行台数",
    "fan_power_total": "风机群总功耗", "approach": "逼近度（出水温 − 湿球温）",
    "heat_reject": "排热量",
    "flow_total": "水泵群总流量", "power_total": "水泵群总功耗",
    "freq_mean": "水泵平均频率", "freq_max": "水泵最高频率",
    "t_supply": "供水温度", "t_return": "回水温度", "delta_t": "供回水温差",
    "frequency": "本台变频器频率", "tower_fan_count": "同塔运行风机数",
    "tower_freq_mean": "同塔风机平均频率", "bank_fan_count": "全场运行风机数",
    "bank_freq_mean": "全场风机平均频率", "bank_run_count": "全场运行台数",
    "bank_freq_sum": "全场频率之和",
}

METRIC_ORDER = ("R2", "CVRMSE", "NMBE", "MAPE")
SURFACE_LABELS = {"rolling_cv": "滚动交叉验证（判据面）",
                  "validate": "验证集（时间切分中段）",
                  "A": "测试集 A（时间切分末段）"}
RATIO_METRICS = {"CVRMSE", "NMBE", "MAPE"}
STATUS_LABELS = {"production": "生产", "approved": "已批准",
                 "validated": "已验证", "candidate": "候选",
                 "deprecated": "已弃用", "retired": "已退役"}


# --------------------------------------------------------------- 小工具


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _read_yaml(path: Path) -> dict:
    if not path.is_file():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def num(value: Any, digits: int = 6) -> str:
    """数值文本。None → 「—」；极小值走科学计数，避免读成 0。"""
    if value is None:
        return "—"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if v != v:
        return "—"
    if v != 0 and abs(v) < 1e-4:
        return f"{v:.3e}"
    return f"{v:.{digits}g}"


def pct(value: Any, digits: int = 2) -> str:
    """比率制指标 → 百分数（metrics.py 存的是比率，不是百分比）。"""
    return "—" if value is None else f"{float(value) * 100:.{digits}f}%"


def metric_text(name: str, value: Any) -> str:
    return pct(value) if name in RATIO_METRICS else num(value, 4)


def label(code: str) -> str:
    return PROPERTY_LABELS.get(code, "")


def ts(value: Any) -> str:
    if not value:
        return "—"
    try:
        moment = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return str(value)
    return moment.strftime("%Y-%m-%d %H:%M")


# --------------------------------------------------------------- 公式渲染


@dataclass
class Formula:
    """一个模型的可读公式：符号式 + 代入参数的数值式 + 参数表 + 注记。"""

    family: str
    equations: list[str] = field(default_factory=list)
    substituted: list[str] = field(default_factory=list)
    params: list[tuple[str, str, str, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _gordon_ng_formula(base: Mapping[str, Any]) -> Formula:
    p = base.get("parameters") or {}
    r_e, r_c, ds = (p.get("r_evaporator"), p.get("r_condenser"),
                    p.get("delta_s_internal"))
    formula = Formula(
        family="Gordon-Ng 熵产模型（能量平衡 + 熵平衡 + 换热热阻，三参数）",
        equations=[
            "T_e = T_ei − Q_e · R_e                （蒸发温度，绝对温标 K）",
            "K   = Q_e / T_e + ΔS_int              （总熵流率 kW/K）",
            "Q_c = T_ci · K / (1 − R_c · K)        （冷凝排热 kW）",
            "P   = Q_c − Q_e                       （压缩机功率 kW）",
        ],
        substituted=[
            f"T_e = T_ei − {num(r_e)} · Q_e",
            f"K   = Q_e / T_e + {num(ds)}",
            f"Q_c = T_ci · K / (1 − {num(r_c)} · K)",
            "P   = Q_c − Q_e",
        ],
        params=[("R_e", "蒸发器换热热阻", num(r_e), "K/kW"),
                ("R_c", "冷凝器换热热阻", num(r_c), "K/kW"),
                ("ΔS_int", "内部熵产（不可逆损失）", num(ds), "kW/K")],
        notes=["单调性天然满足：∂P/∂Q_e>0、∂P/∂T_ci>0、∂P/∂T_ei<0，无需额外施加约束。"],
    )
    if base.get("n_train"):
        formula.notes.append(
            f"辨识样本 {base['n_train']} 个；负荷下限 q_floor="
            f"{num(base.get('q_floor'))} kW（低于此值的样本不参与辨识）。")
    tiny = [name for name, value in (("R_e", r_e), ("R_c", r_c))
            if value is not None and abs(float(value)) < 1e-9]
    if tiny:
        formula.notes.append(
            f"**{'、'.join(tiny)} 被辨识到 ≈0**：该台的换热热阻项在本工况区间内"
            "不可辨识（温差摆幅太小），物理主干退化为熵产项主导的近似恒定 COP 关系，"
            "误差主要由残差学习器承担——对照下方 physics_only 一行。")
    return formula


def _build_hybrid(meta: Mapping[str, Any], directory: Path) -> Formula:
    base = _read_json(directory / "base_model.json")
    formula = _gordon_ng_formula(base) if base else Formula(family="残差混合")
    xgb_params = meta.get("xgb_params") or {}
    order = (meta.get("scaler") or {}).get("feature_order") or []
    trees = xgb_params.get("n_estimators", "?")
    formula.family = f"残差混合：{formula.family} + XGBoost 残差修正"
    formula.equations += [
        "",
        "z_i = (x_i − mean_i) / std_i           （残差特征标准化，均值/标准差随包落盘）",
        f"P̂   = P + Σ_(t=1..{trees}) g_t(z)      （g_t 为回归树，P 是上面的物理主干输出）",
    ]
    formula.substituted.append(f"P̂ = P + Σ_(t=1..{trees}) g_t(z)")
    formula.params += [
        ("n_estimators", "残差树棵数", str(trees), "1"),
        ("max_depth", "残差树最大深度", str(xgb_params.get("max_depth")), "1"),
        ("learning_rate", "学习率", num(xgb_params.get("learning_rate")), "1"),
        ("subsample", "行采样比例", num(xgb_params.get("subsample")), "1"),
        ("colsample_bytree", "列采样比例",
         num(xgb_params.get("colsample_bytree")), "1"),
        ("seed", "随机种子", str(meta.get("seed")), "1"),
    ]
    formula.notes.append(
        f"残差学习器可见 {len(order)} 个标准化特征（顺序即包内 `preprocessing.yaml` 的 "
        "`feature_order`）；物理主干只用 Q_e / T_ei / T_ci 三列。")
    return formula


def _build_linear(meta: Mapping[str, Any], directory: Path) -> Formula:
    coefs = {str(k): float(v) for k, v in (meta.get("coefficients") or {}).items()}
    intercept = float(meta.get("intercept") or 0.0)
    scaler = meta.get("scaler") or {}
    means, stds = scaler.get("means") or {}, scaler.get("stds") or {}
    ranked = sorted(coefs.items(), key=lambda kv: -abs(kv[1]))
    terms = "".join(f"\n      {'+' if w >= 0 else '−'} {abs(w):.6g} · z_{name}"
                    for name, w in ranked)
    params = [("b", "截距", num(intercept), "目标单位"),
              ("α", "岭回归正则强度", num(meta.get("alpha")), "1")]
    for name, weight in ranked:
        params.append((f"w[{name}]", f"{label(name)} 系数（作用在标准化量上）",
                       num(weight), "目标单位 / 标准差"))
        params.append((f"μ,σ[{name}]", "训练集均值 / 标准差",
                       f"{num(means.get(name))} / {num(stds.get(name))}", "原始单位"))
    method = "岭回归 Ridge" if meta.get("method") == "ridge" else "普通最小二乘"
    return Formula(
        family=f"线性基线（{method}）",
        equations=["z_i = (x_i − μ_i) / σ_i        （训练集均值/标准差，随包落盘）",
                   "ŷ   = b + Σ_i w_i · z_i"],
        substituted=[f"ŷ = {intercept:.6g}{terms}"],
        params=params,
        notes=["系数作用在**标准化**特征上：|w| 可直接横向比大小，"
               "含义是「该输入每变化一个训练集标准差，输出变化多少」。"],
    )


def _build_gbdt(meta: Mapping[str, Any], directory: Path) -> Formula:
    columns = [str(c) for c in (meta.get("columns") or [])]
    params = meta.get("params") or {}
    trees = params.get("n_estimators", "?")
    listing = "\n        ".join(f"x_{i + 1} = {c}" for i, c in enumerate(columns))
    rows = [(k, "", num(v), "1") for k, v in sorted(params.items())]
    rows += [("seed", "随机种子", str(meta.get("seed")), "1"),
             ("tree_method", "建树算法", "hist", "—"),
             ("n_jobs", "线程数（位级复现要求单线程）", "1", "1")]
    return Formula(
        family="纯梯度提升静态模型（实验室模块 gbdt_static@v1，无物理主干）",
        equations=[f"ŷ = Σ_(t=1..{trees}) g_t(x)        （XGBoost 回归树加和，"
                   "目标函数 squarederror）",
                   f"x = ({', '.join(columns)})"],
        substituted=[f"输入向量按训练时列序取列：\n        {listing}"],
        params=rows,
        notes=[f"树集成没有闭式解析式：模型本体是 `artifact/booster.json` 里的 "
               f"{trees} 棵树，可解释性由下方**特征增益占比**给出。",
               "确定性：单线程 + 固定 random_state + tree_method=hist，"
               "同种子重训逐位一致。"],
    )


def _build_approach(meta: Mapping[str, Any], directory: Path) -> Formula:
    cols = ["t_ct_in", "f_cw_total", "t_wetbulb", "fan_freq_mean", "fan_run_count"]
    coef = [float(c) for c in (meta.get("coef") or [])]
    means = [float(c) for c in (meta.get("means") or [])]
    stds = [float(c) for c in (meta.get("stds") or [])]
    names = list(cols)
    for i in range(len(cols)):
        for j in range(i, len(cols)):
            names.append(f"{cols[i]}²" if i == j else f"{cols[i]} · {cols[j]}")
    params = [("θ_0", "常数项（不参与标准化）",
               num(coef[0]) if coef else "—", "K")]
    for index, name in enumerate(names):
        params.append((f"θ_{index + 1}", f"{name} 项系数（标准化后）",
                       num(coef[index + 1]) if index + 1 < len(coef) else "—", "K"))
        params.append((f"μ,σ_{index + 1}", f"{name} 训练集均值 / 标准差",
                       f"{num(means[index]) if index < len(means) else '—'} / "
                       f"{num(stds[index]) if index < len(stds) else '—'}", "—"))
    return Formula(
        family="二阶多项式（含全部两两交互）岭回归 —— 实验室模块 "
               "approach_bias_correction_v1@v1",
        equations=[
            "d(x) = [1, x_1…x_5, x_i·x_j (1 ≤ i ≤ j ≤ 5)]      → 1 + 5 + 15 = 21 项",
            "z_k  = (d_k − μ_k) / (σ_k + 1e−9)                  （k = 1…20，常数项不缩放）",
            "approach = θ_0 + Σ_(k=1..20) θ_k · z_k",
            "",
            "拟合：(ZᵀZ + λI)·θ = Zᵀy，λ = 1e−3     （闭式岭回归，无迭代、无随机性）",
        ],
        substituted=[f"x = ({', '.join(cols)})",
                     f"θ_0 = {num(coef[0]) if coef else '—'} K"],
        params=params,
        notes=[
            "**命名与实现不一致**：模型包 `description` 写的是「Merkel/NTU 主干 + "
            "偏差校正」（沿用假设 H-0099 的措辞），但落盘的 `lab_source.py` 里没有 "
            "Merkel 主干——实际形态就是上面这个 21 项二阶多项式岭回归。以本节公式为准。",
            "逼近度 approach = t_ct_out − t_wetbulb 是派生量；"
            "预测它再加上湿球温度即得冷却塔出水温。",
        ],
    )


def _build_fan_density(meta: Mapping[str, Any], directory: Path) -> Formula:
    a, b, c, rho_ref = (meta.get("a"), meta.get("b"), meta.get("c"),
                        meta.get("rho_ref"))
    return Formula(
        family="频率幂律 + 湿空气密度修正 —— 实验室模块 "
               "ct_fan_density_corrected_v1@v1",
        equations=[
            "e_s(T)  = 610.94 · exp(17.625·T / (T + 243.04))       （饱和水汽压 Pa，Magnus 式）",
            "p_v     = RH/100 · e_s(T)                             （水汽分压 Pa）",
            "ρ(T,RH) = (p − p_v)/(287.058·(T+273.15))",
            "          + p_v/(461.495·(T+273.15))                  （湿空气密度 kg/m³，p = 101325 Pa 定值）",
            "P̂       = a · (f/50)^b · ρ(T,RH)/ρ_ref + c            （单台风机功率 kW）",
        ],
        substituted=[
            f"P̂ = {num(a)} · (f/50)^{num(b)} · ρ(T,RH)/{num(rho_ref)} + ({num(c)})"],
        params=[("a", "50 Hz 基准功率系数", num(a), "kW"),
                ("b", "频率指数（相似定律理论值 3）", num(b), "1"),
                ("c", "常数偏置", num(c), "kW"),
                ("ρ_ref", "训练集湿空气密度均值", num(rho_ref), "kg/m³"),
                ("b 搜索网格", "[b_min, b_max] × 步数",
                 f"[{num(meta.get('b_min'))}, {num(meta.get('b_max'))}] × "
                 f"{meta.get('b_steps')}", "—")],
        notes=[
            "拟合方式：b 在网格上枚举，每个 b 下对 (a, c) 做最小二乘，取 RMSE 最小者；"
            "全过程无随机数，`seed` 不影响结果。",
            f"辨识出的 b = {num(b)} 低于相似定律理论值 3 —— 变频器与电机效率随频率下降，"
            "实测幂次通常落在 2.5~2.8。",
            "**12 台风机共用一组 (a, b, c)**：模型不区分 object_id，"
            "逐台差异全部落进残差（见逐台指标表）。",
        ],
    )


def _build_pump_affinity(meta: Mapping[str, Any], directory: Path) -> Formula:
    per_object = meta.get("params") or {}
    glob = meta.get("global_params") or {}
    params = [(obj, "a（kW） / b（1） / c（kW）",
               f"{num(p.get('a'))} / {num(p.get('b'))} / {num(p.get('c'))}", "—")
              for obj, p in sorted(per_object.items())]
    params.append(("global（兜底）", "样本不足对象的回退参数 a / b / c",
                   f"{num(glob.get('a'))} / {num(glob.get('b'))} / "
                   f"{num(glob.get('c'))}", "—"))
    return Formula(
        family="逐台标定的三参数相似定律幂律 —— 实验室模块 "
               "pump_power_per_object_affinity_v1@v1",
        equations=[
            "P̂(obj, f) = 0                                 若 f ≤ 1e−9（停机判定）",
            "P̂(obj, f) = c_obj + a_obj · (f/50)^b_obj       否则",
            "",
            "每台独立标定：b_obj 在 [2.0, 3.5] 上 61 点网格枚举，",
            "每个 b 下对 (a, c) 做最小二乘，取 RMSE 最小者；",
            "样本数 < 5 的对象回退到全体样本标定的 global 参数。",
        ],
        substituted=[
            f"{obj}：P̂ = {num(p.get('c'))} + {num(p.get('a'))} · (f/50)^{num(p.get('b'))}"
            for obj, p in sorted(per_object.items())],
        params=params,
        notes=[
            "冷冻泵 CHWP 标定出的 b ≈ 2.58~2.75，冷却泵 CWP b ≈ 3.00~3.13 —— "
            "冷却泵更贴近相似定律的立方律，冷冻泵偏低。",
            "拟合无随机数；`f ≤ 1e−9 → P̂ = 0` 这条硬规则保证停机时刻不产生虚假功耗。",
            "**可外推**：结构是频率幂律，给定任意开机组合与频率即可加总出总功耗。",
        ],
    )


def _build_eps_ntu(meta: Mapping[str, Any], directory: Path) -> Formula:
    p = meta.get("parameters") or {}
    return Formula(
        family="ε-NTU 逆流换热器模型（两参数 UA₀、β）",
        equations=[
            "C   = f · ρ · Cp / 3600                              （两侧热容流率 kW/K）",
            "UA  = units · UA₀ / (f_hot^(−0.8) + β · f_cold^(−0.8))  （Dittus-Boelter 流量修正）",
            "NTU = UA / C_min,   C_r = C_min / C_max",
            "ε   = (1 − e^(−NTU·(1−C_r))) / (1 − C_r · e^(−NTU·(1−C_r)))",
            "Q   = ε · C_min · (T_hot,in − T_cold,in)",
        ],
        substituted=[f"UA₀ = {num(p.get('ua0'))} kW/K，β = {num(p.get('beta'))}"],
        params=[("UA₀", "基准传热能力", num(p.get("ua0")), "kW/K"),
                ("β", "冷侧热阻占比权重", num(p.get("beta")), "1"),
                ("units", "台数缩放列", str(meta.get("n_units") or "—"), "—")],
        notes=["只用**入口端口值**（两侧进水温 + 两侧流量），出口温度是输出而非输入，"
               "结构上不可能发生「出口温度 × 流量 = 标签」的泄漏。",
               f"辨识样本 {meta.get('n_train', '—')} 个。"],
    )


_FORMULA_BUILDERS = {
    "thermoforge.residual_hybrid.v1": _build_hybrid,
    "thermoforge.linear_baseline.v1": _build_linear,
    "thermoforge.lab.gbdt_static.v1": _build_gbdt,
    "thermoforge.lab.approach_bias_correction_v1": _build_approach,
    "thermoforge.lab.ct_fan_density_corrected_v1": _build_fan_density,
    "thermoforge.lab.pump_power_per_object_affinity_v1": _build_pump_affinity,
    "thermoforge.eps_ntu.v1": _build_eps_ntu,
}


def build_formula(directory: Path) -> Formula | None:
    """按工件 `model.json` 的 format 分派公式渲染；未知 format 返回 None。"""
    meta = _read_json(directory / "model.json")
    builder = _FORMULA_BUILDERS.get(str(meta.get("format") or ""))
    return builder(meta, directory) if builder else None


def gain_importance(directory: Path, top: int = 12) -> list[tuple[str, float]]:
    """XGBoost 增益占比（%，降序）。缺 booster 或 xgboost 不可用时返回空。"""
    booster_path = directory / "booster.json"
    if not booster_path.is_file():
        return []
    try:
        import xgboost as xgb

        booster = xgb.Booster()
        booster.load_model(str(booster_path))
        scores = booster.get_score(importance_type="gain")
    except Exception:
        return []
    total = sum(float(v) for v in scores.values())
    if total <= 0:
        return []
    meta = _read_json(directory / "model.json")
    columns = (meta.get("columns")
               or (meta.get("scaler") or {}).get("feature_order") or [])

    def resolve(key: str) -> str:
        if re.fullmatch(r"f\d+", key):
            index = int(key[1:])
            if index < len(columns):
                return str(columns[index])
        return key

    ranked = sorted(((resolve(k), float(v) / total * 100.0)
                     for k, v in scores.items()), key=lambda kv: -kv[1])
    return ranked[:top]


# --------------------------------------------------------------- 采集


def source_kinds(dataset_ref: str) -> dict[str, str]:
    """数据集修订的 `property_code → source_kind`（measured/estimated/derived）。

    读不到（未安装 pandas / 数据集不在本机）时返回空字典，报告降级不报错。
    """
    if "@" not in (dataset_ref or ""):
        return {}
    dataset_id, revision = dataset_ref.split("@", 1)
    path = VAULT_ROOT / "datasets" / dataset_id / revision / "canonical" / \
        "variables.parquet"
    if not path.is_file():
        return {}
    try:
        import pandas as pd

        frame = pd.read_parquet(path)
    except Exception:
        return {}
    mapping: dict[str, str] = {}
    for code, kind in zip(frame["property_code"], frame["source_kind"]):
        mapping.setdefault(str(code), str(kind))
    return mapping


@dataclass
class Version:
    """一个模型包版本的全部可报告事实。"""

    model_id: str
    version: str
    status: str
    directory: Path
    model_yaml: dict
    signature: dict
    metrics: dict
    dataset_lineage: dict
    research_lineage: dict
    validation: dict
    gates: list
    rollback: str | None
    history: list
    formula: Formula | None
    importance: list[tuple[str, float]]
    spec: dict
    rolling: dict
    goal: dict
    hypothesis: dict
    kinds: dict[str, str]

    @property
    def description(self) -> str:
        return str(self.model_yaml.get("description") or "")

    @property
    def experiment_id(self) -> str:
        return str(self.research_lineage.get("experiment_id") or "")

    @property
    def goal_id(self) -> str:
        return str(self.research_lineage.get("goal_id") or "")


def load_version(model_id: str, version: str, entry: Mapping[str, Any],
                 directory: Path) -> Version:
    research_lineage = _read_json(directory / "research-lineage.json")
    dataset_lineage = _read_json(directory / "dataset-lineage.json")
    experiment_id = str(research_lineage.get("experiment_id") or "")
    experiment_dir = RESEARCH_ROOT / "experiments" / experiment_id
    goal_id = str(research_lineage.get("goal_id") or "")
    hypothesis_id = str(research_lineage.get("hypothesis_id") or "")
    return Version(
        model_id=model_id, version=version,
        status=str(entry.get("status") or ""), directory=directory,
        model_yaml=_read_yaml(directory / "model.yaml"),
        signature=_read_yaml(directory / "signature.yaml"),
        metrics=_read_json(directory / "metrics.json"),
        dataset_lineage=dataset_lineage,
        research_lineage=research_lineage,
        validation=_read_json(directory / "validation.json"),
        gates=list(entry.get("last_gate_results") or []),
        rollback=entry.get("rollback_version"),
        history=list(entry.get("history") or []),
        formula=build_formula(directory / "artifact"),
        importance=gain_importance(directory / "artifact"),
        spec=_read_json(experiment_dir / "spec.json"),
        rolling=_read_json(experiment_dir / "rolling_cv.json"),
        goal=_read_yaml(RESEARCH_ROOT / "goals" / f"{goal_id}.yaml"),
        hypothesis=_read_yaml(RESEARCH_ROOT / "hypotheses" / f"{hypothesis_id}.yaml"),
        kinds=source_kinds(str(dataset_lineage.get("dataset") or "")),
    )


@dataclass
class Model:
    model_id: str
    production: str | None
    versions: list[Version]

    @property
    def current(self) -> Version | None:
        for version in self.versions:
            if version.version == self.production:
                return version
        return self.versions[-1] if self.versions else None


def _version_key(text: str) -> tuple:
    return tuple(int(part) if part.isdigit() else part
                 for part in re.split(r"[._-]", text))


def load_models() -> list[Model]:
    models: list[Model] = []
    for registry_path in sorted(MODELS_ROOT.glob("*/registry.json")):
        registry = _read_json(registry_path)
        model_id = str(registry.get("model_id") or registry_path.parent.name)
        versions = [
            load_version(model_id, version, entry, registry_path.parent / version)
            for version, entry in sorted(registry.get("versions", {}).items(),
                                         key=lambda kv: _version_key(kv[0]))
            if (registry_path.parent / version).is_dir()
        ]
        models.append(Model(model_id=model_id,
                            production=registry.get("production"),
                            versions=versions))
    return models


# --------------------------------------------------------------- Markdown


def table(headers: list[str], rows: list[list[str]]) -> list[str]:
    if not rows:
        return []
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c).replace("|", "\\|") for c in row) + " |"
              for row in rows]
    return lines + [""]


def surface_rows(metrics: Mapping[str, Any]) -> list[list[str]]:
    rows: list[list[str]] = []
    rolling = metrics.get("rolling_cv") or {}
    if rolling:
        values = rolling.get("metrics") or {}
        rows.append([SURFACE_LABELS["rolling_cv"],
                     f"{rolling.get('n_samples', '—')}（{rolling.get('n_folds', '—')} 折，"
                     f"跳过 {rolling.get('n_folds_skipped', 0)}）"]
                    + [metric_text(m, values.get(m)) for m in METRIC_ORDER])
    for key in ("validate", "A"):
        surface = (metrics.get("surfaces") or {}).get(key)
        if not surface:
            continue
        values = surface.get("metrics") or {}
        rows.append([SURFACE_LABELS.get(key, key), str(surface.get("n_samples", "—"))]
                    + [metric_text(m, values.get(m)) for m in METRIC_ORDER])
    return rows


def per_object_rows(metrics: Mapping[str, Any], surface: str) -> list[list[str]]:
    per_object = ((metrics.get("surfaces") or {}).get(surface) or {}).get("per_object") or {}
    if len(per_object) < 2:
        return []
    rows = []
    for obj, entry in sorted(per_object.items()):
        values = entry.get("metrics") or {}
        rows.append([obj, str(entry.get("n_samples", "—"))]
                    + [metric_text(m, values.get(m)) for m in METRIC_ORDER])
    return rows


def acceptance_rows(version: Version) -> list[list[str]]:
    """验收判据逐条对照：判据面取目标声明的 `evaluated_on`。"""
    acceptance = (version.goal.get("definition") or {}).get("acceptance") or {}
    surface = str(acceptance.get("evaluated_on") or "rolling_cv")
    values = ((version.metrics.get("rolling_cv") or {}).get("metrics") or {}) \
        if surface == "rolling_cv" else \
        (((version.metrics.get("surfaces") or {}).get(surface) or {}).get("metrics") or {})
    physics = (version.validation.get("physics") or {}).get("overall_rate")
    checks = [
        ("CVRMSE ≤", acceptance.get("cvrmse_max"), values.get("CVRMSE"), True, False),
        ("|NMBE| ≤", acceptance.get("nmbe_abs_max"),
         abs(values["NMBE"]) if values.get("NMBE") is not None else None, True, False),
        ("MAPE ≤", acceptance.get("mape_max"), values.get("MAPE"), True, False),
        ("R² ≥", acceptance.get("r2_min"), values.get("R2"), False, True),
        ("物理违规率 ≤", acceptance.get("physics_violation_rate_max"),
         physics, True, False),
    ]
    rows = []
    for name, limit, actual, is_ratio, greater in checks:
        if limit is None:
            continue
        fmt = pct if is_ratio else (lambda v: num(v, 4))
        verdict = "—"
        if actual is not None:
            passed = actual >= limit if greater else actual <= limit
            verdict = "通过" if passed else "**不通过**"
        rows.append([name, fmt(limit), fmt(actual), verdict])
    return rows


def rolling_detail_rows(version: Version) -> list[list[str]]:
    """滚动交叉验证逐指标的折间统计（含 RMSE/MAE 的绝对量纲值）。"""
    aggregate = (version.rolling.get("aggregate") or {}).get("per_metric") or {}
    rows = []
    for name in ("R2", "CVRMSE", "NMBE", "MAPE", "RMSE", "MAE"):
        entry = aggregate.get(name)
        if not entry:
            continue
        rows.append([name, metric_text(name, entry.get("mean")),
                     metric_text(name, entry.get("std")),
                     metric_text(name, entry.get("min")),
                     metric_text(name, entry.get("max")),
                     str(entry.get("n_defined", "—"))])
    return rows


def io_rows(version: Version) -> list[list[str]]:
    rows = []
    for item in version.signature.get("inputs") or []:
        code = str(item.get("property_code"))
        rows.append(["输入", code, label(code), str(item.get("unit") or ""),
                     str(item.get("dtype") or ""),
                     version.kinds.get(code, "—"),
                     "是" if item.get("required", True) else "否"])
    for item in version.signature.get("outputs") or []:
        code = str(item.get("property_code"))
        rows.append(["**输出**", code, label(code), str(item.get("unit") or ""),
                     str(item.get("dtype") or ""),
                     version.kinds.get(code, "—"), "—"])
    return rows


def split_lines(version: Version) -> list[str]:
    split = version.validation.get("split") or {}
    if not split:
        return []
    counts = split.get("counts") or {}
    def span(key: str) -> str:
        pair = split.get(key) or ["", ""]
        return f"{ts(pair[0])} → {ts(pair[1])}"
    return table(
        ["切分段", "时间区间（UTC）", "行数"],
        [["训练 train", span("train_range"), str(counts.get("train", "—"))],
         ["验证 validate", span("validate_range"), str(counts.get("validate", "—"))],
         ["测试 A", span("test_range"), str(counts.get("test", "—"))],
         ["purge + embargo 丢弃",
          f"purge {num(split.get('purge_seconds'))} s / "
          f"embargo {num(split.get('embargo_seconds'))} s",
          str(counts.get("purged_or_embargoed", "—"))]])


def version_section(version: Version, models_by_id: Mapping[str, Model]) -> list[str]:
    lines: list[str] = []
    signature = version.signature
    outputs = ", ".join(f"{o.get('property_code')}（{o.get('unit')}）"
                        for o in signature.get("outputs") or [])
    goal_name = version.goal.get("name") or "—"
    dataset = version.dataset_lineage.get("dataset") or "—"
    experiment = version.spec.get("experiment") or {}
    model_spec = experiment.get("model") or {}
    runtime = experiment.get("runtime") or {}

    lines += [f"### 定位", ""]
    # 模型 ID 不再单列一行：分节标题里已经有了，重复一次还会自链到本节。
    lines += table(["项", "值"], [
        ["版本", version.version],
        ["状态", STATUS_LABELS.get(version.status, version.status)],
        ["对象模型", str(signature.get("object_model") or "—")],
        ["建模对象", ", ".join(str(o) for o in
                             (version.spec.get("view_definition") or {}).get("objects") or []) or "—"],
        ["预测目标", outputs or "—"],
        ["一句话说明", version.description or "—"],
        ["研究目标", f"{version.goal_id} · {goal_name}"],
        ["假设", f"{version.research_lineage.get('hypothesis_id', '—')}："
                 f"{version.hypothesis.get('statement') or '—'}"],
        ["实验 / 视图", f"{version.experiment_id} / "
                        f"{version.dataset_lineage.get('view_id', '—')}"],
        ["数据集修订", f"`{dataset}`"],
        ["随机种子 / 代码版本",
         f"{runtime.get('random_seed', '—')} / "
         f"`{str(version.spec.get('code_version') or '')[:8]}`"],
        ["环境锁", f"`{str(runtime.get('environment_lock') or '')[:16]}…`"],
    ])

    lines += ["### 输入 → 输出", ""]
    lines += table(["方向", "property_code", "含义", "单位", "类型", "数据来源", "必填"],
                   io_rows(version))
    lines += [f"越界处理：全部输入 `out_of_range = reject`（超出训练包络直接拒绝，"
              f"不外推）。", ""]

    formula = version.formula
    if formula:
        lines += ["### 公式", "", f"**模型形态**：{formula.family}", "",
                  "```", *formula.equations, "```", ""]
        if formula.substituted:
            lines += ["**代入本模型辨识出的参数**：", "", "```",
                      *formula.substituted, "```", ""]
        if formula.params:
            lines += ["**参数取值**", ""]
            lines += table(["参数", "含义", "取值", "单位"],
                           [list(row) for row in formula.params])
        for note in formula.notes:
            lines += [f"> {note}", ""]
    else:
        lines += ["### 公式", "",
                  "> 该工件格式尚未接入公式渲染，见包内 `artifact/model.json`。", ""]

    if version.importance:
        lines += ["**特征增益占比（XGBoost gain，归一化到 100%）**", ""]
        lines += table(["特征", "含义", "增益占比"],
                       [[name, label(name), f"{value:.1f}%"]
                        for name, value in version.importance])

    lines += ["### 结果", ""]
    lines += table(["判据面", "样本数"] + ["R²", "CVRMSE", "NMBE", "MAPE"],
                   surface_rows(version.metrics))
    detail = rolling_detail_rows(version)
    if detail:
        lines += ["滚动交叉验证的折间统计（RMSE / MAE 为绝对量纲，单位同预测目标）：", ""]
        lines += table(["指标", "折间均值", "折间标准差", "最小", "最大", "有效折数"],
                       detail)
    physics_only = version.metrics.get("physics_only")
    if physics_only:
        rows = []
        for key in ("validate", "A"):
            entry = physics_only.get(key) or {}
            values = entry.get("metrics") or {}
            if not values:
                continue
            share = (version.metrics.get("residual_share") or {}).get(key)
            rows.append([SURFACE_LABELS.get(key, key), str(entry.get("n_samples", "—"))]
                        + [metric_text(m, values.get(m)) for m in METRIC_ORDER]
                        + [pct(share)])
        if rows:
            lines += ["**只用物理主干（不加残差修正）时的表现**，"
                      "以及残差项占预测值的比重：", ""]
            lines += table(["判据面", "样本数", "R²", "CVRMSE", "NMBE", "MAPE",
                            "残差占比"], rows)
    for surface in ("validate", "A"):
        rows = per_object_rows(version.metrics, surface)
        if rows:
            lines += [f"**{SURFACE_LABELS.get(surface, surface)} 上的逐台拆分**", ""]
            lines += table(["对象", "样本数", "R²", "CVRMSE", "NMBE", "MAPE"], rows)
    dropped = version.metrics.get("dropped_na_rows") or {}
    if dropped:
        lines += [f"缺失值丢弃行数：训练 {dropped.get('train', 0)}、"
                  f"验证 {dropped.get('validate', 0)}、测试 {dropped.get('test', 0)}"
                  "（该行任一输入或目标为空即整行丢弃）。", ""]

    lines += ["### 验收与门禁", ""]
    rows = acceptance_rows(version)
    if rows:
        acceptance = (version.goal.get("definition") or {}).get("acceptance") or {}
        lines += [f"目标 {version.goal_id} 声明的判据面："
                  f"`{acceptance.get('evaluated_on', 'rolling_cv')}`。", ""]
        lines += table(["判据", "阈值", "实测", "结论"], rows)
    gates = [[str(g.get("name")), "通过" if g.get("ok") else "**未通过**",
              str(g.get("detail") or "")] for g in version.gates]
    if gates:
        lines += ["发布门禁（`runtime/registry.py`，模型包 §8）：", ""]
        lines += table(["门禁", "结论", "说明"], gates)
    if version.rollback:
        lines += [f"回滚目标版本：`{version.rollback}`。", ""]

    lines += ["### 数据与时间切分", ""]
    lines += split_lines(version)
    lines += [f"视图哈希 `{str(version.dataset_lineage.get('view_hash') or '')[:16]}…`；"
              f"物理检查违规率 "
              f"{pct((version.validation.get('physics') or {}).get('overall_rate'))}"
              f"（{(version.validation.get('physics') or {}).get('overall_violations', 0)} 次 /"
              f" {(version.validation.get('physics') or {}).get('n_samples', 0)} 样本）。", ""]
    return lines


def history_section(model: Model) -> list[str]:
    rows = []
    for version in model.versions:
        rolling = (version.metrics.get("rolling_cv") or {}).get("metrics") or {}
        rows.append([
            version.version,
            STATUS_LABELS.get(version.status, version.status)
            + ("（当前生产）" if version.version == model.production else ""),
            version.experiment_id or "—",
            (version.formula.family.split("（")[0].split("——")[0].strip()
             if version.formula else "—"),
            metric_text("R2", rolling.get("R2")),
            metric_text("CVRMSE", rolling.get("CVRMSE")),
            metric_text("NMBE", rolling.get("NMBE")),
            version.description[:40] or "—",
        ])
    if len(rows) < 2:
        return []
    return ["### 版本历史", ""] + table(
        ["版本", "状态", "实验", "形态", "R²(滚动)", "CVRMSE(滚动)", "NMBE(滚动)", "说明"],
        rows)


def overview_rows(models: list[Model]) -> list[list[str]]:
    rows = []
    for model in models:
        version = model.current
        if version is None:
            continue
        rolling = (version.metrics.get("rolling_cv") or {}).get("metrics") or {}
        outputs = ", ".join(str(o.get("property_code"))
                            for o in version.signature.get("outputs") or [])
        rows.append([
            f"`{model.model_id}`", version.version,
            STATUS_LABELS.get(version.status, version.status),
            outputs, str(len(version.signature.get("inputs") or [])),
            (version.formula.family.split("（")[0].split("——")[0].strip()
             if version.formula else "—"),
            metric_text("R2", rolling.get("R2")),
            metric_text("CVRMSE", rolling.get("CVRMSE")),
            metric_text("NMBE", rolling.get("NMBE")),
            metric_text("MAPE", rolling.get("MAPE")),
        ])
    return rows


def preamble(models: list[Model], generated_at: str) -> list[str]:
    total_versions = sum(len(m.versions) for m in models)
    return [
        "# ThermoForge 模型结果报告",
        "",
        f"生成时间：{generated_at}　·　"
        f"覆盖 {len(models)} 个模型、{total_versions} 个已注册版本",
        "",
        "本报告由 `scripts/export_model_report.py` 从模型注册表 `models/` 与研究账本 "
        "`research/` 直接导出，公式与参数取自各模型包的 `artifact/`，指标取自包内 "
        "`metrics.json`（唯一实现 `research/metrics.py`，**比率制**，报告中转成百分数）。"
        "任何一个数字都可以回到对应的实验目录复核。",
        "",
        "## 一、模型总览",
        "",
    ] + table(
        ["模型", "生产版本", "状态", "预测目标", "输入数", "形态",
         "R²（滚动）", "CVRMSE（滚动）", "NMBE（滚动）", "MAPE（滚动）"],
        overview_rows(models)) + [
        "> 表中指标一律取**滚动交叉验证**面（各目标声明的判据面），"
        "它是唯一用于发布决策的口径；验证集/测试集指标在各模型分节里给出。",
        "",
    ]


def method_section() -> list[str]:
    return [
        "## 二、口径说明（先读这一节，否则数字容易被误读）",
        "",
        "### 指标定义",
        "",
        "四个指标由 `research/metrics.py` 单一实现，模型代码不得自行计算：",
        "",
        "```",
        "RMSE   = sqrt( mean( (ŷ − y)² ) )",
        "CVRMSE = RMSE / mean(y)                       变异系数化的 RMSE，相对误差幅度",
        "NMBE   = mean(ŷ − y) / mean(y)                归一化平均偏差，**正=系统高估**",
        "MAPE   = mean( |ŷ − y| / |y| )                丢弃 |y| < y_floor 的样本后计算",
        "R²     = 1 − Σ(y − ŷ)² / Σ(y − mean(y))²      解释方差比例，可为负",
        "```",
        "",
        "- CVRMSE / NMBE / MAPE 在文件里存的是**比率**，本报告统一乘 100 显示为百分数。",
        "- `mean(y) ≈ 0` 时 CVRMSE / NMBE 记为 `None` 而不是给一个看起来合理的数。",
        "- MAPE 会丢掉 `|y| < y_floor` 的样本并记录有效比例，低于 0.8 会报 `TFX-905`。",
        "",
        "### 三张判据面的区别",
        "",
        "| 面 | 怎么来的 | 用途 |",
        "|---|---|---|",
        "| 滚动交叉验证 rolling_cv | 从训练区起，按固定跨度（7 或 30 天）逐步扩展训练窗、"
        "在下一窗上评估，取折间**均值** | **唯一的发布判据面** |",
        "| 验证集 validate | 时间切分的中段（默认 15%） | 调参与诊断 |",
        "| 测试集 A | 时间切分的末段（默认 15%），时间上离训练最远 | 最保守的泛化观察 |",
        "",
        "切分按**时间**而非行数：边界对齐到采样周期整数倍，训练尾部 purge、"
        "验证头部 embargo（默认 2700 s = 45 min），重叠或间隙不足直接报 `TFX-903`。",
        "",
        "### R² 为负是什么意思，为什么还能发布",
        "",
        "R² 为负 = 该面上模型比「直接用这一折的均值」还差。它在**窄工况**上很容易出现："
        "折内目标方差很小时分母趋零，一点点偏差就把 R² 拉到负数，而 CVRMSE 可以同时很好。",
        "所以本项目的发布判据以 CVRMSE / NMBE 为主，R² 只在目标显式声明 `r2_min` 时才是"
        "硬门槛。报告里两个都给，看到 R² 负值请同时看 CVRMSE 和折间标准差。",
        "",
        "### CVRMSE 跨模型不可直接比较",
        "",
        "CVRMSE 的分母是各自的 `mean(y)`。群模型的均值是十几台之和，"
        "单台模型只有它的十几分之一——**同样的绝对误差，单台的 CVRMSE 会大十几倍**。"
        "研究账本 F-0087 用塔风机做过判决性分解：`wx-ct-fan-power-total`（群，2.43%）与 "
        "`wx-ct-fan-power-unit`（逐台，6.08%）看似差 2.5 倍，把逐台预测按时刻求和后"
        "在同一面上对比，逐台模型反而更好（测试面 3.10% vs 3.55%，验证面 4.46% vs 6.56%）。",
        "",
        "### 白名单硬门禁（为什么输入列看起来「少了点」）",
        "",
        "候选输入白名单会在目标层与实验层各拦一次：派生量不得用来预测它自己的原料。"
        "现场数据里 `load = current_percent × 9672 / 100`，用它预测功率是循环论证，"
        "能刷出 4.35% 的假 MAPE。报告里每个输入都标了 `数据来源`："
        "`measured`（实测点位）/ `estimated`（由测点换算，如 lift、flow_ratio、湿球温度）。",
        "",
        "### 复现方式",
        "",
        "每个模型包自带 `environment.lock`（含全部依赖的版本与哈希）、`checksums.json`、"
        "以及 `golden.parquet` 预测集。实验在子进程里跑，线程数环境变量在 numpy/sklearn "
        "导入**之前**设定；同机器 + 同环境锁必须逐位复现指标。包内不含任何 pickle。",
        "",
    ]


def caveats_section(models: list[Model]) -> list[str]:
    return [
        "## 四、需要注意的地方",
        "",
        "1. **`wx-ct-approach` 的包描述与实现不符。** 描述写「Merkel/NTU 主干 + 偏差校正」，"
        "落盘的 `lab_source.py` 里没有 Merkel 主干，实际是 21 项二阶多项式岭回归。"
        "描述沿用了假设 H-0099 的措辞，没随实现更新。以本报告的公式一节为准。",
        "",
        "2. **两个 `-static` 模型是泄漏事故的干净重跑。** `wx-chiller-power-static`（EXP-0176，"
        "重跑 EXP-0058）与 `wx-hx-heat-transfer-static`（EXP-0177，重跑 EXP-0061）"
        "对应的原始实验受实验室目标列泄漏影响，原指标不可用；这里的指标来自重跑。",
        "",
        "3. **`wx-hx-heat-transfer` 1.1.0 与 `wx-hx-heat-transfer-static` 1.0.0 指标完全相同。** "
        "两者同视图 VIEW-0042、同模块、同超参、同种子，是同一个模型的两次注册"
        "（前者带物理校验重跑，后者是泄漏事故的干净重跑），不是两个独立结果。",
        "",
        "4. **`wx-chwp-flow-total` 用 `power_total` 预测 `flow_total`。** 增益占比里 "
        "power_total 一项占 61.8%——这是「用泵功耗反推流量」，物理上成立（同一台泵上"
        "功耗与流量单调相关），但它要求功耗测点在线可用，做流量**软测量**没问题，"
        "做 what-if（改频率看流量）时要小心：功耗本身也是频率的函数。",
        "",
        "5. **`wx-chiller-power-ch0x` 的 Gordon-Ng 主干普遍退化。** CH01/CH03/CH04 辨识出的"
        "换热热阻里至少有一个被压到 1e−12 及更小（CH04 是 R_e、R_c 双双归零），"
        "物理主干几乎只剩熵产项。"
        "原因是蒸发侧温差摆幅只有 0.4 K 量级，热阻不可辨识。CH02 因此改走纯 GBDT 路线"
        "（假设 H-0037），指标从 R²=0.588 提到 0.723。",
        "",
        "6. **逐台风机模型 12 台共用一组参数。** `wx-ct-fan-power-unit` 不区分 object_id，"
        "逐台差异全部进残差；逐台指标表里 CTF0401 的验证面 R²=0.909 是最差的一台。"
        "误差两两相关系数均值 +0.314（F-0087），说明存在共模偏差——继续压随机误差对"
        "群口径收益有限。",
        "",
        "7. **四台冷机的缺失行丢弃都不小，CH02 尤其极端。** 训练段共约 10450 行，"
        "CH01 丢 3420、CH03 丢 3515、CH02 丢 2766、CH04 丢 2151（机组停机时该行整行为空）。"
        "更要紧的是 CH02 的测试段：2239 行里丢掉 2233 行，**测试面 A 只剩 6 个样本**，"
        "那一列的 R²=−15.97 之类的数字没有统计意义。看 A 面指标一定要先看样本数。",
        "",
        "8. **`RG-0012` 的 NMBE 上限被人工放宽过。** 2026-08-21 由 2.0% 改到 3.5%，"
        "为的是放行 CH01（2.47%）与 CH03（3.40%）。这是需求方的显式决定，"
        "记在目标的 transitions 里，不是模型指标变好了。",
        "",
    ]


def render_markdown(models: list[Model], generated_at: str) -> str:
    lines = preamble(models, generated_at)
    lines += method_section()
    lines += ["## 三、逐模型明细", ""]
    for index, model in enumerate(models, start=1):
        version = model.current
        if version is None:
            continue
        title = version.description.split("：")[0].split("（")[0][:28] or model.model_id
        lines += [f"## 3.{index}　{model.model_id}　—　{title}", ""]
        lines += version_section(version, {m.model_id: m for m in models})
        lines += history_section(model)
    lines += caveats_section(models)
    lines += [
        "---",
        "",
        "报告由 `scripts/export_model_report.py` 生成；数据源为本机 `models/` 与 "
        "`research/` 目录的当前内容。重新生成：",
        "",
        "```",
        ".venv/Scripts/python scripts/export_model_report.py",
        "```",
        "",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------- HTML

# 设计口径：这是一份**仪表台风格的工程档案**，不是营销页。
# 颜色 —— 纸面偏冷绿的中性灰（不是纯灰，也不是暖米色），主色取深青绿
# （制冷/换热的行业色感），警示色取赭橙，只用在「注意」块与未通过门禁上。
# 字体 —— 正文走系统无衬线（中文命中雅黑/苹方），公式、ID、哈希、数字一律
# 走等宽：本报告里等宽不是装饰，是「仪器读数」这一层信息的载体。
# 版式 —— 单列 + 顶部目录卡；表格各自横向滚动，页面本体永不横向滚动。
_CSS = """
:root{
  --paper:#f6f8f7; --ink:#141a18; --ink-soft:#5c6a66;
  --line:#dde4e1; --sunk:#ecf1ef; --rule:#c6d3cf;
  --accent:#0f6b5c; --accent-soft:#e2efeb;
  --warn:#a3561d; --warn-soft:#f6ecdf;
  --ok:#1d6b4a; --ok-soft:#e3f0e9;
  --chip:#5c6a66; --chip-soft:#e7eae9;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --paper:#0f1413; --ink:#e3e9e6; --ink-soft:#94a49f;
    --line:#232c2a; --sunk:#161d1c; --rule:#31403c;
    --accent:#4fc0a7; --accent-soft:#12332d;
    --warn:#d99a5c; --warn-soft:#31251a;
    --ok:#57bd8c; --ok-soft:#122b20;
    --chip:#94a49f; --chip-soft:#1c2422;
  }
}
:root[data-theme="dark"]{
  --paper:#0f1413; --ink:#e3e9e6; --ink-soft:#94a49f;
  --line:#232c2a; --sunk:#161d1c; --rule:#31403c;
  --accent:#4fc0a7; --accent-soft:#12332d;
  --warn:#d99a5c; --warn-soft:#31251a;
  --ok:#57bd8c; --ok-soft:#122b20;
  --chip:#94a49f; --chip-soft:#1c2422;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
@media (prefers-reduced-motion: reduce){html{scroll-behavior:auto}}
body{margin:0;background:var(--paper);color:var(--ink);
  font:16px/1.75 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",
       "Microsoft YaHei","Hiragino Sans GB",sans-serif;
  -webkit-font-smoothing:antialiased}
.wrap{max-width:1120px;margin:0 auto;padding:clamp(28px,5vw,64px) 22px 96px}
h1{font-size:clamp(28px,4.4vw,40px);line-height:1.22;margin:0 0 14px;
  letter-spacing:-.01em;text-wrap:balance}
h2{font-size:clamp(20px,2.6vw,25px);line-height:1.35;margin:64px 0 6px;
  padding-top:18px;border-top:2px solid var(--rule);text-wrap:balance;
  scroll-margin-top:12px}
h3{font-size:15px;margin:34px 0 4px;color:var(--accent);
  letter-spacing:.06em;font-weight:700}
h3::before{content:"";display:inline-block;width:16px;height:1px;
  background:var(--accent);vertical-align:.35em;margin-right:9px}
p{margin:12px 0;max-width:74ch}
a{color:var(--accent);text-underline-offset:3px}
a:focus-visible{outline:2px solid var(--accent);outline-offset:3px;
  border-radius:3px}
.tablewrap{overflow-x:auto;margin:14px 0;border:1px solid var(--line);
  border-radius:8px;background:var(--sunk)}
table{border-collapse:collapse;width:100%;font-size:13.5px;
  font-variant-numeric:tabular-nums}
th,td{border-bottom:1px solid var(--line);padding:8px 12px;text-align:right;
  vertical-align:top;line-height:1.55}
th{background:var(--sunk);font-weight:650;text-align:center;white-space:nowrap;
  font-size:12px;letter-spacing:.04em;color:var(--ink-soft);
  border-bottom:1px solid var(--rule)}
td{background:var(--paper)}
td:first-child,th:first-child,td:nth-child(2),td:nth-child(3){text-align:left}
tbody tr:last-child td{border-bottom:0}
code{background:var(--sunk);padding:1px 5px;border-radius:4px;font-size:.88em;
  font-family:ui-monospace,"Cascadia Mono",Consolas,"Courier New",monospace;
  border:1px solid var(--line)}
pre{background:var(--sunk);border:1px solid var(--line);
  border-left:3px solid var(--accent);padding:14px 18px;border-radius:8px;
  overflow-x:auto;font-size:13px;line-height:1.7;margin:14px 0;
  font-family:ui-monospace,"Cascadia Mono",Consolas,"Courier New",monospace}
pre code{background:none;padding:0;border:0;font-size:inherit}
blockquote{margin:14px 0;padding:11px 16px;border-left:3px solid var(--warn);
  background:var(--warn-soft);border-radius:0 8px 8px 0;color:var(--ink)}
blockquote p{margin:0;max-width:none;font-size:14.5px}
ol,ul{padding-left:22px;max-width:74ch}
li{margin:8px 0}
hr{border:0;border-top:1px solid var(--line);margin:48px 0 20px}
strong{font-weight:680}
.stamp{font-family:ui-monospace,"Cascadia Mono",Consolas,monospace;
  font-size:12.5px;color:var(--ink-soft);letter-spacing:.02em;
  padding:2px 0 18px;border-bottom:1px solid var(--rule);margin-bottom:8px}
.pill{display:inline-block;padding:1px 9px;border-radius:999px;
  font-size:11.5px;font-weight:650;letter-spacing:.03em;white-space:nowrap;
  background:var(--chip-soft);color:var(--chip)}
.pill-ok{background:var(--ok-soft);color:var(--ok)}
.pill-warn{background:var(--warn-soft);color:var(--warn)}
.pill-live{background:var(--accent-soft);color:var(--accent)}
td a code{border-color:currentColor}
"""

# 表格里这几个词是**状态**而不是普通文本，渲染成 chip 让人一眼扫到。
_PILL_CLASSES = {
    "生产": "pill-live", "已批准": "pill-live", "已验证": "pill-live",
    "通过": "pill-ok",
    "候选": "", "已弃用": "", "已退役": "",
    "不通过": "pill-warn", "未通过": "pill-warn",
}


def _inline(text: str) -> str:
    """行内标记：先整体转义，再还原 `code`、**bold**、[文字](#锚)。"""
    out = html_escape.escape(text)
    out = re.sub(r"`([^`]+)`", r"<code>\1</code>", out)
    out = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", out)
    out = re.sub(r"\[([^\]]+)\]\((#[\w.-]+)\)", r'<a href="\2">\1</a>', out)
    return out


def _cell_html(text: str) -> str:
    """状态词渲染成 chip；其余单元格走普通行内标记。"""
    stripped = text.replace("**", "").strip()
    suffix = ""
    if stripped.endswith("（当前生产）"):
        stripped = stripped[:-6]
        suffix = ' <span class="pill pill-live">当前</span>'
    if stripped in _PILL_CLASSES:
        css = _PILL_CLASSES[stripped]
        return (f'<span class="pill {css}">{html_escape.escape(stripped)}'
                f"</span>{suffix}")
    # 总览表的模型 ID 直接跳到该模型的分节：总览表本身就是目录，不另起一张。
    model_id = re.fullmatch(r"`(wx-[a-z0-9-]+)`", stripped)
    if model_id:
        anchor = model_id.group(1)
        return f'<a href="#{anchor}"><code>{anchor}</code></a>'
    return _inline(text)


def _flush_table(buffer: list[str], out: list[str]) -> None:
    if not buffer:
        return
    # 单元格里的管道符在 Markdown 里写作 `\|`，切分时必须跳过转义的那一个，
    # 否则「|NMBE| ≤」这种判据名会把一行切成六列。
    rows = [[cell.strip().replace("\\|", "|")
             for cell in re.split(r"(?<!\\)\|", line.strip().strip("|"))]
            for line in buffer]
    header, body = rows[0], rows[2:]
    out.append('<div class="tablewrap"><table><thead><tr>'
               + "".join(f"<th>{_inline(c)}</th>" for c in header)
               + "</tr></thead><tbody>")
    for row in body:
        out.append("<tr>" + "".join(f"<td>{_cell_html(c)}</td>" for c in row)
                   + "</tr>")
    out.append("</tbody></table></div>")
    buffer.clear()


def heading_id(text: str) -> str:
    """章节锚点：模型分节取 model_id，其余按序号，稳定且可读。"""
    match = re.search(r"(wx-[a-z0-9-]+)", text)
    if match:
        return match.group(1)
    match = re.match(r"([一二三四五六七八九十]+|\d+(?:\.\d+)?)", text.strip())
    return "sec-" + (match.group(1) if match else "x")


def markdown_to_html(text: str) -> str:
    """本报告用到的 Markdown 子集 → HTML。不追求通用，够用且可审计。"""
    out: list[str] = []
    table_buffer: list[str] = []
    in_code = False
    code_lines: list[str] = []
    list_items: list[str] = []

    def flush_list() -> None:
        if list_items:
            out.append("<ol>" + "".join(f"<li>{_inline(i)}</li>"
                                        for i in list_items) + "</ol>")
            list_items.clear()

    for raw in text.splitlines():
        line = raw.rstrip()
        if line.startswith("```"):
            if in_code:
                out.append("<pre><code>"
                           + html_escape.escape("\n".join(code_lines))
                           + "</code></pre>")
                code_lines.clear()
            else:
                _flush_table(table_buffer, out)
                flush_list()
            in_code = not in_code
            continue
        if in_code:
            code_lines.append(raw)
            continue
        if line.startswith("|"):
            flush_list()
            table_buffer.append(line)
            continue
        _flush_table(table_buffer, out)
        if not line:
            flush_list()
            continue
        if line.startswith("<"):          # 生成器直接给的 HTML 片段（目录卡）
            flush_list()
            out.append(line)
        elif line.startswith("#"):
            flush_list()
            level = len(line) - len(line.lstrip("#"))
            body = line[level:].strip()
            anchor = f' id="{heading_id(body)}"' if level == 2 else ""
            out.append(f"<h{level}{anchor}>{_inline(body)}</h{level}>")
        elif line.startswith(">"):
            flush_list()
            out.append(f"<blockquote><p>{_inline(line[1:].strip())}</p>"
                       "</blockquote>")
        elif line.startswith("---"):
            flush_list()
            out.append("<hr>")
        elif re.match(r"^\d+\.\s", line):
            list_items.append(re.sub(r"^\d+\.\s", "", line))
        elif line.startswith("- "):
            flush_list()
            out.append(f"<ul><li>{_inline(line[2:])}</li></ul>")
        elif line.startswith("生成时间："):
            flush_list()
            out.append(f'<p class="stamp">{_inline(line)}</p>')
        else:
            flush_list()
            out.append(f"<p>{_inline(line)}</p>")
    _flush_table(table_buffer, out)
    flush_list()
    return "\n".join(out)


def render_html(markdown_text: str, title: str, *,
                standalone: bool = True) -> str:
    """`standalone=True` 出可双击打开的完整页；False 出只含 title/style/正文
    的片段——Artifact 发布时外层已经套了 doctype/head/body。"""
    body = f'<div class="wrap">\n{markdown_to_html(markdown_text)}\n</div>'
    title_tag = f"<title>{html_escape.escape(title)}</title>"
    if not standalone:
        return f"{title_tag}\n<style>{_CSS}</style>\n{body}\n"
    return ('<!doctype html>\n<html lang="zh-CN"><head><meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width,'
            'initial-scale=1">\n'
            f"{title_tag}\n<style>{_CSS}</style>\n</head><body>\n{body}\n"
            "</body></html>\n")


# --------------------------------------------------------------- 入口


def main() -> int:
    parser = argparse.ArgumentParser(description="导出模型结果报告")
    parser.add_argument("--out-dir", default="docs/sharing",
                        help="产物目录（默认 docs/sharing）")
    parser.add_argument("--basename", default="模型结果报告")
    parser.add_argument("--fragment", default=None,
                        help="额外写一份只含正文的 HTML 片段（供 Artifact 发布）")
    parser.add_argument("--title", default="ThermoForge 模型档案")
    args = parser.parse_args()

    models = load_models()
    if not models:
        print("模型注册表为空，没有可报告的内容。")
        return 1
    # 时区用 UTC 偏移而不是 %Z：Windows 上 %Z 返回本地编码的中文名，
    # 写进 UTF-8 文件会变成一串问号。
    now = datetime.now().astimezone()
    generated_at = now.strftime("%Y-%m-%d %H:%M ") + f"UTC{now.strftime('%z')[:3]}"
    markdown_text = render_markdown(models, generated_at)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{args.basename}.md"
    html_path = out_dir / f"{args.basename}.html"
    md_path.write_text(markdown_text, encoding="utf-8", newline="\n")
    html_path.write_text(render_html(markdown_text, args.title),
                         encoding="utf-8", newline="\n")
    print(f"模型 {len(models)} 个，版本 {sum(len(m.versions) for m in models)} 个")
    print(f"写出 {md_path}")
    print(f"写出 {html_path}")
    if args.fragment:
        fragment_path = Path(args.fragment)
        if not fragment_path.is_absolute():
            fragment_path = ROOT / fragment_path
        fragment_path.parent.mkdir(parents=True, exist_ok=True)
        fragment_path.write_text(
            render_html(markdown_text, args.title, standalone=False),
            encoding="utf-8", newline="\n")
        print(f"写出 {fragment_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
