"""模型结构解析：从工件目录还原「这个模型到底是什么」。

纯函数层：只读文件，不 import streamlit / 绘图库，可直接被 pytest 测。
页面渲染在 `screens/structure.py`，报告导出（`services/reports.py`）复用
同一份解析结果——公式、参数、结构在每个展示位都是同一口径。

分派依据是工件 `model.json` 的 `format` 字段（模型层常量，import 而不抄
字符串，防漂移）。系统目前没有神经网络路线；将来出现新 format 时在这里
加一个 `_xxx` 解析分支即可。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from thermoforge_models.baseline import MODEL_FORMAT as LINEAR_FORMAT
from thermoforge_models.hybrid import MODEL_FORMAT as HYBRID_FORMAT
from thermoforge_models.identification import GN_FORMAT, NTU_FORMAT
from thermoforge_models.physics import (
    MODEL_FORMAT as PHYSICS_V1_FORMAT,
    MODEL_FORMAT_V2 as PHYSICS_V2_FORMAT,
)

# 特征重要性只画前几条，宽表全画出来图没法看
TOP_IMPORTANCE = 12


@dataclass(frozen=True)
class ParamRow:
    """一个可展示参数：取值 + 单位 + 合法范围（有的话）+ 备注。"""

    name: str
    value: float | None
    unit: str = ""
    bounds: tuple[float, float] | None = None
    note: str = ""


@dataclass(frozen=True)
class ModelDoc:
    """一个模型工件的结构化描述。

    `equations` 为符号式（纯文本，与模型包 params.yaml 同口径，报告导出
    也用这份）；`substituted` 为代入辨识参数后的数值式；`latex` 供页面
    `st.latex` 渲染。hybrid 的主干在 `base`（递归 ModelDoc）。
    """

    kind: str
    format: str
    summary: str
    equations: tuple[str, ...] = ()
    substituted: tuple[str, ...] = ()
    latex: tuple[str, ...] = ()
    params: tuple[ParamRow, ...] = ()
    inputs: dict[str, str] = field(default_factory=dict)
    notes: tuple[str, ...] = ()
    coefficients: dict[str, float] = field(default_factory=dict)
    intercept: float | None = None
    scaler: dict[str, Any] | None = None
    base: "ModelDoc | None" = None
    xgb: dict[str, Any] | None = None
    lab_source: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


def load_model_doc(directory: str | Path) -> ModelDoc | None:
    """读模型工件目录。

    目录或 model.json 不存在返回 None（调用方直接不展示）；格式未知返回
    kind="unknown" 的兜底文档（界面展示原文清单），不抛异常。
    """
    directory = Path(directory)
    meta_path = directory / "model.json"
    if not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    fmt = str(meta.get("format") or "")
    if fmt == HYBRID_FORMAT:
        return _hybrid(directory, meta)
    if fmt == LINEAR_FORMAT:
        return _linear(meta)
    if fmt in (PHYSICS_V1_FORMAT, PHYSICS_V2_FORMAT):
        return _physics_family(directory)
    if fmt == GN_FORMAT:
        return _gordon_ng(meta)
    if fmt == NTU_FORMAT:
        return _eps_ntu(meta)
    if fmt.startswith("thermoforge.lab."):
        return _lab(directory, meta)
    return ModelDoc(kind="unknown", format=fmt or "（未标注）",
                    summary="未识别的模型格式，展示原始清单。", raw=meta)


# ---------------------------------------------------------------- 线性基线


def _fmt_num(value: float | None) -> str:
    return "—" if value is None else f"{float(value):.6g}"


def _linear(meta: Mapping[str, Any]) -> ModelDoc:
    coefs = {str(k): float(v)
             for k, v in (meta.get("coefficients") or {}).items()}
    intercept = float(meta.get("intercept") or 0.0)
    method = str(meta.get("method") or "linear")
    alpha = meta.get("alpha")
    is_ridge = method == "ridge"
    method_label = ("岭回归 Ridge，α=" + _fmt_num(alpha)) if is_ridge \
        else "普通最小二乘"
    summary = (f"线性基线（{method_label}）："
               f"{len(coefs)} 个标准化特征加权求和")
    terms = "".join(f" {'+' if w >= 0 else '−'} {abs(w):.4g}·z_{name}"
                    for name, w in coefs.items())
    params = [ParamRow("intercept", intercept, note="截距（目标单位）")]
    if is_ridge:
        params.append(ParamRow("alpha", float(alpha or 0.0),
                               note="岭回归正则强度"))
    return ModelDoc(
        kind="linear", format=LINEAR_FORMAT, summary=summary,
        equations=("z_i = (x_i − mean_i) / std_i　（训练集均值/标准差，"
                   "随模型落盘）",
                   "ŷ = intercept + Σᵢ wᵢ·zᵢ"),
        substituted=(f"ŷ = {intercept:.4g}{terms}",),
        latex=(r"z_i = \dfrac{x_i - \mu_i}{\sigma_i}",
               r"\hat{y} = b + \sum_i w_i\, z_i"),
        params=tuple(params),
        notes=("系数作用在标准化特征上：|w| 可直接比大小，"
               "含义是「该特征每动一个标准差，目标动多少」。",),
        coefficients=coefs, intercept=intercept,
        scaler=dict(meta.get("scaler") or {}),
    )


# ---------------------------------------------------------------- 能量平衡族


def _physics_family(directory: Path) -> ModelDoc:
    """v1/v2 的参数真源是 params.yaml（hybrid 的 model.json 是混合元信息）。"""
    params_path = directory / "params.yaml"
    if not params_path.is_file():
        return ModelDoc(kind="unknown", format="physics（缺 params.yaml）",
                        summary="物理模型参数文件缺失。")
    doc = yaml.safe_load(params_path.read_text(encoding="utf-8")) or {}
    fmt = str(doc.get("format") or "")
    equations = tuple(str(e) for e in doc.get("equations") or ())
    inputs = {str(k): str(v) for k, v in (doc.get("inputs") or {}).items()}
    if fmt == PHYSICS_V2_FORMAT:
        return _physics_v2(doc, equations, inputs)
    return _physics_v1(doc, equations, inputs)


def _param_rows(params: Mapping[str, Any]) -> list[ParamRow]:
    """params.yaml 的 parameters 段 → 展示行（递归展开嵌套段）。"""
    rows: list[ParamRow] = []
    for name, entry in params.items():
        if not isinstance(entry, Mapping):
            continue
        if "value" in entry:
            bounds = entry.get("bounds")
            rows.append(ParamRow(
                str(name),
                float(entry["value"]) if entry["value"] is not None else None,
                unit=str(entry.get("unit") or ""),
                bounds=(tuple(float(b) for b in bounds)
                        if isinstance(bounds, (list, tuple)) else None),
                note=str(entry.get("note") or "")))
        else:  # 嵌套段（cop_coefficients / curves.capft…）：名称带前缀展开
            for sub, sub_entry in entry.items():
                if isinstance(sub_entry, Mapping) and "value" in sub_entry:
                    bounds = sub_entry.get("bounds")
                    rows.append(ParamRow(
                        f"{name}.{sub}",
                        float(sub_entry["value"]),
                        unit=str(sub_entry.get("unit") or ""),
                        bounds=(tuple(float(b) for b in bounds)
                                if isinstance(bounds, (list, tuple)) else None),
                        note=str(sub_entry.get("note") or "")))
    return rows


def _identification_notes(ident: Mapping[str, Any]) -> list[str]:
    notes: list[str] = []
    method = {"least_squares": "最小二乘",
              "alternating_least_squares": "交替最小二乘"}.get(
        str(ident.get("method") or ""), str(ident.get("method") or ""))
    if method:
        n = ident.get("n_samples")
        notes.append(f"参数辨识：{method}"
                     + (f"，{n} 个样本" if n else ""))
    if ident.get("cop_r2") is not None:
        notes.append(f"COP 拟合 R²={float(ident['cop_r2']):.4f}")
    if ident.get("power_rel_rmse") is not None:
        notes.append(f"训练相对 RMSE={float(ident['power_rel_rmse']):.4f}")
    clipped = ident.get("clipped_coefficients") or []
    if clipped:
        notes.append(f"系数被裁剪到物理合法范围：{', '.join(map(str, clipped))}")
    return notes


def _physics_v1(doc: Mapping[str, Any], equations: tuple[str, ...],
                inputs: dict[str, str]) -> ModelDoc:
    params = doc.get("parameters") or {}
    cop = params.get("cop_coefficients") or {}
    c = {name: (cop.get(name) or {}).get("value") for name in ("c0", "c1", "c2", "c3")}
    ident = doc.get("identification") or {}
    return ModelDoc(
        kind="physics_v1", format=PHYSICS_V1_FORMAT,
        summary="能量平衡物理模型 v1：Q=m·Cp·ΔT，COP 线性回归，P=Q/COP",
        equations=equations,
        substituted=(f"COP = {_fmt_num(c['c0'])} + {_fmt_num(c['c1'])}·T_chws "
                     f"+ {_fmt_num(c['c2'])}·T_cws + {_fmt_num(c['c3'])}·PLR",),
        latex=(r"Q = \frac{\rho\, f_{chw}}{3600}\, C_p\, (T_{ret} - T_{sup})",
               r"COP = c_0 + c_1 T_{chws} + c_2 T_{cws} + c_3\, PLR",
               r"P = Q \,/\, COP"),
        params=tuple(_param_rows(params)),
        inputs=inputs,
        notes=tuple(_identification_notes(ident)),
    )


def _physics_v2(doc: Mapping[str, Any], equations: tuple[str, ...],
                inputs: dict[str, str]) -> ModelDoc:
    params = doc.get("parameters") or {}
    curves = params.get("curves") or {}
    norm = params.get("input_normalization") or {}
    ident = doc.get("identification") or {}

    def _curve_line(name: str, terms: tuple[str, ...],
                    template: str) -> str:
        coefs = {t: ((curves.get(name) or {}).get(t) or {}).get("value")
                 for t in terms}
        return template.format(**{t: _fmt_num(coefs[t]) for t in terms})

    substituted = [
        _curve_line("capft", ("const", "t1", "t1_sq", "t2", "t2_sq", "t1_t2"),
                    "CAPFT = {const} + {t1}·u_chws + {t1_sq}·u_chws² "
                    "+ {t2}·u_cond + {t2_sq}·u_cond² + {t1_t2}·u_chws·u_cond"),
        _curve_line("eirft", ("const", "t1", "t1_sq", "t2", "t2_sq", "t1_t2"),
                    "EIRFT = {const} + {t1}·u_chws + {t1_sq}·u_chws² "
                    "+ {t2}·u_cond + {t2_sq}·u_cond² + {t1_t2}·u_chws·u_cond"),
        _curve_line("eirfplr", ("const", "plr", "plr_sq"),
                    "EIRFPLR = {const} + {plr}·PLR + {plr_sq}·PLR²"),
    ]
    for key, entry in norm.items():
        if isinstance(entry, Mapping):
            substituted.append(
                f"{key}：u = (T − {_fmt_num(entry.get('center'))}) / "
                f"{_fmt_num(entry.get('scale'))}")
    notes = _identification_notes(ident)
    proxy = (ident.get("condenser_proxy") or {})
    if proxy.get("selected"):
        notes.append(f"冷凝侧温度代理：{proxy['selected']}（按辨识残差选出）")
    return ModelDoc(
        kind="physics_v2", format=PHYSICS_V2_FORMAT,
        summary="能量平衡物理模型 v2（DOE-2 三曲线）：CAPFT / EIRFT / EIRFPLR",
        equations=equations,
        substituted=tuple(substituted),
        latex=(r"Q = \frac{\rho\, f_{chw}}{3600}\, C_p\, (T_{ret} - T_{sup})",
               r"PLR = \frac{Q}{Q_{rated} \cdot units \cdot CAPFT}",
               r"P = P_{rated} \cdot units \cdot PLR \cdot EIRFT \cdot EIRFPLR"),
        params=tuple(_param_rows(params)),
        inputs=inputs,
        notes=tuple(notes),
    )


# ---------------------------------------------------------------- 系统辨识族


def _gordon_ng(meta: Mapping[str, Any]) -> ModelDoc:
    p = meta.get("parameters") or {}
    inputs = {str(k): str(v) for k, v in (meta.get("inputs") or {}).items()}
    notes = ["单调性天然满足：∂P/∂Q_e>0、∂P/∂T_ci>0、∂P/∂T_ei<0"]
    if meta.get("n_train"):
        notes.append(f"辨识样本 {meta['n_train']} 个")
    if meta.get("q_floor") is not None:
        notes.append(f"负荷下限 q_floor={_fmt_num(meta['q_floor'])} kW")
    return ModelDoc(
        kind="gordon_ng", format=GN_FORMAT,
        summary="Gordon-Ng 熵产模型：能量平衡 + 熵平衡 + 换热热阻，三参数闭式解",
        equations=("T_e = T_ei − Q_e·R_e　（蒸发温度随换热逼近温差下降）",
                   "K = Q_e / T_e + ΔS_int",
                   "Q_c = T_ci·K / (1 − R_c·K)　（冷凝排热）",
                   "P = Q_c − Q_e　（压缩机功率）"),
        latex=(r"T_e = T_{ei} - Q_e\, R_e",
               r"K = \frac{Q_e}{T_e} + \Delta S_{int}",
               r"Q_c = \frac{T_{ci}\, K}{1 - R_c\, K}",
               r"P = Q_c - Q_e"),
        params=(ParamRow("r_evaporator", p.get("r_evaporator"), "K/kW",
                         note="蒸发器换热热阻"),
                ParamRow("r_condenser", p.get("r_condenser"), "K/kW",
                         note="冷凝器换热热阻"),
                ParamRow("delta_s_internal", p.get("delta_s_internal"), "kW/K",
                         note="内部熵产（不可逆损失）")),
        inputs=inputs,
        notes=tuple(notes),
    )


def _eps_ntu(meta: Mapping[str, Any]) -> ModelDoc:
    p = meta.get("parameters") or {}
    inputs = {str(k): str(v) for k, v in (meta.get("inputs") or {}).items()}
    notes = []
    if meta.get("n_train"):
        notes.append(f"辨识样本 {meta['n_train']} 个")
    if meta.get("n_units"):
        notes.append(f"按 `{meta['n_units']}` 台数缩放 UA")
    return ModelDoc(
        kind="eps_ntu", format=NTU_FORMAT,
        summary="ε-NTU 换热器模型：逆流效能-传热单元数，两参数（UA0、β）",
        equations=("C = f·ρ·Cp/3600　（两侧热容流率 kW/K）",
                   "UA = units·UA0 / (f_hot^(−0.8) + β·f_cold^(−0.8))",
                   "NTU = UA/C_min，C_r = C_min/C_max",
                   "ε = (1 − e^(−NTU·(1−C_r))) / (1 − C_r·e^(−NTU·(1−C_r)))",
                   "Q = ε·C_min·(T_hot_in − T_cold_in)"),
        latex=(r"UA = \frac{units \cdot UA_0}{f_{hot}^{-0.8} + \beta\, f_{cold}^{-0.8}}",
               r"\varepsilon = \frac{1 - e^{-NTU(1-C_r)}}{1 - C_r\, e^{-NTU(1-C_r)}}",
               r"Q = \varepsilon\, C_{min}\, (T_{hot,in} - T_{cold,in})"),
        params=(ParamRow("ua0", p.get("ua0"), "kW/K",
                         note="基准传热能力（单位流量下）"),
                ParamRow("beta", p.get("beta"), "1",
                         note="冷侧热阻占比权重")),
        inputs=inputs,
        notes=tuple(notes),
    )


# ---------------------------------------------------------------- 混合与实验室


def _booster_importance(path: Path) -> dict[str, float]:
    """XGBoost 特征重要性（gain 占比 %，降序 Top N）。文件缺失/损坏返回 {}。

    展示失败不该炸页面——重要性是附加信息，不是模型本体。
    """
    if not path.is_file():
        return {}
    try:
        import xgboost as xgb

        booster = xgb.Booster()
        booster.load_model(str(path))
        scores = booster.get_score(importance_type="gain")
    except Exception:
        return {}
    total = sum(float(v) for v in scores.values())
    if total <= 0:
        return {}
    ranked = sorted(((str(k), float(v) / total * 100.0)
                     for k, v in scores.items()),
                    key=lambda kv: kv[1], reverse=True)
    return dict(ranked[:TOP_IMPORTANCE])


def _hybrid(directory: Path, meta: Mapping[str, Any]) -> ModelDoc:
    base_fmt = str(meta.get("base_format") or "")
    base: ModelDoc | None = None
    if base_fmt in ("", PHYSICS_V1_FORMAT, PHYSICS_V2_FORMAT):
        # 旧包没有 base_format；能量平衡族真源是 params.yaml
        if (directory / "params.yaml").is_file():
            base = _physics_family(directory)
    else:
        base_path = directory / "base_model.json"
        if base_path.is_file():
            try:
                base_doc = json.loads(base_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                base_doc = None
            if base_doc:
                if base_fmt == GN_FORMAT:
                    base = _gordon_ng(base_doc)
                elif base_fmt == NTU_FORMAT:
                    base = _eps_ntu(base_doc)

    xgb_params = {str(k): v for k, v in (meta.get("xgb_params") or {}).items()}
    importance = _booster_importance(directory / "booster.json")
    n_trees = xgb_params.get("n_estimators")
    base_name = base.summary.split("：", 1)[0] if base else (base_fmt or "物理主干")
    summary = (f"残差混合：{base_name} + XGBoost 残差修正"
               f"（{n_trees or '?'} 棵树，深度 ≤ {xgb_params.get('max_depth', '?')}）")
    equations = tuple(base.equations if base else ()) + (
        "z_i = (x_i − mean_i)/std_i　（残差特征标准化）",
        "ŷ = ŷ_主干 + Σ_t g_t(z)　（t = 1..n_trees，g_t 为回归树）")
    latex = tuple(base.latex if base else ()) + (
        r"\hat{y} = f_{base}(x) + \sum_{t=1}^{T} g_t(z)",)
    return ModelDoc(
        kind="hybrid", format=HYBRID_FORMAT, summary=summary,
        equations=equations,
        substituted=tuple(base.substituted if base else ()),
        latex=latex,
        params=tuple(base.params if base else ()),
        inputs=dict(base.inputs if base else {}),
        notes=tuple(base.notes if base else ()),
        base=base,
        xgb={"params": xgb_params,
             "monotone_constraints": {str(k): int(v) for k, v in
                                      (meta.get("monotone_constraints")
                                       or {}).items()},
             "seed": meta.get("seed"),
             "n_trees": n_trees,
             "importance": importance},
        scaler=dict(meta.get("scaler") or {}),
    )


def hybrid_flow_spec(doc: ModelDoc) -> tuple[list[tuple[float, float, str]],
                                             list[tuple[int, int]]]:
    """hybrid 组合框图的节点与连线（页面 plotly 与报告 matplotlib 共用）。

    节点文本用 "\\n" 分行；plotly 渲染层自行转成 <br>。
    """
    base_label = (doc.base.summary.split("：", 1)[0] if doc.base else "物理主干")
    n_trees = (doc.xgb or {}).get("n_trees") or "?"
    nodes = [(0.0, 1.0, "输入特征 x"),
             (1.3, 1.0, f"物理主干\n{base_label}"),
             (2.5, 1.0, "Σ"),
             (3.5, 1.0, "预测 ŷ"),
             (1.9, 0.0, f"XGBoost 残差\n{n_trees} 棵树")]
    return nodes, [(0, 1), (1, 2), (4, 2), (2, 3)]


def _lab(directory: Path, meta: Mapping[str, Any]) -> ModelDoc:
    fmt = str(meta.get("format") or "")
    columns = [str(c) for c in meta.get("columns") or []]
    params = {str(k): v for k, v in (meta.get("params") or {}).items()}
    booster_file = str(meta.get("booster_file") or "booster.json")
    importance = _booster_importance(directory / booster_file)
    source_path = directory / "lab_source.py"
    lab_source = None
    if source_path.is_file():
        try:
            lab_source = source_path.read_text(encoding="utf-8")
        except OSError:
            lab_source = None
    return ModelDoc(
        kind="lab", format=fmt,
        summary=f"模型实验室模块（{fmt}）：GBDT 直接回归，{len(columns)} 个输入列",
        equations=("ŷ = Σ_t g_t(x)　（梯度提升树，模型实验室模块）",),
        latex=(r"\hat{y} = \sum_{t=1}^{T} g_t(x)",),
        xgb={"params": params, "n_trees": params.get("n_estimators"),
             "monotone_constraints": {}, "seed": None,
             "importance": importance},
        lab_source=lab_source,
        raw=dict(meta),
    )
