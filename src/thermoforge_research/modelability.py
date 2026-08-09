"""可建模性报告（gap-analysis.md G3、data-survey.md §F1/F3/F6）。

统计层质量报告（data-contract §8 的 6 类检查）只看单变量分布，探查发现
的三个阻断级问题一个都发现不了。本模块把探查阶段一次性做过的**语义层**
检查固化为系统能力：输入 = 数据集 revision + Research Goal（target +
candidate_inputs 白名单），输出结构化报告，结论 FAIL（存在 blocker）时
状态机门禁不允许进入建模（G3：MODELABILITY_ASSESSMENT）。

五类检查（分级 blocker / warning / info）：

1. **派生链分析**（derivation_chain）：候选输入或目标为 TFOM derived
   属性时展开 expression 上游闭包；候选链含目标、或目标链含候选，即
   循环论证（blocker）。与 whitelist.py 协同：whitelist 做准入拦截，
   本报告给出完整依赖链证据。
2. **同源检测**（same_origin）：候选与目标 |r| > 0.98（可配）或序列
   完全相同 → blocker（F1：power~current_percent r=0.9955）；候选两两
   完全相同/相关性≈1 → warning（冗余）。
3. **设备区分度**（device_diversity）：同类实例同名属性逐点相同或
   |r|≥0.9999 → warning（F3：四台冷机温度完全相同）。
4. **物理自洽**（physical_plausibility）：由数据派生的 COP/ΔT 等
   无量纲量落在参数化范围内；范围来源写明，内置默认不写死（F6 已撤销：
   高温离心机 COP 9~11 合理，默认 cop_range=(0,40) 可配置）。
5. **有效工况覆盖**（operating_coverage）：目标可用样本占比（blocker
   下限可配）、时间覆盖、负荷分档（目标三分位低/中/高档样本数）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from thermoforge_data.importer import TfomRegistry, default_registry, expression_dependencies

LEVELS = ("info", "warning", "blocker")

# 属性名别名（不同物模型对同一物理量的命名）
_FLOW_PROPS = ("chw_flow", "evap_chw_flow")
_CHW_SUPPLY_PROPS = ("chw_supply_temp", "evap_chw_supply_temp")
_CHW_RETURN_PROPS = ("chw_return_temp", "evap_chw_return_temp")
_CW_SUPPLY_PROPS = ("cw_supply_temp", "condenser_supply_temp")
_CW_RETURN_PROPS = ("cw_return_temp", "condenser_return_t")

_RHO = 998.0   # kg/m3（定值物性，同 physics.py [草案]）
_CP = 4.186    # kJ/(kg·K)


@dataclass(frozen=True)
class ModelabilityConfig:
    """可配置阈值（参考阈值，需用真实数据标定——implementation-notes §14）。"""

    same_origin_corr: float = 0.98       # 候选-目标 |r| 上限（blocker）
    redundant_corr: float = 0.9999       # 候选两两冗余（warning）
    device_corr: float = 0.9999          # 设备间序列相关性（warning）
    min_target_fraction: float = 0.10    # 目标可用占比下限（blocker）
    warn_target_fraction: float = 0.30   # 目标可用占比告警线（warning）
    min_bin_samples: int = 30            # 负荷分档每档最小样本数（warning）
    ratio_outlier_tol: float = 0.05      # 物理量超范围样本占比容忍（warning）
    # 同源检测只在目标有效工况（非零）样本上评估：停机工况下输入与目标
    # 同为零会制造虚假高相关（功率类目标的零即「未运行」）
    same_origin_positive_target: bool = True
    # 内置物理可能范围（参数化；F6 教训：不得写死常规冷机口径）
    cop_range: tuple[float, float] = (0.0, 40.0)
    delta_t_range: tuple[float, float] = (0.0, 20.0)  # K
    quantity_ranges: Mapping[str, tuple[float, float]] = field(
        default_factory=dict)  # 逐量覆盖，如 {"cop_estimate": (2.0, 25.0)}


def _check(name: str, level: str, passed: bool, summary: str,
           evidence: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if level not in LEVELS:
        raise ValueError(f"非法分级: {level!r}")
    return {"name": name, "level": level, "passed": bool(passed),
            "summary": summary, "evidence": dict(evidence or {})}


def _corr(a: np.ndarray, b: np.ndarray) -> float | None:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return None
    x, y = a[mask], b[mask]
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _identical(a: np.ndarray, b: np.ndarray) -> bool:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 1:
        return False
    return bool(np.array_equal(a[mask], b[mask]))


# ---------------------------------------------------------------- 1. 派生链


def _upstream_closure(model, prop: str) -> tuple[list[str], bool]:
    """TFOM expression 上游闭包。返回 (有序依赖链, 是否有环)。"""
    chain: list[str] = []
    visiting: set[str] = set()
    visited: set[str] = set()
    cyclic = False

    def walk(p: str) -> None:
        nonlocal cyclic
        if p in visiting:
            cyclic = True
            return
        if p in visited:
            return
        visiting.add(p)
        tfom_prop = model.properties.get(p)
        if tfom_prop is not None and tfom_prop.expression:
            for dep in sorted(expression_dependencies(tfom_prop.expression)):
                if dep not in chain:
                    chain.append(dep)
                walk(dep)
        visiting.discard(p)
        visited.add(p)

    walk(prop)
    return chain, cyclic


def check_derivation_chain(
    target: str,
    candidate_inputs: Sequence[str],
    *,
    object_model: str,
    registry: TfomRegistry,
) -> dict[str, Any]:
    """派生链分析：候选/目标的 expression 上游闭包是否互相包含（循环）。"""
    model = registry.get(object_model)
    if model is None:
        return _check("derivation_chain", "warning", False,
                      f"对象模型未注册，无法做派生链分析: {object_model}")
    findings: list[dict[str, Any]] = []
    level = "info"

    t_prop = model.properties.get(target)
    target_chain: list[str] = []
    if t_prop is not None and t_prop.expression:
        target_chain, cyclic = _upstream_closure(model, target)
        if cyclic:
            findings.append({"kind": "cycle", "detail":
                             f"目标 {target} 的 expression 依赖图有环"})
            level = "blocker"

    for entry in candidate_inputs:
        prop = str(entry).partition(".")[2] if "." in str(entry) else str(entry)
        cand_prop = model.properties.get(prop)
        # 目标为派生量且候选在其上游链中 → 循环（候选本身无需 derived）
        if prop in target_chain:
            findings.append({
                "kind": "circular_target", "candidate": str(entry),
                "chain": [target, *target_chain],
                "detail": f"目标 {target} 由候选输入 {prop} 派生（循环论证）",
            })
            level = "blocker"
            continue
        if cand_prop is None or not cand_prop.expression:
            continue
        chain, cyclic = _upstream_closure(model, prop)
        if target in chain:
            findings.append({
                "kind": "circular_input", "candidate": str(entry),
                "chain": [prop, *chain],
                "detail": f"候选输入 {prop} 由目标 {target} 派生（循环论证）",
            })
            level = "blocker"
        else:
            findings.append({
                "kind": "derived_input", "candidate": str(entry),
                "chain": [prop, *chain],
                "detail": f"候选输入 {prop} 为派生量（准入层应已拦截；链路证据如上）",
            })
            if level == "info":
                level = "warning"
        if cyclic:
            level = "blocker"
    passed = level != "blocker"
    summary = (f"{len(findings)} 项派生发现"
               if findings else "候选输入与目标均无派生依赖交叉")
    return _check("derivation_chain", level, passed, summary,
                  {"findings": findings, "target_chain": target_chain})


# ---------------------------------------------------------------- 2. 同源


def check_same_origin(
    series: Mapping[str, np.ndarray],
    target: str,
    candidate_inputs: Sequence[str],
    *,
    config: ModelabilityConfig,
) -> dict[str, Any]:
    """同源检测：候选-目标相关性高到不合理（blocker）；候选两两冗余（warning）。

    `series`：property_code → 数值序列（范围内对象已按行池化）。
    """
    t = series.get(target)
    if t is None:
        return _check("same_origin", "warning", False,
                      f"目标序列不可用: {target}")
    level = "info"
    target_hits: list[dict[str, Any]] = []
    for entry in candidate_inputs:
        prop = str(entry)
        v = series.get(prop)
        if v is None:
            continue
        if _identical(v, t):
            target_hits.append({"candidate": prop, "reason": "identical",
                                "detail": "与目标序列完全相同"})
            level = "blocker"
            continue
        r = _corr(v, t)
        if r is not None and abs(r) > config.same_origin_corr:
            target_hits.append({"candidate": prop, "r": r,
                                "detail": f"|r|={abs(r):.4f} > "
                                          f"{config.same_origin_corr}，疑似同源（F1）"})
            level = "blocker"

    redundant: list[dict[str, Any]] = []
    props = [str(c) for c in candidate_inputs if series.get(str(c)) is not None]
    for i, a in enumerate(props):
        for b in props[i + 1:]:
            va, vb = series[a], series[b]
            if _identical(va, vb):
                redundant.append({"pair": [a, b], "reason": "identical"})
                if level == "info":
                    level = "warning"
            else:
                r = _corr(va, vb)
                if r is not None and abs(r) >= config.redundant_corr:
                    redundant.append({"pair": [a, b], "r": r})
                    if level == "info":
                        level = "warning"
    summary = (f"同源疑似 {len(target_hits)} 项，冗余对 {len(redundant)} 组"
               if (target_hits or redundant) else "候选与目标相关性均在合理范围")
    return _check("same_origin", level, level != "blocker", summary,
                  {"target_correlations": target_hits,
                   "redundant_pairs": redundant})


# ---------------------------------------------------------------- 3. 设备区分度


def check_device_diversity(
    per_object: Mapping[str, Mapping[str, np.ndarray]],
    *,
    config: ModelabilityConfig,
) -> dict[str, Any]:
    """设备区分度：同类实例同名属性逐点相同 / 相关性≈1 → warning（F3）。

    `per_object`：object_id → {property_code: 序列}（同一 object_model）。
    """
    objects = sorted(per_object)
    if len(objects) < 2:
        return _check("device_diversity", "info", True,
                      "范围内实例数 < 2，无需设备区分度检查",
                      {"objects": objects})
    props = sorted({p for o in objects for p in per_object[o]})
    findings: list[dict[str, Any]] = []
    for prop in props:
        for i, oa in enumerate(objects):
            for ob in objects[i + 1:]:
                va = per_object[oa].get(prop)
                vb = per_object[ob].get(prop)
                if va is None or vb is None:
                    continue
                if _identical(va, vb):
                    findings.append({"property": prop, "objects": [oa, ob],
                                     "reason": "identical"})
                else:
                    r = _corr(va, vb)
                    if r is not None and abs(r) >= config.device_corr:
                        findings.append({"property": prop,
                                         "objects": [oa, ob], "r": r})
    if findings:
        return _check("device_diversity", "warning", True,
                      f"{len(findings)} 组实例-属性无实质差异（留一设备验证"
                      "对这些属性无区分度，F3）",
                      {"findings": findings[:50],
                       "findings_total": len(findings)})
    return _check("device_diversity", "info", True,
                  f"{len(objects)} 个实例间输入均有实质差异",
                  {"objects": objects})


# ---------------------------------------------------------------- 4. 物理自洽


def _first_present(props: Mapping[str, np.ndarray],
                   aliases: Sequence[str]) -> np.ndarray | None:
    for name in aliases:
        if props.get(name) is not None:
            return props[name]
    return None


def check_physical_plausibility(
    series: Mapping[str, np.ndarray],
    target: str,
    *,
    config: ModelabilityConfig,
) -> dict[str, Any]:
    """物理自洽：数据派生的无量纲量是否落在参数化物理可能范围。

    内置量（范围来源均写入 evidence）：
    - `chw_delta_t` / `cw_delta_t`：供回水温差（K），范围 config.delta_t_range；
    - `cop_estimate`：Q = 流量·ρ·Cp·ΔT，COP = Q/目标功率，
      范围 config.cop_range（**参数化默认 (0,40)**——F6 已撤销，高温离心
      工况 COP 9~11 合理，不得写死 5~7）。
    """
    ranges = {
        "chw_delta_t": config.delta_t_range,
        "cw_delta_t": config.delta_t_range,
        "cop_estimate": config.cop_range,
        **dict(config.quantity_ranges),
    }
    flow = _first_present(series, _FLOW_PROPS)
    chw_s = _first_present(series, _CHW_SUPPLY_PROPS)
    chw_r = _first_present(series, _CHW_RETURN_PROPS)
    cw_s = _first_present(series, _CW_SUPPLY_PROPS)
    cw_r = _first_present(series, _CW_RETURN_PROPS)

    quantities: dict[str, np.ndarray] = {}
    if chw_s is not None and chw_r is not None:
        quantities["chw_delta_t"] = chw_r - chw_s
    if cw_s is not None and cw_r is not None:
        quantities["cw_delta_t"] = cw_r - cw_s
    if flow is not None and chw_s is not None and chw_r is not None \
            and series.get(target) is not None:
        q = flow * _RHO / 3600.0 * _CP * (chw_r - chw_s)
        p = series[target]
        mask = (q > 0) & (p > 0) & np.isfinite(q) & np.isfinite(p)
        cop = np.full_like(q, np.nan, dtype=np.float64)
        cop[mask] = q[mask] / p[mask]
        quantities["cop_estimate"] = cop

    if not quantities:
        return _check("physical_plausibility", "info", True,
                      "无可计算的派生物理量（缺少流量/温度对）")
    findings: list[dict[str, Any]] = []
    level = "info"
    for name, values in quantities.items():
        lo, hi = ranges[name]
        finite = values[np.isfinite(values)]
        if not len(finite):
            continue
        out = (finite < lo) | (finite > hi)
        frac = float(out.mean())
        evidence = {
            "range": [lo, hi],
            "range_source": ("config.quantity_ranges"
                             if name in config.quantity_ranges
                             else "内置参数化默认（F6 撤销后不写死常规口径）"),
            "p05": float(np.percentile(finite, 5)),
            "p50": float(np.percentile(finite, 50)),
            "p95": float(np.percentile(finite, 95)),
            "out_of_range_fraction": frac,
        }
        if frac > config.ratio_outlier_tol:
            findings.append({"quantity": name, **evidence})
            level = "warning"
        else:
            findings.append({"quantity": name, **evidence, "ok": True})
    passed = level != "blocker"
    bad = [f for f in findings if not f.get("ok")]
    summary = (f"{len(bad)} 项派生物理量超范围"
               if bad else f"{len(findings)} 项派生物理量均在物理可能范围")
    return _check("physical_plausibility", level, passed, summary,
                  {"quantities": findings})


# ---------------------------------------------------------------- 5. 工况覆盖


def check_operating_coverage(
    series: Mapping[str, np.ndarray],
    target: str,
    timestamps: np.ndarray | None,
    *,
    config: ModelabilityConfig,
) -> dict[str, Any]:
    """有效工况覆盖：目标可用占比 / 时间覆盖 / 负荷分档（低中高档样本数）。"""
    t = series.get(target)
    if t is None or not len(t):
        return _check("operating_coverage", "blocker", False,
                      f"目标序列不可用: {target}")
    usable = np.isfinite(t)
    fraction = float(usable.mean())
    evidence: dict[str, Any] = {
        "rows": int(len(t)),
        "usable_rows": int(usable.sum()),
        "usable_fraction": fraction,
        "zero_fraction": float((t[usable] == 0).mean()) if usable.any() else None,
    }
    if timestamps is not None and len(timestamps) >= 2:
        span_s = float(timestamps[-1] - timestamps[0])
        evidence["time_span_days"] = span_s / 86400.0
    vals = t[usable]
    pos = vals[vals > 0] if (vals > 0).any() else vals
    if len(pos) >= 3:
        edges = np.quantile(pos, [0, 1 / 3, 2 / 3, 1.0])
        bins = []
        for i, label in enumerate(("low", "mid", "high")):
            lo, hi = edges[i], edges[i + 1]
            mask = (vals >= lo) & (vals <= hi if i == 2 else vals < hi)
            bins.append({"bin": label, "range": [float(lo), float(hi)],
                         "n_samples": int(mask.sum())})
        evidence["load_bins"] = bins
    level = "info"
    reasons: list[str] = []
    if fraction < config.min_target_fraction:
        level = "blocker"
        reasons.append(f"目标可用占比 {fraction:.3f} < "
                       f"{config.min_target_fraction}")
    elif fraction < config.warn_target_fraction:
        level = "warning"
        reasons.append(f"目标可用占比 {fraction:.3f} 偏低")
    for b in evidence.get("load_bins", []):
        if b["n_samples"] < config.min_bin_samples and level == "info":
            level = "warning"
            reasons.append(f"{b['bin']} 档样本 {b['n_samples']} < "
                           f"{config.min_bin_samples}")
    summary = ("；".join(reasons) if reasons
               else f"目标可用占比 {fraction:.1%}，负荷三档覆盖充足")
    return _check("operating_coverage", level, level != "blocker",
                  summary, evidence)


# ---------------------------------------------------------------- 报告装配


def build_modelability_report(
    vault,
    ref: str,
    *,
    target: str,
    candidate_inputs: Sequence[str],
    object_model: str | None = None,
    registry: TfomRegistry | None = None,
    config: ModelabilityConfig | None = None,
) -> dict[str, Any]:
    """装配完整可建模性报告。verdict = FAIL 当且仅当存在 blocker 级检查。"""
    config = config or ModelabilityConfig()
    registry = registry or default_registry()
    info = vault.resolve(ref)
    table = vault.load_data(ref)
    objects = vault.load_objects(ref)

    if object_model is None:
        # 取 target 所在变量的 object_model
        variables = vault.load_variables(ref)
        hits = {v["object_id"]: v for v in variables
                if v["property_code"] == target}
        model_of = {o["object_id"]: o["object_model_id"] for o in objects}
        models = {model_of[o] for o in hits if o in model_of}
        object_model = sorted(models)[0] if models else None

    scope_objects = sorted(
        o["object_id"] for o in objects
        if o["object_model_id"] == object_model
    )
    cols = {name: _to_float_array(table, name)
            for name in table.column_names if name != "timestamp"}
    ts = _timestamps_epoch(table)

    # 对齐池化：以「拥有目标列的对象」为轴逐对象拼接；缺列对象用 NaN
    # 填充保持块对齐（物理量按行位置配对）。同源检测按对象分别评估，
    # 避免跨对象池化稀释（实测：单机 r=0.995，混池后仅 0.57）。
    target_objects = [o for o in scope_objects if f"{o}.{target}" in cols]
    if not target_objects:
        target_objects = scope_objects
    n_rows = table.num_rows

    def pool(prop: str, obj: str | None = None) -> np.ndarray | None:
        if obj is not None:
            arr = cols.get(f"{obj}.{prop}")
            return arr
        parts = []
        for o in target_objects:
            arr = cols.get(f"{o}.{prop}")
            parts.append(arr if arr is not None
                         else np.full(n_rows, np.nan))
        if not parts:
            return None
        return np.concatenate(parts) if len(parts) > 1 else parts[0]

    series: dict[str, np.ndarray] = {}
    for entry in [target, *candidate_inputs]:
        text = str(entry)
        if "." in text:
            obj, _, prop = text.partition(".")
            arr = pool(prop, obj)
        else:
            arr = pool(text)
        if arr is not None:
            series[text] = arr

    per_object = {
        o: {p: cols[f"{o}.{p}"] for p in
            [target, *[str(c) for c in candidate_inputs if "." not in str(c)]]
            if f"{o}.{p}" in cols}
        for o in scope_objects
    }

    # 同源检测：逐对象评估后取最重级别（两层候选为外部驱动量，对每个
    # 目标对象都同一序列）；仅在目标有效工况（非零）样本上评估
    origin_per_object: list[dict[str, Any]] = []
    for o in target_objects:
        series_o: dict[str, np.ndarray] = {}
        for entry in [target, *candidate_inputs]:
            text = str(entry)
            if "." in text:
                obj, _, prop = text.partition(".")
                arr = cols.get(f"{obj}.{prop}")
            else:
                arr = cols.get(f"{o}.{text}")
            if arr is None:
                continue
            if config.same_origin_positive_target:
                t_o = cols.get(f"{o}.{target}")
                if t_o is not None:
                    active = np.isfinite(t_o) & (t_o > 0)
                    arr = np.where(active, arr, np.nan)
            series_o[text] = arr
        out = check_same_origin(series_o, target, candidate_inputs,
                                config=config)
        out["object"] = o
        origin_per_object.append(out)
    origin_check = _merge_per_object("same_origin", origin_per_object)

    checks = [
        check_derivation_chain(target, candidate_inputs,
                               object_model=object_model or "",
                               registry=registry),
        origin_check,
        check_device_diversity(per_object, config=config),
        check_physical_plausibility(series, target, config=config),
        check_operating_coverage(series, target, ts, config=config),
    ]
    blockers = [c["name"] for c in checks if c["level"] == "blocker"]
    warnings = [c["name"] for c in checks if c["level"] == "warning"]
    return {
        "kind": "modelability_report",
        "dataset_ref": info.ref,
        "content_sha256": info.content_sha256,
        "object_model": object_model,
        "scope_objects": scope_objects,
        "target": target,
        "candidate_inputs": [str(c) for c in candidate_inputs],
        "verdict": "FAIL" if blockers else "PASS",
        "blockers": blockers,
        "warnings": warnings,
        "checks": checks,
        "config": {
            "same_origin_corr": config.same_origin_corr,
            "redundant_corr": config.redundant_corr,
            "device_corr": config.device_corr,
            "min_target_fraction": config.min_target_fraction,
            "same_origin_positive_target": config.same_origin_positive_target,
            "cop_range": list(config.cop_range),
            "delta_t_range": list(config.delta_t_range),
        },
    }


def _merge_per_object(
    name: str, results: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """逐对象检查结果合并：级别取最重，证据按对象归属。"""
    rank = {lv: i for i, lv in enumerate(LEVELS)}
    level = max((r["level"] for r in results), key=lambda lv: rank[lv],
                default="info")
    per_object = [
        {"object": r.get("object"), "level": r["level"],
         "summary": r["summary"], "evidence": r["evidence"]}
        for r in results
    ]
    hits = [r for r in results if r["level"] != "info"]
    summary = (f"{len(hits)}/{len(results)} 个对象存在发现（最重 {level}）"
               if hits else "候选与目标相关性均在合理范围")
    return _check(name, level, level != "blocker", summary,
                  {"per_object": per_object})


def _to_float_array(table, name: str) -> np.ndarray:
    return np.array([float(v) if v is not None else np.nan
                     for v in table.column(name).to_pylist()],
                    dtype=np.float64)


def _timestamps_epoch(table) -> np.ndarray | None:
    if "timestamp" not in table.column_names:
        return None
    ts = table.column("timestamp").to_pylist()
    if not ts:
        return None
    return np.array([t.timestamp() for t in ts], dtype=np.float64)
