"""Research diagnostics computed only from explicitly permitted metric surfaces."""

from __future__ import annotations

import math
from typing import Any, Mapping

from thermoforge_v2.evidence_projection import numeric_metrics


def _surface(value: Any) -> dict[str, Any]:
    surface = value if isinstance(value, Mapping) else {}
    metrics = numeric_metrics(surface.get("metrics"))
    count = surface.get("n_samples")
    result: dict[str, Any] = {"status": "available" if metrics else "unavailable", "metrics": metrics}
    if isinstance(count, (int, float)) and not isinstance(count, bool) and math.isfinite(count) and count >= 0:
        result["n_samples"] = int(count)
    if not metrics:
        result["reason"] = "本次实验未提供该面可用指标；不能以其他评价面替代。"
    return result


def research_diagnostics(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Do not load artifacts or infer missing training/physics measurements."""
    surfaces = summary.get("surfaces")
    surfaces = surfaces if isinstance(surfaces, Mapping) else {}
    train, validate = _surface(surfaces.get("train")), _surface(surfaces.get("validate"))
    if "train" not in surfaces:
        train["reason"] = "当前内核未输出训练集指标，无法据此判断拟合差距。"
    nmbe = validate["metrics"].get("NMBE")
    bias: dict[str, Any] = {"status": "unavailable", "reason": "验证面没有可用的 NMBE。"}
    if nmbe is not None:
        bias = {"status": "available", "nmbe": nmbe,
                "direction": "normalized_positive" if nmbe > 0 else "normalized_negative" if nmbe < 0 else "zero",
                "convention": "NMBE=sum(y-y_pred)/(n*mean(y))，采用比率；目标均值为正时，正值表示低估，负值表示高估。"}
    differences = {key: validate["metrics"][key] - value
                   for key, value in train["metrics"].items() if value is not None
                   and validate["metrics"].get(key) is not None
                   and math.isfinite(validate["metrics"][key] - value)}
    gap: dict[str, Any] = {"status": "available" if differences else "unavailable",
                           "validate_minus_train": differences,
                           "convention": "同名指标的验证值减训练值；差值本身不证明过拟合，R2 与误差指标的优劣方向不同。"}
    if not differences:
        gap["reason"] = "缺少训练与验证面同名有效指标，未计算差距。"
    return {"train": train, "validate": validate, "bias": bias, "generalization_gap": gap,
            "physics": {"status": "unavailable", "reason": "当前内核未提供独立的训练/验证物理检查结果。"}}
