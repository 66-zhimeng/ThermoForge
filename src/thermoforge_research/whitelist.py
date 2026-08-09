"""候选输入白名单的机器校验（DD-16、data-survey.md §F1/F4 建议 4）。

DD-16：`candidate_inputs` 是封闭白名单——建模只允许使用其中列出的变量
计算 target。本模块把这条契约落成两个机器可判定的检查：

1. **Goal 级**（`check_candidate_inputs`）：白名单条目必须存在于数据
   版本，且 `source_kind` 不得为 `derived`——派生量不是独立测量，
   与目标同源的派生输入构成循环论证（§F1：`load = current_percent
   × 9672 / 100`，用它预测功率能拿 MAPE 4.35% 的虚假精度）。
2. **实验级**（`check_view_within_whitelist`）：Dataset View 的
   features 必须是白名单子集、target 必须等于 goal.target。两层的
   机器校验保证「禁用变量」无法经任何路径进入特征。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def _split_entry(entry: str) -> tuple[str | None, str]:
    """`object.property` → (object, property)；裸 property → (None, property)。"""
    obj, sep, prop = str(entry).partition(".")
    return (obj, prop) if sep else (None, obj)


def check_candidate_inputs(
    candidate_inputs: Sequence[str],
    *,
    target: str,
    variables: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Goal 级白名单校验。返回违规列表（空 = 通过）。

    `variables`：数据版本的变量元数据（variable_id / property_code /
    source_kind），来自 `DataVault.load_variables`。
    """
    by_id = {str(v["variable_id"]): v for v in variables}
    by_prop: dict[str, list[Mapping[str, Any]]] = {}
    for v in variables:
        by_prop.setdefault(str(v["property_code"]), []).append(v)

    violations: list[str] = []
    if not any(str(v["property_code"]) == str(target) for v in variables):
        violations.append(f"target 在数据版本中不存在: {target}")
    for entry in candidate_inputs:
        obj, prop = _split_entry(str(entry))
        if obj is not None:
            records = [by_id[str(entry)]] if str(entry) in by_id else []
            label = str(entry)
        else:
            records = by_prop.get(prop, [])
            label = prop
        if not records:
            violations.append(f"候选输入在数据版本中不存在: {label}")
            continue
        for rec in records:
            if str(rec.get("source_kind")) == "derived":
                violations.append(
                    f"候选输入为派生量（非独立测量，循环论证风险 §F1）: "
                    f"{rec['variable_id']}"
                )
    return violations


def check_view_within_whitelist(
    *,
    features: Sequence[str],
    view_target: str | None,
    view_objects: Sequence[str],
    candidate_inputs: Sequence[str],
    goal_target: str,
) -> list[str]:
    """实验级白名单校验（DD-16）。返回违规列表（空 = 通过）。

    - 裸 property 条目：允许该 property（任意在范围对象）。
    - `object.property` 条目：仅允许该对象的该 property（跨设备取数）。
    """
    plain: set[str] = set()
    by_object: dict[str, set[str]] = {}
    for entry in candidate_inputs:
        obj, prop = _split_entry(str(entry))
        if obj is None:
            plain.add(prop)
        else:
            by_object.setdefault(obj, set()).add(prop)

    objects = {str(o) for o in view_objects}
    allowed = set(plain)
    for obj in objects:
        allowed |= by_object.get(obj, set())

    violations = [
        f"特征不在 candidate_inputs 白名单内（DD-16）: {f}"
        for f in features
        if str(f) not in allowed
    ]
    if view_target is not None and str(view_target) != str(goal_target):
        violations.append(
            f"View 目标 {view_target} 与 Research Goal target {goal_target} 不一致"
        )
    return violations
