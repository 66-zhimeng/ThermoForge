"""预处理规则库与工具化测试（I-49）。

覆盖：规则 Schema 正反例、三个规则类型、审批门禁、规则集存储、
工具信封（propose/approve/apply/list）。真实工作簿等价性验证见
tests/test_preprocess_equivalence.py（slow）。
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest
from openpyxl import Workbook
from pydantic import ValidationError

from thermoforge_data.legacy import LegacySheet
from thermoforge_data.preprocess import (
    TFPP_RULE_NOT_APPROVED,
    PreprocessError,
    PreprocessRule,
    RuleSet,
    RuleStore,
    apply_ruleset,
)
from thermoforge_research.tools import (
    ToolContext,
    tf_preprocess_apply,
    tf_preprocess_approve,
    tf_preprocess_list,
    tf_preprocess_propose,
)

BASE = datetime(2025, 1, 1, 0, 0, 0)


def make_sheet(name, instances, model, props, rows) -> LegacySheet:
    """构造 4 行多级表头的小号内存表。"""
    r1, r2, r3, r4 = ["设备类"], ["设备实例"], ["物模型"], ["物模型属性"]
    for inst in instances:
        for k, p in enumerate(props):
            r1.append(name if k == 0 else None)
            r2.append(inst if k == 0 else None)
            r3.append(model if k == 0 else None)
            r4.append(p)
    return LegacySheet(name=name, header_rows=[r1, r2, r3, r4], data_rows=rows)


def fill_sheets() -> dict[str, LegacySheet]:
    t = [BASE + timedelta(hours=i) for i in range(3)]
    ct = make_sheet(
        "冷却塔", ["ct_01", "ct_02"], "cooling_tower",
        ["supply_t", "return_t", "supply_p", "return_p", "instant_flow"],
        [(ts,) + (None,) * 10 for ts in t],
    )
    cw = make_sheet(
        "冷却水总管", ["cw_A1"], "cooling_water_header",
        ["f", "t_supply", "t_return"],
        [(t[0], 900.0, 30.0, 35.0),
         (t[1], 800.0, 31.0, 36.0),
         (t[2], 700.0, 32.0, 37.0)],
    )
    fans = make_sheet(
        "冷却塔风机", ["ct_01_f_01", "ct_02_f_01"], "cooling_tower_fan",
        ["status_run"],
        [(t[0], 1, 0), (t[1], 1, 1), (t[2], 0, 0)],
    )
    return {"冷却塔": ct, "冷却水总管": cw, "冷却塔风机": fans}


FILL_PARAMS = {
    "source_sheet": "冷却水总管",
    "source_instance": "cw_A1",
    "value_map": {"supply_t": "t_supply", "return_t": "t_return"},
    "divide_prop": "instant_flow",
    "divide_source_prop": "f",
    "running_sheet": "冷却塔风机",
    "running_prop": "status_run",
    "leave_empty": ["supply_p", "return_p"],
    "idle_value": 0,
}


def fill_rule(**over) -> PreprocessRule:
    doc = {"rule_id": "fill-ct", "rule_type": "fill_from_header_divide_by_count",
           "sheet": "冷却塔", "params": FILL_PARAMS}
    doc.update(over)
    return PreprocessRule(**doc)


# ---------------------------------------------------------------- Schema 正反例


def test_rule_schema_valid():
    rule = fill_rule()
    assert rule.status == "proposed"
    assert rule.proposer == "agent"


def test_rule_schema_rejects_unknown_type():
    with pytest.raises(ValidationError):
        PreprocessRule(rule_id="x", rule_type="drop_columns", sheet="冷却塔")


def test_rule_schema_rejects_bad_rule_id():
    with pytest.raises(ValidationError):
        PreprocessRule(rule_id="bad id!", rule_type="fix_header_labels",
                       sheet="冷却塔")


def test_rule_params_validated_at_propose():
    rule = fill_rule(params={"source_sheet": "冷却水总管"})  # 缺必需参数
    from thermoforge_data.preprocess import validate_rule_params
    with pytest.raises(ValidationError):
        validate_rule_params(rule)


def test_ruleset_content_hash_ignores_approval_metadata():
    rs1 = RuleSet(ruleset_id="RS", version=1, rules=[fill_rule()])
    approved = fill_rule(status="approved", proposer="human")
    rs2 = RuleSet(ruleset_id="RS", version=1, rules=[approved])
    assert rs1.content_hash() == rs2.content_hash()  # 审批不改内容指纹
    changed = RuleSet(ruleset_id="RS", version=1,
                      rules=[fill_rule(params={**FILL_PARAMS, "idle_value": 1})])
    assert changed.content_hash() != rs1.content_hash()


def test_ruleset_duplicate_rule_id_rejected():
    with pytest.raises(ValidationError):
        RuleSet(ruleset_id="RS", version=1,
                rules=[fill_rule(), fill_rule()])


# ---------------------------------------------------------------- 三个规则类型


def test_fill_from_header_divide_by_count():
    sheets = fill_sheets()
    rs = RuleSet(ruleset_id="RS", version=1, rules=[fill_rule()])
    (ex,) = apply_ruleset(sheets, rs)
    assert not ex.skipped
    assert ex.input_sha256 != ex.output_sha256
    rows = sheets["冷却塔"].data_rows
    # t0：ct_01 运行（唯一运行塔）→ f/1；ct_02 停机 → 0
    assert rows[0][1] == 30.0    # ct_01.supply_t = t_supply
    assert rows[0][2] == 35.0    # ct_01.return_t
    assert rows[0][5] == 900.0   # ct_01.instant_flow = 900 / 1
    assert rows[0][6] == 0       # ct_02.supply_t 停机填 0
    assert rows[0][10] == 0      # ct_02.instant_flow
    assert rows[0][3] is None    # supply_p 恒留空
    # t1：两塔运行 → f/2
    assert rows[1][5] == 400.0
    assert rows[1][10] == 400.0
    # t2：全停 → 0
    assert rows[2][1] == 0
    assert rows[2][5] == 0


def test_fill_round_decimals():
    sheets = fill_sheets()
    rule = fill_rule(params={**FILL_PARAMS, "round_decimals": 1})
    apply_ruleset(sheets, RuleSet(ruleset_id="RS", version=1, rules=[rule]))
    rows = sheets["冷却塔"].data_rows
    assert rows[1][5] == 400.0  # 800/2 精确
    # 构造非整除：t0 f=900/1 精确；改用 7/3 验证 round
    cw = sheets["冷却水总管"]
    cw.data_rows = [(cw.data_rows[0][0], 7.0, 30.0, 35.0), *cw.data_rows[1:]]
    apply_ruleset(sheets, RuleSet(ruleset_id="RS", version=2, rules=[rule]))
    assert sheets["冷却塔"].data_rows[0][5] == round(7.0 / 1, 1)


def test_repair_time_axis():
    t = [BASE + timedelta(hours=i) for i in range(4)]
    target = make_sheet(
        "冷冻水泵", ["chwp_01"], "chilled_water_pump", ["power"],
        [(t[0], 10.0), (t[1], 11.0), (t[1], 99.0), (t[1], 98.0)],
    )
    reference = make_sheet(
        "冷水主机", ["chiller_01"], "chiller", ["power"],
        [(ts, 1.0) for ts in t],
    )
    sheets = {"冷冻水泵": target, "冷水主机": reference}
    rule = PreprocessRule(
        rule_id="fix-axis", rule_type="repair_time_axis", sheet="冷冻水泵",
        params={"reference_sheet": "冷水主机"},
    )
    (ex,) = apply_ruleset(sheets, RuleSet(ruleset_id="RS", version=1,
                                          rules=[rule]))
    rows = sheets["冷冻水泵"].data_rows
    assert len(rows) == 4
    assert [r[0] for r in rows] == t
    assert rows[1][1] == 11.0   # 重复时间戳保留首次出现
    assert rows[2][1] is None   # 缺失时刻补空行（不插值）
    assert rows[3][1] is None
    assert ex.detail["duplicates_removed"] == 2
    assert ex.detail["blank_rows_appended"] == 2


def test_repair_time_axis_drops_off_axis():
    t = [BASE + timedelta(hours=i) for i in range(2)]
    stray = BASE - timedelta(days=1)
    target = make_sheet(
        "冷冻水泵", ["chwp_01"], "chilled_water_pump", ["power"],
        [(stray, 5.0), (t[0], 10.0), (t[1], 11.0)],
    )
    reference = make_sheet(
        "冷水主机", ["chiller_01"], "chiller", ["power"],
        [(ts, 1.0) for ts in t],
    )
    rule = PreprocessRule(
        rule_id="fix-axis", rule_type="repair_time_axis", sheet="冷冻水泵",
        params={"reference_sheet": "冷水主机"},
    )
    (ex,) = apply_ruleset(
        {"冷冻水泵": target, "冷水主机": reference},
        RuleSet(ruleset_id="RS", version=1, rules=[rule]),
    )
    assert ex.detail["off_axis_removed"] == 1
    assert [r[0] for r in target.data_rows] == t


def test_fix_header_labels():
    sheets = fill_sheets()
    rule = PreprocessRule(
        rule_id="fix-labels", rule_type="fix_header_labels", sheet="冷却塔",
        params={"labels": {"2": "设备名称", "3": "物模型"}},
    )
    (ex,) = apply_ruleset(sheets, RuleSet(ruleset_id="RS", version=1,
                                          rules=[rule]))
    assert sheets["冷却塔"].header_rows[1][0] == "设备名称"
    assert sheets["冷却塔"].header_rows[2][0] == "物模型"
    assert ex.detail["labels_fixed"]["2"]["from"] == "设备实例"


# ---------------------------------------------------------------- 审批门禁与存储


def test_require_approved_blocks_proposed():
    sheets = fill_sheets()
    rs = RuleSet(ruleset_id="RS", version=1, rules=[fill_rule()])
    with pytest.raises(PreprocessError) as exc_info:
        apply_ruleset(sheets, rs, require_approved=True)
    assert exc_info.value.code == TFPP_RULE_NOT_APPROVED
    assert sheets["冷却塔"].data_rows[0][1] is None  # 未做任何修改


def test_approved_rule_runs_and_deprecated_skipped():
    sheets = fill_sheets()
    rs = RuleSet(ruleset_id="RS", version=1, rules=[
        fill_rule(status="approved"),
        fill_rule(rule_id="old", status="deprecated"),
    ])
    executions = apply_ruleset(sheets, rs, require_approved=True)
    assert [e.skipped for e in executions] == [False, True]
    assert sheets["冷却塔"].data_rows[0][1] == 30.0


def test_rule_store_roundtrip_and_version_conflict(tmp_path):
    store = RuleStore(tmp_path / "preprocess")
    rs = RuleSet(ruleset_id="RS", version=1, rules=[fill_rule()])
    store.save(rs)
    loaded = store.load("RS")
    assert loaded.content_hash() == rs.content_hash()
    # 同版本内容变更 → TFPP-005
    changed = RuleSet(ruleset_id="RS", version=1,
                      rules=[fill_rule(params={**FILL_PARAMS, "idle_value": 9})])
    with pytest.raises(PreprocessError) as exc_info:
        store.save(changed)
    assert exc_info.value.code == "TFPP-005"
    # 元数据（审批）更新允许
    approved = RuleSet(ruleset_id="RS", version=1,
                       rules=[fill_rule(status="approved")])
    store.save(approved)
    assert store.load("RS").rules[0].status == "approved"
    # 版本递增与 latest
    store.save(RuleSet(ruleset_id="RS", version=2, rules=[changed.rules[0]]))
    assert store.latest_version("RS") == 2
    assert store.load("RS").version == 2
    with pytest.raises(PreprocessError):
        store.load("NOPE")
    assert len(store.list_rulesets()) == 2


# ---------------------------------------------------------------- 工具信封与门禁链路


def _small_legacy_workbook(path: Path) -> Path:
    wb = Workbook()
    wb.remove(wb.active)
    ws = wb.create_sheet("冷水主机")
    ws.append(["设备类", "冷水主机", None])
    ws.append(["设备名称", "chiller_01", None])
    ws.append(["物模型", "chiller", None])
    ws.append(["物模型属性", "power", "current_percent"])
    for i in range(4):
        ws.append([BASE + timedelta(minutes=15 * i), 300.0 + i, 50.0 + i])
    wb.save(path)
    return path


def _ctx(tmp_path, actor="agent") -> ToolContext:
    return ToolContext(vault_root=tmp_path / "vault",
                       research_root=tmp_path / "research", actor=actor)


def _ruleset_doc() -> dict:
    return {
        "ruleset_id": "WX_FIX", "version": 1,
        "rules": [{
            "rule_id": "fix-labels", "rule_type": "fix_header_labels",
            "sheet": "冷水主机",
            "params": {"labels": {"2": "设备名称"}},
        }],
    }


def test_tool_envelope_and_approval_chain(tmp_path):
    src = _small_legacy_workbook(tmp_path / "legacy.xlsx")
    ctx = _ctx(tmp_path)

    # propose：信封结构 + 强制 proposed
    env = tf_preprocess_propose(ctx, {**_ruleset_doc(),
                                      "rules": [{**_ruleset_doc()["rules"][0],
                                                 "status": "approved"}]})
    assert env["ok"] and env["tool"] == "tf_preprocess_propose"
    assert env["id"] == "WX_FIX@v1" and env["status"] == "PROPOSED"
    assert env["summary"]["rules"][0]["status"] == "proposed"  # 不信任输入
    assert not env["truncated"]

    # agent 审批被拒
    env = tf_preprocess_approve(ctx, "WX_FIX")
    assert not env["ok"]
    assert env["diagnostics"][0]["code"] == "TFPP-006"

    # proposed 规则预览执行允许（不落 vault）
    env = tf_preprocess_apply(ctx, src, "WX_FIX")
    assert env["ok"] and env["status"] == "APPLIED"
    assert env["summary"]["rules"][0]["input_sha256"]

    # proposed 规则落 vault 被门禁拦截
    env = tf_preprocess_apply(ctx, src, "WX_FIX", import_into_vault=True)
    assert not env["ok"]
    assert env["diagnostics"][0]["code"] == "TFPP-002"

    # human 审批 → 放行 + 留痕
    human_ctx = _ctx(tmp_path, actor="human")
    env = tf_preprocess_approve(human_ctx, "WX_FIX", note="复核通过")
    assert env["ok"] and env["status"] == "APPROVED"
    rule = env["summary"]["rules"][0]
    assert rule["status"] == "approved"
    assert rule["approvals"][0]["actor"] == "human"
    assert rule["approvals"][0]["note"] == "复核通过"

    env = tf_preprocess_apply(human_ctx, src, "WX_FIX", import_into_vault=True)
    assert env["ok"], env
    assert env["status"] == "IMPORTED"
    assert env["id"] == "WX_2025_HVAC@rev_0001"
    lineage_artifact = next(a for a in env["artifacts"]
                            if a["kind"] == "preprocess_lineage")
    assert Path(lineage_artifact["path"]).exists()

    # list
    env = tf_preprocess_list(human_ctx)
    assert env["ok"]
    assert env["summary"]["count"] == 1
    assert env["summary"]["rulesets"][0]["content_hash"]


def test_tool_propose_invalid_schema(tmp_path):
    ctx = _ctx(tmp_path)
    env = tf_preprocess_propose(ctx, {"ruleset_id": "X", "version": 1,
                                      "rules": [{"rule_id": "r",
                                                 "rule_type": "nope",
                                                 "sheet": "s"}]})
    assert not env["ok"]
    assert env["status"] == "FAILED"


def test_tool_apply_unknown_ruleset(tmp_path):
    ctx = _ctx(tmp_path)
    env = tf_preprocess_apply(ctx, "whatever.xlsx", "NOPE")
    assert not env["ok"]
    assert env["diagnostics"][0]["code"] == "TFPP-003"
