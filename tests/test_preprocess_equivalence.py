"""规则库复现人工处理的等价性验证（I-49，slow）。

对**原始**工作簿应用「冷却塔填充 + 冷冻水泵修复 + 表头标签修正」规则集，
导入 vault；断言其 content_sha256 与既有处理后副本导入的 revision 指纹一致
——这是「规则库复现人工脚本」的强验证。原始工作簿只读。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from thermoforge_data.importer import default_registry, import_parsed
from thermoforge_data.legacy import (
    convert_legacy_tables,
    convert_legacy_workbook,
    load_workbook_model,
)
from thermoforge_data.preprocess import (
    PreprocessRule,
    RuleSet,
    apply_ruleset,
)
from thermoforge_data.vault import DataVault

REPO_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = REPO_ROOT / "data" / "数据处理_算法导入训练_0723_WX_已加入表冷器.xlsx"
PROCESSED = (REPO_ROOT / "data"
             / "数据处理_算法导入训练_0723_WX_已加入表冷器_已填充冷却塔.xlsx")

pytestmark = pytest.mark.slow

# 复现 scripts/fill_cooling_tower.py + scripts/fix_chwp.py 的规则集。
# round_decimals=10 复现 fill 脚本 fmt() 的 round(v, 10) 写出精度。
WX_NORMALIZE_RULESET = RuleSet(
    ruleset_id="WX_0723_NORMALIZE",
    version=1,
    rules=[
        PreprocessRule(
            rule_id="fill-cooling-tower",
            rule_type="fill_from_header_divide_by_count",
            sheet="冷却塔",
            status="approved",
            proposer="human",
            params={
                "source_sheet": "冷却水总管",
                "source_instance": "cw_A1",
                "value_map": {"supply_t": "t_supply", "return_t": "t_return"},
                "divide_prop": "instant_flow",
                "divide_source_prop": "f",
                "running_sheet": "冷却塔风机",
                "running_prop": "status_run",
                "member_pattern": "{instance}_f",
                "leave_empty": ["supply_p", "return_p"],
                "idle_value": 0,
                "round_decimals": 10,
            },
        ),
        PreprocessRule(
            rule_id="repair-chwp-axis",
            rule_type="repair_time_axis",
            sheet="冷冻水泵",
            status="approved",
            proposer="human",
            params={"reference_sheet": "冷水主机"},
        ),
        PreprocessRule(
            rule_id="fix-chwp-labels",
            rule_type="fix_header_labels",
            sheet="冷冻水泵",
            status="approved",
            proposer="human",
            params={"labels": {"2": "设备名称", "3": "物模型"}},
        ),
    ],
)


def _import_tables(vault: DataVault, sheets, source: Path) -> str:
    registry = default_registry()
    conv = convert_legacy_tables(sheets, registry, source_name=source.name)
    result = import_parsed(
        conv.dataset, conv.timestamps, conv.columns, registry=registry,
        pre_diagnostics=conv.pre_diagnostics,
        degradations=conv.degradations, source_path=source,
    )
    assert result.ok, [d for d in result.diagnostics if d.level.value == "ERROR"]
    return vault.store(result, source_path=source, lineage=conv.lineage(source))


@pytest.mark.skipif(not ORIGINAL.exists() or not PROCESSED.exists(),
                    reason="真实工作簿不存在")
def test_ruleset_reproduces_manual_scripts(tmp_path):
    vault = DataVault(tmp_path / "vault")

    # 路径 A：原始工作簿 + 规则集
    sheets = load_workbook_model(ORIGINAL)
    executions = apply_ruleset(sheets, WX_NORMALIZE_RULESET,
                               require_approved=True)
    assert all(not e.skipped for e in executions)
    repair = next(e for e in executions if e.rule_id == "repair-chwp-axis")
    assert repair.detail["duplicates_removed"] == 2049  # data-survey F7
    assert repair.detail["blank_rows_appended"] == 481
    ref_a = _import_tables(vault, sheets, ORIGINAL)

    # 路径 B：人工脚本处理后的副本
    conv_b = convert_legacy_workbook(PROCESSED, default_registry())
    result_b = import_parsed(
        conv_b.dataset, conv_b.timestamps, conv_b.columns,
        registry=default_registry(),
        pre_diagnostics=conv_b.pre_diagnostics,
        degradations=conv_b.degradations, source_path=PROCESSED,
    )
    assert result_b.ok
    ref_b = vault.store(result_b, source_path=PROCESSED,
                        lineage=conv_b.lineage(PROCESSED))

    info_a = vault.resolve(ref_a)
    info_b = vault.resolve(ref_b)
    assert info_a.content_sha256 == info_b.content_sha256, (
        f"规则库产出与人工脚本指纹不一致:\n"
        f"  规则集: {ref_a} {info_a.content_sha256}\n"
        f"  人工:   {ref_b} {info_b.content_sha256}"
    )
    # 同内容去重：两次导入应复用同一 revision
    assert ref_a == ref_b
