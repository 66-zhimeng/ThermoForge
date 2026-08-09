"""真实 43MB 旧格式工作簿的集成测试（slow，默认跳过，用 -m slow 触发）。

断言：revision 创建、行数 35040、时间范围覆盖 2025 全年、
诊断中含负流量（TFDC-601）。原始工作簿只读。
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from thermoforge_core.errors import Level
from thermoforge_data.importer import default_registry, import_parsed
from thermoforge_data.legacy import convert_legacy_workbook
from thermoforge_data.vault import DataVault

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = REPO_ROOT / "data" / "数据处理_算法导入训练_0723_WX_已加入表冷器_已填充冷却塔.xlsx"

pytestmark = pytest.mark.slow


@pytest.mark.skipif(not WORKBOOK.exists(), reason="真实工作簿不存在")
def test_real_workbook_end_to_end(tmp_path):
    t0 = time.monotonic()
    registry = default_registry()
    conv = convert_legacy_workbook(WORKBOOK, registry)
    result = import_parsed(
        conv.dataset, conv.timestamps, conv.columns,
        pre_diagnostics=conv.pre_diagnostics,
        degradations=conv.degradations,
        source_path=WORKBOOK,
    )
    elapsed_import = time.monotonic() - t0

    assert result.ok, [
        d for d in result.diagnostics if d.level == Level.ERROR
    ]
    assert result.table.num_rows == 35040

    ts = result.table.column("timestamp").to_pylist()
    # 2025 全年（Asia/Shanghai 本地）：UTC 2024-12-31T16:00 → 2025-12-31T15:45
    assert ts[0].isoformat() == "2024-12-31T16:00:00+00:00"
    assert ts[-1].isoformat() == "2025-12-31T15:45:00+00:00"

    # 负流量等已知缺陷以诊断呈现，而非静默通过
    range_violations = [d for d in result.diagnostics if d.code == "TFDC-601"]
    assert range_violations, "应检出负流量（TFDC-601 RANGE_VIOLATION）"
    violated = {d.location for d in range_violations}
    assert any("cw_A1.f" in loc or "cw_A2.f" in loc for loc in violated)

    vault = DataVault(tmp_path / "vault")
    ref = vault.store(result, lineage=conv.lineage(WORKBOOK))
    assert ref == "WX_2025_HVAC@rev_0001"
    assert vault.load_data(ref).num_rows == 35040
    elapsed_total = time.monotonic() - t0
    print(f"\nimport {elapsed_import:.1f}s, total {elapsed_total:.1f}s")
    for d in result.diagnostics:
        print(f"  {d.code} {d.level.value} {d.location} count={d.count}")
