"""Data Vault 测试：revision 去重、指纹稳定性、不可变性与完整性校验。

指纹稳定性（implementation-notes §13.2）：同逻辑数据列序/行序调换 →
同 content_sha256 复用同一 revision；值改动 → 新 revision。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from openpyxl import Workbook, load_workbook

from thermoforge_data.importer import import_xlsx
from thermoforge_data.vault import DataVault, VaultError

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
VALID = FIXTURES_DIR / "valid_minimal.xlsx"


def _save_variant(path: Path, *, permute: bool = False, change_value: bool = False):
    """基于 valid_minimal 生成变体：列序+行序调换，或改动一个值。"""
    src = load_workbook(VALID)
    wb = Workbook()
    wb.remove(wb.active)
    for name in src.sheetnames:
        if name == "data":
            continue
        ws_new = wb.create_sheet(name)
        for row in src[name].iter_rows(values_only=True):
            ws_new.append(list(row))
    ws_data = wb.create_sheet("data")
    header = [c.value for c in src["data"][1]]
    rows = [[c.value for c in row] for row in src["data"].iter_rows(min_row=2)]
    if change_value:
        rows[0][1] = rows[0][1] + 0.5
    if permute:
        order = [0, 3, 1, 2]  # timestamp 保持首列，其余列序调换
        header = [header[i] for i in order]
        rows = [[row[i] for i in order] for row in rows]
        rows = [rows[2], rows[0], rows[3], rows[1]]  # 行序打乱
    ws_data.append(header)
    for row in rows:
        ws_data.append(row)
        for cell in ws_data[ws_data.max_row]:
            if cell.column == 1 and isinstance(cell.value, str):
                cell.number_format = "@"
    wb.save(path)
    return path


def _fingerprints(vault: DataVault, ref: str) -> dict:
    info = vault.resolve(ref)
    with open(info.path / "fingerprint.json", encoding="utf-8") as fp:
        return json.load(fp)


def test_store_creates_revision_layout(tmp_path):
    vault = DataVault(tmp_path / "vault")
    result = import_xlsx(VALID)
    ref = vault.store(result)
    assert ref == "DC01_2026_CHILLER@rev_0001"
    info = vault.resolve(ref)
    for rel in ("canonical/data.parquet", "canonical/variables.parquet",
                "canonical/objects.parquet", "canonical/parameters.parquet",
                "manifest.json", "quality.json", "profile.json",
                "lineage.json", "fingerprint.json"):
        assert (info.path / rel).exists(), rel
    assert (info.path / "source" / "valid_minimal.xlsx").exists()
    # source 写后为只读（§10.1）
    mode = os.stat(info.path / "source" / "valid_minimal.xlsx").st_mode
    assert not mode & 0o200
    # DuckDB 元数据索引
    assert vault.list_revisions("DC01_2026_CHILLER") == ["rev_0001"]
    table = vault.load_data(ref)
    assert table.num_rows == 4


def test_content_dedup_reuses_revision(tmp_path):
    """同一文件重复导入 → 复用同一 revision。"""
    vault = DataVault(tmp_path / "vault")
    ref1 = vault.store(import_xlsx(VALID))
    ref2 = vault.store(import_xlsx(VALID))
    assert ref1 == ref2
    assert vault.list_revisions("DC01_2026_CHILLER") == ["rev_0001"]


def test_fingerprint_stable_under_permutation(tmp_path):
    """列序/行序调换 → 同 content_sha256，复用同一 revision（§5.2）。"""
    vault = DataVault(tmp_path / "vault")
    ref1 = vault.store(import_xlsx(VALID))
    variant = _save_variant(tmp_path / "permuted.xlsx", permute=True)
    ref2 = vault.store(import_xlsx(variant), source_path=variant)
    assert ref1 == ref2
    fp1 = _fingerprints(vault, ref1)
    assert fp1["content_sha256"]
    assert vault.list_revisions("DC01_2026_CHILLER") == ["rev_0001"]


def test_value_change_creates_new_revision(tmp_path):
    vault = DataVault(tmp_path / "vault")
    ref1 = vault.store(import_xlsx(VALID))
    variant = _save_variant(tmp_path / "changed.xlsx", change_value=True)
    ref2 = vault.store(import_xlsx(variant), source_path=variant)
    assert ref2 == "DC01_2026_CHILLER@rev_0002"
    fp1 = _fingerprints(vault, ref1)
    fp2 = _fingerprints(vault, ref2)
    assert fp1["content_sha256"] != fp2["content_sha256"]


def test_error_import_rejected(tmp_path):
    vault = DataVault(tmp_path / "vault")
    bad = import_xlsx(FIXTURES_DIR / "invalid_TFDC-105_merged_cell.xlsx")
    with pytest.raises(VaultError) as exc_info:
        vault.store(bad)
    assert exc_info.value.code == "TFV-701"


def test_fingerprint_mismatch_detected(tmp_path):
    """篡改落盘 parquet 后读取报 TFV-702。"""
    vault = DataVault(tmp_path / "vault")
    ref = vault.store(import_xlsx(VALID))
    info = vault.resolve(ref)
    target = info.path / "canonical" / "objects.parquet"
    data = bytearray(target.read_bytes())
    data[-10] ^= 0xFF
    target.write_bytes(bytes(data))
    with pytest.raises(VaultError) as exc_info:
        vault.load_data(ref)
    assert exc_info.value.code == "TFV-702"


def test_revision_not_found(tmp_path):
    vault = DataVault(tmp_path / "vault")
    with pytest.raises(VaultError) as exc_info:
        vault.load_data("NOPE@rev_0001")
    assert exc_info.value.code == "TFV-703"
