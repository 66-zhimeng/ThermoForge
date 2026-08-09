"""Model Package 构建与校验（model-package §2、implementation-notes §8）。

- 必需文件齐全 + checksums 全文件 sha256；
- 篡改 / 缺文件 → TFM-1002；
- golden 预测集覆盖工况边界（boundary_rows）；
- signature 扩展键 history_required / cold_start 的拆分与读取。
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from thermoforge_runtime.errors import ModelRegistryError
from thermoforge_runtime.package import (
    PACKAGE_REQUIRED_FILES,
    boundary_rows,
    compute_checksums,
    load_constraints,
    load_model_meta,
    load_signature,
    split_signature_doc,
    verify_package,
    write_checksums,
)

from phase34_helpers import FEATURES, build_test_package, training_frame


@pytest.fixture()
def pkg(tmp_path):
    return build_test_package(tmp_path / "pkg")


def test_package_structure_and_checksums(pkg):
    for rel in PACKAGE_REQUIRED_FILES:
        assert (pkg / rel).is_file(), rel
    assert any((pkg / "artifact").iterdir())
    result = verify_package(pkg)
    assert result["ok"] and result["files"] > len(PACKAGE_REQUIRED_FILES) - 1
    meta = load_model_meta(pkg)
    assert meta.model_id == "chiller-power" and meta.version == "1.0.0"
    assert meta.status == "candidate"


def test_golden_covers_operating_boundary(pkg):
    """golden 输入 = 每个特征 min/max 所在行（工况边界而非随机抽样）。"""
    golden = pd.read_parquet(pkg / "golden.parquet")
    input_cols = [c for c in golden.columns if not c.startswith("output__")]
    assert list(input_cols) == list(FEATURES)  # 特征顺序显式存储
    df = training_frame()
    for feat in FEATURES:
        col = golden[feat]
        assert col.min() == pytest.approx(df[feat].min())
        assert col.max() == pytest.approx(df[feat].max())
    assert any(c.startswith("output__") for c in golden.columns)


def test_tampered_file_detected_tfm1002(pkg):
    readme = pkg / "README.md"
    readme.write_text(readme.read_text(encoding="utf-8") + "tampered\n",
                      encoding="utf-8", newline="\n")
    with pytest.raises(ModelRegistryError) as excinfo:
        verify_package(pkg)
    assert excinfo.value.code == "TFM-1002"


def test_missing_required_file_tfm1002(pkg):
    (pkg / "metrics.json").unlink()
    with pytest.raises(ModelRegistryError) as excinfo:
        verify_package(pkg)
    assert excinfo.value.code == "TFM-1002"


def test_unregistered_extra_file_tfm1002(pkg):
    (pkg / "stray.txt").write_text("x", encoding="utf-8", newline="\n")
    with pytest.raises(ModelRegistryError) as excinfo:
        verify_package(pkg)
    assert excinfo.value.code == "TFM-1002"


def test_checksum_rewrite_repairs(pkg):
    readme = pkg / "README.md"
    readme.write_text("rewritten\n", encoding="utf-8", newline="\n")
    write_checksums(pkg)
    assert verify_package(pkg)["ok"]
    recorded = json.loads((pkg / "checksums.json")
                          .read_text(encoding="utf-8"))["files"]
    assert recorded == compute_checksums(pkg)


def test_signature_extensions_roundtrip(tmp_path):
    history = {"window": "30min", "resolution": "1min", "min_samples": 25}
    pkg = build_test_package(tmp_path / "pkg", history_required=history,
                             cold_start="reject")
    signature, hist, cold = load_signature(pkg)
    assert hist == history and cold == "reject"
    assert [p.property_code for p in signature.inputs] == list(FEATURES)
    # 扩展键不进入契约部分
    base, ext = split_signature_doc(
        {"model_id": "m", "history_required": history, "cold_start": "reject"})
    assert "history_required" not in base
    assert ext == {"history_required": history, "cold_start": "reject"}


def test_constraints_roundtrip(pkg):
    constraints = load_constraints(pkg)
    by_prop = {c.property_code: c for c in constraints.inputs}
    assert by_prop["evap_chw_flow"].out_of_range == "reject"
    assert by_prop["evap_chw_supply_temp"].out_of_range == "clamp"
    assert (by_prop["evap_chw_return_temp"].out_of_range
            == "passthrough_with_flag")
    assert constraints.output_min_value == 0.0


def test_boundary_rows_dedup_and_first_row():
    df = training_frame()
    rows = boundary_rows(df, list(FEATURES))
    assert rows[0] == {f: float(df[f].iloc[0]) for f in FEATURES}
    assert len(rows) <= 1 + 2 * len(FEATURES)
