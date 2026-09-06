"""切分子集分布画像测试（train/validate/test 各段自变量+目标分布）。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from thermoforge_research.split_profile import build_split_profile


def _frame(values: dict[str, list[float]]) -> pd.DataFrame:
    return pd.DataFrame(values)


def test_reports_per_subset_quantiles():
    subsets = {
        "train": _frame({"x": [0.0, 1.0, 2.0, 3.0, 4.0]}),
        "validate": _frame({"x": [1.5, 2.5]}),
        "test": _frame({"x": [2.0]}),
    }
    doc = build_split_profile(subsets, ["x"])
    assert doc["reference"] == "train"
    train = doc["subsets"]["train"]["variables"]["x"]
    assert train["min"] == 0.0
    assert train["max"] == 4.0
    assert train["count"] == 5
    assert train["quantiles"]["p50"] == 2.0


def test_out_of_range_fraction_relative_to_train():
    subsets = {
        "train": _frame({"x": [0.0, 10.0]}),
        "validate": _frame({"x": [-5.0, 5.0, 15.0, 20.0]}),
    }
    doc = build_split_profile(subsets, ["x"])
    entry = doc["subsets"]["validate"]["variables"]["x"]
    # -5 与 20 越界，5 在界内，15 越界 -> 3/4
    assert entry["out_of_train_range_fraction"] == 0.75
    # 参照段自身不算越界占比
    assert "out_of_train_range_fraction" not in doc["subsets"]["train"]["variables"]["x"]


def test_empty_subset_has_no_range_fields():
    subsets = {
        "train": _frame({"x": [1.0, 2.0, 3.0]}),
        "validate": _frame({"x": pd.Series([], dtype=float)}),
    }
    doc = build_split_profile(subsets, ["x"])
    entry = doc["subsets"]["validate"]["variables"]["x"]
    assert entry == {"count": 0, "missing": 0}


def test_all_missing_column_reports_missing_without_crashing():
    subsets = {"train": _frame({"x": [np.nan, np.nan]})}
    doc = build_split_profile(subsets, ["x"])
    entry = doc["subsets"]["train"]["variables"]["x"]
    assert entry == {"count": 0, "missing": 2}


def test_missing_column_in_one_subset_is_skipped():
    subsets = {
        "train": _frame({"x": [1.0, 2.0], "y": [1.0, 2.0]}),
        "validate": _frame({"x": [1.0]}),
    }
    doc = build_split_profile(subsets, ["x", "y"])
    assert "y" not in doc["subsets"]["validate"]["variables"]
    assert "y" in doc["subsets"]["train"]["variables"]


def test_custom_reference_subset():
    subsets = {
        "train": _frame({"x": [0.0, 100.0]}),
        "test": _frame({"x": [0.0, 1.0, 2.0]}),
    }
    doc = build_split_profile(subsets, ["x"], reference="test")
    assert doc["reference"] == "test"
    assert "out_of_test_range_fraction" in doc["subsets"]["train"]["variables"]["x"]
    assert "out_of_test_range_fraction" not in doc["subsets"]["test"]["variables"]["x"]
