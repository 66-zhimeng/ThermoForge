"""指标模块测试（implementation-notes.md §5）。

公式精确值、y_floor 边界与 mape_valid_fraction、TFX-905、
CVRMSE/NMBE 零均值未定义、micro + per-object 口径。
"""

from __future__ import annotations

import math

import pytest

from thermoforge_research.errors import ResearchError
from thermoforge_research.metrics import (
    METRIC_NAMES,
    compute_metrics,
    cvrmse,
    mape,
    nmbe,
    rmse,
)
import numpy as np


def test_formulas_exact():
    y = np.array([2.0, 4.0, 5.0, 4.0])
    p = np.array([2.5, 3.0, 5.0, 6.0])
    assert rmse(y, p) == pytest.approx(
        math.sqrt((0.25 + 1.0 + 0.0 + 4.0) / 4))
    assert cvrmse(y, p) == pytest.approx(rmse(y, p) / 3.75)
    assert nmbe(y, p) == pytest.approx((-0.5 + 1.0 + 0.0 - 2.0) / (4 * 3.75))
    value, fraction = mape(y, p, y_floor=0.0)
    assert fraction == 1.0
    assert value == pytest.approx(
        (0.5 / 2 + 1.0 / 4 + 0.0 / 5 + 2.0 / 4) / 4)


def test_mape_y_floor_excludes_small_targets():
    y = [100.0, 100.0, 1.0, 2.0]  # y_floor=5 → 后两个剔除
    p = [110.0, 90.0, 100.0, 100.0]
    value, fraction = mape(np.array(y), np.array(p), y_floor=5.0)
    assert value == pytest.approx(0.1)
    assert fraction == pytest.approx(0.5)


def test_default_y_floor_is_5_percent_of_max():
    # 默认 y_floor = 0.05 * max|y|（§5.1 [草案：额定功率的 5%] 的代理）
    y = [100.0] * 19 + [4.0]  # max=100 → floor=5.0，4.0 被剔除
    p = [100.0] * 19 + [0.0]
    report = compute_metrics(y, p, ["MAPE"])
    assert report.y_floor == pytest.approx(5.0)
    assert report.mape_valid_fraction == pytest.approx(19 / 20)


def test_mape_low_valid_fraction_raises_tfx905():
    y = [100.0, 1.0, 1.0, 1.0, 1.0]  # 有效比例 0.2 < 0.8
    p = [100.0, 0.0, 0.0, 0.0, 0.0]
    with pytest.raises(ResearchError) as excinfo:
        compute_metrics(y, p, ["MAPE"], y_floor=5.0)
    assert excinfo.value.code == "TFX-905"


def test_cvrmse_undefined_when_mean_near_zero():
    y = [1.0, -1.0, 1.0, -1.0]  # mean = 0
    p = [0.0] * 4
    report = compute_metrics(y, p, ["CVRMSE", "NMBE"])
    assert report.metrics["CVRMSE"] is None
    assert report.metrics["NMBE"] is None
    assert "CVRMSE" in report.undefined and "NMBE" in report.undefined


def test_per_object_breakdown_and_micro_default():
    y = [10.0, 20.0, 100.0, 200.0]
    p = [11.0, 19.0, 100.0, 200.0]
    objs = ["A", "A", "B", "B"]
    report = compute_metrics(y, p, ["RMSE", "MAE"], object_ids=objs)
    # micro：全样本合并
    assert report.metrics["RMSE"] == pytest.approx(math.sqrt((1 + 1) / 4))
    assert set(report.per_object) == {"A", "B"}
    assert report.per_object["A"]["metrics"]["MAE"] == pytest.approx(1.0)
    assert report.per_object["B"]["metrics"]["MAE"] == pytest.approx(0.0)


def test_nan_input_is_defect_not_silent():
    with pytest.raises(ResearchError) as excinfo:
        compute_metrics([1.0, float("nan")], [1.0, 1.0], ["RMSE"])
    assert excinfo.value.code == "TFX-905"


def test_empty_input_raises():
    with pytest.raises(ResearchError):
        compute_metrics([], [], ["RMSE"])


def test_unknown_metric_rejected():
    with pytest.raises(ValueError):
        compute_metrics([1.0], [1.0], ["SMAPE"])
    assert "NMBE" in METRIC_NAMES  # DD-14：NMBE 为必报偏差指标
