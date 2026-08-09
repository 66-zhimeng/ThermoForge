"""物理验证测试（implementation-notes.md §6）。

- 可证伪硬约束逐条触发与各自口径。
- 总体口径：任一约束违规即计入（§6.4）。
- 单调性：训练后模型的受控扰动扫描（§6.3）。
- 发布门禁绑定总体口径：违反硬约束不得进入发布候选。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thermoforge_research.physics_checks import (
    CEL_TO_K,
    check_hard_constraints,
    check_monotonicity,
    combine_reports,
)


def _df(power, cooling, t_chws=7.0, t_cwr=32.0):
    n = len(power)
    return pd.DataFrame({
        "input_power": power,
        "cooling": cooling,
        "chw_supply": [t_chws] * n,
        "cw_return": [t_cwr] * n,
    })


def test_each_hard_constraint_fires():
    # cop_positive / positive_cooling_positive_power：Q>0 但 P<=0
    report = check_hard_constraints(
        _df([0.0, 100.0], [500.0, 500.0]),
        power_col="input_power", cooling_col="cooling",
    )
    assert report.hard_constraints["cop_positive"].violations == 1
    assert report.hard_constraints["cop_positive"].rate == pytest.approx(0.5)

    # cop_below_carnot：COP = 500/10 = 50，Carnot ≈ 280/(305-280) ≈ 11.2
    report = check_hard_constraints(
        _df([10.0], [500.0], t_chws=7.0, t_cwr=32.0),
        power_col="input_power", cooling_col="cooling",
        chw_supply_col="chw_supply", cw_return_col="cw_return",
    )
    carnot = report.hard_constraints["cop_below_carnot"]
    assert carnot.violations == 1 and carnot.rate == 1.0

    # power_within_rated：超出 rated × 上限系数
    report = check_hard_constraints(
        _df([1300.0, 500.0], [0.0, 0.0]),
        power_col="input_power", rated_power=1000.0,  # 上限 1100
    )
    rated = report.hard_constraints["power_within_rated"]
    assert rated.violations == 1 and rated.rate == pytest.approx(0.5)


def test_carnot_uses_kelvin_not_celsius():
    # 若误用摄氏度，7/(32-7)=0.28，一切 COP>1 都会误判违规；
    # 正常 COP=5 的样本必须通过（开尔文 Carnot ≈ 11.2）
    report = check_hard_constraints(
        _df([100.0], [500.0]),
        power_col="input_power", cooling_col="cooling",
        chw_supply_col="chw_supply", cw_return_col="cw_return",
    )
    assert report.hard_constraints["cop_below_carnot"].violations == 0


def test_overall_rate_counts_any_violation():
    # 样本 0 违反 cop（P<=0），样本 1 违反 rated（超上限）：
    # 每条单独口径 50%，总体口径 100%（任一违规即计入，§6.4）
    report = check_hard_constraints(
        _df([0.0, 2000.0], [500.0, 500.0]),
        power_col="input_power", cooling_col="cooling", rated_power=1000.0,
    )
    assert report.hard_constraints["cop_positive"].rate == pytest.approx(0.5)
    assert report.hard_constraints["power_within_rated"].rate == pytest.approx(0.5)
    assert report.overall_rate == pytest.approx(1.0)
    assert not report.is_publish_candidate(0.001)  # 硬约束失败不得发布
    clean = check_hard_constraints(
        _df([100.0, 200.0], [500.0, 900.0]),
        power_col="input_power", cooling_col="cooling",
        chw_supply_col="chw_supply", cw_return_col="cw_return",
        rated_power=1000.0,
    )
    assert clean.overall_rate == 0.0
    assert clean.is_publish_candidate(0.001)


def test_constraints_skip_inapplicable_samples():
    # 制冷量为 0 的样本不进入 COP 类约束的判定
    report = check_hard_constraints(
        _df([0.0, 100.0], [0.0, 500.0]),
        power_col="input_power", cooling_col="cooling",
    )
    assert report.hard_constraints["cop_positive"].applicable == 1
    assert report.hard_constraints["cop_positive"].violations == 0


def test_monotonicity_controlled_perturbation():
    def up(df):  # 单调递增模型
        return df["x"].to_numpy() * 2.0
    def down(df):  # 单调递减模型
        return -df["x"].to_numpy()
    def zigzag(df):  # 非单调模型
        return np.sin(df["x"].to_numpy() * np.pi * 2)

    base = {"x": 0.0, "other": 1.0}
    grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    assert check_monotonicity(up, base, "x", grid, direction=1).violations == 0
    assert check_monotonicity(down, base, "x", grid, direction=-1).violations == 0
    bad = check_monotonicity(zigzag, base, "x", grid, direction=1)
    assert bad.violations > 0 and bad.rate > 0
    # 方向判反时单调模型也会违规
    assert check_monotonicity(up, base, "x", grid, direction=-1).violations == 4


def test_combine_reports_merges_monotonicity_into_overall():
    hard = check_hard_constraints(
        _df([100.0, 200.0], [500.0, 900.0]),
        power_col="input_power", cooling_col="cooling",
    )
    mono = check_monotonicity(
        lambda df: np.sin(df["x"].to_numpy() * np.pi * 2),
        {"x": 0.0}, "x", [0.0, 0.25, 0.5, 0.75, 1.0], direction=1,
    )
    assert mono.violations == 2
    combined = combine_reports(hard, {mono.name: mono})
    assert mono.name in combined.monotonicity
    assert combined.overall_violations == hard.overall_violations + mono.violations


def test_rated_power_per_sample_column():
    # rated_power 传列名：逐样本额定（v2 单台额定 × run_count 的口径）
    df = pd.DataFrame({
        "input_power": [2000.0, 900.0, 3500.0],
        "rated": [1100.0, 1100.0, 3300.0],  # 1 台 / 1 台 / 3 台运行
    })
    report = check_hard_constraints(df, power_col="input_power",
                                    rated_power="rated")
    rated = report.hard_constraints["power_within_rated"]
    # 上限 1.1×：2000 > 1210 违规；900 合规；3500 > 3630? 否，合规
    assert rated.violations == 1
    assert rated.rate == pytest.approx(1 / 3)
