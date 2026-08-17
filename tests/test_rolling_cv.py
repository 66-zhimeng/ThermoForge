"""滚动原点（rolling-origin）时序交叉验证测试（research-loop §6）。

fold 语义：fold k 的原点 origin_k = floor_res(b0 + k·step)；expanding 模式
训练集为 [t0, origin_k − purge)，sliding 模式为 [origin_k − W, origin_k − purge)
（W 恒定），评估集为 [origin_k + embargo, floor_res(origin_k + horizon))。
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from thermoforge_research.errors import ResearchError
from thermoforge_research.ledger import ResearchLedger
from thermoforge_research.metrics import aggregate_fold_metrics, compute_metrics
from thermoforge_research.runner import run_experiment, verify_reproducibility
from thermoforge_research.splits import (
    check_no_leakage,
    rolling_origin_splits,
    temporal_split,
)

from phase2_helpers import build_chiller_vault, make_experiment, view_definition

UTC = timezone.utc
RES = 900  # 15 min
ACTOR = "test-rolling"


def _ts(n: int, start_minute: int = 7) -> list[datetime]:
    base = datetime(2026, 1, 1, 0, start_minute, tzinfo=UTC)
    return [base + timedelta(seconds=RES * i) for i in range(n)]


def _cfg(**over):
    cfg = {
        "initial_train_fraction": 0.5,
        "horizon_seconds": 6 * 3600,
        "step_seconds": 6 * 3600,
        "embargo_seconds": 0.0,
    }
    cfg.update(over)
    return cfg


# ---------------------------------------------------------------- fold 边界


def test_fold_boundaries_aligned_to_resolution():
    ts = _ts(192)  # 起点 :07 不对齐 15 min
    result = rolling_origin_splits(ts, RES, **_cfg())
    assert len(result.folds) == 4
    for fold in result.folds:
        origin = datetime.fromisoformat(fold.boundaries["origin"])
        assert origin.timestamp() % RES == 0  # 向下对齐 resolution
        tr = fold.boundaries["train_range"]
        ev = fold.boundaries["eval_range"]
        # 区间语义：train [start, origin)，eval [origin, origin+horizon)
        assert tr[1] == fold.boundaries["origin"]
        assert ev[0] == fold.boundaries["origin"]
        assert fold.boundaries["counts"]["eval"] == 24  # 6 h / 15 min
    # 原点等距 step
    origins = [f.boundaries["origin"] for f in result.folds]
    deltas = [
        datetime.fromisoformat(b) - datetime.fromisoformat(a)
        for a, b in zip(origins, origins[1:])
    ]
    assert all(d == timedelta(hours=6) for d in deltas)


def test_expanding_vs_sliding():
    ts = _ts(192)
    expanding = rolling_origin_splits(ts, RES, **_cfg(mode="expanding"))
    sliding = rolling_origin_splits(ts, RES, **_cfg(mode="sliding"))
    # expanding：训练起点恒为 t0，窗口递增；sliding：窗口长度恒定
    assert all(f.boundaries["train_range"][0] == expanding.config["t0"]
               for f in expanding.folds)
    exp_sizes = [f.boundaries["counts"]["train"] for f in expanding.folds]
    sli_sizes = [f.boundaries["counts"]["train"] for f in sliding.folds]
    assert exp_sizes == sorted(exp_sizes) and len(set(exp_sizes)) > 1
    assert len(set(sli_sizes)) == 1
    # 两种模式 fold 0 相同（尚未滑动）
    assert expanding.folds[0].train_idx == sliding.folds[0].train_idx


def test_max_folds_cap_and_default_step():
    ts = _ts(192)
    capped = rolling_origin_splits(ts, RES, **_cfg(max_folds=2))
    assert len(capped.folds) == 2
    # step 默认 = horizon；step 更小 → fold 更多
    default_step = rolling_origin_splits(ts, RES, **_cfg())
    small_step = rolling_origin_splits(ts, RES, **_cfg(step_seconds=3 * 3600))
    assert default_step.config["step_seconds"] == 6 * 3600
    assert len(small_step.folds) > len(default_step.folds)


def test_initial_train_by_duration():
    ts = _ts(192)
    by_seconds = rolling_origin_splits(
        ts, RES, initial_train_seconds=24 * 3600, horizon_seconds=6 * 3600,
        embargo_seconds=0)
    b0 = datetime.fromisoformat(by_seconds.config["b0"])
    t0 = datetime.fromisoformat(by_seconds.config["t0"])
    assert (b0 - t0) <= timedelta(hours=24)  # 向下对齐后不超声明时长
    with pytest.raises(ValueError):
        rolling_origin_splits(ts, RES, horizon_seconds=3600)  # 两个初值都没给
    with pytest.raises(ValueError):
        rolling_origin_splits(  # 两个都给
            ts, RES, initial_train_fraction=0.5, initial_train_seconds=3600,
            horizon_seconds=3600)


# ---------------------------------------------------------------- purge/embargo 与 TFX-903


def test_purge_and_embargo_applied_per_fold():
    ts = _ts(192)
    purge, embargo = 4 * RES, 2 * RES
    result = rolling_origin_splits(
        ts, RES, **_cfg(purge_seconds=purge, embargo_seconds=embargo))
    for fold in result.folds:
        origin = datetime.fromisoformat(fold.boundaries["origin"])
        train_max = max(ts[i] for i in fold.train_idx)
        eval_min = min(ts[i] for i in fold.eval_idx)
        assert train_max < origin - timedelta(seconds=purge)
        assert eval_min >= origin + timedelta(seconds=embargo)


def test_rolling_feature_without_purge_caught_per_fold():
    """逐 fold 的 TFX-903：未 purge 的滚动特征场景被 check_no_leakage 捕获。"""
    ts = _ts(192)
    window = 8 * RES  # 滚动窗口 2 h
    no_purge = rolling_origin_splits(ts, RES, **_cfg())
    for fold in no_purge.folds:
        with pytest.raises(ResearchError) as excinfo:
            check_no_leakage(ts, fold.train_idx, fold.eval_idx,
                             min_gap_seconds=window)
        assert excinfo.value.code == "TFX-903"
    # 正确 purge 后同一检查逐 fold 通过
    purged = rolling_origin_splits(ts, RES, **_cfg(purge_seconds=window))
    for fold in purged.folds:
        check_no_leakage(ts, fold.train_idx, fold.eval_idx,
                         min_gap_seconds=window)


def test_no_valid_fold_raises():
    ts = _ts(50)
    with pytest.raises(ValueError, match="fold"):
        rolling_origin_splits(ts, RES, initial_train_fraction=0.9,
                              horizon_seconds=3600, embargo_seconds=10 * 3600)


# ---------------------------------------------------------------- 聚合口径


def test_single_fold_degenerate_matches_holdout():
    """单 fold 退化：train == holdout train，eval == holdout validate+test。"""
    ts = _ts(192)
    roll = rolling_origin_splits(
        ts, RES, initial_train_fraction=0.7, horizon_seconds=10**7,
        max_folds=1, embargo_seconds=0)
    hold = temporal_split(ts, RES, 0.70, 0.15, 0.15, embargo_seconds=0)
    assert roll.folds[0].train_idx == hold.train_idx
    assert roll.folds[0].eval_idx == hold.validate_idx + hold.test_idx


def test_aggregation_deterministic_and_consistent():
    y = [100.0 + i for i in range(24)]
    p1 = [v * 1.01 for v in y]
    p2 = [v * 0.98 for v in y]
    metrics = ["RMSE", "MAE", "MAPE", "CVRMSE", "NMBE"]
    r1 = compute_metrics(y, p1, metrics)
    r2 = compute_metrics(y, p2, metrics)
    agg_a = aggregate_fold_metrics([r1, r2])
    agg_b = aggregate_fold_metrics([r1, r2])
    assert agg_a == agg_b  # 确定性：同输入两跑一致
    rmse_stats = agg_a["per_metric"]["RMSE"]
    assert rmse_stats["min"] == min(r1.metrics["RMSE"], r2.metrics["RMSE"])
    assert rmse_stats["max"] == max(r1.metrics["RMSE"], r2.metrics["RMSE"])
    assert rmse_stats["mean"] == pytest.approx(
        (r1.metrics["RMSE"] + r2.metrics["RMSE"]) / 2)
    assert rmse_stats["std"] == pytest.approx(
        abs(r1.metrics["RMSE"] - r2.metrics["RMSE"]) / 2)  # ddof=0
    assert "NMBE" in agg_a["per_metric"]  # NMBE 必报
    # 单 fold 退化：mean 即该 fold 值、std 为 0
    single = aggregate_fold_metrics([r1])
    assert single["per_metric"]["RMSE"]["mean"] == r1.metrics["RMSE"]
    assert single["per_metric"]["RMSE"]["std"] == 0.0


# ---------------------------------------------------------------- 契约


def test_contract_backward_compatible():
    exp = make_experiment("EXP-0001")  # 旧定义：无 rolling_cv
    assert exp.validation.rolling_cv.enabled is False
    assert exp.validation.rolling_cv.mode == "expanding"
    assert exp.validation.rolling_cv.max_folds == 10
    # 启用但缺初始训练窗口 → 校验拒绝
    with pytest.raises(ValidationError):
        make_experiment("EXP-0002", rolling_cv={"enabled": True})
    # 合法配置
    ok = make_experiment("EXP-0003", rolling_cv={
        "enabled": True, "mode": "sliding",
        "initial_train_fraction": 0.4, "horizon_seconds": 21600,
    })
    assert ok.validation.rolling_cv.enabled is True
    assert ok.validation.rolling_cv.mode == "sliding"


# ---------------------------------------------------------------- Runner 集成


@pytest.fixture()
def env(tmp_path):
    vault, ref = build_chiller_vault(tmp_path)
    ledger = ResearchLedger(tmp_path / "research")
    ledger.create_goal("g", actor=ACTOR)
    ledger.register_view(view_definition(ref), actor=ACTOR)
    ledger.create_hypothesis("RG-0001", "h", actor=ACTOR)
    return tmp_path, ledger


def _run(env, exp_id, **kwargs):
    tmp_path, ledger = env
    experiment = make_experiment(exp_id, **kwargs)
    ledger.register_experiment(
        experiment.model_dump(by_alias=True, mode="json"), actor=ACTOR)
    return run_experiment(
        experiment, research_root=tmp_path / "research",
        vault_root=tmp_path / "vault", ledger=ledger, actor=ACTOR,
        purge_seconds=0.0, embargo_seconds=2700.0)


def test_runner_rolling_cv_end_to_end(env, tmp_path):
    rolling_cv = {
        "enabled": True, "mode": "expanding",
        "initial_train_fraction": 0.4, "horizon_seconds": 6 * 3600,
        "step_seconds": 6 * 3600, "max_folds": 5,
    }
    report = _run(env, "EXP-0001", rolling_cv=rolling_cv)
    assert report["status"] == "completed", report.get("error")
    rolling = report["rolling_cv"]
    assert rolling is not None
    assert len(rolling["folds"]) >= 2
    assert rolling["config"]["n_folds"] == len(rolling["folds"])
    # 逐 fold 指标 + 跨 fold 聚合，口径与 holdout 一致（micro、NMBE 必报）
    for fold in rolling["folds"]:
        assert fold["n_train"] > 0
        assert fold["metrics"]["n_samples"] > 0
        assert "NMBE" in fold["metrics"]["metrics"]
        assert fold["metrics"]["per_object"]  # micro 默认 + per-object 明细
    agg = rolling["aggregate"]["per_metric"]
    assert agg["CVRMSE"]["n_defined"] == len(rolling["folds"])
    exp_dir = tmp_path / "research" / "experiments" / "EXP-0001"
    assert (exp_dir / "rolling_cv.json").exists()  # 边界落盘进制品
    doc = json.loads((exp_dir / "rolling_cv.json").read_text(encoding="utf-8"))
    assert doc["folds"][0]["boundaries"]["train_range"]


def test_runner_rolling_cv_reproducible(env):
    """同一定义两跑：rolling 聚合指标 bit-exact（§7.3 机器判定）。"""
    rolling_cv = {
        "enabled": True, "initial_train_fraction": 0.4,
        "horizon_seconds": 6 * 3600,
    }
    ra = _run(env, "EXP-0001", rolling_cv=rolling_cv)
    rb = _run(env, "EXP-0002", rolling_cv=rolling_cv)
    assert ra["status"] == rb["status"] == "completed"
    verify_reproducibility(ra["rolling_cv"], rb["rolling_cv"])
    assert json.dumps(ra["rolling_cv"], sort_keys=True) == json.dumps(
        rb["rolling_cv"], sort_keys=True)


def _late_start_env(tmp_path):
    """造一个「对象晚投运」的 vault：CH-02 前半段全空。

    真实建模集的时间轴是全部对象的并集，单对象视图在该对象投运之前
    一行都没有 —— 这不是缺陷，是覆盖事实。
    """
    import phase2_helpers as ph
    from thermoforge_data.importer import import_parsed
    from thermoforge_data.vault import DataVault

    n = ph.N_STEPS
    manifest = ph.TfdcManifest(
        contract="TFDC", contract_version="1.0", dataset_id="TEST02_LATE",
        dataset_version=1, site_id="T02", timezone="Asia/Shanghai",
        time_resolution="15min")
    units = {"evap_chw_flow": "m3/h", "evap_chw_supply_temp": "Cel",
             "evap_chw_return_temp": "Cel", "cw_supply_temp": "Cel",
             "input_power": "kW"}
    objects = [ph.ObjectRecord(object_id=o, object_model_id="chiller.v1")
               for o in ph.OBJECTS]
    variables = [
        ph.VariableRecord(
            variable_id=f"{o}.{p}", object_id=o, property_code=p, unit=u,
            dtype="float", role="target" if p == ph.TARGET else "state",
            source_kind="measured")
        for o in ph.OBJECTS for p, u in units.items()]
    ts = [ph.datetime(2026, 3, 1, tzinfo=UTC) + ph.timedelta(minutes=15 * i)
          for i in range(n)]
    columns = {}
    for k, obj in enumerate(ph.OBJECTS):
        for prop, values in ph._series(n, offset=20.0 * k).items():
            # CH-02 前 60% 时间轴无数据（晚投运）
            columns[f"{obj}.{prop}"] = (
                [None] * (n * 6 // 10) + values[n * 6 // 10:]
                if obj == "CH-02" else values)
    result = import_parsed(
        ph.TfdcDataset(manifest=manifest, objects=objects,
                       variables=variables), ts, columns)
    assert result.ok, result.diagnostics
    ref = DataVault(tmp_path / "vault").store(result)
    ledger = ResearchLedger(tmp_path / "research")
    ledger.create_goal("g", actor=ACTOR)
    view = view_definition(ref)
    view["objects"] = ["CH-02"]          # 单对象视图：早期折必空
    ledger.register_view(view, actor=ACTOR)
    ledger.create_hypothesis("RG-0001", "h", actor=ACTOR)
    return tmp_path, ledger


def test_rolling_cv_skips_empty_folds_instead_of_aborting(tmp_path):
    """早期折没有样本时跳过并入账，不让整个实验挂掉（TFX-905）。"""
    env2 = _late_start_env(tmp_path)
    report = _run(env2, "EXP-0001", rolling_cv={
        "enabled": True, "mode": "expanding", "initial_train_fraction": 0.2,
        "horizon_seconds": 6 * 3600, "step_seconds": 6 * 3600, "max_folds": 8,
    })
    assert report["status"] == "completed", report.get("error")
    rolling = report["rolling_cv"]
    assert rolling["n_folds_skipped"] > 0, "构造的数据本应产生空折"
    assert rolling["n_folds_evaluated"] > 0
    skipped = [f for f in rolling["folds"] if f.get("skipped")]
    # 训练窗空与评估窗空都算空折，两者都不该产生指标
    assert all(f["metrics"] is None for f in skipped)
    assert all(f["n_train"] == 0 or f["n_eval"] == 0 for f in skipped)
    # 聚合只统计跑过的折，不把空折算成 0 分
    agg = rolling["aggregate"]["per_metric"]["CVRMSE"]
    assert agg["n_defined"] == rolling["n_folds_evaluated"]


def test_runner_without_rolling_cv_unchanged(env, tmp_path):
    """向后兼容：不配 rolling_cv 的旧定义行为不变（无 rolling 制品）。"""
    report = _run(env, "EXP-0001")
    assert report["status"] == "completed", report.get("error")
    assert report["rolling_cv"] is None
    assert report["artifacts"]["rolling_cv"] is None
    exp_dir = tmp_path / "research" / "experiments" / "EXP-0001"
    assert not (exp_dir / "rolling_cv.json").exists()
    assert (exp_dir / "metrics.json").exists()  # holdout 制品照常
