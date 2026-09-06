"""误差剖面分析：同一个模型在不同切分口径、不同工况档、不同月份下的误差。

回答三个问题：

1. **误差在不同工况上是多少** —— 按负荷档（目标分位）与驱动量档
   （PLR / 频率 / 湿球温 / 驱动温差，逐模型指定）分箱统计。
2. **误差会不会随时间变大** —— 按站点本地月份分箱统计。
3. **发布口径的「测试面」为什么误差大** —— 同一模型、同一超参、同一种子，
   在三种留出协议下各跑一遍，把「模型能力不足」和「外推到未见时段」分开：

   | 协议 | 训练/评估关系 | 回答什么 |
   |---|---|---|
   | `walk_forward` | 扩展窗因果滚动（发布判据面的形态） | 真·预测未来的水平 |
   | `time_block`   | 连续时间块逐块留出（非因果） | 去掉「只能看过去」这一条后还剩多少误差 |
   | `random`       | 行随机 K 折（非因果，且相邻样本会互相泄漏） | 纯插值上限（乐观） |

   `random` 的数值必然最好看，但 15 分钟序列相邻两行几乎相同，随机切分让
   训练集里躺着测试点的「邻居」——这部分好看是自相关泄漏换来的，不是外推
   能力。`time_block` 是两者之间诚实的中间值。

模型的重建完全复用实验规格：`_child._build_model` + 冻结的实验室模块，
超参、种子、特征列都取自 `research/experiments/<EXP>/spec.json`，
指标一律走 `research/metrics.py` 的唯一实现。

用法：
    .venv/Scripts/python scripts/analyze_model_errors.py [--models a,b] [--folds 5]
产物：
    research/tool_artifacts/analysis/model_error_profile.json
    research/tool_artifacts/analysis/oos_predictions/<model_id>.parquet
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

# 线程数必须在 numpy/sklearn/xgboost import 之前定死，否则规约顺序会漂，
# 与实验运行器同一条规矩（implementation-notes §7.1）。
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
os.environ.setdefault("PYTHONHASHSEED", "0")

import numpy as np                                            # noqa: E402
import pandas as pd                                           # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thermoforge_research import _child                       # noqa: E402
from thermoforge_research.metrics import compute_metrics      # noqa: E402
from thermoforge_research.splits import rolling_origin_splits  # noqa: E402

MODELS_ROOT = ROOT / "models"
RESEARCH_ROOT = ROOT / "research"
VIEW_CACHE = RESEARCH_ROOT / "view_cache"
OUT_DIR = RESEARCH_ROOT / "tool_artifacts" / "analysis"

SITE_TZ_OFFSET_HOURS = 8          # 数据契约声明的站点时区 Asia/Shanghai
METRIC_NAMES = ("RMSE", "MAE", "MAPE", "CVRMSE", "NMBE", "R2")
PROTOCOLS = ("walk_forward", "time_block", "random")

# 逐模型的「驱动量」：分档看误差时最有物理意义的那一列 + 固定档位边界。
# 固定边界而不是分位数：档位要能跨模型、跨版本对齐，分位数会随数据漂。
DRIVERS: dict[str, tuple[str, str, tuple[float, ...]]] = {
    "wx-chiller-power-ch01": ("plr", "部分负荷率 PLR", (0.30, 0.50, 0.60, 0.70)),
    "wx-chiller-power-ch02": ("plr", "部分负荷率 PLR", (0.30, 0.50, 0.60, 0.70)),
    "wx-chiller-power-ch03": ("plr", "部分负荷率 PLR", (0.30, 0.50, 0.60, 0.70)),
    "wx-chiller-power-ch04": ("plr", "部分负荷率 PLR", (0.30, 0.50, 0.60, 0.70)),
    "wx-chiller-power-static": ("plr", "部分负荷率 PLR", (0.30, 0.50, 0.60, 0.70)),
    "wx-hx-heat-transfer": ("drive_dt", "驱动温差 K", (5.0, 6.5, 7.5, 8.5)),
    "wx-hx-heat-transfer-static": ("drive_dt", "驱动温差 K", (5.0, 6.5, 7.5, 8.5)),
    "wx-chwp-flow-total": ("freq_mean", "水泵平均频率 Hz", (33.0, 35.0, 40.0, 43.0)),
    "wx-pump-power-unit": ("frequency", "本台频率 Hz", (30.0, 35.0, 40.0, 45.0)),
    "wx-ct-fan-power-unit": ("frequency", "本台频率 Hz", (30.0, 35.0, 40.0, 45.0)),
    "wx-ct-fan-power-total": ("fan_freq_mean", "风机平均频率 Hz",
                              (30.0, 35.0, 40.0, 45.0)),
    "wx-ct-supply-temp": ("t_wetbulb", "室外湿球温度 ℃", (0.0, 8.0, 16.0, 22.0)),
    "wx-ct-approach": ("t_wetbulb", "室外湿球温度 ℃", (12.0, 16.0, 20.0, 23.0)),
}

LOAD_BAND_LABELS = ("低负荷", "中低负荷", "中负荷", "中高负荷", "高负荷")


# --------------------------------------------------------------- 装载


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


@dataclass
class Case:
    """一个待分析的模型：包，加上重建它所需要的实验规格与视图数据。"""

    model_id: str
    version: str
    experiment_id: str
    exp_dir: Path
    spec: dict
    frame: pd.DataFrame
    features: list[str]
    target: str

    @property
    def model_spec(self) -> dict:
        return self.spec["experiment"]["model"]

    @property
    def seed(self) -> int:
        return int(self.spec["experiment"]["runtime"]["random_seed"])

    @property
    def resolution_seconds(self) -> int:
        split = _read_json(self.exp_dir / "split.json")
        return int(split.get("resolution_seconds") or 900)

    @property
    def rolling_config(self) -> dict:
        return dict(self.spec["experiment"]["validation"]["rolling_cv"])


def load_cases(only: Sequence[str] | None = None) -> list[Case]:
    cases: list[Case] = []
    for registry_path in sorted(MODELS_ROOT.glob("*/registry.json")):
        registry = _read_json(registry_path)
        model_id = str(registry.get("model_id") or registry_path.parent.name)
        if only and model_id not in only:
            continue
        version = registry.get("production")
        if not version:
            continue
        package = registry_path.parent / str(version)
        lineage = _read_json(package / "research-lineage.json")
        experiment_id = str(lineage.get("experiment_id") or "")
        exp_dir = RESEARCH_ROOT / "experiments" / experiment_id
        spec = _read_json(exp_dir / "spec.json")
        if not spec:
            print(f"  跳过 {model_id}：读不到 {experiment_id}/spec.json")
            continue
        view = spec["view_definition"]
        cache_dir = VIEW_CACHE / str(view["view_hash"])[:16]
        data_path = cache_dir / "data.parquet"
        if not data_path.is_file():
            print(f"  跳过 {model_id}：视图缓存缺失 {cache_dir.name}")
            continue
        frame = pd.read_parquet(data_path)
        features = list(spec["experiment"].get("model", {}).get("features") or [])
        if not features:
            features = [c for c in view["features"] if c in frame.columns]
        target = str(view["target"])
        frame = frame.dropna(subset=features + [target]).reset_index(drop=True)
        frame = frame.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
        cases.append(Case(model_id=model_id, version=str(version),
                          experiment_id=experiment_id, exp_dir=exp_dir,
                          spec=spec, frame=frame, features=features,
                          target=target))
    return cases


# --------------------------------------------------------------- 协议


def _fit_predict(case: Case, train_idx: np.ndarray,
                 eval_idx: np.ndarray) -> np.ndarray:
    """按实验规格重建模型，在 train 上拟合，返回 eval 上的预测。"""
    lab_module = _child._load_lab_module(case.spec, case.exp_dir)
    model = _child._build_model(case.model_spec, case.seed,
                                lab_module=lab_module)
    train_df = case.frame.iloc[train_idx]
    _child._fit(model, train_df,
                train_df[case.target].to_numpy(np.float64), case.features)
    eval_df = case.frame.iloc[eval_idx]
    return np.asarray(
        model.predict(_child._model_frame(eval_df, case.features)),
        dtype=np.float64)


def protocol_walk_forward(case: Case) -> tuple[np.ndarray, dict]:
    """扩展窗因果滚动：训练只用评估窗之前的数据，覆盖时间轴后段。"""
    config = case.rolling_config
    result = rolling_origin_splits(
        list(case.frame["timestamp"]),
        case.resolution_seconds,
        initial_train_fraction=float(config.get("initial_train_fraction") or 0.3),
        horizon_seconds=float(config.get("horizon_seconds") or 604800.0),
        step_seconds=float(config.get("step_seconds")
                           or config.get("horizon_seconds") or 604800.0),
        mode=str(config.get("mode") or "expanding"),
        max_folds=None,                    # 不设上限：要覆盖到时间轴末尾
    )
    preds = np.full(len(case.frame), np.nan)
    for fold in result.folds:
        eval_idx = np.asarray(fold.eval_idx, dtype=int)
        train_idx = np.asarray(fold.train_idx, dtype=int)
        if not len(eval_idx) or len(train_idx) < 10:
            continue
        preds[eval_idx] = _fit_predict(case, train_idx, eval_idx)
    return preds, {"n_folds": len(result.folds),
                   "note": "扩展窗因果滚动，训练集只含评估窗之前的样本"}


def protocol_time_block(case: Case, n_folds: int) -> tuple[np.ndarray, dict]:
    """连续时间块逐块留出：块内是未见时段，但训练集含该块之后的数据。"""
    n = len(case.frame)
    edges = np.linspace(0, n, n_folds + 1).astype(int)
    preds = np.full(n, np.nan)
    for k in range(n_folds):
        eval_idx = np.arange(edges[k], edges[k + 1])
        train_idx = np.concatenate([np.arange(0, edges[k]),
                                    np.arange(edges[k + 1], n)])
        if not len(eval_idx) or len(train_idx) < 10:
            continue
        preds[eval_idx] = _fit_predict(case, train_idx, eval_idx)
    return preds, {"n_folds": n_folds,
                   "note": "连续时间块留出，非因果（训练集含该块之后的样本）"}


def protocol_random(case: Case, n_folds: int) -> tuple[np.ndarray, dict]:
    """行随机 K 折：相邻样本会落进训练集，指标偏乐观（自相关泄漏）。"""
    n = len(case.frame)
    order = np.random.default_rng(case.seed).permutation(n)
    preds = np.full(n, np.nan)
    for k in range(n_folds):
        eval_idx = order[k::n_folds]
        train_idx = np.setdiff1d(np.arange(n), eval_idx, assume_unique=False)
        if not len(eval_idx) or len(train_idx) < 10:
            continue
        preds[eval_idx] = _fit_predict(case, train_idx, eval_idx)
    return preds, {"n_folds": n_folds,
                   "note": "行随机 K 折，相邻时刻互相可见（自相关泄漏，偏乐观）"}


# --------------------------------------------------------------- 分箱


def load_bands(y: np.ndarray) -> pd.Series:
    """负荷档：目标值的五分位。跨模型统一口径，标签固定。"""
    quantiles = np.quantile(y, [0.2, 0.4, 0.6, 0.8])
    edges = np.unique(np.concatenate([[-np.inf], quantiles, [np.inf]]))
    labels = LOAD_BAND_LABELS[:len(edges) - 1]
    return pd.cut(y, bins=edges, labels=list(labels), include_lowest=True)


def driver_bands(values: np.ndarray, edges: Sequence[float]) -> pd.Series:
    cuts = [-np.inf, *edges, np.inf]
    labels = []
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        lo_text = "−∞" if lo == -np.inf else f"{lo:g}"
        hi_text = "+∞" if hi == np.inf else f"{hi:g}"
        labels.append(f"{lo_text}~{hi_text}")
    return pd.cut(values, bins=cuts, labels=labels, include_lowest=True)


def bin_metrics(frame: pd.DataFrame, column: str, target: str,
                y_floor: float, min_samples: int = 30) -> list[dict]:
    """按 `column` 分箱算指标。样本太少的箱只报数量，不报会误导的比率。"""
    rows: list[dict] = []
    grouped = frame.groupby(column, observed=True, sort=True)
    for key, part in grouped:
        entry: dict[str, Any] = {"bin": str(key), "n_samples": int(len(part)),
                                 "y_mean": float(part["y_true"].mean())}
        if len(part) >= min_samples:
            report = compute_metrics(
                part["y_true"].tolist(), part["y_pred"].tolist(), METRIC_NAMES,
                y_floor=y_floor, min_valid_fraction=0.0)
            entry.update(report.metrics)
            entry["mape_valid_fraction"] = report.mape_valid_fraction
        else:
            entry["too_few"] = True
        rows.append(entry)
    return rows


# --------------------------------------------------------------- 主流程


def analyse(case: Case, n_folds: int) -> tuple[dict, pd.DataFrame]:
    y = case.frame[case.target].to_numpy(np.float64)
    # y_floor 全局取一次：各箱共用同一门槛，MAPE 才可横向比较
    y_floor = 0.05 * float(np.max(np.abs(y)))
    local = (case.frame["timestamp"].dt.tz_convert(
        timezone.utc).dt.tz_localize(None)
        + pd.Timedelta(hours=SITE_TZ_OFFSET_HOURS))
    driver_col, driver_label, driver_edges = DRIVERS.get(
        case.model_id, (case.target, "目标值", ()))
    has_driver = driver_col in case.frame.columns

    base = pd.DataFrame({
        "object_id": case.frame["object_id"].astype(str),
        "timestamp": case.frame["timestamp"],
        "month": local.dt.strftime("%Y-%m"),
        "y_true": y,
        "load_band": load_bands(y).astype(str),
    })
    if has_driver:
        driver_values = case.frame[driver_col].to_numpy(np.float64)
        base["driver_value"] = driver_values
        base["driver_band"] = driver_bands(
            driver_values, driver_edges or (np.quantile(
                driver_values, [0.2, 0.4, 0.6, 0.8]).tolist())).astype(str)

    doc: dict[str, Any] = {
        "model_id": case.model_id, "version": case.version,
        "experiment_id": case.experiment_id, "target": case.target,
        "n_rows": int(len(case.frame)),
        "objects": sorted(base["object_id"].unique().tolist()),
        "time_range": [str(case.frame["timestamp"].min()),
                       str(case.frame["timestamp"].max())],
        "y_floor": y_floor,
        "y_mean": float(np.mean(y)),
        "driver": {"column": driver_col if has_driver else None,
                   "label": driver_label, "edges": list(driver_edges)},
        "protocols": {}, "bins": {"month": {}, "load_band": {},
                                  "driver_band": {}, "object": {}},
    }

    runners = {
        "walk_forward": lambda: protocol_walk_forward(case),
        "time_block": lambda: protocol_time_block(case, n_folds),
        "random": lambda: protocol_random(case, n_folds),
    }
    collected: list[pd.DataFrame] = []
    for name, runner in runners.items():
        started = time.perf_counter()
        preds, info = runner()
        mask = np.isfinite(preds)
        if not mask.any():
            doc["protocols"][name] = {"n_samples": 0, **info}
            continue
        part = base.loc[mask].copy()
        part["y_pred"] = preds[mask]
        part["protocol"] = name
        report = compute_metrics(part["y_true"].tolist(), part["y_pred"].tolist(),
                                 METRIC_NAMES, y_floor=y_floor,
                                 min_valid_fraction=0.0)
        doc["protocols"][name] = {
            **info,
            "n_samples": int(mask.sum()),
            "coverage": float(mask.sum() / len(case.frame)),
            "seconds": round(time.perf_counter() - started, 1),
            "metrics": report.metrics,
            "mape_valid_fraction": report.mape_valid_fraction,
        }
        for dimension in ("month", "load_band", "driver_band", "object"):
            column = "object_id" if dimension == "object" else dimension
            if column not in part.columns:
                continue
            if dimension == "object" and part["object_id"].nunique() < 2:
                continue
            doc["bins"][dimension][name] = bin_metrics(
                part, column, case.target, y_floor)
        collected.append(part)
        metrics = doc["protocols"][name]["metrics"]
        print(f"    {name:<13} n={int(mask.sum()):>7}  "
              f"CVRMSE={metrics.get('CVRMSE') or float('nan'):.4f}  "
              f"R2={metrics.get('R2') if metrics.get('R2') is not None else float('nan'):.4f}"
              f"  {doc['protocols'][name]['seconds']}s")
    predictions = (pd.concat(collected, ignore_index=True)
                   if collected else pd.DataFrame())
    return doc, predictions


def main() -> int:
    parser = argparse.ArgumentParser(description="模型误差剖面分析")
    parser.add_argument("--models", default="",
                        help="逗号分隔的 model_id，缺省跑全部生产版本")
    parser.add_argument("--folds", type=int, default=5,
                        help="time_block / random 的折数（默认 5）")
    args = parser.parse_args()

    only = [m.strip() for m in args.models.split(",") if m.strip()] or None
    cases = load_cases(only)
    if not cases:
        print("没有可分析的模型。")
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions_dir = OUT_DIR / "oos_predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    doc: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "site_tz_offset_hours": SITE_TZ_OFFSET_HOURS,
        "n_folds": args.folds,
        "protocol_notes": {
            "walk_forward": "扩展窗因果滚动（发布判据面的形态）",
            "time_block": "连续时间块逐块留出，非因果",
            "random": "行随机 K 折，非因果且相邻样本互相可见",
        },
        "models": {},
    }
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case.model_id}@{case.version} "
              f"({case.experiment_id}, {len(case.frame)} 行)")
        try:
            model_doc, predictions = analyse(case, args.folds)
        except Exception as error:                    # 单个模型失败不拖垮全局
            print(f"    失败：{type(error).__name__}: {error}")
            doc["models"][case.model_id] = {"error": f"{type(error).__name__}: {error}"}
            continue
        doc["models"][case.model_id] = model_doc
        if len(predictions):
            predictions.to_parquet(predictions_dir / f"{case.model_id}.parquet",
                                   index=False)
    out_path = OUT_DIR / "model_error_profile.json"
    out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2,
                                   allow_nan=True, sort_keys=True),
                        encoding="utf-8", newline="\n")
    print(f"写出 {out_path}")
    print(f"写出 {predictions_dir}/*.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
