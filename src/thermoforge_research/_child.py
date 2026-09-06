"""Experiment Runner 的子进程入口（implementation-notes §7.1）。

由 `runner.run_experiment` 以 `python -m thermoforge_research._child spec.json`
启动；线程数环境变量与 PYTHONHASHSEED 已被父进程预置（import 前生效）。
本进程内不得假定任何随机源未固定：种子清单一并写出（§7.2）。

契约性失败（TFX-9xx）写 child_result.json（status=failed + error_code）
并以退出码 2 结束；未捕获异常写 stderr 并以退出码 1 结束。
"""

from __future__ import annotations

import json
import os
import random
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from thermoforge_data.vault import DataVault
from thermoforge_data.views import materialize_view
from thermoforge_models.baseline import LinearBaseline
from thermoforge_models.hybrid import ResidualHybrid
from thermoforge_models.identification import EffectivenessNTU, GordonNgChiller
from thermoforge_models.physics import (
    ChillerPhysicsModel,
    ChillerPhysicsV2,
)
from thermoforge_research.errors import ResearchError
from thermoforge_research.metrics import aggregate_fold_metrics, compute_metrics
from thermoforge_research.physics_checks import (
    check_hard_constraints,
    check_monotonicity,
    combine_reports,
)
from thermoforge_research.runner import current_environment_lock
from thermoforge_research.split_profile import build_split_profile
from thermoforge_research.splits import (
    check_no_leakage,
    rolling_origin_splits,
    temporal_split,
)
from thermoforge_core.timeutil import parse_time_resolution

MONOTONE_GRID_POINTS = 21  # 单调性受控扰动扫描的点数（§6.3）


def _write_json(path: Path, doc: Any) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fp:
        json.dump(doc, fp, ensure_ascii=False, sort_keys=True, indent=2,
                  allow_nan=False)
        fp.write("\n")
    os.replace(tmp, path)


def _parse_monotone(raw: Any) -> dict[str, int]:
    """hyperparameters 中的单调性约束字符串："feat:1;feat2:-1"。"""
    if not raw:
        return {}
    out: dict[str, int] = {}
    for item in str(raw).split(";"):
        name, _, direction = item.partition(":")
        out[name.strip()] = int(direction)
    return out


def _parse_inputs(raw: Any) -> dict[str, str] | None:
    """physics 输入列映射字符串："chw_flow=evap_chw_flow;cw_supply_temp=..."。

    （Experiment 契约的 hyperparameters 只支持标量，映射以字符串编码。）

    分隔符 `;` 与 `,` 都接受；条目省略 `=` 时视为**同名映射**
    （`"cooling_load,t_cond_in"` ≡ `"cooling_load=cooling_load;t_cond_in=t_cond_in"`）。
    容忍这两种写法是刻意的：列名与角色名相同是最常见的情形，强制写成
    `a=a;b=b` 只会制造无谓的失败（实测 agent 连续三次栽在这里）。
    """
    if not raw:
        return None
    out: dict[str, str] = {}
    for chunk in str(raw).replace(",", ";").split(";"):
        item = chunk.strip()
        if not item:
            continue
        key, sep, col = item.partition("=")
        key, col = key.strip(), col.strip()
        out[key] = col if (sep and col) else key
    return out or None


def _build_physics(hp: Mapping[str, Any],
                   physics: str | None = None) -> ChillerPhysicsModel:
    """按方程版本分派（v1 保留兼容，v2 为 DOE-2 三曲线）。"""
    if "rated_capacity_kw" not in hp:
        raise ValueError("physics 模型缺少超参 rated_capacity_kw（额定参数必须显式声明）")
    # 键必须与 model_catalog.PHYSICS_BALANCE 一致（test_model_catalog 校验）
    cls = {
        None: ChillerPhysicsModel,
        "cooling_balance_v1": ChillerPhysicsModel,
        "cooling_balance_v2": ChillerPhysicsV2,
    }.get(physics)
    if cls is None:
        raise ValueError(f"未知物理方程版本: {physics!r}")
    return cls(
        rated_capacity_kw=float(hp["rated_capacity_kw"]),
        rated_power_kw=(
            float(hp["rated_power_kw"]) if hp.get("rated_power_kw") else None
        ),
        inputs=_parse_inputs(hp.get("inputs")),
    )


# 系统辨识模型族：参数少、可解释、单调性由方程结构保证（identification.py）
# 键必须与 model_catalog.PHYSICS_IDENTIFICATION 一致（test_model_catalog 校验）
_IDENT_FAMILIES = {
    "gordon_ng": GordonNgChiller,
    "eps_ntu": EffectivenessNTU,
}


def _build_identification(physics: str, hp: Mapping[str, Any]):
    cls = _IDENT_FAMILIES[physics]
    kwargs: dict[str, Any] = {"inputs": _parse_inputs(hp.get("inputs")) or None}
    if physics == "gordon_ng" and "q_floor" in hp:
        kwargs["q_floor"] = float(hp["q_floor"])
    if physics == "eps_ntu":
        kwargs["n_units"] = hp.get("n_units", "run_count") or None
    return cls(**kwargs)


def _load_lab_module(spec: Mapping[str, Any], exp_dir: Path):
    """按 spec.lab_module 加载冻结在实验目录里的实验室模块源码。

    校验 content_hash（快照完整性）；实验只依赖实验目录内的源码，
    与实验室存储脱钩（runner._resolve_lab_module 已在父进程过可运行门禁）。
    """
    info = spec.get("lab_module")
    if not info:
        return None
    import hashlib
    import importlib.util

    path = exp_dir / str(info["path"])
    with open(path, "rb") as fp:
        digest = hashlib.sha256(fp.read()).hexdigest()
    if digest != info.get("content_hash"):
        raise ResearchError(
            "TFX-902",
            f"实验室模块快照哈希不符: {info['name']}@v{info['version']} "
            f"（期望 {info.get('content_hash')!r}，实际 {digest!r}）",
        )
    module_spec = importlib.util.spec_from_file_location("tf_lab_frozen", path)
    if module_spec is None or module_spec.loader is None:
        raise ResearchError("TFX-902", f"无法加载实验室模块: {path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def _build_model(model_spec: Mapping[str, Any], seed: int,
                 lab_module: Any | None = None):
    category = model_spec["category"]
    hp = model_spec.get("hyperparameters", {})
    if category == "data":
        estimator = model_spec.get("estimator") or model_spec.get("residual")
        if estimator in ("linear", "ridge"):
            return LinearBaseline(method=str(estimator),
                                  alpha=float(hp.get("alpha", 1.0)))
        raise ResearchError("TFX-902", f"未支持的 data estimator: {estimator!r}")
    if category == "physics":
        physics = model_spec.get("physics")
        if physics in _IDENT_FAMILIES:
            return _build_identification(str(physics), hp)
        return _build_physics(hp, physics)
    if category == "lab":
        if lab_module is None:
            raise ResearchError(
                "TFX-902", "lab 实验缺少冻结的实验室模块（spec.lab_module）")
        return lab_module.build_model(hp, seed=seed)
    if category == "hybrid":
        xgb_params = {
            k: hp[k] for k in
            ("n_estimators", "max_depth", "learning_rate", "subsample",
             "colsample_bytree")
            if k in hp
        }
        physics = model_spec.get("physics")
        base = (_build_identification(str(physics), hp)
                if physics in _IDENT_FAMILIES
                else _build_physics(hp, physics))
        return ResidualHybrid(
            base, seed=seed, nthread=1,
            xgb_params=xgb_params,
            monotone_constraints=_parse_monotone(hp.get("monotone_constraints")),
        )
    raise ResearchError("TFX-902", f"未支持的模型类别: {category!r}")


def _model_frame(df: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    """交给模型的列：视图特征 + 记账列，**绝不含目标列**。

    DD-16 的白名单门禁管的是「视图的特征 ⊆ 目标白名单」，管不到模型自己
    从 DataFrame 里伸手拿列。内置模型都按 feature_order / inputs 取列，
    不受影响；实验室模块拿到的是整张表，只要写一句「除 object_id/timestamp
    外都当特征」就会把目标列训进去——R² 立刻变成 0.99，而那是拿 y predict y。
    这类假精度正是 DD-16 存在的理由，因此在这里从结构上堵死：模型看不到
    的东西，就不可能用。
    """
    keep = [c for c in ("object_id", "timestamp") if c in df.columns]
    keep += [c for c in features if c in df.columns and c not in keep]
    return df[keep]


def _fit(model: Any, df: pd.DataFrame, y: np.ndarray,
         features: Sequence[str]) -> None:
    frame = _model_frame(df, features)
    if isinstance(model, LinearBaseline | ResidualHybrid):
        model.fit(frame, y, feature_order=features)
    else:
        model.fit(frame, y)


def _seed_manifest(seed: int) -> dict[str, Any]:
    """种子清单（§7.2）：逐一设置并记录。"""
    random.seed(seed)
    rng = np.random.default_rng(seed)  # 显式 Generator，不用全局 np.random.seed
    return {
        "random_seed": seed,
        "entries": {
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
            "random.seed": seed,
            "numpy.default_rng": seed,
            "sklearn.random_state": seed,
            "xgboost.seed": seed,
            "xgboost.nthread": 1,
            "temporal_split": "deterministic-time-boundary",  # 无随机
        },
        "thread_env": {
            k: os.environ.get(k)
            for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                      "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                      "VECLIB_MAXIMUM_THREADS")
        },
        "_rng_probe": float(rng.random()),  # 验证 Generator 可用且确定
    }


def run(spec_path: Path) -> dict[str, Any]:
    with open(spec_path, encoding="utf-8") as fp:
        spec = json.load(fp)
    exp = spec["experiment"]
    exp_dir = Path(spec["experiment_dir"])

    # ---- 种子（TFX-902）
    seed = (exp.get("runtime") or {}).get("random_seed")
    if seed is None:
        raise ResearchError("TFX-902", "实验未声明随机种子（runtime.random_seed）")
    manifest = _seed_manifest(int(seed))
    _write_json(exp_dir / "seed_manifest.json", manifest)

    # ---- 环境指纹（TFX-901）
    lock, env_doc = current_environment_lock()
    env_doc["code_version"] = spec.get("code_version")
    _write_json(exp_dir / "environment.json", env_doc)
    declared = (exp.get("runtime") or {}).get("environment_lock")
    if declared != lock:
        raise ResearchError(
            "TFX-901",
            f"运行环境与实验声明不符: declared={declared} actual={lock}",
        )

    # ---- 数据：View 物化（revision + view_hash 记录）
    view_definition = spec["view_definition"]
    vault = DataVault(spec["vault_root"])
    mv = materialize_view(vault, view_definition, spec["view_cache_root"])
    df = mv.table.to_pandas()
    features = [str(f) for f in view_definition["features"]]
    target = str(view_definition["target"])

    with open(Path(spec["vault_root"]) / "datasets"
              / str(view_definition["dataset"]).split("@")[0]
              / str(view_definition["dataset"]).split("@")[1]
              / "manifest.json", encoding="utf-8") as fp:
        manifest_doc = json.load(fp)
    resolution_seconds = parse_time_resolution(manifest_doc["time_resolution"])
    if view_definition.get("resolution"):
        resolution_seconds = parse_time_resolution(str(view_definition["resolution"]))

    # ---- 切分（时间边界 + purge/embargo；边界记录进制品）
    ts_unique = sorted(df["timestamp"].unique())
    ts_list = [pd.Timestamp(t).to_pydatetime() for t in ts_unique]
    split = temporal_split(
        ts_list, resolution_seconds,
        train=exp["validation"]["temporal_split"]["train"],
        validate=exp["validation"]["temporal_split"]["validate"],
        test=exp["validation"]["temporal_split"]["test"],
        purge_seconds=float(spec["purge_seconds"]),
        embargo_seconds=float(spec["embargo_seconds"]),
    )
    boundaries = dict(split.boundaries)
    boundaries["dataset"] = str(view_definition["dataset"])
    boundaries["view_hash"] = mv.view_hash
    _write_json(exp_dir / "split.json", boundaries)

    holdout = (exp["validation"].get("equipment_holdout") or {})
    holdout_objects = list(holdout.get("holdout_objects") or [])
    if holdout.get("enabled") and not holdout_objects:
        raise ValueError("equipment_holdout.enabled 但未给出 holdout_objects")
    holdout_set = set(holdout_objects)

    # ---- 泄漏自检（TFX-903）
    check_no_leakage(ts_list, split.train_idx, split.validate_idx,
                     min_gap_seconds=float(spec["purge_seconds"]),
                     eval_label="验证集")
    check_no_leakage(ts_list, split.train_idx, split.test_idx,
                     min_gap_seconds=float(spec["purge_seconds"]),
                     eval_label="测试集")

    # 按边界区间划分行（train/validate 左闭右开；test 含 t_end）
    from thermoforge_core.timeutil import parse_timestamp

    b = split.boundaries
    tr_lo, tr_hi = (parse_timestamp(b["train_range"][0]),
                    parse_timestamp(b["train_range"][1]))
    va_lo, va_hi = (parse_timestamp(b["validate_range"][0]),
                    parse_timestamp(b["validate_range"][1]))
    te_lo = parse_timestamp(b["test_range"][0])
    ts_col = df["timestamp"]
    is_holdout = df["object_id"].astype(str).isin(holdout_set)
    subsets = {
        "train": df[~is_holdout & (ts_col >= tr_lo) & (ts_col < tr_hi)],
        "validate": df[~is_holdout & (ts_col >= va_lo) & (ts_col < va_hi)],
        "test": df[~is_holdout & (ts_col >= te_lo)],
    }

    if holdout_set:
        seen_train_objects = set(
            subsets["train"]["object_id"].astype(str).unique()
        )
        overlap = seen_train_objects & holdout_set
        if overlap:
            raise ResearchError(
                "TFX-903", f"留一设备验证下训练集包含留出设备: {sorted(overlap)}"
            )

    needed = features + [target]
    dropped = {}
    for name, sub in subsets.items():
        before = len(sub)
        subsets[name] = sub.dropna(subset=needed)
        dropped[name] = before - len(subsets[name])
    if not len(subsets["train"]):
        raise ResearchError("TFX-905", "训练集清洗后无样本")

    # ---- 切分子集分布画像（训练/验证/测试各段的自变量+目标分布对比,
    # 供人工/Agent 判断是否存在工况覆盖断层；只算数字不下结论）
    _write_json(exp_dir / "split_profile.json",
                build_split_profile(subsets, needed))

    # ---- 训练
    lab_module = _load_lab_module(spec, exp_dir)
    model = _build_model(exp["model"], int(seed), lab_module=lab_module)
    train_df = subsets["train"]
    y_train = train_df[target].to_numpy(np.float64)
    _fit(model, train_df, y_train, features)
    model_dir = exp_dir / "model"
    # 目录必须先建好：`_lab_check` 在调 save() 前是 mkdir 过的，这里不建
    # 就意味着「过了五连检的模块仍可能在真实实验里 FileNotFoundError」——
    # 五连检的承诺是「过了就能跑」，两边的前置条件必须一致（实测 EXP-0100）。
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_dir)
    if lab_module is not None:
        # 发布包自包含：lab 源码随模型制品走（artifact/lab_source.py），
        # 冷加载冒烟不再依赖实验室存储
        import shutil

        shutil.copy2(exp_dir / str(spec["lab_module"]["path"]),
                     model_dir / "lab_source.py")

    # ---- 评估：validate / test（面 A）+ 设备留出面 B/C
    y_floor = spec.get("y_floor")
    metrics_doc: dict[str, Any] = {"dropped_na_rows": dropped, "surfaces": {}}
    prediction_frames: list[pd.DataFrame] = []

    surfaces: dict[str, pd.DataFrame] = {
        "validate": subsets["validate"], "A": subsets["test"],
    }
    if holdout_set:
        # 面 B：未见设备 × 已见时间（< 测试起点）；面 C：未见设备 × 未来
        held = df[is_holdout].dropna(subset=needed)
        held_ts = ts_col.loc[held.index]
        surfaces["B"] = held[held_ts < te_lo]
        surfaces["C"] = held[held_ts >= te_lo]

    for surface, sub in surfaces.items():
        if not len(sub):
            metrics_doc["surfaces"][surface] = {"n_samples": 0, "metrics": {}}
            continue
        y_true = sub[target].to_numpy(np.float64)
        y_pred = np.asarray(
            model.predict(_model_frame(sub, features)), dtype=np.float64)
        report = compute_metrics(
            y_true.tolist(), y_pred.tolist(), exp["metrics"],
            y_floor=y_floor,
            object_ids=sub["object_id"].astype(str).tolist(),
        )
        metrics_doc["surfaces"][surface] = report.to_dict()
        prediction_frames.append(pd.DataFrame({
            "surface": surface,
            "object_id": sub["object_id"].astype(str).to_numpy(),
            "timestamp": sub["timestamp"].to_numpy(),
            "y_true": y_true,
            "y_pred": y_pred,
        }))

    # DD-07 配套：残差混合必须同时报告物理主干单独的指标与残差占比
    if isinstance(model, ResidualHybrid):
        physics_only: dict[str, Any] = {}
        for surface, sub in surfaces.items():
            if not len(sub):
                continue
            y_true = sub[target].to_numpy(np.float64)
            y_pred = np.asarray(model.physics_only(sub), dtype=np.float64)
            physics_only[surface] = compute_metrics(
                y_true.tolist(), y_pred.tolist(), exp["metrics"],
                y_floor=y_floor,
            ).to_dict()
        metrics_doc["physics_only"] = physics_only
        metrics_doc["residual_share"] = {
            surface: model.residual_share(sub)
            for surface, sub in surfaces.items() if len(sub)
        }

    if prediction_frames:
        pd.concat(prediction_frames).to_parquet(
            exp_dir / "predictions.parquet", compression="zstd", index=False
        )

    # ---- 滚动原点 CV（可选增强验证，research-loop §6；逐 fold 训练+评估）
    rolling_doc: dict[str, Any] | None = None
    rolling_cfg = (exp["validation"].get("rolling_cv") or {})
    if rolling_cfg.get("enabled"):
        rolling = rolling_origin_splits(
            ts_list, resolution_seconds,
            initial_train_fraction=rolling_cfg.get("initial_train_fraction"),
            initial_train_seconds=rolling_cfg.get("initial_train_seconds"),
            horizon_seconds=float(rolling_cfg["horizon_seconds"]),
            step_seconds=(
                float(rolling_cfg["step_seconds"])
                if rolling_cfg.get("step_seconds") else None
            ),
            mode=str(rolling_cfg.get("mode", "expanding")),
            max_folds=int(rolling_cfg.get("max_folds", 10)),
            purge_seconds=float(spec["purge_seconds"]),
            embargo_seconds=float(spec["embargo_seconds"]),
        )
        t_end_ts = pd.Timestamp(ts_list[-1])
        fold_reports = []
        fold_entries: list[dict[str, Any]] = []
        for fold in rolling.folds:
            fb = fold.boundaries
            ftr_lo = parse_timestamp(fb["train_range"][0])
            ftr_hi = parse_timestamp(fb["train_range"][1])
            fev_lo = parse_timestamp(fb["eval_range"][0])
            fev_hi = parse_timestamp(fb["eval_range"][1])
            # 评估窗达到数据末尾时含 t_end（与 splits 的口径一致）
            inclusive_end = pd.Timestamp(fev_hi) >= t_end_ts
            ftrain = df[~is_holdout & (ts_col >= ftr_lo) & (ts_col < ftr_hi)]
            feval = df[
                ~is_holdout & (ts_col >= fev_lo)
                & ((ts_col <= t_end_ts) if inclusive_end else (ts_col < fev_hi))
            ]
            ftrain = ftrain.dropna(subset=needed)
            feval = feval.dropna(subset=needed)
            # 空折跳过而不是中止：数据集的时间轴是全部对象的并集，
            # 单对象视图在该对象还没投运的早期折里一行都没有 —— 这是
            # 数据覆盖的事实，不是缺陷。以前它让 compute_metrics 报
            # TFX-905，整个实验白跑（实测 EXP-0025）。
            # 空折照样入账（n_train/n_eval + skipped），覆盖缺口在
            # 产物里看得见；全部折皆空才是真失败，在循环后判。
            if not len(ftrain) or not len(feval):
                fold_entries.append({
                    "fold": fold.fold, "boundaries": fb,
                    "n_train": len(ftrain), "n_eval": len(feval),
                    "skipped": "empty_fold", "metrics": None,
                })
                continue
            fold_model = _build_model(exp["model"], int(seed),
                                      lab_module=lab_module)
            _fit(fold_model, ftrain,
                 ftrain[target].to_numpy(np.float64), features)
            y_true = feval[target].to_numpy(np.float64)
            y_pred = np.asarray(
                fold_model.predict(_model_frame(feval, features)),
                dtype=np.float64)
            report = compute_metrics(
                y_true.tolist(), y_pred.tolist(), exp["metrics"],
                y_floor=y_floor,
                object_ids=feval["object_id"].astype(str).tolist(),
            )
            fold_reports.append(report)
            fold_entries.append({
                "fold": fold.fold,
                "boundaries": fb,
                "n_train": len(ftrain),
                "n_eval": len(feval),
                "metrics": report.to_dict(),
            })
        if not fold_reports:
            raise ResearchError(
                "TFX-905",
                f"rolling_cv 全部 {len(fold_entries)} 折都没有可用样本："
                "检查视图的对象范围与数据集时间轴是否匹配",
            )
        rolling_doc = {
            "config": rolling.config,
            "folds": fold_entries,
            "n_folds_evaluated": len(fold_reports),
            "n_folds_skipped": len(fold_entries) - len(fold_reports),
            "aggregate": aggregate_fold_metrics(fold_reports),
            "note": "每个 fold 独立训练同一模型定义（同一种子），"
                    "仅训练窗口不同；fold 模型不落盘",
        }
        _write_json(exp_dir / "rolling_cv.json", rolling_doc)
        # 折间均值也写进 metrics：验收若判在滚动交叉验证上，模型包必须
        # 自带这个数 —— 门禁只读包内 metrics.json，不回头翻实验目录。
        # 放在 surfaces 之外：它不是一个「面」，n_samples 是各折评估集之和。
        metrics_doc["rolling_cv"] = {
            "n_folds": len(fold_reports),
            "n_folds_skipped": rolling_doc["n_folds_skipped"],
            "n_samples": sum(int(e.get("n_eval") or 0) for e in fold_entries),
            "metrics": {name: agg["mean"] for name, agg
                        in rolling_doc["aggregate"]["per_metric"].items()
                        if agg.get("mean") is not None},
        }

    # metrics.json 必须在滚动块之后落盘：验收可判在 rolling_cv 上，
    # 制品与 report 里的 metrics 必须是同一份内容
    _write_json(exp_dir / "metrics.json", metrics_doc)

    # ---- 物理验证（§6）
    physics_doc: dict[str, Any] | None = None
    if (exp.get("physics_tests") or {}).get("enabled"):
        eval_df = subsets["test"] if len(subsets["test"]) else subsets["validate"]
        if len(eval_df):
            y_pred = np.asarray(
                model.predict(_model_frame(eval_df, features)),
                dtype=np.float64)
            rated_power = (exp["model"].get("hyperparameters") or {}).get(
                "rated_power_kw"
            )
            check_df = eval_df.copy()
            check_df["__pred_power__"] = y_pred
            kwargs: dict[str, Any] = {
                "power_col": "__pred_power__",
                "rated_power": float(rated_power) if rated_power else None,
            }
            # 只有能量平衡族才提供 cooling_capacity/condenser_col 这套接口。
            # 系统辨识族（gordon_ng/eps_ntu）没有，硬检查降级为只查功率。
            # 判据必须落在**解包后**的物理模型上：hybrid 包住 gordon_ng 时，
            # 外层 ResidualHybrid 会通过 isinstance，内层却没有这些方法，
            # 于是 hybrid+gordon_ng 每次都在物理检查处崩掉（实测 EXP-0040~0043）。
            physics_model = (model.physics if isinstance(model, ResidualHybrid)
                             else model)
            if isinstance(physics_model, ChillerPhysicsModel):
                check_df["__cooling__"] = physics_model.cooling_capacity(eval_df)
                kwargs["cooling_col"] = "__cooling__"
                kwargs["chw_supply_col"] = physics_model.inputs["chw_supply_temp"]
                # I-40 修正：冷凝温度用模型辨识选定的代理列（v2）；
                # v1 仍为冷却水供水温度近似
                kwargs["cw_return_col"] = physics_model.condenser_col
                # v2 的 rated_power_kw 是单台额定：逐样本乘以 run_count
                if (rated_power and getattr(physics_model, "per_unit_rated", False)
                        and physics_model.inputs.get("run_count") in eval_df.columns):
                    check_df["__rated__"] = (
                        float(rated_power)
                        * pd.to_numeric(eval_df[physics_model.inputs["run_count"]])
                    )
                    kwargs["rated_power"] = "__rated__"
            hard = check_hard_constraints(check_df, **kwargs)

            mono_results: dict[str, Any] = {}
            monotone = getattr(model, "monotone_constraints", None) or {}
            if monotone and isinstance(model, ResidualHybrid):
                base_row = {
                    f: float(np.nanmedian(
                        pd.to_numeric(train_df[f], errors="coerce")))
                    for f in model.feature_order
                }
                for feat, direction in monotone.items():
                    col = pd.to_numeric(train_df[feat], errors="coerce")
                    grid = np.linspace(float(col.min()), float(col.max()),
                                       MONOTONE_GRID_POINTS)
                    result = check_monotonicity(
                        model.predict, base_row, feat, grid.tolist(),
                        direction=int(direction),
                    )
                    mono_results[result.name] = result
            combined = combine_reports(hard, mono_results)
            physics_doc = combined.to_dict()
            physics_doc["condenser_col"] = kwargs.get("cw_return_col")
            physics_doc["note"] = (
                "cop_below_carnot 冷凝温度列见 condenser_col"
                "（v2 为辨识选定的代理，v1 为冷却水供水近似，I-40）"
            )
            _write_json(exp_dir / "physics_report.json", physics_doc)

    artifacts = {
        "model_dir": "model/",
        "metrics": "metrics.json",
        "predictions": "predictions.parquet",
        "split": "split.json",
        "seed_manifest": "seed_manifest.json",
        "environment": "environment.json",
        "physics_report": (
            "physics_report.json" if physics_doc is not None else None
        ),
        "rolling_cv": "rolling_cv.json" if rolling_doc is not None else None,
    }
    conclusion = {
        "summary": "实验完成",
        "dataset_revision": str(view_definition["dataset"]),
        "view_hash": mv.view_hash,
        "view_cache_reused": mv.reused,
        "surfaces": {
            name: surf.get("metrics", {})
            for name, surf in metrics_doc["surfaces"].items()
        },
    }
    return {
        "status": "completed",
        "artifacts": artifacts,
        "metrics": metrics_doc,
        "physics": physics_doc,
        "rolling_cv": rolling_doc,
        "conclusion": conclusion,
        "next_questions": [],
    }


def main(argv: Sequence[str]) -> int:
    spec_path = Path(argv[1] if len(argv) > 1 else "spec.json")
    exp_dir = spec_path.resolve().parent
    try:
        result = run(spec_path)
    except ResearchError as exc:
        _write_json(exp_dir / "child_result.json", {
            "status": "failed", "error_code": exc.code, "error": str(exc),
        })
        print(str(exc), file=sys.stderr)
        return 2
    except Exception:
        traceback.print_exc()
        _write_json(exp_dir / "child_result.json", {
            "status": "failed", "error_code": "CHILD_EXCEPTION",
            "error": traceback.format_exc(limit=3),
        })
        return 1
    _write_json(exp_dir / "child_result.json", result)
    print(f"experiment completed: {result['conclusion']['dataset_revision']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
