"""简易入口：菜单式操作,不需要记命令行参数。

面向不熟悉命令行的使用者,把最常用的五件事做成数字菜单：
看状态 / 跑示例 / 导入 Excel / 训练模型 / 用模型预测。

启动方式：双击仓库根目录的 `启动.bat`,或::

    .venv/Scripts/python tools/menu.py

设计约束：

- 只调用 `thermoforge_research.tools` 的信封工具,不绕过校验管线——
  白名单(DD-16)与可建模性门禁(G3)在这里同样生效,失败时翻译成人话。
- 只覆盖「单对象 + 线性基线」这条最简路径。物理/混合模型需要额定容量
  等参数,不适合无引导填写,菜单会提示改用 `tf agent` 或示例脚本。
- 控制台强制 UTF-8(Windows 默认 cp1252 会让中文信封直接崩,§10.3)。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SEED = 20260808
DEFAULT_ACCEPTANCE = {"cvrmse_max": 0.13, "inference_latency_ms_max": 5.0}
LINE = "=" * 58
# 常见的「设备在运行」标志位名;用于把它和故障/维护标志区分开
RUN_FLAGS = ("status_run", "any_running", "running", "run", "is_running")


def _setup_console() -> None:
    """Windows 控制台默认 cp1252,中文输出会 UnicodeEncodeError(§10.3)。"""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            reconfigure(encoding="utf-8", errors="replace")


def _ctx():
    from thermoforge_research.tools import ToolContext

    return ToolContext(
        vault_root=REPO_ROOT / "vault",
        research_root=REPO_ROOT / "research",
        models_root=REPO_ROOT / "models",
        actor="menu",
    )


# ---------------------------------------------------------------- 输入辅助


def _ask(prompt: str, default: str = "") -> str:
    suffix = f"（直接回车 = {default}）" if default else ""
    try:
        text = input(f"{prompt}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return ""
    return text or default


def _ask_yes(prompt: str, default: bool = True) -> bool:
    hint = "Y/n" if default else "y/N"
    text = _ask(f"{prompt} [{hint}]").lower()
    if not text:
        return default
    return text in ("y", "yes", "是", "好")


def _pick(items: Sequence[Any], label: Callable[[Any], str],
          prompt: str) -> Any | None:
    """编号选择。返回选中项;空输入或非法输入返回 None。"""
    for i, item in enumerate(items, 1):
        print(f"  {i}. {label(item)}")
    text = _ask(prompt)
    if not text.isdigit() or not 1 <= int(text) <= len(items):
        print("  没选中任何项,回到菜单。")
        return None
    return items[int(text) - 1]


def _pick_many(items: Sequence[Any], label: Callable[[Any], str],
               prompt: str, default_idx: Sequence[int]) -> list[Any]:
    """多选(逗号分隔编号)。空输入取 default_idx。"""
    for i, item in enumerate(items, 1):
        print(f"  {i}. {label(item)}")
    default_text = ",".join(str(i + 1) for i in default_idx)
    text = _ask(prompt, default_text)
    picked: list[Any] = []
    for part in text.replace("，", ",").split(","):
        part = part.strip()
        if part.isdigit() and 1 <= int(part) <= len(items):
            item = items[int(part) - 1]
            if item not in picked:
                picked.append(item)
    return picked


def _clean_path(text: str) -> Path:
    """处理拖拽进来的路径：去掉首尾引号与空白。"""
    return Path(text.strip().strip('"').strip("'"))


def _pause() -> None:
    _ask("\n按回车返回菜单")


# ---------------------------------------------------------------- 1 状态


def show_status() -> None:
    from thermoforge_cli.status import build_status, render_text

    print(render_text(build_status(_ctx(), recent=5)))


# ---------------------------------------------------------------- 2 示例


def run_demo() -> None:
    script = REPO_ROOT / "examples" / "chiller_power" / "run_demo.py"
    print("即将用仓库自带的真实数据跑一遍完整流程。")
    print("首次运行要导入 43MB 的 Excel,约 1~2 分钟;之后会复用,几十秒。\n")
    if not _ask_yes("现在开始吗?"):
        return
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run([sys.executable, str(script)], cwd=str(REPO_ROOT),
                          env=env)
    if proc.returncode == 0:
        print("\n跑完了。实验报告在 examples\\chiller_power\\report.md")
        if _ask_yes("现在打开报告吗?"):
            open_report()
    else:
        print(f"\n没跑成功(退出码 {proc.returncode})。上面的输出里有原因。")


# ---------------------------------------------------------------- 3 导入


def import_excel() -> None:
    from thermoforge_research.tools import tf_dataset_import

    print("把 Excel 文件拖到这个窗口里,然后按回车(或手动输入完整路径)。")
    text = _ask("文件路径")
    if not text:
        return
    path = _clean_path(text)
    if not path.exists():
        print(f"找不到这个文件：{path}")
        return
    print("\n导入中,大文件需要几十秒……")
    env = tf_dataset_import(_ctx(), path)
    if env["ok"]:
        s = env["summary"]
        print(f"\n导入成功：{env['id']}")
        print(f"  {s.get('rows')} 行 × {s.get('variables')} 个变量")
        print(f"  时间范围 {s.get('time_range')}")
        warns = [d for d in env["diagnostics"] if d["level"] != "ERROR"]
        if warns:
            print(f"  有 {len(warns)} 类提醒(不影响使用)：")
            for d in warns[:5]:
                print(f"    - {d['code']} ×{d['count']}: {d['message'][:60]}")
        return
    print("\n导入失败。原因：")
    for d in env["diagnostics"][:8]:
        if d["level"] == "ERROR":
            print(f"  [{d['code']}] {d['message'][:100]}")
    print("\n最常见的情况是：这份 Excel 不是系统要求的四表格式")
    print("（manifest / objects / variables / data）。")
    print("这种情况需要先写一个格式适配——让 Claude 帮你写,或参考")
    print("src\\thermoforge_data\\legacy.py 里已有的适配器。")


# ---------------------------------------------------------------- 4 训练


def _choose_dataset(ctx) -> tuple[str, list[dict], list[dict]] | None:
    """选数据集 → 返回 (ref, variables, objects)。"""
    datasets = ctx.vault.list_datasets()
    if not datasets:
        print("还没有任何数据。先用菜单 3 导入,或用菜单 2 跑示例。")
        return None
    picked = _pick(datasets,
                   lambda d: f"{d['dataset_id']}"
                             f"（{len(d['revisions'])} 个版本）",
                   "选一个数据集(输编号)")
    if picked is None:
        return None
    ref = picked["revisions"][-1]["ref"]
    return ref, ctx.vault.load_variables(ref), ctx.vault.load_objects(ref)


def _choose_object(objects: list[dict]) -> dict | None:
    if len(objects) == 1:
        return objects[0]
    print("\n这份数据里有多个设备。简易入口一次只处理一个：")
    return _pick(objects,
                 lambda o: f"{o['object_id']}"
                           f"（{o.get('object_name') or o['object_model_id']}）",
                 "选一个设备(输编号)")


def _var_label(v: dict) -> str:
    """derived 会被白名单拒绝(DD-16),单独标记;estimated 只做提示。"""
    kind = {"derived": " ⚠派生量", "estimated": " ·推算值"}.get(
        v["source_kind"], "")
    return f"{v['property_code']:<22} {v['unit']:<8}{kind}"


def _choose_filter(own: list[dict]) -> dict[str, Any]:
    """可选的状态位过滤。

    停机时段目标值接近 0,混进来会让 CVRMSE/NMBE 直接未定义
    (metrics.py：mean(y)≈0 时不产生一个看似正常的数字)。
    """
    flags = [v for v in own if v["dtype"] == "boolean"]
    if not flags:
        return {}
    # 状态位里既有运行标志也有故障/维护标志,选错方向就把数据筛反了。
    # 把常见的运行标志排前面并标注,不靠使用者猜。
    flags.sort(key=lambda v: (v["property_code"] not in RUN_FLAGS,
                              v["property_code"]))
    print("\n【第 3 步】要不要只用设备「运行中」的时段?")
    print("  停机时段的目标值接近 0,混在一起会让误差指标失真、甚至算不出来。")
    options: list[dict | None] = [*flags, None]

    def _label(v: dict | None) -> str:
        if v is None:
            return "不过滤,所有时段都用"
        tag = "  ← 这个是运行标志" if v["property_code"] in RUN_FLAGS else ""
        return f"{v['property_code']}{tag}"

    picked = _pick(options, _label, "选一个表示「设备在运行」的字段")
    if picked is None:
        return {}
    print(f"  只保留 {picked['property_code']} = 是 的时段")
    return {picked["property_code"]: True}


def _blocked_inputs(env: dict) -> dict[str, str]:
    """从可建模性报告里挑出「该去掉哪个输入」及人话原因。

    信封只带摘要(「4/4 个对象存在发现」),具体是哪个变量在 artifact
    里,不读它就只能让人瞎猜。
    """
    out: dict[str, str] = {}
    for artifact in env.get("artifacts", []):
        path = Path(artifact["path"])
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as fp:
            doc = json.load(fp)
        for check in doc.get("checks", []):
            if check.get("level") != "blocker":
                continue
            for obj in (check.get("evidence") or {}).get("per_object", []):
                evidence = obj.get("evidence") or {}
                for corr in evidence.get("target_correlations", []):
                    name = corr.get("candidate")
                    if not name:
                        continue
                    r = abs(float(corr.get("r", 0)))
                    old = out.get(name, "")
                    detail = f"和目标的相关系数 {r:.4f}，几乎是同一个量"
                    if not old or r > 0:
                        out[name] = detail
    return out


def _pass_gate(ctx, ref: str, obj: dict, target_prop: str,
               input_props: list[str]) -> list[str] | None:
    """建 Goal 之前先过可建模性门禁,不通过就地调整。

    门禁支持不带 goal_id 直接检查,所以反复试不会在 Ledger 里留下
    一串废弃 Goal。返回可用的输入清单;放弃时返回 None。
    """
    from thermoforge_research.tools import tf_dataset_modelability

    while True:
        print(f"\n【第 4 步】检查这 {len(input_props)} 个变量能不能支撑目标……")
        env = tf_dataset_modelability(
            ctx, ref, target=target_prop, candidate_inputs=input_props,
            object_model=obj["object_model_id"])
        for c in env["summary"].get("checks", []):
            mark = {"blocker": "✗", "warning": "!"}.get(c["level"], "·")
            print(f"  {mark} {c['name']}: {c['summary'][:80]}")
        if env["status"] != "FAIL":
            return input_props

        blocked = {k: v for k, v in _blocked_inputs(env).items()
                   if k in input_props}
        if not blocked:
            print("\n检查没通过,而且不是某一个输入变量的问题。")
            print("常见原因：可用样本太少,或目标变量长期不变。")
            return None

        print("\n这几个变量不能用来预测目标：")
        for name, why in blocked.items():
            print(f"  - {name}：{why}")
        print("  用它们预测能得到虚高的精度,但换到实际场景没有价值。")

        remaining = [p for p in input_props if p not in blocked]
        if not remaining:
            print("\n去掉之后就没有输入变量了——这台设备在这份数据里")
            print("建不出诚实的模型。建议换一个数据集或设备再试。")
            return None
        print(f"\n去掉后还剩 {len(remaining)} 个：{', '.join(remaining)}")
        if not _ask_yes("去掉它们,用剩下的继续吗?"):
            return None
        input_props = remaining


def train_model() -> None:
    from thermoforge_research.runner import current_environment_lock
    from thermoforge_research.tools import (
        tf_dataset_materialize, tf_experiment_plan, tf_experiment_run,
        tf_goal_create, tf_hypothesis_create,
    )

    ctx = _ctx()
    chosen = _choose_dataset(ctx)
    if chosen is None:
        return
    ref, variables, objects = chosen
    obj = _choose_object(objects)
    if obj is None:
        return

    own = sorted((v for v in variables if v["object_id"] == obj["object_id"]),
                 key=lambda v: v["property_code"])
    numeric = [v for v in own if v["dtype"] in ("float", "integer")]
    if len(numeric) < 2:
        print("这个设备的数值变量太少,没法建模。")
        return

    print(f"\n【第 1 步】要预测什么?（{obj['object_id']} 的变量）")
    target = _pick(numeric, _var_label, "选目标变量(输编号)")
    if target is None:
        return

    candidates = [v for v in numeric if v["property_code"]
                  != target["property_code"]]
    usable_idx = [i for i, v in enumerate(candidates)
                  if v["source_kind"] != "derived"]
    print(f"\n【第 2 步】用哪些变量来预测 {target['property_code']}?")
    print("  标 ⚠派生量 的是由别的变量算出来的,用它预测容易变成「用答案算")
    print("  答案」,系统会拒绝。默认勾选除它们以外的全部。")
    inputs = _pick_many(candidates, _var_label,
                        "输入编号,逗号分隔", usable_idx)
    if not inputs:
        print("没选输入变量,回到菜单。")
        return

    view_filter = _choose_filter(own)

    target_prop = target["property_code"]
    input_props = [v["property_code"] for v in inputs]

    # 派生量必被白名单(DD-16)拒绝,不必等建 Goal 才发现
    derived = [v["property_code"] for v in inputs
               if v["source_kind"] == "derived"]
    if derived:
        print(f"\n这几个是派生量,系统不允许：{', '.join(derived)}")
        print("  它们由别的变量算出来,用来预测等于「用答案算答案」。")
        if not _ask_yes("自动去掉它们继续吗?"):
            return
        input_props = [p for p in input_props if p not in derived]
        if not input_props:
            print("去掉之后一个输入都不剩,回到菜单。")
            return

    passed = _pass_gate(ctx, ref, obj, target_prop, input_props)
    if passed is None:
        return
    input_props = passed
    if len(input_props) < 2:
        print("\n  提醒：只剩 1 个输入变量,模型大概率不会准。")
        print("  这说明这份数据里,能独立解释目标的测量量太少了。")

    print(f"\n准备训练：用 {len(input_props)} 个变量预测 {target_prop}")
    print(f"  输入：{', '.join(input_props)}")
    if not _ask_yes("开始吗?"):
        return

    goal = tf_goal_create(ctx, {
        "name": f"{obj['object_id']} 的 {target_prop} 模型（简易入口）",
        "object_model": obj["object_model_id"],
        "purpose": "optimization",
        "target": target_prop,
        "candidate_inputs": input_props,
        "model_types": {"physics": False, "data": True, "hybrid": False},
        "acceptance": dict(DEFAULT_ACCEPTANCE),
    }, dataset_ref=ref)
    if not goal["ok"]:
        print(f"\n目标没能建立：{goal['summary'].get('error')}")
        return
    print(f"  研究目标已建立：{goal['id']}")

    view = tf_dataset_materialize(ctx, {
        "dataset": ref,
        "scope": {"object_model": obj["object_model_id"]},
        "objects": [obj["object_id"]],
        "features": input_props,
        "target": target_prop,
        **({"filter": view_filter} if view_filter else {}),
    })
    if not view["ok"]:
        print(f"\n数据准备失败：{view['summary'].get('error')}")
        return
    print(f"\n  可用样本 {view['summary']['rows']} 行")

    hyp = tf_hypothesis_create(
        ctx, goal["id"], f"用 {len(input_props)} 个实测变量线性预测 {target_prop}")
    plan = tf_experiment_plan(ctx, {
        "goal_id": goal["id"],
        "hypothesis_id": hyp["id"],
        "dataset_view": view["id"],
        "model": {"category": "data", "estimator": "ridge",
                  "hyperparameters": {"alpha": 1.0}},
        "target": target_prop,
        "validation": {
            "temporal_split": {"train": 0.70, "validate": 0.15, "test": 0.15},
            "equipment_holdout": {"enabled": False, "holdout_objects": []}},
        # 不要 MAPE：设备停机时段 y≈0,有效样本不足会直接 TFX-905 中止,
        # 而 CVRMSE 对同一份数据仍然有定义(metrics.py §5.1)。
        "metrics": ["RMSE", "MAE", "CVRMSE", "NMBE"],
        "runtime": {"environment_lock": current_environment_lock()[0],
                    "random_seed": SEED},
        "description": "简易入口生成",
    })
    if not plan["ok"]:
        print(f"\n实验没能登记：{plan['summary'].get('error')}")
        return

    print("\n【第 5 步】训练中……")
    run = tf_experiment_run(ctx, plan["id"])
    if not run["ok"]:
        code = run["summary"].get("error_code", "")
        print(f"训练失败：{code} {run['summary'].get('error', '')}")
        print(f"  {_explain_error(code)}")
        return
    _show_result(run, plan["id"], target_prop)


ERROR_HINTS = {
    "TFX-903": "训练和测试的数据在时间上重叠了,这会让结果虚高,所以被拦下。",
    "TFX-905": "指标算不出来——通常是目标变量有大量接近 0 的样本(比如设备"
               "停机时段)。可以先筛掉停机数据再训。",
    "TFX-902": "缺随机种子,实验无法复现,所以不允许运行。",
    "TFX-901": "运行环境和实验登记时不一致(可能装过新库)。",
    "TFV-803": "按你的条件筛完之后一行数据都不剩。",
}


def _explain_error(code: str) -> str:
    return ERROR_HINTS.get(code, "把这个错误码发给 Claude,它能查到具体含义。")


def _show_result(run: dict, exp_id: str, target_prop: str) -> None:
    """把指标翻译成人话。

    优先看面 A(未来时段,最接近真实使用);面 A 的 CVRMSE 可能为 None
    ——目标均值≈0 时该指标无定义(metrics.py 不编造数字),此时退回
    validate 面并说明差别。
    """
    surfaces = run["summary"]["surfaces"]
    print(f"\n{LINE}\n训练完成：{exp_id}\n{LINE}")
    surface_name, metrics = "A", (surfaces.get("A") or {}).get("metrics", {})
    if metrics.get("CVRMSE") is None and surfaces.get("validate"):
        print("  注意：最后一段时间里设备基本没运行(目标值接近 0),")
        print("  相对误差在这段上没有定义,下面换用中间那段验证数据。")
        surface_name = "validate"
        metrics = surfaces["validate"].get("metrics", {})

    n = (surfaces.get(surface_name) or {}).get("n_samples")
    where = "模型没见过的未来时段" if surface_name == "A" else "中间验证时段"
    print(f"\n  测试范围：{where}（{n} 条数据）")

    cvrmse = metrics.get("CVRMSE")
    if cvrmse is not None:
        print(f"  预测 {target_prop} 的相对误差(CVRMSE)：{cvrmse:.1%}")
    if metrics.get("MAE") is not None:
        print(f"  平均差多少：{metrics['MAE']:.1f}（和目标同单位）")
    nmbe = metrics.get("NMBE")
    if nmbe is not None:
        print(f"  系统性偏差(NMBE)：{nmbe:+.1%}"
              f"（整体{'偏高' if nmbe > 0 else '偏低'}）")
    if cvrmse is None:
        print("  相对误差无法计算：目标变量的均值太接近 0。")
        print("  下次训练时在第 3 步选上「只用运行中的时段」。")
    else:
        verdict = ("10% 以内,通常够用" if cvrmse <= 0.10
                   else "13% 以内,勉强可用" if cvrmse <= 0.13
                   else "误差偏大,建议换输入变量、加过滤或补数据再试")
        print(f"  参考判断：{verdict}。")
    print(f"\n  完整报告：research\\experiments\\{exp_id}\\report.json")


# ---------------------------------------------------------------- 5 预测


def predict() -> None:
    import pandas as pd

    from thermoforge_runtime.inference import InferenceSession

    models_root = REPO_ROOT / "models"
    packages: list[tuple[str, str, Path]] = []
    for model_dir in sorted(p for p in models_root.glob("*") if p.is_dir()):
        for version_dir in sorted(p for p in model_dir.glob("*") if p.is_dir()):
            if (version_dir / "signature.yaml").exists():
                packages.append((model_dir.name, version_dir.name, version_dir))
    if not packages:
        print("还没有发布过模型。先跑一次菜单 2 的示例。")
        return
    picked = _pick(packages, lambda p: f"{p[0]} 版本 {p[1]}",
                   "选一个模型(输编号)")
    if picked is None:
        return
    _, _, pkg_dir = picked

    print("\n加载中……")
    # 不传 binding：手工输入按签名的 property_code 直通,不需要现场点位映射
    session = InferenceSession(pkg_dir)
    feats = [p.property_code for p in session.signature.inputs]
    out_prop = session.signature.outputs[0].property_code

    defaults: dict[str, float] = {}
    golden_path = pkg_dir / "golden.parquet"
    if golden_path.exists():
        row = pd.read_parquet(golden_path).iloc[0]
        defaults = {c: float(row[c]) for c in feats if c in row}

    print(f"\n这个模型用 {len(feats)} 个输入预测 {out_prop}。")
    print("逐个输入数值,直接回车就用括号里的参考值。\n")
    values: dict[str, float] = {}
    for prop in feats:
        spec = next(p for p in session.signature.inputs
                    if p.property_code == prop)
        default = defaults.get(prop)
        text = _ask(f"  {prop}（{spec.unit}）",
                    f"{default:.2f}" if default is not None else "")
        try:
            values[prop] = float(text)
        except ValueError:
            print(f"  「{text}」不是数字,取消。")
            return

    result = session.handle({
        "contract": "TFDC", "version": "1.0",
        "timestamp": "2026-01-01T00:00:00+08:00",
        "values": values,
    })
    if result["status"] != "ok":
        print(f"\n没能预测：{result['status']} {result.get('flags')}")
        return
    value = result["predictions"][out_prop]
    unit = session.signature.outputs[0].unit
    print(f"\n{LINE}")
    print(f"  预测 {out_prop} = {value:.2f} {unit}")
    print(f"  耗时 {result['latency_ms']:.1f} 毫秒")
    if result.get("flags"):
        print(f"  提醒：{result['flags']}（有输入超出训练时见过的范围）")
    print(LINE)


# ---------------------------------------------------------------- 6 报告


def open_report() -> None:
    report = REPO_ROOT / "examples" / "chiller_power" / "report.md"
    if not report.exists():
        print("报告还不存在。先跑一次菜单 2 的示例。")
        return
    try:
        os.startfile(str(report))  # type: ignore[attr-defined]
        print(f"已用默认程序打开：{report}")
    except Exception:
        print(f"打不开,请手动打开：{report}")


# ---------------------------------------------------------------- 主循环


MENU = [
    ("看看现在有什么（数据、实验、模型）", show_status),
    ("跑一遍示例，看完整效果", run_demo),
    ("导入我自己的 Excel 数据", import_excel),
    ("训练一个模型（一步步问你）", train_model),
    ("用训好的模型做一次预测", predict),
    ("打开示例的实验报告", open_report),
]


def main() -> int:
    _setup_console()
    sys.path.insert(0, str(REPO_ROOT / "src"))
    while True:
        print(f"\n{LINE}")
        print("  ThermoForge 简易入口")
        print(LINE)
        for i, (label, _fn) in enumerate(MENU, 1):
            print(f"  {i}. {label}")
        print("  0. 退出")
        choice = _ask("\n输入编号")
        if choice in ("0", "q", "exit", ""):
            print("再见。")
            return 0
        if not choice.isdigit() or not 1 <= int(choice) <= len(MENU):
            print("没有这个编号。")
            continue
        print()
        try:
            MENU[int(choice) - 1][1]()
        except KeyboardInterrupt:
            print("\n已取消。")
        except Exception as exc:  # 菜单不因单个操作失败而退出
            print(f"\n出错了：{type(exc).__name__}: {exc}")
            print("把这段话发给 Claude,它能告诉你怎么回事。")
        _pause()


if __name__ == "__main__":
    sys.exit(main())
