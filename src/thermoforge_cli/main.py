"""`tf` CLI 主入口：参数解析 → ToolContext → 工具 → 信封 JSON。

结构：每个工具一个子命令，常用参数为显式选项，其余收
`--param k=v`（值按 JSON 解析，失败按字符串）或 `--params-json`。
所有 handler 协议统一为 `handler(ctx, args, extras) -> envelope`，
extras 即 --param/--params-json 合并出的额外关键字参数。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import yaml

from thermoforge_research.tools import TOOL_REGISTRY, ToolContext

from .status import build_status, render_json, render_text


class CliError(Exception):
    """CLI 自身错误（exit 2）。"""


# ---------------------------------------------------------------- 参数辅助


def _coerce_value(text: str) -> Any:
    try:
        return json.loads(text)
    except ValueError:
        return text


def _parse_kv_params(items: Sequence[str] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items or []:
        key, sep, value = str(item).partition("=")
        if not sep or not key:
            raise CliError(f"--param 需要 k=v 形式: {item!r}")
        out[key] = _coerce_value(value)
    return out


def _extras(args: argparse.Namespace) -> dict[str, Any]:
    """合并 --param / --params-json（所有工具子命令通用）。"""
    out = _parse_kv_params(getattr(args, "param", None))
    params_json = getattr(args, "params_json", None)
    if params_json:
        out.update(_load_doc(params_json, None, what="--params-json"))
    return out


def _load_doc(text: str | None, file: str | None, *, what: str) -> dict[str, Any]:
    """从 --*-json / --*-file 载入结构化参数（JSON 优先，YAML 兜底）。"""
    if text is None and file is None:
        raise CliError(f"缺少 {what}：--*-json 或 --*-file 二选一")
    raw = (text if text is not None
           else Path(file).read_text(encoding="utf-8"))  # type: ignore[arg-type]
    try:
        doc = json.loads(raw)
    except ValueError:
        doc = yaml.safe_load(raw)
    if not isinstance(doc, dict):
        raise CliError(f"{what} 必须是 JSON/YAML mapping")
    return doc


def _csv(*values: str | None) -> list[str]:
    """逗号分隔选项 → list；None/空串跳过。"""
    out: list[str] = []
    for v in values:
        if v is None:
            continue
        out.extend(p for p in (p.strip() for p in str(v).split(",")) if p)
    return out


# ---------------------------------------------------------------- 输出


def _emit(envelope: dict[str, Any], pretty: bool) -> None:
    print(json.dumps(envelope, ensure_ascii=False,
                     indent=2 if pretty else None,
                     sort_keys=pretty))


# ---------------------------------------------------------------- 命令表


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tf",
        description="ThermoForge 工具 CLI（Pi Agent 接入面，信封 JSON 输出）",
    )
    parser.add_argument("--vault-root", default="vault")
    parser.add_argument("--research-root", default="research")
    parser.add_argument("--models-root", default="models")
    parser.add_argument("--actor", default="cli")
    parser.add_argument("--pretty", action="store_true",
                        help="缩进 JSON（默认单行紧凑）")
    sub = parser.add_subparsers(dest="group", required=True)

    def tool_cmd(group_sub, name: str, help_text: str):
        p = group_sub.add_parser(name, help=help_text)
        p.add_argument("--param", action="append",
                       help="额外工具参数 k=v（值按 JSON 解析）")
        p.add_argument("--params-json", help="额外工具参数（JSON mapping）")
        return p

    T = TOOL_REGISTRY

    # ---- dataset ----
    ds = sub.add_parser("dataset").add_subparsers(dest="command",
                                                  required=True)
    p = tool_cmd(ds, "import", "导入 TFDC-XLSX")
    p.add_argument("path")
    p.set_defaults(handler=lambda ctx, a, x:
                   T["tf_dataset_import"](ctx, a.path, **x))
    tool_cmd(ds, "list", "列出数据集").set_defaults(
        handler=lambda ctx, a, x: T["tf_dataset_list"](ctx, **x))
    for name, help_text in (("get", "数据集摘要"), ("schema", "变量 Schema"),
                            ("profile", "数据画像")):
        p = tool_cmd(ds, name, help_text)
        p.add_argument("ref")
        p.set_defaults(handler=lambda ctx, a, x, t=f"tf_dataset_{name}":
                       T[t](ctx, a.ref, **x))
    p = tool_cmd(ds, "query", "聚合查询")
    p.add_argument("ref")
    p.add_argument("--variables", required=True, help="逗号分隔 variable_id")
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--aggregations", help="逗号分隔（count/missing/mean/...）")
    p.set_defaults(handler=lambda ctx, a, x: T["tf_dataset_query"](
        ctx, a.ref, variable_ids=_csv(a.variables), start=a.start, end=a.end,
        aggregations=_csv(a.aggregations)
        or ["count", "missing", "mean", "min", "max"], **x))
    p = tool_cmd(ds, "sample", "有限样本（≤200 行）")
    p.add_argument("ref")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--variables", help="逗号分隔 variable_id")
    p.add_argument("--start")
    p.add_argument("--end")
    p.set_defaults(handler=lambda ctx, a, x: T["tf_dataset_sample"](
        ctx, a.ref, n=a.n, variable_ids=_csv(a.variables) or None,
        start=a.start, end=a.end, **x))
    p = tool_cmd(ds, "materialize", "登记并物化 Dataset View")
    p.add_argument("--definition-json")
    p.add_argument("--definition-file")
    p.set_defaults(handler=lambda ctx, a, x: T["tf_dataset_materialize"](
        ctx, _load_doc(a.definition_json, a.definition_file,
                       what="View 定义"), **x))
    p = tool_cmd(ds, "compare", "两版本对比")
    p.add_argument("ref_a")
    p.add_argument("ref_b")
    p.set_defaults(handler=lambda ctx, a, x:
                   T["tf_dataset_compare"](ctx, a.ref_a, a.ref_b, **x))
    p = tool_cmd(ds, "modelability", "可建模性报告（G3）")
    p.add_argument("ref")
    p.add_argument("--goal-id")
    p.add_argument("--target")
    p.add_argument("--candidate-inputs", help="逗号分隔")
    p.add_argument("--object-model")
    p.set_defaults(handler=lambda ctx, a, x: T["tf_dataset_modelability"](
        ctx, a.ref, goal_id=a.goal_id, target=a.target,
        candidate_inputs=_csv(a.candidate_inputs) or None,
        object_model=a.object_model, **x))

    # ---- goal / research / hypothesis ----
    g = sub.add_parser("goal").add_subparsers(dest="command", required=True)
    p = tool_cmd(g, "create", "创建 Research Goal")
    p.add_argument("--definition-json")
    p.add_argument("--definition-file")
    p.add_argument("--dataset-ref", help="提供时做白名单机器校验（DD-16）")
    p.set_defaults(handler=lambda ctx, a, x: T["tf_goal_create"](
        ctx, _load_doc(a.definition_json, a.definition_file,
                       what="Goal 定义"),
        dataset_ref=a.dataset_ref, **x))

    r = sub.add_parser("research").add_subparsers(dest="command",
                                                  required=True)
    p = tool_cmd(r, "status", "研究进展")
    p.add_argument("--goal-id")
    p.set_defaults(handler=lambda ctx, a, x:
                   T["tf_research_status"](ctx, goal_id=a.goal_id, **x))

    h = sub.add_parser("hypothesis").add_subparsers(dest="command",
                                                    required=True)
    p = tool_cmd(h, "create", "创建假设")
    p.add_argument("goal_id")
    p.add_argument("statement")
    p.add_argument("--basis", help="逗号分隔证据 ID（F-/EXP-）")
    p.set_defaults(handler=lambda ctx, a, x: T["tf_hypothesis_create"](
        ctx, a.goal_id, a.statement, basis=_csv(a.basis), **x))

    # ---- experiment ----
    e = sub.add_parser("experiment").add_subparsers(dest="command",
                                                    required=True)
    p = tool_cmd(e, "plan", "登记实验定义")
    p.add_argument("--definition-json")
    p.add_argument("--definition-file")
    p.set_defaults(handler=lambda ctx, a, x: T["tf_experiment_plan"](
        ctx, _load_doc(a.definition_json, a.definition_file,
                       what="实验定义"), **x))
    p = tool_cmd(e, "run", "执行实验（子进程隔离）")
    p.add_argument("experiment_id")
    p.add_argument("--purge-seconds", type=float, default=0.0)
    p.add_argument("--embargo-seconds", type=float, default=None)
    p.add_argument("--y-floor", type=float, default=None)
    p.add_argument("--timeout-seconds", type=float, default=None)
    p.set_defaults(handler=lambda ctx, a, x: T["tf_experiment_run"](
        ctx, a.experiment_id,
        **{k: v for k, v in {"purge_seconds": a.purge_seconds,
                             "embargo_seconds": a.embargo_seconds,
                             "y_floor": a.y_floor,
                             "timeout_seconds": a.timeout_seconds}.items()
           if v is not None}, **x))
    p = tool_cmd(e, "get", "实验结果查询")
    p.add_argument("experiment_id")
    p.set_defaults(handler=lambda ctx, a, x:
                   T["tf_experiment_get"](ctx, a.experiment_id, **x))

    # ---- model ----
    m = sub.add_parser("model").add_subparsers(dest="command", required=True)
    p = tool_cmd(m, "compare", "模型比较")
    p.add_argument("experiment_ids", nargs="+", help="空格或逗号分隔")
    p.set_defaults(handler=lambda ctx, a, x:
                   T["tf_model_compare"](ctx, _csv(*a.experiment_ids), **x))
    p = tool_cmd(m, "publish", "构建模型包并走发布门禁")
    p.add_argument("experiment_id")
    p.add_argument("--model-id", required=True)
    p.add_argument("--version", required=True)
    p.add_argument("--description")
    p.add_argument("--default-out-of-range", default=None,
                   choices=["reject", "clamp", "passthrough_with_flag"])
    p.add_argument("--no-smoke", action="store_true",
                   help="跳过冷加载冒烟（调试用）")
    p.set_defaults(handler=lambda ctx, a, x: T["tf_model_publish"](
        ctx, a.experiment_id, model_id=a.model_id, version=a.version,
        description=a.description,
        **({"default_out_of_range": a.default_out_of_range}
           if a.default_out_of_range else {}),
        run_smoke=not a.no_smoke, **x))

    # ---- preprocess ----
    pp = sub.add_parser("preprocess").add_subparsers(dest="command",
                                                     required=True)
    p = tool_cmd(pp, "propose", "提交预处理规则集")
    p.add_argument("--ruleset-json")
    p.add_argument("--ruleset-file")
    p.set_defaults(handler=lambda ctx, a, x: T["tf_preprocess_propose"](
        ctx, _load_doc(a.ruleset_json, a.ruleset_file, what="规则集"), **x))
    p = tool_cmd(pp, "approve", "审批规则（需 --actor human）")
    p.add_argument("ruleset_id")
    p.add_argument("--version", type=int)
    p.add_argument("--rule-ids", help="逗号分隔；缺省全部")
    p.add_argument("--note")
    p.set_defaults(handler=lambda ctx, a, x: T["tf_preprocess_approve"](
        ctx, a.ruleset_id, version=a.version,
        rule_ids=_csv(a.rule_ids) or None, note=a.note, **x))
    p = tool_cmd(pp, "apply", "执行规则集")
    p.add_argument("path")
    p.add_argument("ruleset_id")
    p.add_argument("--version", type=int)
    p.add_argument("--import-into-vault", action="store_true")
    p.add_argument("--write-workbook", action="store_true")
    p.add_argument("--output-path")
    p.set_defaults(handler=lambda ctx, a, x: T["tf_preprocess_apply"](
        ctx, a.path, a.ruleset_id, version=a.version,
        import_into_vault=a.import_into_vault,
        write_workbook=a.write_workbook, output_path=a.output_path, **x))
    tool_cmd(pp, "list", "列出规则集").set_defaults(
        handler=lambda ctx, a, x: T["tf_preprocess_list"](ctx, **x))

    # ---- status（面板，非工具信封）----
    s = sub.add_parser("status", help="汇总面板（默认人类可读）")
    s.add_argument("--json", action="store_true", help="机读 JSON")
    s.add_argument("--recent", type=int, default=5, help="最近实验条数")
    s.set_defaults(handler=None)
    return parser


# ---------------------------------------------------------------- 入口


def main(argv: Sequence[str] | None = None) -> int:
    # Windows 控制台默认 cp1252，信封含中文会 UnicodeEncodeError（§10.3）
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            reconfigure(encoding="utf-8", errors="replace")
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        ctx = ToolContext(
            vault_root=args.vault_root,
            research_root=args.research_root,
            models_root=args.models_root,
            actor=args.actor,
        )
        if args.group == "status":
            status = build_status(ctx, recent=args.recent)
            print(render_json(status) if args.json else render_text(status))
            return 0
        handler = getattr(args, "handler", None)
        if handler is None:
            raise CliError(f"命令未实现: {args.group} {args.command}")
        _emit(handler(ctx, args, _extras(args)), args.pretty)
        return 0
    except CliError as exc:
        print(f"tf: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # CLI 自身错误（非工具级失败）：exit 2
        print(f"tf: 未预期的 CLI 错误: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
