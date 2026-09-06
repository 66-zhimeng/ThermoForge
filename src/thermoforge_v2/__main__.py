"""`tf v2` / `python -m thermoforge_v2` 的统一控制入口。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from .client import REPO_ROOT, V2Client


def build_parser():
    parser = argparse.ArgumentParser(prog="tf v2", description="后台 Codex 研究：准备、启动、观察、暂停恢复与报告")
    shared = argparse.ArgumentParser(add_help=False)
    for name in ("research", "vault", "models"):
        shared.add_argument(f"--{name}-root")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("serve", parents=[shared], help="运行后台服务（通常由客户端自动启动）")
    sub.add_parser("list", parents=[shared], help="列出已持久保存的研究")
    prepare = sub.add_parser("prepare", parents=[shared], help="发现目标、数据和运行能力")
    prepare.add_argument("--config-file")
    start = sub.add_parser("start", parents=[shared], help="按完整配置启动研究，返回 run_id")
    start.add_argument("--config-file", required=True)
    start.add_argument("--idempotency-key", required=True)
    for name in ("status", "events", "report"):
        p = sub.add_parser(name, parents=[shared])
        p.add_argument("run_id")
        if name == "events":
            p.add_argument("--after", type=int, default=0)
            p.add_argument("--limit", type=int, default=100)
        if name == "report":
            p.add_argument("--track-id")
    control = sub.add_parser("control", parents=[shared])
    control.add_argument("run_id")
    control.add_argument("action", choices=("pause", "resume", "cancel", "update"))
    control.add_argument("--expected-version", type=int)
    control.add_argument("--changes-file")
    return parser


def _document(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig")) if path else None


def main(argv=None, *, roots=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    paths = {f"{name}_root": getattr(args, f"{name}_root") or (roots or {}).get(f"{name}_root")
             or os.environ.get(f"TF_{name.upper()}_ROOT") or str(REPO_ROOT / name)
             for name in ("research", "vault", "models")}
    try:
        if args.command == "serve":
            from thermoforge_research.tools import ToolContext
            from .service import serve
            serve(ToolContext(**paths, actor="v2-service"))
            return 0
        client = V2Client(**paths)
        if args.command == "prepare":
            result = client.prepare(_document(args.config_file))
        elif args.command == "start":
            result = client.start(_document(args.config_file), args.idempotency_key)
        elif args.command == "list":
            result = client.list_runs()
        elif args.command == "status":
            result = client.get_run(args.run_id)
        elif args.command == "events":
            result = client.events(args.run_id, args.after, args.limit)
        elif args.command == "control":
            result = client.control(args.run_id, args.action, args.expected_version, _document(args.changes_file))
        else:
            result = client.get_report(args.run_id, args.track_id)
        print(json.dumps({"ok": True, "tool": f"tf_v2_{args.command}", "summary": result}, ensure_ascii=False, default=str))
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "code": getattr(exc, "code", "TFV2-ERROR"), "error": str(exc)}, ensure_ascii=False))
        return 0  # 与现有 CLI 相同，工具结果须读取 ok；参数错误由 argparse 返回 2。


if __name__ == "__main__":
    raise SystemExit(main())
