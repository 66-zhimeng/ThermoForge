"""仅启动 V2 MCP 适配器；研究服务及六个 Codex 进程由 ThermoForge 管理。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


PLUGIN_ROOT = Path(__file__).resolve().parents[1]


def resolve_project(explicit: str | None = None) -> Path:
    configured = explicit or os.environ.get("TF_PROJECT_ROOT")
    connection = PLUGIN_ROOT / "connection.json"
    if not configured and connection.is_file():
        config = json.loads(connection.read_text(encoding="utf-8"))
        configured = config.get("project_root")
    if configured:
        candidates = [Path(configured).expanduser().resolve()]
    else:
        candidates = list(PLUGIN_ROOT.parents)
    for root in candidates:
        if (root / "pyproject.toml").is_file() and (root / "src" / "thermoforge_v2").is_dir():
            return root
    raise ValueError("无法定位 ThermoForge 安装。设置 TF_PROJECT_ROOT，或先运行插件的 "
                     "scripts/configure.py --project-root <软件目录>；无需提供或复制认证信息。")


def resolve_python(root: Path) -> Path:
    for relative in (".venv/Scripts/python.exe", ".venv/bin/python"):
        python = root / relative
        if python.is_file():
            return python
    raise ValueError(f"ThermoForge 虚拟环境不存在：{root}。请先在软件目录运行 uv sync。")


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root")
    parser.add_argument("--check", action="store_true", help="仅核对路径，不启动 MCP、研究服务或模型")
    args = parser.parse_args(argv)
    try:
        root = resolve_project(args.project_root)
        python = resolve_python(root)
        if args.check:
            print(json.dumps({"ok": True, "project_root": str(root), "python": str(python),
                              "module": "thermoforge_v2.mcp"}, ensure_ascii=False))
            return 0
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        process = subprocess.Popen([str(python), "-m", "thermoforge_v2.mcp"],
                                   cwd=root, creationflags=flags)
        try:
            return process.wait()
        except KeyboardInterrupt:
            process.terminate()
            try:
                return process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                return 130
    except (OSError, ValueError, TypeError) as exc:
        # stdio 协议占用 stdout；启动错误只进 stderr。
        print(f"ThermoForge MCP 启动失败：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
