"""将软件目录保存到插件本地连接文件；不修改 Codex 全局配置或认证。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

from launch_mcp import PLUGIN_ROOT, resolve_project, resolve_python


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    args = parser.parse_args()
    root = resolve_project(args.project_root)
    resolve_python(root)
    destination = PLUGIN_ROOT / "connection.json"
    fd, name = tempfile.mkstemp(prefix=".connection-", suffix=".tmp", dir=PLUGIN_ROOT)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump({"project_root": str(root)}, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(name, destination)
    finally:
        Path(name).unlink(missing_ok=True)
    print(f"已保存插件连接位置：{destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
