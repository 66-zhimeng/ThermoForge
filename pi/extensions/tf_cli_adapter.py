"""Pi Agent 侧最小适配示例：把 `tf` CLI 包装成函数调用。

用法（Pi Agent 的 shell 工具层）::

    from tf_cli_adapter import Tf

    tf = Tf(cwd="/path/to/ThermoForge")  # 或显式 tf_bin / roots
    env = tf("dataset", "list")
    if env["ok"]:
        for ds in env["summary"]["datasets"]:
            ...
    env = tf("dataset", "sample", "WX_2025_PLANT@rev_0001",
             "--n", "50", "--variables", "PLANT.total_power")

约定（见 pi/README.md）：exit 0 含工具级失败，成败读信封 ok；
exit 2 为 CLI 自身错误（参数问题，重试前先修参数）。
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Sequence


class TfCliError(RuntimeError):
    """CLI 自身错误（exit 2）：参数非法等，不是工具级失败。"""


class Tf:
    """`tf` 进程调用的薄封装：参数数组进，信封 dict 出。"""

    def __init__(
        self,
        cwd: str | Path,
        *,
        tf_bin: str = ".venv/Scripts/tf",
        actor: str = "agent",
        timeout_seconds: float = 600.0,
    ):
        self.cwd = Path(cwd)
        self.tf_bin = tf_bin
        self.actor = actor
        self.timeout_seconds = timeout_seconds

    def __call__(self, *argv: str) -> dict[str, Any]:
        cmd = [self.tf_bin, "--actor", self.actor, *argv]
        proc = subprocess.run(
            cmd, cwd=self.cwd, capture_output=True, text=True,
            timeout=self.timeout_seconds, encoding="utf-8", errors="replace",
        )
        if proc.returncode == 2:
            raise TfCliError(f"CLI 参数错误: {proc.stderr.strip()}")
        if proc.returncode != 0:
            raise TfCliError(f"CLI 异常退出 {proc.returncode}: "
                             f"{proc.stderr.strip()[-300:]}")
        try:
            return json.loads(proc.stdout)
        except ValueError as exc:
            raise TfCliError(
                f"stdout 不是合法信封 JSON: {proc.stdout[:200]!r}"
            ) from exc

    def status(self) -> dict[str, Any]:
        """tf status --json（面板数据的机读形式）。"""
        proc = subprocess.run(
            [self.tf_bin, "status", "--json"], cwd=self.cwd,
            capture_output=True, text=True, timeout=60,
            encoding="utf-8", errors="replace",
        )
        if proc.returncode != 0:
            raise TfCliError(proc.stderr.strip())
        return json.loads(proc.stdout)


if __name__ == "__main__":
    import sys

    tf = Tf(cwd=Path(__file__).resolve().parents[2])
    envelope = tf(*sys.argv[1:]) if len(sys.argv) > 1 else tf("dataset", "list")
    print(json.dumps(envelope, ensure_ascii=False, indent=2))
