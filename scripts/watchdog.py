"""无人值守研究的守护脚本：保证「循环不死、简报照发」这条底线。

**这个脚本不做判断**。它只做三件确定性的事，因此可以交给 Windows 计划
任务跑，不依赖任何 AI 会话存活：

1. 查 `autoresearch.py` 进程在不在；不在且上次不是正常收工（exhausted）
   就按记录下来的命令行重新拉起。
2. 读状态文件，拼一份事实简报。
3. 发邮件（`agently-cli`）。额度/鉴权类中断（`status=fatal`）会改标题加急。

「该不该换策略、要不要改判据面」这类判断仍然要人或 AI 会话来做 —— 这里
只保证到那之前循环还活着、你还能收到数。

计划任务注册（管理员 PowerShell，每小时一次）::

    schtasks /Create /TN ThermoForgeWatchdog /SC HOURLY /MO 1 /ST 00:17 ^
      /TR "\"D:\\...\\.venv\\Scripts\\python.exe\" \"D:\\...\\scripts\\watchdog.py\" --to zhangqian@synthcool.com" ^
      /F

或直接用 `--install`（本脚本代为拼好命令并注册）。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

REPO = Path(__file__).resolve().parents[1]
DEFAULT_STATE = REPO / "research" / "autoresearch_state.json"
DEFAULT_LOG = REPO / "research" / "autoresearch.log"
#: 循环正常收工的状态，此时不该再拉起来（否则每小时空转一次烧钱）
DONE_STATUSES = {"exhausted", "max_cycles"}


def _local_now() -> datetime:
    return datetime.now().astimezone()


def _to_local(text: str | None) -> str:
    """留痕是 UTC，给人看的一律换成本机本地时间。"""
    if not text:
        return "—"
    try:
        return (datetime.fromisoformat(text).astimezone()
                .strftime("%Y-%m-%d %H:%M:%S"))
    except ValueError:
        return text


def _loop_running() -> list[int]:
    """返回 autoresearch.py 的进程号列表（空 = 没在跑）。"""
    script = "autoresearch.py"
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name like '%python%'\" | "
             f"Where-Object {{ $_.CommandLine -like '*{script}*' }} | "
             "Select-Object -ExpandProperty ProcessId"],
            capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.SubprocessError):
        return []
    return [int(line) for line in out.stdout.split() if line.strip().isdigit()]


def _restart(state: Mapping[str, Any], log_file: Path) -> str:
    """按状态文件里记下的命令行重新拉起循环。"""
    argv = state.get("argv")
    if not argv:
        return "状态文件里没有 argv，无法自动重启（需要人工指定命令行）"
    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("TF_AGENT_TIMEOUT", "600")
    env.setdefault("TF_AGENT_MAX_RETRIES", "2")
    with open(log_file, "a", encoding="utf-8", newline="\n") as fp:
        fp.write(f"[{datetime.now(timezone.utc).isoformat()}] "
                 f"watchdog 检测到进程不在，重新拉起\n")
    subprocess.Popen([sys.executable, *argv], cwd=str(REPO), env=env,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     creationflags=getattr(subprocess, "DETACHED_PROCESS", 0))
    return f"已重新拉起：{' '.join(argv[:4])} …"


def _tail(path: Path, lines: int = 12) -> str:
    if not path.exists():
        return "（无日志）"
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(text[-lines:])


def _compose(state: Mapping[str, Any], action: str, pids: list[int],
             log_file: Path) -> tuple[str, str]:
    status = str(state.get("status") or "unknown")
    fatal = state.get("fatal") or {}
    now = _local_now().strftime("%m-%d %H:%M")
    urgent = status == "fatal"
    subject = (f"[ThermoForge] 研究循环中断 · {now}" if urgent
               else f"[ThermoForge] 无人值守研究简报 · {now}")

    lines = [
        f"（守护脚本自动发出；时间为本机本地时间，系统留痕是 UTC）",
        "",
        f"循环状态：{status}    进程：{'在跑 ' + str(pids) if pids else '不在'}",
        f"已跑圈数：{state.get('cycle')}    目标队列：{state.get('queue')}",
        f"启动于 {_to_local(state.get('started_at'))}，"
        f"最后更新 {_to_local(state.get('updated_at'))}",
        f"本次动作：{action}",
        "",
    ]
    if fatal:
        lines += ["【中断原因】",
                  f"  目标：{fatal.get('goal')}",
                  f"  错误：{fatal.get('error')}",
                  f"  判断：{fatal.get('likely_cause')}",
                  ""]
    lines.append("【各目标进展】")
    goals = state.get("goals") or {}
    if not goals:
        lines.append("  （本次运行还没有目标跑完一轮）")
    for goal_id, result in goals.items():
        lines.append(f"  {goal_id}：停于 {result.get('stop_reason')}，"
                     f"{result.get('rounds')} 轮，"
                     f"最优 CVRMSE={result.get('best_cvrmse')}")
        metrics = result.get("last_metrics") or {}
        if metrics:
            lines.append("      最近一轮：" + "，".join(
                f"{k}={v:.4f}" for k, v in metrics.items()
                if isinstance(v, (int, float))))
        if result.get("acceptance_unmet"):
            lines.append(f"      差在：{result['acceptance_unmet']}")
    lines += ["", "【日志尾部】", _tail(log_file)]
    return subject, "\n".join(lines)


def _send(to: str, subject: str, body: str, dry_run: bool) -> str:
    if dry_run:
        print(f"To: {to}\nSubject: {subject}\n\n{body}")
        return "dry-run，未发送"
    tmp = REPO / "research" / ".watchdog_brief.txt"
    tmp.write_text(body, encoding="utf-8", newline="\n")
    try:
        out = subprocess.run(
            ["agently-cli", "message", "+send", "--to", to,
             "--subject", subject, "--body-file", f"./{tmp.name}",
             "--confirmed"],
            cwd=str(tmp.parent), capture_output=True, text=True,
            timeout=120, check=False, shell=True)
    except (OSError, subprocess.SubprocessError) as exc:
        return f"发信失败（{type(exc).__name__}: {exc}）"
    finally:
        tmp.unlink(missing_ok=True)
    if out.returncode != 0:
        return f"发信失败（exit {out.returncode}）：{out.stdout.strip()[:300]}"
    return "简报邮件已发送"


def _install(to: str, at: str) -> int:
    """注册 Windows 计划任务（每小时一次）。"""
    python = Path(sys.executable)
    command = (f'"{python}" "{Path(__file__).resolve()}" --to {to}')
    argv = ["schtasks", "/Create", "/TN", "ThermoForgeWatchdog",
            "/SC", "HOURLY", "/MO", "1", "/ST", at,
            "/TR", command, "/F"]
    result = subprocess.run(argv, capture_output=True, text=True, check=False)
    print(result.stdout or result.stderr)
    if result.returncode == 0:
        print(f"已注册计划任务 ThermoForgeWatchdog，每小时 {at[-2:]} 分执行。\n"
              f"查看：schtasks /Query /TN ThermoForgeWatchdog\n"
              f"删除：schtasks /Delete /TN ThermoForgeWatchdog /F")
    return result.returncode


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--to", default="zhangqian@synthcool.com", help="简报收件人")
    ap.add_argument("--state-file", type=Path, default=DEFAULT_STATE)
    ap.add_argument("--log-file", type=Path, default=DEFAULT_LOG)
    ap.add_argument("--no-restart", action="store_true",
                    help="只发简报，不自动拉起循环")
    ap.add_argument("--only-if-fatal", action="store_true",
                    help="只在中断时发信（正常跑着就不打扰）")
    ap.add_argument("--dry-run", action="store_true", help="打印而不发信")
    ap.add_argument("--install", action="store_true",
                    help="注册成 Windows 计划任务（每小时一次）")
    ap.add_argument("--at", default="00:17",
                    help="计划任务的起始时刻 HH:MM（避开整点，默认 00:17）")
    args = ap.parse_args(argv)

    if args.install:
        return _install(args.to, args.at)

    # 驱动器按目标集给状态文件命名，可能同时有多个在跑；--state-file 没显式
    # 指定时，取最近更新的那个。
    state_file = args.state_file
    if not state_file.exists():
        candidates = sorted(DEFAULT_STATE.parent.glob("autoresearch_state*.json"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            print(f"状态文件不存在: {state_file}")
            return 1
        state_file = candidates[0]
        print(f"（改用最近更新的状态文件: {state_file.name}）")
    args.state_file = state_file
    state = json.loads(state_file.read_text(encoding="utf-8"))
    pids = _loop_running()
    status = str(state.get("status") or "")

    action = "无需动作"
    if pids:
        action = "进程正常，未干预"
    elif status in DONE_STATUSES:
        action = f"循环已正常收工（{status}），不再拉起"
    elif args.no_restart:
        action = "进程不在，但 --no-restart 生效，未拉起"
    else:
        action = _restart(state, args.log_file)

    if args.only_if_fatal and status != "fatal":
        print(f"状态 {status}，非 fatal，不发信。动作：{action}")
        return 0

    subject, body = _compose(state, action, pids, args.log_file)
    print(_send(args.to, subject, body, args.dry_run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
