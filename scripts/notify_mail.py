"""把无人值守研究的异常状态发到指定邮箱。

**为什么单独一个脚本**：`autoresearch.py` 只负责跑研究，出了额度/鉴权类
错误就写 `status=fatal` 退出；通知谁、怎么通知是运维口径，不该编进研究
循环里。值守流程（cron / 人）读状态文件，判断要不要调本脚本。

SMTP 参数只从环境变量读，不落盘、不进仓库::

    TF_MAIL_HOST      smtp.example.com
    TF_MAIL_PORT      465（默认 465=SSL；587 走 STARTTLS）
    TF_MAIL_USER      发件账号
    TF_MAIL_PASSWORD  密码或授权码
    TF_MAIL_FROM      发件地址（默认取 TF_MAIL_USER）

用法::

    # 只在状态文件是 fatal 时才发（值守里这样挂）
    python scripts/notify_mail.py --to zhangqian@synthcool.com --only-if-fatal
    # 无条件发一份当前进展
    python scripts/notify_mail.py --to zhangqian@synthcool.com
"""

from __future__ import annotations

import argparse
import json
import os
import smtplib
import ssl
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Mapping

REQUIRED_ENV = ("TF_MAIL_HOST", "TF_MAIL_USER", "TF_MAIL_PASSWORD")


def _body(state: Mapping[str, Any], state_file: Path) -> tuple[str, str]:
    status = str(state.get("status") or "unknown")
    fatal = state.get("fatal") or {}
    lines = [
        f"ThermoForge 无人值守研究 · 状态 {status}",
        f"状态文件: {state_file}",
        f"启动于 {state.get('started_at')}，最后更新 {state.get('updated_at')}",
        f"已跑 {state.get('cycle')} 圈，队列 {state.get('queue')}",
        "",
    ]
    if fatal:
        lines += [
            "【中断原因】",
            f"  目标: {fatal.get('goal')}",
            f"  错误: {fatal.get('error')}",
            f"  判断: {fatal.get('likely_cause')}",
            "",
        ]
    lines.append("【各目标进展】")
    for goal_id, result in (state.get("goals") or {}).items():
        lines.append(
            f"  {goal_id}: 停于 {result.get('stop_reason')}，"
            f"{result.get('rounds')} 轮，最优 CVRMSE={result.get('best_cvrmse')}")
        if result.get("acceptance_unmet"):
            lines.append(f"      差在: {result['acceptance_unmet']}")
    subject = (f"[ThermoForge] 研究循环中断：{fatal.get('likely_cause')}"
               if fatal else f"[ThermoForge] 研究循环状态 {status}")
    return subject, "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--to", required=True, help="收件人，逗号分隔")
    ap.add_argument("--state-file", type=Path,
                    default=Path("research/autoresearch_state.json"))
    ap.add_argument("--only-if-fatal", action="store_true",
                    help="只有状态是 fatal 才发信；否则静默退出 0")
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印将要发送的内容，不连 SMTP")
    args = ap.parse_args(argv)

    if not args.state_file.exists():
        print(f"状态文件不存在: {args.state_file}")
        return 1
    state = json.loads(args.state_file.read_text(encoding="utf-8"))
    if args.only_if_fatal and str(state.get("status")) != "fatal":
        print(f"状态 {state.get('status')}，非 fatal，不发信。")
        return 0

    subject, body = _body(state, args.state_file)
    if args.dry_run:
        print(f"To: {args.to}\nSubject: {subject}\n\n{body}")
        return 0

    missing = [name for name in REQUIRED_ENV if not os.environ.get(name)]
    if missing:
        print(f"缺少 SMTP 环境变量: {missing}。先设置再重试；"
              f"内容可用 --dry-run 先看。")
        return 2

    host = os.environ["TF_MAIL_HOST"]
    port = int(os.environ.get("TF_MAIL_PORT") or 465)
    user = os.environ["TF_MAIL_USER"]
    password = os.environ["TF_MAIL_PASSWORD"]
    sender = os.environ.get("TF_MAIL_FROM") or user

    message = EmailMessage()
    message["From"] = sender
    message["To"] = args.to
    message["Subject"] = subject
    message.set_content(body)

    context = ssl.create_default_context()
    if port == 465:
        with smtplib.SMTP_SSL(host, port, context=context, timeout=30) as smtp:
            smtp.login(user, password)
            smtp.send_message(message)
    else:
        with smtplib.SMTP(host, port, timeout=30) as smtp:
            smtp.starttls(context=context)
            smtp.login(user, password)
            smtp.send_message(message)
    print(f"已发送到 {args.to}: {subject}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
