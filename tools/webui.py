"""本地 Web 控制台启动器。

真正的界面在 `src/thermoforge_webui/`；这个文件只负责用正确的参数把
Streamlit 拉起来，好让 `启动网页版.bat` 保持「双击就能用」。

用法（仓库根目录）::

    .venv/Scripts/python tools/webui.py            # 默认 http://127.0.0.1:8765
    .venv/Scripts/python tools/webui.py 8899       # 换端口
    .venv/Scripts/python tools/webui.py --no-browser

安全边界：**只绑 127.0.0.1**。这个界面能改 API 密钥、能跑实验、能批准
预处理规则，绑 0.0.0.0 等于把这些权限开放给同网段的任何人。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
APP = REPO_ROOT / "src" / "thermoforge_webui" / "app.py"
HOST = "127.0.0.1"
DEFAULT_PORT = 8765


def _force_utf8_console() -> None:
    """Windows 控制台默认 cp1252/cp936，打印中文会直接抛 UnicodeEncodeError。

    与 `cli/main.py` 同一处理：进程一起来就把标准流切到 UTF-8。
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONUTF8", "1")


def _skip_first_run_prompt() -> None:
    """跳过 Streamlit 首次运行的邮箱询问。

    没有 `credentials.toml` 时，Streamlit 会在终端里打印欢迎语并**等待输入
    邮箱**。双击 .bat 的人看到的就是一个卡住不动的黑框——界面永远起不来。
    这里先写一个空邮箱的凭据文件把这一步跳过；不上报任何东西。
    """
    path = Path.home() / ".streamlit" / "credentials.toml"
    if path.exists():
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('[general]\nemail = ""\n', encoding="utf-8",
                        newline="\n")
    except OSError:
        pass  # 写不了就算了，最多是首次启动要按一次回车


def _parse(argv: list[str]) -> tuple[int, bool]:
    port = DEFAULT_PORT
    open_browser = True
    for arg in argv:
        if arg in ("--no-browser", "-n"):
            open_browser = False
        elif arg.isdigit():
            port = int(arg)
        elif arg.startswith("--port="):
            port = int(arg.split("=", 1)[1])
    return port, open_browser


def main(argv: list[str] | None = None) -> int:
    _force_utf8_console()
    _skip_first_run_prompt()
    port, open_browser = _parse(list(argv if argv is not None else sys.argv[1:]))
    if not APP.is_file():
        print(f"[!] 找不到界面入口：{APP}", file=sys.stderr)
        return 1
    try:
        from streamlit.web import cli as streamlit_cli
    except ImportError:
        print("[!] 缺少 streamlit。在仓库根目录跑一次：uv sync", file=sys.stderr)
        return 1

    # src 布局：以源码方式运行时（未安装包）也要能 import thermoforge_*
    src = str(REPO_ROOT / "src")
    existing = os.environ.get("PYTHONPATH", "")
    if src not in existing.split(os.pathsep):
        os.environ["PYTHONPATH"] = (f"{src}{os.pathsep}{existing}"
                                    if existing else src)

    # flush：重定向到文件时 stdout 是块缓冲的，不刷新的话这行会一直压到
    # 进程退出才出现——而这个进程正常情况下不会退出
    print(f"\n  ThermoForge 控制台 → http://{HOST}:{port}\n"
          f"  按 Ctrl+C 停止。\n", flush=True)
    sys.argv = [
        "streamlit", "run", str(APP),
        "--server.address", HOST,
        "--server.port", str(port),
        "--server.headless", "false" if open_browser else "true",
        "--browser.gatherUsageStats", "false",
        # 右上角工具栏整个隐藏。默认的 auto 模式在 localhost 上会显示
        # 「Deploy」按钮，点开是 Streamlit 官方的公有云部署引导——对一个
        # 只跑本机、能改密钥和跑实验的控制台，那是纯噪声且容易点错；
        # 剩下的 Settings/Print/About 菜单又只有英文，无法本地化。
        # minimal 模式下没有任何外部注入的菜单项，菜单直接不显示。
        "--client.toolbarMode", "minimal",
        # 关掉 Streamlit 自己的英文欢迎语与「安装 skills」推广，
        # 启动提示由上面那行中文负责。
        "--logger.hideWelcomeMessage", "true",
        # 只服务本机，CORS/XSRF 保持默认开启即可；这里显式写出来是为了
        # 提醒后来者：不要为了「让别人也能看」去关掉它们或改 address。
        "--server.enableCORS", "true",
        "--server.enableXsrfProtection", "true",
    ]
    return int(streamlit_cli.main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
