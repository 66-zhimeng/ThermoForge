@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo   [!] Python environment not found.
    echo   [!] Run this once in a terminal:  uv sync
    echo.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" "tools\webui.py" %1

pause
