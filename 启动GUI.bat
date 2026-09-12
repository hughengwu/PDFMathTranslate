@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [错误] 未找到虚拟环境 .venv，请先在项目目录下运行：
    echo     uv venv --python 3.12 .venv
    echo     uv pip install -e . --python .venv
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m pdf2zh.native_gui
if errorlevel 1 (
    echo.
    echo [错误] GUI 异常退出，请查看上方日志。
    pause
)
