@echo off
chcp 65001 >nul
cd /d "%~dp0backend"
".venv\Scripts\python.exe" -m scripts.start_workspace --port 8000
if errorlevel 1 (
    echo.
    echo 启动失败，请查看上方提示或 backend\.runtime 中的日志。
    pause
    exit /b 1
)
start "" "http://127.0.0.1:8000/#chat"
echo.
echo 本地版已启动：http://127.0.0.1:8000/#chat
echo 浏览器没有打开时，可复制上方地址访问。
echo 服务在后台运行，关闭此窗口不会停止服务。
echo 需要停止时，请双击项目根目录的 stop-workspace.bat。
pause
