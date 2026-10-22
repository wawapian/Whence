@echo off
cd /d "%~dp0backend"
if not exist ".venv\Scripts\python.exe" py -3.11 -m venv .venv
if not exist ".venv\Scripts\python.exe" exit /b 1
".venv\Scripts\python.exe" -m pip install -e .
if errorlevel 1 exit /b 1
if not exist .env copy /y .env.example .env >nul
pause
