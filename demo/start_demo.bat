@echo off
REM Use the existing research environment without installing or upgrading packages.
cd /d "%~dp0"
"..\research_v1\.venv_model\Scripts\python.exe" -B backend\app.py --port 8765
pause
