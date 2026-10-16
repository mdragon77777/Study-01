@echo off
cd /d "%~dp0"
"..\.venv\Scripts\python.exe" digit_recognizer.py
if errorlevel 1 pause
