@echo off
rem Run one ingest. Task Scheduler reads the exit code: 0 = success, anything else = failure.
cd /d "%~dp0.."
".venv\Scripts\python.exe" -m hp.ingest
exit /b %errorlevel%