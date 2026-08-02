@echo off
setlocal
cd /d "%~dp0"
".venv-pybullet\Scripts\python.exe" play_pybullet_continuous_jump.py %*
