@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv-pybullet\Scripts\python.exe" (
  py -3 -m venv .venv-pybullet
  if errorlevel 1 exit /b 1
)
".venv-pybullet\Scripts\python.exe" -m pip install -r simulation\requirements-pybullet.txt
if errorlevel 1 exit /b 1
cd simulation
"..\.venv-pybullet\Scripts\python.exe" train_pybullet_jump.py %*
