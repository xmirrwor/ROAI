@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv-pybullet\Scripts\python.exe" (
  echo Run run_pybullet_jump.bat first to create the environment and train a policy.
  exit /b 1
)
cd simulation
"..\.venv-pybullet\Scripts\python.exe" play_pybullet_jump.py %*
