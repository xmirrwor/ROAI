#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONDA_PREFIX="${MINIDUCK_CONDA_PREFIX:-/opt/conda/envs/miniduck_cu118}"
ISAAC_BINDINGS="$PWD/third_party/isaacgym/python/isaacgym/_bindings/linux-x86_64"
DISPLAY_NUMBER="${MINIDUCK_DISPLAY:-:20}"
LOG_DIR="logs/visualization"
PID_FILE="$LOG_DIR/emergency_stop.pid"
LOG_FILE="$LOG_DIR/emergency_stop.log"
mkdir -p "$LOG_DIR"

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "visualization already running: pid=$old_pid"
    echo "log=$LOG_FILE"
    exit 0
  fi
fi

if [[ -n "${MINIDUCK_VIS_RUN:-}" ]]; then
  load_run="$MINIDUCK_VIS_RUN"
else
  latest_run="$(find logs/flat_miniduck -maxdepth 1 -type d -name '*emergency_stop_stage3_phasefix_12500' -printf '%T@ %f\n' | sort -nr | head -1 | cut -d' ' -f2-)"
  if [[ -z "$latest_run" ]]; then
    echo "no completed emergency-stop run found" >&2
    exit 1
  fi
  load_run="$latest_run"
fi

export CONDA_PREFIX
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$ISAAC_BINDINGS:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD:$PWD/third_party/isaacgym/python:$PWD/third_party/rsl_rl:${PYTHONPATH:-}"
export DISPLAY="$DISPLAY_NUMBER"
export PYTHONUNBUFFERED=1
export LOAD_RUN="$load_run" LOG_FILE PID_FILE

setsid bash -c '
  echo $$ > "$PID_FILE"
  exec python legged_panguin/scripts/play_miniduck.py \
    --task miniduck_flat \
    --load_run "$LOAD_RUN" \
    --checkpoint 12500 \
    --demo emergency_stop >> "$LOG_FILE" 2>&1
' < /dev/null > /dev/null 2>&1 &

sleep 4
pid="$(cat "$PID_FILE" 2>/dev/null || true)"
if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
  echo "started emergency-stop visualization: pid=$pid"
  echo "run=$load_run"
  echo "display=$DISPLAY_NUMBER"
  echo "log=$LOG_FILE"
else
  echo "failed to start visualization; check log=$LOG_FILE" >&2
  exit 1
fi
