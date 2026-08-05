#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONDA_PREFIX="${MINIDUCK_CONDA_PREFIX:-/opt/conda/envs/miniduck_cu118}"
ISAAC_BINDINGS="$PWD/third_party/isaacgym/python/isaacgym/_bindings/linux-x86_64"
DISPLAY_NUMBER="${MINIDUCK_DISPLAY:-:20}"
LOG_DIR="logs/visualization"
PID_FILE="$LOG_DIR/race_2m.pid"
LOG_FILE="$LOG_DIR/race_2m.log"
CHECKPOINT="${MINIDUCK_RACE_CHECKPOINT:-checkpoints/miniduck_race_lateral_2m_straight_model_12040.pt}"
mkdir -p "$LOG_DIR"

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    kill "$old_pid"
    for _ in {1..20}; do
      kill -0 "$old_pid" 2>/dev/null || break
      sleep 0.1
    done
  fi
fi

export CONDA_PREFIX
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$ISAAC_BINDINGS:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD:$PWD/third_party/isaacgym/python:$PWD/third_party/rsl_rl:${PYTHONPATH:-}"
export DISPLAY="$DISPLAY_NUMBER"
export PYTHONUNBUFFERED=1
export CHECKPOINT LOG_FILE PID_FILE

: > "$LOG_FILE"
setsid bash -c '
  echo $$ > "$PID_FILE"
  exec python legged_panguin/scripts/play_miniduck.py \
    --task miniduck_flat \
    --demo race_2m \
    --fixed_camera \
    --checkpoint_path_override "$CHECKPOINT" >> "$LOG_FILE" 2>&1
' < /dev/null > /dev/null 2>&1 &

sleep 6
pid="$(cat "$PID_FILE" 2>/dev/null || true)"
if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
  echo "started 2 m race visualization: pid=$pid"
  echo "checkpoint=$CHECKPOINT"
  echo "display=$DISPLAY_NUMBER"
  echo "log=$LOG_FILE"
else
  echo "failed to start visualization; check log=$LOG_FILE" >&2
  tail -n 80 "$LOG_FILE" >&2 || true
  exit 1
fi
