#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

RUN_NAME="${MINIDUCK_RUN_NAME:?set MINIDUCK_RUN_NAME}"
LOG_DIR="logs/train_stdout"
PID_FILE="$LOG_DIR/${RUN_NAME}.pid"
LOG_FILE="$LOG_DIR/${RUN_NAME}.log"
mkdir -p "$LOG_DIR"

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "training already running: pid=$old_pid"
    echo "log=$LOG_FILE"
    exit 0
  fi
fi

export LOG_FILE PID_FILE
setsid bash -c '
  echo $$ > "$PID_FILE"
  exec bash scripts/run_action_stage_training.sh >> "$LOG_FILE" 2>&1
' < /dev/null > /dev/null 2>&1 &

sleep 2
pid="$(cat "$PID_FILE" 2>/dev/null || true)"
if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
  echo "started detached training: pid=$pid"
  echo "log=$LOG_FILE"
else
  echo "failed to start detached training; check log=$LOG_FILE" >&2
  exit 1
fi
