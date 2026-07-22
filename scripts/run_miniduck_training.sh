#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CONDA_PREFIX="${MINIDUCK_CONDA_PREFIX:-/home/robot/miniconda3/envs/miniducktraining_zero}"
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

RUN_NAME="${MINIDUCK_RUN_NAME:-miniduck_race_v1_12000}"
NUM_ENVS="${MINIDUCK_NUM_ENVS:-4096}"
MAX_ITERATIONS="${MINIDUCK_MAX_ITERATIONS:-12000}"

python --version
which python

set +e
stdbuf -oL -eL python legged_panguin/scripts/train.py \
  --task miniduck_flat \
  --headless \
  --num_envs "$NUM_ENVS" \
  --max_iterations "$MAX_ITERATIONS" \
  --run_name "$RUN_NAME"
status=$?
set -e
echo "train_exit_status=$status"
exit "$status"
