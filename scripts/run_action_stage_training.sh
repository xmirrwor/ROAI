#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONDA_PREFIX="${MINIDUCK_CONDA_PREFIX:-/opt/conda/envs/miniduck_cu118}"
ISAAC_BINDINGS="$PWD/third_party/isaacgym/python/isaacgym/_bindings/linux-x86_64"
export CONDA_PREFIX
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$ISAAC_BINDINGS:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD:$PWD/third_party/isaacgym/python:$PWD/third_party/rsl_rl:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

RUN_NAME="${MINIDUCK_RUN_NAME:?set MINIDUCK_RUN_NAME}"
NUM_ENVS="${MINIDUCK_NUM_ENVS:-4096}"
MAX_ITERATIONS="${MINIDUCK_MAX_ITERATIONS:?set MINIDUCK_MAX_ITERATIONS}"
LOAD_RUN="${MINIDUCK_LOAD_RUN:?set MINIDUCK_LOAD_RUN}"
CHECKPOINT="${MINIDUCK_CHECKPOINT:?set MINIDUCK_CHECKPOINT}"

stdbuf -oL -eL python legged_panguin/scripts/train.py \
  --task miniduck_flat \
  --headless \
  --num_envs "$NUM_ENVS" \
  --max_iterations "$MAX_ITERATIONS" \
  --resume \
  --load_run "$LOAD_RUN" \
  --checkpoint "$CHECKPOINT" \
  --run_name "$RUN_NAME"
