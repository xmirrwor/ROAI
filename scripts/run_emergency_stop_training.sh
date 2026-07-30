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

RUN_NAME="${MINIDUCK_RUN_NAME:-emergency_stop_stage3_12500}"
NUM_ENVS="${MINIDUCK_NUM_ENVS:-4096}"
MAX_ITERATIONS="${MINIDUCK_MAX_ITERATIONS:-500}"
LOAD_RUN="${MINIDUCK_LOAD_RUN:-Jul17_04-41-06_miniduck_stable_compressed_curriculum_resume_to_12000_from_8850}"
CHECKPOINT="${MINIDUCK_CHECKPOINT:-12000}"

python --version
stdbuf -oL -eL python legged_panguin/scripts/train.py \
  --task miniduck_flat \
  --headless \
  --num_envs "$NUM_ENVS" \
  --max_iterations "$MAX_ITERATIONS" \
  --resume \
  --load_run "$LOAD_RUN" \
  --checkpoint "$CHECKPOINT" \
  --run_name "$RUN_NAME"
