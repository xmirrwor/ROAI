#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" -m pip install -r "${ROOT_DIR}/simulation/requirements-pybullet.txt"
cd "${ROOT_DIR}/simulation"
exec "${PYTHON_BIN}" train_pybullet_jump.py "$@"
