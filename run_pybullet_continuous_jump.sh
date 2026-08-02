#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" -m pip install -r "${ROOT_DIR}/requirements-pybullet.txt"
cd "${ROOT_DIR}"
exec "${PYTHON_BIN}" train_pybullet_continuous_jump.py "$@"
