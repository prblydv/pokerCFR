#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source /venv/main/bin/activate

if [[ ! -f .vast_setup_complete ]]; then
  echo "Run 'bash setup_vast.sh' first." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MPLBACKEND="${MPLBACKEND:-module://matplotlib_inline.backend_inline}"
export POKER_RUN_TESTS="${POKER_RUN_TESTS:-1}"

JUPYTER_PORT="${JUPYTER_PORT:-8080}"
echo "Starting JupyterLab on port ${JUPYTER_PORT}."
echo "Open heads_up_training.ipynb and choose Run All Cells."

exec jupyter lab \
  --ServerApp.ip=0.0.0.0 \
  --ServerApp.port="$JUPYTER_PORT" \
  --ServerApp.port_retries=0 \
  --ServerApp.open_browser=False \
  --ServerApp.allow_root=True \
  --ServerApp.root_dir="$ROOT"
