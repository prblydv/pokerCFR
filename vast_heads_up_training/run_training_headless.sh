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
export MPLBACKEND=Agg
export POKER_RUN_TESTS="${POKER_RUN_TESTS:-1}"

exec jupyter nbconvert \
  --to notebook \
  --execute heads_up_training.ipynb \
  --output heads_up_training_vast.ipynb \
  --ExecutePreprocessor.timeout=-1 \
  2>&1 | tee vast-training.log
