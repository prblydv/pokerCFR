#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source /venv/main/bin/activate

if ! python -c "import pybind11" >/dev/null 2>&1; then
  uv pip install pybind11
fi

EXTENSION_SUFFIX="$(python3-config --extension-suffix)"
c++ -O3 -DNDEBUG -shared -std=c++20 -fPIC \
  $(python -m pybind11 --includes) \
  heads_up_native_engine.cpp \
  -o "heads_up_native_engine${EXTENSION_SUFFIX}"

chmod +x run_genetic_ensemble_optimizer_vast.sh
touch .genetic_ensemble_stage_ready
echo "Genetic ensemble stage compiled. Optimizer was not started."
