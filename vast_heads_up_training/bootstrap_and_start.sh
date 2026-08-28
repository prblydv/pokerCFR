#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ ! -f .vast_setup_complete ]]; then
  bash setup_vast.sh
fi

exec bash start_jupyter.sh
