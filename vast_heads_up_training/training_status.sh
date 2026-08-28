#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source /venv/main/bin/activate

ARTIFACT="artifacts/heads_up_v4_hidden384"
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv

python - <<'PY'
import json
from pathlib import Path

artifact = Path("artifacts/heads_up_v4_hidden384")
latest = artifact / "latest.json"
metrics = artifact / "metrics.jsonl"

if latest.exists():
    value = json.loads(latest.read_text(encoding="utf-8"))
    print(f"Latest recoverable checkpoint: iteration {int(value['iteration'])}")
else:
    print("Latest recoverable checkpoint: none yet")

if metrics.exists() and metrics.stat().st_size:
    value = json.loads(metrics.read_text(encoding="utf-8").strip().splitlines()[-1])
    print(f"Latest completed metric: iteration {int(value['iteration'])}")
    print(f"Last iteration time: {float(value.get('seconds', float('nan'))):.2f} seconds")
    if "eval_composite_mean_ev_bb" in value:
        print(
            "Last evaluation composite EV: "
            f"{float(value['eval_composite_mean_ev_bb']):+.4f} bb/hand"
        )
    if "league_mean_ev_bb" in value:
        print(f"Historical league EV: {float(value['league_mean_ev_bb']):+.4f} bb/hand")
else:
    print("Latest completed metric: none yet")
PY

if [[ -f vast-training.log ]]; then
  tail -n 20 vast-training.log
fi
