#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/vast_poker_training"
STAGE="$ROOT/.staged_quality_params_20260722"
BACKUP="$ROOT/.backup_before_quality_params_20260722"

if pgrep -af 'multiprocessing.spawn|--multiprocessing-fork' >/dev/null; then
  echo "Refusing to activate while traversal workers are running." >&2
  echo "Interrupt the notebook training cell and wait for its emergency checkpoint." >&2
  exit 1
fi
if [[ -e "$BACKUP" ]]; then
  echo "Refusing to activate twice; backup already exists at $BACKUP" >&2
  exit 1
fi
if [[ ! -f "$STAGE/three_player_training.ipynb" ]]; then
  echo "Staged notebook is missing." >&2
  exit 1
fi

/venv/main/bin/python - "$STAGE/three_player_training.ipynb" <<'PY'
import json
import sys

path = sys.argv[1]
notebook = json.load(open(path, encoding="utf-8"))
source = "\n".join("".join(cell.get("source", ())) for cell in notebook["cells"])
required = (
    "'max_depth': 32, 'max_strategy_importance': 50.0",
    "'exploration': 0.05",
    "'target_iteration': 10_000, 'traversals_per_player': 1_088",
    "'advantage_steps': 256, 'policy_steps': 128, 'batch_size': 8192",
    "trainer.max_strategy_importance = float(TRAINER_CONFIG['max_strategy_importance'])",
)
missing = [value for value in required if value not in source]
if missing:
    raise SystemExit(f"staged notebook validation failed; missing: {missing}")
print("Staged quality parameters validated.")
PY

mkdir -p "$BACKUP"
cp -a "$ROOT/three_player_training.ipynb" "$BACKUP/"
cp -a "$STAGE/three_player_training.ipynb" "$ROOT/three_player_training.ipynb.new"
mv -f "$ROOT/three_player_training.ipynb.new" "$ROOT/three_player_training.ipynb"

echo "Quality-parameter notebook activated."
echo "Backup: $BACKUP/three_player_training.ipynb"
echo "Reload the notebook from disk, restart its kernel, and rerun the cells."
