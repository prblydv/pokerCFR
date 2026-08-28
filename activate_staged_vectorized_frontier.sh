#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/vast_poker_training"
STAGE="$ROOT/.staged_vectorized_frontier_20260722"
BACKUP="$ROOT/.backup_before_vectorized_frontier_20260722"

if [[ ! -f "$STAGE/STAGING_VALIDATED" ]]; then
  echo "Refusing to activate: staged vectorized traversal has not passed validation." >&2
  exit 1
fi

mkdir -p "$BACKUP"
cp -a "$ROOT/three_player_cfr.py" "$BACKUP/"
cp -a "$ROOT/test_three_player_training.py" "$BACKUP/"
cp -a "$ROOT/test_native_engine.py" "$BACKUP/"

cp -a "$STAGE/three_player_cfr.py" "$ROOT/"
cp -a "$STAGE/test_three_player_training.py" "$ROOT/"
cp -a "$STAGE/test_native_engine.py" "$ROOT/"

echo "Vectorized traversal frontier activated. Restart the notebook kernel, then resume."
echo "Backup: $BACKUP"
