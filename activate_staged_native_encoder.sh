#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/vast_poker_training"
STAGE="$ROOT/.staged_native_encoder_20260722"
BACKUP="$ROOT/.backup_before_native_encoder_20260722"

if [[ ! -f "$STAGE/STAGING_VALIDATED" ]]; then
  echo "Refusing to activate: staged build has not passed validation." >&2
  exit 1
fi

mkdir -p "$BACKUP/engine C"
cp -a "$ROOT/three_player_models.py" "$BACKUP/"
cp -a "$ROOT/three_player_native.py" "$BACKUP/"
cp -a "$ROOT/test_native_engine.py" "$BACKUP/"
cp -a "$ROOT/engine C/poker_native_engine.cpp" "$BACKUP/engine C/"

cp -a "$STAGE/three_player_models.py" "$ROOT/"
cp -a "$STAGE/three_player_native.py" "$ROOT/"
cp -a "$STAGE/test_native_engine.py" "$ROOT/"
cp -a "$STAGE/poker_native_engine.cpp" "$ROOT/engine C/"

suffix="$(python3-config --extension-suffix)"
cp -a "$STAGE/poker_native_engine${suffix}" "$ROOT/poker_native_engine${suffix}.new"
mv -f "$ROOT/poker_native_engine${suffix}.new" "$ROOT/poker_native_engine${suffix}"

echo "Native encoder activated. Restart the notebook kernel, then resume training."
echo "Backup: $BACKUP"
