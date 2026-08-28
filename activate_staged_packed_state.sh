#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/vast_poker_training"
STAGE="$ROOT/.staged_packed_state_20260722"
BACKUP="$ROOT/.backup_before_packed_state_20260722"

if [[ ! -f "$STAGE/STAGING_VALIDATED" ]]; then
  echo "Refusing to activate: packed-state build has not passed validation." >&2
  exit 1
fi

if [[ -e "$BACKUP" ]]; then
  echo "Refusing to activate twice: the original-engine backup already exists at $BACKUP" >&2
  exit 1
fi

suffix="$(python3-config --extension-suffix)"
mkdir -p "$BACKUP/engine C"
cp -a "$ROOT/engine C/poker_native_engine.cpp" "$BACKUP/engine C/"
cp -a "$ROOT/poker_native_engine${suffix}" "$BACKUP/"

cp -a "$STAGE/poker_native_engine.cpp" "$ROOT/engine C/"
cp -a "$STAGE/poker_native_engine${suffix}" "$ROOT/poker_native_engine${suffix}.new"
mv -f "$ROOT/poker_native_engine${suffix}.new" "$ROOT/poker_native_engine${suffix}"

echo "Packed native state engine activated. Restart the notebook kernel, then resume."
echo "Backup: $BACKUP"
