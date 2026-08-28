#!/usr/bin/env bash
set -euo pipefail

root=/workspace/vast_poker_training
stage="$root/.staged_recent_reservoir_20260722"
backup="$root/.backup_before_recent_reservoir_20260722"

if pgrep -af 'multiprocessing.spawn|spawn_main' >/dev/null; then
  echo "Training workers are still active. Interrupt the notebook cell, wait for the emergency checkpoint, then run this script again." >&2
  exit 2
fi

test -f "$stage/three_player_cfr.py"
test -f "$stage/three_player_production.py"
test -f "$stage/three_player_training.ipynb"

mkdir -p "$backup"
for name in three_player_cfr.py three_player_production.py three_player_training.ipynb; do
  if [[ ! -e "$backup/$name" ]]; then
    cp -p "$root/$name" "$backup/$name"
  fi
done

install -m 0644 "$stage/three_player_cfr.py" "$root/three_player_cfr.py"
install -m 0644 "$stage/three_player_production.py" "$root/three_player_production.py"
install -m 0644 "$stage/three_player_training.ipynb" "$root/three_player_training.ipynb"
rm -f -- "$root/three_player_training_recent.ipynb" "$root/three_player_training_recent_50k.ipynb"

source /venv/main/bin/activate
cd "$root"
python -m py_compile three_player_cfr.py three_player_production.py
python -m json.tool three_player_training.ipynb >/dev/null

echo "Recent reservoir activated. Open three_player_training.ipynb, restart its kernel, and run all cells."
