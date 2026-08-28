#!/usr/bin/env bash
set -euo pipefail

root=/workspace/vast_poker_training
stage="$root/.staged_deep_cfr_v3_20260723"
backup="$root/.backup_before_deep_cfr_v3_20260723"

if pgrep -af 'multiprocessing.spawn|spawn_main' >/dev/null; then
  echo "Training workers are active. Interrupt training before activation." >&2
  exit 2
fi
test -f "$stage/STAGING_VALIDATED"
for name in \
  three_player_models.py \
  three_player_training.ipynb \
  migrate_deep_cfr_v3.py \
  validate_v3_architecture.py
do
  test -f "$stage/$name"
done

mkdir -p "$backup"
for name in three_player_models.py three_player_training.ipynb; do
  if [[ ! -e "$backup/$name" ]]; then
    cp -p "$root/$name" "$backup/$name"
  fi
done

install -m 0644 "$stage/three_player_models.py" "$root/three_player_models.py"
install -m 0644 "$stage/three_player_training.ipynb" "$root/three_player_training.ipynb"
install -m 0644 "$stage/migrate_deep_cfr_v3.py" "$root/migrate_deep_cfr_v3.py"
install -m 0644 "$stage/validate_v3_architecture.py" "$root/validate_v3_architecture.py"

source /venv/main/bin/activate
cd "$root"
python -m py_compile \
  three_player_models.py \
  migrate_deep_cfr_v3.py \
  validate_v3_architecture.py
python -m json.tool three_player_training.ipynb >/dev/null
echo "deep_cfr_branch_v3 source activated; run the migration command next."
