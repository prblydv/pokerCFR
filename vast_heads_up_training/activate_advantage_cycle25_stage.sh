#!/usr/bin/env bash
set -euo pipefail

live_dir=/workspace/vast_heads_up_training_hidden384
stage_dir=/workspace/hu_advantage_cycle25_stage_20260731
backup_dir="$live_dir/pre_cycle25_code_backup_20260731"

if [[ ! -f "$live_dir/artifacts/heads_up_v4_hidden384/latest.json" ]]; then
    echo "ERROR: expected live HU artifact manifest is missing" >&2
    exit 1
fi
if [[ ! -f "$stage_dir/STAGED_SHA256SUMS" ]]; then
    echo "ERROR: staged checksum manifest is missing" >&2
    exit 1
fi

cd "$stage_dir"
sha256sum --check STAGED_SHA256SUMS

mkdir -p "$backup_dir"
for file in \
    heads_up_cfr.py \
    heads_up_production.py \
    train_heads_up.py \
    heads_up_training.ipynb \
    test_heads_up_training.py \
    README_VAST.md; do
    if [[ -f "$live_dir/$file" ]]; then
        cp -a "$live_dir/$file" "$backup_dir/$file"
    fi
    cp -a "$stage_dir/$file" "$live_dir/$file"
done

echo "Activated 25-iteration advantage reset cycle."
echo "Training artifacts were not modified. Resume from the checkpoint in latest.json."
echo "Previous code is in: $backup_dir"
