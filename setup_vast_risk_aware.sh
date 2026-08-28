#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source /venv/main/bin/activate

if command -v apt-get >/dev/null 2>&1; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential git python3-dev python3-tk tmux
fi

python -m pip install --upgrade pip
python -m pip install -r requirements-heads-up.txt

EXTENSION_SUFFIX="$(python3-config --extension-suffix)"
c++ -O3 -DNDEBUG -shared -std=c++20 -fPIC \
  $(python -m pybind11 --includes) \
  heads_up_native_engine.cpp \
  -o "heads_up_native_engine${EXTENSION_SUFFIX}"

python -m unittest -q \
  test_heads_up_engine.py \
  test_heads_up_models.py \
  test_heads_up_native_engine.py \
  test_heads_up_training.py \
  test_heads_up_ensemble_profitability.py \
  test_heads_up_action_conditioned_eval.py

python - <<'PY'
import os
import shutil
import torch

import heads_up_native_engine
from heads_up_cfr import HeadsUpNeuralCFR, NETWORK_ARCHITECTURE
from heads_up_native import HeadsUpHoldemEngine
from heads_up_models import information_state_size

memory_kib = 0
with open("/proc/meminfo", encoding="utf-8") as stream:
    for line in stream:
        if line.startswith("MemTotal:"):
            memory_kib = int(line.split()[1])
            break
memory_gib = memory_kib / (1024 * 1024)
disk_free_gib = shutil.disk_usage(".").free / (1024**3)
cpu_count = os.cpu_count() or 1
if hasattr(os, "sched_getaffinity"):
    cpu_count = min(cpu_count, len(os.sched_getaffinity(0)))

if cpu_count < 8:
    raise SystemExit(f"Need at least 8 allocated CPUs; detected {cpu_count}.")
if memory_gib < 110:
    raise SystemExit(f"Need at least 110 GiB RAM; got {memory_gib:.1f}.")
if disk_free_gib < 145:
    raise SystemExit(f"Need at least 145 GiB free; got {disk_free_gib:.1f}.")
if not HeadsUpHoldemEngine.native_backend:
    raise SystemExit("The compiled heads-up C++ engine was not activated.")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot see the rented NVIDIA GPU.")

env = HeadsUpHoldemEngine(starting_stack=200, small_blind=1, big_blind=2, seed=1)
trainer = HeadsUpNeuralCFR(
    env,
    device="cpu",
    hidden=512,
    blocks=2,
    advantage_capacity=8,
    policy_capacity=8,
    range_capacity=1,
    range_loss_weight=0.0,
    network_architecture=NETWORK_ARCHITECTURE,
    policy_network_architecture=NETWORK_ARCHITECTURE,
    enable_range_training=False,
    risk_aware_all_in=True,
    all_in_risk_threshold=2.0,
    all_in_superiority_margin_bb=0.25,
    robust_advantage_loss=True,
    fit_reservoir_once_per_iteration=True,
    advantage_reinitialize_from_iteration=25,
    advantage_reinitialize_cycle=1,
)
if trainer.input_dim != information_state_size() or trainer.input_dim != 1038:
    raise SystemExit(f"Expected restored 1,038-state input; got {trainer.input_dim}.")
if trainer.range_buffers or hasattr(trainer.policy_nets[0], "range_head"):
    raise SystemExit("Fresh policy unexpectedly contains range training state.")
output = trainer.policy_nets[0](torch.zeros(2, trainer.input_dim))
if output.shape != (2, 10):
    raise SystemExit(f"Ten-action output contract is wrong: {output.shape}.")

print(f"C++ engine: {heads_up_native_engine.__file__}")
print(f"Resources: {cpu_count} CPUs, {memory_gib:.1f} GiB RAM, {disk_free_gib:.1f} GiB free")
print(f"CUDA: {torch.cuda.get_device_name(0)}")
print("Input/actions: 1038 / 10")
print("Range head/reservoir/loss: absent")
print("Risk-aware traversal + Smooth-L1 + one-pass fitting: enabled")
PY

touch .vast_risk_aware_setup_complete
echo "Setup complete. Open heads_up_risk_aware_training.ipynb."
