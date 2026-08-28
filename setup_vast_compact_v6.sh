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
  "engine C HU/heads_up_native_engine.cpp" \
  -o "heads_up_native_engine${EXTENSION_SUFFIX}"

python -m unittest -q \
  test_heads_up_engine.py \
  test_heads_up_models.py \
  test_heads_up_native_engine.py \
  test_heads_up_training.py \
  test_heads_up_analysis.py \
  test_heads_up_compact.py \
  test_heads_up_ensemble_profitability.py

python - <<'PY'
import os
import shutil
import torch

import heads_up_native_engine
from heads_up_compact import (
    COMPACT_DEFAULT_MAX_HISTORY,
    COMPACT_ENCODER_SCHEMA_VERSION,
    compact_information_state_size,
)
from heads_up_models import (
    COMPACT_V6_ARCHITECTURE,
    COMPACT_V6_POLICY_RANGE_ARCHITECTURE,
    build_advantage_network,
    build_policy_network,
)
from heads_up_native import HeadsUpHoldemEngine

if int(heads_up_native_engine.NATIVE_ABI_VERSION) != 6:
    raise SystemExit("Compact package requires native ABI 6.")
if heads_up_native_engine.COMPACT_INPUT_DIM != 782:
    raise SystemExit("Native compact width is not 782.")

memory_kib = 0
with open("/proc/meminfo", encoding="utf-8") as stream:
    for line in stream:
        if line.startswith("MemTotal:"):
            memory_kib = int(line.split()[1])
            break
memory_gib = memory_kib / (1024 * 1024)
try:
    value = open("/sys/fs/cgroup/memory.max", encoding="utf-8").read().strip()
    if value != "max":
        memory_gib = min(memory_gib, int(value) / (1024**3))
except (OSError, ValueError):
    pass
disk_free_gib = shutil.disk_usage(".").free / (1024**3)
cpu_counts = [os.cpu_count() or 1]
if hasattr(os, "sched_getaffinity"):
    cpu_counts.append(len(os.sched_getaffinity(0)))
try:
    quota, period = open("/sys/fs/cgroup/cpu.max", encoding="utf-8").read().split()
    if quota != "max":
        cpu_counts.append(max(1, int(quota) // int(period)))
except (OSError, ValueError):
    pass
cpu_count = max(1, min(cpu_counts))

if cpu_count < 12:
    raise SystemExit(f"Need at least 12 allocated CPUs; detected {cpu_count}.")
if memory_gib < 110:
    raise SystemExit(f"Need at least 110 GiB RAM; got {memory_gib:.1f}.")
if disk_free_gib < 145:
    raise SystemExit(f"Need at least 145 GiB free; got {disk_free_gib:.1f}.")
if not HeadsUpHoldemEngine.native_backend:
    raise SystemExit("The compiled heads-up C++ engine was not activated.")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot see the rented NVIDIA GPU.")
vram_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
if vram_gib < 14:
    raise SystemExit(f"Need a nominal 16-GB GPU; detected {vram_gib:.1f} GiB.")

input_dim = compact_information_state_size(COMPACT_DEFAULT_MAX_HISTORY)
advantage = build_advantage_network(
    COMPACT_V6_ARCHITECTURE, input_dim, hidden=384, blocks=2
)
policy = build_policy_network(
    COMPACT_V6_POLICY_RANGE_ARCHITECTURE,
    input_dim,
    hidden=384,
    blocks=2,
)
actions, ranges = policy.forward_with_range(torch.zeros(2, input_dim))
if actions.shape != (2, 10) or ranges.shape != (2, 1326):
    raise SystemExit(f"Compact output contract is wrong: {actions.shape}, {ranges.shape}.")

x = torch.randn(1024, 1024, device="cuda")
y = x @ x
torch.cuda.synchronize()
print(f"C++ engine: {heads_up_native_engine.__file__}")
print(f"Encoder: {COMPACT_ENCODER_SCHEMA_VERSION}, width={input_dim}")
print(f"Resources: {cpu_count} CPUs, {memory_gib:.1f} GiB RAM, {disk_free_gib:.1f} GiB free")
print(f"CUDA: {torch.cuda.get_device_name(0)} ({vram_gib:.1f} GiB)")
print(f"Hidden-384 advantage parameters: {sum(p.numel() for p in advantage.parameters()):,}")
print(f"Hidden-384 policy+range parameters: {sum(p.numel() for p in policy.parameters()):,}")
print(f"CUDA test result: {float(y[0, 0]):.6f}")
PY

touch .vast_compact_v6_setup_complete
echo "Setup complete. Open heads_up_compact_training.ipynb."
