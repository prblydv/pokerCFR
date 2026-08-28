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

python -m unittest -v \
  test_heads_up_engine.py \
  test_heads_up_models.py \
  test_heads_up_native_engine.py \
  test_heads_up_search.py \
  test_heads_up_training.py \
  test_heads_up_analysis.py

python - <<'PY'
import os
import shutil
import torch

import heads_up_native_engine
from heads_up_native import HeadsUpHoldemEngine
from heads_up_models import (
    POLICY_RANGE_AUX_ARCHITECTURE,
    build_advantage_network,
    build_policy_network,
    information_state_size,
)

memory_kib = 0
with open("/proc/meminfo", encoding="utf-8") as stream:
    for line in stream:
        if line.startswith("MemTotal:"):
            memory_kib = int(line.split()[1])
            break
memory_gib = memory_kib / (1024 * 1024)
try:
    cgroup_memory = open(
        "/sys/fs/cgroup/memory.max", encoding="utf-8"
    ).read().strip()
    if cgroup_memory != "max":
        memory_gib = min(memory_gib, int(cgroup_memory) / (1024**3))
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
    raise SystemExit(
        f"Need at least 110 GiB effective RAM for four 8M reservoirs; "
        f"detected only {memory_gib:.1f} GiB."
    )
if disk_free_gib < 145:
    raise SystemExit(
        f"Need at least 145 GiB free for atomic 8M-reservoir checkpoints; "
        f"detected {disk_free_gib:.1f} GiB."
    )
if not HeadsUpHoldemEngine.native_backend:
    raise SystemExit("The compiled heads-up C++ engine was not activated.")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot see the rented NVIDIA GPU.")
vram_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
if vram_gib < 14:
    raise SystemExit(
        f"Need a nominal 16-GB GPU for the structured model; detected {vram_gib:.1f} GiB."
    )

advantage_network = build_advantage_network(
    "hu_deep_cfr_compact_v4",
    information_state_size(),
    hidden=384,
    blocks=2,
)
policy_network = build_policy_network(
    POLICY_RANGE_AUX_ARCHITECTURE,
    information_state_size(),
    hidden=384,
    blocks=2,
)
advantage_parameters = sum(
    parameter.numel() for parameter in advantage_network.parameters()
)
policy_parameters = sum(
    parameter.numel() for parameter in policy_network.parameters()
)
total_parameters = 2 * advantage_parameters + 2 * policy_parameters
model_mib = total_parameters * 4 / (1024**2)
if advantage_parameters != 1_842_512:
    raise SystemExit(
        f"Unexpected advantage-network size: {advantage_parameters:,}."
    )
if policy_parameters != 2_353_022 or total_parameters != 8_391_068:
    raise SystemExit(
        f"Unexpected range-policy/four-network size: "
        f"{policy_parameters:,}/{total_parameters:,}."
    )
action_logits, range_logits = policy_network.forward_with_range(
    torch.zeros(2, information_state_size())
)
if action_logits.shape != (2, 10) or range_logits.shape != (2, 1326):
    raise SystemExit("Dual-head policy output contract is incorrect.")
del advantage_network, policy_network

x = torch.randn(1024, 1024, device="cuda")
y = x @ x
torch.cuda.synchronize()

print(f"C++ engine: {heads_up_native_engine.__file__}")
print(f"Resources: {cpu_count} CPUs, {memory_gib:.1f} GiB RAM, {disk_free_gib:.1f} GiB free")
print(f"CUDA: {torch.cuda.get_device_name(0)} ({vram_gib:.1f} GiB)")
print(
    f"Networks: advantage={advantage_parameters:,}, "
    f"policy={policy_parameters:,}, total={total_parameters:,} "
    f"({model_mib:.1f} MiB)"
)
print(f"PyTorch: {torch.__version__}; CUDA runtime: {torch.version.cuda}")
print(f"CUDA test result: {float(y[0, 0]):.6f}")
PY

touch .vast_setup_complete
echo "Setup complete. Open heads_up_training.ipynb in the Vast-managed JupyterLab."
