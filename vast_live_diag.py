import json
import os
import time
from pathlib import Path

import psutil


ROOT = Path("/workspace/vast_heads_up_training_hidden384")
ARTIFACT = ROOT / "artifacts" / "heads_up_v4_hidden384"


def read(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError as error:
        return f"<{type(error).__name__}: {error}>"


print("cpu_count_logical", psutil.cpu_count())
print("cpu_count_physical", psutil.cpu_count(logical=False))
print("affinity_count", len(os.sched_getaffinity(0)))
print("cpu_max", read("/sys/fs/cgroup/cpu.max"))
print("memory_max", read("/sys/fs/cgroup/memory.max"))
print("loadavg", os.getloadavg())

processes = []
for process in psutil.process_iter(
    ["pid", "ppid", "name", "cmdline", "cpu_percent", "memory_info"]
):
    command = " ".join(process.info.get("cmdline") or ())
    if (
        "ipykernel" in command
        or "heads_up" in command
        or "multiprocessing" in command
        or "spawn_main" in command
    ):
        process.cpu_percent(None)
        processes.append(process)
time.sleep(3)
print("matching_processes")
for process in processes:
    try:
        print(
            process.pid,
            process.ppid(),
            f"cpu={process.cpu_percent(None):.1f}",
            f"rss_gib={process.memory_info().rss / 2**30:.2f}",
            " ".join(process.cmdline())[:240],
        )
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

print("per_cpu_percent", psutil.cpu_percent(interval=2, percpu=True))

run_config = ARTIFACT / "run_config.json"
if run_config.exists():
    value = json.loads(run_config.read_text(encoding="utf-8"))
    print("campaign_config", json.dumps(value.get("campaign", {}), sort_keys=True))
    print("trainer_config", json.dumps(value.get("trainer", {}), sort_keys=True))
else:
    print("run_config missing")

metrics = ARTIFACT / "metrics.jsonl"
if metrics.exists():
    rows = [
        json.loads(line)
        for line in metrics.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    keys = (
        "iteration",
        "seconds",
        "traversal_seconds",
        "reservoir_merge_seconds",
        "advantage_fit_seconds",
        "policy_fit_seconds",
        "nodes",
        "traversal_nodes_per_second",
        "rollouts",
        "node_cutoffs",
        "depth_cutoffs",
    )
    print("latest_metrics")
    for row in rows[-8:]:
        print(json.dumps({key: row.get(key) for key in keys}, sort_keys=True))
else:
    print("metrics missing")
