"""Print the resumability-critical metadata of the active checkpoint."""

import json
from pathlib import Path

import torch


artifact = Path(
    "artifacts/three_player_tournament_deep_cfr_bootstrap25_v1"
)
latest = json.loads((artifact / "latest.json").read_text(encoding="utf-8"))
checkpoint = torch.load(
    latest["checkpoint"], map_location="cpu", weights_only=False
)
keys = (
    "network_architecture",
    "hidden",
    "blocks",
    "policy_capacity",
    "recent_capacity",
    "recent_batch_fraction",
    "exploration",
    "max_strategy_importance",
)
print("latest=", latest)
print("config=", {key: checkpoint["config"].get(key) for key in keys})
for name in ("advantage_buffers", "policy_buffers"):
    print(
        name,
        [
            (
                buffer.get("capacity"),
                buffer.get("size"),
                len(buffer.get("fields", [])),
            )
            for buffer in checkpoint[name]
        ],
    )
