"""Export only the deployable average policy from a full HU Deep CFR checkpoint."""

from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import torch


ACTION_NAMES = (
    "fold",
    "check",
    "call",
    "min_raise",
    "third_pot",
    "half_pot",
    "three_quarter_pot",
    "pot",
    "overbet",
    "all_in",
)


def export_policy(checkpoint_path: Path, output_path: Path) -> dict:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if checkpoint.get("kind") != "heads_up_deep_cfr":
        raise ValueError("source is not a heads-up Deep CFR checkpoint")
    checkpoint_version = int(checkpoint.get("version", -1))
    if checkpoint_version not in (2, 4):
        raise ValueError("unsupported heads-up checkpoint version")
    if int(checkpoint.get("num_players", -1)) != 2:
        raise ValueError("checkpoint must contain exactly two players")
    if int(checkpoint.get("num_actions", -1)) != len(ACTION_NAMES):
        raise ValueError("checkpoint action count does not match")
    if tuple(checkpoint.get("action_names", ())) != ACTION_NAMES:
        raise ValueError("checkpoint action order does not match")
    policy_nets = list(checkpoint.get("policy_nets", ()))
    if len(policy_nets) != 2:
        raise ValueError("checkpoint must contain two average-policy networks")
    config = dict(checkpoint["config"])
    encoder = dict(checkpoint["encoder"])
    iteration = int(checkpoint["iteration"])
    last_fitted = int(checkpoint.get("last_fitted_iteration", iteration))
    if last_fitted != iteration:
        raise ValueError(
            f"iteration {iteration} is not fully fitted; last fitted {last_fitted}"
        )
    has_range_head = checkpoint_version >= 4
    payload = {
        "version": 3 if has_range_head else 2,
        "kind": "heads_up_policy_snapshot",
        "iteration": iteration,
        "input_dim": int(checkpoint["input_dim"]),
        "hidden": int(config["hidden"]),
        "blocks": int(config["blocks"]),
        "network_architecture": str(checkpoint["network_architecture"]),
        **(
            {
                "policy_network_architecture": str(
                    checkpoint["policy_network_architecture"]
                ),
                "range_schema_version": str(
                    checkpoint["range_schema_version"]
                ),
            }
            if has_range_head
            else {}
        ),
        "max_history": int(config["max_history"]),
        "action_names": ACTION_NAMES,
        "environment": dict(checkpoint["environment"]),
        "policy_nets": policy_nets,
        "metadata": {
            "source_checkpoint": str(checkpoint_path.resolve()),
            "source_last_fitted_iteration": last_fitted,
            "source_encoder": encoder,
            "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output_path)
    with output_path.open("rb") as stream:
        sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "iteration": iteration,
        "last_fitted_iteration": last_fitted,
        "output": str(output_path.resolve()),
        "bytes": output_path.stat().st_size,
        "sha256": sha256,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = export_policy(args.checkpoint, args.output)
    for key, value in result.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
