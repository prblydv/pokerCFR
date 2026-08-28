"""Migrate a resumable checkpoint to the history-aware v3 network."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from three_player_cfr import ThreePlayerNeuralCFR
from three_player_engine import ACTION_NAMES
from three_player_native import ThreePlayerHoldemEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--advantage-steps", type=int, default=256)
    parser.add_argument("--policy-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--recent-batch-fraction", type=float, default=0.2)
    parser.add_argument("--exploration", type=float, default=0.15)
    parser.add_argument("--max-strategy-importance", type=float, default=100.0)
    parser.add_argument(
        "--remove-source",
        action="store_true",
        help="remove the source checkpoint only after the migrated file validates",
    )
    return parser.parse_args()


def _write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def _save_pre_migration_policy(checkpoint: dict, artifact_dir: Path) -> Path:
    config = dict(checkpoint["config"])
    environment = dict(checkpoint.get("environment", {}))
    environment["tournament_total_chips"] = float(
        config.get("tournament_total_chips", 3.0 * environment["stack_size"])
    )
    path = (
        artifact_dir
        / "snapshots"
        / f"policy_{int(checkpoint['iteration']):08d}_pre_v3.pt"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "kind": "three_player_policy_snapshot",
        "iteration": int(checkpoint["iteration"]),
        "input_dim": int(checkpoint["input_dim"]),
        "hidden": int(config["hidden"]),
        "blocks": int(config["blocks"]),
        "network_architecture": str(
            config.get("network_architecture", "residual_mlp")
        ),
        "max_history": int(config["max_history"]),
        "action_names": tuple(ACTION_NAMES),
        "environment": environment,
        "include_tournament_features": bool(
            config.get("include_tournament_features", False)
        ),
        "tournament_features": bool(
            config.get("include_tournament_features", False)
        ),
        "encoder": {
            "include_tournament_features": bool(
                config.get("include_tournament_features", False)
            ),
            "tournament_total_chips": environment["tournament_total_chips"],
        },
        "training_mode": {
            key: config[key]
            for key in (
                "variable_stack_training",
                "heads_up_root_fraction",
                "continuation_root_fraction",
                "minimum_live_stack",
                "root_stack_concentration",
            )
            if key in config
        },
        "policy_nets": checkpoint["policy_nets"],
        "metadata": {"purpose": "pre_v3_migration_backup"},
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return path


def _restore_training_state(
    trainer: ThreePlayerNeuralCFR, checkpoint: dict
) -> None:
    trainer.iteration = int(checkpoint["iteration"])
    trainer.last_fitted_iteration = int(
        checkpoint.get("last_fitted_iteration", trainer.iteration - 1)
    )
    trainer.can_resume_training = True
    trainer._position_cycle = int(checkpoint.get("position_cycle", 0))
    trainer.metrics = list(checkpoint.get("metrics", []))
    trainer.rng.setstate(checkpoint["rng_state"])
    trainer.eval_rng.setstate(checkpoint["eval_rng_state"])
    torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    if "env_rng_state" in checkpoint:
        trainer.env.rng.setstate(checkpoint["env_rng_state"])
        trainer.env._last_button = int(checkpoint["env_last_button"])
    if "sample_env_rng_state" in checkpoint:
        trainer._sample_env.rng.setstate(checkpoint["sample_env_rng_state"])
        trainer._sample_env._last_button = int(
            checkpoint["sample_env_last_button"]
        )
    trainer._continuation_stacks = [
        tuple(float(value) for value in stacks)
        for stacks in checkpoint.get("continuation_stacks", [])
    ]
    trainer._continuation_states_seen = int(
        checkpoint.get(
            "continuation_states_seen", len(trainer._continuation_stacks)
        )
    )
    for buffer, state in zip(
        trainer.advantage_buffers, checkpoint["advantage_buffers"]
    ):
        buffer.load_state_dict(state)
    for buffer, state in zip(
        trainer.policy_buffers, checkpoint["policy_buffers"]
    ):
        buffer.load_state_dict(state)


def main() -> int:
    args = parse_args()
    source = args.checkpoint.resolve()
    artifact_dir = args.artifact_dir.resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if args.hidden < 256 or args.blocks <= 0:
        raise ValueError("hidden must be at least 256 and blocks must be positive")
    if not 0.0 <= args.recent_batch_fraction <= 1.0:
        raise ValueError("recent batch fraction must be in [0, 1]")
    print(f"Loading resumable checkpoint: {source}", flush=True)
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    if "advantage_buffers" not in checkpoint or "policy_buffers" not in checkpoint:
        raise ValueError("migration requires a full checkpoint with all reservoirs")
    old_config = dict(checkpoint["config"])
    if str(old_config.get("network_architecture")) == "deep_cfr_branch_v3":
        print("Checkpoint is already v3; no migration needed.", flush=True)
        return 0
    backup_snapshot = _save_pre_migration_policy(checkpoint, artifact_dir)
    print(f"Saved compact pre-v3 policy backup: {backup_snapshot}", flush=True)

    environment = dict(checkpoint["environment"])
    env = ThreePlayerHoldemEnv(
        stack_size=float(environment["stack_size"]),
        sb=float(environment["sb"]),
        bb=float(environment["bb"]),
        seed=int(old_config.get("seed", 442)),
    )
    new_config = dict(old_config)
    new_config["network_architecture"] = "deep_cfr_branch_v3"
    new_config["hidden"] = int(args.hidden)
    new_config["blocks"] = int(args.blocks)
    new_config["recent_batch_fraction"] = float(args.recent_batch_fraction)
    new_config["exploration"] = float(args.exploration)
    new_config["max_strategy_importance"] = float(
        args.max_strategy_importance
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = ThreePlayerNeuralCFR(env, device=device, **new_config)
    _restore_training_state(trainer, checkpoint)
    # New policy and advantage parameters must be fitted from cumulative data.
    trainer.last_fitted_iteration = min(
        trainer.last_fitted_iteration, trainer.iteration - 1
    )
    del checkpoint
    print(
        f"Refitting six v3 networks at iteration {trainer.iteration} on {device}...",
        flush=True,
    )
    recovery = trainer.recover_incomplete_fit(
        advantage_steps=args.advantage_steps,
        policy_steps=args.policy_steps,
        batch_size=args.batch_size,
    )
    output = source.with_name(f"step_{trainer.iteration:08d}_v3.pt")
    print(f"Writing migrated checkpoint atomically: {output}", flush=True)
    trainer.save(output, include_buffers=True)
    # Validate the written metadata before removing the old full checkpoint.
    validation = torch.load(output, map_location="cpu", weights_only=False)
    validated_config = dict(validation["config"])
    if (
        int(validation["iteration"]) != trainer.iteration
        or validated_config.get("network_architecture") != "deep_cfr_branch_v3"
        or int(validated_config.get("hidden", -1)) != args.hidden
        or int(validated_config.get("blocks", -1)) != args.blocks
        or float(validated_config.get("recent_batch_fraction", -1.0))
        != args.recent_batch_fraction
        or float(validated_config.get("exploration", -1.0)) != args.exploration
        or float(validated_config.get("max_strategy_importance", -1.0))
        != args.max_strategy_importance
        or any(len(buffer.get("fields", [])) != 4 for buffer in validation["advantage_buffers"])
        or any(len(buffer.get("fields", [])) != 4 for buffer in validation["policy_buffers"])
    ):
        raise RuntimeError("written v3 checkpoint failed validation; source retained")
    del validation
    if args.remove_source and output != source:
        source.unlink()

    latest_path = artifact_dir / "latest.json"
    latest = (
        json.loads(latest_path.read_text(encoding="utf-8"))
        if latest_path.exists()
        else {"version": 1}
    )
    latest.update(
        {
            "version": 1,
            "iteration": trainer.iteration,
            "checkpoint": str(output),
            "emergency": False,
            "failed": False,
            "last_fitted_iteration": trainer.last_fitted_iteration,
            "incomplete_fit": False,
            "saved_at_unix": time.time(),
            "migration": "deep_cfr_branch_v3",
        }
    )
    _write_json_atomic(latest_path, latest)

    run_config_path = artifact_dir / "run_config.json"
    if run_config_path.exists():
        run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        previous_hidden = run_config["trainer"].get("hidden")
        previous_blocks = run_config["trainer"].get("blocks")
        previous_architecture = run_config["trainer"].get("network_architecture")
        previous_recent_fraction = run_config["trainer"].get(
            "recent_batch_fraction"
        )
        previous_exploration = run_config["trainer"].get("exploration")
        previous_importance_cap = run_config["trainer"].get(
            "max_strategy_importance"
        )
        run_config["trainer"]["hidden"] = int(args.hidden)
        run_config["trainer"]["blocks"] = int(args.blocks)
        run_config["trainer"]["network_architecture"] = "deep_cfr_branch_v3"
        run_config["trainer"]["recent_batch_fraction"] = float(
            args.recent_batch_fraction
        )
        run_config["trainer"]["exploration"] = float(args.exploration)
        run_config["trainer"]["max_strategy_importance"] = float(
            args.max_strategy_importance
        )
        _write_json_atomic(run_config_path, run_config)
        history_path = artifact_dir / "run_config_history.jsonl"
        with history_path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "changed_at_utc": datetime.now(timezone.utc).isoformat(),
                        "resumed_iteration": trainer.iteration,
                        "campaign_changes": {},
                        "trainer_changes": {
                            "hidden": {
                                "previous": previous_hidden,
                                "current": args.hidden,
                            },
                            "blocks": {
                                "previous": previous_blocks,
                                "current": args.blocks,
                            },
                            "network_architecture": {
                                "previous": previous_architecture,
                                "current": "deep_cfr_branch_v3",
                            },
                            "recent_batch_fraction": {
                                "previous": previous_recent_fraction,
                                "current": args.recent_batch_fraction,
                            },
                            "exploration": {
                                "previous": previous_exploration,
                                "current": args.exploration,
                            },
                            "max_strategy_importance": {
                                "previous": previous_importance_cap,
                                "current": args.max_strategy_importance,
                            },
                        },
                        "migration": "reservoir_preserving_v3_refit",
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )

    report = {
        "source_checkpoint": str(source),
        "source_removed_after_validation": bool(args.remove_source),
        "checkpoint": str(output),
        "iteration": trainer.iteration,
        "hidden": args.hidden,
        "blocks": args.blocks,
        "network_architecture": trainer.network_architecture,
        "recent_batch_fraction": args.recent_batch_fraction,
        "exploration": args.exploration,
        "max_strategy_importance": args.max_strategy_importance,
        "recovery": recovery,
        "pre_v3_policy_backup": str(backup_snapshot),
    }
    _write_json_atomic(artifact_dir / "v3_migration.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
