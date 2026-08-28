"""Shrink a full heads-up Deep CFR checkpoint without biasing its reservoirs.

Taking a uniform subset of an already-uniform reservoir produces a uniform
reservoir at the smaller capacity. The original ``seen`` counters are retained,
so future Algorithm-R replacement probabilities remain correct.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import torch


BUFFER_KEYS = ("advantage_buffers", "policy_buffers")


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _uniform_subset(
    state: dict[str, Any],
    capacity: int,
    *,
    generator: torch.Generator,
) -> dict[str, Any]:
    fields = list(state.get("fields", ()))
    if not fields:
        raise ValueError("cannot shrink an empty reservoir")
    length = int(fields[0].shape[0])
    if any(int(field.shape[0]) != length for field in fields):
        raise ValueError("reservoir fields have inconsistent lengths")
    if length < capacity:
        raise ValueError(
            f"reservoir has {length:,} rows, below requested {capacity:,} capacity"
        )
    if int(state.get("seen", length)) < length:
        raise ValueError("reservoir seen counter is smaller than its row count")

    if length == capacity:
        selected = fields
    else:
        indices = torch.randperm(length, generator=generator)[:capacity]
        selected = [field.index_select(0, indices) for field in fields]
    return {
        **state,
        "capacity": capacity,
        "fields": selected,
    }


def migrate(
    source: Path,
    destination: Path,
    *,
    capacity: int,
    seed: int,
) -> dict[str, Any]:
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    if source.resolve() == destination.resolve():
        raise ValueError("source and destination must differ")
    if destination.exists() or destination.with_suffix(
        destination.suffix + ".tmp"
    ).exists():
        raise FileExistsError(f"destination already exists: {destination}")

    checkpoint = torch.load(
        source,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    previous_capacity = int(checkpoint["config"]["advantage_capacity"])
    if capacity >= previous_capacity:
        raise ValueError(
            f"new capacity {capacity:,} must be below {previous_capacity:,}"
        )
    if int(checkpoint["config"]["policy_capacity"]) != previous_capacity:
        raise ValueError("advantage and policy capacities differ")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    buffer_report: dict[str, list[dict[str, int]]] = {}
    for key in BUFFER_KEYS:
        migrated_states = []
        reports = []
        for player, state in enumerate(checkpoint[key]):
            before = int(state["fields"][0].shape[0])
            migrated = _uniform_subset(
                state,
                capacity,
                generator=generator,
            )
            migrated_states.append(migrated)
            reports.append(
                {
                    "player": player,
                    "before": before,
                    "after": int(migrated["fields"][0].shape[0]),
                    "seen": int(migrated["seen"]),
                }
            )
        checkpoint[key] = migrated_states
        buffer_report[key] = reports

    checkpoint["config"]["advantage_capacity"] = capacity
    checkpoint["config"]["policy_capacity"] = capacity
    checkpoint["reservoir_capacity_migration"] = {
        "source": str(source.resolve()),
        "previous_capacity": previous_capacity,
        "capacity": capacity,
        "selection": "uniform_subset_without_replacement",
        "seed": int(seed),
        "seen_counters_preserved": True,
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(destination)

    verification = torch.load(
        destination,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if int(verification["config"]["advantage_capacity"]) != capacity:
        raise RuntimeError("migrated advantage capacity verification failed")
    if int(verification["config"]["policy_capacity"]) != capacity:
        raise RuntimeError("migrated policy capacity verification failed")
    for key in BUFFER_KEYS:
        for state in verification[key]:
            if int(state["capacity"]) != capacity:
                raise RuntimeError(f"{key} capacity verification failed")
            if any(int(field.shape[0]) != capacity for field in state["fields"]):
                raise RuntimeError(f"{key} row-count verification failed")

    return {
        "source": str(source.resolve()),
        "destination": str(destination.resolve()),
        "iteration": int(verification["iteration"]),
        "last_fitted_iteration": int(verification["last_fitted_iteration"]),
        "previous_capacity": previous_capacity,
        "capacity": capacity,
        "seed": int(seed),
        "buffers": buffer_report,
        "destination_bytes": destination.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--capacity", type=int, required=True)
    parser.add_argument("--seed", type=int, default=2_000_275)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    report = migrate(
        args.source,
        args.destination,
        capacity=args.capacity,
        seed=args.seed,
    )
    if args.report is not None:
        _atomic_json(args.report, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
