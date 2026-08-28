"""Reproducible reciprocal match between two heads-up policy snapshots."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Sequence

import torch

from heads_up_cfr import HeadsUpNeuralCFR
from heads_up_engine import ACTION_NAMES
from heads_up_native import HeadsUpHoldemEngine
from heads_up_production import load_policy_snapshot


def _draw_action(
    probabilities: torch.Tensor,
    rng: random.Random,
    mode: str,
) -> int:
    if mode == "argmax":
        return int(torch.argmax(probabilities).item())
    threshold = rng.random()
    cumulative = 0.0
    fallback = int(torch.argmax(probabilities).item())
    for action, probability in enumerate(probabilities.tolist()):
        if probability <= 0.0:
            continue
        fallback = action
        cumulative += float(probability)
        if threshold <= cumulative + 1e-12:
            return action
    return fallback


def _validate_compatible(candidate, baseline) -> dict[str, Any]:
    keys = (
        "input_dim",
        "hidden",
        "blocks",
        "network_architecture",
        "max_history",
        "action_names",
        "environment",
    )
    differences = {
        key: (candidate.metadata.get(key), baseline.metadata.get(key))
        for key in keys
        if candidate.metadata.get(key) != baseline.metadata.get(key)
    }
    if differences:
        raise ValueError(f"policy snapshots are incompatible: {differences}")
    return candidate.metadata


def _stderr(values: Sequence[float]) -> float:
    return stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0


def run_match(
    candidate_path: Path,
    baseline_path: Path,
    *,
    hands: int,
    seed: int,
    mode: str,
    device: str,
    inference_batch_size: int,
) -> dict[str, Any]:
    if hands <= 0 or hands % 2:
        raise ValueError("hands must be a positive even number")
    candidate = load_policy_snapshot(candidate_path, device=device)
    baseline = load_policy_snapshot(baseline_path, device=device)
    metadata = _validate_compatible(candidate, baseline)
    environment = metadata["environment"]
    games_per_assignment = hands // 2
    encoder_env = HeadsUpHoldemEngine(
        starting_stack=int(environment["starting_stack"]),
        small_blind=int(environment["small_blind"]),
        big_blind=int(environment["big_blind"]),
        seed=seed,
    )
    trainer = HeadsUpNeuralCFR(
        encoder_env,
        device=device,
        hidden=int(metadata["hidden"]),
        blocks=int(metadata["blocks"]),
        advantage_capacity=1,
        policy_capacity=1,
        max_history=int(metadata["max_history"]),
        seed=seed,
    )
    values_by_assignment: list[list[float]] = []
    positions_by_assignment: list[list[bool]] = []
    candidate_actions: Counter[int] = Counter()
    baseline_actions: Counter[int] = Counter()
    started = time.perf_counter()

    for candidate_seat in range(2):
        env = HeadsUpHoldemEngine(
            starting_stack=int(environment["starting_stack"]),
            small_blind=int(environment["small_blind"]),
            big_blind=int(environment["big_blind"]),
            seed=seed,
        )
        states = [
            # Keep the public/chance sequence identical for both policy-seat
            # assignments. This makes the two assignments truly reciprocal:
            # swapping candidate and baseline reproduces the same games with
            # every payoff sign reversed.
            env.new_hand(button=game % 2)
            for game in range(games_per_assignment)
        ]
        candidate_is_button = [
            int(state.button) == candidate_seat for state in states
        ]
        action_rngs = [
            random.Random(seed + 1_000_003 * (game + 1))
            for game in range(games_per_assignment)
        ]
        steps = [0] * games_per_assignment
        while True:
            live = [index for index, state in enumerate(states) if not state.terminal]
            if not live:
                break
            candidate_indices = [
                index
                for index in live
                if int(states[index].to_act) == candidate_seat
            ]
            baseline_indices = [
                index
                for index in live
                if int(states[index].to_act) != candidate_seat
            ]
            probabilities: dict[int, torch.Tensor] = {}
            if candidate_indices:
                predictions = trainer.average_policy_batch(
                    [states[index] for index in candidate_indices],
                    policy_nets=candidate.policy_nets,
                    batch_size=inference_batch_size,
                )
                probabilities.update(zip(candidate_indices, predictions))
            if baseline_indices:
                predictions = trainer.average_policy_batch(
                    [states[index] for index in baseline_indices],
                    policy_nets=baseline.policy_nets,
                    batch_size=inference_batch_size,
                )
                probabilities.update(zip(baseline_indices, predictions))
            for index in live:
                state = states[index]
                actor = int(state.to_act)
                action = _draw_action(probabilities[index], action_rngs[index], mode)
                if actor == candidate_seat:
                    candidate_actions[action] += 1
                else:
                    baseline_actions[action] += 1
                states[index] = env.step(state, action)
                steps[index] += 1
                if steps[index] > 512:
                    raise RuntimeError("evaluation hand exceeded 512 actions")
        values_by_assignment.append(
            [float(state.payoffs[candidate_seat]) / float(env.bb) for state in states]
        )
        positions_by_assignment.append(candidate_is_button)

    values = values_by_assignment[0] + values_by_assignment[1]
    paired_values = [
        mean((values_by_assignment[0][game], values_by_assignment[1][game]))
        for game in range(games_per_assignment)
    ]
    stderr = _stderr(paired_values)
    button_values = [
        value
        for assignment_values, assignment_positions in zip(
            values_by_assignment,
            positions_by_assignment,
        )
        for value, is_button in zip(assignment_values, assignment_positions)
        if is_button
    ]
    blind_values = [
        value
        for assignment_values, assignment_positions in zip(
            values_by_assignment,
            positions_by_assignment,
        )
        for value, is_button in zip(assignment_values, assignment_positions)
        if not is_button
    ]
    result = {
        "candidate": str(candidate_path),
        "candidate_iteration": candidate.iteration,
        "baseline": str(baseline_path),
        "baseline_iteration": baseline.iteration,
        "mode": mode,
        "seed": seed,
        "hands": hands,
        "reciprocal_pairs": games_per_assignment,
        "mean_ev_bb_per_hand": mean(values),
        "mean_ev_mbb_per_hand": 1_000.0 * mean(values),
        "paired_stderr_bb_per_hand": stderr,
        "ci95_low_bb_per_hand": mean(values) - 1.96 * stderr,
        "ci95_high_bb_per_hand": mean(values) + 1.96 * stderr,
        "candidate_as_p0_ev_bb_per_hand": mean(values_by_assignment[0]),
        "candidate_as_p1_ev_bb_per_hand": mean(values_by_assignment[1]),
        "candidate_button_ev_bb_per_hand": mean(button_values),
        "candidate_big_blind_ev_bb_per_hand": mean(blind_values),
        "wins": sum(value > 0.0 for value in values),
        "losses": sum(value < 0.0 for value in values),
        "ties": sum(value == 0.0 for value in values),
        "candidate_action_counts": {
            ACTION_NAMES[action]: int(candidate_actions[action])
            for action in range(len(ACTION_NAMES))
        },
        "baseline_action_counts": {
            ACTION_NAMES[action]: int(baseline_actions[action])
            for action in range(len(ACTION_NAMES))
        },
        "elapsed_seconds": time.perf_counter() - started,
        "device": device,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--hands", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=702_600)
    parser.add_argument("--mode", choices=("sample", "argmax"), default="sample")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--inference-batch-size", type=int, default=2_048)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_match(
        args.candidate,
        args.baseline,
        hands=args.hands,
        seed=args.seed,
        mode=args.mode,
        device=args.device,
        inference_batch_size=args.inference_batch_size,
    )
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
