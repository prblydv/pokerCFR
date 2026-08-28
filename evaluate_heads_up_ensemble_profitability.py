"""Reciprocal profitability suite for an averaged top-k HU policy ensemble.

The evaluator keeps chance sequences fixed while swapping the candidate between
physical seats.  It therefore measures the selected policy/controller in the
exact rake-free HU engine without confounding the result with a lucky seat or
deck sequence.  Finite confidence intervals are evidence, not a claim of
certainty against every possible opponent.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from collections import Counter
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Protocol, Sequence

import torch

from heads_up_cfr import HeadsUpNeuralCFR
from heads_up_engine import (
    ACTION_ALL_IN,
    ACTION_CALL,
    ACTION_CHECK,
    ACTION_FOLD,
    ACTION_MIN_RAISE,
    ACTION_NAMES,
    NUM_ACTIONS,
)
from heads_up_native import HeadsUpHoldemEngine
from heads_up_production import (
    OPPONENT_PROFILES,
    ScriptedOpponent,
    _normalised_policy,
    load_policy_snapshot,
)


DEFAULT_POLICIES = (
    Path("artifacts/heads_up_v4_paper3x/snapshots/policy_00000725.pt"),
    Path("artifacts/heads_up_v4_paper3x/snapshots/policy_00000950.pt"),
    Path("artifacts/heads_up_v4_paper3x/snapshots/policy_00001025.pt"),
)
DEFAULT_OUTPUT = Path(
    "artifacts/heads_up_v4_paper3x/evaluations/"
    "ensemble_725_950_1025_top3_profitability.json"
)


class ProbabilityProvider(Protocol):
    name: str

    def probabilities_batch(self, states: Sequence[Any]) -> torch.Tensor: ...


def _compatibility_metadata(snapshots) -> dict[str, Any]:
    keys = (
        "input_dim",
        "hidden",
        "blocks",
        "network_architecture",
        "policy_network_architecture",
        "max_history",
        "action_names",
        "environment",
        "action_schema_version",
        "engine_schema_version",
    )
    first = snapshots[0].metadata
    differences = {
        key: [snapshot.metadata.get(key) for snapshot in snapshots]
        for key in keys
        if any(snapshot.metadata.get(key) != first.get(key) for snapshot in snapshots[1:])
    }
    if differences:
        raise ValueError(f"policy snapshots are incompatible: {differences}")
    return first


def top_k_probabilities(probabilities: torch.Tensor, top_k: int) -> torch.Tensor:
    """Keep the highest positive actions per row and renormalize exactly."""

    if probabilities.ndim != 2 or probabilities.shape[1] != NUM_ACTIONS:
        raise ValueError(f"probabilities must have shape [N, {NUM_ACTIONS}]")
    if top_k <= 0 or top_k >= NUM_ACTIONS:
        return probabilities / probabilities.sum(dim=1, keepdim=True)
    if torch.any(probabilities.sum(dim=1) <= 0.0):
        raise RuntimeError("policy row contains no positive legal action")
    # Stable descending sort preserves the GUI tie-break: lower action index
    # wins when probabilities are identical. Zero-probability padding can be
    # selected only when a row has fewer than top_k legal actions and remains
    # zero, exactly matching the scalar implementation's effect.
    selected = torch.argsort(
        probabilities,
        dim=1,
        descending=True,
        stable=True,
    )[:, : int(top_k)]
    adjusted = torch.zeros_like(probabilities)
    adjusted.scatter_(1, selected, probabilities.gather(1, selected))
    return adjusted / adjusted.sum(dim=1, keepdim=True)


def ensemble_top_k_from_name(name: str) -> int | None:
    """Parse opponent names such as ``ensemble_top4``."""

    match = re.fullmatch(r"ensemble_top([1-9][0-9]*)", name)
    return int(match.group(1)) if match is not None else None


class SnapshotProvider:
    def __init__(self, trainer, snapshot, *, name: str | None = None) -> None:
        self.trainer = trainer
        self.snapshot = snapshot
        self.name = name or f"policy_{snapshot.iteration}"

    @torch.inference_mode()
    def probabilities_batch(self, states: Sequence[Any]) -> torch.Tensor:
        if not states:
            return torch.empty((0, NUM_ACTIONS), dtype=torch.float32)
        return torch.stack(
            self.trainer.average_policy_batch(
                states,
                policy_nets=self.snapshot.policy_nets,
                batch_size=max(1, len(states)),
            )
        )


class TrainerProvider:
    """Expose a live trainer through the evaluator's batched policy contract."""

    def __init__(self, trainer, *, top_k: int = 0, name: str = "live_trainer") -> None:
        self.trainer = trainer
        self.top_k = int(top_k)
        self.name = str(name)

    @torch.inference_mode()
    def probabilities_batch(self, states: Sequence[Any]) -> torch.Tensor:
        if not states:
            return torch.empty((0, NUM_ACTIONS), dtype=torch.float32)
        probabilities = torch.stack(
            self.trainer.average_policy_batch(
                states,
                batch_size=max(1, len(states)),
            )
        )
        return top_k_probabilities(probabilities, self.top_k)


class EnsembleProvider:
    def __init__(
        self,
        trainer,
        snapshots,
        *,
        top_k: int,
        name: str | None = None,
    ) -> None:
        self.trainer = trainer
        self.snapshots = tuple(snapshots)
        self.top_k = int(top_k)
        iterations = "_".join(str(snapshot.iteration) for snapshot in snapshots)
        suffix = f"top{self.top_k}" if self.top_k > 0 else "full"
        self.name = name or f"ensemble_{iterations}_{suffix}"

    @torch.inference_mode()
    def probabilities_batch(self, states: Sequence[Any]) -> torch.Tensor:
        if not states:
            return torch.empty((0, NUM_ACTIONS), dtype=torch.float32)
        components = [
            torch.stack(
                self.trainer.average_policy_batch(
                    states,
                    policy_nets=snapshot.policy_nets,
                    batch_size=max(1, len(states)),
                )
            )
            for snapshot in self.snapshots
        ]
        averaged = torch.stack(components).mean(dim=0)
        averaged = averaged / averaged.sum(dim=1, keepdim=True)
        return top_k_probabilities(averaged, self.top_k)


class ScriptedProvider:
    def __init__(self, env, opponent: ScriptedOpponent) -> None:
        self.env = env
        self.opponent = opponent
        self.name = opponent.name

    def probabilities_batch(self, states: Sequence[Any]) -> torch.Tensor:
        return torch.stack(
            [
                self.opponent.probabilities(self.env, state, int(state.to_act))
                for state in states
            ]
        )


class LooseAggressiveOpponent(ScriptedOpponent):
    name = "loose_aggressive"

    def probabilities(self, env, state, player: int) -> torch.Tensor:
        legal = env.legal_actions(state)
        raises = [action for action in legal if action >= ACTION_MIN_RAISE]
        middle_raise = raises[len(raises) // 2] if raises else None
        weights: dict[int, float] = {}
        if ACTION_CALL in legal:
            weights = {ACTION_FOLD: 0.18, ACTION_CALL: 0.32}
            if middle_raise is not None:
                weights[middle_raise] = 0.50
        elif ACTION_CHECK in legal:
            weights = {ACTION_CHECK: 0.32}
            if middle_raise is not None:
                weights[middle_raise] = 0.68
        return _normalised_policy(weights, legal)


class AllInPressureOpponent(ScriptedOpponent):
    name = "all_in_pressure"

    def probabilities(self, env, state, player: int) -> torch.Tensor:
        legal = env.legal_actions(state)
        weights: dict[int, float] = {}
        if ACTION_CALL in legal:
            weights = {ACTION_FOLD: 0.10, ACTION_CALL: 0.20}
        elif ACTION_CHECK in legal:
            weights = {ACTION_CHECK: 0.15}
        if ACTION_ALL_IN in legal:
            weights[ACTION_ALL_IN] = 0.70 if ACTION_CALL in legal else 0.85
        elif ACTION_MIN_RAISE in legal:
            weights[ACTION_MIN_RAISE] = 0.70
        return _normalised_policy(weights, legal)


class BlindStealPressureOpponent(ScriptedOpponent):
    """Attack excessive blind and postflop folds with frequent small raises."""

    name = "blind_steal_pressure"

    def probabilities(self, env, state, player: int) -> torch.Tensor:
        legal = env.legal_actions(state)
        raises = [action for action in legal if action >= ACTION_MIN_RAISE]
        smallest_raise = raises[0] if raises else None
        if int(state.street) == 0 and int(player) == int(state.button):
            weights = {ACTION_CALL: 0.05}
            if smallest_raise is not None:
                weights[smallest_raise] = 0.95
            return _normalised_policy(weights, legal)
        if ACTION_CALL in legal:
            weights = {ACTION_FOLD: 0.72, ACTION_CALL: 0.20}
            if smallest_raise is not None:
                weights[smallest_raise] = 0.08
            return _normalised_policy(weights, legal)
        weights = {ACTION_CHECK: 0.15}
        if smallest_raise is not None:
            weights[smallest_raise] = 0.85
        return _normalised_policy(weights, legal)


EXTRA_PROFILES: dict[str, type[ScriptedOpponent]] = {
    LooseAggressiveOpponent.name: LooseAggressiveOpponent,
    AllInPressureOpponent.name: AllInPressureOpponent,
    BlindStealPressureOpponent.name: BlindStealPressureOpponent,
}


def _draw_actions(probabilities: torch.Tensor, rngs, indices) -> list[int]:
    actions: list[int] = []
    for row, rng, _ in zip(probabilities.tolist(), rngs, indices):
        threshold = rng.random()
        cumulative = 0.0
        fallback = max(range(len(row)), key=row.__getitem__)
        for action, probability in enumerate(row):
            if probability <= 0.0:
                continue
            fallback = action
            cumulative += float(probability)
            if threshold <= cumulative + 1e-12:
                fallback = action
                break
        actions.append(int(fallback))
    return actions


def _stderr(values: Sequence[float]) -> float:
    return stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return float("nan")
    position = max(0.0, min(1.0, float(fraction))) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _ratio_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "minimum": None,
            "maximum": None,
        }
    return {
        "count": len(finite),
        "mean": mean(finite),
        "median": _percentile(finite, 0.5),
        "p90": _percentile(finite, 0.9),
        "minimum": min(finite),
        "maximum": max(finite),
    }


@torch.inference_mode()
def run_reciprocal_match(
    candidate: ProbabilityProvider,
    opponent_factory,
    *,
    environment: dict[str, Any],
    hands: int,
    seed: int,
    inference_batch_size: int,
    simulation_batch_size: int = 20_000,
) -> dict[str, Any]:
    if hands <= 0 or hands % 2:
        raise ValueError("hands must be a positive even number")
    games_per_assignment = hands // 2
    assignment_values: list[list[float]] = []
    assignment_buttons: list[list[bool]] = []
    candidate_actions: Counter[int] = Counter()
    opponent_actions: Counter[int] = Counter()
    candidate_actions_by_street: Counter[tuple[int, int]] = Counter()
    all_in_bet_to_pot_before: list[float] = []
    all_in_payment_over_pot_after_call: list[float] = []
    all_in_raise_over_pot_after_call: list[float] = []
    all_in_spr_after_call: list[float] = []
    all_in_ratios_by_street: dict[int, dict[str, list[float]]] = {
        street: {
            "bet_to_pot_before": [],
            "payment_over_pot_after_call": [],
            "raise_over_pot_after_call": [],
            "spr_after_call": [],
        }
        for street in range(4)
    }
    all_in_flags_by_assignment: list[list[bool]] = []
    started = time.perf_counter()

    for candidate_seat in range(2):
        env = HeadsUpHoldemEngine(
            starting_stack=int(environment["starting_stack"]),
            small_blind=int(environment["small_blind"]),
            big_blind=int(environment["big_blind"]),
            seed=int(seed),
        )
        opponent = opponent_factory(env)
        seat_values: list[float] = []
        seat_buttons: list[bool] = []
        seat_all_in: list[bool] = []
        for batch_start in range(0, games_per_assignment, int(simulation_batch_size)):
            batch_count = min(
                int(simulation_batch_size), games_per_assignment - batch_start
            )
            global_games = range(batch_start, batch_start + batch_count)
            states = [env.new_hand(button=game % 2) for game in global_games]
            candidate_is_button = [int(state.button) == candidate_seat for state in states]
            action_rngs = [
                random.Random(int(seed) + 1_000_003 * (game + 1))
                for game in global_games
            ]
            steps = [0] * batch_count
            candidate_all_in = [False] * batch_count
            while True:
                live = [index for index, state in enumerate(states) if not state.terminal]
                if not live:
                    break
                for start in range(0, len(live), int(inference_batch_size)):
                    indices = live[start : start + int(inference_batch_size)]
                    candidate_indices = [
                        index
                        for index in indices
                        if int(states[index].to_act) == candidate_seat
                    ]
                    opponent_indices = [
                        index for index in indices if index not in candidate_indices
                    ]
                    chosen: dict[int, int] = {}
                    if candidate_indices:
                        probabilities = candidate.probabilities_batch(
                            [states[index] for index in candidate_indices]
                        )
                        actions = _draw_actions(
                            probabilities,
                            [action_rngs[index] for index in candidate_indices],
                            candidate_indices,
                        )
                        chosen.update(zip(candidate_indices, actions))
                    if opponent_indices:
                        probabilities = opponent.probabilities_batch(
                            [states[index] for index in opponent_indices]
                        )
                        actions = _draw_actions(
                            probabilities,
                            [action_rngs[index] for index in opponent_indices],
                            opponent_indices,
                        )
                        chosen.update(zip(opponent_indices, actions))
                    for index in indices:
                        state = states[index]
                        actor = int(state.to_act)
                        action = int(chosen[index])
                        if action not in env.legal_actions(state):
                            raise RuntimeError(f"provider selected illegal action {action}")
                        if actor == candidate_seat:
                            candidate_actions[action] += 1
                            candidate_actions_by_street[(int(state.street), action)] += 1
                            candidate_all_in[index] |= action == ACTION_ALL_IN
                            if action == ACTION_ALL_IN:
                                pot_before = float(state.pot)
                                stack_before = float(state.stacks[actor])
                                to_call = max(
                                    0.0,
                                    float(state.current_bet)
                                    - float(state.street_contrib[actor]),
                                )
                                call_payment = min(stack_before, to_call)
                                pot_after_call = pot_before + call_payment
                                aggressive_payment = max(
                                    0.0, stack_before - call_payment
                                )
                                effective_after_call = min(
                                    max(0.0, stack_before - call_payment),
                                    float(state.stacks[1 - actor]),
                                )
                                bet_ratio = stack_before / pot_before
                                payment_ratio = stack_before / pot_after_call
                                raise_ratio = aggressive_payment / pot_after_call
                                spr_after_call = (
                                    effective_after_call / pot_after_call
                                )
                                all_in_bet_to_pot_before.append(bet_ratio)
                                all_in_payment_over_pot_after_call.append(payment_ratio)
                                all_in_raise_over_pot_after_call.append(raise_ratio)
                                all_in_spr_after_call.append(spr_after_call)
                                street_ratios = all_in_ratios_by_street[
                                    int(state.street)
                                ]
                                street_ratios["bet_to_pot_before"].append(bet_ratio)
                                street_ratios[
                                    "payment_over_pot_after_call"
                                ].append(payment_ratio)
                                street_ratios[
                                    "raise_over_pot_after_call"
                                ].append(raise_ratio)
                                street_ratios["spr_after_call"].append(
                                    spr_after_call
                                )
                        else:
                            opponent_actions[action] += 1
                        states[index] = env.step(state, action)
                        steps[index] += 1
                        if steps[index] > 512:
                            raise RuntimeError("evaluation hand exceeded 512 actions")
            seat_values.extend(
                float(state.payoffs[candidate_seat]) / float(env.bb) for state in states
            )
            seat_buttons.extend(candidate_is_button)
            seat_all_in.extend(candidate_all_in)
        assignment_values.append(seat_values)
        assignment_buttons.append(seat_buttons)
        all_in_flags_by_assignment.append(seat_all_in)

    values = assignment_values[0] + assignment_values[1]
    paired = [
        mean((assignment_values[0][game], assignment_values[1][game]))
        for game in range(games_per_assignment)
    ]
    estimate = mean(values)
    stderr = _stderr(paired)
    button_values = [
        value
        for values_for_seat, buttons_for_seat in zip(assignment_values, assignment_buttons)
        for value, is_button in zip(values_for_seat, buttons_for_seat)
        if is_button
    ]
    blind_values = [
        value
        for values_for_seat, buttons_for_seat in zip(assignment_values, assignment_buttons)
        for value, is_button in zip(values_for_seat, buttons_for_seat)
        if not is_button
    ]
    all_in_values = [
        value
        for values_for_seat, flags_for_seat in zip(assignment_values, all_in_flags_by_assignment)
        for value, flag in zip(values_for_seat, flags_for_seat)
        if flag
    ]
    non_all_in_values = [
        value
        for values_for_seat, flags_for_seat in zip(assignment_values, all_in_flags_by_assignment)
        for value, flag in zip(values_for_seat, flags_for_seat)
        if not flag
    ]

    intervals = {}
    for label, z in (("95", 1.959963984540054), ("99", 2.5758293035489004), ("99_9", 3.2905267314919255)):
        intervals[label] = {
            "low_bb_per_hand": estimate - z * stderr,
            "high_bb_per_hand": estimate + z * stderr,
            "low_bb_per_100": 100.0 * (estimate - z * stderr),
            "high_bb_per_100": 100.0 * (estimate + z * stderr),
        }
    if intervals["99"]["low_bb_per_hand"] > 0.0:
        verdict_99 = "profitable"
    elif intervals["99"]["high_bb_per_hand"] < 0.0:
        verdict_99 = "unprofitable"
    else:
        verdict_99 = "inconclusive"

    return {
        "candidate": candidate.name,
        "opponent": opponent.name,
        "hands": int(hands),
        "reciprocal_pairs": games_per_assignment,
        "seed": int(seed),
        "mean_ev_bb_per_hand": estimate,
        "mean_ev_bb_per_100": 100.0 * estimate,
        "paired_stderr_bb_per_hand": stderr,
        "confidence_intervals": intervals,
        "verdict_at_99_percent": verdict_99,
        "candidate_as_seat_0_bb_per_hand": mean(assignment_values[0]),
        "candidate_as_seat_1_bb_per_hand": mean(assignment_values[1]),
        "candidate_button_bb_per_hand": mean(button_values),
        "candidate_big_blind_bb_per_hand": mean(blind_values),
        "wins": sum(value > 0.0 for value in values),
        "losses": sum(value < 0.0 for value in values),
        "ties": sum(value == 0.0 for value in values),
        "candidate_action_counts": {
            ACTION_NAMES[action]: int(candidate_actions[action]) for action in range(NUM_ACTIONS)
        },
        "candidate_action_counts_by_street": {
            f"{street}:{ACTION_NAMES[action]}": int(count)
            for (street, action), count in sorted(candidate_actions_by_street.items())
        },
        "opponent_action_counts": {
            ACTION_NAMES[action]: int(opponent_actions[action]) for action in range(NUM_ACTIONS)
        },
        "candidate_all_in_hands": len(all_in_values),
        "candidate_all_in_hand_rate": len(all_in_values) / len(values),
        "candidate_all_in_net_bb": sum(all_in_values),
        "candidate_non_all_in_net_bb": sum(non_all_in_values),
        "candidate_all_in_bet_to_pot_ratio": {
            "definition": "full payment now divided by pot before acting",
            **_ratio_summary(all_in_bet_to_pot_before),
        },
        "candidate_all_in_payment_over_pot_after_call": {
            "definition": "full payment now divided by pot after matching the call",
            **_ratio_summary(all_in_payment_over_pot_after_call),
        },
        "candidate_all_in_raise_over_pot_after_call": {
            "definition": "payment beyond the call divided by pot after matching the call",
            **_ratio_summary(all_in_raise_over_pot_after_call),
        },
        "candidate_all_in_spr_after_call": {
            "definition": (
                "effective stack remaining after matching the current bet "
                "divided by pot after that call; sampled only when the "
                "candidate chooses all-in"
            ),
            **_ratio_summary(all_in_spr_after_call),
        },
        "candidate_all_in_ratios_by_street": {
            str(street): {
                name: _ratio_summary(values)
                for name, values in ratios.items()
            }
            for street, ratios in all_in_ratios_by_street.items()
        },
        "elapsed_seconds": time.perf_counter() - started,
    }


def build_suite(
    policy_paths: Sequence[Path],
    *,
    device: str,
    top_k: int,
    starting_stack: int | None = None,
):
    snapshots = [load_policy_snapshot(Path(path), device=device) for path in policy_paths]
    metadata = _compatibility_metadata(snapshots)
    environment = dict(metadata["environment"])
    if starting_stack is not None:
        if int(starting_stack) <= int(environment["big_blind"]):
            raise ValueError("starting_stack must exceed the big blind")
        environment["starting_stack"] = int(starting_stack)
    encoder_env = HeadsUpHoldemEngine(
        starting_stack=int(environment["starting_stack"]),
        small_blind=int(environment["small_blind"]),
        big_blind=int(environment["big_blind"]),
        seed=1,
    )
    trainer = HeadsUpNeuralCFR(
        encoder_env,
        device=device,
        hidden=int(metadata["hidden"]),
        blocks=int(metadata["blocks"]),
        advantage_capacity=1,
        policy_capacity=1,
        max_history=int(metadata["max_history"]),
        seed=1,
    )
    candidate = EnsembleProvider(trainer, snapshots, top_k=top_k)
    full_ensemble = EnsembleProvider(trainer, snapshots, top_k=0)
    components = [SnapshotProvider(trainer, snapshot) for snapshot in snapshots]
    return candidate, full_ensemble, components, environment, snapshots


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policies", nargs="+", type=Path, default=list(DEFAULT_POLICIES))
    parser.add_argument("--hands", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=725_950_1025)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--starting-stack",
        type=int,
        help="evaluation-only starting stack override in chips",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--inference-batch-size", type=int, default=2_048)
    parser.add_argument("--simulation-batch-size", type=int, default=20_000)
    parser.add_argument(
        "--opponents",
        nargs="+",
        default=(
            "random",
            "calling_station",
            "tight_aggressive",
            "loose_aggressive",
            "all_in_pressure",
            "blind_steal_pressure",
            "ensemble_full",
            "policy_725",
            "policy_950",
            "policy_1025",
        ),
    )
    parser.add_argument("--include-self-check", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    candidate, full_ensemble, components, environment, snapshots = build_suite(
        args.policies,
        device=args.device,
        top_k=args.top_k,
        starting_stack=args.starting_stack,
    )
    policy_by_name = {provider.name: provider for provider in components}
    ensemble_opponents: dict[int, EnsembleProvider] = {}
    matches = []
    requested = list(args.opponents)
    if args.include_self_check:
        requested.insert(0, "self_top_k")
    for match_index, opponent_name in enumerate(requested):
        if opponent_name == "self_top_k":
            factory = lambda env, provider=candidate: provider
        elif opponent_name == "ensemble_full":
            factory = lambda env, provider=full_ensemble: provider
        elif (opponent_top_k := ensemble_top_k_from_name(opponent_name)) is not None:
            provider = ensemble_opponents.setdefault(
                opponent_top_k,
                EnsembleProvider(
                    candidate.trainer,
                    snapshots,
                    top_k=opponent_top_k,
                ),
            )
            factory = lambda env, provider=provider: provider
        elif opponent_name in policy_by_name:
            factory = lambda env, provider=policy_by_name[opponent_name]: provider
        elif opponent_name in OPPONENT_PROFILES:
            opponent_type = OPPONENT_PROFILES[opponent_name]
            factory = lambda env, opponent_type=opponent_type: ScriptedProvider(
                env, opponent_type()
            )
        elif opponent_name in EXTRA_PROFILES:
            opponent_type = EXTRA_PROFILES[opponent_name]
            factory = lambda env, opponent_type=opponent_type: ScriptedProvider(
                env, opponent_type()
            )
        else:
            raise ValueError(f"unknown opponent {opponent_name!r}")
        result = run_reciprocal_match(
            candidate,
            factory,
            environment=environment,
            hands=int(args.hands),
            seed=int(args.seed) + match_index * 10_000_019,
            inference_batch_size=int(args.inference_batch_size),
            simulation_batch_size=int(args.simulation_batch_size),
        )
        matches.append(result)
        print(
            f"{result['opponent']}: {result['mean_ev_bb_per_100']:+.3f} BB/100 "
            f"99% CI [{result['confidence_intervals']['99']['low_bb_per_100']:+.3f}, "
            f"{result['confidence_intervals']['99']['high_bb_per_100']:+.3f}] "
            f"=> {result['verdict_at_99_percent']}",
            flush=True,
        )

    result = {
        "method": "common-deal reciprocal seat-swapped sampled matches",
        "scope": "rake-free two-player no-limit Hold'em engine only",
        "finite_test_warning": (
            "No finite opponent suite can prove universal profitability with 100% certainty."
        ),
        "candidate": candidate.name,
        "component_iterations": [snapshot.iteration for snapshot in snapshots],
        "top_k": int(args.top_k),
        "device": str(args.device),
        "environment": environment,
        "hands_per_match": int(args.hands),
        "matches": matches,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
