"""Differential validation of scalar-Python and optimized HU root search."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import torch

from heads_up_engine import ACTION_NAMES, HeadsUpHoldemEngine
from heads_up_native import (
    HeadsUpHoldemEngine as NativeHeadsUpHoldemEngine,
    reference_state_to_native,
)
from heads_up_pluribus_search import BlueprintPublicRange, PublicRangeSnapshot
from heads_up_root_policy_search import HeadsUpRootPolicySearch
from play_heads_up_gui import HeadsUpSnapshotPolicy


class ScalarPythonPolicy:
    """Expose the original one-state-at-a-time policy contract."""

    def __init__(self, policy: HeadsUpSnapshotPolicy) -> None:
        self.policy = policy

    def probabilities(self, env, state) -> torch.Tensor:
        return self.policy._legacy_single_probabilities(env, state)


def _continuation_action(env, state, rng: random.Random) -> int:
    legal = [int(action) for action in env.legal_actions(state)]
    by_name = {ACTION_NAMES[action]: action for action in legal}
    if "check" in by_name:
        if rng.random() < 0.68:
            return by_name["check"]
        candidates = [
            by_name[name]
            for name in ("third_pot", "half_pot", "three_quarter_pot")
            if name in by_name
        ]
        return rng.choice(candidates) if candidates else by_name["check"]
    if "call" in by_name:
        if rng.random() < 0.78:
            return by_name["call"]
        candidates = [
            by_name[name]
            for name in ("min_raise", "half_pot", "three_quarter_pot")
            if name in by_name
        ]
        return rng.choice(candidates) if candidates else by_name["call"]
    candidates = [
        action
        for action in legal
        if ACTION_NAMES[action] not in {"fold", "all_in"}
    ]
    return rng.choice(candidates or legal)


def collect_states(count: int, seed: int) -> list:
    env = HeadsUpHoldemEngine(200, 1, 2, seed=seed)
    rng = random.Random(seed)
    per_street = max(1, math.ceil(count / 4))
    street_counts: Counter[int] = Counter()
    states = []
    hand = 0
    while len(states) < count and hand < count * 8:
        state = env.new_hand(button=hand % 2)
        actions = 0
        while not state.terminal and actions < 48:
            street = int(state.street)
            if street_counts[street] < per_street:
                states.append(copy.deepcopy(state))
                street_counts[street] += 1
                if len(states) >= count:
                    break
            state = env.step(state, _continuation_action(env, state, rng))
            actions += 1
        hand += 1
    if len(states) < count:
        raise RuntimeError(
            f"could collect only {len(states)} of {count} requested states"
        )
    return states[:count]


def _synthetic_range(state, index: int) -> PublicRangeSnapshot:
    hero = int(state.to_act)
    public_range = BlueprintPublicRange()
    public_range.reset([*state.hole[hero], *state.board])
    snapshot = public_range.snapshot()
    if index % 2 == 0:
        return snapshot
    raw = [
        float(1 + ((17 * first + 31 * second + index) % 29))
        for first, second in snapshot.combos
    ]
    total = sum(raw)
    weights = tuple(value / total for value in raw)
    return PublicRangeSnapshot(
        combos=snapshot.combos,
        weights=weights,
        effective_sample_size=1.0 / sum(value * value for value in weights),
        updates=3,
    )


def _candidate_map(result) -> dict[int, object]:
    return {int(row.action.action): row for row in result.candidates}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("policy", type=Path)
    parser.add_argument("--states", type=int, default=24)
    parser.add_argument("--iterations-per-state", type=int, default=16)
    parser.add_argument("--gui-floor", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.states <= 0:
        parser.error("--states must be positive")
    if args.iterations_per_state <= 0:
        parser.error("--iterations-per-state must be positive")
    if not 0.0 <= args.gui_floor < 1.0:
        parser.error("--gui-floor must be in [0, 1)")

    policy = HeadsUpSnapshotPolicy(
        args.policy,
        device="auto",
        seed=args.seed,
    )
    scalar_policy = ScalarPythonPolicy(policy)
    env = HeadsUpHoldemEngine(200, 1, 2, seed=args.seed)
    native_env = NativeHeadsUpHoldemEngine(200, 1, 2, seed=args.seed)
    states = collect_states(args.states, args.seed)

    max_policy_probability_difference = 0.0
    policy_argmax_mismatches = 0
    for state in states:
        scalar = policy._legacy_single_probabilities(env, state)
        native_state = reference_state_to_native(state)
        optimized = policy.probabilities_batch(native_env, [native_state])[0]
        difference = float(torch.max(torch.abs(scalar - optimized)).item())
        max_policy_probability_difference = max(
            max_policy_probability_difference,
            difference,
        )
        policy_argmax_mismatches += int(
            int(torch.argmax(scalar)) != int(torch.argmax(optimized))
        )

    rows = []
    maximum_ev_difference = 0.0
    maximum_strategy_difference = 0.0
    maximum_standard_error_difference = 0.0
    choice_mismatches = 0
    sample_count_mismatches = 0
    old_seconds = 0.0
    optimized_seconds = 0.0
    gui_floor_choice_changes = 0
    gui_floor_pruned_candidates = 0
    gui_floor_max_strategy_difference = 0.0
    gui_floor_max_ev_difference = 0.0
    for index, state in enumerate(states):
        blueprint = policy._legacy_single_probabilities(env, state)
        public_range = _synthetic_range(state, index)
        legal_count = len(env.legal_actions(state))
        max_rollouts = legal_count * args.iterations_per_state
        search_seed = args.seed + 10_000 + index

        started = time.perf_counter()
        old = HeadsUpRootPolicySearch(
            scalar_policy,
            time_budget_ms=60_000,
            max_rollouts=max_rollouts,
            batch_iterations=1,
            use_native_rollouts=False,
            use_batched_action_sampling=False,
            use_batch_step=False,
            range_mode="inferred",
            range_temperature=0.65,
            uniform_contamination=0.25,
            min_strategy_probability=0.0,
            seed=search_seed,
        ).resolve(env, state, blueprint, public_range)
        old_elapsed = time.perf_counter() - started

        started = time.perf_counter()
        optimized = HeadsUpRootPolicySearch(
            policy,
            time_budget_ms=60_000,
            max_rollouts=max_rollouts,
            batch_iterations=args.iterations_per_state,
            use_native_rollouts=True,
            use_batched_action_sampling=True,
            use_batch_step=True,
            range_mode="inferred",
            range_temperature=0.65,
            uniform_contamination=0.25,
            min_strategy_probability=0.0,
            seed=search_seed,
        ).resolve(env, state, blueprint, public_range)
        optimized_elapsed = time.perf_counter() - started
        old_seconds += old_elapsed
        optimized_seconds += optimized_elapsed

        gui_floor = HeadsUpRootPolicySearch(
            policy,
            time_budget_ms=60_000,
            max_rollouts=max_rollouts,
            batch_iterations=args.iterations_per_state,
            use_native_rollouts=True,
            use_batched_action_sampling=True,
            use_batch_step=True,
            range_mode="inferred",
            range_temperature=0.65,
            uniform_contamination=0.25,
            min_strategy_probability=args.gui_floor,
            seed=search_seed,
        ).resolve(env, state, blueprint, public_range)

        old_candidates = _candidate_map(old)
        new_candidates = _candidate_map(optimized)
        gui_floor_candidates = _candidate_map(gui_floor)
        state_ev_difference = 0.0
        state_strategy_difference = 0.0
        state_standard_error_difference = 0.0
        for action in old_candidates:
            old_row = old_candidates[action]
            new_row = new_candidates[action]
            state_ev_difference = max(
                state_ev_difference,
                abs(
                    float(old_row.expected_final_payoff_bb)
                    - float(new_row.expected_final_payoff_bb)
                ),
            )
            state_strategy_difference = max(
                state_strategy_difference,
                abs(
                    float(old_row.strategy_probability)
                    - float(new_row.strategy_probability)
                ),
            )
            state_standard_error_difference = max(
                state_standard_error_difference,
                abs(
                    float(old_row.standard_error_bb)
                    - float(new_row.standard_error_bb)
                ),
            )
            sample_count_mismatches += int(
                int(old_row.samples) != int(new_row.samples)
            )
            floor_row = gui_floor_candidates[action]
            gui_floor_max_strategy_difference = max(
                gui_floor_max_strategy_difference,
                abs(
                    float(new_row.strategy_probability)
                    - float(floor_row.strategy_probability)
                ),
            )
            gui_floor_max_ev_difference = max(
                gui_floor_max_ev_difference,
                abs(
                    float(new_row.expected_final_payoff_bb)
                    - float(floor_row.expected_final_payoff_bb)
                ),
            )
            gui_floor_pruned_candidates += int(floor_row.safety_pruned)
        maximum_ev_difference = max(
            maximum_ev_difference,
            state_ev_difference,
        )
        maximum_strategy_difference = max(
            maximum_strategy_difference,
            state_strategy_difference,
        )
        maximum_standard_error_difference = max(
            maximum_standard_error_difference,
            state_standard_error_difference,
        )
        choice_mismatch = int(
            int(old.choice.action) != int(optimized.choice.action)
        )
        choice_mismatches += choice_mismatch
        floor_choice_change = int(
            int(optimized.choice.action) != int(gui_floor.choice.action)
        )
        gui_floor_choice_changes += floor_choice_change
        rows.append(
            {
                "index": index,
                "street": int(state.street),
                "actor": int(state.to_act),
                "history_events": len(state.history),
                "legal_actions": legal_count,
                "range_updates": public_range.updates,
                "rollouts": max_rollouts,
                "old_choice": old.choice.label,
                "optimized_choice": optimized.choice.label,
                "choice_mismatch": bool(choice_mismatch),
                "gui_floor_choice": gui_floor.choice.label,
                "gui_floor_choice_changed": bool(floor_choice_change),
                "gui_floor_pruned_candidates": sum(
                    int(row.safety_pruned)
                    for row in gui_floor.candidates
                ),
                "max_ev_abs_difference_bb": state_ev_difference,
                "max_strategy_abs_difference": state_strategy_difference,
                "max_standard_error_abs_difference_bb": (
                    state_standard_error_difference
                ),
                "old_seconds": old_elapsed,
                "optimized_seconds": optimized_elapsed,
            }
        )

    tolerance = 1e-6
    equivalent = (
        max_policy_probability_difference <= tolerance
        and policy_argmax_mismatches == 0
        and maximum_ev_difference <= tolerance
        and maximum_strategy_difference <= tolerance
        and maximum_standard_error_difference <= tolerance
        and choice_mismatches == 0
        and sample_count_mismatches == 0
    )
    report = {
        "policy": str(args.policy.resolve()),
        "policy_iteration": policy.iteration,
        "policy_sha256": policy.sha256,
        "device": str(policy.device),
        "states": len(states),
        "street_counts": dict(
            sorted(Counter(int(state.street) for state in states).items())
        ),
        "iterations_per_state": args.iterations_per_state,
        "policy_inference": {
            "max_probability_abs_difference": (
                max_policy_probability_difference
            ),
            "argmax_mismatches": policy_argmax_mismatches,
        },
        "fixed_rollout_search": {
            "max_ev_abs_difference_bb": maximum_ev_difference,
            "max_strategy_abs_difference": maximum_strategy_difference,
            "max_standard_error_abs_difference_bb": (
                maximum_standard_error_difference
            ),
            "choice_mismatches": choice_mismatches,
            "sample_count_mismatches": sample_count_mismatches,
            "old_scalar_python_seconds": old_seconds,
            "optimized_seconds": optimized_seconds,
            "fixed_count_speed_multiplier": (
                old_seconds / optimized_seconds
                if optimized_seconds > 0.0
                else float("inf")
            ),
        },
        "intentional_gui_strategy_change": {
            "minimum_strategy_probability": args.gui_floor,
            "choice_changes": gui_floor_choice_changes,
            "pruned_candidates": gui_floor_pruned_candidates,
            "max_strategy_abs_difference": (
                gui_floor_max_strategy_difference
            ),
            "max_ev_abs_difference_bb": gui_floor_max_ev_difference,
            "note": (
                "The GUI floor is applied after EV estimation. It leaves "
                "rollout EVs unchanged but intentionally changes the final "
                "mixed strategy and sampled action."
            ),
        },
        "tolerance": tolerance,
        "mathematically_equivalent_within_tolerance": equivalent,
        "states_detail": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(report, indent=2))
    return 0 if equivalent else 1


if __name__ == "__main__":
    raise SystemExit(main())
