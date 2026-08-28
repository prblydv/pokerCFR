"""Measure how far HU root search moves from its policy at two rollout caps."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from heads_up_engine import ACTION_NAMES, HeadsUpHoldemEngine
from heads_up_root_policy_search import HeadsUpRootPolicySearch
from play_heads_up_gui import HeadsUpSnapshotPolicy
from validate_heads_up_root_search_equivalence import (
    _synthetic_range,
    collect_states,
)


def _legal_policy(blueprint, legal: list[int]) -> dict[int, float]:
    total = sum(float(blueprint[action]) for action in legal)
    if total <= 1e-12:
        return {action: 1.0 / len(legal) for action in legal}
    return {action: float(blueprint[action]) / total for action in legal}


def _result_strategy(result) -> dict[int, float]:
    return {
        int(candidate.action.action): float(candidate.strategy_probability)
        for candidate in result.candidates
    }


def _total_variation(
    first: dict[int, float],
    second: dict[int, float],
) -> float:
    return 0.5 * sum(
        abs(first[action] - second[action]) for action in first
    )


def _run_search(
    policy,
    env,
    state,
    blueprint,
    public_range,
    *,
    rollout_cap: int,
    seed: int,
):
    legal_count = len(env.legal_actions(state))
    batch_iterations = min(1024, max(1, rollout_cap // legal_count))
    started = time.perf_counter()
    result = HeadsUpRootPolicySearch(
        policy,
        time_budget_ms=120_000,
        max_rollouts=rollout_cap,
        batch_iterations=batch_iterations,
        use_native_rollouts=True,
        use_batched_action_sampling=True,
        use_batch_step=True,
        range_mode="inferred",
        range_temperature=0.65,
        uniform_contamination=0.25,
        blueprint_weight=0.35,
        min_strategy_probability=0.10,
        seed=seed,
    ).resolve(env, state, blueprint, public_range)
    return result, time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("policy", type=Path)
    parser.add_argument("--scenarios", type=int, default=10)
    parser.add_argument("--low-rollouts", type=int, default=750)
    parser.add_argument("--high-rollouts", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.scenarios <= 0:
        parser.error("--scenarios must be positive")
    if args.low_rollouts <= 0 or args.high_rollouts <= 0:
        parser.error("rollout caps must be positive")
    if args.low_rollouts >= args.high_rollouts:
        parser.error("--low-rollouts must be below --high-rollouts")

    policy = HeadsUpSnapshotPolicy(args.policy, device="auto", seed=args.seed)
    env = HeadsUpHoldemEngine(200, 1, 2, seed=args.seed)
    states = collect_states(args.scenarios, args.seed)
    rows = []

    for index, state in enumerate(states):
        legal = [int(action) for action in env.legal_actions(state)]
        blueprint_tensor = policy._legacy_single_probabilities(env, state)
        blueprint = _legal_policy(blueprint_tensor, legal)
        public_range = _synthetic_range(state, index)
        search_seed = args.seed + 10_000 + index

        low, low_seconds = _run_search(
            policy,
            env,
            state,
            blueprint_tensor,
            public_range,
            rollout_cap=args.low_rollouts,
            seed=search_seed,
        )
        high, high_seconds = _run_search(
            policy,
            env,
            state,
            blueprint_tensor,
            public_range,
            rollout_cap=args.high_rollouts,
            seed=search_seed,
        )
        low_strategy = _result_strategy(low)
        high_strategy = _result_strategy(high)
        rows.append(
            {
                "scenario": index + 1,
                "street": int(state.street),
                "range_kind": "uniform" if index % 2 == 0 else "skewed",
                "range_ess": float(public_range.effective_sample_size),
                "legal_actions": [ACTION_NAMES[action] for action in legal],
                "policy": {
                    ACTION_NAMES[action]: blueprint[action] for action in legal
                },
                "low_final_strategy": {
                    ACTION_NAMES[action]: low_strategy[action]
                    for action in legal
                },
                "high_final_strategy": {
                    ACTION_NAMES[action]: high_strategy[action]
                    for action in legal
                },
                "low_total_variation_from_policy": _total_variation(
                    blueprint,
                    low_strategy,
                ),
                "high_total_variation_from_policy": _total_variation(
                    blueprint,
                    high_strategy,
                ),
                "low_vs_high_total_variation": _total_variation(
                    low_strategy,
                    high_strategy,
                ),
                "low_actual_rollouts": int(low.terminal_rollouts),
                "high_actual_rollouts": int(high.terminal_rollouts),
                "low_cfr_iterations": int(low.cfr_iterations),
                "high_cfr_iterations": int(high.cfr_iterations),
                "low_seconds": low_seconds,
                "high_seconds": high_seconds,
            }
        )
        print(
            f"{index + 1:02d}/{args.scenarios}: "
            f"low {100.0 * rows[-1]['low_total_variation_from_policy']:.2f}% "
            f"high {100.0 * rows[-1]['high_total_variation_from_policy']:.2f}%"
        )

    low_distances = [
        row["low_total_variation_from_policy"] for row in rows
    ]
    high_distances = [
        row["high_total_variation_from_policy"] for row in rows
    ]
    paired_differences = [
        high - low for high, low in zip(high_distances, low_distances)
    ]
    report = {
        "policy": str(args.policy.resolve()),
        "seed": args.seed,
        "scenario_count": args.scenarios,
        "configuration": {
            "low_rollout_cap": args.low_rollouts,
            "high_rollout_cap": args.high_rollouts,
            "blueprint_weight": 0.35,
            "search_weight": 0.65,
            "range_mode": "inferred",
            "range_temperature": 0.65,
            "uniform_contamination": 0.25,
            "minimum_strategy_probability": 0.10,
            "time_budget_ms": 120_000,
        },
        "metric": (
            "total variation distance from raw policy; "
            "0 means identical, 1 means disjoint"
        ),
        "summary": {
            "low_mean_distance": statistics.fmean(low_distances),
            "low_median_distance": statistics.median(low_distances),
            "high_mean_distance": statistics.fmean(high_distances),
            "high_median_distance": statistics.median(high_distances),
            "mean_paired_high_minus_low": statistics.fmean(
                paired_differences
            ),
            "states_high_farther": sum(
                difference > 1e-12 for difference in paired_differences
            ),
            "states_low_farther": sum(
                difference < -1e-12 for difference in paired_differences
            ),
            "states_equal": sum(
                abs(difference) <= 1e-12
                for difference in paired_differences
            ),
            "low_total_rollouts": sum(
                row["low_actual_rollouts"] for row in rows
            ),
            "high_total_rollouts": sum(
                row["high_actual_rollouts"] for row in rows
            ),
            "low_total_seconds": sum(row["low_seconds"] for row in rows),
            "high_total_seconds": sum(row["high_seconds"] for row in rows),
        },
        "scenarios": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
