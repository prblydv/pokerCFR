"""Reciprocal HU match between two root-search rollout budgets."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

from heads_up_engine import HeadsUpHoldemEngine
from heads_up_pluribus_search import (
    BlueprintPublicRange,
    observed_action_likelihoods,
)
from heads_up_root_policy_search import HeadsUpRootPolicySearch
from play_heads_up_gui import HeadsUpSnapshotPolicy


@dataclass(frozen=True)
class SearchSpec:
    name: str
    time_budget_ms: int
    max_rollouts: int
    batch_iterations: int


def _summary(values: list[float]) -> dict:
    count = len(values)
    mean = statistics.fmean(values) if values else 0.0
    sd = statistics.stdev(values) if count > 1 else 0.0
    se = sd / math.sqrt(count) if count else float("inf")
    return {
        "samples": count,
        "mean_bb": mean,
        "sample_sd_bb": sd,
        "standard_error_bb": se,
        "ci95_low_bb": mean - 1.96 * se,
        "ci95_high_bb": mean + 1.96 * se,
    }


def _observe_action(
    policy,
    env,
    before,
    after,
    observer_seat: int,
    public_range: BlueprintPublicRange,
) -> None:
    event = after.history[-1]
    public_range.filter_known(
        [*before.hole[observer_seat], *before.board]
    )
    probabilities = policy.probabilities_for_holes(
        env,
        before,
        int(before.to_act),
        public_range.combos,
    )
    likelihoods = observed_action_likelihoods(
        env,
        before,
        probabilities,
        kind=str(event.kind),
        raise_to=(
            int(event.raise_to)
            if str(event.kind) in {"bet", "raise"}
            else None
        ),
    )
    public_range.condition(likelihoods)


def _play_hand(
    policy,
    *,
    deck: list[int],
    button: int,
    seat_specs: dict[int, SearchSpec],
    role_seeds: dict[str, int],
) -> dict:
    env = HeadsUpHoldemEngine(200, 1, 2, seed=role_seeds["deal"])
    state = env.new_hand(button=button, deck=list(deck))
    ranges = {}
    searches = {}
    metrics = {}
    for seat, spec in seat_specs.items():
        public_range = BlueprintPublicRange()
        public_range.reset([*state.hole[seat], *state.board])
        ranges[seat] = public_range
        searches[seat] = HeadsUpRootPolicySearch(
            policy,
            time_budget_ms=spec.time_budget_ms,
            max_rollouts=spec.max_rollouts,
            batch_iterations=spec.batch_iterations,
            use_native_rollouts=True,
            use_batched_action_sampling=True,
            use_batch_step=True,
            range_mode="inferred",
            range_temperature=0.65,
            uniform_contamination=0.25,
            min_strategy_probability=0.0,
            seed=role_seeds[spec.name],
        )
        metrics[spec.name] = {
            "decisions": 0,
            "rollouts": 0,
            "search_seconds": 0.0,
            "maximum_rollouts_in_one_decision": 0,
        }

    actions = 0
    while not state.terminal:
        before = state
        actor = int(state.to_act)
        spec = seat_specs[actor]
        public_range = ranges[actor]
        public_range.filter_known(
            [*state.hole[actor], *state.board]
        )
        blueprint = policy.probabilities(env, state)
        started = time.perf_counter()
        result = searches[actor].resolve(
            env,
            state,
            blueprint,
            public_range.snapshot(),
        )
        decision_seconds = time.perf_counter() - started
        role_metrics = metrics[spec.name]
        role_metrics["decisions"] += 1
        role_metrics["rollouts"] += int(result.terminal_rollouts)
        role_metrics["search_seconds"] += decision_seconds
        role_metrics["maximum_rollouts_in_one_decision"] = max(
            int(role_metrics["maximum_rollouts_in_one_decision"]),
            int(result.terminal_rollouts),
        )
        if int(result.terminal_rollouts) > spec.max_rollouts:
            raise RuntimeError(
                f"{spec.name} exceeded rollout cap: "
                f"{result.terminal_rollouts} > {spec.max_rollouts}"
            )
        state = env.step(state, int(result.choice.action))
        observer = 1 - actor
        _observe_action(
            policy,
            env,
            before,
            state,
            observer,
            ranges[observer],
        )
        actions += 1
        if actions > 64:
            raise RuntimeError("evaluation hand exceeded 64 actions")

    if int(state.payoffs[0]) != -int(state.payoffs[1]):
        raise RuntimeError("terminal payoff is not exactly zero-sum")
    role_payoffs = {
        seat_specs[seat].name: float(state.payoffs[seat]) / float(env.bb)
        for seat in (0, 1)
    }
    return {
        "button": button,
        "role_payoffs_bb": role_payoffs,
        "metrics": metrics,
        "actions": actions,
    }


def _merge_metrics(rows: list[dict], role: str) -> dict:
    decisions = sum(
        int(row["metrics"][role]["decisions"]) for row in rows
    )
    rollouts = sum(
        int(row["metrics"][role]["rollouts"]) for row in rows
    )
    seconds = sum(
        float(row["metrics"][role]["search_seconds"]) for row in rows
    )
    maximum = max(
        (
            int(
                row["metrics"][role][
                    "maximum_rollouts_in_one_decision"
                ]
            )
            for row in rows
        ),
        default=0,
    )
    return {
        "decisions": decisions,
        "terminal_rollouts": rollouts,
        "mean_rollouts_per_decision": (
            rollouts / decisions if decisions else 0.0
        ),
        "search_seconds": seconds,
        "mean_search_seconds_per_decision": (
            seconds / decisions if decisions else 0.0
        ),
        "maximum_rollouts_in_one_decision": maximum,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("policy", type=Path)
    parser.add_argument("--hands", type=int, default=1000)
    parser.add_argument("--limited-rollouts", type=int, default=250)
    parser.add_argument("--limited-batch-iterations", type=int, default=1024)
    parser.add_argument("--timed-budget-ms", type=int, default=700)
    parser.add_argument("--timed-max-rollouts", type=int, default=150_000)
    parser.add_argument("--timed-batch-iterations", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.hands <= 0 or args.hands % 2:
        parser.error("--hands must be a positive even number")
    if args.limited_rollouts <= 0:
        parser.error("--limited-rollouts must be positive")
    if args.timed_budget_ms <= 0 or args.timed_max_rollouts <= 0:
        parser.error("timed search limits must be positive")

    limited = SearchSpec(
        "limited_250",
        time_budget_ms=60_000,
        max_rollouts=args.limited_rollouts,
        batch_iterations=args.limited_batch_iterations,
    )
    timed = SearchSpec(
        "timed_1s",
        time_budget_ms=args.timed_budget_ms,
        max_rollouts=args.timed_max_rollouts,
        batch_iterations=args.timed_batch_iterations,
    )
    policy = HeadsUpSnapshotPolicy(
        args.policy,
        mode="sample",
        device="auto",
        seed=args.seed,
    )
    deck_rng = random.Random(args.seed)
    hands = []
    pair_values = []
    started = time.perf_counter()
    for pair_index in range(args.hands // 2):
        deck = list(range(52))
        deck_rng.shuffle(deck)
        button = pair_index % 2
        pair_seed = args.seed + pair_index * 10
        role_seeds = {
            "deal": pair_seed,
            limited.name: pair_seed ^ 0xD1B54A32D192ED03,
            timed.name: pair_seed ^ 0x9E3779B97F4A7C15,
        }
        first = _play_hand(
            policy,
            deck=deck,
            button=button,
            seat_specs={0: limited, 1: timed},
            role_seeds=role_seeds,
        )
        second = _play_hand(
            policy,
            deck=deck,
            button=button,
            seat_specs={0: timed, 1: limited},
            role_seeds=role_seeds,
        )
        hands.extend((first, second))
        pair_values.append(
            0.5
            * (
                float(first["role_payoffs_bb"][timed.name])
                + float(second["role_payoffs_bb"][timed.name])
            )
        )
        if (pair_index + 1) % 10 == 0:
            print(
                json.dumps(
                    {
                        "pairs_complete": pair_index + 1,
                        "hands_complete": 2 * (pair_index + 1),
                        "elapsed_seconds": time.perf_counter() - started,
                        "timed_1s_ev_bb_per_hand": statistics.fmean(
                            pair_values
                        ),
                    }
                ),
                flush=True,
            )

    elapsed = time.perf_counter() - started
    timed_payoffs = [
        float(row["role_payoffs_bb"][timed.name]) for row in hands
    ]
    wins = sum(value > 0.0 for value in timed_payoffs)
    losses = sum(value < 0.0 for value in timed_payoffs)
    ties = len(timed_payoffs) - wins - losses
    report = {
        "policy": str(args.policy.resolve()),
        "policy_iteration": policy.iteration,
        "policy_sha256": policy.sha256,
        "hands": args.hands,
        "reciprocal_deal_pairs": args.hands // 2,
        "one_second_search": {
            "internal_time_budget_ms": args.timed_budget_ms,
            "max_rollouts": args.timed_max_rollouts,
            "batch_iterations": args.timed_batch_iterations,
            **_merge_metrics(hands, timed.name),
        },
        "limited_search": {
            "rollout_cap": args.limited_rollouts,
            "batch_iterations": args.limited_batch_iterations,
            **_merge_metrics(hands, limited.name),
        },
        "one_second_ev_vs_limited": _summary(pair_values),
        "limited_ev_bb_per_hand": -statistics.fmean(pair_values),
        "one_second_wins": wins,
        "one_second_losses": losses,
        "ties": ties,
        "hand_level_one_second_payoff": _summary(timed_payoffs),
        "wall_seconds": elapsed,
        "zero_sum_verified": True,
        "minimum_strategy_probability": 0.0,
        "range_model": {
            "mode": "inferred",
            "temperature": 0.65,
            "uniform_contamination": 0.25,
        },
        "pairing": (
            f"{args.hands // 2} shared deck/button pairs; "
            "250-rollout and one-second controllers swap seats within "
            "every pair"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
