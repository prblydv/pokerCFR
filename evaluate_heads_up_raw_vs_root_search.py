"""Reciprocal raw-policy versus inferred-range root-search evaluation."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from pathlib import Path

from heads_up_engine import HeadsUpHoldemEngine
from heads_up_pluribus_search import (
    BlueprintPublicRange,
    observed_action_likelihoods,
)
from heads_up_root_policy_search import HeadsUpRootPolicySearch
from play_heads_up_gui import HeadsUpSnapshotPolicy


def _sample(probabilities, legal, rng: random.Random) -> int:
    threshold = rng.random()
    cumulative = 0.0
    fallback = legal[-1]
    for action in legal:
        probability = float(probabilities[action])
        if probability <= 0.0:
            continue
        fallback = action
        cumulative += probability
        if threshold <= cumulative + 1e-12:
            return action
    return fallback


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


def _observe_raw_action(
    policy,
    env,
    before,
    after,
    search_seat: int,
    public_range: BlueprintPublicRange,
) -> None:
    event = after.history[-1]
    public_range.filter_known(
        [*before.hole[search_seat], *before.board]
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
    search_seat: int,
    seed: int,
    search_budget_ms: int,
    batch_iterations: int,
) -> dict:
    env = HeadsUpHoldemEngine(200, 1, 2, seed=seed)
    state = env.new_hand(button=button, deck=list(deck))
    raw_rng = random.Random(seed ^ 0xD1B54A32D192ED03)
    public_range = BlueprintPublicRange()
    public_range.reset(
        [*state.hole[search_seat], *state.board]
    )
    search = HeadsUpRootPolicySearch(
        policy,
        time_budget_ms=search_budget_ms,
        max_rollouts=150_000,
        batch_iterations=batch_iterations,
        use_native_rollouts=True,
        use_batched_action_sampling=True,
        use_batch_step=True,
        range_mode="inferred",
        range_temperature=0.65,
        uniform_contamination=0.25,
        seed=seed ^ 0x9E3779B97F4A7C15,
    )
    search_decisions = 0
    search_rollouts = 0
    search_seconds = 0.0
    actions = 0
    while not state.terminal:
        before = state
        actor = int(state.to_act)
        if actor == search_seat:
            public_range.filter_known(
                [*state.hole[search_seat], *state.board]
            )
            blueprint = policy.probabilities(env, state)
            started = time.perf_counter()
            result = search.resolve(
                env,
                state,
                blueprint,
                public_range.snapshot(),
            )
            search_seconds += time.perf_counter() - started
            search_decisions += 1
            search_rollouts += int(result.terminal_rollouts)
            action = int(result.choice.action)
        else:
            probabilities = policy.probabilities(env, state)
            legal = [int(value) for value in env.legal_actions(state)]
            action = _sample(probabilities, legal, raw_rng)
        state = env.step(state, action)
        if actor != search_seat:
            _observe_raw_action(
                policy,
                env,
                before,
                state,
                search_seat,
                public_range,
            )
        actions += 1
        if actions > 64:
            raise RuntimeError("evaluation hand exceeded 64 actions")
    if int(state.payoffs[0]) != -int(state.payoffs[1]):
        raise RuntimeError("terminal payoff is not exactly zero-sum")
    payoff_bb = float(state.payoffs[search_seat]) / float(env.bb)
    return {
        "search_seat": search_seat,
        "button": button,
        "payoff_bb": payoff_bb,
        "search_decisions": search_decisions,
        "search_rollouts": search_rollouts,
        "search_seconds": search_seconds,
        "actions": actions,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("policy", type=Path)
    parser.add_argument("--hands", type=int, default=1000)
    parser.add_argument("--search-budget-ms", type=int, default=800)
    parser.add_argument("--batch-iterations", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.hands <= 0 or args.hands % 2:
        parser.error("--hands must be a positive even number")
    if args.search_budget_ms <= 0:
        parser.error("--search-budget-ms must be positive")
    if args.batch_iterations <= 0:
        parser.error("--batch-iterations must be positive")

    policy = HeadsUpSnapshotPolicy(
        args.policy,
        mode="sample",
        device="auto",
        seed=args.seed,
    )
    deck_rng = random.Random(args.seed)
    hands: list[dict] = []
    pair_values: list[float] = []
    started = time.perf_counter()
    for pair_index in range(args.hands // 2):
        deck = list(range(52))
        deck_rng.shuffle(deck)
        button = pair_index % 2
        first = _play_hand(
            policy,
            deck=deck,
            button=button,
            search_seat=0,
            seed=args.seed + pair_index * 2,
            search_budget_ms=args.search_budget_ms,
            batch_iterations=args.batch_iterations,
        )
        second = _play_hand(
            policy,
            deck=deck,
            button=button,
            search_seat=1,
            seed=args.seed + pair_index * 2 + 1,
            search_budget_ms=args.search_budget_ms,
            batch_iterations=args.batch_iterations,
        )
        hands.extend((first, second))
        pair_values.append(0.5 * (first["payoff_bb"] + second["payoff_bb"]))
        if (pair_index + 1) % 10 == 0:
            elapsed = time.perf_counter() - started
            print(
                json.dumps(
                    {
                        "pairs_complete": pair_index + 1,
                        "hands_complete": 2 * (pair_index + 1),
                        "elapsed_seconds": elapsed,
                        "search_ev_bb_per_hand": statistics.fmean(
                            pair_values
                        ),
                    }
                ),
                flush=True,
            )

    elapsed = time.perf_counter() - started
    payoffs = [row["payoff_bb"] for row in hands]
    wins = sum(value > 0.0 for value in payoffs)
    losses = sum(value < 0.0 for value in payoffs)
    ties = len(payoffs) - wins - losses
    total_decisions = sum(row["search_decisions"] for row in hands)
    total_rollouts = sum(row["search_rollouts"] for row in hands)
    total_search_seconds = sum(row["search_seconds"] for row in hands)
    report = {
        "policy": str(args.policy.resolve()),
        "policy_iteration": policy.iteration,
        "policy_sha256": policy.sha256,
        "hands": args.hands,
        "reciprocal_deal_pairs": args.hands // 2,
        "search_budget_ms_internal": args.search_budget_ms,
        "batch_iterations": args.batch_iterations,
        "search_policy_ev": _summary(pair_values),
        "raw_policy_ev_bb_per_hand": -statistics.fmean(pair_values),
        "hand_level_payoff_summary": _summary(payoffs),
        "search_wins": wins,
        "search_losses": losses,
        "ties": ties,
        "search_decisions": total_decisions,
        "terminal_rollouts": total_rollouts,
        "mean_rollouts_per_search_decision": (
            total_rollouts / total_decisions
            if total_decisions
            else 0.0
        ),
        "mean_search_seconds_per_decision": (
            total_search_seconds / total_decisions
            if total_decisions
            else 0.0
        ),
        "wall_seconds": elapsed,
        "zero_sum_verified": True,
        "pairing": (
            f"{args.hands // 2} shared deck/button pairs; "
            "search and raw controllers swap "
            "seats within every pair"
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
