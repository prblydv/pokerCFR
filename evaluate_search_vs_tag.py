"""Paired fixed-stack evaluation of checkpoint search versus two TAG bots."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from pathlib import Path

import torch

from play_three_player_gui import SnapshotPolicy, sample_action
from real_time_search import RealTimeResolver
from three_player_native import ThreePlayerHoldemEnv
from three_player_production import TightAggressiveOpponent


def _summary(values: list[float]) -> dict[str, float]:
    count = len(values)
    mean = statistics.fmean(values)
    standard_error = (
        statistics.stdev(values) / math.sqrt(count) if count > 1 else 0.0
    )
    return {
        "hands": count,
        "mean_bb_per_hand": mean,
        "standard_error": standard_error,
        "ci95_low": mean - 1.96 * standard_error,
        "ci95_high": mean + 1.96 * standard_error,
        "win_fraction": sum(value > 0.0 for value in values) / count,
    }


def _deals(hands: int, seed: int) -> list[tuple[int, int, list[int]]]:
    rng = random.Random(seed)
    result: list[tuple[int, int, list[int]]] = []
    for index in range(hands):
        deck = list(range(52))
        rng.shuffle(deck)
        # Hero seats and buttons rotate independently, covering all nine
        # seat/position combinations in each consecutive block of nine.
        result.append((index % 3, (index // 3) % 3, deck))
    return result


def _play_mode(
    policy: SnapshotPolicy,
    deals: list[tuple[int, int, list[int]]],
    *,
    mode: str,
    seed: int,
    search_ms: int,
    search_rollouts: int,
    blind_search: bool = False,
) -> tuple[list[float], dict[str, float]]:
    env = ThreePlayerHoldemEnv(
        starting_stack=policy.stack_size,
        small_blind=policy.small_blind,
        big_blind=policy.big_blind,
        seed=seed,
    )
    tag = TightAggressiveOpponent()
    resolvers = {
        hero: RealTimeResolver(
            policy,
            None if blind_search else tag,
            tag_seat=None if blind_search else (hero + 1) % 3,
            scripted_opponents=(
                None
                if blind_search
                else {seat: tag for seat in range(3) if seat != hero}
            ),
            time_budget_ms=search_ms,
            max_rollouts=search_rollouts,
            seed=seed + 100_000 + hero,
        )
        for hero in range(3)
    }
    payoffs: list[float] = []
    search_times: list[float] = []
    search_rollout_counts: list[int] = []
    started = time.perf_counter()

    for index, (hero, button, deck) in enumerate(deals):
        state = env.new_hand(button=button, deck=deck)
        action_rng = random.Random(seed + 1_000_000 + index)
        while not state.terminal:
            actor = int(state.to_act)
            if actor == hero:
                if mode == "search":
                    result = resolvers[hero].resolve(env, state)
                    action = result.action
                    search_times.append(result.elapsed_ms)
                    search_rollout_counts.append(result.rollouts)
                else:
                    action = sample_action(
                        policy.probabilities(env, state), action_rng
                    )
            else:
                action = sample_action(
                    tag.probabilities(env, state, actor), action_rng
                )
            state = env.step(state, action)
        payoffs.append(float(state.payoffs[hero]) / float(env.bb))
        if (index + 1) % 50 == 0 or index + 1 == len(deals):
            elapsed = time.perf_counter() - started
            print(
                f"{mode}: {index + 1}/{len(deals)} hands "
                f"({elapsed:.1f}s)",
                flush=True,
            )

    diagnostics = {
        "elapsed_seconds": time.perf_counter() - started,
        "search_decisions": len(search_times),
        "mean_search_ms": statistics.fmean(search_times) if search_times else 0.0,
        "p95_search_ms": (
            sorted(search_times)[min(len(search_times) - 1, int(0.95 * len(search_times)))]
            if search_times
            else 0.0
        ),
        "mean_rollouts": (
            statistics.fmean(search_rollout_counts)
            if search_rollout_counts
            else 0.0
        ),
    }
    return payoffs, diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/downloaded_blueprints/policy_00008100.pt"),
    )
    parser.add_argument("--hands", type=int, default=300)
    parser.add_argument("--seed", type=int, default=8100)
    parser.add_argument("--search-ms", type=int, default=900)
    parser.add_argument("--search-rollouts", type=int, default=192)
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="CPU inference threads used by this worker (default: 1)",
    )
    parser.add_argument(
        "--blind-search",
        action="store_true",
        help="hide the opponents' scripted profile from search rollouts",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/search_vs_tag_8100.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.hands <= 0:
        raise ValueError("hands must be positive")
    if args.torch_threads <= 0:
        raise ValueError("torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)
    policy = SnapshotPolicy(args.checkpoint.resolve())
    deals = _deals(args.hands, args.seed)
    raw, raw_diagnostics = _play_mode(
        policy,
        deals,
        mode="raw",
        seed=args.seed,
        search_ms=args.search_ms,
        search_rollouts=args.search_rollouts,
        blind_search=args.blind_search,
    )
    search, search_diagnostics = _play_mode(
        policy,
        deals,
        mode="search",
        seed=args.seed,
        search_ms=args.search_ms,
        search_rollouts=args.search_rollouts,
        blind_search=args.blind_search,
    )
    delta = [search_value - raw_value for search_value, raw_value in zip(search, raw)]
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "iteration": policy.iteration,
        "seed": args.seed,
        "search_ms": args.search_ms,
        "search_rollouts": args.search_rollouts,
        "torch_threads": args.torch_threads,
        "blind_search": args.blind_search,
        "raw_policy_vs_two_tag": _summary(raw),
        "search_policy_vs_two_tag": _summary(search),
        "paired_search_minus_raw": _summary(delta),
        "raw_diagnostics": raw_diagnostics,
        "search_diagnostics": search_diagnostics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
