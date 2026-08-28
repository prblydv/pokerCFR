"""Benchmark one six-second depth-limited HU CFR decision."""

from __future__ import annotations

import argparse
from pathlib import Path

from heads_up_engine import HeadsUpHoldemEngine
from heads_up_pluribus_search import BlueprintPublicRange, MultiprocessPluribusSearch
from play_heads_up_gui import HeadsUpSnapshotPolicy


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("policy", type=Path)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--python-backend", action="store_true")
    args = parser.parse_args()
    env = HeadsUpHoldemEngine(seed=991)
    state = env.new_hand(button=0)
    hero = int(state.to_act)
    policy = HeadsUpSnapshotPolicy(args.policy, mode="sample", seed=991)
    public_range = BlueprintPublicRange()
    public_range.reset(state.hole[hero])
    search = MultiprocessPluribusSearch(
        args.policy,
        workers=args.workers,
        time_budget_seconds=args.seconds,
        depth_limit=args.depth,
        native_backend=not args.python_backend,
        seed=991,
    )
    try:
        result = search.resolve(
            env,
            state,
            policy.probabilities(env, state),
            public_range.snapshot(),
        )
    finally:
        search.close(wait_for_workers=True)
    print(
        f"choice={result.choice.label!r} elapsed={result.elapsed_ms / 1000:.3f}s "
        f"iterations={result.cfr_iterations} "
        f"continuation_rollouts={result.terminal_rollouts} "
        f"workers={result.workers_responded}/{args.workers} "
        f"range_ess={result.range_effective_sample_size:.1f} "
        f"backend={'native' if result.native_backend else 'python'}"
    )
    for row in sorted(
        result.candidates,
        key=lambda item: item.strategy_probability,
        reverse=True,
    ):
        print(
            f"{row.action.label:20s} strategy={row.strategy_probability:7.2%} "
            f"EV={row.expected_final_payoff_bb:+8.3f} BB "
            f"CI=[{row.ci95_low_bb:+.3f},{row.ci95_high_bb:+.3f}] "
            f"n={row.samples}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
