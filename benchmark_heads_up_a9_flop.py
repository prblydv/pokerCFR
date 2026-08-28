"""Reproduce and benchmark the A9s 12-pot flop-shove regression."""

from __future__ import annotations

import argparse
from pathlib import Path

from heads_up_engine import HeadsUpHoldemEngine
from heads_up_pluribus_search import (
    BlueprintPublicRange,
    MultiprocessPluribusSearch,
    observed_action_likelihoods,
    recommended_search_workers,
)
from play_heads_up_gui import HeadsUpSnapshotPolicy


def _card(text: str) -> int:
    return "cdhs".index(text[1]) * 13 + "23456789TJQKA".index(text[0])


def build_state(policy):
    p0 = [_card("Jh"), _card("6h")]
    p1 = [_card("As"), _card("9s")]
    pop_order = [
        p1[0],
        p0[0],
        p1[1],
        p0[1],
        _card("2d"),
        _card("7s"),
        _card("3c"),
        _card("Jc"),
        _card("5d"),
        _card("7c"),
        _card("8d"),
        _card("4c"),
    ]
    deck = [
        card for card in range(52) if card not in pop_order
    ] + list(reversed(pop_order))
    env = HeadsUpHoldemEngine(starting_stack=243, seed=1)
    state = env.new_hand(button=0, stacks=[243, 157], deck=deck)
    public_range = BlueprintPublicRange()
    public_range.reset(state.hole[1])

    def human_action(kind: str, raise_to: int | None = None) -> None:
        nonlocal state
        public_range.filter_known([*state.hole[1], *state.board])
        matrix = policy.probabilities_for_holes(
            env,
            state,
            0,
            public_range.combos,
        )
        public_range.condition(
            observed_action_likelihoods(
                env,
                state,
                matrix,
                kind=kind,
                raise_to=raise_to,
            )
        )
        state = env.step_exact(state, kind, raise_to)

    human_action("call")
    state = env.step_exact(state, "raise_to", 6)
    human_action("call")
    return env, state, public_range


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--python-backend", action="store_true")
    args = parser.parse_args()
    policy_path = Path(
        "artifacts/heads_up_v4_paper3x/snapshots/policy_00000600.pt"
    )
    policy = HeadsUpSnapshotPolicy(policy_path, mode="sample", seed=41)
    env, state, public_range = build_state(policy)
    workers = args.workers or recommended_search_workers()
    search = MultiprocessPluribusSearch(
        policy_path,
        workers=workers,
        time_budget_seconds=args.seconds,
        native_backend=not args.python_backend,
        seed=73,
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
    backend = "native" if result.native_backend else "python"
    print(
        f"backend={backend} workers={workers} choice={result.choice.label} "
        f"elapsed={result.elapsed_ms / 1000.0:.3f}s "
        f"iterations={result.cfr_iterations} "
        f"rollouts={result.terminal_rollouts} "
        f"validation={result.validation_samples} "
        f"converged={result.converged} "
        f"fallback={result.used_blueprint_fallback}"
    )
    print(f"reason={result.convergence_reason}")
    for row in sorted(
        result.candidates,
        key=lambda value: value.strategy_probability,
        reverse=True,
    ):
        print(
            f"{row.action.label:20s} strategy={row.strategy_probability:7.2%} "
            f"CFR={row.expected_final_payoff_bb:+8.3f} "
            f"validation={row.validation_ev_bb:+8.3f} "
            f"CI=[{row.validation_ci95_low_bb:+.3f},"
            f"{row.validation_ci95_high_bb:+.3f}] "
            f"n={row.validation_samples} "
            f"dominated={row.statistically_dominated} "
            f"safety_pruned={row.safety_pruned}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
