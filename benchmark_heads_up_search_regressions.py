"""Benchmark the known HU search blunders against iteration 600."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmark_heads_up_a9_flop import build_state as build_a9_state
from heads_up_engine import ACTION_NAMES, HeadsUpHoldemEngine
from heads_up_pluribus_search import (
    BlueprintPublicRange,
    MultiprocessPluribusSearch,
    observed_action_likelihoods,
    recommended_search_workers,
)
from play_heads_up_gui import HeadsUpSnapshotPolicy


POLICY_PATH = Path(
    "artifacts/heads_up_v4_paper3x/snapshots/policy_00000600.pt"
)


def _card(text: str) -> int:
    return "cdhs".index(text[1]) * 13 + "23456789TJQKA".index(text[0])


def _deck_for_button_one(
    p0: list[int],
    p1: list[int],
    board: list[int] | None = None,
) -> list[int]:
    board = list(board or [])
    filler = [
        card
        for card in range(52)
        if card not in {*p0, *p1, *board}
    ]
    pops = [p0[0], p1[0], p0[1], p1[1]]
    if board:
        pops.extend([filler.pop(), *board[:3]])
        if len(board) >= 4:
            pops.extend([filler.pop(), board[3]])
        if len(board) >= 5:
            pops.extend([filler.pop(), board[4]])
    remaining = [card for card in filler if card not in pops]
    return remaining + list(reversed(pops))


def build_a7_state(policy):
    p0 = [_card("8h"), _card("6h")]
    p1 = [_card("7s"), _card("Ac")]
    board = [_card("4s"), _card("8d"), _card("7h")]
    env = HeadsUpHoldemEngine(starting_stack=208, seed=2)
    state = env.new_hand(
        button=1,
        stacks=[208, 192],
        deck=_deck_for_button_one(p0, p1, board),
    )
    public_range = BlueprintPublicRange()
    public_range.reset(state.hole[1])

    state = env.step_exact(state, "call")
    public_range.filter_known([*state.hole[1], *state.board])
    matrix = policy.probabilities_for_holes(
        env, state, 0, public_range.combos
    )
    public_range.condition(
        observed_action_likelihoods(
            env, state, matrix, kind="check"
        )
    )
    state = env.step_exact(state, "check")

    public_range.filter_known([*state.hole[1], *state.board])
    matrix = policy.probabilities_for_holes(
        env, state, 0, public_range.combos
    )
    public_range.condition(
        observed_action_likelihoods(
            env, state, matrix, kind="check"
        )
    )
    state = env.step_exact(state, "check")
    return env, state, public_range


def build_aj_state(_policy):
    p0 = [_card("7h"), _card("4h")]
    p1 = [_card("Ad"), _card("Jc")]
    env = HeadsUpHoldemEngine(starting_stack=207, seed=3)
    state = env.new_hand(
        button=1,
        stacks=[193, 207],
        deck=_deck_for_button_one(p0, p1),
    )
    public_range = BlueprintPublicRange()
    public_range.reset(state.hole[1])
    return env, state, public_range


def _build_hand7_state(policy, stage: str):
    """Reproduce a selected Kd3h decision from GUI Hand 7."""

    p0 = [_card("Qs"), _card("Jd")]
    p1 = [_card("3h"), _card("Kd")]
    board = [
        _card("2d"),
        _card("Jc"),
        _card("3s"),
        _card("Ah"),
        _card("5d"),
    ]
    pops = [
        p1[0],
        p0[0],
        p1[1],
        p0[1],
        _card("4c"),
        *board[:3],
        _card("6c"),
        board[3],
        _card("7c"),
        board[4],
    ]
    deck = [
        card for card in range(52) if card not in pops
    ] + list(reversed(pops))
    env = HeadsUpHoldemEngine(starting_stack=210, seed=4)
    state = env.new_hand(
        button=0,
        stacks=[190, 210],
        deck=deck,
    )
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

    human_action("raise_to", 8)
    if stage == "preflop":
        return env, state, public_range
    state = env.step_exact(state, "raise_to", 28)
    human_action("call")
    if stage == "flop":
        return env, state, public_range
    state = env.step_exact(state, "raise_to", 70)
    human_action("raise_to", 162)
    return env, state, public_range


def build_hand7_preflop_state(policy):
    return _build_hand7_state(policy, "preflop")


def build_hand7_flop_state(policy):
    return _build_hand7_state(policy, "flop")


def build_hand7_call_state(policy):
    return _build_hand7_state(policy, "call")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        choices=(
            "a9",
            "a7",
            "aj",
            "hand7_preflop",
            "hand7_flop",
            "hand7",
            "all",
        ),
        default="all",
    )
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    policy = HeadsUpSnapshotPolicy(POLICY_PATH, mode="sample", seed=41)
    builders = {
        "a9": build_a9_state,
        "a7": build_a7_state,
        "aj": build_aj_state,
        "hand7_preflop": build_hand7_preflop_state,
        "hand7_flop": build_hand7_flop_state,
        "hand7": build_hand7_call_state,
    }
    selected = (
        list(builders)
        if args.scenario == "all"
        else [args.scenario]
    )
    workers = args.workers or recommended_search_workers()
    search = MultiprocessPluribusSearch(
        POLICY_PATH,
        workers=workers,
        time_budget_seconds=args.seconds,
        seed=73,
    )
    try:
        for name in selected:
            env, state, public_range = builders[name](policy)
            result = search.resolve(
                env,
                state,
                policy.probabilities(env, state),
                public_range.snapshot(),
            )
            print(
                f"{name}: choice={result.choice.label} "
                f"elapsed={result.elapsed_ms / 1000.0:.3f}s "
                f"iterations={result.cfr_iterations} "
                f"rollouts={result.terminal_rollouts} "
                f"fallback={result.used_blueprint_fallback} "
                f"agreement={result.worker_agreement:.0%} "
                f"gap={result.strategy_gap:.1%}"
            )
            print(f"  reason={result.convergence_reason}")
            for row in sorted(
                result.candidates,
                key=lambda item: item.strategy_probability,
                reverse=True,
            ):
                print(
                    f"  {row.action.label:20s} "
                    f"strategy={row.strategy_probability:7.2%} "
                    f"EV={row.expected_final_payoff_bb:+8.3f} "
                    f"validation={row.validation_ev_bb:+8.3f} "
                    f"pruned={row.safety_pruned}"
                )
    finally:
        search.close(wait_for_workers=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
