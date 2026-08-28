"""Regression benchmark for the K-heart/6-heart turn decision."""

from __future__ import annotations

from pathlib import Path
import random

from heads_up_engine import HeadsUpHoldemEngine, evaluate_7card
from heads_up_pluribus_search import (
    BlueprintPublicRange,
    MultiprocessPluribusSearch,
    observed_action_likelihoods,
)
from play_heads_up_gui import HeadsUpSnapshotPolicy


def _card(rank: str, suit: str) -> int:
    return "cdhs".index(suit) * 13 + "23456789TJQKA".index(rank)


def main() -> int:
    policy_path = Path(
        "artifacts/heads_up_v4_paper3x/snapshots/policy_00000600.pt"
    )
    human_hole = [_card("6", "s"), _card("2", "c")]
    bot_hole = [_card("K", "h"), _card("6", "h")]
    pop_order = [
        human_hole[0],
        bot_hole[0],
        human_hole[1],
        bot_hole[1],
        _card("A", "c"),
        _card("Q", "h"),
        _card("8", "h"),
        _card("3", "c"),
        _card("A", "d"),
        _card("4", "s"),
    ]
    remaining = [card for card in range(52) if card not in pop_order]
    deck = remaining + list(reversed(pop_order))
    env = HeadsUpHoldemEngine(starting_stack=257, seed=1)
    state = env.new_hand(button=1, stacks=[257, 143], deck=deck)
    policy = HeadsUpSnapshotPolicy(policy_path, mode="sample", seed=41)
    public_range = BlueprintPublicRange()
    public_range.reset(state.hole[1])

    def human_action(kind: str, raise_to: int | None = None) -> None:
        nonlocal state
        public_range.filter_known([*state.hole[1], *state.board])
        matrix = policy.probabilities_for_holes(
            env, state, 0, public_range.combos
        )
        likelihoods = observed_action_likelihoods(
            env,
            state,
            matrix,
            kind=kind,
            raise_to=raise_to,
        )
        public_range.condition(likelihoods)
        state = env.step_exact(state, kind, raise_to)

    state = env.step_exact(state, "call")
    human_action("check")
    human_action("raise_to", 2)
    state = env.step_exact(state, "call")
    human_action("raise_to", 2)
    rng = random.Random(101)
    snapshot = public_range.snapshot()
    wins = ties = 0
    trials = 20_000
    known = set(state.hole[1] + state.board)
    for _ in range(trials):
        index = rng.choices(
            range(len(snapshot.combos)),
            weights=snapshot.weights,
            k=1,
        )[0]
        opponent = snapshot.combos[index]
        river_pool = [
            card
            for card in range(52)
            if card not in known and card not in opponent
        ]
        river = rng.choice(river_pool)
        hero_score = evaluate_7card(state.hole[1], state.board + [river])
        opponent_score = evaluate_7card(opponent, state.board + [river])
        if hero_score > opponent_score:
            wins += 1
        elif hero_score == opponent_score:
            ties += 1
    print(
        f"posterior_checkdown_equity={(wins + 0.5 * ties) / trials:.4%}"
    )
    search = MultiprocessPluribusSearch(
        policy_path,
        workers=5,
        time_budget_seconds=6.0,
        depth_limit=3,
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
    print(
        f"choice={result.choice.label} range_ess="
        f"{result.range_effective_sample_size:.1f} iterations="
        f"{result.cfr_iterations} rollouts={result.terminal_rollouts}"
    )
    for row in sorted(
        result.candidates,
        key=lambda value: value.strategy_probability,
        reverse=True,
    ):
        print(
            f"{row.action.label:20s} strategy={row.strategy_probability:7.2%} "
            f"EV={row.expected_final_payoff_bb:+7.3f} "
            f"CI=[{row.ci95_low_bb:+.3f},{row.ci95_high_bb:+.3f}] "
            f"n={row.samples}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
