"""Persistent-stack tournament orchestration for the Hold'em hand engine.

The core engine deliberately settles one hand at a time.  This module carries
the resulting stacks into the next hand, skips eliminated seats, and supplies
the winner-take-all terminal reward used by tournament evaluation.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from three_player_engine import EPSILON, NUM_ACTIONS, NUM_PLAYERS, ThreePlayerHoldemEnv


Policy = Callable[[ThreePlayerHoldemEnv, Any, int], int | Sequence[float]]


@dataclass(frozen=True)
class TournamentHandResult:
    hand_number: int
    button: int
    starting_stacks: tuple[float, float, float]
    ending_stacks: tuple[float, float, float]
    payoffs: tuple[float, float, float]
    winners: tuple[int, ...]
    actions: int


@dataclass(frozen=True)
class TournamentResult:
    winner: int
    rewards: tuple[float, float, float]
    final_stacks: tuple[float, float, float]
    hands: tuple[TournamentHandResult, ...]
    eliminated_on_hand: tuple[int | None, int | None, int | None]

    @property
    def hands_played(self) -> int:
        return len(self.hands)


def winner_take_all_rewards(winner: int) -> tuple[float, float, float]:
    """Return the zero-sum +2/-1/-1 reward for a three-seat tournament."""

    if isinstance(winner, bool) or winner not in range(NUM_PLAYERS):
        raise ValueError("winner must be seat 0, 1, or 2")
    return tuple(2.0 if seat == winner else -1.0 for seat in range(NUM_PLAYERS))


def _sample_probabilities(
    values: Sequence[float], legal: Sequence[int], rng: random.Random
) -> int:
    if len(values) != NUM_ACTIONS:
        raise ValueError(f"policy must return {NUM_ACTIONS} action probabilities")
    legal_set = set(int(action) for action in legal)
    weights = []
    for action, raw in enumerate(values):
        value = float(raw)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("policy probabilities must be finite and nonnegative")
        weights.append(value if action in legal_set else 0.0)
    total = sum(weights)
    if total <= 0.0:
        raise ValueError("policy assigns no probability to a legal action")
    threshold = rng.random() * total
    cumulative = 0.0
    fallback = int(legal[-1])
    for action in legal:
        cumulative += weights[int(action)]
        fallback = int(action)
        if threshold <= cumulative + 1e-12:
            return int(action)
    return fallback


def play_tournament(
    env: ThreePlayerHoldemEnv,
    policies: Sequence[Policy],
    *,
    rng: random.Random | None = None,
    starting_stacks: Sequence[float] | None = None,
    max_hands: int = 10_000,
    max_actions_per_hand: int = 256,
) -> TournamentResult:
    """Play hands until one stack remains and return tournament-level rewards.

    A policy may return a legal action ID directly or a length-nine probability
    vector.  The latter is sampled with ``rng``.  Constant blinds can make a
    tournament long, so explicit safety limits guard integration tests and
    unattended evaluation from a broken/non-terminating policy.
    """

    if len(policies) != NUM_PLAYERS:
        raise ValueError("policies must contain exactly three seat policies")
    if max_hands <= 0 or max_actions_per_hand <= 0:
        raise ValueError("tournament safety limits must be positive")
    if rng is None:
        rng = random.Random()
    if starting_stacks is None:
        stacks = [float(env.stack_size)] * NUM_PLAYERS
    else:
        if len(starting_stacks) != NUM_PLAYERS:
            raise ValueError("starting_stacks must contain exactly three values")
        stacks = [float(value) for value in starting_stacks]
        if any(not math.isfinite(value) or value < 0.0 for value in stacks):
            raise ValueError("starting stacks must be finite and nonnegative")
    if sum(value > EPSILON for value in stacks) < 2:
        raise ValueError("a tournament must start with at least two live players")

    chip_total = sum(stacks)
    eliminated_on_hand: list[int | None] = [
        0 if stack <= EPSILON else None for stack in stacks
    ]
    records: list[TournamentHandResult] = []

    for hand_number in range(1, max_hands + 1):
        before = tuple(stacks)
        state = env.new_hand(stacks=stacks)
        button = int(state.button)
        actions = 0
        while not state.terminal:
            player = int(state.to_act)
            legal = env.legal_actions(state)
            decision = policies[player](env, state, player)
            if isinstance(decision, bool):
                raise ValueError("a boolean is not a poker action")
            if isinstance(decision, int):
                action = decision
                if action not in legal:
                    raise ValueError(f"seat {player} policy returned illegal action {action}")
            else:
                action = _sample_probabilities(decision, legal, rng)
            state = env.step(state, action)
            actions += 1
            if actions > max_actions_per_hand:
                raise RuntimeError("tournament hand exceeded the action safety limit")

        stacks = [0.0 if value <= EPSILON else float(value) for value in state.stacks]
        if not math.isclose(sum(stacks), chip_total, rel_tol=0.0, abs_tol=1e-7):
            raise RuntimeError("tournament chip conservation failed between hands")
        for seat, (old, new) in enumerate(zip(before, stacks)):
            if old > EPSILON and new <= EPSILON and eliminated_on_hand[seat] is None:
                eliminated_on_hand[seat] = hand_number
        records.append(
            TournamentHandResult(
                hand_number=hand_number,
                button=button,
                starting_stacks=before,
                ending_stacks=tuple(stacks),
                payoffs=tuple(float(value) for value in state.payoffs),
                winners=tuple(state.winners),
                actions=actions,
            )
        )

        survivors = [seat for seat, stack in enumerate(stacks) if stack > EPSILON]
        if len(survivors) == 1:
            winner = survivors[0]
            return TournamentResult(
                winner=winner,
                rewards=winner_take_all_rewards(winner),
                final_stacks=tuple(stacks),
                hands=tuple(records),
                eliminated_on_hand=tuple(eliminated_on_hand),
            )

    raise RuntimeError(f"tournament did not finish within {max_hands} hands")


__all__ = [
    "Policy",
    "TournamentHandResult",
    "TournamentResult",
    "play_tournament",
    "winner_take_all_rewards",
]
