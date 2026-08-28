"""Compatibility wrapper for the compiled C++ three-player poker engine.

The native extension owns packed state and betting logic.  This module keeps
constructor aliases and public names compatible with ``three_player_engine`` so
the CFR trainer can switch backends without changing its algorithm.
"""

from __future__ import annotations

import copyreg

from three_player_engine import (
    ACTION_ALL_IN,
    ACTION_CALL,
    ACTION_CHECK,
    ACTION_FOLD,
    ACTION_HALF_POT,
    ACTION_MIN_RAISE,
    ACTION_NAMES,
    ACTION_POT,
    ACTION_RAISE_2X,
    ACTION_RAISE_3X,
    ACTION_RAISE_MIN,
    DEFAULT_BIG_BLIND,
    DEFAULT_SMALL_BLIND,
    DEFAULT_STACK,
    EPSILON,
    NUM_ACTIONS,
    NUM_PLAYERS,
    STREET_FLOP,
    STREET_NAMES,
    STREET_PREFLOP,
    STREET_RIVER,
    STREET_TURN,
    card_to_string,
)

try:
    import poker_native_engine as _native
except ImportError as exc:  # pragma: no cover - depends on local compilation
    raise ImportError(
        "The C++ poker engine is not built. Run: engine C\\build.bat"
    ) from exc


ActionRecord = _native.ActionRecord
SidePot = _native.SidePot
ThreePlayerState = _native.ThreePlayerState
GameState = ThreePlayerState


def _restore_native_state(payload):
    return _native.state_from_dict(payload)


def _reduce_native_state(state):
    return _restore_native_state, (_native.state_to_dict(state),)


copyreg.pickle(ThreePlayerState, _reduce_native_state)


def evaluate_5card(cards):
    return _native.evaluate_5card(list(cards))


def evaluate_7card(cards_or_hole, board=None):
    cards = list(cards_or_hole)
    if board is not None:
        if len(cards) != 2 or len(board) != 5:
            raise ValueError("hole/board form requires exactly 2 and 5 cards")
        cards.extend(board)
    return _native.evaluate_7card(cards)


def calculate_side_pots(contributions, folded):
    return _native.calculate_side_pots(list(contributions), list(folded))


def encode_information_state_native(
    state,
    hero,
    legal_actions,
    stack_size,
    max_history=32,
    *,
    include_tournament_features=False,
    tournament_total_chips=None,
):
    """Encode a packed native state without materialising its Python fields."""

    return _native.encode_information_state(
        state,
        int(hero),
        list(legal_actions),
        float(stack_size),
        int(max_history),
        include_tournament_features=bool(include_tournament_features),
        tournament_total_chips=tournament_total_chips,
    )


def poker_relational_features_native(cards, street_one_hot):
    """Return the 66 relational poker features from CPU NumPy arrays."""

    return _native.poker_relational_features(cards, street_one_hot)


class ThreePlayerHoldemEnv:
    """Python-compatible facade over the packed native environment."""

    native_backend = True

    def __init__(
        self,
        starting_stack=DEFAULT_STACK,
        small_blind=DEFAULT_SMALL_BLIND,
        big_blind=DEFAULT_BIG_BLIND,
        seed=None,
        **legacy_kwargs,
    ):
        if "stack_size" in legacy_kwargs:
            starting_stack = legacy_kwargs.pop("stack_size")
        if "sb" in legacy_kwargs:
            small_blind = legacy_kwargs.pop("sb")
        if "bb" in legacy_kwargs:
            big_blind = legacy_kwargs.pop("bb")
        if legacy_kwargs:
            unknown = ", ".join(sorted(legacy_kwargs))
            raise TypeError(f"unexpected constructor argument(s): {unknown}")
        self._engine = _native.ThreePlayerHoldemEnv(
            float(starting_stack), float(small_blind), float(big_blind), seed
        )

    @property
    def starting_stack(self):
        return self._engine.starting_stack

    @property
    def small_blind(self):
        return self._engine.small_blind

    @property
    def big_blind(self):
        return self._engine.big_blind

    @property
    def stack_size(self):
        return self._engine.stack_size

    @property
    def sb(self):
        return self._engine.sb

    @property
    def bb(self):
        return self._engine.bb

    @property
    def rng(self):
        return self._engine.rng

    @rng.setter
    def rng(self, value):
        self._engine.rng = value

    @property
    def _last_button(self):
        return self._engine._last_button

    @_last_button.setter
    def _last_button(self, value):
        self._engine._last_button = int(value)

    def new_hand(self, button=None, *, stacks=None, deck=None):
        return self._engine.new_hand(button, stacks=stacks, deck=deck)

    @staticmethod
    def clone(state):
        return _native.ThreePlayerHoldemEnv.clone(state)

    def amount_to_call(self, state, player=None):
        return self._engine.amount_to_call(
            state, -1 if player is None else int(player)
        )

    def legal_actions(self, state):
        return self._engine.legal_actions(state)

    def legal_action_mask(self, state):
        return self._engine.legal_action_mask(state)

    def action_target(self, state, action):
        return self._engine.action_target(state, int(action))

    def step(self, state, action):
        return self._engine.step(state, int(action))

    def resolve_showdown(self, state):
        return self._engine.resolve_showdown(state)

    def terminal_payoff(self, state, player):
        return self._engine.terminal_payoff(state, int(player))


ThreePlayerPokerEnv = ThreePlayerHoldemEnv
ThreePlayerHoldemEngine = ThreePlayerHoldemEnv


__all__ = [name for name in globals() if name.isupper()] + [
    "ActionRecord",
    "SidePot",
    "ThreePlayerState",
    "GameState",
    "ThreePlayerHoldemEnv",
    "ThreePlayerPokerEnv",
    "ThreePlayerHoldemEngine",
    "evaluate_5card",
    "evaluate_7card",
    "card_to_string",
    "calculate_side_pots",
    "encode_information_state_native",
    "poker_relational_features_native",
]
