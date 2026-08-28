"""Compatibility facade for the native heads-up Hold'em engine.

The Python engine remains the readable correctness reference. This module
exposes the same public action constants and environment methods while storing
and advancing each state in the isolated C++ extension.
"""

from __future__ import annotations

import copyreg

import heads_up_engine as _reference

try:
    import heads_up_native_engine as _native
except ImportError as exc:  # pragma: no cover - depends on local compilation
    raise ImportError(
        "The heads-up C++ engine is not built. Run engine C HU\\build.bat "
        "on Windows or bash setup_vast.sh in the Vast.ai package."
    ) from exc


_SUPPORTED_NATIVE_ABI_VERSIONS = (5, 6)
_native_contract = {
    "NUM_PLAYERS": (getattr(_native, "NUM_PLAYERS", None), _reference.NUM_PLAYERS),
    "NUM_ACTIONS": (getattr(_native, "NUM_ACTIONS", None), _reference.NUM_ACTIONS),
    "ENGINE_SCHEMA_VERSION": (
        getattr(_native, "ENGINE_SCHEMA_VERSION", None),
        _reference.ENGINE_SCHEMA_VERSION,
    ),
    "ACTION_SCHEMA_VERSION": (
        getattr(_native, "ACTION_SCHEMA_VERSION", None),
        _reference.ACTION_SCHEMA_VERSION,
    ),
    "ENCODER_SCHEMA_VERSION": (
        getattr(_native, "ENCODER_SCHEMA_VERSION", None),
        "hu_information_state_v1",
    ),
    "DEFAULT_MAX_HISTORY": (
        getattr(_native, "DEFAULT_MAX_HISTORY", None),
        32,
    ),
}
_native_mismatches = [
    f"{name}: native={actual!r}, expected={expected!r}"
    for name, (actual, expected) in _native_contract.items()
    if actual != expected
]
if tuple(getattr(_native, "ACTION_NAMES", ())) != tuple(_reference.ACTION_NAMES):
    _native_mismatches.append("ACTION_NAMES/order differs")
if getattr(_native, "NATIVE_ABI_VERSION", None) not in _SUPPORTED_NATIVE_ABI_VERSIONS:
    _native_mismatches.append(
        "NATIVE_ABI_VERSION is unsupported: "
        f"{getattr(_native, 'NATIVE_ABI_VERSION', None)!r}"
    )
if _native_mismatches:
    raise ImportError(
        "heads_up_native_engine is stale or schema-incompatible; rebuild with "
        r"engine C HU\build.bat. "
        + "; ".join(_native_mismatches)
    )


for _name in dir(_reference):
    if _name.isupper():
        globals()[_name] = getattr(_reference, _name)


ActionRecord = _native.ActionRecord
HeadsUpState = _native.HeadsUpState
GameState = HeadsUpState


def _restore_native_state(payload):
    return _native.state_from_dict(payload)


def _reduce_native_state(state):
    return _restore_native_state, (_native.state_to_dict(state),)


copyreg.pickle(HeadsUpState, _reduce_native_state)


def reference_state_to_native(state):
    """Convert one fully determinized Python HU state to packed native form.

    Search roots deliberately hide unknown cards with ``None``.  Such roots
    must be determinized before conversion; accepting placeholders here would
    let an invalid native state reach the C++ hot path.
    """

    zones = [
        *list(state.deck),
        *list(state.board),
        *list(state.burned),
        *list(state.hole[0]),
        *list(state.hole[1]),
    ]
    if any(card is None for card in zones):
        raise ValueError("native search states must be fully determinized")

    last_action_bet = []
    has_last_action_bet = []
    for value in state.last_action_bet:
        has_value = value is not None
        has_last_action_bet.append(has_value)
        last_action_bet.append(0 if value is None else int(value))

    history = []
    for event in state.history:
        action = getattr(event, "action", None)
        history.append(
            (
                int(event.player),
                int(event.street),
                -1 if action is None else int(action),
                str(event.kind),
                int(event.amount),
                int(event.raise_to),
                int(event.contribution_after),
                int(event.current_bet_before),
                int(event.current_bet_after),
                int(event.pot_before),
                int(event.pot_after),
                int(event.to_call_before),
                bool(event.full_raise),
                bool(event.all_in),
            )
        )

    pending_mask = 0
    for player in state.pending_actors:
        pending_mask |= 1 << int(player)
    payoffs = [0, 0] if state.payoffs is None else list(state.payoffs)
    payouts = [0, 0] if state.payouts is None else list(state.payouts)
    return _native.state_from_dict(
        {
            "deck": list(state.deck),
            "board": list(state.board),
            "burned": list(state.burned),
            "hole": [list(state.hole[0]), list(state.hole[1])],
            "stacks": list(state.stacks),
            "initial_stacks": list(state.initial_stacks),
            "total_contrib": list(state.total_contrib),
            "street_contrib": list(state.street_contrib),
            "folded": list(state.folded),
            "all_in": list(state.all_in),
            "raise_rights": list(state.raise_rights),
            "last_action_bet": last_action_bet,
            "has_last_action_bet": has_last_action_bet,
            "uncalled_returns": list(state.uncalled_returns),
            "small_blind": int(state.small_blind),
            "big_blind": int(state.big_blind),
            "pot": int(state.pot),
            "current_bet": int(state.current_bet),
            "min_raise": int(state.min_raise),
            "to_act": -1 if state.to_act is None else int(state.to_act),
            "street": int(state.street),
            "button": int(state.button),
            "sb_player": int(state.sb_player),
            "bb_player": int(state.bb_player),
            "last_full_raiser": (
                -1
                if state.last_full_raiser is None
                else int(state.last_full_raiser)
            ),
            "pending_mask": pending_mask,
            "terminal": bool(state.terminal),
            "has_payoffs": state.payoffs is not None,
            "has_payouts": state.payouts is not None,
            "payoffs": payoffs,
            "payouts": payouts,
            "winners": list(state.winners),
            "history": history,
        }
    )


def evaluate_5card(cards):
    return _native.evaluate_5card(list(cards))


def evaluate_7card(cards_or_hole, board=None):
    cards = list(cards_or_hole)
    if board is not None:
        if len(cards) != 2 or len(board) != 5:
            raise ValueError("hole/board form requires exactly 2 and 5 cards")
        cards.extend(board)
    return _native.evaluate_7card(cards)


def estimate_all_in_ev(
    hero_hole,
    board,
    opponent_holes,
    weights,
    call_probabilities,
    *,
    fold_payoff,
    win_payoff,
    tie_payoff=0.0,
    loss_payoff,
    samples=50_000,
    seed=0,
    robust_best_response=False,
):
    """Native blocker-aware all-in validation over a weighted public range."""

    return _native.estimate_all_in_ev(
        list(hero_hole),
        list(board),
        [list(cards) for cards in opponent_holes],
        list(weights),
        list(call_probabilities),
        float(fold_payoff),
        float(win_payoff),
        float(tie_payoff),
        float(loss_payoff),
        int(samples),
        int(seed),
        bool(robust_best_response),
    )


def bayesian_condition(weights, likelihoods, likelihood_floor=1e-6):
    """Normalize a Bayesian range update in the native hot path."""

    return _native.bayesian_condition(
        list(weights),
        list(likelihoods),
        float(likelihood_floor),
    )


def regret_match_root(regrets, allowed, value_scores):
    """Return a search-owned root strategy; no blueprint fallback is used."""

    return _native.regret_match_root(
        list(regrets),
        [bool(value) for value in allowed],
        list(value_scores),
    )


def hierarchical_regret_match_root(regrets, allowed, value_scores, families):
    """Family-first native regret matching with conditional size selection."""
    return _native.hierarchical_regret_match_root(
        list(regrets),
        [bool(value) for value in allowed],
        list(value_scores),
        [int(value) for value in families],
    )


def estimate_terminal_call_scenarios(
    hero_hole,
    board,
    opponent_holes,
    weights,
    *,
    fold_payoff,
    win_payoff,
    tie_payoff=0.0,
    loss_payoff,
    nominal_samples=50_000,
    seed=0,
):
    """Price a terminal call over robust public-range scenarios."""

    return _native.estimate_terminal_call_scenarios(
        list(hero_hole),
        list(board),
        [list(cards) for cards in opponent_holes],
        list(weights),
        float(fold_payoff),
        float(win_payoff),
        float(tie_payoff),
        float(loss_payoff),
        int(nominal_samples),
        int(seed),
    )


card_to_string = _reference.card_to_string


def encode_information_state_native(
    state,
    hero,
    legal_actions,
    big_blind,
    max_history=32,
    *,
    action_descriptors=None,
):
    """Return the native form of ``heads_up_models``' 1-D observation."""

    return _native.encode_information_state(
        state,
        int(hero),
        list(legal_actions),
        float(big_blind),
        int(max_history),
        action_descriptors=action_descriptors,
    )


def encode_information_states_native(env, states, max_history=32):
    """Encode live native states and legal masks in one C++ boundary call."""

    return env._engine.encode_batch(list(states), int(max_history))


def encode_compact_information_state_native(
    state,
    hero,
    legal_actions,
    big_blind,
    max_history=106,
    *,
    action_descriptors=None,
):
    """Return the native 40 + 7L compact observation in its dense form."""

    if not hasattr(_native, "encode_compact_information_state"):
        raise AttributeError(
            "native compact encoder requires ABI 6; rebuild after restarting "
            "processes that have the old .pyd loaded"
        )
    return _native.encode_compact_information_state(
        state,
        int(hero),
        list(legal_actions),
        float(big_blind),
        int(max_history),
        action_descriptors=action_descriptors,
    )


def encode_compact_information_states_native(env, states, max_history=106):
    if not hasattr(env._engine, "encode_compact_batch"):
        raise AttributeError("native compact batch encoder requires ABI 6")
    return env._engine.encode_compact_batch(list(states), int(max_history))


class HeadsUpHoldemEngine:
    """Python-compatible facade over the integer-chip C++ engine."""

    native_backend = True

    def __init__(
        self,
        starting_stack=None,
        small_blind=None,
        big_blind=None,
        seed=None,
        **legacy_kwargs,
    ):
        if starting_stack is None:
            starting_stack = getattr(_reference, "DEFAULT_STACK", 200)
        if small_blind is None:
            small_blind = getattr(_reference, "DEFAULT_SMALL_BLIND", 1)
        if big_blind is None:
            big_blind = getattr(_reference, "DEFAULT_BIG_BLIND", 2)
        if "stack_size" in legacy_kwargs:
            starting_stack = legacy_kwargs.pop("stack_size")
        if "sb" in legacy_kwargs:
            small_blind = legacy_kwargs.pop("sb")
        if "bb" in legacy_kwargs:
            big_blind = legacy_kwargs.pop("bb")
        if legacy_kwargs:
            unknown = ", ".join(sorted(legacy_kwargs))
            raise TypeError(f"unexpected constructor argument(s): {unknown}")
        for name, value in (
            ("starting_stack", starting_stack),
            ("small_blind", small_blind),
            ("big_blind", big_blind),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer chip amount")
        self._engine = _native.HeadsUpHoldemEngine(
            starting_stack,
            small_blind,
            big_blind,
            seed,
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
        if button is not None and (
            isinstance(button, bool) or button not in (0, 1)
        ):
            raise ValueError("button must be seat 0 or 1")
        if stacks is not None:
            if len(stacks) != NUM_PLAYERS:
                raise ValueError("stacks must contain exactly two values")
            for index, value in enumerate(stacks):
                if isinstance(value, bool) or not isinstance(value, int):
                    raise TypeError(
                        f"stacks[{index}] must be an integer chip amount"
                    )
                if value <= 0:
                    raise ValueError(f"stacks[{index}] must be positive")
        if deck is not None:
            deck = list(deck)
            if len(deck) != 52:
                raise ValueError(f"expected 52 cards, received {len(deck)}")
            if any(
                isinstance(card, bool) or not isinstance(card, int)
                for card in deck
            ):
                raise TypeError("cards must be integer indices")
            if any(card < 0 or card >= 52 for card in deck):
                raise ValueError("card indices must be in the range 0..51")
            if len(set(deck)) != len(deck):
                raise ValueError("duplicate cards are not valid")
        return self._engine.new_hand(button, stacks=stacks, deck=deck)

    @staticmethod
    def clone(state):
        return _native.HeadsUpHoldemEngine.clone(state)

    def amount_to_call(self, state, player=None):
        if player is not None and (
            isinstance(player, bool)
            or not isinstance(player, int)
            or player not in (0, 1)
        ):
            raise ValueError("player must be seat 0 or 1")
        return self._engine.amount_to_call(
            state,
            -1 if player is None else int(player),
        )

    def legal_actions(self, state):
        return self._engine.legal_actions(state)

    def legal_action_mask(self, state):
        return self._engine.legal_action_mask(state)

    def action_target(self, state, action):
        if isinstance(action, bool) or not isinstance(action, int):
            raise TypeError("action must be an integer in 0..9")
        return self._engine.action_target(state, action)

    def action_payment(self, state, action):
        if isinstance(action, bool) or not isinstance(action, int):
            raise TypeError("action must be an integer in 0..9")
        return self._engine.action_payment(state, action)

    def action_descriptors(self, state):
        return self._engine.action_descriptors(state)

    def step(self, state, action):
        if isinstance(action, bool) or not isinstance(action, int):
            raise TypeError(f"action must be an integer in 0..{NUM_ACTIONS - 1}")
        return self._engine.step(state, action)

    def step_batch(self, states, actions):
        if len(states) != len(actions):
            raise ValueError("states and actions must have identical lengths")
        return self._engine.step_batch(
            list(states),
            [int(action) for action in actions],
        )

    def step_exact(self, state, kind, raise_to=None):
        if not isinstance(kind, str):
            raise TypeError("kind must be a string")
        if raise_to is not None and (
            isinstance(raise_to, bool) or not isinstance(raise_to, int)
        ):
            raise TypeError("raise_to must be an integer chip amount")
        return self._engine.step_exact(state, kind, raise_to)

    def resolve_showdown(self, state):
        return self._engine.resolve_showdown(state)

    def terminal_payoff(self, state, player):
        if isinstance(player, bool) or not isinstance(player, int):
            raise ValueError("player must be seat 0 or 1")
        if player not in (0, 1):
            raise ValueError("player must be seat 0 or 1")
        return self._engine.terminal_payoff(state, player)


HeadsUpHoldemEnv = HeadsUpHoldemEngine
HeadsUpPokerEnv = HeadsUpHoldemEngine


__all__ = [name for name in globals() if name.isupper()] + [
    "ActionRecord",
    "HeadsUpState",
    "GameState",
    "HeadsUpHoldemEngine",
    "HeadsUpHoldemEnv",
    "HeadsUpPokerEnv",
    "evaluate_5card",
    "evaluate_7card",
    "estimate_all_in_ev",
    "card_to_string",
    "encode_information_state_native",
    "encode_information_states_native",
    "encode_compact_information_state_native",
    "encode_compact_information_states_native",
    "reference_state_to_native",
]
