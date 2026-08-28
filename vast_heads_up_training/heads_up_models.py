"""Heads-up information-state encoding and compact Deep-CFR networks.

The poker engine owns exact chip arithmetic.  The trained blueprint owns only
ten stable action slots.  This module keeps that boundary explicit:

* history records semantic actions (fold/check/call/bet/raise) and exact
  amounts, never the blueprint slot which happened to produce an action;
* the policy observation contains a fixed ten-bit legal mask;
* each policy slot also carries a descriptor of its exact effect in the
  current state.

Only ``hero`` hole cards are encoded.  The other player's hole cards are never
read, including at terminal states.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from heads_up_engine import (
    ACTION_ALL_IN,
    ACTION_NAMES,
    ACTION_SCHEMA_VERSION,
    ENGINE_SCHEMA_VERSION,
    NUM_ACTIONS,
)


if NUM_ACTIONS != 10:
    raise RuntimeError(
        f"heads-up model schema requires exactly 10 actions, got {NUM_ACTIONS}"
    )

_EXPECTED_ENGINE_SCHEMA_VERSION = "hu_nlhe_engine_v1"
_EXPECTED_ACTION_SCHEMA_VERSION = "hu_nlhe_actions_v1_10"
_EXPECTED_ACTION_NAMES = (
    "fold",
    "check",
    "call",
    "min_raise",
    "third_pot",
    "half_pot",
    "three_quarter_pot",
    "pot",
    "overbet",
    "all_in",
)
if ENGINE_SCHEMA_VERSION != _EXPECTED_ENGINE_SCHEMA_VERSION:
    raise RuntimeError(
        "heads-up encoder was built for engine schema "
        f"{_EXPECTED_ENGINE_SCHEMA_VERSION!r}, got {ENGINE_SCHEMA_VERSION!r}"
    )
if ACTION_SCHEMA_VERSION != _EXPECTED_ACTION_SCHEMA_VERSION:
    raise RuntimeError(
        "heads-up encoder was built for action schema "
        f"{_EXPECTED_ACTION_SCHEMA_VERSION!r}, got {ACTION_SCHEMA_VERSION!r}"
    )
if tuple(ACTION_NAMES) != _EXPECTED_ACTION_NAMES:
    raise RuntimeError("heads-up action names/order do not match the encoder schema")


CARD_FEATURES = 18  # 13 ranks + 4 suits + present bit
CARD_TOKEN_COUNT = 7  # hero's two cards plus five public board slots
DEFAULT_MAX_HISTORY = 32
ENCODER_VERSION = 2
ENCODER_SCHEMA_VERSION = "hu_information_state_v2_recent_history"
SEMANTIC_ACTIONS = ("fold", "check", "call", "bet", "raise")

SEAT_FEATURE_NAMES = (
    "stack_bb",
    "initial_stack_bb",
    "total_contribution_bb",
    "street_contribution_bb",
    "folded",
    "all_in",
    "pending",
    "raise_right",
    "last_action_bet_bb",
    "last_action_bet_present",
)

GLOBAL_FEATURE_NAMES = (
    "pot_bb",
    "current_bet_bb",
    "minimum_raise_increment_bb",
    "minimum_raise_to_bb",
    "hero_to_call_bb",
    "hero_call_payment_bb",
    "hero_maximum_raise_to_bb",
    "pot_after_hero_call_bb",
    "hero_effective_stack_bb",
    "hero_effective_stack_after_call_bb",
    "hero_spr_after_call",
    "hero_pot_odds",
    "current_bet_over_pot",
    "minimum_raise_to_over_pot_after_call",
    "maximum_raise_to_over_pot_after_call",
    "board_progress",
    "active_players_fraction",
    "pending_players_fraction",
    "hero_raise_right",
    "small_blind_over_big_blind",
    "total_initial_chips_bb",
)

# present + street(4) + relative actor(2) + semantic action(5) + flags(2)
# + nine exact/relative amount fields.
HISTORY_FEATURE_NAMES = (
    "present",
    "street_preflop",
    "street_flop",
    "street_turn",
    "street_river",
    "actor_hero",
    "actor_opponent",
    "kind_fold",
    "kind_check",
    "kind_call",
    "kind_bet",
    "kind_raise",
    "all_in",
    "full_raise",
    "amount_added_bb",
    "contribution_after_bb",
    "current_bet_before_bb",
    "current_bet_after_bb",
    "raise_increment_bb",
    "pot_before_bb",
    "pot_after_bb",
    "amount_over_pot_before",
    "target_over_pot_before",
)
HISTORY_FEATURES = len(HISTORY_FEATURE_NAMES)

ACTION_DESCRIPTOR_FEATURE_NAMES = (
    "payment_bb",
    "target_bb",
    "resulting_pot_bb",
    "payment_over_pot_after_call",
    "target_over_pot_after_call",
    "remaining_stack_bb",
    "resulting_spr",
    "is_all_in",
    "is_aggressive",
    "is_full_raise",
    "reopens_betting",
)
ACTION_DESCRIPTOR_FEATURES = len(ACTION_DESCRIPTOR_FEATURE_NAMES)
POKER_RELATIONAL_FEATURES = 66
POLICY_RANGE_AUX_ARCHITECTURE = "hu_deep_cfr_compact_v4_policy_range_v1"
NUM_OPPONENT_COMBOS = 1_326
NETWORK_ARCHITECTURES = (
    "residual_mlp",
    "hu_deep_cfr_compact_v4",
    POLICY_RANGE_AUX_ARCHITECTURE,
)

# Prefix order is deliberately public and stable so a native implementation can
# be differential-tested against this reference encoder.
POSITION_FEATURES = 4 + 2 + 3 + 3 + 3
PUBLIC_PREFIX_FEATURES = (
    POSITION_FEATURES + 2 * len(SEAT_FEATURE_NAMES) + len(GLOBAL_FEATURE_NAMES)
)
CARD_STATE_PREFIX_FEATURES = PUBLIC_PREFIX_FEATURES
CARD_STATE_FEATURES = CARD_TOKEN_COUNT * CARD_FEATURES
HISTORY_OFFSET = CARD_STATE_PREFIX_FEATURES + CARD_STATE_FEATURES


def information_state_size(max_history: int = DEFAULT_MAX_HISTORY) -> int:
    """Return the exact fixed observation width."""

    if isinstance(max_history, bool) or int(max_history) <= 0:
        raise ValueError("max_history must be a positive integer")
    return (
        PUBLIC_PREFIX_FEATURES
        + CARD_STATE_FEATURES
        + int(max_history) * HISTORY_FEATURES
        + NUM_ACTIONS
        + NUM_ACTIONS * ACTION_DESCRIPTOR_FEATURES
    )


def legal_mask_offset(max_history: int = DEFAULT_MAX_HISTORY) -> int:
    return HISTORY_OFFSET + int(max_history) * HISTORY_FEATURES


def action_descriptor_offset(max_history: int = DEFAULT_MAX_HISTORY) -> int:
    return legal_mask_offset(max_history) + NUM_ACTIONS


def encoder_metadata(max_history: int = DEFAULT_MAX_HISTORY) -> dict[str, object]:
    """Return the locked checkpoint/replay compatibility contract."""

    width = information_state_size(max_history)
    max_history = int(max_history)
    return {
        "engine_schema_version": ENGINE_SCHEMA_VERSION,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "encoder_version": ENCODER_VERSION,
        "encoder_schema_version": ENCODER_SCHEMA_VERSION,
        "width": width,
        "input_dim": width,
        "max_history": max_history,
        "history_policy": "most_recent",
        "num_actions": NUM_ACTIONS,
        "action_names": tuple(ACTION_NAMES),
        "history_feature_names": HISTORY_FEATURE_NAMES,
        "action_descriptor_feature_names": ACTION_DESCRIPTOR_FEATURE_NAMES,
    }


@dataclass(frozen=True)
class ActionDescriptor:
    """Exact effect of one legal blueprint action in raw engine chips."""

    action: int
    target: float
    payment: float
    resulting_pot: float
    remaining_stack: float
    resulting_effective_stack: float
    is_all_in: bool
    is_aggressive: bool
    is_full_raise: bool
    reopens_betting: bool


def _sequence(state, name: str, fallback: Sequence[object]) -> list[object]:
    value = getattr(state, name, fallback)
    return list(value)


def _pending_actors(state) -> set[int]:
    return set(getattr(state, "pending_actors", getattr(state, "pending", ())))


def _minimum_raise_increment(state) -> float:
    return float(
        getattr(
            state,
            "min_raise",
            getattr(state, "minimum_raise", getattr(state, "min_raise_increment", 0.0)),
        )
    )


def _small_blind(state, big_blind: float) -> float:
    return float(
        getattr(
            state,
            "small_blind",
            getattr(state, "sb", 0.5 * float(big_blind)),
        )
    )


def _last_full_raiser(state) -> int | None:
    value = getattr(state, "last_full_raiser", None)
    return None if value is None else int(value)


def _card_features(card: int | None) -> list[float]:
    values = [0.0] * CARD_FEATURES
    if card is None or int(card) < 0:
        return values
    if isinstance(card, bool) or not isinstance(card, int) or card >= 52:
        raise ValueError(f"card index must be in [0, 51], got {card!r}")
    values[card % 13] = 1.0
    values[13 + card // 13] = 1.0
    values[17] = 1.0
    return values


def _event_value(event, names: Sequence[str], default: object = None) -> object:
    if isinstance(event, Mapping):
        for name in names:
            if name in event:
                return event[name]
    else:
        for name in names:
            if hasattr(event, name):
                return getattr(event, name)
    return default


def _semantic_event(event) -> tuple[int, int, str, bool, bool, list[float]]:
    """Return public event data without retaining a blueprint bucket ID."""

    if isinstance(event, (tuple, list)):
        if len(event) < 4:
            raise ValueError("history tuples need street, player, action, amount")
        street = int(event[0])
        player = int(event[1])
        raw_kind: object = event[2]
        amount = float(event[3])
        contribution_after = float(event[4]) if len(event) > 4 else amount
        current_before = float(event[5]) if len(event) > 5 else 0.0
        current_after = float(event[6]) if len(event) > 6 else contribution_after
        pot_after = float(event[7]) if len(event) > 7 else amount
        all_in = bool(event[8]) if len(event) > 8 else False
        full_raise = bool(event[9]) if len(event) > 9 else False
    else:
        street = int(_event_value(event, ("street",), 0))
        player = int(_event_value(event, ("player", "actor"), 0))
        raw_kind = _event_value(
            event,
            ("kind", "semantic_action", "action_name", "name", "action"),
            "",
        )
        amount = float(_event_value(event, ("amount", "amount_added", "payment"), 0.0))
        contribution_after = float(
            _event_value(
                event,
                ("contribution_after", "target", "raise_to"),
                amount,
            )
        )
        current_before = float(
            _event_value(event, ("current_bet_before",), 0.0)
        )
        current_after = float(
            _event_value(
                event,
                ("current_bet_after",),
                max(current_before, contribution_after),
            )
        )
        pot_after = float(
            _event_value(event, ("pot_after",), amount)
        )
        all_in = bool(_event_value(event, ("all_in", "is_all_in"), False))
        full_raise = bool(
            _event_value(event, ("full_raise", "is_full_raise"), False)
        )

    raw_name = str(raw_kind).strip().lower().replace("-", "_").replace(" ", "_")
    if raw_name in ("fold", "check", "call"):
        kind = raw_name
    elif raw_name in ("bet", "raise"):
        kind = raw_name
    elif current_after > current_before + 1e-9:
        kind = "bet" if current_before <= 1e-9 else "raise"
    elif amount > 1e-9:
        kind = "call"
    else:
        kind = "check"

    pot_before = max(0.0, pot_after - amount)
    raise_increment = max(0.0, current_after - current_before)
    numeric = [
        amount,
        contribution_after,
        current_before,
        current_after,
        raise_increment,
        pot_before,
        pot_after,
        amount / pot_before if pot_before > 1e-9 else 0.0,
        contribution_after / pot_before if pot_before > 1e-9 else 0.0,
    ]
    return street, player, kind, all_in, full_raise, numeric


def build_action_descriptors(
    env, state
) -> list[ActionDescriptor | Mapping[str, object] | None]:
    """Build exact per-slot effects using the engine's canonical targets.

    Callers should use this helper instead of reimplementing sizing formulas.
    Engines may expose their own ``action_descriptors`` fast path.  Otherwise
    this consumes ``legal_actions`` and ``action_target`` from the same engine
    instance that will execute ``step``, preventing mask/execution drift.
    """

    engine_descriptors = getattr(env, "action_descriptors", None)
    if callable(engine_descriptors):
        values = list(engine_descriptors(state))
        if len(values) != NUM_ACTIONS:
            raise RuntimeError(
                f"engine action_descriptors returned {len(values)} entries; "
                f"expected {NUM_ACTIONS}"
            )
        return values

    descriptors: list[ActionDescriptor | None] = [None] * NUM_ACTIONS
    if bool(getattr(state, "terminal", False)):
        return descriptors
    actor = getattr(state, "to_act", None)
    if actor is None:
        return descriptors
    actor = int(actor)
    legal = [int(action) for action in env.legal_actions(state)]
    street_contrib = [float(value) for value in state.street_contrib]
    stacks = [float(value) for value in state.stacks]
    opponent = 1 - actor
    current_bet = float(state.current_bet)
    minimum_raise = _minimum_raise_increment(state)
    pot = float(state.pot)
    for action in legal:
        target = float(env.action_target(state, action))
        payment = max(0.0, target - street_contrib[actor])
        remaining = max(0.0, stacks[actor] - payment)
        aggressive = target > current_bet + 1e-9
        full_raise = aggressive and target - current_bet + 1e-9 >= minimum_raise
        reopens = (
            full_raise
            and remaining > 1e-9
            and stacks[opponent] > 1e-9
        )
        descriptors[action] = ActionDescriptor(
            action=action,
            target=target,
            payment=payment,
            resulting_pot=pot + payment,
            remaining_stack=remaining,
            resulting_effective_stack=min(remaining, stacks[opponent]),
            is_all_in=action == ACTION_ALL_IN or remaining <= 1e-9,
            is_aggressive=aggressive,
            is_full_raise=full_raise,
            reopens_betting=reopens,
        )
    return descriptors


def _descriptor_by_action(
    descriptors: Sequence[ActionDescriptor | None]
    | Mapping[int, ActionDescriptor | Mapping[str, object] | None]
    | None,
) -> list[ActionDescriptor | Mapping[str, object] | None]:
    if descriptors is None:
        return [None] * NUM_ACTIONS
    if isinstance(descriptors, Mapping):
        return [descriptors.get(action) for action in range(NUM_ACTIONS)]
    values = list(descriptors)
    if len(values) != NUM_ACTIONS:
        raise ValueError(f"action_descriptors must contain {NUM_ACTIONS} entries")
    return values


def _descriptor_field(
    descriptor: ActionDescriptor | Mapping[str, object], name: str
) -> object:
    if isinstance(descriptor, Mapping):
        if name not in descriptor:
            raise ValueError(f"action descriptor is missing {name!r}")
        return descriptor[name]
    return getattr(descriptor, name)


def encode_information_state(
    state,
    hero: int,
    legal_actions: Iterable[int],
    big_blind: float,
    max_history: int = DEFAULT_MAX_HISTORY,
    *,
    action_descriptors: Sequence[ActionDescriptor | None]
    | Mapping[int, ActionDescriptor | Mapping[str, object] | None]
    | None = None,
) -> torch.Tensor:
    """Encode one hero-visible, fixed-width heads-up information state.

    Monetary features use big blinds or dimensionless pot/stack ratios.  If
    every chip quantity and both blinds are multiplied by the same positive
    factor, the resulting tensor is unchanged.
    """

    if hero not in (0, 1):
        raise ValueError(f"hero must be seat 0 or 1, got {hero!r}")
    if isinstance(big_blind, bool) or float(big_blind) <= 0.0:
        raise ValueError("big_blind must be positive")
    if isinstance(max_history, bool) or int(max_history) <= 0:
        raise ValueError("max_history must be a positive integer")
    bb = float(big_blind)
    state_big_blind = getattr(state, "big_blind", getattr(state, "bb", None))
    if (
        state_big_blind is not None
        and abs(float(state_big_blind) - bb) > 1e-9
    ):
        raise ValueError(
            f"big_blind={bb:g} does not match state.big_blind="
            f"{float(state_big_blind):g}"
        )
    max_history = int(max_history)
    opponent = 1 - hero
    legal_values = tuple(int(action) for action in legal_actions)
    legal_set = set(legal_values)
    if any(action < 0 or action >= NUM_ACTIONS for action in legal_set):
        raise ValueError(f"legal action IDs must be in 0..{NUM_ACTIONS - 1}")
    if legal_set and action_descriptors is None:
        raise ValueError(
            "live decisions require exact action_descriptors; use "
            "build_action_descriptors(env, state)"
        )
    state_actor = getattr(state, "to_act", None)
    if legal_set and (state_actor is None or int(state_actor) != hero):
        raise ValueError(
            "live decision encoding requires hero == state.to_act so legal "
            "actions and exact descriptors belong to the encoded player"
        )

    full_history = list(state.history)

    if type(state).__module__ == "heads_up_native_engine":
        try:
            from heads_up_native import encode_information_state_native

            encoded = encode_information_state_native(
                state,
                hero,
                legal_values,
                bb,
                max_history,
                action_descriptors=action_descriptors,
            )
            return torch.from_numpy(encoded)
        except (ImportError, AttributeError):
            # Source remains usable before the optional extension is rebuilt.
            pass

    stacks = [float(value) for value in state.stacks]
    if len(stacks) != 2:
        raise ValueError("state.stacks must contain exactly two values")
    total_contrib = [float(value) for value in state.total_contrib]
    street_contrib = [float(value) for value in state.street_contrib]
    folded = [bool(value) for value in state.folded]
    all_in = [bool(value) for value in state.all_in]
    for name, values in (
        ("total_contrib", total_contrib),
        ("street_contrib", street_contrib),
        ("folded", folded),
        ("all_in", all_in),
    ):
        if len(values) != 2:
            raise ValueError(f"state.{name} must contain exactly two values")

    initial = [
        float(value)
        for value in getattr(
            state,
            "initial_stacks",
            [stacks[seat] + total_contrib[seat] for seat in range(2)],
        )
    ]
    if len(initial) != 2:
        raise ValueError("state.initial_stacks must contain exactly two values")
    pending = _pending_actors(state)
    raise_rights = [
        bool(value)
        for value in getattr(state, "raise_rights", [True, True])
    ]
    last_action_bet = list(getattr(state, "last_action_bet", [None, None]))
    if len(raise_rights) != 2 or len(last_action_bet) != 2:
        raise ValueError("heads-up raise metadata must contain exactly two values")

    x: list[float] = []

    street = int(state.street)
    if street not in range(4):
        raise ValueError(f"street must be in 0..3, got {street}")
    x.extend(float(street == value) for value in range(4))

    button = int(state.button)
    if button not in (0, 1):
        raise ValueError("state.button must be seat 0 or 1")
    relative_button = (button - hero) % 2
    x.extend(float(relative_button == value) for value in range(2))
    x.extend(
        (
            float(hero == button),
            float(hero == int(state.sb_player)),
            float(hero == int(state.bb_player)),
        )
    )

    actor = state_actor
    x.extend(
        (
            float(actor is None),
            float(actor is not None and int(actor) == hero),
            float(actor is not None and int(actor) == opponent),
        )
    )
    last_raiser = _last_full_raiser(state)
    x.extend(
        (
            float(last_raiser is None),
            float(last_raiser == hero),
            float(last_raiser == opponent),
        )
    )

    for seat in (hero, opponent):
        prior = last_action_bet[seat]
        x.extend(
            (
                stacks[seat] / bb,
                initial[seat] / bb,
                total_contrib[seat] / bb,
                street_contrib[seat] / bb,
                float(folded[seat]),
                float(all_in[seat]),
                float(seat in pending),
                float(raise_rights[seat]),
                float(prior) / bb if prior is not None else 0.0,
                float(prior is not None),
            )
        )

    pot = float(state.pot)
    current_bet = float(state.current_bet)
    minimum_raise = _minimum_raise_increment(state)
    minimum_raise_to = current_bet + minimum_raise
    to_call = max(0.0, current_bet - street_contrib[hero])
    call_payment = min(stacks[hero], to_call)
    maximum_raise_to = street_contrib[hero] + stacks[hero]
    pot_after_call = pot + call_payment
    effective_stack = min(stacks[hero], stacks[opponent])
    hero_after_call = max(0.0, stacks[hero] - call_payment)
    effective_after_call = min(hero_after_call, stacks[opponent])
    small_blind = _small_blind(state, bb)
    active_count = sum(not value for value in folded)
    x.extend(
        (
            pot / bb,
            current_bet / bb,
            minimum_raise / bb,
            minimum_raise_to / bb,
            to_call / bb,
            call_payment / bb,
            maximum_raise_to / bb,
            pot_after_call / bb,
            effective_stack / bb,
            effective_after_call / bb,
            effective_after_call / pot_after_call if pot_after_call > 1e-9 else 0.0,
            call_payment / pot_after_call if pot_after_call > 1e-9 else 0.0,
            current_bet / pot if pot > 1e-9 else 0.0,
            minimum_raise_to / pot_after_call if pot_after_call > 1e-9 else 0.0,
            maximum_raise_to / pot_after_call if pot_after_call > 1e-9 else 0.0,
            len(state.board) / 5.0,
            active_count / 2.0,
            len(pending) / 2.0,
            float(raise_rights[hero]),
            small_blind / bb,
            sum(initial) / bb,
        )
    )

    hole: Sequence[Sequence[int]] = state.hole
    if len(hole) != 2 or len(hole[hero]) != 2:
        raise ValueError("state.hole must contain two cards for each seat")
    for card in sorted(int(value) for value in hole[hero]):
        x.extend(_card_features(card))
    board = [int(value) for value in state.board]
    if len(board) > 5:
        raise ValueError("board cannot contain more than five cards")
    canonical_board = sorted(board[:3]) + board[3:]
    for index in range(5):
        x.extend(
            _card_features(
                canonical_board[index] if index < len(canonical_board) else None
            )
        )

    history = full_history[-max_history:]
    x.extend([0.0] * ((max_history - len(history)) * HISTORY_FEATURES))
    for event in history:
        (
            event_street,
            event_player,
            kind,
            event_all_in,
            full_raise,
            numeric,
        ) = _semantic_event(event)
        if event_street not in range(4):
            raise ValueError(f"history street must be in 0..3, got {event_street}")
        if event_player not in (0, 1):
            raise ValueError(f"history player must be 0 or 1, got {event_player}")
        x.append(1.0)
        x.extend(float(event_street == value) for value in range(4))
        x.extend((float(event_player == hero), float(event_player == opponent)))
        x.extend(float(kind == value) for value in SEMANTIC_ACTIONS)
        x.extend((float(event_all_in), float(full_raise)))
        # The first seven quantities are raw chips; the final two are already
        # dimensionless ratios.
        x.extend(float(value) / bb for value in numeric[:7])
        x.extend(float(value) for value in numeric[7:])

    x.extend(float(action in legal_set) for action in range(NUM_ACTIONS))

    descriptor_values = _descriptor_by_action(action_descriptors)
    if action_descriptors is not None:
        missing = [
            action
            for action in legal_set
            if descriptor_values[action] is None
        ]
        if missing:
            raise ValueError(f"missing descriptors for legal actions: {missing}")
        extras = [
            action
            for action, descriptor in enumerate(descriptor_values)
            if descriptor is not None and action not in legal_set
        ]
        if extras:
            raise ValueError(f"descriptors supplied for illegal actions: {extras}")

    for action, descriptor in enumerate(descriptor_values):
        if descriptor is None:
            x.extend([0.0] * ACTION_DESCRIPTOR_FEATURES)
            continue
        target = float(_descriptor_field(descriptor, "target"))
        payment = float(_descriptor_field(descriptor, "payment"))
        resulting_pot = float(_descriptor_field(descriptor, "resulting_pot"))
        remaining = float(_descriptor_field(descriptor, "remaining_stack"))
        resulting_effective = float(
            _descriptor_field(descriptor, "resulting_effective_stack")
        )
        x.extend(
            (
                payment / bb,
                target / bb,
                resulting_pot / bb,
                payment / pot_after_call if pot_after_call > 1e-9 else 0.0,
                target / pot_after_call if pot_after_call > 1e-9 else 0.0,
                remaining / bb,
                (
                    resulting_effective / resulting_pot
                    if resulting_pot > 1e-9
                    else 0.0
                ),
                float(bool(_descriptor_field(descriptor, "is_all_in"))),
                float(bool(_descriptor_field(descriptor, "is_aggressive"))),
                float(bool(_descriptor_field(descriptor, "is_full_raise"))),
                float(bool(_descriptor_field(descriptor, "reopens_betting"))),
            )
        )

    expected = information_state_size(max_history)
    if len(x) != expected:
        raise RuntimeError(f"encoder produced {len(x)} values; expected {expected}")
    return torch.tensor(x, dtype=torch.float32)


class ResidualBlock(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.fc1(F.silu(self.norm(x)))
        x = self.fc2(F.silu(x))
        return residual + x


class _ResidualNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden: int = 128,
        blocks: int = 2,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden <= 0 or blocks < 0:
            raise ValueError("network dimensions must be positive")
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_layer = nn.Linear(input_dim, hidden)
        self.blocks = nn.ModuleList(ResidualBlock(hidden) for _ in range(blocks))
        self.output_norm = nn.LayerNorm(hidden)
        self.output_layer = nn.Linear(hidden, NUM_ACTIONS)
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.input_layer(self.input_norm(x)))
        for block in self.blocks:
            x = block(x)
        return self.output_layer(self.output_norm(x))


class AdvantageNetwork(_ResidualNetwork):
    """Outputs one unconstrained counterfactual-regret value per action slot."""


class PolicyNetwork(_ResidualNetwork):
    """Outputs ten logits; callers apply the exact legal mask separately."""


def _straight_window_counts(rank_present: torch.Tensor) -> torch.Tensor:
    windows = torch.tensor(
        (
            (12, 0, 1, 2, 3),
            (0, 1, 2, 3, 4),
            (1, 2, 3, 4, 5),
            (2, 3, 4, 5, 6),
            (3, 4, 5, 6, 7),
            (4, 5, 6, 7, 8),
            (5, 6, 7, 8, 9),
            (6, 7, 8, 9, 10),
            (7, 8, 9, 10, 11),
            (8, 9, 10, 11, 12),
        ),
        dtype=torch.long,
        device=rank_present.device,
    )
    return rank_present[:, windows].sum(dim=2)


def poker_relational_features(
    cards: torch.Tensor, street_one_hot: torch.Tensor
) -> torch.Tensor:
    """Derive deterministic made-hand, draw, and board-texture features."""

    if cards.ndim != 3 or cards.shape[1:] != (CARD_TOKEN_COUNT, CARD_FEATURES):
        raise ValueError("cards must have shape [batch, 7, 18]")
    present = cards[:, :, 17]
    ranks = cards[:, :, :13] * present.unsqueeze(2)
    suits = cards[:, :, 13:17] * present.unsqueeze(2)
    hole_ranks = ranks[:, :2]
    hole_suits = suits[:, :2]
    board_ranks = ranks[:, 2:]
    board_suits = suits[:, 2:]

    rank_counts = ranks.sum(dim=1)
    suit_counts = suits.sum(dim=1)
    board_rank_counts = board_ranks.sum(dim=1)
    board_suit_counts = board_suits.sum(dim=1)
    rank_present = (rank_counts > 0).to(cards.dtype)
    straight_counts = _straight_window_counts(rank_present)
    has_straight = (straight_counts >= 5).any(dim=1)
    has_flush = (suit_counts >= 5).any(dim=1)

    straight_flush = torch.zeros_like(has_straight)
    for suit in range(4):
        suited_ranks = (
            ranks * suits[:, :, suit : suit + 1]
        ).sum(dim=1).clamp(max=1.0)
        straight_flush |= (_straight_window_counts(suited_ranks) >= 5).any(dim=1)

    pair_count = (rank_counts >= 2).sum(dim=1)
    trip_count = (rank_counts >= 3).sum(dim=1)
    has_quads = (rank_counts >= 4).any(dim=1)
    has_full_house = (trip_count >= 2) | ((trip_count >= 1) & (pair_count >= 2))
    has_trips = trip_count >= 1
    has_two_pair = pair_count >= 2
    has_pair = pair_count >= 1
    category = torch.stack(
        (
            ~(
                straight_flush
                | has_quads
                | has_full_house
                | has_flush
                | has_straight
                | has_trips
                | has_two_pair
                | has_pair
            ),
            has_pair & ~has_two_pair & ~has_trips,
            has_two_pair & ~has_trips,
            has_trips & ~has_full_house & ~has_straight & ~has_flush,
            has_straight & ~has_flush & ~has_full_house & ~has_quads,
            has_flush & ~has_full_house & ~has_quads & ~straight_flush,
            has_full_house & ~has_quads,
            has_quads & ~straight_flush,
            straight_flush,
        ),
        dim=1,
    ).to(cards.dtype)

    street_index = street_one_hot.argmax(dim=1)
    can_draw = street_index < 3
    straight_four = straight_counts == 4
    has_straight_draw = straight_four.any(dim=1) & can_draw & ~has_straight
    window_presence = rank_present[:, torch.tensor(
        (
            (12, 0, 1, 2, 3),
            (0, 1, 2, 3, 4),
            (1, 2, 3, 4, 5),
            (2, 3, 4, 5, 6),
            (3, 4, 5, 6, 7),
            (4, 5, 6, 7, 8),
            (5, 6, 7, 8, 9),
            (6, 7, 8, 9, 10),
            (7, 8, 9, 10, 11),
            (8, 9, 10, 11, 12),
        ),
        dtype=torch.long,
        device=cards.device,
    )]
    missing = 1.0 - window_presence
    open_ended = (
        straight_four
        & ((missing[:, :, 0] > 0.5) | (missing[:, :, 4] > 0.5))
    ).any(dim=1) & can_draw & ~has_straight
    gutshot = (
        straight_four & (missing[:, :, 1:4].sum(dim=2) > 0.5)
    ).any(dim=1) & can_draw & ~has_straight
    flush_draw = (suit_counts == 4).any(dim=1) & can_draw & ~has_flush
    backdoor_flush = (
        (street_index == 1) & (suit_counts == 3).any(dim=1) & ~has_flush
    )

    hole_rank_index = hole_ranks.argmax(dim=2)
    hole_suit_index = hole_suits.argmax(dim=2)
    pocket_pair = hole_rank_index[:, 0] == hole_rank_index[:, 1]
    hole_suited = hole_suit_index[:, 0] == hole_suit_index[:, 1]
    raw_gap = (hole_rank_index[:, 0] - hole_rank_index[:, 1]).abs()
    ace_low_gap = torch.where(
        (hole_rank_index == 12).any(dim=1),
        torch.minimum(raw_gap, 13 - raw_gap),
        raw_gap,
    )
    gap_bucket = F.one_hot(ace_low_gap.clamp(max=4), num_classes=5).to(cards.dtype)
    board_present = board_rank_counts > 0
    hole_board_matches = board_present.gather(1, hole_rank_index).sum(dim=1)
    board_card_count = present[:, 2:].sum(dim=1)
    board_max_rank = torch.where(
        board_present,
        torch.arange(13, device=cards.device).unsqueeze(0),
        torch.full_like(board_rank_counts, -1.0),
    ).amax(dim=1)
    overcards = (hole_rank_index > board_max_rank.unsqueeze(1)).sum(dim=1)
    board_pair_count = (board_rank_counts >= 2).sum(dim=1)
    board_trip_count = (board_rank_counts >= 3).sum(dim=1)
    max_board_suit = board_suit_counts.amax(dim=1)

    scalar = torch.stack(
        tuple(
            value.to(cards.dtype)
            for value in (
                pocket_pair,
                hole_suited,
                hole_board_matches / 2.0,
                overcards / 2.0,
                has_straight_draw,
                open_ended,
                gutshot,
                flush_draw,
                backdoor_flush,
                board_pair_count > 0,
                board_trip_count > 0,
                (board_card_count >= 3) & (max_board_suit == board_card_count),
                (board_card_count >= 3) & (max_board_suit == 2),
                board_card_count / 5.0,
                pair_count.to(cards.dtype) / 3.0,
                trip_count.to(cards.dtype) / 2.0,
                has_flush,
                has_straight,
            )
        ),
        dim=1,
    )
    features = torch.cat(
        (
            rank_counts / 4.0,
            suit_counts / 7.0,
            board_rank_counts / 3.0,
            board_suit_counts / 5.0,
            category,
            gap_bucket,
            scalar,
        ),
        dim=1,
    )
    if int(features.shape[1]) != POKER_RELATIONAL_FEATURES:
        raise RuntimeError("unexpected poker relational feature width")
    return features


def _structured_history_steps(input_dim: int) -> int:
    fixed = (
        HISTORY_OFFSET
        + NUM_ACTIONS
        + NUM_ACTIONS * ACTION_DESCRIPTOR_FEATURES
    )
    history_values = int(input_dim) - fixed
    if history_values <= 0 or history_values % HISTORY_FEATURES:
        raise ValueError(
            f"input width {input_dim} is not a supported HU information state"
        )
    return history_values // HISTORY_FEATURES


class HeadsUpDeepCFRCompactV4Backbone(nn.Module):
    """Compact structured HU model retaining the complete information state."""

    def __init__(self, input_dim: int, hidden: int = 128, blocks: int = 2):
        super().__init__()
        if hidden <= 0 or blocks < 0:
            raise ValueError("hidden must be positive and blocks cannot be negative")
        self.input_dim = int(input_dim)
        self.hidden = int(hidden)
        self.history_steps = _structured_history_steps(input_dim)
        self.history_token_dim = 32
        self.action_token_dim = 32
        embedding_dim = 32
        card_hidden = 128

        self.rank_embedding = nn.Embedding(13, embedding_dim)
        self.suit_embedding = nn.Embedding(4, embedding_dim)
        self.exact_card_embedding = nn.Embedding(52, embedding_dim)
        self.card_fc1 = nn.Linear(4 * embedding_dim, card_hidden)
        self.card_fc2 = nn.Linear(card_hidden, card_hidden)
        self.card_fc3 = nn.Linear(card_hidden, hidden)

        self.poker_feature_norm = nn.LayerNorm(POKER_RELATIONAL_FEATURES)
        self.poker_feature_fc = nn.Linear(POKER_RELATIONAL_FEATURES, hidden)

        public_dim = CARD_STATE_PREFIX_FEATURES + NUM_ACTIONS
        self.public_norm = nn.LayerNorm(public_dim)
        self.public_fc = nn.Linear(public_dim, hidden)
        self.public_residual = ResidualBlock(hidden)

        self.action_projection = nn.Linear(
            ACTION_DESCRIPTOR_FEATURES, self.action_token_dim
        )
        self.action_positions = nn.Parameter(
            torch.zeros(1, NUM_ACTIONS, self.action_token_dim)
        )
        action_layer = nn.TransformerEncoderLayer(
            d_model=self.action_token_dim,
            nhead=4,
            dim_feedforward=2 * self.action_token_dim,
            dropout=0.05,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.action_attention = nn.TransformerEncoder(
            action_layer, num_layers=1, enable_nested_tensor=False
        )
        self.action_fusion = nn.Linear(self.action_token_dim, hidden)

        self.history_projection = nn.Linear(
            HISTORY_FEATURES, self.history_token_dim
        )
        self.history_positions = nn.Parameter(
            torch.zeros(1, self.history_steps + 1, self.history_token_dim)
        )
        self.history_summary_token = nn.Parameter(
            torch.zeros(1, 1, self.history_token_dim)
        )
        history_layer = nn.TransformerEncoderLayer(
            d_model=self.history_token_dim,
            nhead=4,
            dim_feedforward=2 * self.history_token_dim,
            dropout=0.05,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.history_attention = nn.TransformerEncoder(
            history_layer, num_layers=1, enable_nested_tensor=False
        )
        self.history_memory = nn.GRU(
            self.history_token_dim,
            self.history_token_dim,
            num_layers=1,
            batch_first=True,
        )
        self.history_fusion = nn.Linear(2 * self.history_token_dim, hidden)

        self.combine = nn.Linear(5 * hidden, hidden)
        self.trunk = nn.ModuleList(
            ResidualBlock(hidden) for _ in range(int(blocks))
        )
        nn.init.normal_(self.action_positions, std=0.02)
        nn.init.normal_(self.history_positions, std=0.02)
        nn.init.normal_(self.history_summary_token, std=0.02)

    def _card_representation(self, cards: torch.Tensor) -> torch.Tensor:
        present = cards[:, :, 17] > 0.5
        ranks = cards[:, :, :13].argmax(dim=2)
        suits = cards[:, :, 13:17].argmax(dim=2)
        exact = suits * 13 + ranks
        embedded = (
            self.rank_embedding(ranks)
            + self.suit_embedding(suits)
            + self.exact_card_embedding(exact)
        )
        embedded = embedded * present.unsqueeze(2)
        grouped = torch.stack(
            (
                embedded[:, 0:2].sum(dim=1),
                embedded[:, 2:5].sum(dim=1),
                embedded[:, 5],
                embedded[:, 6],
            ),
            dim=1,
        ).flatten(1)
        return F.silu(
            self.card_fc3(F.silu(self.card_fc2(F.silu(self.card_fc1(grouped)))))
        )

    def _history_representation(self, x: torch.Tensor) -> torch.Tensor:
        batch = int(x.shape[0])
        history_end = HISTORY_OFFSET + self.history_steps * HISTORY_FEATURES
        history = x[:, HISTORY_OFFSET:history_end].reshape(
            batch, self.history_steps, HISTORY_FEATURES
        )
        present = history[:, :, 0] > 0.5
        tokens = self.history_projection(history)
        lengths = present.sum(dim=1)
        maximum_length = int(lengths.max().item())
        if maximum_length:
            start = self.history_steps - lengths
            order = (
                torch.arange(self.history_steps, device=x.device).unsqueeze(0)
                + start.unsqueeze(1)
            ) % self.history_steps
            gather = order.unsqueeze(2).expand(-1, -1, self.history_token_dim)
            compact = tokens.gather(1, gather)[:, :maximum_length]
            positions = self.history_positions[:, 1:].expand(
                batch, -1, -1
            ).gather(1, gather)[:, :maximum_length]
            padding = (
                torch.arange(maximum_length, device=x.device).unsqueeze(0)
                >= lengths.unsqueeze(1)
            )
        else:
            compact = tokens[:, :0]
            positions = self.history_positions[:, 1:1].expand(batch, 0, -1)
            padding = torch.zeros(
                batch, 0, dtype=torch.bool, device=x.device
            )
        summary = self.history_summary_token.expand(batch, -1, -1)
        sequence = torch.cat((summary, compact), dim=1)
        sequence = sequence + torch.cat(
            (
                self.history_positions[:, :1].expand(batch, -1, -1),
                positions,
            ),
            dim=1,
        )
        sequence_padding = torch.cat(
            (
                torch.zeros(batch, 1, dtype=torch.bool, device=x.device),
                padding,
            ),
            dim=1,
        )
        attention = self.history_attention(
            sequence, src_key_padding_mask=sequence_padding
        )[:, 0]
        if maximum_length:
            memory_sequence, _ = self.history_memory(compact)
            final_index = (lengths - 1).clamp(min=0)
            memory = memory_sequence[
                torch.arange(batch, device=x.device), final_index
            ]
            memory = memory * (lengths > 0).unsqueeze(1)
        else:
            memory = tokens.new_zeros(batch, self.history_token_dim)
        return F.silu(self.history_fusion(torch.cat((attention, memory), dim=1)))

    def _action_representation(
        self, descriptors: torch.Tensor, legal_mask: torch.Tensor
    ) -> torch.Tensor:
        tokens = self.action_projection(descriptors) + self.action_positions
        padding = legal_mask <= 0
        attended = self.action_attention(
            tokens, src_key_padding_mask=padding
        )
        weights = legal_mask.unsqueeze(2)
        pooled = (attended * weights).sum(dim=1) / weights.sum(
            dim=1
        ).clamp(min=1.0)
        return F.silu(self.action_fusion(pooled))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or int(x.shape[1]) != self.input_dim:
            raise ValueError(
                f"HU Deep CFR compact V4 expects [batch, {self.input_dim}] inputs"
            )
        batch = int(x.shape[0])
        history_end = HISTORY_OFFSET + self.history_steps * HISTORY_FEATURES
        legal_end = history_end + NUM_ACTIONS
        cards = x[:, CARD_STATE_PREFIX_FEATURES:HISTORY_OFFSET].reshape(
            batch, CARD_TOKEN_COUNT, CARD_FEATURES
        )
        legal_mask = x[:, history_end:legal_end]
        descriptors = x[:, legal_end:].reshape(
            batch, NUM_ACTIONS, ACTION_DESCRIPTOR_FEATURES
        )
        card_representation = self._card_representation(cards)
        poker_representation = F.silu(
            self.poker_feature_fc(
                self.poker_feature_norm(
                    poker_relational_features(cards, x[:, :4])
                )
            )
        )
        public = torch.cat(
            (x[:, :CARD_STATE_PREFIX_FEATURES], legal_mask), dim=1
        )
        public_representation = self.public_residual(
            F.silu(self.public_fc(self.public_norm(public)))
        )
        action_representation = self._action_representation(
            descriptors, legal_mask
        )
        history_representation = self._history_representation(x)
        fused = F.silu(
            self.combine(
                torch.cat(
                    (
                        card_representation,
                        poker_representation,
                        public_representation,
                        action_representation,
                        history_representation,
                    ),
                    dim=1,
                )
            )
        )
        for block in self.trunk:
            fused = block(fused)
        return F.normalize(fused, p=2.0, dim=1, eps=1e-8)


class HeadsUpDeepCFRCompactV4Network(nn.Module):
    """Structured backbone with independent heads for all four streets."""

    def __init__(self, input_dim: int, hidden: int = 128, blocks: int = 2):
        super().__init__()
        self.backbone = HeadsUpDeepCFRCompactV4Backbone(
            input_dim, hidden, blocks
        )
        self.street_heads = nn.ModuleList(
            nn.Linear(hidden, NUM_ACTIONS) for _ in range(4)
        )
        for head in self.street_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def _action_logits(
        self,
        representation: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        outputs = torch.stack(
            [head(representation) for head in self.street_heads], dim=1
        )
        return (outputs * x[:, :4].unsqueeze(2)).sum(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._action_logits(self.backbone(x), x)


class HeadsUpDeepCFRCompactV4PolicyRangeNetwork(
    HeadsUpDeepCFRCompactV4Network
):
    """Persistent average-policy model with an auxiliary exact-range head."""

    def __init__(self, input_dim: int, hidden: int = 128, blocks: int = 2):
        super().__init__(input_dim, hidden, blocks)
        self.range_head = nn.Linear(hidden, NUM_OPPONENT_COMBOS)
        nn.init.zeros_(self.range_head.weight)
        nn.init.zeros_(self.range_head.bias)

    def forward_with_range(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        representation = self.backbone(x)
        return (
            self._action_logits(representation, x),
            self.range_head(representation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        representation = self.backbone(x)
        return self._action_logits(representation, x)


def build_advantage_network(
    architecture: str,
    input_dim: int,
    hidden: int = 128,
    blocks: int = 2,
) -> nn.Module:
    if architecture == "residual_mlp":
        return AdvantageNetwork(input_dim, hidden, blocks)
    if architecture == "hu_deep_cfr_compact_v4":
        return HeadsUpDeepCFRCompactV4Network(input_dim, hidden, blocks)
    raise ValueError(f"unknown heads-up network architecture: {architecture!r}")


def build_policy_network(
    architecture: str,
    input_dim: int,
    hidden: int = 128,
    blocks: int = 2,
) -> nn.Module:
    if architecture == "residual_mlp":
        return PolicyNetwork(input_dim, hidden, blocks)
    if architecture == "hu_deep_cfr_compact_v4":
        return HeadsUpDeepCFRCompactV4Network(input_dim, hidden, blocks)
    if architecture == POLICY_RANGE_AUX_ARCHITECTURE:
        return HeadsUpDeepCFRCompactV4PolicyRangeNetwork(
            input_dim,
            hidden,
            blocks,
        )
    raise ValueError(f"unknown heads-up network architecture: {architecture!r}")


def masked_softmax(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """Normalize logits over legal actions only."""

    if logits.shape != legal_mask.shape:
        raise ValueError("logits and legal_mask must have the same shape")
    if torch.any(legal_mask.sum(dim=-1) <= 0):
        raise ValueError("each policy row must contain at least one legal action")
    masked = logits.masked_fill(legal_mask <= 0, -1e9)
    return torch.softmax(masked, dim=-1)


__all__ = [
    "ACTION_DESCRIPTOR_FEATURE_NAMES",
    "ACTION_DESCRIPTOR_FEATURES",
    "ActionDescriptor",
    "AdvantageNetwork",
    "CARD_FEATURES",
    "CARD_STATE_FEATURES",
    "CARD_STATE_PREFIX_FEATURES",
    "CARD_TOKEN_COUNT",
    "DEFAULT_MAX_HISTORY",
    "ENCODER_SCHEMA_VERSION",
    "ENCODER_VERSION",
    "GLOBAL_FEATURE_NAMES",
    "HISTORY_FEATURE_NAMES",
    "HISTORY_FEATURES",
    "HISTORY_OFFSET",
    "HeadsUpDeepCFRCompactV4Backbone",
    "HeadsUpDeepCFRCompactV4Network",
    "HeadsUpDeepCFRCompactV4PolicyRangeNetwork",
    "NETWORK_ARCHITECTURES",
    "NUM_OPPONENT_COMBOS",
    "POLICY_RANGE_AUX_ARCHITECTURE",
    "POKER_RELATIONAL_FEATURES",
    "PolicyNetwork",
    "PUBLIC_PREFIX_FEATURES",
    "SEAT_FEATURE_NAMES",
    "SEMANTIC_ACTIONS",
    "action_descriptor_offset",
    "build_action_descriptors",
    "build_advantage_network",
    "build_policy_network",
    "encode_information_state",
    "encoder_metadata",
    "information_state_size",
    "legal_mask_offset",
    "masked_softmax",
    "poker_relational_features",
]
