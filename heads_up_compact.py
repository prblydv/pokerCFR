"""Compact, lossless-through-100BB HU information-state encoding.

The logical observation is ``40 + 7 * history_length`` typed fields.  The
training tensor reserves the proven 100BB live-decision maximum of 106 events
so reservoirs and the native batch boundary remain dense and fast.  Padding is
identified by categorical PAD ids and must be masked by the network.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch

from heads_up_engine import (
    ACTION_NAMES,
    ACTION_SCHEMA_VERSION,
    ENGINE_SCHEMA_VERSION,
    NUM_ACTIONS,
)


COMPACT_ENCODER_VERSION = 1
COMPACT_ENCODER_SCHEMA_VERSION = "hu_compact_information_state_v1_full_history"
COMPACT_DEFAULT_MAX_HISTORY = 106
COMPACT_MAX_EFFECTIVE_DEPTH_BB = 100
COMPACT_MAX_TOTAL_INITIAL_BB = 200.0

COMPACT_CONTEXT_FEATURES = 13
COMPACT_CARD_SLOTS = 7
COMPACT_HISTORY_FEATURES = 7
COMPACT_ACTION_FEATURES = 2

COMPACT_CONTEXT_OFFSET = 0
COMPACT_CARD_OFFSET = COMPACT_CONTEXT_FEATURES
COMPACT_HISTORY_OFFSET = COMPACT_CARD_OFFSET + COMPACT_CARD_SLOTS

COMPACT_CONTEXT_FEATURE_NAMES = (
    "street_id",
    "hero_is_button",
    "last_full_raiser_id",
    "hero_raise_right",
    "opponent_raise_right",
    "log_total_initial_depth",
    "hero_initial_stack_share",
    "hero_stack_share",
    "opponent_stack_share",
    "hero_street_contribution_share",
    "opponent_street_contribution_share",
    "minimum_raise_share",
    "small_blind_over_big_blind",
)
COMPACT_CARD_SLOT_NAMES = (
    "hero_hole_1",
    "hero_hole_2",
    "flop_1",
    "flop_2",
    "flop_3",
    "turn",
    "river",
)
COMPACT_HISTORY_FEATURE_NAMES = (
    "street_id",
    "actor_id",
    "semantic_action_id",
    "exact_target_share",
    "pot_before_share",
    "all_in",
    "full_raise",
)
COMPACT_ACTION_FEATURE_NAMES = ("legal", "exact_target_share")
COMPACT_SEMANTIC_ACTIONS = ("fold", "check", "call", "bet", "raise")


def compact_action_offset(
    max_history: int = COMPACT_DEFAULT_MAX_HISTORY,
) -> int:
    return COMPACT_HISTORY_OFFSET + int(max_history) * COMPACT_HISTORY_FEATURES


def compact_information_state_size(
    max_history: int = COMPACT_DEFAULT_MAX_HISTORY,
) -> int:
    if isinstance(max_history, bool) or int(max_history) <= 0:
        raise ValueError("max_history must be a positive integer")
    return compact_action_offset(max_history) + NUM_ACTIONS * COMPACT_ACTION_FEATURES


def compact_encoder_metadata(
    max_history: int = COMPACT_DEFAULT_MAX_HISTORY,
) -> dict[str, Any]:
    max_history = int(max_history)
    width = compact_information_state_size(max_history)
    return {
        "engine_schema_version": ENGINE_SCHEMA_VERSION,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "encoder_version": COMPACT_ENCODER_VERSION,
        "encoder_schema_version": COMPACT_ENCODER_SCHEMA_VERSION,
        "width": width,
        "input_dim": width,
        "max_history": max_history,
        "history_policy": "full_error_on_overflow",
        "max_effective_depth_bb": COMPACT_MAX_EFFECTIVE_DEPTH_BB,
        "logical_width": "40 + 7 * history_length",
        "card_storage": "seven_exact_card_ids_plus_one_with_zero_pad",
        "context_feature_names": COMPACT_CONTEXT_FEATURE_NAMES,
        "card_slot_names": COMPACT_CARD_SLOT_NAMES,
        "history_feature_names": COMPACT_HISTORY_FEATURE_NAMES,
        "action_feature_names": COMPACT_ACTION_FEATURE_NAMES,
        "action_names": tuple(ACTION_NAMES),
        "num_actions": NUM_ACTIONS,
    }


def _field(value: object, name: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _descriptor_values(
    descriptors: Sequence[object | None] | Mapping[int, object | None] | None,
) -> list[object | None]:
    if descriptors is None:
        return [None] * NUM_ACTIONS
    if isinstance(descriptors, Mapping):
        return [descriptors.get(action) for action in range(NUM_ACTIONS)]
    values = list(descriptors)
    if len(values) != NUM_ACTIONS:
        raise ValueError(f"action_descriptors must contain {NUM_ACTIONS} entries")
    return values


def _semantic_kind(event: object) -> str:
    raw = str(_field(event, "kind", "")).strip().lower().replace("-", "_")
    if raw in COMPACT_SEMANTIC_ACTIONS:
        return raw
    current_before = float(_field(event, "current_bet_before", 0.0))
    current_after = float(_field(event, "current_bet_after", current_before))
    amount = float(_field(event, "amount", 0.0))
    if current_after > current_before + 1e-9:
        return "bet" if current_before <= 1e-9 else "raise"
    return "call" if amount > 1e-9 else "check"


def _card_slots(state: object, hero: int) -> list[int]:
    hole = list(_field(state, "hole", ()))
    if len(hole) != 2 or len(hole[hero]) != 2:
        raise ValueError("state.hole must contain two cards for each seat")
    hero_cards = sorted(int(card) for card in hole[hero])
    board = [int(card) for card in _field(state, "board", ())]
    if len(board) > 5:
        raise ValueError("board cannot contain more than five cards")
    canonical_board = sorted(board[:3]) + board[3:]
    values = hero_cards + canonical_board + [-1] * (5 - len(canonical_board))
    if any(card < -1 or card >= 52 for card in values):
        raise ValueError("cards must use exact ids 0..51 or -1 for padding")
    return values


def encode_compact_information_state(
    state: object,
    hero: int,
    legal_actions: Iterable[int],
    big_blind: float,
    max_history: int = COMPACT_DEFAULT_MAX_HISTORY,
    *,
    action_descriptors: Sequence[object | None]
    | Mapping[int, object | None]
    | None = None,
) -> torch.Tensor:
    """Encode a live HU decision without truncating its public history."""

    if hero not in (0, 1):
        raise ValueError("hero must be seat 0 or 1")
    if isinstance(big_blind, bool) or float(big_blind) <= 0.0:
        raise ValueError("big_blind must be positive")
    max_history = int(max_history)
    if max_history <= 0:
        raise ValueError("max_history must be positive")
    legal = tuple(int(action) for action in legal_actions)
    legal_set = set(legal)
    if any(action < 0 or action >= NUM_ACTIONS for action in legal_set):
        raise ValueError("legal actions must be in 0..9")
    actor = _field(state, "to_act", None)
    if legal_set and (actor is None or int(actor) != hero):
        raise ValueError("live compact encoding requires hero == state.to_act")
    if legal_set and action_descriptors is None:
        raise ValueError("live compact encoding requires exact action descriptors")

    full_history = list(_field(state, "history", ()))
    if len(full_history) > max_history:
        raise ValueError(
            "compact information-state history exceeds the lossless capacity: "
            f"{len(full_history)} > {max_history}"
        )

    if type(state).__module__ == "heads_up_native_engine":
        try:
            from heads_up_native import encode_compact_information_state_native

            encoded = encode_compact_information_state_native(
                state,
                hero,
                legal,
                float(big_blind),
                max_history,
                action_descriptors=action_descriptors,
            )
            return torch.from_numpy(encoded)
        except (ImportError, AttributeError):
            pass

    opponent = 1 - hero
    stacks = [float(value) for value in _field(state, "stacks", ())]
    initial = [float(value) for value in _field(state, "initial_stacks", ())]
    street_contrib = [float(value) for value in _field(state, "street_contrib", ())]
    raise_rights = [bool(value) for value in _field(state, "raise_rights", ())]
    if not all(len(values) == 2 for values in (stacks, initial, street_contrib, raise_rights)):
        raise ValueError("compact encoder requires two-seat stack metadata")
    total_initial = sum(initial)
    if total_initial <= 0.0:
        raise ValueError("total initial chips must be positive")
    bb = float(big_blind)
    street = int(_field(state, "street", -1))
    if street not in range(4):
        raise ValueError("street must be in 0..3")
    button = int(_field(state, "button", -1))
    if button not in (0, 1):
        raise ValueError("button must be seat 0 or 1")
    last_raiser = _field(state, "last_full_raiser", None)
    last_raiser_id = (
        0 if last_raiser is None else 1 if int(last_raiser) == hero else 2
    )
    minimum_raise = float(
        _field(state, "min_raise", _field(state, "minimum_raise", 0.0))
    )
    small_blind = float(_field(state, "small_blind", 0.5 * bb))

    values: list[float] = [
        float(street),
        float(hero == button),
        float(last_raiser_id),
        float(raise_rights[hero]),
        float(raise_rights[opponent]),
        math.log1p(total_initial / bb) / math.log1p(COMPACT_MAX_TOTAL_INITIAL_BB),
        initial[hero] / total_initial,
        stacks[hero] / total_initial,
        stacks[opponent] / total_initial,
        street_contrib[hero] / total_initial,
        street_contrib[opponent] / total_initial,
        minimum_raise / total_initial,
        small_blind / bb,
    ]
    values.extend(float(card + 1) for card in _card_slots(state, hero))

    semantic_ids = {
        name: index + 1 for index, name in enumerate(COMPACT_SEMANTIC_ACTIONS)
    }
    for event in full_history:
        event_street = int(_field(event, "street", -1))
        event_actor = int(_field(event, "player", _field(event, "actor", -1)))
        if event_street not in range(4) or event_actor not in (0, 1):
            raise ValueError("history contains an invalid street or actor")
        contribution_after = float(
            _field(event, "contribution_after", _field(event, "raise_to", 0.0))
        )
        pot_before = float(_field(event, "pot_before", 0.0))
        values.extend(
            (
                float(event_street + 1),
                1.0 if event_actor == hero else 2.0,
                float(semantic_ids[_semantic_kind(event)]),
                contribution_after / total_initial,
                pot_before / total_initial,
                float(bool(_field(event, "all_in", False))),
                float(bool(_field(event, "full_raise", False))),
            )
        )
    values.extend(
        [0.0] * ((max_history - len(full_history)) * COMPACT_HISTORY_FEATURES)
    )

    descriptors = _descriptor_values(action_descriptors)
    missing = [a for a in legal_set if descriptors[a] is None]
    extras = [a for a, item in enumerate(descriptors) if item is not None and a not in legal_set]
    if missing:
        raise ValueError(f"missing descriptors for legal actions: {missing}")
    if extras:
        raise ValueError(f"descriptors supplied for illegal actions: {extras}")
    for action, descriptor in enumerate(descriptors):
        if descriptor is None:
            values.extend((0.0, 0.0))
        else:
            target = float(_field(descriptor, "target", 0.0))
            values.extend((1.0, target / total_initial))

    expected = compact_information_state_size(max_history)
    if len(values) != expected:
        raise RuntimeError(f"compact encoder produced {len(values)} != {expected}")
    return torch.tensor(values, dtype=torch.float32)


def compact_street_indices(x: torch.Tensor) -> torch.Tensor:
    return x[:, COMPACT_CONTEXT_OFFSET].round().to(torch.long).clamp(0, 3)


def compact_street_one_hot(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.one_hot(
        compact_street_indices(x), num_classes=4
    ).to(x.dtype)


def compact_visible_card_ids(x: torch.Tensor) -> torch.Tensor:
    values = x[:, COMPACT_CARD_OFFSET : COMPACT_CARD_OFFSET + COMPACT_CARD_SLOTS]
    return values.round().to(torch.long) - 1


def is_compact_information_state_tensor(x: torch.Tensor) -> bool:
    """Identify a compact observation from its locked width and typed prefix."""

    if x.ndim != 2:
        return False
    width = int(x.shape[1])
    fixed = COMPACT_HISTORY_OFFSET + NUM_ACTIONS * COMPACT_ACTION_FEATURES
    history_values = width - fixed
    if history_values <= 0 or history_values % COMPACT_HISTORY_FEATURES:
        return False
    # The first and card fields are categorical integers in the compact ABI.
    prefix = x[:, : COMPACT_CARD_OFFSET + COMPACT_CARD_SLOTS]
    if prefix.numel() == 0 or not bool(torch.isfinite(prefix).all()):
        return False
    streets = x[:, COMPACT_CONTEXT_OFFSET]
    cards = x[:, COMPACT_CARD_OFFSET : COMPACT_CARD_OFFSET + COMPACT_CARD_SLOTS]
    return bool(
        torch.allclose(streets, streets.round(), atol=1e-5, rtol=0.0)
        and bool(((streets >= 0.0) & (streets <= 3.0)).all())
        and torch.allclose(cards, cards.round(), atol=1e-5, rtol=0.0)
        and bool(((cards >= 0.0) & (cards <= 52.0)).all())
    )


__all__ = [name for name in globals() if name.startswith("COMPACT_")] + [
    "compact_action_offset",
    "compact_encoder_metadata",
    "compact_information_state_size",
    "is_compact_information_state_tensor",
    "compact_street_indices",
    "compact_street_one_hot",
    "compact_visible_card_ids",
    "encode_compact_information_state",
]
