"""Exact opponent-combination indexing and blocker masks for heads-up Hold'em."""

from __future__ import annotations

import math
from typing import Sequence

import torch

from heads_up_models import (
    CARD_FEATURES,
    CARD_STATE_PREFIX_FEATURES,
    CARD_TOKEN_COUNT,
    HISTORY_OFFSET,
)


NUM_CARDS = 52
NUM_OPPONENT_COMBOS = math.comb(NUM_CARDS, 2)
RANK_LABELS = "23456789TJQKA"
SUIT_LABELS = "cdhs"

OPPONENT_COMBOS = tuple(
    (first, second)
    for first in range(NUM_CARDS)
    for second in range(first + 1, NUM_CARDS)
)
if len(OPPONENT_COMBOS) != 1_326:
    raise RuntimeError("the opponent-combination table must contain 1,326 rows")

_COMBO_TO_INDEX = {
    combo: index for index, combo in enumerate(OPPONENT_COMBOS)
}
COMBO_FIRST_CARD = torch.tensor(
    [combo[0] for combo in OPPONENT_COMBOS],
    dtype=torch.long,
)
COMBO_SECOND_CARD = torch.tensor(
    [combo[1] for combo in OPPONENT_COMBOS],
    dtype=torch.long,
)


def card_label(card: int) -> str:
    card = int(card)
    if not 0 <= card < NUM_CARDS:
        raise ValueError(f"card must be in 0..51, got {card}")
    return f"{RANK_LABELS[card % 13]}{SUIT_LABELS[card // 13]}"


def opponent_combo_index(cards: Sequence[int]) -> int:
    if len(cards) != 2:
        raise ValueError("an opponent hand must contain exactly two cards")
    first, second = sorted(int(card) for card in cards)
    if first == second or not 0 <= first < second < NUM_CARDS:
        raise ValueError("opponent cards must be two distinct cards in 0..51")
    return _COMBO_TO_INDEX[(first, second)]


def opponent_combo_label(index: int) -> str:
    try:
        first, second = OPPONENT_COMBOS[int(index)]
    except (IndexError, TypeError) as error:
        raise ValueError("opponent combination index must be in 0..1325") from error
    return f"{card_label(first)}{card_label(second)}"


def _hand_class_label(combo: tuple[int, int]) -> str:
    first, second = combo
    first_rank = first % 13
    second_rank = second % 13
    if first_rank == second_rank:
        rank = RANK_LABELS[first_rank]
        return f"{rank}{rank}"
    high = max(first_rank, second_rank)
    low = min(first_rank, second_rank)
    suited = first // 13 == second // 13
    return (
        f"{RANK_LABELS[high]}{RANK_LABELS[low]}"
        f"{'s' if suited else 'o'}"
    )


HAND_CLASS_LABELS = tuple(
    sorted({_hand_class_label(combo) for combo in OPPONENT_COMBOS})
)
if len(HAND_CLASS_LABELS) != 169:
    raise RuntimeError("the exact combinations must aggregate to 169 hand classes")
_HAND_CLASS_TO_INDEX = {
    label: index for index, label in enumerate(HAND_CLASS_LABELS)
}
COMBO_HAND_CLASS_INDEX = torch.tensor(
    [
        _HAND_CLASS_TO_INDEX[_hand_class_label(combo)]
        for combo in OPPONENT_COMBOS
    ],
    dtype=torch.long,
)


def valid_combo_mask_from_encoded(
    information_states: torch.Tensor,
) -> torch.Tensor:
    """Return a blocker-valid ``[batch, 1326]`` mask from encoded visible cards."""

    if information_states.ndim != 2:
        raise ValueError("information_states must have shape [batch, features]")
    if int(information_states.shape[1]) < HISTORY_OFFSET:
        raise ValueError("information-state width is too small for visible cards")
    batch = int(information_states.shape[0])
    cards = information_states[
        :, CARD_STATE_PREFIX_FEATURES:HISTORY_OFFSET
    ].reshape(batch, CARD_TOKEN_COUNT, CARD_FEATURES)
    present = cards[:, :, 17] > 0.5
    ranks = cards[:, :, :13].argmax(dim=2)
    suits = cards[:, :, 13:17].argmax(dim=2)
    exact_cards = suits * 13 + ranks

    blocked = torch.zeros(
        batch,
        NUM_CARDS,
        dtype=torch.bool,
        device=information_states.device,
    )
    blocked.scatter_(1, exact_cards, present)
    first = COMBO_FIRST_CARD.to(information_states.device)
    second = COMBO_SECOND_CARD.to(information_states.device)
    return ~blocked[:, first] & ~blocked[:, second]


def masked_range_probabilities(
    logits: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    if logits.shape != valid_mask.shape:
        raise ValueError("range logits and valid mask must have identical shapes")
    if int(logits.shape[-1]) != NUM_OPPONENT_COMBOS:
        raise ValueError("range head must output exactly 1,326 logits")
    if torch.any(valid_mask.sum(dim=1) <= 0):
        raise ValueError("every range row must contain a valid combination")
    return torch.softmax(logits.masked_fill(~valid_mask, -1e9), dim=1)


__all__ = [
    "COMBO_FIRST_CARD",
    "COMBO_HAND_CLASS_INDEX",
    "COMBO_SECOND_CARD",
    "HAND_CLASS_LABELS",
    "NUM_OPPONENT_COMBOS",
    "OPPONENT_COMBOS",
    "card_label",
    "masked_range_probabilities",
    "opponent_combo_index",
    "opponent_combo_label",
    "valid_combo_mask_from_encoded",
]
