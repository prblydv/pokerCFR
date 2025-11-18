# abstraction.py
import itertools
from typing import List

import torch

from config import STACK_SIZE


# --- Card utilities ----------------------------------------------------------

def card_rank(card: int) -> int:
    """Return rank 2..14 (2..10,J=11,Q=12,K=13,A=14) from 0..51."""
    r = card % 13  # 0..12
    return r + 2

def card_suit(card: int) -> int:
    """Return suit 0..3 from 0..51."""
    return card // 13


# --- 5-card evaluator --------------------------------------------------------

# Categories:
# 1: High card
# 2: One pair
# 3: Two pair
# 4: Trips
# 5: Straight
# 6: Flush
# 7: Full house
# 8: Quads
# 9: Straight flush

def evaluate_5card(cards: List[int]) -> int:
    """
    Return a score for a 5-card hand.
    Higher is better.
    """
    assert len(cards) == 5, "Internal error: evaluate_5card() must be given exactly 5 cards."

    ranks = sorted([card_rank(c) for c in cards], reverse=True)
    suits = [card_suit(c) for c in cards]

    # Count ranks
    rank_counts = {}
    for r in ranks:
        rank_counts[r] = rank_counts.get(r, 0) + 1

    # Sort by frequency (desc), then rank (desc)
    counts = sorted(rank_counts.items(), key=lambda x: (-x[1], -x[0]))
    freqs = [c[1] for c in counts]

    is_flush = len(set(suits)) == 1

    # Straight detection -------------------------------------------------------
    unique_ranks = sorted(set(ranks), reverse=True)
    is_straight = False
    high_straight = 0

    if len(unique_ranks) >= 5:
        # normal straights
        for i in range(len(unique_ranks) - 4):
            seq = unique_ranks[i:i+5]
            if seq[0] - seq[4] == 4:
                is_straight = True
                high_straight = seq[0]
                break

    # Wheel A-5: A,5,4,3,2
    if not is_straight and {14,5,4,3,2}.issubset(set(unique_ranks)):
        is_straight = True
        high_straight = 5

    # Straight flush
    if is_flush and is_straight:
        category = 9
        return category * 10**8 + high_straight * 10**6

    # Quads
    if freqs[0] == 4:
        quad = counts[0][0]
        kicker = max(r for r in ranks if r != quad)
        category = 8
        return category * 10**8 + quad * 10**6 + kicker * 10**4

    # Full house
    if freqs[0] == 3 and freqs[1] >= 2:
        trip = counts[0][0]
        pair = counts[1][0]
        category = 7
        return category * 10**8 + trip * 10**6 + pair * 10**4

    # Flush
    if is_flush:
        category = 6
        score = category * 10**8
        for i, r in enumerate(sorted(ranks, reverse=True)):
            score += r * (10 ** (6 - 2 * i))
        return score

    # Straight
    if is_straight:
        category = 5
        return category * 10**8 + high_straight * 10**6

    # Trips
    if freqs[0] == 3:
        trip = counts[0][0]
        kickers = sorted((r for r in ranks if r != trip), reverse=True)[:2]
        category = 4
        score = category * 10**8 + trip * 10**6
        for i, k in enumerate(kickers):
            score += k * (10 ** (4 - 2 * i))
        return score

    # Two pair
    if freqs[0] == 2 and freqs[1] == 2:
        p1 = counts[0][0]
        p2 = counts[1][0]
        hi, lo = max(p1, p2), min(p1, p2)
        kicker = max(r for r in ranks if r != p1 and r != p2)
        category = 3
        return (category * 10**8 +
                hi * 10**6 +
                lo * 10**4 +
                kicker * 10**2)

    # One pair
    if freqs[0] == 2:
        pair = counts[0][0]
        kickers = sorted((r for r in ranks if r != pair), reverse=True)[:3]
        category = 2
        score = category * 10**8 + pair * 10**6
        for i, k in enumerate(kickers):
            score += k * (10 ** (4 - 2 * i))
        return score

    # High card
    category = 1
    score = category * 10**8
    for i, r in enumerate(sorted(ranks, reverse=True)):
        score += r * (10 ** (6 - 2 * i))
    return score


# --- 7-card evaluation -------------------------------------------------------

def evaluate_7card(hole: List[int], board: List[int]) -> int:
    """
    Best 5-card hand from up to 7 cards.
    If < 5 cards, returns 0 (safe).
    """
    cards = hole + board
    if len(cards) < 5:
        return 0  # cannot evaluate a real hand yet

    best = 0
    for combo in itertools.combinations(cards, 5):
        val = evaluate_5card(list(combo))
        if val > best:
            best = val
    return best


# --- Strength abstraction ----------------------------------------------------

# --- Strength estimator ------------------------------------------------------

def normalized_strength(hole: List[int], board: List[int]) -> float:
    """
    Returns a continuous strength estimator in [0,1].
    Preflop (0-2 cards on board): use hole-card ranks only.
    Flop/turn/river: use 7-card evaluation.
    """
    if len(board) < 3:
        if len(hole) < 2:
            return 0.5  # neutral if we somehow don't have 2 cards yet
        r = sorted([card_rank(c) for c in hole], reverse=True)
        return (r[0] + r[1]) / (2 * 14.0)  # avg rank / max_rank

    raw = evaluate_7card(hole, board)
    return raw / 1e9  # large enough to keep in [0,1)-ish


# --- Hole-card feature encoding ---------------------------------------------

def encode_hole_cards(hole: List[int]) -> List[float]:
    """
    Encode the private 2-card hand in a simple, information-rich way.

    Features:
      - hi_rank_norm, lo_rank_norm in [0,1]
      - suited flag (0/1)
      - pair flag (0/1)
    """
    assert len(hole) == 2
    r1 = card_rank(hole[0]) - 2  # 0..12
    r2 = card_rank(hole[1]) - 2
    s1 = hole[0] // 13           # 0..3
    s2 = hole[1] // 13

    # order hi, lo
    if r2 > r1:
        r1, r2 = r2, r1
        s1, s2 = s2, s1

    hi_rank_norm = r1 / 12.0
    lo_rank_norm = r2 / 12.0
    suited = 1.0 if s1 == s2 else 0.0
    pair = 1.0 if r1 == r2 else 0.0

    return [hi_rank_norm, lo_rank_norm, suited, pair]


# --- State encoding ----------------------------------------------------------

def encode_state(state, player: int) -> torch.Tensor:
    """
    Encode public + private information for Deep CFR.

    Layout:
      [ street_one_hot(4),
        acting_player_flag,
        pot_norm,
        stack0_norm, stack1_norm,
        current_bet_norm,
        last_aggressor_flag,
        hand_strength,
        board_strength,
        hole_hi_rank_norm,
        hole_lo_rank_norm,
        hole_suited_flag,
        hole_pair_flag ]
    """
    from poker_env import STREET_PREFLOP, STREET_FLOP, STREET_TURN, STREET_RIVER

    # Street one-hot
    street_oh = [0.0, 0.0, 0.0, 0.0]
    street_oh[state.street] = 1.0

    # Public scalars
    pot_norm = state.pot / (STACK_SIZE * 2)
    stacks_norm = [s / STACK_SIZE for s in state.stacks]
    curr_bet_norm = state.current_bet / STACK_SIZE

    # Last aggressor from current player's perspective
    if state.last_aggressor == -1:
        last_agg_flag = 0.0
    else:
        last_agg_flag = float(state.last_aggressor == player)

    # Strength estimates
    hand_str = normalized_strength(state.hole[player], state.board)
    board_str = normalized_strength([], state.board) if state.board else 0.0

    # Private hole-card identity features
    hole_feats = encode_hole_cards(state.hole[player])

    vec = street_oh + [
        float(player),
        pot_norm,
        stacks_norm[0],
        stacks_norm[1],
        curr_bet_norm,
        last_agg_flag,
        hand_str,
        board_str,
    ] + hole_feats

    return torch.tensor(vec, dtype=torch.float32)
