# abstraction.py

# expand the hole features (e.g., 10-dim richer encoding, Kicker gap, connectedness, suitedness one-hot, rank buckets, etc.).

import itertools
from typing import List

import torch

from config import STACK_SIZE
import itertools
import os
import time
import random
import pickle
from typing import List

from config import STACK_SIZE
# ------------------------------------------------------------------
# Hand-strength LUT: maps 7-card raw score -> normalized [0,1]
# ------------------------------------------------------------------

_LUT_PATH = "hand_strength_lut.pkl"
_SCORE_TO_PCTL = None  # filled by _load_or_build_lut()


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
from typing import List

# Precompute rank/suit for 0..51 once (faster than calling helpers each time)
_RANK_LUT = [c % 13 + 2 for c in range(52)]   # 2..14
_SUIT_LUT = [c // 13       for c in range(52)]  # 0..3


def evaluate_5card(cards: List[int]) -> int:
    """
    Fast 5-card hand evaluator.
    Input:  list of 5 cards (0..51).
    Output: integer score, higher is better.
    """

    # --- decode ranks & suits ---
    r0 = _RANK_LUT[cards[0]]
    r1 = _RANK_LUT[cards[1]]
    r2 = _RANK_LUT[cards[2]]
    r3 = _RANK_LUT[cards[3]]
    r4 = _RANK_LUT[cards[4]]

    s0 = _SUIT_LUT[cards[0]]
    s1 = _SUIT_LUT[cards[1]]
    s2 = _SUIT_LUT[cards[2]]
    s3 = _SUIT_LUT[cards[3]]
    s4 = _SUIT_LUT[cards[4]]

    ranks = [r0, r1, r2, r3, r4]
    ranks.sort()                # ascending
    r_desc = ranks[::-1]        # descending

    # --- flush check ---
    is_flush = (s0 == s1 == s2 == s3 == s4)

    # --- straight check ---
    uniq = sorted(set(ranks))
    is_straight = False
    high_straight = 0
    if len(uniq) == 5:
        # wheel: A-5 = [2,3,4,5,14]
        if uniq == [2, 3, 4, 5, 14]:
            is_straight = True
            high_straight = 5
        elif uniq[-1] - uniq[0] == 4:
            is_straight = True
            high_straight = uniq[-1]

    # --- rank frequencies (no dict, fixed small array) ---
    cnt = [0] * 15  # 0..14
    for r in ranks:
        cnt[r] += 1
    distinct = [r for r in range(2, 15) if cnt[r] > 0]
    # sort distinct ranks by (freq, rank) descending
    distinct.sort(key=lambda r: (cnt[r], r), reverse=True)
    freqs = [cnt[r] for r in distinct]

    # scoring scheme: cat in [1..9], then 5 kickers with 2 digits each
    BASE = 10 ** 10

    # --- Straight flush ---
    if is_flush and is_straight:
        cat = 9
        return cat * BASE + high_straight * 10**8

    # --- Four of a kind ---
    if freqs[0] == 4:
        quad = distinct[0]
        kicker = distinct[1]
        cat = 8
        return (cat * BASE +
                quad * 10**8 +
                kicker * 10**6)

    # --- Full house ---
    if freqs[0] == 3 and freqs[1] == 2:
        trip = distinct[0]
        pair = distinct[1]
        cat = 7
        return (cat * BASE +
                trip * 10**8 +
                pair * 10**6)

    # --- Flush ---
    if is_flush:
        cat = 6
        # use all 5 ranks as kickers
        a, b, c, d, e = r_desc
        return (cat * BASE +
                a * 10**8 +
                b * 10**6 +
                c * 10**4 +
                d * 10**2 +
                e)

    # --- Straight ---
    if is_straight:
        cat = 5
        return cat * BASE + high_straight * 10**8

    # --- Trips ---
    if freqs[0] == 3:
        trip = distinct[0]
        # remaining two kickers (highest first)
        kickers = [r for r in r_desc if r != trip][:2]
        k1, k2 = kickers
        cat = 4
        return (cat * BASE +
                trip * 10**8 +
                k1 * 10**6 +
                k2 * 10**4)

    # --- Two pair ---
    if freqs[0] == 2 and freqs[1] == 2:
        p1, p2 = distinct[0], distinct[1]
        hi, lo = max(p1, p2), min(p1, p2)
        kicker = [r for r in r_desc if r != hi and r != lo][0]
        cat = 3
        return (cat * BASE +
                hi * 10**8 +
                lo * 10**6 +
                kicker * 10**4)

    # --- One pair ---
    if freqs[0] == 2:
        pair = distinct[0]
        kickers = [r for r in r_desc if r != pair][:3]
        k1, k2, k3 = kickers
        cat = 2
        return (cat * BASE +
                pair * 10**8 +
                k1 * 10**6 +
                k2 * 10**4 +
                k3 * 10**2)

    # --- High card ---
    cat = 1
    a, b, c, d, e = r_desc
    return (cat * BASE +
            a * 10**8 +
            b * 10**6 +
            c * 10**4 +
            d * 10**2 +
            e)


# --- 7-card evaluation -------------------------------------------------------
def evaluate_7card(hole: List[int], board: List[int]) -> int:
    """
    Best 5-card hand from 7 cards.
    Same interface as before.
    Uses unrolled combinations for speed (21 evals only).
    """

    cards = hole + board
    n = len(cards)
    if n < 5:
        return 0

    # If exactly 5, skip loops entirely.
    if n == 5:
        return evaluate_5card(cards)

    # If 6 or 7 → evaluate all 5-card subsets quickly
    # Using unrolled indices eliminates itertools + list alloc.
    best = 0

    c0, c1, c2, c3, c4, c5, c6 = (cards + [0,0])[:7]  # safe padding

    # 21 five-card subsets for 7 cards:
    combos = [
        (c0, c1, c2, c3, c4),
        (c0, c1, c2, c3, c5),
        (c0, c1, c2, c3, c6),
        (c0, c1, c2, c4, c5),
        (c0, c1, c2, c4, c6),
        (c0, c1, c2, c5, c6),
        (c0, c1, c3, c4, c5),
        (c0, c1, c3, c4, c6),
        (c0, c1, c3, c5, c6),
        (c0, c1, c4, c5, c6),
        (c0, c2, c3, c4, c5),
        (c0, c2, c3, c4, c6),
        (c0, c2, c3, c5, c6),
        (c0, c2, c4, c5, c6),
        (c0, c3, c4, c5, c6),
        (c1, c2, c3, c4, c5),
        (c1, c2, c3, c4, c6),
        (c1, c2, c3, c5, c6),
        (c1, c2, c4, c5, c6),
        (c1, c3, c4, c5, c6),
        (c2, c3, c4, c5, c6),
    ]

    for c in combos:
        v = evaluate_5card(c)
        if v > best:
            best = v

    return best



# --- Strength abstraction ----------------------------------------------------
def _build_strength_lut(num_samples: int = 500_000, batch_size: int = 5_000) -> dict:
    """
    Build a LUT mapping evaluate_7card() scores to percentiles in [0,1].
    Uses random 7-card samples.
    Shows progress + rough ETA.
    """
    print(f"[LUT] Building hand-strength LUT with {num_samples} samples...")
    start_time = time.time()
    scores = []

    cards_all = list(range(52))

    for i in range(0, num_samples, batch_size):
        batch_end = min(num_samples, i + batch_size)
        for _ in range(i, batch_end):
            # sample 7 distinct cards
            seven = random.sample(cards_all, 7)
            v = evaluate_7card(seven[:2], seven[2:])  # hole=2, board=5
            scores.append(v)

        # progress + ETA
        done = batch_end
        elapsed = time.time() - start_time
        frac = done / num_samples
        rate = done / elapsed if elapsed > 0 else 0.0
        remaining = (num_samples - done) / rate if rate > 0 else 0.0

        print(
            f"[LUT] {done}/{num_samples} "
            f"({frac:6.2%}) "
            f"elapsed={elapsed:6.1f}s "
            f"ETA={remaining:6.1f}s",
            end="\r",
            flush=True,
        )

    print()  # newline after progress

    # build percentile mapping
    unique_scores = sorted(set(scores))
    n = len(unique_scores)
    print(f"[LUT] Unique scores: {n}")

    score_to_pctl = {}
    if n == 1:
        score_to_pctl[unique_scores[0]] = 0.5
    else:
        for idx, s in enumerate(unique_scores):
            score_to_pctl[s] = idx / (n - 1)

    # save to disk
    with open(_LUT_PATH, "wb") as f:
        pickle.dump(score_to_pctl, f)

    total_time = time.time() - start_time
    print(f"[LUT] Built and saved LUT to '{_LUT_PATH}' in {total_time:.1f}s")

    return score_to_pctl
def _load_or_build_lut() -> dict:
    """
    Load LUT from disk if present; otherwise build it.
    """
    global _SCORE_TO_PCTL

    if _SCORE_TO_PCTL is not None:
        return _SCORE_TO_PCTL

    if os.path.exists(_LUT_PATH):
        print(f"[LUT] Loading hand-strength LUT from '{_LUT_PATH}'...")
        with open(_LUT_PATH, "rb") as f:
            _SCORE_TO_PCTL = pickle.load(f)
        print(f"[LUT] Loaded {len(_SCORE_TO_PCTL)} entries.")
    else:
        _SCORE_TO_PCTL = _build_strength_lut()

    return _SCORE_TO_PCTL


# initialize on import
_load_or_build_lut()

# --- Strength estimator ------------------------------------------------------

def normalized_strength(hole: List[int], board: List[int]) -> float:
    """
    Returns a continuous strength estimator in [0,1].

    Preflop (0-2 cards on board): simple hole-card rank heuristic.
    Flop/turn/river: evaluate best 5-card hand out of 7, then map
    its raw score through the LUT to a [0,1] percentile.
    """
    # Preflop heuristic stays cheap and simple
    if len(board) < 3:
        if len(hole) < 2:
            return 0.5  # neutral if we somehow don't have 2 cards yet
        r = sorted([card_rank(c) for c in hole], reverse=True)
        return (r[0] + r[1]) / (2 * 14.0)  # avg rank / max_rank

    # Postflop: use 7-card evaluation + LUT
    scores_lut = _load_or_build_lut()  # ensures LUT is ready

    raw = evaluate_7card(hole, board)
    strength = scores_lut.get(raw, None)

    if strength is None:
        # Very rare: raw score not seen in sampling; fall back to
        # nearest neighbor by value (simple linear search).
        # This should almost never trigger if num_samples is large enough.
        # To keep it cheap, just use normalized raw as backup.
        strength = raw / 1e9

    return float(strength)

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
    # print("card1", hole[0], "rank", r1, "suit", s1)
    # print("card2", hole[1], "rank", r2, "suit", s2)

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


