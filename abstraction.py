# ---------------------------------------------------------------------------
# File overview:
#   abstraction.py centralizes all poker hand abstraction helpers, including
#   card utilities, Treys-based evaluators, LUT generation, and state encoding.
#   Run instructions: this module is utility-only; import it from training or
#   simulation scripts. There is no standalone CLI entry point.
# ---------------------------------------------------------------------------

import itertools
import os
import time
import random
import pickle
from typing import List

import torch
from treys import Evaluator, Card

from config import STACK_SIZE
# ------------------------------------------------------------------
# Treys Evaluator (thread-safe singleton)
# ------------------------------------------------------------------

_EVALUATOR = Evaluator()

_LUT_PATH = "hand_strength_lut.pkl"
_SCORE_TO_PCTL = None  # filled by _load_or_build_lut()


# --- Card utilities ----------------------------------------------------------

# Function metadata:
#   Inputs: card (int) -> single card encoded 0..51  # dtype=int
#   Sample: card_sample = 51  # dtype=int (Ace of clubs)
#           output_sample = card_rank(card_sample)  # 14  # dtype=int
def card_rank(card: int) -> int:
    """Return rank 2..14 (2..10,J=11,Q=12,K=13,A=14) from 0..51."""
    r = card % 13  # 0..12
    return r + 2

# Function metadata:
#   Inputs: card (int) encoded 0..51  # dtype=int
#   Sample: card_sample = 27  # dtype=int (4 of diamonds)
#           output_sample = card_suit(card_sample)  # 2  # dtype=int
def card_suit(card: int) -> int:
    """Return suit 0..3 from 0..51."""
    return card // 13

# Function metadata:
#   Inputs: card (int) encoded 0..51  # dtype=int
#   Sample:
#       card_sample = 0  # dtype=int (2 of spades)
#       output_sample = _card_0_51_to_treys(card_sample)  # dtype=int Treys code e.g. 268442665
def _card_0_51_to_treys(card: int) -> int:
    """Convert card index 0..51 to Treys Card format.
    
    Card encoding 0..51:
      0-12: 2-A of spades
      13-25: 2-A of hearts
      26-38: 2-A of diamonds
      39-51: 2-A of clubs
    """
    rank_idx = card % 13  # 0..12
    suit_idx = card // 13  # 0..3
    
    # Treys uses character codes: 2,3,4,5,6,7,8,9,T,J,Q,K,A
    rank_chars = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    suit_chars = ['s', 'h', 'd', 'c']  # 0..3
    
    return Card.new(f"{rank_chars[rank_idx]}{suit_chars[suit_idx]}")


# --- 5-card evaluator using Treys -----------------------------------------------

# Function metadata:
#   Inputs: cards (List[int]) length 5 of unique card ids 0..51  # dtype=List[int]
#   Sample:
#       cards_sample = [0, 5, 13, 22, 44]
#       output_sample = evaluate_5card(cards_sample)  # e.g. -7462  # dtype=int
def evaluate_5card(cards: List[int]) -> int:
    """
    Treys-based 5-card hand evaluator.
    Input:  list of 5 cards (0..51).
    Output: integer score (lower is better in Treys, so we negate).
    """
    treys_cards = [_card_0_51_to_treys(c) for c in cards]
    score = _EVALUATOR.evaluate(treys_cards, [])
    # Treys returns lower score for better hands, negate for consistency
    return -score


# --- 7-card evaluation using Treys -----------------------------------------------
# Function metadata:
#   Inputs: hole (List[int]) length 2, board (List[int]) length 0-5  # dtype=List[int]
#   Sample:
#       hole_sample = [12, 25]; board_sample = [1, 14, 27, 40, 48]
#       output_sample = evaluate_7card(hole_sample, board_sample)  # e.g. -8123  # dtype=int
def evaluate_7card(hole: List[int], board: List[int]) -> int:
    """
    Best 5-card hand from 7 cards using Treys.
    """
    cards = hole + board
    n = len(cards)
    if n < 5:
        return 0

    treys_cards = [_card_0_51_to_treys(c) for c in cards]
    
    if n == 5:
        score = _EVALUATOR.evaluate(treys_cards, [])
    else:
        # For 6 or 7 cards, Treys handles best 5-card selection
        score = _EVALUATOR.evaluate(treys_cards, [])
    
    # Negate for consistency (lower is better in Treys)
    return -score



# --- Strength abstraction ----------------------------------------------------
# Function metadata:
#   Inputs: num_samples (int), batch_size (int)  # dtype=int
#   Sample:
#       output_sample = _build_strength_lut(num_samples=10_000, batch_size=1_000)
#       # dtype=dict mapping score (int) -> percentile (float)
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
# Function metadata:
#   Inputs: None
#   Sample:
#       output_sample = _load_or_build_lut()  # dtype=dict[int, float]
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

# Function metadata:
#   Inputs: hole (List[int]), board (List[int]) # dtype=List[int]
#   Sample:
#       hole_sample = [10, 23]; board_sample = [5, 18, 31]
#       output_sample = normalized_strength(hole_sample, board_sample)  # 0.73 float
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

# Function metadata:
#   Inputs: hole (List[int]) length 2  # dtype=List[int]
#   Sample:
#       hole_sample = [0, 13]
#       output_sample = encode_hole_cards(hole_sample)  # [0.0, 0.0, 0.0, 0.0]  # dtype=List[float]
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

# Function metadata:
#   Inputs: state (poker_env.PokerState-like object), player (int) # dtype=(object,int)
#   Sample:
#       state_sample = mock_state  # dtype=PokerState
#       player_sample = 0
#       output_sample = encode_state(state_sample, player_sample)
#       # dtype=torch.FloatTensor shape (13,)
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


