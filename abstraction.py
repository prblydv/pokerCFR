# # ---------------------------------------------------------------------------
# # File overview:
# #   abstraction.py centralizes all poker hand abstraction helpers, including
# #   card utilities, Treys-based evaluators, LUT generation, and state encoding.
# #   Run instructions: this module is utility-only; import it from training or
# #   simulation scripts. There is no standalone CLI entry point.
# # ---------------------------------------------------------------------------

# import itertools
# import os
# import time
# import random
# import pickle
# from typing import List

# import torch
# from treys import Evaluator, Card

# from config import STACK_SIZE
# # ------------------------------------------------------------------
# # Treys Evaluator (thread-safe singleton)
# # ------------------------------------------------------------------

# _EVALUATOR = Evaluator()

# _LUT_PATH = "hand_strength_lut.pkl"
# _SCORE_TO_PCTL = None  # filled by _load_or_build_lut()


# # --- Card utilities ----------------------------------------------------------

# # Function metadata:
# #   Inputs: card (int) -> single card encoded 0..51  # dtype=int
# #   Sample: card_sample = 51  # dtype=int (Ace of clubs)
# #           output_sample = card_rank(card_sample)  # 14  # dtype=int
# def card_rank(card: int) -> int:
#     """Return rank 2..14 (2..10,J=11,Q=12,K=13,A=14) from 0..51."""
#     r = card % 13  # 0..12
#     return r + 2

# # Function metadata:
# #   Inputs: card (int) encoded 0..51  # dtype=int
# #   Sample: card_sample = 27  # dtype=int (4 of diamonds)
# #           output_sample = card_suit(card_sample)  # 2  # dtype=int
# def card_suit(card: int) -> int:
#     """Return suit 0..3 from 0..51."""
#     return card // 13

# # Function metadata:
# #   Inputs: card (int) encoded 0..51  # dtype=int
# #   Sample:
# #       card_sample = 0  # dtype=int (2 of spades)
# #       output_sample = _card_0_51_to_treys(card_sample)  # dtype=int Treys code e.g. 268442665
# def _card_0_51_to_treys(card: int) -> int:
#     """Convert card index 0..51 to Treys Card format.
    
#     Card encoding 0..51:
#       0-12: 2-A of spades
#       13-25: 2-A of hearts
#       26-38: 2-A of diamonds
#       39-51: 2-A of clubs
#     """
#     rank_idx = card % 13  # 0..12
#     suit_idx = card // 13  # 0..3
    
#     # Treys uses character codes: 2,3,4,5,6,7,8,9,T,J,Q,K,A
#     rank_chars = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
#     suit_chars = ['s', 'h', 'd', 'c']  # 0..3
    
#     return Card.new(f"{rank_chars[rank_idx]}{suit_chars[suit_idx]}")


# # --- 5-card evaluator using Treys -----------------------------------------------

# # Function metadata:
# #   Inputs: cards (List[int]) length 5 of unique card ids 0..51  # dtype=List[int]
# #   Sample:
# #       cards_sample = [0, 5, 13, 22, 44]
# #       output_sample = evaluate_5card(cards_sample)  # e.g. -7462  # dtype=int
# def evaluate_5card(cards: List[int]) -> int:
#     """
#     Treys-based 5-card hand evaluator.
#     Input:  list of 5 cards (0..51).
#     Output: integer score (lower is better in Treys, so we negate).
#     """
#     treys_cards = [_card_0_51_to_treys(c) for c in cards]
#     score = _EVALUATOR.evaluate(treys_cards, [])
#     # Treys returns lower score for better hands, negate for consistency
#     return -score


# # --- 7-card evaluation using Treys -----------------------------------------------
# # Function metadata:
# #   Inputs: hole (List[int]) length 2, board (List[int]) length 0-5  # dtype=List[int]
# #   Sample:
# #       hole_sample = [12, 25]; board_sample = [1, 14, 27, 40, 48]
# #       output_sample = evaluate_7card(hole_sample, board_sample)  # e.g. -8123  # dtype=int
# def evaluate_7card(hole: List[int], board: List[int]) -> int:
#     """
#     Best 5-card hand from 7 cards using Treys.
#     """
#     cards = hole + board
#     n = len(cards)
#     if n < 5:
#         return 0

#     treys_cards = [_card_0_51_to_treys(c) for c in cards]
    
#     if n == 5:
#         score = _EVALUATOR.evaluate(treys_cards, [])
#     else:
#         # For 6 or 7 cards, Treys handles best 5-card selection
#         score = _EVALUATOR.evaluate(treys_cards, [])
    
#     # Negate for consistency (lower is better in Treys)
#     return -score



# # --- Strength abstraction ----------------------------------------------------
# # Function metadata:
# #   Inputs: num_samples (int), batch_size (int)  # dtype=int
# #   Sample:
# #       output_sample = _build_strength_lut(num_samples=10_000, batch_size=1_000)
# #       # dtype=dict mapping score (int) -> percentile (float)
# def _build_strength_lut(num_samples: int = 500_000, batch_size: int = 5_000) -> dict:
#     """
#     Build a LUT mapping evaluate_7card() scores to percentiles in [0,1].
#     Uses random 7-card samples.
#     Shows progress + rough ETA.
#     """
#     print(f"[LUT] Building hand-strength LUT with {num_samples} samples...")
#     start_time = time.time()
#     scores = []

#     cards_all = list(range(52))

#     for i in range(0, num_samples, batch_size):
#         batch_end = min(num_samples, i + batch_size)
#         for _ in range(i, batch_end):
#             # sample 7 distinct cards
#             seven = random.sample(cards_all, 7)
#             v = evaluate_7card(seven[:2], seven[2:])  # hole=2, board=5
#             scores.append(v)

#         # progress + ETA
#         done = batch_end
#         elapsed = time.time() - start_time
#         frac = done / num_samples
#         rate = done / elapsed if elapsed > 0 else 0.0
#         remaining = (num_samples - done) / rate if rate > 0 else 0.0

#         print(
#             f"[LUT] {done}/{num_samples} "
#             f"({frac:6.2%}) "
#             f"elapsed={elapsed:6.1f}s "
#             f"ETA={remaining:6.1f}s",
#             end="\r",
#             flush=True,
#         )

#     print()  # newline after progress

#     # build percentile mapping
#     unique_scores = sorted(set(scores))
#     n = len(unique_scores)
#     print(f"[LUT] Unique scores: {n}")

#     score_to_pctl = {}
#     if n == 1:
#         score_to_pctl[unique_scores[0]] = 0.5
#     else:
#         for idx, s in enumerate(unique_scores):
#             score_to_pctl[s] = idx / (n - 1)

#     # save to disk
#     with open(_LUT_PATH, "wb") as f:
#         pickle.dump(score_to_pctl, f)

#     total_time = time.time() - start_time
#     print(f"[LUT] Built and saved LUT to '{_LUT_PATH}' in {total_time:.1f}s")

#     return score_to_pctl
# # Function metadata:
# #   Inputs: None
# #   Sample:
# #       output_sample = _load_or_build_lut()  # dtype=dict[int, float]
# def _load_or_build_lut() -> dict:
#     """
#     Load LUT from disk if present; otherwise build it.
#     """
#     global _SCORE_TO_PCTL

#     if _SCORE_TO_PCTL is not None:
#         return _SCORE_TO_PCTL

#     if os.path.exists(_LUT_PATH):
#         print(f"[LUT] Loading hand-strength LUT from '{_LUT_PATH}'...")
#         with open(_LUT_PATH, "rb") as f:
#             _SCORE_TO_PCTL = pickle.load(f)
#         print(f"[LUT] Loaded {len(_SCORE_TO_PCTL)} entries.")
#     else:
#         _SCORE_TO_PCTL = _build_strength_lut()

#     return _SCORE_TO_PCTL


# # initialize on import
# _load_or_build_lut()

# # --- Strength estimator ------------------------------------------------------

# # Function metadata:
# #   Inputs: hole (List[int]), board (List[int]) # dtype=List[int]
# #   Sample:
# #       hole_sample = [10, 23]; board_sample = [5, 18, 31]
# #       output_sample = normalized_strength(hole_sample, board_sample)  # 0.73 float
# def normalized_strength(hole: List[int], board: List[int]) -> float:
#     """
#     Returns a continuous strength estimator in [0,1].

#     Preflop (0-2 cards on board): simple hole-card rank heuristic.
#     Flop/turn/river: evaluate best 5-card hand out of 7, then map
#     its raw score through the LUT to a [0,1] percentile.
#     """
#     # Preflop heuristic stays cheap and simple
#     if len(board) < 3:
#         if len(hole) < 2:
#             return 0.5  # neutral if we somehow don't have 2 cards yet
#         r = sorted([card_rank(c) for c in hole], reverse=True)
#         return (r[0] + r[1]) / (2 * 14.0)  # avg rank / max_rank

#     # Postflop: use 7-card evaluation + LUT
#     scores_lut = _load_or_build_lut()  # ensures LUT is ready

#     raw = evaluate_7card(hole, board)
#     strength = scores_lut.get(raw, None)

#     if strength is None:
#         # Very rare: raw score not seen in sampling; fall back to
#         # nearest neighbor by value (simple linear search).
#         # This should almost never trigger if num_samples is large enough.
#         # To keep it cheap, just use normalized raw as backup.
#         strength = raw / 1e9

#     return float(strength)

# # --- Hole-card feature encoding ---------------------------------------------

# # Function metadata:
# #   Inputs: hole (List[int]) length 2  # dtype=List[int]
# #   Sample:
# #       hole_sample = [0, 13]
# #       output_sample = encode_hole_cards(hole_sample)  # [0.0, 0.0, 0.0, 0.0]  # dtype=List[float]
# def encode_hole_cards(hole: List[int]) -> List[float]:
#     """
#     Encode the private 2-card hand in a simple, information-rich way.

#     Features:
#       - hi_rank_norm, lo_rank_norm in [0,1]
#       - suited flag (0/1)
#       - pair flag (0/1)
#     """
#     assert len(hole) == 2
#     r1 = card_rank(hole[0]) - 2  # 0..12
#     r2 = card_rank(hole[1]) - 2
#     s1 = hole[0] // 13           # 0..3
#     s2 = hole[1] // 13
#     # print("card1", hole[0], "rank", r1, "suit", s1)
#     # print("card2", hole[1], "rank", r2, "suit", s2)

#     # order hi, lo
#     if r2 > r1:
#         r1, r2 = r2, r1
#         s1, s2 = s2, s1

#     hi_rank_norm = r1 / 12.0
#     lo_rank_norm = r2 / 12.0
#     suited = 1.0 if s1 == s2 else 0.0
#     pair = 1.0 if r1 == r2 else 0.0

#     return [hi_rank_norm, lo_rank_norm, suited, pair]


# # --- State encoding ----------------------------------------------------------

# # Function metadata:
# #   Inputs: state (poker_env.PokerState-like object), player (int) # dtype=(object,int)
# #   Sample:
# #       state_sample = mock_state  # dtype=PokerState
# #       player_sample = 0
# #       output_sample = encode_state(state_sample, player_sample)
# #       # dtype=torch.FloatTensor shape (13,)
# def encode_state(state, player: int) -> torch.Tensor:
#     """
#     Encode public + private information for Deep CFR.

#     Layout (variable length with num_players=N):
#       - street_one_hot (4)
#       - hero_position_one_hot (N, 0 = button, 1 = SB for N>2, etc.)
#       - acting_player_index_norm
#       - pot_norm, current_bet_norm, to_call_norm(hero), pot_after_call_norm(hero), spr(hero)
#       - last_aggressor_present, last_aggressor_rel_norm
#       - hero_is_sb, hero_is_bb
#       - hand_strength, board_strength
#       - per-seat features rotated to hero first (for each seat: stack_norm, contrib_norm, to_call_norm, folded, all_in, acted, to_act_flag)
#       - hole_feats (4)
#       - action_seq encoding (ACTION_SEQ_LEN * 10)
#     """
#     from poker_env import (
#         STREET_PREFLOP, STREET_FLOP, STREET_TURN, STREET_RIVER,
#         ACTION_FOLD, ACTION_CHECK, ACTION_CALL, ACTION_BET_POT_25, ACTION_BET_POT_50, ACTION_BET_POT_100, ACTION_BET_POT_200, ACTION_ALL_IN, ACTION_SEQ_LEN
#     )

#     # Derive reference stack size per hand/session so normalization stays valid
#     # when playing with variable or drifting stacks.
#     ref_stack = STACK_SIZE
#     if hasattr(state, "initial_stacks") and state.initial_stacks:
#         try:
#             ref_stack = max(max(state.initial_stacks), 1e-6)
#         except Exception:
#             ref_stack = STACK_SIZE

#     # Street one-hot
#     street_oh = [0.0, 0.0, 0.0, 0.0]
#     street_oh[state.street] = 1.0

#     num_players = getattr(state, "num_players", len(state.stacks))
#     button = getattr(state, "button_player", getattr(state, "sb_player", 0))
#     hero_pos = (player - button) % num_players
#     hero_pos_oh = [1.0 if i == hero_pos else 0.0 for i in range(num_players)]

#     # Public scalars
#     pot_norm = state.pot / (ref_stack * max(2, num_players))
#     curr_bet_norm = state.current_bet / ref_stack
#     to_call = max(0.0, state.current_bet - state.contrib[player])
#     to_call_norm = to_call / ref_stack
#     pot_after_call_norm = (state.pot + to_call) / (ref_stack * max(2, num_players))
#     # Stack-to-pot ratio: critical for bet sizing / bluffing logic
#     spr = state.stacks[player] / max(1.0, state.pot)
#     hero_is_sb = 1.0 if getattr(state, "sb_player", -1) == player else 0.0
#     hero_is_bb = 1.0 if getattr(state, "bb_player", -1) == player else 0.0
#     last_agg_present = 0.0 if state.last_aggressor is None or state.last_aggressor < 0 else 1.0
#     last_agg_rel = 0.0
#     if last_agg_present > 0.0:
#         last_agg_rel = ((player - state.last_aggressor) % num_players) / max(1.0, num_players - 1)
#     # Strength estimates
#     hand_str = normalized_strength(state.hole[player], state.board)
#     board_str = normalized_strength([], state.board) if state.board else 0.0

#     # Private hole-card identity features
#     hole_feats = encode_hole_cards(state.hole[player])

#     def encode_action_sequence():
#         vec_seq = []
#         seq = getattr(state, "action_seq", []) or []
#         k = ACTION_SEQ_LEN
#         for i in range(k):
#             if i < len(seq):
#                 actor, act, size_norm = seq[-1 - i]  # latest first
#                 actor_offset = ((actor - player) % num_players) / max(1.0, num_players - 1)
#                 is_fold = 1.0 if act == ACTION_FOLD else 0.0
#                 is_check = 1.0 if act == ACTION_CHECK else 0.0
#                 is_call = 1.0 if act == ACTION_CALL else 0.0
#                 is_rsmall = 1.0 if act == ACTION_BET_POT_50 else 0.0
#                 is_rbig = 1.0 if act in (ACTION_BET_POT_100, ACTION_ALL_IN) else 0.0
#                 vec_seq.extend([
#                     actor_offset,
#                     is_fold,
#                     is_check,
#                     is_call,
#                     is_rsmall,
#                     is_rbig,
#                     float(size_norm),
#                 ])
#             else:
#                 vec_seq.extend([0.0] * 7)
#         return vec_seq

#     per_seat = []
#     order = [(player + i) % num_players for i in range(num_players)]
#     acted = getattr(state, "players_acted", [False] * num_players)
#     for pid in order:
#         per_seat.extend([
#             state.stacks[pid] / ref_stack,
#             state.contrib[pid] / ref_stack,
#             max(0.0, state.current_bet - state.contrib[pid]) / ref_stack,
#             1.0 if getattr(state, "folded", [False] * num_players)[pid] else 0.0,
#             1.0 if state.stacks[pid] <= 0 else 0.0,
#             1.0 if acted[pid] else 0.0,
#             1.0 if state.to_act == pid else 0.0,
#         ])

#     vec = (
#         street_oh
#         + hero_pos_oh
#         + [
#             float(player) / max(1.0, num_players - 1),
#             pot_norm,
#             to_call_norm,
#             pot_after_call_norm,
#             curr_bet_norm,
#             spr,
#             hero_is_sb,
#             hero_is_bb,
#             last_agg_present,
#             last_agg_rel,
#             hand_str,
#             board_str,
#         ]
#         + per_seat
#         + hole_feats
#         + encode_action_sequence()
#     )

#     return torch.tensor(vec, dtype=torch.float32)
























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




def _board_texture(board):
    if not board:
        return 0.0, 0.0, 0.0
    ranks = sorted({card_rank(c) for c in board})
    highness = max(ranks) / 14.0 if ranks else 0.0
    ranks_set = set(ranks)
    if 14 in ranks_set:
        ranks_set.add(1)
    max_run = 1
    for start in range(1, 11):
        run = 0
        for r in range(start, start + 5):
            if r in ranks_set:
                run += 1
            else:
                break
        if run > max_run:
            max_run = run
    connectedness = max(0.0, min(1.0, (max_run - 1) / 4.0))
    suits = [card_suit(c) for c in board]
    suit_counts = [suits.count(i) for i in range(4)]
    suitedness = max(suit_counts) / len(board)
    return highness, connectedness, suitedness


def _has_flush_draw(cards):
    if len(cards) < 4:
        return False
    suits = [card_suit(c) for c in cards]
    return max(suits.count(i) for i in range(4)) >= 4


def _has_straight_draw(cards):
    if len(cards) < 4:
        return False
    ranks = {card_rank(c) for c in cards}
    if 14 in ranks:
        ranks.add(1)
    for start in range(1, 11):
        window = set(range(start, start + 5))
        if len(window & ranks) >= 4:
            return True
    return False


def _hand_class(hole, board):
    cards = hole + board
    if len(cards) < 5:
        return None
    treys_cards = [_card_0_51_to_treys(c) for c in cards]
    score = _EVALUATOR.evaluate(treys_cards, [])
    class_id = _EVALUATOR.get_rank_class(score)
    return _EVALUATOR.class_to_string(class_id)


def _preflop_bucket(hole):
    if len(hole) < 2:
        return [0.0, 1.0, 0.0]
    r1 = card_rank(hole[0])
    r2 = card_rank(hole[1])
    suited = card_suit(hole[0]) == card_suit(hole[1])
    high = max(r1, r2)
    low = min(r1, r2)
    pair = r1 == r2
    gap = abs(r1 - r2)

    if (pair and high >= 12) or (high >= 14 and low >= 11) or (suited and high >= 13 and low >= 10):
        return [1.0, 0.0, 0.0]
    if (pair and high >= 7) or (high >= 12 and low >= 9) or (suited and high >= 11 and low >= 8) or (gap == 1 and high >= 9):
        return [0.0, 1.0, 0.0]
    return [0.0, 0.0, 1.0]


def _flop_bucket(hole, board):
    class_str = _hand_class(hole, board)
    made_classes = {
        "Pair",
        "Two Pair",
        "Three of a Kind",
        "Straight",
        "Flush",
        "Full House",
        "Four of a Kind",
        "Straight Flush",
    }
    if class_str in made_classes:
        return [0.0, 0.0, 1.0]
    cards = hole + board
    if _has_flush_draw(cards) or _has_straight_draw(cards):
        return [0.0, 1.0, 0.0]
    return [1.0, 0.0, 0.0]


def _turn_river_bucket(hole, board):
    class_str = _hand_class(hole, board)
    if class_str is None or class_str in {"High Card", "Pair"}:
        return [1.0, 0.0, 0.0]
    if class_str in {"Two Pair", "Three of a Kind"}:
        return [0.0, 1.0, 0.0]
    return [0.0, 0.0, 1.0]


def _street_strength_bucket(hole, board, street):
    if street == 0:
        return _preflop_bucket(hole)
    if street == 1:
        return _flop_bucket(hole, board)
    return _turn_river_bucket(hole, board)

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

    Layout (variable length with num_players=N):
      - street_one_hot (4)
      - hero_position_one_hot (N, 0 = button, 1 = SB for N>2, etc.)
      - pot_norm, current_bet_norm, to_call_norm(hero), pot_after_call_norm(hero), spr(hero)
      - last_aggressor_present, last_aggressor_rel_norm
      - hero_is_sb, hero_is_bb
      - board_strength (public only)
      - street-aware strength bucket (3)
      - range_advantage, hero_polarized, villain_polarized
      - fold_equity_proxy, pot_pressure
      - blocker flags (top pair, nut flush, high card on flush board)
      - line_consistency
      - per-seat features rotated to hero first (for each seat: stack_norm, contrib_norm, to_call_norm, folded, all_in, acted, to_act_flag)
      - hole_feats (4)
      - action_seq encoding (ACTION_SEQ_LEN * 10)
    """
    from poker_env import (
        STREET_PREFLOP, STREET_FLOP, STREET_TURN, STREET_RIVER,
        ACTION_FOLD, ACTION_CHECK, ACTION_CALL, ACTION_BET_POT_25, ACTION_BET_POT_50, ACTION_BET_POT_100, ACTION_BET_POT_200, ACTION_ALL_IN, ACTION_SEQ_LEN
    )

    # Derive reference stack size per hand/session so normalization stays valid
    # when playing with variable or drifting stacks.
    ref_stack = STACK_SIZE
    if hasattr(state, "initial_stacks") and state.initial_stacks:
        try:
            ref_stack = max(max(state.initial_stacks), 1e-6)
        except Exception:
            ref_stack = STACK_SIZE

    # Street one-hot
    street_oh = [0.0, 0.0, 0.0, 0.0]
    street_oh[state.street] = 1.0

    num_players = getattr(state, "num_players", len(state.stacks))
    button = getattr(state, "button_player", getattr(state, "sb_player", 0))
    hero_pos = (player - button) % num_players
    hero_pos_oh = [1.0 if i == hero_pos else 0.0 for i in range(num_players)]

    # Public scalars
    pot_norm = state.pot / (ref_stack * max(2, num_players))
    curr_bet_norm = state.current_bet / ref_stack
    to_call = max(0.0, state.current_bet - state.contrib[player])
    to_call_norm = to_call / ref_stack
    pot_after_call_norm = (state.pot + to_call) / (ref_stack * max(2, num_players))
    spr_raw = state.stacks[player] / max(1.0, state.pot)
    spr = min(spr_raw, 20.0) / 20.0
    hero_is_sb = 1.0 if getattr(state, "sb_player", -1) == player else 0.0
    hero_is_bb = 1.0 if getattr(state, "bb_player", -1) == player else 0.0
    last_agg_present = 0.0 if state.last_aggressor is None or state.last_aggressor < 0 else 1.0
    last_agg_rel = 0.0
    if last_agg_present > 0.0:
        last_agg_rel = ((player - state.last_aggressor) % num_players) / max(1.0, num_players - 1)

    # Public-only board strength
    board_str = normalized_strength([], state.board) if state.board else 0.0

    # Street-aware strength bucket
    strength_bucket = _street_strength_bucket(state.hole[player], state.board, state.street)

    # Range advantage proxy
    board_highness, board_connectedness, board_suitedness = _board_texture(state.board)
    dryness = 0.5 * (1.0 - board_connectedness) + 0.5 * (1.0 - board_suitedness)
    base_adv = 0.6 * board_highness + 0.4 * dryness

    action_seq = getattr(state, "action_seq", []) or []
    last_aggr_actor = None
    last_aggr_size = 0.0
    for actor, act, size_norm in reversed(action_seq):
        if act in (ACTION_BET_POT_25, ACTION_BET_POT_50, ACTION_BET_POT_100, ACTION_BET_POT_200, ACTION_ALL_IN):
            last_aggr_actor = actor
            last_aggr_size = size_norm
            break

    if last_aggr_actor is None:
        range_advantage = 0.5
    elif last_aggr_actor == player:
        range_advantage = base_adv
    else:
        range_advantage = 1.0 - base_adv
    range_advantage = max(0.0, min(1.0, range_advantage))

    spr_low = spr_raw <= 3.0
    hero_polarized = 1.0 if (last_aggr_actor == player and last_aggr_size >= 0.6 and spr_low) else 0.0
    villain_polarized = 1.0 if (last_aggr_actor not in (None, player) and last_aggr_size >= 0.6 and spr_low) else 0.0

    def pick_primary_villain():
        if state.to_act is not None and state.to_act >= 0 and state.to_act != player:
            return state.to_act
        for i in range(1, num_players + 1):
            idx = (player + i) % num_players
            if not state.folded[idx] and state.stacks[idx] > 0:
                return idx
        return None

    villain = pick_primary_villain()
    if villain is None:
        fold_equity_proxy = 0.0
        pot_pressure = 0.0
    else:
        villain_to_call = max(0.0, state.current_bet - state.contrib[villain])
        villain_stack = max(1e-6, state.stacks[villain])
        fold_equity_proxy = min(villain_to_call / villain_stack, 1.0)
        pot_pressure = min(villain_to_call / max(1.0, state.pot), 4.0) / 4.0

    # Blocker flags
    hero_blocks_top_pair = 0.0
    hero_blocks_nut_flush = 0.0
    hero_has_high_card_on_flush_board = 0.0
    if state.board:
        board_ranks = [card_rank(c) for c in state.board]
        top_rank = max(board_ranks)
        hero_ranks = [card_rank(c) for c in state.hole[player]]
        if top_rank in hero_ranks:
            hero_blocks_top_pair = 1.0
        suits = [card_suit(c) for c in state.board]
        suit_counts = [suits.count(i) for i in range(4)]
        max_suit = max(suit_counts)
        if max_suit >= 3:
            flush_suit = suit_counts.index(max_suit)
            hero_flush_cards = [c for c in state.hole[player] if card_suit(c) == flush_suit]
            if hero_flush_cards:
                hero_flush_ranks = [card_rank(c) for c in hero_flush_cards]
                board_flush_ranks = [card_rank(c) for c in state.board if card_suit(c) == flush_suit]
                if 14 in hero_flush_ranks:
                    hero_blocks_nut_flush = 1.0
                if board_flush_ranks and max(hero_flush_ranks) >= max(board_flush_ranks):
                    hero_has_high_card_on_flush_board = 1.0

    # Line consistency (aggressive actions / opportunities)
    opps = 0
    aggr = 0
    for actor, act, _ in action_seq:
        if actor == player:
            opps += 1
            if act in (ACTION_BET_POT_25, ACTION_BET_POT_50, ACTION_BET_POT_100, ACTION_BET_POT_200, ACTION_ALL_IN):
                aggr += 1
    line_consistency = aggr / opps if opps > 0 else 0.0

    # Private hole-card identity features
    hole_feats = encode_hole_cards(state.hole[player])

    def encode_action_sequence():
        vec_seq = []
        seq = getattr(state, "action_seq", []) or []
        k = ACTION_SEQ_LEN
        for i in range(k):
            if i < len(seq):
                actor, act, size_norm = seq[-1 - i]  # latest first
                actor_offset = ((actor - player) % num_players) / max(1.0, num_players - 1)
                is_fold = 1.0 if act == ACTION_FOLD else 0.0
                is_check = 1.0 if act == ACTION_CHECK else 0.0
                is_call = 1.0 if act == ACTION_CALL else 0.0
                is_b25 = 1.0 if act == ACTION_BET_POT_25 else 0.0
                is_b50 = 1.0 if act == ACTION_BET_POT_50 else 0.0
                is_b100 = 1.0 if act == ACTION_BET_POT_100 else 0.0
                is_b200 = 1.0 if act == ACTION_BET_POT_200 else 0.0
                is_all_in = 1.0 if act == ACTION_ALL_IN else 0.0
                vec_seq.extend([
                    actor_offset,
                    is_fold,
                    is_check,
                    is_call,
                    is_b25,
                    is_b50,
                    is_b100,
                    is_b200,
                    is_all_in,
                    float(size_norm),
                ])
            else:
                vec_seq.extend([0.0] * 10)
        return vec_seq

    per_seat = []
    order = [(player + i) % num_players for i in range(num_players)]
    acted = getattr(state, "players_acted", [False] * num_players)
    for pid in order:
        per_seat.extend([
            state.stacks[pid] / ref_stack,
            state.contrib[pid] / ref_stack,
            max(0.0, state.current_bet - state.contrib[pid]) / ref_stack,
            1.0 if getattr(state, "folded", [False] * num_players)[pid] else 0.0,
            1.0 if state.stacks[pid] <= 0 else 0.0,
            1.0 if acted[pid] else 0.0,
            1.0 if state.to_act == pid else 0.0,
        ])

    vec = (
        street_oh
        + hero_pos_oh
        + [
            pot_norm,
            to_call_norm,
            pot_after_call_norm,
            curr_bet_norm,
            spr,
            hero_is_sb,
            hero_is_bb,
            last_agg_present,
            last_agg_rel,
            board_str,
        ]
        + strength_bucket
        + [
            range_advantage,
            hero_polarized,
            villain_polarized,
            fold_equity_proxy,
            pot_pressure,
            hero_blocks_top_pair,
            hero_blocks_nut_flush,
            hero_has_high_card_on_flush_board,
            line_consistency,
        ]
        + per_seat
        + hole_feats
        + encode_action_sequence()
    )

    return torch.tensor(vec, dtype=torch.float32)
