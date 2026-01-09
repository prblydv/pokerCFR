# ---------------------------------------------------------------------------
# File overview:
#   abstraction.py centralizes all poker hand abstraction helpers, including
#   card utilities, Treys-based evaluators, and state encoding.
#   Run instructions: this module is utility-only; import it from training or
#   simulation scripts. There is no standalone CLI entry point.
# ---------------------------------------------------------------------------

from typing import List

import torch
from treys import Evaluator, Card

from config import STACK_SIZE

# ------------------------------------------------------------------
# Treys Evaluator (thread-safe singleton)
# ------------------------------------------------------------------

_EVALUATOR = Evaluator()


# --- Card utilities ----------------------------------------------------------

def card_rank(card: int) -> int:
    """Return rank 2..14 (2..10,J=11,Q=12,K=13,A=14) from 0..51."""
    r = card % 13  # 0..12
    return r + 2


def card_suit(card: int) -> int:
    """Return suit 0..3 from 0..51."""
    return card // 13


def _card_0_51_to_treys(card: int) -> int:
    """Convert card index 0..51 to Treys Card format."""
    rank_idx = card % 13  # 0..12
    suit_idx = card // 13  # 0..3
    rank_chars = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    suit_chars = ['s', 'h', 'd', 'c']
    return Card.new(f"{rank_chars[rank_idx]}{suit_chars[suit_idx]}")


def evaluate_5card(cards: List[int]) -> int:
    """
    Treys-based 5-card hand evaluator.
    Output: integer score (lower is better in Treys, so we negate).
    """
    treys_cards = [_card_0_51_to_treys(c) for c in cards]
    score = _EVALUATOR.evaluate(treys_cards, [])
    return -score


def evaluate_7card(hole: List[int], board: List[int]) -> int:
    """Best 5-card hand from 7 cards using Treys."""
    cards = hole + board
    if len(cards) < 5:
        return 0
    treys_cards = [_card_0_51_to_treys(c) for c in cards]
    score = _EVALUATOR.evaluate(treys_cards, [])
    return -score


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


def _straight_high(cards):
    if len(cards) < 5:
        return 0
    ranks = {card_rank(c) for c in cards}
    if 14 in ranks:
        ranks.add(1)
    best = 0
    for start in range(1, 11):
        window = set(range(start, start + 5))
        if window.issubset(ranks):
            best = max(best, start + 4)
    return best


def _straight_draw_high(cards):
    if len(cards) < 4:
        return 0
    ranks = {card_rank(c) for c in cards}
    if 14 in ranks:
        ranks.add(1)
    best = 0
    for start in range(1, 11):
        window = set(range(start, start + 5))
        if len(window & ranks) >= 4:
            best = max(best, start + 4)
    return best


def _straight_draw_type(cards):
    if len(cards) < 4:
        return "none"
    ranks = {card_rank(c) for c in cards}
    if 14 in ranks:
        ranks.add(1)
    best = "none"
    for start in range(1, 11):
        window = set(range(start, start + 5))
        if len(window & ranks) == 4:
            missing = list(window - ranks)
            if missing and (missing[0] == start or missing[0] == start + 4):
                best = "oesd"
            elif best != "oesd":
                best = "gutshot"
    return best


def _flush_draw_info(hole, board):
    cards = hole + board
    if len(cards) < 4:
        return False, False, None
    suits = [card_suit(c) for c in cards]
    suit_counts = [suits.count(i) for i in range(4)]
    max_suit = max(suit_counts)
    if max_suit < 4:
        return False, False, None
    flush_suit = suit_counts.index(max_suit)
    hero_flush_cards = [c for c in hole if card_suit(c) == flush_suit]
    nut_flush_draw = any(card_rank(c) == 14 for c in hero_flush_cards)
    return True, nut_flush_draw, flush_suit


def _backdoor_flush_draw(hole, board):
    if len(board) != 3:
        return False
    suits = [card_suit(c) for c in hole + board]
    suit_counts = [suits.count(i) for i in range(4)]
    return max(suit_counts) == 3


def _made_hand_flags(hole, board):
    class_str = _hand_class(hole, board)
    if class_str is None:
        return 0.0, 0.0, 0.0, 0.0

    strong_classes = {
        "Two Pair",
        "Three of a Kind",
        "Straight",
        "Flush",
        "Full House",
        "Four of a Kind",
        "Straight Flush",
    }
    if class_str in strong_classes:
        return 1.0, 0.0, 0.0, 1.0

    if class_str == "Pair":
        board_ranks = [card_rank(c) for c in board] if board else []
        top_board = max(board_ranks) if board_ranks else 0
        ranks = [card_rank(c) for c in hole + board]
        pair_rank = None
        for r in set(ranks):
            if ranks.count(r) == 2:
                pair_rank = r
                break
        pocket_pair = card_rank(hole[0]) == card_rank(hole[1]) if len(hole) >= 2 else False
        is_overpair = pocket_pair and pair_rank and pair_rank > top_board
        is_top_pair = pair_rank == top_board if pair_rank else False
        if is_overpair or is_top_pair:
            return 0.0, 1.0, 0.0, 1.0
        return 0.0, 0.0, 1.0, 1.0

    return 0.0, 0.0, 1.0, 0.0


def coarse_strength(hole, board) -> int:
    """
    Discrete strength bucket for scripted policies and reports.
    Returns: 2=strong, 1=medium, 0=weak.
    """
    if len(board) < 3:
        bucket = _preflop_bucket(hole)
        if bucket[0] == 1.0:
            return 2
        if bucket[1] == 1.0:
            return 1
        return 0

    made_strong, made_medium, made_weak, _ = _made_hand_flags(hole, board)
    if made_strong:
        return 2
    if made_medium:
        return 1
    if _has_flush_draw(hole + board) or _has_straight_draw(hole + board):
        return 1
    if made_weak:
        return 0
    return 0

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
      - street-aware strength bucket (3)
      - has_initiative, my_range_capped, opp_range_capped
      - nut/leverage awareness (3)
      - made hand class (4)
      - draw state (3)
      - board texture (3)
      - spr regime (3)
      - nutless polarization (1)
      - per-seat features rotated to hero first (for each seat: stack_norm, contrib_norm, to_call_norm, folded, all_in, acted, to_act_flag)
      - hole_feats (4)
    """
    from poker_env import (
        STREET_PREFLOP, STREET_FLOP, STREET_TURN, STREET_RIVER,
        ACTION_BET_POT_25, ACTION_BET_POT_50, ACTION_BET_POT_100, ACTION_BET_POT_200, ACTION_ALL_IN
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

    # Street-aware strength bucket
    strength_bucket = _street_strength_bucket(state.hole[player], state.board, state.street)

    # Board texture
    _, board_connectedness, board_suitedness = _board_texture(state.board)

    action_seq = getattr(state, "action_seq", []) or []
    last_aggr_actor = None
    last_aggr_size = 0.0
    for actor, act, size_norm in reversed(action_seq):
        if act in (ACTION_BET_POT_25, ACTION_BET_POT_50, ACTION_BET_POT_100, ACTION_BET_POT_200, ACTION_ALL_IN):
            last_aggr_actor = actor
            last_aggr_size = size_norm
            break

    # Range and initiative (simple caps + who last bet)
    has_initiative = 1.0 if last_aggr_actor == player else 0.0
    my_range_capped = 1.0 if (last_aggr_actor not in (None, player) and last_aggr_size >= 0.5) else 0.0
    opp_range_capped = 1.0 if (last_aggr_actor == player and last_aggr_size >= 0.5) else 0.0

    # Nut / leverage awareness
    cards = state.hole[player] + state.board
    straight_high = _straight_high(cards)
    straight_draw_high = _straight_draw_high(cards)
    flush_draw, nut_flush_draw, flush_suit = _flush_draw_info(state.hole[player], state.board)
    class_str = _hand_class(state.hole[player], state.board)
    has_nuts = 0.0
    if class_str in {"Straight Flush", "Four of a Kind"}:
        has_nuts = 1.0
    elif class_str == "Flush" and nut_flush_draw:
        has_nuts = 1.0
    elif class_str == "Straight" and straight_high == 14:
        has_nuts = 1.0
    has_nut_draw = 1.0 if (nut_flush_draw or straight_draw_high == 14) else 0.0
    has_nut_blocker = 0.0
    if state.board and flush_suit is not None:
        hero_flush_cards = [c for c in state.hole[player] if card_suit(c) == flush_suit]
        if any(card_rank(c) == 14 for c in hero_flush_cards):
            has_nut_blocker = 1.0

    # Made hand class (exactly one postflop)
    if len(state.board) >= 3:
        made_strong, made_medium, made_weak, showdown_value = _made_hand_flags(state.hole[player], state.board)
    else:
        made_strong, made_medium, made_weak, showdown_value = 0.0, 0.0, 1.0, 0.0

    # Draw state
    draw_type = _straight_draw_type(cards)
    strong_draw = 1.0 if (draw_type == "oesd" or nut_flush_draw or (flush_draw and draw_type != "none")) else 0.0
    weak_draw = 1.0 if (draw_type == "gutshot" or _backdoor_flush_draw(state.hole[player], state.board)) else 0.0
    reverse_io_draw = 1.0 if (flush_draw and not nut_flush_draw) else 0.0

    # Board texture one-hot
    board_dry = 1.0 if (board_connectedness < 0.35 and board_suitedness < 0.5) else 0.0
    board_wet = 1.0 if (board_connectedness >= 0.6 or board_suitedness >= 0.6) else 0.0
    board_semi = 1.0 if (board_dry == 0.0 and board_wet == 0.0) else 0.0

    # SPR regime
    spr_low = 1.0 if spr_raw < 3.0 else 0.0
    spr_mid = 1.0 if 3.0 <= spr_raw <= 8.0 else 0.0
    spr_high = 1.0 if spr_raw > 8.0 else 0.0

    # Nutless polarization safety
    pot_base = max(1.0, state.pot)
    large_bet_available = state.stacks[player] >= pot_base
    nutless_polarization = 1.0 if (large_bet_available and not (has_nuts or has_nut_draw or has_nut_blocker)) else 0.0

    # Private hole-card identity features
    hole_feats = encode_hole_cards(state.hole[player])

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
        ]
        + strength_bucket
        + [
            has_initiative,
            my_range_capped,
            opp_range_capped,
        ]
        + [
            has_nuts,
            has_nut_draw,
            has_nut_blocker,
            made_strong,
            made_medium,
            made_weak,
            showdown_value,
            strong_draw,
            weak_draw,
            reverse_io_draw,
            board_dry,
            board_semi,
            board_wet,
            spr_low,
            spr_mid,
            spr_high,
            nutless_polarization,
        ]
        + per_seat
        + hole_feats
    )

    return torch.tensor(vec, dtype=torch.float32)
