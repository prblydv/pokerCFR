# traversal_only.py
# Worker-safe CFR traversal. NO abstraction import.
# Produces 16-dim state vectors matching abstraction.encode_state layout.

import random
from typing import List

import torch
from config import STACK_SIZE
from poker_env import NUM_ACTIONS


# --- Card utils (copy from abstraction, LUT-free) ---
_RANK_LUT = [c % 13 + 2 for c in range(52)]     # 2..14
_SUIT_LUT = [c // 13 for c in range(52)]        # 0..3


def card_rank(card: int) -> int:
    return _RANK_LUT[card]


def evaluate_5card(cards: List[int]) -> int:
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
    ranks.sort()
    r_desc = ranks[::-1]

    is_flush = (s0 == s1 == s2 == s3 == s4)

    uniq = sorted(set(ranks))
    is_straight = False
    high_straight = 0
    if len(uniq) == 5:
        if uniq == [2, 3, 4, 5, 14]:
            is_straight = True
            high_straight = 5
        elif uniq[-1] - uniq[0] == 4:
            is_straight = True
            high_straight = uniq[-1]

    cnt = [0] * 15
    for r in ranks:
        cnt[r] += 1
    distinct = [r for r in range(2, 15) if cnt[r] > 0]
    distinct.sort(key=lambda r: (cnt[r], r), reverse=True)
    freqs = [cnt[r] for r in distinct]

    BASE = 10 ** 10

    if is_flush and is_straight:
        return 9 * BASE + high_straight * 10**8

    if freqs[0] == 4:
        quad = distinct[0]
        kicker = distinct[1]
        return 8 * BASE + quad * 10**8 + kicker * 10**6

    if freqs[0] == 3 and freqs[1] == 2:
        trip = distinct[0]
        pair = distinct[1]
        return 7 * BASE + trip * 10**8 + pair * 10**6

    if is_flush:
        a, b, c, d, e = r_desc
        return 6 * BASE + a * 10**8 + b * 10**6 + c * 10**4 + d * 10**2 + e

    if is_straight:
        return 5 * BASE + high_straight * 10**8

    if freqs[0] == 3:
        trip = distinct[0]
        kickers = [r for r in r_desc if r != trip][:2]
        k1, k2 = kickers
        return 4 * BASE + trip * 10**8 + k1 * 10**6 + k2 * 10**4

    if freqs[0] == 2 and freqs[1] == 2:
        p1, p2 = distinct[0], distinct[1]
        hi, lo = max(p1, p2), min(p1, p2)
        kicker = [r for r in r_desc if r != hi and r != lo][0]
        return 3 * BASE + hi * 10**8 + lo * 10**6 + kicker * 10**4

    if freqs[0] == 2:
        pair = distinct[0]
        kickers = [r for r in r_desc if r != pair][:3]
        k1, k2, k3 = kickers
        return 2 * BASE + pair * 10**8 + k1 * 10**6 + k2 * 10**4 + k3 * 10**2

    a, b, c, d, e = r_desc
    return 1 * BASE + a * 10**8 + b * 10**6 + c * 10**4 + d * 10**2 + e


def evaluate_7card(hole: List[int], board: List[int]) -> int:
    cards = hole + board
    n = len(cards)
    if n < 5:
        return 0
    if n == 5:
        return evaluate_5card(cards)

    best = 0
    c0, c1, c2, c3, c4, c5, c6 = (cards + [0, 0])[:7]
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
        v = evaluate_5card(list(c))
        if v > best:
            best = v
    return best


def normalized_strength_worker(hole: List[int], board: List[int]) -> float:
    # same preflop heuristic as abstraction
    if len(board) < 3:
        if len(hole) < 2:
            return 0.5
        r = sorted([card_rank(c) for c in hole], reverse=True)
        return (r[0] + r[1]) / (2 * 14.0)

    raw = evaluate_7card(hole, board)

    # LUT-free normalization: map raw score to ~[0,1]
    # raw max is about 9*1e10 + 14*1e8 ≈ 9.14e10
    strength = raw / 9.14e10
    return float(max(0.0, min(1.0, strength)))


def encode_hole_cards_worker(hole: List[int]) -> List[float]:
    r1 = card_rank(hole[0]) - 2  # 0..12
    r2 = card_rank(hole[1]) - 2
    s1 = hole[0] // 13
    s2 = hole[1] // 13

    if r2 > r1:
        r1, r2 = r2, r1
        s1, s2 = s2, s1

    hi_rank_norm = r1 / 12.0
    lo_rank_norm = r2 / 12.0
    suited = 1.0 if s1 == s2 else 0.0
    pair = 1.0 if r1 == r2 else 0.0
    return [hi_rank_norm, lo_rank_norm, suited, pair]


def encode_state_worker(state, player: int) -> torch.Tensor:
    # street one-hot
    street_oh = [0.0, 0.0, 0.0, 0.0]
    street_oh[state.street] = 1.0

    pot_norm = state.pot / (STACK_SIZE * 2)
    stacks_norm = [s / STACK_SIZE for s in state.stacks]
    curr_bet_norm = state.current_bet / STACK_SIZE

    if state.last_aggressor == -1:
        last_agg_flag = 0.0
    else:
        last_agg_flag = float(state.last_aggressor == player)

    hand_str = normalized_strength_worker(state.hole[player], state.board)
    board_str = normalized_strength_worker([], state.board) if state.board else 0.0

    hole_feats = encode_hole_cards_worker(state.hole[player])

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


def regret_matching(adv: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    adv = adv.clone()
    adv[mask == 0] = 0.0
    pos = torch.clamp(adv, min=0.0)
    total = pos.sum()
    if total.item() <= 0.0:
        legal_count = mask.sum().item()
        return mask / legal_count if legal_count > 0 else mask
    return pos / total


def traverse_once(state, player: int, env, state_dim: int):
    samples = []

    def recurse(s):
        if s.terminal:
            return env.terminal_payoff(s, player)

        legal = env.legal_actions(s)
        if len(legal) == 0:
            return env.terminal_payoff(s, player)

        to_act = s.to_act
        x = encode_state_worker(s, to_act)

        mask = torch.zeros(NUM_ACTIONS, dtype=torch.float32)
        for a in legal:
            mask[a] = 1.0

        advantages = torch.rand(NUM_ACTIONS)
        probs = regret_matching(advantages, mask)

        if to_act == player:
            action_vals = []
            for a in range(NUM_ACTIONS):
                if a not in legal:
                    action_vals.append(0.0)
                    continue
                next_s = env.step(s, a)
                v = recurse(next_s)
                action_vals.append(v)

            node_val = sum(probs[a].item() * action_vals[a] for a in range(NUM_ACTIONS))
            adv_vec = torch.tensor(
                [action_vals[a] - node_val if a in legal else 0.0 for a in range(NUM_ACTIONS)],
                dtype=torch.float32,
            )
            samples.append((x, adv_vec, mask))
            return node_val

        # opponent node sampling
        if len(legal) == 1:
            chosen = legal[0]
        else:
            r = random.random()
            cum = 0.0
            chosen = legal[-1]
            for a in legal:
                cum += probs[a].item()
                if r <= cum:
                    chosen = a
                    break

        return recurse(env.step(s, chosen))

    recurse(state)
    return samples
