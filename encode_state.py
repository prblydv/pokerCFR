import torch
import torch.nn.functional as F

from engine import (
    GameState,
    NUM_ACTIONS,
    STACK_SIZE,
    BIG_BLIND,
    STREET_PREFLOP,
    STREET_FLOP,
    STREET_TURN,
    STREET_RIVER,
)

DEVICE = torch.device("cpu")


# ============================================================
# Helper: encode a single visible card into 7 dims
# ============================================================
def encode_card(card: int) -> torch.Tensor:
    """
    card index 0–51 or -1 for not dealt.
    Output: 7D vector
    """
    if card < 0:
        return torch.zeros(7)

    rank = (card % 13) / 12.0  # normalized rank
    suit = card // 13          # 0–3
    suit_oh = F.one_hot(torch.tensor(suit), num_classes=4).float()

    return torch.cat([
        torch.tensor([rank], dtype=torch.float32),
        torch.tensor([float(suit)], dtype=torch.float32),
        suit_oh,
        torch.tensor([1.0], dtype=torch.float32),    # mask=1
    ])


# ============================================================
# Main CPU encoder
# ============================================================
def encode_state(state: GameState, hero: int) -> torch.Tensor:
    """
    CPU version of encode_e1_batch.
    No cheating: hero only sees own hole cards + public info.
    Returns 1D tensor (≈260 dims).
    """

    x = []

    # Identify opponent
    opp = 1 - hero

    # =====================================================================
    # 1. Street one-hot (4 dims)
    # =====================================================================
    street_oh = F.one_hot(torch.tensor(state.street), num_classes=4).float()
    x.append(street_oh)

    # =====================================================================
    # 2. Hero index (1 dim)
    # =====================================================================
    x.append(torch.tensor([float(hero)]))

    # =====================================================================
    # 3. Betting geometry (14 dims)
    # =====================================================================
    pot = state.pot
    cb = state.current_bet
    hero_c = state.contrib[hero]
    opp_c = state.contrib[opp]
    hero_stack = state.stacks[hero]
    opp_stack = state.stacks[opp]
    to_call = max(cb - hero_c, 0.0)

    geom = torch.tensor([
        pot / (STACK_SIZE * 2),
        cb / BIG_BLIND,
        to_call / STACK_SIZE,
        hero_c / STACK_SIZE,
        opp_c / STACK_SIZE,
        hero_stack / STACK_SIZE,
        opp_stack / STACK_SIZE,
        (hero_stack + hero_c) / STACK_SIZE,
        # pot odds
        to_call / max(pot + to_call, 1e-6),
        # SPR
        hero_stack / max(pot, 1e-6),
        # Opp committed fraction
        opp_c / max(pot + 1, 1e-6),
        pot / (STACK_SIZE + 1),
        float(state.last_aggressor == hero),
        float(state.last_aggressor == opp),
    ], dtype=torch.float32)

    x.append(geom)

    # =====================================================================
    # 4. Hero hole cards (2×7 dims = 14)
    # =====================================================================
    h0, h1 = state.hole[hero]
    x.append(encode_card(h0))
    x.append(encode_card(h1))

    # =====================================================================
    # 5. Public board cards (5×7 = 35 dims)
    # =====================================================================
    board = state.board
    for i in range(5):
        if i < len(board):
            x.append(encode_card(board[i]))
        else:
            x.append(encode_card(-1))

    # =====================================================================
    # 6. Board texture (rank/suit histograms etc.)
    # =====================================================================
    if len(board) == 0:
        # empty board → append zeros of correct shape
        num_board = torch.zeros(1)
        rank_hist = torch.zeros(13)
        suit_hist = torch.zeros(4)
        has_pair = torch.zeros(1)
        has_trips = torch.zeros(1)
        has_quads = torch.zeros(1)
        flush3 = torch.zeros(1)
        flush4 = torch.zeros(1)
        flush5 = torch.zeros(1)
        small_gaps = torch.zeros(1)
        wet_board = torch.zeros(1)
    else:
        ranks = [(c % 13) for c in board]
        suits = [(c // 13) for c in board]

        num_board = torch.tensor([float(len(board))])

        # Rank histogram (13 dims)
        rank_hist = torch.zeros(13)
        for r in ranks:
            rank_hist[r] += 1

        # Suit histogram (4 dims)
        suit_hist = torch.zeros(4)
        for s in suits:
            suit_hist[s] += 1

        # Paired flags
        has_pair = torch.tensor([float((rank_hist >= 2).any())])
        has_trips = torch.tensor([float((rank_hist >= 3).any())])
        has_quads = torch.tensor([float((rank_hist >= 4).any())])

        max_suit = suit_hist.max()

        flush3 = torch.tensor([float(max_suit >= 3)])
        flush4 = torch.tensor([float(max_suit >= 4)])
        flush5 = torch.tensor([float(max_suit >= 5)])

        # straight texture: sort ranks
        sorted_r = sorted(ranks)
        diffs = [sorted_r[i+1] - sorted_r[i] for i in range(len(sorted_r)-1)]
        small_gaps = torch.tensor([float(sum(d <= 2 for d in diffs))])

        wet_board = torch.tensor([float((max_suit >= 3) or (small_gaps > 1))])

    x.append(num_board)
    x.append(rank_hist)
    x.append(suit_hist)
    x.append(has_pair)
    x.append(has_trips)
    x.append(has_quads)
    x.append(flush3)
    x.append(flush4)
    x.append(flush5)
    x.append(small_gaps)
    x.append(wet_board)

    # =====================================================================
    # 7. Betting dynamics (public) — 6 dims
    # =====================================================================
    bet_dyn = torch.tensor([
        float(state.actions_since_raise),
        float(to_call > 0),
        float(cb > 0),
        float(hero_stack < 0.25 * STACK_SIZE),
        float(cb > 4 * BIG_BLIND),
        float(hero_c > opp_c)
    ])
    x.append(bet_dyn)

    # =====================================================================
    # 8. Legal actions mask (9 dims)
    # =====================================================================
    legal = state.env.legal_actions(state) if hasattr(state, "env") else None
    # Trainer supplies legal mask separately → do NOT compute here.
    # Instead, expect trainer to concatenate mask later.

    # But to keep output consistent with GPU version, we append a ZERO MASK placeholder.
    # Trainer will overwrite it.
    legal_placeholder = torch.zeros(NUM_ACTIONS, dtype=torch.float32)
    x.append(legal_placeholder)

    # =====================================================================
    # CONCAT everything
    # =====================================================================
    return torch.cat(x).float()
# ============================================================
# Final hero-centric encoder used by CFR trainer
# ============================================================

def encode_state_hero(state: GameState, hero: int, legal_mask=None) -> torch.Tensor:
    """
    Wraps encode_state() and appends the TRUE legal-action mask.
    Trainer will always pass real legal_mask.
    """
    base = encode_state(state, hero)   # ~260 dims

    if legal_mask is None:
        # must match NUM_ACTIONS dims
        lm = torch.zeros(NUM_ACTIONS, dtype=torch.float32)
    else:
        lm = torch.tensor(legal_mask, dtype=torch.float32)

    return torch.cat([base[:-NUM_ACTIONS], lm])  
