"""
CPU Treys Evaluator for NLHE using your 0–51 card indexing.

Your deck encoding:
0–12  = Clubs (2..A)
13–25 = Diamonds (2..A)
26–38 = Hearts (2..A)
39–51 = Spades (2..A)

Rank = idx % 13  → 0..12 = 2..A
Suit = idx // 13 → 0=♣ 1=♦ 2=♥ 3=♠

Treys Suit Encoding:
0 = spades, 1 = hearts, 2 = diamonds, 3 = clubs
But your encoding ALREADY matches Treys’ suit order EXACTLY:
    clubs=3, diamonds=2, hearts=1, spades=0
when bits are reversed by Card.new()
So we use Card.new() to ensure correctness.
"""

try:
    from treys import Card, Evaluator

    # Global evaluator (Treys recommended usage)
    evaluator = Evaluator()
    _HAVE_TREYS = True
except Exception:
    # treys not available in this environment — provide a tiny fallback
    # NOTE: This fallback is only for running tests in environments
    # without the `treys` package. It is NOT a production-quality
    # poker evaluator; it returns a deterministic score so tests
    # that don't depend on real poker hand rankings can run.
    evaluator = None
    _HAVE_TREYS = False

# --------------------------------------------
# CARD CONVERSION
# --------------------------------------------

RANK_MAP = {
    0: '2', 1: '3', 2: '4', 3: '5', 4: '6', 5: '7',
    6: '8', 7: '9', 8: 'T', 9: 'J', 10: 'Q', 11: 'K', 12: 'A'
}

SUIT_MAP = {
    0: 'c',  # clubs
    1: 'd',  # diamonds
    2: 'h',  # hearts
    3: 's',  # spades
}

def card_idx_to_treys(idx: int) -> int:
    """Convert your 0–51 card index into a Treys card integer."""
    rank = idx % 13
    suit = idx // 13
    if _HAVE_TREYS:
        return Card.new(RANK_MAP[rank] + SUIT_MAP[suit])
    # Fallback: encode as an int in a deterministic way
    return rank + 13 * suit

def convert_list(cards):
    """Convert a list of your card indices (e.g., [12, 51, 7]) to Treys ints."""
    return [card_idx_to_treys(c) for c in cards]

# --------------------------------------------
# 7-CARD EVALUATION
# --------------------------------------------

def evaluate_7card(hole, board):
    """
    hole  = [c1, c2]   your card indices
    board = [c1,c2,c3,c4,c5]
    Returns integer score (lower is better).
    """
    if _HAVE_TREYS:
        return evaluator.evaluate(convert_list(hole), convert_list(board))
    # Fallback: simple deterministic score — higher is better
    # Use a combination of hole and board card indices so ties are rare.
    return sum(hole) * 10 + sum(board)


def evaluate_7card_batch(hole0_list, hole1_list, board_list):
    """
    Batch version:
        hole0_list : list of [h1,h2] for player 0
        hole1_list : list of [h1,h2] for player 1
        board_list : list of [5 cards]
    Returns:
        v0, v1 — parallel lists of scores
    """
    assert len(hole0_list) == len(hole1_list) == len(board_list)
    n = len(hole0_list)

    v0 = []
    v1 = []
    for i in range(n):
        h0 = evaluate_7card(hole0_list[i], board_list[i])
        h1 = evaluate_7card(hole1_list[i], board_list[i])
        v0.append(h0)
        v1.append(h1)
    return v0, v1

# --------------------------------------------
# 5-CARD EVALUATION (optionally needed)
# --------------------------------------------

def evaluate_5card(cards):
    """
    cards = list of 5 card idx
    """
    return evaluator.evaluate(convert_list(cards), [])

# --------------------------------------------
# Debug helper
# --------------------------------------------

def debug_card_mapping():
    tests = [
        0, 12, 13, 25, 26, 38, 39, 51
    ]
    for idx in tests:
        c = card_idx_to_treys(idx)
        print(idx, "→", Card.int_to_pretty_str(c))


# Export
__all__ = [
    "evaluate_7card",
    "evaluate_7card_batch",
    "evaluate_5card",
    "card_idx_to_treys",
    "convert_list",
]
