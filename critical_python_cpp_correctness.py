"""Targeted high-risk differential checks for Python vs packed C++ engines."""

from __future__ import annotations

import copy

import poker_native_engine

from test_native_engine import state_view
from three_player_engine import (
    ACTION_ALL_IN,
    ACTION_CALL,
    ACTION_FOLD,
    ACTION_MIN_RAISE,
    ThreePlayerHoldemEnv as PythonEnv,
    evaluate_5card as python_evaluate_5card,
    evaluate_7card as python_evaluate_7card,
)
from three_player_native import (
    ThreePlayerHoldemEnv as NativeEnv,
    evaluate_5card as native_evaluate_5card,
    evaluate_7card as native_evaluate_7card,
)

RANKS = "23456789TJQKA"
SUITS = "cdhs"


def card(text: str) -> int:
    return SUITS.index(text[1]) * 13 + RANKS.index(text[0])


def assert_same(py_state, cpp_state, label: str) -> None:
    if state_view(py_state) != state_view(cpp_state):
        raise AssertionError(f"{label}: Python/C++ states differ")


def paired_start(*, stacks=(200, 200, 200), deck=None):
    py_env, cpp_env = PythonEnv(seed=7), NativeEnv(seed=7)
    py_state = py_env.new_hand(button=0, stacks=stacks, deck=deck)
    cpp_state = cpp_env.new_hand(button=0, stacks=stacks, deck=deck)
    assert_same(py_state, cpp_state, "new hand")
    return py_env, cpp_env, py_state, cpp_state


def paired_step(py_env, cpp_env, py_state, cpp_state, action, label):
    py_parent = copy.deepcopy(py_state)
    cpp_parent = state_view(cpp_state)
    py_child = py_env.step(py_state, action)
    cpp_child = cpp_env.step(cpp_state, action)
    if py_state != py_parent or state_view(cpp_state) != cpp_parent:
        raise AssertionError(f"{label}: parent state mutated")
    assert_same(py_child, cpp_child, label)
    return py_child, cpp_child


def showdown_states(initial, contrib, holes, board, folded=None):
    py_env, cpp_env, py_state, cpp_state = paired_start(stacks=initial)
    py_state.initial_stacks = [float(x) for x in initial]
    py_state.stacks = [float(initial[i] - contrib[i]) for i in range(3)]
    py_state.total_contrib = [float(x) for x in contrib]
    py_state.pot = float(sum(contrib))
    py_state.hole = [[card(x) for x in hand] for hand in holes]
    py_state.board = [card(x) for x in board]
    py_state.folded = list(folded or [False, False, False])
    py_state.terminal = False
    py_state.payoffs = None
    py_state.payouts = None
    py_state.winners = ()

    payload = poker_native_engine.state_to_dict(cpp_state)
    payload.update(
        initial_stacks=[float(x) for x in initial],
        stacks=[float(initial[i] - contrib[i]) for i in range(3)],
        total_contrib=[float(x) for x in contrib],
        pot=float(sum(contrib)),
        hole=[[card(x) for x in hand] for hand in holes],
        board=[card(x) for x in board],
        folded=list(folded or [False, False, False]),
        terminal=False,
        has_payoffs=False,
        has_payouts=False,
        payoffs=[0.0, 0.0, 0.0],
        payouts=[0.0, 0.0, 0.0],
        winners=[],
    )
    cpp_state = poker_native_engine.state_from_dict(payload)
    return py_env, cpp_env, py_state, cpp_state


def main() -> None:
    # Critical evaluator categories, wheel handling, and best-five-of-seven.
    hands = (
        ("Ac", "Jd", "8h", "5s", "2c"),
        ("Tc", "Td", "Ah", "7s", "2c"),
        ("Jc", "Jd", "8h", "8s", "Ac"),
        ("Qc", "Qd", "Qh", "8s", "2c"),
        ("9c", "Td", "Jh", "Qs", "Kc"),
        ("Ac", "Jc", "8c", "5c", "2c"),
        ("Kc", "Kd", "Kh", "2s", "2d"),
        ("Ac", "Ad", "Ah", "As", "2c"),
        ("9s", "Ts", "Js", "Qs", "Ks"),
        ("Ac", "2d", "3h", "4s", "5c"),
    )
    scores = []
    for hand in hands:
        cards = [card(x) for x in hand]
        py_score = python_evaluate_5card(cards)
        cpp_score = native_evaluate_5card(cards)
        if py_score != cpp_score:
            raise AssertionError(f"five-card evaluator differs for {hand}")
        scores.append(py_score)
    seven = [card(x) for x in ("As", "Ah", "Ad", "Kc", "Kd", "2s", "3s")]
    if python_evaluate_7card(seven) != native_evaluate_7card(seven):
        raise AssertionError("seven-card best-hand selection differs")

    # Fold settlement and immutable branching.
    py_env, cpp_env, py_state, cpp_state = paired_start()
    py_state, cpp_state = paired_step(
        py_env, cpp_env, py_state, cpp_state, ACTION_FOLD, "first fold"
    )
    py_state, cpp_state = paired_step(
        py_env, cpp_env, py_state, cpp_state, ACTION_FOLD, "second fold settlement"
    )
    if py_state.payoffs != [-0.0, -1.0, 1.0] or sum(py_state.payoffs) != 0:
        raise AssertionError("fold settlement is not zero sum")

    # A short all-in must not illegally reopen raising.
    py_env, cpp_env, py_state, cpp_state = paired_start(stacks=(200, 200, 5))
    for action, label in (
        (ACTION_MIN_RAISE, "full raise"),
        (ACTION_CALL, "call full raise"),
        (ACTION_ALL_IN, "short all-in"),
    ):
        py_state, cpp_state = paired_step(
            py_env, cpp_env, py_state, cpp_state, action, label
        )
    if py_env.legal_actions(py_state) != [ACTION_FOLD, ACTION_CALL]:
        raise AssertionError("short all-in incorrectly reopened raising")
    if cpp_env.legal_actions(cpp_state) != [ACTION_FOLD, ACTION_CALL]:
        raise AssertionError("native short all-in incorrectly reopened raising")

    # Main pot, side pot, and unmatched overbet refund go to different players.
    setup = showdown_states(
        [50, 100, 200],
        [50, 100, 200],
        [("As", "Ad"), ("Qs", "Qd"), ("Js", "Jd")],
        ("2c", "3d", "7h", "9s", "Kc"),
    )
    py_env, cpp_env, py_state, cpp_state = setup
    py_result = py_env.resolve_showdown(py_state)
    cpp_result = cpp_env.resolve_showdown(cpp_state)
    assert_same(py_result, cpp_result, "side pots and unmatched refund")
    if py_result.payouts != [150.0, 100.0, 100.0]:
        raise AssertionError("incorrect side-pot payouts")

    # Board straight ties every main/side pot and must preserve all chips.
    setup = showdown_states(
        [50, 100, 100],
        [50, 100, 100],
        [("2d", "3d"), ("4h", "5h"), ("6s", "7s")],
        ("Tc", "Jc", "Qc", "Kc", "Ac"),
    )
    py_env, cpp_env, py_state, cpp_state = setup
    py_result = py_env.resolve_showdown(py_state)
    cpp_result = cpp_env.resolve_showdown(cpp_state)
    assert_same(py_result, cpp_result, "three-way tied side pots")
    if py_result.winners != (0, 1, 2) or sum(py_result.payoffs) != 0:
        raise AssertionError("tie settlement failed")

    print("critical_evaluator_cases=11")
    print("critical_betting_cases=3")
    print("critical_settlement_cases=2")
    print("ALL CRITICAL PYTHON/C++ CHECKS EXACT")


if __name__ == "__main__":
    main()
