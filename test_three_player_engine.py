import copy
import random
import unittest

from three_player_engine import (
    ACTION_ALL_IN,
    ACTION_CALL,
    ACTION_CHECK,
    ACTION_FOLD,
    ACTION_MIN_RAISE,
    ACTION_NAMES,
    NUM_ACTIONS,
    STREET_FLOP,
    STREET_PREFLOP,
    ThreePlayerHoldemEnv,
    calculate_side_pots,
    evaluate_5card,
    evaluate_7card,
)


RANKS = "23456789TJQKA"
SUITS = "cdhs"


def card(text):
    return SUITS.index(text[1]) * 13 + RANKS.index(text[0])


class HandEvaluatorTests(unittest.TestCase):
    def test_categories_and_high_is_better(self):
        straight_flush = [card(value) for value in ("9s", "Ts", "Js", "Qs", "Ks")]
        quads = [card(value) for value in ("Ac", "Ad", "Ah", "As", "2c")]
        full_house = [card(value) for value in ("Kc", "Kd", "Kh", "2s", "2d")]
        flush = [card(value) for value in ("Ac", "Jc", "8c", "5c", "2c")]
        straight = [card(value) for value in ("9c", "Td", "Jh", "Qs", "Kc")]
        trips = [card(value) for value in ("Qc", "Qd", "Qh", "8s", "2c")]
        two_pair = [card(value) for value in ("Jc", "Jd", "8h", "8s", "Ac")]
        pair = [card(value) for value in ("Tc", "Td", "Ah", "7s", "2c")]
        high = [card(value) for value in ("Ac", "Jd", "8h", "5s", "2c")]
        hands = [high, pair, two_pair, trips, straight, flush, full_house, quads, straight_flush]
        scores = [evaluate_5card(hand) for hand in hands]
        self.assertEqual(scores, sorted(scores))

    def test_wheel_and_best_of_seven(self):
        wheel = [card(value) for value in ("Ac", "2d", "3h", "4s", "5c")]
        six_high = [card(value) for value in ("2c", "3d", "4h", "5s", "6c")]
        self.assertGreater(evaluate_5card(six_high), evaluate_5card(wheel))

        seven = [card(value) for value in ("As", "Ah", "Ad", "Kc", "Kd", "2s", "3s")]
        expected = evaluate_5card([card(value) for value in ("As", "Ah", "Ad", "Kc", "Kd")])
        self.assertEqual(evaluate_7card(seven), expected)

    def test_rejects_duplicate_cards(self):
        with self.assertRaises(ValueError):
            evaluate_5card([0, 0, 1, 2, 3])


class BettingTests(unittest.TestCase):
    def setUp(self):
        self.env = ThreePlayerHoldemEnv(seed=7)

    def test_positions_and_action_order(self):
        state = self.env.new_hand(button=0)
        self.assertEqual((state.button, state.sb_player, state.bb_player), (0, 1, 2))
        self.assertEqual(state.to_act, 0)
        self.assertEqual(state.street_contrib, [0.0, 1.0, 2.0])

        state = self.env.step(state, ACTION_CALL)
        self.assertEqual(state.to_act, 1)
        state = self.env.step(state, ACTION_CALL)
        self.assertEqual(state.to_act, 2)
        state = self.env.step(state, ACTION_CHECK)

        self.assertEqual(state.street, STREET_FLOP)
        self.assertEqual(len(state.board), 3)
        self.assertEqual(state.to_act, 1)  # first active seat left of button
        self.assertEqual(state.street_contrib, [0.0, 0.0, 0.0])
        self.assertEqual([record.player for record in state.history], [0, 1, 2])

    def test_automatic_button_rotation(self):
        env = ThreePlayerHoldemEnv(seed=1)
        self.assertEqual([env.new_hand().button for _ in range(5)], [0, 1, 2, 0, 1])

    def test_step_is_copy_safe_and_invalid_actions_are_strict(self):
        state = self.env.new_hand(button=0)
        snapshot = copy.deepcopy(state)
        branch = self.env.step(state, ACTION_CALL)
        self.assertEqual(state, snapshot)
        self.assertNotEqual(branch, state)

        with self.assertRaises(ValueError):
            self.env.step(state, ACTION_CHECK)  # button owes the big blind
        self.assertEqual(state, snapshot)

    def test_legal_actions_are_deduplicated_and_match_execution(self):
        # Seat 0 can put in exactly four chips: min-raise and all-in have the
        # same effect, so only the lower canonical action ID is exposed.
        state = self.env.new_hand(button=0, stacks=[4, 200, 200])
        legal = self.env.legal_actions(state)
        self.assertIn(ACTION_MIN_RAISE, legal)
        self.assertNotIn(ACTION_ALL_IN, legal)
        self.assertEqual(len(legal), len(set(legal)))
        self.assertEqual(self.env.legal_action_mask(state), [int(a in legal) for a in range(NUM_ACTIONS)])

        for action in legal:
            child = self.env.step(state, action)
            self.assertEqual(state.street, STREET_PREFLOP)
            self.assertIsNot(child, state)

    def test_short_all_in_does_not_reopen_raising(self):
        # BB starts with five: after a full raise to four, it can only make a
        # short all-in raise to five.  The callers owe one but cannot re-raise.
        state = self.env.new_hand(button=0, stacks=[200, 200, 5])
        state = self.env.step(state, ACTION_MIN_RAISE)
        state = self.env.step(state, ACTION_CALL)
        self.assertIn(ACTION_ALL_IN, self.env.legal_actions(state))
        state = self.env.step(state, ACTION_ALL_IN)

        self.assertEqual(state.current_bet, 5.0)
        self.assertFalse(state.raise_rights[0])
        legal = self.env.legal_actions(state)
        self.assertEqual(legal, [ACTION_FOLD, ACTION_CALL])

    def test_two_folds_are_zero_sum(self):
        state = self.env.new_hand(button=0)
        state = self.env.step(state, ACTION_FOLD)
        state = self.env.step(state, ACTION_FOLD)
        self.assertTrue(state.terminal)
        self.assertEqual(state.winners, (2,))
        self.assertEqual(state.payoffs, [-0.0, -1.0, 1.0])
        self.assertAlmostEqual(sum(state.payoffs), 0.0)
        self.assertAlmostEqual(sum(state.stacks), 600.0)


class SettlementTests(unittest.TestCase):
    def setUp(self):
        self.env = ThreePlayerHoldemEnv(seed=11)

    def _showdown_state(self, initial, contrib, hole, board, folded=None):
        state = self.env.new_hand(button=0, stacks=initial)
        state.initial_stacks = [float(value) for value in initial]
        state.stacks = [initial[p] - contrib[p] for p in range(3)]
        state.total_contrib = [float(value) for value in contrib]
        state.pot = float(sum(contrib))
        state.hole = [[card(value) for value in cards] for cards in hole]
        state.board = [card(value) for value in board]
        state.folded = list(folded or [False, False, False])
        state.terminal = False
        state.payoffs = None
        return state

    def test_side_pots_include_uncalled_refund(self):
        pots = calculate_side_pots([50, 100, 200], [False, False, False])
        self.assertEqual([pot.amount for pot in pots], [150.0, 100.0, 100.0])
        self.assertEqual(pots[-1].eligible, (2,))

        # P0 wins the main pot, P1 the first side pot, and P2 receives its
        # unmatched final 100 back.
        state = self._showdown_state(
            [50, 100, 200],
            [50, 100, 200],
            [("As", "Ad"), ("Qs", "Qd"), ("Js", "Jd")],
            ("2c", "3d", "7h", "9s", "Kc"),
        )
        state = self.env.resolve_showdown(state)
        self.assertEqual(state.payouts, [150.0, 100.0, 100.0])
        self.assertEqual(state.stacks, [150.0, 100.0, 100.0])
        self.assertEqual(state.payoffs, [100.0, 0.0, -100.0])
        self.assertAlmostEqual(sum(state.payoffs), 0.0)

    def test_ties_split_each_side_pot(self):
        state = self._showdown_state(
            [50, 100, 100],
            [50, 100, 100],
            [("2d", "3d"), ("4h", "5h"), ("6s", "7s")],
            ("Tc", "Jc", "Qc", "Kc", "Ac"),
        )
        state = self.env.resolve_showdown(state)
        self.assertEqual(state.payouts, [50.0, 100.0, 100.0])
        self.assertEqual(state.payoffs, [0.0, 0.0, 0.0])
        self.assertEqual(state.winners, (0, 1, 2))


class RandomRolloutTests(unittest.TestCase):
    def test_random_legal_rollouts_preserve_invariants(self):
        chooser = random.Random(12345)
        env = ThreePlayerHoldemEnv(seed=54321)
        for _ in range(300):
            state = env.new_hand()
            actions = 0
            while not state.terminal:
                legal = env.legal_actions(state)
                self.assertTrue(legal)
                self.assertIn(state.to_act, state.pending_actors)
                self.assertGreater(state.stacks[state.to_act], 0.0)
                self.assertAlmostEqual(sum(state.stacks) + state.pot, 600.0)
                self.assertAlmostEqual(sum(state.total_contrib), state.pot)

                state = env.step(state, chooser.choice(legal))
                actions += 1
                self.assertLess(actions, 100)

            self.assertEqual(state.pot, 0.0)
            self.assertIsNone(state.to_act)
            self.assertAlmostEqual(sum(state.stacks), 600.0)
            self.assertAlmostEqual(sum(state.payoffs), 0.0)
            self.assertTrue(all(stack >= 0.0 for stack in state.stacks))
            self.assertTrue(all(0 <= record.action < len(ACTION_NAMES) for record in state.history))


if __name__ == "__main__":
    unittest.main()
