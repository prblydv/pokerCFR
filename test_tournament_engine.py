import math
import random
import unittest

from three_player_engine import (
    ACTION_CALL,
    ACTION_CHECK,
    ACTION_FOLD,
    STREET_FLOP,
    ThreePlayerHoldemEnv,
)


RANKS = "23456789TJQKA"
SUITS = "cdhs"


def card(text):
    return SUITS.index(text[1]) * 13 + RANKS.index(text[0])


class TournamentHandSetupTests(unittest.TestCase):
    def setUp(self):
        self.env = ThreePlayerHoldemEnv(seed=17)

    def test_zero_stack_seat_is_skipped_everywhere(self):
        state = self.env.new_hand(button=0, stacks=[350, 0, 250])

        self.assertEqual(state.alive, [True, False, True])
        self.assertEqual(state.eliminated, [False, True, False])
        self.assertEqual(state.players_remaining, 2)
        self.assertEqual([len(cards) for cards in state.hole], [2, 0, 2])
        self.assertEqual((state.button, state.sb_player, state.bb_player), (0, 0, 2))
        self.assertEqual(state.street_contrib, [1.0, 0.0, 2.0])
        self.assertEqual(state.stacks, [349.0, 0.0, 248.0])
        self.assertTrue(state.folded[1])
        self.assertFalse(state.all_in[1])
        self.assertNotIn(1, state.pending_actors)
        self.assertFalse(state.raise_rights[1])

    def test_tournament_stack_validation(self):
        invalid = (
            [100, 0, 0],
            [0, 0, 0],
            [100, -1, 100],
            [100, math.nan, 100],
            [100, math.inf, 100],
        )
        for stacks in invalid:
            with self.subTest(stacks=stacks):
                with self.assertRaises(ValueError):
                    self.env.new_hand(stacks=stacks)

        with self.assertRaises(ValueError):
            self.env.new_hand(button=1, stacks=[100, 0, 100])

    def test_automatic_button_rotation_skips_eliminated_seats(self):
        buttons = [
            self.env.new_hand(stacks=[400, 0, 200]).button
            for _ in range(6)
        ]
        self.assertEqual(buttons, [0, 2, 0, 2, 0, 2])


class HeadsUpRulesTests(unittest.TestCase):
    def setUp(self):
        self.env = ThreePlayerHoldemEnv(seed=23)

    def test_button_is_small_blind_and_acts_first_preflop(self):
        state = self.env.new_hand(button=0, stacks=[300, 0, 300])
        self.assertEqual((state.sb_player, state.bb_player), (0, 2))
        self.assertEqual(state.to_act, 0)

        state = self.env.step(state, ACTION_CALL)
        self.assertEqual(state.to_act, 2)
        state = self.env.step(state, ACTION_CHECK)

        self.assertEqual(state.street, STREET_FLOP)
        self.assertEqual(state.to_act, 2)  # BB acts first postflop.
        self.assertEqual([record.player for record in state.history], [0, 2])

    def test_single_heads_up_fold_awards_the_pot(self):
        state = self.env.new_hand(button=0, stacks=[350, 0, 250])
        state = self.env.step(state, ACTION_FOLD)

        self.assertTrue(state.terminal)
        self.assertEqual(state.winners, (2,))
        self.assertEqual(state.stacks, [349.0, 0.0, 251.0])
        self.assertEqual(state.payoffs, [-1.0, 0.0, 1.0])
        self.assertEqual(state.players_remaining, 2)

    def test_heads_up_random_rollouts_conserve_tournament_chips(self):
        chooser = random.Random(123)
        env = ThreePlayerHoldemEnv(seed=456)
        for _ in range(100):
            state = env.new_hand(stacks=[475, 0, 125])
            actions = 0
            while not state.terminal:
                self.assertNotEqual(state.to_act, 1)
                self.assertAlmostEqual(sum(state.stacks) + state.pot, 600.0)
                state = env.step(state, chooser.choice(env.legal_actions(state)))
                actions += 1
                self.assertLess(actions, 100)

            self.assertAlmostEqual(sum(state.stacks), 600.0)
            self.assertAlmostEqual(sum(state.payoffs), 0.0)
            self.assertFalse(state.alive[1])
            self.assertTrue(state.eliminated[1])


class TournamentSettlementTests(unittest.TestCase):
    def test_terminal_state_marks_two_new_bustouts(self):
        env = ThreePlayerHoldemEnv(seed=31)
        state = env.new_hand(button=0, stacks=[10, 10, 10])

        # Build a deterministic three-way all-in showdown.  Seat 0's aces beat
        # both lower pairs, eliminating both opponents at settlement.
        state.stacks = [0.0, 0.0, 0.0]
        state.total_contrib = [10.0, 10.0, 10.0]
        state.street_contrib = [10.0, 10.0, 10.0]
        state.pot = 30.0
        state.hole = [
            [card("As"), card("Ad")],
            [card("Ks"), card("Kd")],
            [card("Qs"), card("Qd")],
        ]
        state.board = [
            card("2c"),
            card("3d"),
            card("7h"),
            card("9s"),
            card("Jc"),
        ]
        state.all_in = [True, True, True]
        state.pending_actors.clear()
        state.to_act = None

        settled = env.resolve_showdown(state)
        self.assertEqual(settled.stacks, [30.0, 0.0, 0.0])
        self.assertEqual(settled.alive, [True, False, False])
        self.assertEqual(settled.eliminated, [False, True, True])
        self.assertEqual(settled.players_remaining, 1)
        self.assertAlmostEqual(sum(settled.stacks), 30.0)
        self.assertAlmostEqual(sum(settled.payoffs), 0.0)

        # A tournament controller must stop here rather than deal a one-player
        # hand, and the engine enforces that boundary.
        with self.assertRaises(ValueError):
            env.new_hand(stacks=settled.stacks)


if __name__ == "__main__":
    unittest.main()
