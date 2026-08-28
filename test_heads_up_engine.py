import copy
import random
import unittest

from heads_up_engine import (
    ACTION_ALL_IN,
    ACTION_CALL,
    ACTION_CHECK,
    ACTION_FOLD,
    ACTION_HALF_POT,
    ACTION_MIN_RAISE,
    ACTION_OVERBET,
    ACTION_POT,
    ACTION_THIRD_POT,
    ACTION_THREE_QUARTER_POT,
    NUM_ACTIONS,
    STREET_FLOP,
    STREET_PREFLOP,
    STREET_RIVER,
    STREET_TURN,
    HeadsUpHoldemEngine,
    evaluate_5card,
    evaluate_7card,
)


RANKS = "23456789TJQKA"
SUITS = "cdhs"


def card(text):
    return SUITS.index(text[1]) * 13 + RANKS.index(text[0])


class HandEvaluatorTests(unittest.TestCase):
    def test_categories_are_strictly_high_is_better(self):
        hands = [
            ("Ac", "Jd", "8h", "5s", "2c"),
            ("Tc", "Td", "Ah", "7s", "2c"),
            ("Jc", "Jd", "8h", "8s", "Ac"),
            ("Qc", "Qd", "Qh", "8s", "2c"),
            ("9c", "Td", "Jh", "Qs", "Kc"),
            ("Ac", "Jc", "8c", "5c", "2c"),
            ("Kc", "Kd", "Kh", "2s", "2d"),
            ("Ac", "Ad", "Ah", "As", "2c"),
            ("9s", "Ts", "Js", "Qs", "Ks"),
        ]
        scores = [evaluate_5card([card(value) for value in hand]) for hand in hands]
        self.assertEqual(scores, sorted(scores))
        self.assertEqual(len(scores), len(set(scores)))

    def test_wheel_and_best_five_of_seven(self):
        wheel = [card(value) for value in ("Ac", "2d", "3h", "4s", "5c")]
        six_high = [card(value) for value in ("2c", "3d", "4h", "5s", "6c")]
        self.assertGreater(evaluate_5card(six_high), evaluate_5card(wheel))

        seven = [card(value) for value in ("As", "Ah", "Ad", "Kc", "Kd", "2s", "3s")]
        expected = evaluate_5card(
            [card(value) for value in ("As", "Ah", "Ad", "Kc", "Kd")]
        )
        self.assertEqual(evaluate_7card(seven), expected)
        self.assertEqual(evaluate_7card(seven[:2], seven[2:]), expected)

    def test_evaluator_rejects_bad_cards(self):
        with self.assertRaises(ValueError):
            evaluate_5card([0, 0, 1, 2, 3])
        with self.assertRaises(ValueError):
            evaluate_5card([0, 1, 2, 3])
        with self.assertRaises(ValueError):
            evaluate_7card([0, 1, 2, 3, 4, 5, 52])


class DealAndOrderTests(unittest.TestCase):
    def setUp(self):
        self.env = HeadsUpHoldemEngine(seed=11)

    def test_button_posts_small_blind_and_acts_first_preflop(self):
        state = self.env.new_hand(button=0)
        self.assertEqual((state.button, state.sb_player, state.bb_player), (0, 0, 1))
        self.assertEqual(state.street_contrib, [1, 2])
        self.assertEqual(state.stacks, [199, 198])
        self.assertEqual(state.pot, 3)
        self.assertEqual(state.current_bet, 2)
        self.assertEqual(state.to_act, 0)

        state = self.env.step_exact(state, "call")
        self.assertEqual(state.to_act, 1)
        state = self.env.step_exact(state, "check")
        self.assertEqual(state.street, STREET_FLOP)
        self.assertEqual(state.to_act, 1)  # BB/non-button first postflop.
        self.assertEqual(state.street_contrib, [0, 0])
        self.assertEqual([event.player for event in state.history], [0, 1])

    def test_first_card_is_dealt_to_big_blind(self):
        state = self.env.new_hand(button=0, deck=list(range(52)))
        self.assertEqual(state.hole[1], [51, 49])
        self.assertEqual(state.hole[0], [50, 48])
        self.assertEqual(state.deck[-1], 47)

    def test_burn_and_board_order_are_deterministic(self):
        state = self.env.new_hand(button=0, deck=list(range(52)))
        state = self.env.step_exact(state, "call")
        state = self.env.step_exact(state, "check")
        self.assertEqual(state.burned, [47])
        self.assertEqual(state.board, [46, 45, 44])

        state = self.env.step_exact(state, "check")
        state = self.env.step_exact(state, "check")
        self.assertEqual(state.street, STREET_TURN)
        self.assertEqual(state.burned, [47, 43])
        self.assertEqual(state.board, [46, 45, 44, 42])

        state = self.env.step_exact(state, "check")
        state = self.env.step_exact(state, "check")
        self.assertEqual(state.street, STREET_RIVER)
        self.assertEqual(state.burned, [47, 43, 41])
        self.assertEqual(state.board, [46, 45, 44, 42, 40])

    def test_button_rotates_between_the_two_seats(self):
        env = HeadsUpHoldemEngine(seed=5)
        self.assertEqual([env.new_hand().button for _ in range(6)], [0, 1, 0, 1, 0, 1])

    def test_constructor_and_hand_require_integer_chips(self):
        with self.assertRaises((TypeError, ValueError)):
            HeadsUpHoldemEngine(starting_stack=100.5)
        with self.assertRaises((TypeError, ValueError)):
            HeadsUpHoldemEngine(small_blind=True)
        with self.assertRaises(ValueError):
            HeadsUpHoldemEngine(small_blind=2, big_blind=2)
        with self.assertRaises((TypeError, ValueError)):
            self.env.new_hand(stacks=[100, 99.5])
        with self.assertRaises(ValueError):
            self.env.new_hand(stacks=[100])


class ExactActionTests(unittest.TestCase):
    def setUp(self):
        self.env = HeadsUpHoldemEngine(seed=17)

    def test_arbitrary_raise_to_is_exact_and_losslessly_recorded(self):
        original = self.env.new_hand(button=0)
        snapshot = copy.deepcopy(original)
        state = self.env.step_exact(original, "raise_to", 7)

        self.assertEqual(original, snapshot)
        self.assertEqual(state.street_contrib, [7, 2])
        self.assertEqual(state.stacks, [193, 198])
        self.assertEqual(state.pot, 9)
        self.assertEqual(state.current_bet, 7)
        self.assertEqual(state.min_raise, 5)
        self.assertEqual(state.to_act, 1)

        event = state.history[-1]
        self.assertIsNone(event.action)
        self.assertEqual(event.kind, "raise")
        self.assertEqual(event.amount, 6)
        self.assertEqual(event.raise_to, 7)
        self.assertEqual(event.contribution_after, 7)
        self.assertEqual(event.current_bet_before, 2)
        self.assertEqual(event.current_bet_after, 7)
        self.assertEqual(event.pot_before, 3)
        self.assertEqual(event.pot_after, 9)
        self.assertEqual(event.to_call_before, 1)
        self.assertTrue(event.full_raise)
        self.assertFalse(event.all_in)

    def test_exact_raise_boundaries_are_strict_and_never_clamped(self):
        state = self.env.new_hand(button=0)
        snapshot = copy.deepcopy(state)
        for target in (2, 3, 201):
            with self.subTest(target=target):
                with self.assertRaises(ValueError):
                    self.env.step_exact(state, "raise_to", target)
                self.assertEqual(state, snapshot)
        for target in (4.0, True):
            with self.subTest(target=target):
                with self.assertRaises((TypeError, ValueError)):
                    self.env.step_exact(state, "raise_to", target)

        minimum = self.env.step_exact(state, "raise_to", 4)
        maximum = self.env.step_exact(state, "raise_to", 200)
        self.assertEqual(minimum.street_contrib[0], 4)
        self.assertEqual(maximum.street_contrib[0], 200)
        self.assertTrue(maximum.all_in[0])

    def test_passive_action_legality_is_semantic(self):
        state = self.env.new_hand(button=0)
        with self.assertRaises(ValueError):
            self.env.step_exact(state, "check")
        with self.assertRaises(ValueError):
            self.env.step_exact(state, "call", 4)

        state = self.env.step_exact(state, "call")
        with self.assertRaises(ValueError):
            self.env.step_exact(state, "call")
        # A real room may accept the dominated in-turn fold even though the
        # finite training abstraction masks it when checking is available.
        folded = self.env.step_exact(state, "fold")
        self.assertTrue(folded.terminal)
        self.assertEqual(folded.payoffs, [2, -2])
        self.assertNotIn(ACTION_FOLD, self.env.legal_actions(state))
        checked = self.env.step_exact(state, "check")
        self.assertEqual(checked.street, STREET_FLOP)

    def test_check_raise_reopens_and_updates_minimum_increment(self):
        state = self.env.new_hand(button=0)
        state = self.env.step_exact(state, "call")
        state = self.env.step_exact(state, "check")
        self.assertEqual(state.to_act, 1)

        state = self.env.step_exact(state, "check")
        self.assertFalse(state.raise_rights[1])
        self.assertEqual(state.to_act, 0)
        state = self.env.step_exact(state, "bet", 5)
        self.assertEqual(state.history[-1].kind, "bet")
        self.assertTrue(state.raise_rights[1])
        self.assertEqual(state.to_act, 1)

        state = self.env.step_exact(state, "raise", 10)
        self.assertEqual(state.history[-1].kind, "raise")
        self.assertEqual(state.min_raise, 5)
        self.assertTrue(state.raise_rights[0])
        self.assertEqual(state.to_act, 0)

    def test_short_all_in_raise_does_not_reopen_prior_raiser(self):
        state = self.env.new_hand(button=0, stacks=[100, 5])
        state = self.env.step_exact(state, "raise_to", 4)
        self.assertFalse(state.raise_rights[0])
        state = self.env.step_exact(state, "all_in")

        self.assertEqual(state.current_bet, 5)
        self.assertEqual(state.min_raise, 2)
        self.assertFalse(state.history[-1].full_raise)
        self.assertTrue(state.history[-1].all_in)
        self.assertFalse(state.raise_rights[0])
        self.assertEqual(self.env.legal_actions(state), [ACTION_FOLD, ACTION_CALL])
        with self.assertRaises(ValueError):
            self.env.step_exact(state, "raise_to", 7)

    def test_full_reraise_reopens_prior_raiser(self):
        state = self.env.new_hand(button=0)
        state = self.env.step_exact(state, "raise_to", 4)
        state = self.env.step_exact(state, "raise_to", 8)
        self.assertTrue(state.history[-1].full_raise)
        self.assertEqual(state.min_raise, 4)
        self.assertTrue(state.raise_rights[0])
        self.assertEqual(state.to_act, 0)
        state = self.env.step_exact(state, "raise_to", 12)
        self.assertEqual(state.current_bet, 12)

    def test_fold_transition_is_immutable_and_zero_sum(self):
        original = self.env.new_hand(button=0)
        snapshot = copy.deepcopy(original)
        terminal = self.env.step_exact(original, "fold")
        self.assertEqual(original, snapshot)
        self.assertTrue(terminal.terminal)
        self.assertEqual(terminal.winners, (1,))
        self.assertEqual(terminal.stacks, [199, 201])
        self.assertEqual(terminal.uncalled_returns, [0, 1])
        self.assertEqual(terminal.street_contrib, [1, 1])
        self.assertEqual(terminal.current_bet, 1)
        self.assertEqual(terminal.payouts, [0, 2])
        self.assertEqual(terminal.payoffs, [-1, 1])
        self.assertEqual(sum(terminal.payoffs), 0)


class AbstractActionTests(unittest.TestCase):
    def setUp(self):
        self.env = HeadsUpHoldemEngine(seed=23)

    def test_preflop_targets_use_pot_after_call_and_are_deduplicated(self):
        state = self.env.new_hand(button=0)
        # pot_after_call = 3 + 1 = 4.  Half-pot therefore raises to 4,
        # colliding with min-raise; the lower MIN slot is canonical.
        expected = {
            ACTION_FOLD: 1,
            ACTION_CALL: 2,
            ACTION_MIN_RAISE: 4,
            ACTION_THREE_QUARTER_POT: 5,
            ACTION_POT: 6,
            ACTION_OVERBET: 8,
            ACTION_ALL_IN: 200,
        }
        self.assertEqual(
            {action: self.env.action_target(state, action) for action in self.env.legal_actions(state)},
            expected,
        )
        self.assertNotIn(ACTION_THIRD_POT, self.env.legal_actions(state))
        self.assertNotIn(ACTION_HALF_POT, self.env.legal_actions(state))
        self.assertEqual(
            self.env.legal_action_mask(state),
            [int(action in expected) for action in range(NUM_ACTIONS)],
        )

    def test_all_postflop_fraction_targets_use_integer_half_up_rounding(self):
        state = self.env.new_hand(button=0)
        state = self.env.step_exact(state, "raise_to", 4)
        state = self.env.step_exact(state, "call")
        self.assertEqual(state.pot, 8)
        state = self.env.step_exact(state, "check")
        self.assertEqual(state.to_act, 0)

        targets = {
            action: self.env.action_target(state, action)
            for action in self.env.legal_actions(state)
        }
        self.assertEqual(targets[ACTION_MIN_RAISE], 2)
        self.assertEqual(targets[ACTION_THIRD_POT], 3)  # round(8 / 3)
        self.assertEqual(targets[ACTION_HALF_POT], 4)
        self.assertEqual(targets[ACTION_THREE_QUARTER_POT], 6)
        self.assertEqual(targets[ACTION_POT], 8)
        self.assertEqual(targets[ACTION_OVERBET], 12)

    def test_min_raise_and_all_in_collision_has_one_canonical_slot(self):
        state = self.env.new_hand(button=0, stacks=[4, 100])
        legal = self.env.legal_actions(state)
        self.assertIn(ACTION_MIN_RAISE, legal)
        self.assertNotIn(ACTION_ALL_IN, legal)
        self.assertEqual(self.env.action_target(state, ACTION_MIN_RAISE), 4)

    def test_short_all_in_call_is_canonicalized_to_call(self):
        state = self.env.new_hand(button=0, stacks=[100, 5])
        state = self.env.step_exact(state, "raise_to", 10)
        self.assertEqual(self.env.legal_actions(state), [ACTION_FOLD, ACTION_CALL])
        called = self.env.step(state, ACTION_CALL)
        self.assertTrue(called.terminal)

    def test_fixed_aggressive_slot_records_generic_semantic_kind(self):
        state = self.env.new_hand(button=0)
        raised = self.env.step(state, ACTION_MIN_RAISE)
        self.assertEqual(raised.history[-1].kind, "raise")
        self.assertEqual(raised.history[-1].action, ACTION_MIN_RAISE)

        state = self.env.new_hand(button=0)
        state = self.env.step_exact(state, "call")
        state = self.env.step_exact(state, "check")
        state = self.env.step_exact(state, "check")
        # In a four-chip limped pot HALF_POT collides with MIN_RAISE, so use
        # the distinct pot-sized slot for this semantic-history check.
        bet = self.env.step(state, ACTION_POT)
        self.assertEqual(bet.history[-1].kind, "bet")
        self.assertEqual(bet.history[-1].action, ACTION_POT)

    def test_descriptors_match_mask_target_payment_and_execution(self):
        state = self.env.new_hand(button=0)
        descriptors = self.env.action_descriptors(state)
        self.assertEqual(len(descriptors), NUM_ACTIONS)
        legal = set(self.env.legal_actions(state))
        for action, descriptor in enumerate(descriptors):
            if action not in legal:
                self.assertIsNone(descriptor)
                continue
            self.assertIsNotNone(descriptor)
            self.assertEqual(descriptor["target"], self.env.action_target(state, action))
            self.assertEqual(descriptor["payment"], self.env.action_payment(state, action))
            child = self.env.step(state, action)
            event = child.history[-1]
            self.assertEqual(event.raise_to, descriptor["target"])
            self.assertEqual(event.amount, descriptor["payment"])

    def test_short_opening_all_in_descriptor_does_not_claim_a_raise_reopens(self):
        state = self.env.new_hand(button=0, stacks=[3, 100])
        state = self.env.step_exact(state, "call")
        state = self.env.step_exact(state, "check")
        state = self.env.step_exact(state, "check")

        descriptor = self.env.action_descriptors(state)[ACTION_ALL_IN]
        self.assertIsNotNone(descriptor)
        self.assertTrue(descriptor["is_aggressive"])
        self.assertFalse(descriptor["is_full_raise"])
        self.assertFalse(descriptor["reopens_betting"])

        state = self.env.step(state, ACTION_ALL_IN)
        self.assertFalse(state.raise_rights[state.to_act])
        self.assertEqual(self.env.legal_actions(state), [ACTION_FOLD, ACTION_CALL])


class SettlementTests(unittest.TestCase):
    def setUp(self):
        self.env = HeadsUpHoldemEngine(seed=31)

    def _showdown_state(self, initial, contributions, hole, board):
        state = self.env.new_hand(button=0, stacks=initial)
        state.initial_stacks = list(initial)
        state.stacks = [
            initial[player] - contributions[player] for player in range(2)
        ]
        state.total_contrib = list(contributions)
        state.street_contrib = list(contributions)
        state.pot = sum(contributions)
        state.hole = [[card(value) for value in cards] for cards in hole]
        state.board = [card(value) for value in board]
        state.burned = []
        state.folded = [False, False]
        state.all_in = [stack == 0 for stack in state.stacks]
        state.pending_actors.clear()
        state.to_act = None
        state.terminal = False
        state.payoffs = None
        state.payouts = None
        state.winners = ()
        state.uncalled_returns = [0, 0]
        return state

    def test_uncalled_excess_is_explicitly_refunded_before_showdown(self):
        state = self._showdown_state(
            [100, 20],
            [100, 20],
            [("Ks", "Kd"), ("As", "Ad")],
            ("2c", "3d", "7h", "9s", "Jc"),
        )
        settled = self.env.resolve_showdown(state)
        self.assertEqual(settled.uncalled_returns, [80, 0])
        self.assertEqual(settled.total_contrib, [20, 20])
        self.assertEqual(settled.street_contrib, [20, 20])
        self.assertEqual(settled.current_bet, 20)
        self.assertEqual(settled.payouts, [0, 40])
        self.assertEqual(settled.stacks, [80, 40])
        self.assertEqual(settled.payoffs, [-20, 20])
        self.assertEqual(sum(settled.payoffs), 0)

    def test_uncalled_excess_is_refunded_before_uncontested_pot_is_awarded(self):
        state = self.env.new_hand(button=0, stacks=[200, 200])
        state = self.env.step_exact(state, "raise_to", 100)
        settled = self.env.step_exact(state, "fold")

        self.assertTrue(settled.terminal)
        self.assertEqual(settled.winners, (0,))
        self.assertEqual(settled.uncalled_returns, [98, 0])
        self.assertEqual(settled.total_contrib, [2, 2])
        self.assertEqual(settled.street_contrib, [2, 2])
        self.assertEqual(settled.current_bet, 2)
        self.assertEqual(settled.payouts, [4, 0])
        self.assertEqual(settled.stacks, [202, 198])
        self.assertEqual(settled.payoffs, [2, -2])

    def test_all_in_sequence_runs_board_and_refunds_uncalled_raise(self):
        state = self.env.new_hand(button=0, stacks=[100, 5], deck=list(range(52)))
        state = self.env.step_exact(state, "raise_to", 10)
        state = self.env.step_exact(state, "all_in")
        self.assertTrue(state.terminal)
        self.assertEqual(state.uncalled_returns, [5, 0])
        self.assertEqual(state.street_contrib, [5, 5])
        self.assertEqual(state.current_bet, 5)
        self.assertEqual(len(state.board), 5)
        self.assertEqual(len(state.burned), 3)
        self.assertEqual(len(state.deck), 40)
        self.assertEqual(sum(state.stacks), 105)
        self.assertEqual(sum(state.payoffs), 0)

    def test_blind_completion_all_in_runs_out_without_synthetic_bb_check(self):
        state = self.env.new_hand(button=0, stacks=[2, 100], deck=list(range(52)))
        self.assertEqual(state.to_act, 0)
        state = self.env.step_exact(state, "call")

        self.assertTrue(state.terminal)
        self.assertEqual(len(state.history), 1)
        self.assertEqual(state.history[0].player, 0)
        self.assertEqual(state.history[0].kind, "call")
        self.assertEqual(len(state.board), 5)
        self.assertEqual(len(state.burned), 3)
        self.assertEqual(sum(state.stacks), 102)
        self.assertEqual(sum(state.payoffs), 0)

    def test_board_royal_flush_splits_the_pot(self):
        state = self._showdown_state(
            [100, 100],
            [25, 25],
            [("2d", "3d"), ("4h", "5h")],
            ("Tc", "Jc", "Qc", "Kc", "Ac"),
        )
        settled = self.env.resolve_showdown(state)
        self.assertEqual(settled.winners, (0, 1))
        self.assertEqual(settled.payouts, [25, 25])
        self.assertEqual(settled.stacks, [100, 100])
        self.assertEqual(settled.payoffs, [0, 0])

    def test_resolve_showdown_is_copy_safe(self):
        state = self._showdown_state(
            [100, 100],
            [20, 20],
            [("As", "Ad"), ("Ks", "Kd")],
            ("2c", "3d", "7h", "9s", "Jc"),
        )
        snapshot = copy.deepcopy(state)
        settled = self.env.resolve_showdown(state)
        self.assertEqual(state, snapshot)
        self.assertTrue(settled.terminal)


class RandomInvariantTests(unittest.TestCase):
    def test_random_fixed_and_off_tree_rollouts_preserve_exact_invariants(self):
        chooser = random.Random(123456)
        env = HeadsUpHoldemEngine(seed=654321)
        for hand in range(400):
            stacks = [chooser.randint(3, 250), chooser.randint(3, 250)]
            state = env.new_hand(button=hand % 2, stacks=stacks)
            total_chips = sum(stacks)
            actions = 0
            while not state.terminal:
                self.assertIsNotNone(state.to_act)
                self.assertIn(state.to_act, state.pending_actors)
                self.assertEqual(sum(state.stacks) + state.pot, total_chips)
                self.assertEqual(sum(state.total_contrib), state.pot)
                self.assertTrue(all(isinstance(value, int) for value in state.stacks))
                self.assertTrue(all(value >= 0 for value in state.stacks))

                legal = env.legal_actions(state)
                self.assertTrue(legal)
                self.assertEqual(
                    env.legal_action_mask(state),
                    [int(action in legal) for action in range(NUM_ACTIONS)],
                )
                effects = []
                for action in legal:
                    target = env.action_target(state, action)
                    payment = env.action_payment(state, action)
                    self.assertEqual(
                        target,
                        state.street_contrib[state.to_act] + payment,
                    )
                    kind = "fold" if action == ACTION_FOLD else (
                        "check" if action == ACTION_CHECK else "commit"
                    )
                    effects.append((kind, target))
                self.assertEqual(len(effects), len(set(effects)))

                actor = state.to_act
                opponent = 1 - actor
                maximum = state.street_contrib[actor] + state.stacks[actor]
                minimum = state.current_bet + state.min_raise
                can_exact_raise = (
                    state.raise_rights[actor]
                    and not state.folded[opponent]
                    and not state.all_in[opponent]
                    and state.stacks[opponent] > 0
                    and maximum > state.current_bet
                )
                snapshot = copy.deepcopy(state)
                if can_exact_raise and chooser.random() < 0.30:
                    if maximum >= minimum:
                        target = chooser.randint(minimum, maximum)
                    else:
                        target = maximum  # the sole legal short all-in raise
                    state = env.step_exact(state, "raise_to", target)
                    self.assertEqual(state.history[-1].raise_to, target)
                else:
                    state = env.step(state, chooser.choice(legal))
                self.assertEqual(snapshot.history, state.history[:-1])
                self.assertLessEqual(snapshot.pot, state.pot if not state.terminal else total_chips)
                actions += 1
                self.assertLess(actions, 100)

            self.assertEqual(state.pot, 0)
            self.assertIsNone(state.to_act)
            self.assertEqual(sum(state.stacks), total_chips)
            self.assertEqual(sum(state.payoffs), 0)
            self.assertTrue(all(isinstance(value, int) for value in state.payoffs))
            self.assertTrue(all(value >= 0 for value in state.stacks))
            if len(state.board) == 5:
                exposed = state.hole[0] + state.hole[1] + state.board + state.burned
                self.assertEqual(len(exposed), len(set(exposed)))


if __name__ == "__main__":
    unittest.main()
