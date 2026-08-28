import pickle
import random
import unittest

import numpy as np

import heads_up_engine as python_engine
import heads_up_native as native_engine
from heads_up_models import (
    build_action_descriptors,
    encode_information_state,
    information_state_size,
)


def deterministic_deck(pop_order):
    """Return a complete deck whose end pops in ``pop_order`` order."""

    if len(pop_order) != len(set(pop_order)):
        raise ValueError("pop order contains duplicate cards")
    remainder = [card for card in range(52) if card not in pop_order]
    return remainder + list(reversed(pop_order))


class NativeHeadsUpEngineTests(unittest.TestCase):
    def setUp(self):
        self.env = native_engine.HeadsUpHoldemEngine(
            starting_stack=10_000,
            small_blind=50,
            big_blind=100,
            seed=7,
        )

    def test_native_binary_schema_handshake_matches_python_contract(self):
        extension = native_engine._native
        self.assertEqual(extension.NATIVE_ABI_VERSION, 4)
        self.assertEqual(extension.NUM_PLAYERS, python_engine.NUM_PLAYERS)
        self.assertEqual(extension.NUM_ACTIONS, python_engine.NUM_ACTIONS)
        self.assertEqual(
            extension.ENGINE_SCHEMA_VERSION,
            python_engine.ENGINE_SCHEMA_VERSION,
        )
        self.assertEqual(
            extension.ACTION_SCHEMA_VERSION,
            python_engine.ACTION_SCHEMA_VERSION,
        )
        self.assertEqual(
            tuple(extension.ACTION_NAMES),
            tuple(python_engine.ACTION_NAMES),
        )

    def test_native_bayesian_update_and_root_regret_matching(self):
        posterior = native_engine.bayesian_condition(
            [0.5, 0.5],
            [0.25, 0.75],
        )
        self.assertEqual(posterior["weights"], [0.25, 0.75])
        self.assertAlmostEqual(
            posterior["effective_sample_size"],
            1.6,
        )
        strategy = native_engine.regret_match_root(
            [-2.0, 4.0, 2.0],
            [True, True, False],
            [10.0, 0.0, 100.0],
        )
        self.assertEqual(strategy, [0.0, 1.0, 0.0])
        no_positive = native_engine.regret_match_root(
            [-2.0, -4.0, -1.0],
            [True, True, False],
            [1.0, 3.0, 100.0],
        )
        self.assertEqual(no_positive, [0.0, 1.0, 0.0])
        hierarchical = native_engine.hierarchical_regret_match_root(
            [1.0, 2.0, 3.0, 1.0],
            [True, True, True, True],
            [0.0, 0.0, 0.0, 0.0],
            [0, 1, 2, 2],
        )
        self.assertEqual(hierarchical["family_strategy"], [1 / 7, 2 / 7, 4 / 7])
        self.assertEqual(
            hierarchical["action_strategy"],
            [1 / 7, 2 / 7, 3 / 7, 1 / 7],
        )

    def test_native_batch_step_matches_scalar_transitions(self):
        first = self.env.new_hand(button=0)
        second = self.env.new_hand(button=1)
        actions = [
            int(self.env.legal_actions(first)[0]),
            int(self.env.legal_actions(second)[0]),
        ]
        scalar = [
            self.env.step(self.env.clone(state), action)
            for state, action in zip((first, second), actions)
        ]
        batched = self.env.step_batch((first, second), actions)
        self.assertEqual(
            [native_engine._native.state_to_dict(state) for state in batched],
            [native_engine._native.state_to_dict(state) for state in scalar],
        )

    def test_robust_all_in_validator_ignores_a_faulty_fold_prior(self):
        def card(text):
            return "cdhs".index(text[1]) * 13 + "23456789TJQKA".index(text[0])

        result = native_engine.estimate_all_in_ev(
            [card("7s"), card("Ac")],
            [card("4s"), card("8d"), card("7h")],
            [[card("8h"), card("6h")]],
            [1.0],
            [0.0],
            fold_payoff=2.0,
            win_payoff=96.0,
            loss_payoff=-96.0,
            samples=50_000,
            seed=93,
            robust_best_response=True,
        )
        self.assertTrue(result["robust_best_response"])
        self.assertEqual(result["robust_call_hands"], 1)
        self.assertEqual(result["call_rate"], 1.0)
        self.assertLess(result["mean"], -40.0)

    def test_terminal_call_scenarios_include_value_heavy_stress_range(self):
        def card(text):
            return "cdhs".index(text[1]) * 13 + "23456789TJQKA".index(text[0])

        result = native_engine.estimate_terminal_call_scenarios(
            [card("3h"), card("Kd")],
            [card("2d"), card("Jc"), card("3s")],
            [
                [card("Qs"), card("Jd")],
                [card("7h"), card("6h")],
            ],
            [0.5, 0.5],
            fold_payoff=-98.0,
            win_payoff=190.0,
            loss_payoff=-190.0,
            nominal_samples=50_000,
            seed=94,
        )
        rows = {str(row["name"]): row for row in result["scenarios"]}
        self.assertEqual(
            set(rows),
            {"posterior", "tempered", "contaminated", "value_heavy"},
        )
        self.assertEqual(result["worst_name"], "value_heavy")
        self.assertLess(
            rows["value_heavy"]["mean"],
            rows["posterior"]["mean"],
        )

    def test_native_all_in_validator_prices_the_a9s_failure_correctly(self):
        def card(text):
            return "cdhs".index(text[1]) * 13 + "23456789TJQKA".index(text[0])

        result = native_engine.estimate_all_in_ev(
            [card("As"), card("9s")],
            [card("7s"), card("3c"), card("Jc")],
            [[card("Jh"), card("6h")]],
            [1.0],
            [1.0],
            fold_payoff=6.0,
            win_payoff=157.0,
            loss_payoff=-157.0,
            samples=100_000,
            seed=91,
        )
        self.assertLess(result["mean"], -95.0)
        self.assertGreater(result["mean"], -108.0)
        self.assertAlmostEqual(result["call_rate"], 1.0)
        folded = native_engine.estimate_all_in_ev(
            [card("As"), card("9s")],
            [card("7s"), card("3c"), card("Jc")],
            [[card("Jh"), card("6h")]],
            [1.0],
            [0.0],
            fold_payoff=6.0,
            win_payoff=157.0,
            loss_payoff=-157.0,
            samples=1_000,
            seed=92,
        )
        self.assertEqual(folded["mean"], 6.0)

    def test_heads_up_positions_deal_and_action_order(self):
        state = self.env.new_hand(button=0, deck=list(range(52)))

        self.assertEqual(state.button, 0)
        self.assertEqual(state.sb_player, 0)
        self.assertEqual(state.bb_player, 1)
        self.assertEqual(state.to_act, 0)
        # Cards are dealt first to the seat left of the dealer: the BB.
        self.assertEqual(state.hole[1], [51, 49])
        self.assertEqual(state.hole[0], [50, 48])
        self.assertEqual(state.stacks, [9_950, 9_900])
        self.assertEqual(state.street_contrib, [50, 100])
        self.assertEqual(state.pot, 150)

        state = self.env.step_exact(state, "call")
        self.assertEqual(state.to_act, 1)
        state = self.env.step_exact(state, "check")
        self.assertEqual(state.street, native_engine.STREET_FLOP)
        self.assertEqual(state.to_act, 1)  # BB acts first after the flop.
        self.assertEqual(len(state.board), 3)
        self.assertEqual(len(state.burned), 1)

    def test_finite_policy_slots_are_strictly_deduplicated(self):
        state = self.env.new_hand(button=0, deck=list(range(52)))
        legal = self.env.legal_actions(state)
        targets = {
            action: self.env.action_target(state, action)
            for action in legal
        }

        self.assertEqual(
            legal,
            [
                native_engine.ACTION_FOLD,
                native_engine.ACTION_CALL,
                native_engine.ACTION_MIN_RAISE,
                native_engine.ACTION_THREE_QUARTER_POT,
                native_engine.ACTION_POT,
                native_engine.ACTION_OVERBET,
                native_engine.ACTION_ALL_IN,
            ],
        )
        self.assertEqual(targets[native_engine.ACTION_MIN_RAISE], 200)
        self.assertEqual(targets[native_engine.ACTION_THREE_QUARTER_POT], 250)
        self.assertEqual(targets[native_engine.ACTION_POT], 300)
        self.assertEqual(targets[native_engine.ACTION_OVERBET], 400)
        self.assertEqual(targets[native_engine.ACTION_ALL_IN], 10_000)
        effects = [
            (
                action in (native_engine.ACTION_FOLD, native_engine.ACTION_CHECK),
                target,
            )
            for action, target in targets.items()
        ]
        self.assertEqual(len(effects), len(set(effects)))
        self.assertEqual(
            self.env.legal_action_mask(state),
            [int(action in legal) for action in range(native_engine.NUM_ACTIONS)],
        )

    def test_arbitrary_raise_is_exact_and_does_not_change_policy_width(self):
        state = self.env.new_hand(button=0, deck=list(range(52)))
        state = self.env.step_exact(state, "raise_to", 237)

        self.assertEqual(state.current_bet, 237)
        self.assertEqual(state.street_contrib, [237, 100])
        self.assertEqual(state.pot, 337)
        self.assertEqual(state.history[-1].kind, "raise")
        self.assertEqual(state.history[-1].raise_to, 237)
        self.assertEqual(state.history[-1].amount, 187)
        self.assertEqual(len(self.env.legal_action_mask(state)), 10)

        state = self.env.step_exact(state, "call")
        self.assertEqual(state.street, native_engine.STREET_FLOP)
        self.assertEqual(state.pot, 474)
        self.assertEqual(state.to_act, state.bb_player)

    def test_short_all_in_does_not_reopen_raising(self):
        env = native_engine.HeadsUpHoldemEngine(2_000, 50, 100, seed=1)
        state = env.new_hand(
            button=0,
            stacks=[2_000, 250],
            deck=list(range(52)),
        )
        state = env.step_exact(state, "raise_to", 200)
        state = env.step_exact(state, "raise_to", 250)

        self.assertFalse(state.history[-1].full_raise)
        self.assertFalse(state.raise_rights[0])
        self.assertEqual(
            env.legal_actions(state),
            [native_engine.ACTION_FOLD, native_engine.ACTION_CALL],
        )
        with self.assertRaisesRegex(ValueError, "not been reopened"):
            env.step_exact(state, "raise_to", 500)

    def test_short_opening_all_in_does_not_offer_a_meaningless_raise(self):
        env = native_engine.HeadsUpHoldemEngine(1_000, 50, 100, seed=1)
        state = env.new_hand(
            button=0,
            stacks=[150, 1_000],
            deck=list(range(52)),
        )
        state = env.step_exact(state, "call")
        state = env.step_exact(state, "check")
        state = env.step_exact(state, "check")  # BB checks the flop.
        descriptor = env.action_descriptors(state)[native_engine.ACTION_ALL_IN]
        self.assertFalse(descriptor["reopens_betting"])
        state = env.step_exact(state, "all_in")  # BTN opens for only 50.

        self.assertFalse(state.history[-1].full_raise)
        self.assertFalse(state.raise_rights[state.bb_player])
        self.assertEqual(
            env.legal_actions(state),
            [native_engine.ACTION_FOLD, native_engine.ACTION_CALL],
        )

    def test_matched_all_in_completion_runs_out_without_synthetic_check(self):
        env = native_engine.HeadsUpHoldemEngine(1_000, 50, 100, seed=1)
        state = env.new_hand(
            button=0,
            stacks=[100, 1_000],
            deck=list(range(52)),
        )
        state = env.step_exact(state, "call")

        self.assertTrue(state.terminal)
        self.assertEqual(len(state.history), 1)
        self.assertEqual(state.history[0].kind, "call")
        self.assertEqual(len(state.board), 5)

    def test_uncalled_overbet_is_refunded_before_showdown(self):
        env = native_engine.HeadsUpHoldemEngine(10_000, 50, 100, seed=1)
        state = env.new_hand(
            button=0,
            stacks=[10_000, 2_000],
            deck=list(range(52)),
        )
        state = env.step_exact(state, "raise_to", 10_000)
        state = env.step_exact(state, "call")

        self.assertTrue(state.terminal)
        self.assertEqual(state.total_contrib, [2_000, 2_000])
        self.assertEqual(state.street_contrib, [2_000, 2_000])
        self.assertEqual(state.current_bet, 2_000)
        self.assertEqual(state.uncalled_returns, [8_000, 0])
        self.assertEqual(sum(state.stacks), 12_000)
        self.assertEqual(sum(state.payoffs), 0)
        self.assertLessEqual(abs(state.payoffs[0]), 2_000)

    def test_uncontested_payoff_uses_net_chips(self):
        env = native_engine.HeadsUpHoldemEngine(100, 1, 2, seed=1)
        state = env.new_hand(button=0, deck=list(range(52)))
        state = env.step_exact(state, "fold")

        self.assertTrue(state.terminal)
        self.assertEqual(state.stacks, [99, 101])
        self.assertEqual(state.payoffs, [-1, 1])
        self.assertEqual(state.uncalled_returns, [0, 1])
        self.assertEqual(state.street_contrib, [1, 1])
        self.assertEqual(state.current_bet, 1)
        self.assertEqual(state.payouts, [0, 2])
        self.assertEqual(state.winners, (1,))

    def test_exact_room_interface_accepts_dominated_fold_when_checking(self):
        env = native_engine.HeadsUpHoldemEngine(100, 1, 2, seed=1)
        state = env.new_hand(button=0, deck=list(range(52)))
        state = env.step_exact(state, "call")
        self.assertNotIn(native_engine.ACTION_FOLD, env.legal_actions(state))

        state = env.step_exact(state, "fold")
        self.assertTrue(state.terminal)
        self.assertEqual(state.payoffs, [2, -2])

    def test_unmatched_raise_is_refunded_before_fold_payout(self):
        env = native_engine.HeadsUpHoldemEngine(200, 1, 2, seed=1)
        state = env.new_hand(button=0, deck=list(range(52)))
        state = env.step_exact(state, "raise_to", 100)
        state = env.step_exact(state, "fold")

        self.assertEqual(state.uncalled_returns, [98, 0])
        self.assertEqual(state.total_contrib, [2, 2])
        self.assertEqual(state.street_contrib, [2, 2])
        self.assertEqual(state.current_bet, 2)
        self.assertEqual(state.payouts, [4, 0])
        self.assertEqual(state.stacks, [202, 198])
        self.assertEqual(state.payoffs, [2, -2])

    def test_board_play_tie_splits_exactly(self):
        # Board is a royal flush; neither private hand can improve it.
        pop_order = [0, 1, 2, 3, 4, 8, 9, 10, 5, 11, 6, 12]
        env = native_engine.HeadsUpHoldemEngine(100, 1, 2, seed=1)
        state = env.new_hand(
            button=0,
            deck=deterministic_deck(pop_order),
        )
        while not state.terminal:
            legal = env.legal_actions(state)
            action = (
                native_engine.ACTION_CHECK
                if native_engine.ACTION_CHECK in legal
                else native_engine.ACTION_CALL
            )
            state = env.step(state, action)

        self.assertEqual(state.board, [8, 9, 10, 11, 12])
        self.assertEqual(state.stacks, [100, 100])
        self.assertEqual(state.payoffs, [0, 0])
        self.assertEqual(state.winners, (0, 1))

    def test_odd_tie_chip_is_awarded_to_big_blind(self):
        pop_order = [0, 1, 2, 3, 4, 8, 9, 10, 5, 11, 6, 12]
        env = native_engine.HeadsUpHoldemEngine(100, 1, 2, seed=1)
        state = env.new_hand(
            button=0,
            deck=deterministic_deck(pop_order),
        )
        # Reach a complete board without settling the final betting round.
        while state.street != native_engine.STREET_RIVER:
            legal = env.legal_actions(state)
            action = (
                native_engine.ACTION_CHECK
                if native_engine.ACTION_CHECK in legal
                else native_engine.ACTION_CALL
            )
            state = env.step(state, action)
        # Model one odd dead chip while preserving whole-state conservation.
        stacks = list(state.stacks)
        stacks[state.sb_player] -= 1
        state.stacks = stacks
        state.pot += 1
        state = env.resolve_showdown(state)

        self.assertEqual(state.payouts[state.bb_player], 3)
        self.assertEqual(state.payouts[state.sb_player], 2)
        self.assertEqual(sum(state.payoffs), 0)

    def test_clone_and_pickle_are_independent(self):
        state = self.env.new_hand(button=0, deck=list(range(52)))
        clone = self.env.clone(state)
        advanced = self.env.step_exact(clone, "raise_to", 237)

        self.assertEqual(state.pot, 150)
        self.assertEqual(len(state.history), 0)
        self.assertEqual(advanced.pot, 337)
        restored = pickle.loads(pickle.dumps(advanced))
        self.assertEqual(restored.stacks, advanced.stacks)
        self.assertEqual(restored.history[-1].raise_to, 237)
        self.assertEqual(self.env.legal_actions(restored), self.env.legal_actions(advanced))

    def test_native_encoder_is_bit_exact_with_python_reference(self):
        state = self.env.new_hand(button=0, deck=list(range(52)))
        state = self.env.step_exact(state, "raise_to", 237)
        legal = self.env.legal_actions(state)
        descriptors = self.env.action_descriptors(state)

        reference = encode_information_state(
            state,
            state.to_act,
            legal,
            self.env.big_blind,
            action_descriptors=descriptors,
        ).numpy()
        native = native_engine.encode_information_state_native(
            state,
            state.to_act,
            legal,
            self.env.big_blind,
            action_descriptors=descriptors,
        )

        self.assertEqual(native.shape, (information_state_size(),))
        np.testing.assert_array_equal(native, reference)

    def test_live_encoder_rejects_missing_action_descriptors(self):
        state = self.env.new_hand(button=0, deck=list(range(52)))
        legal = self.env.legal_actions(state)
        with self.assertRaisesRegex(ValueError, "action_descriptors"):
            native_engine.encode_information_state_native(
                state,
                state.to_act,
                legal,
                self.env.big_blind,
            )

    def test_native_encoder_rejects_nonacting_hero_and_uses_recent_history(self):
        state = self.env.new_hand(button=0, deck=list(range(52)))
        legal = self.env.legal_actions(state)
        descriptors = self.env.action_descriptors(state)
        with self.assertRaisesRegex(ValueError, "hero == state.to_act"):
            native_engine.encode_information_state_native(
                state,
                1 - state.to_act,
                legal,
                self.env.big_blind,
                action_descriptors=descriptors,
            )

        for _ in range(9):
            state = self.env.step(state, native_engine.ACTION_MIN_RAISE)
        legal = self.env.legal_actions(state)
        descriptors = self.env.action_descriptors(state)
        encoded = native_engine.encode_information_state_native(
            state,
            state.to_act,
            legal,
            self.env.big_blind,
            8,
            action_descriptors=descriptors,
        )
        self.assertEqual(encoded.shape, (information_state_size(8),))


class NativePythonDifferentialTests(unittest.TestCase):
    def test_fixed_policy_paths_match_python_reference(self):
        rng = random.Random(91)
        for hand in range(40):
            stacks = [rng.randint(300, 3_000), rng.randint(300, 3_000)]
            button = hand % 2
            deck = list(range(52))
            rng.shuffle(deck)
            py_env = python_engine.HeadsUpHoldemEngine(
                2_000,
                50,
                100,
                seed=hand,
            )
            native_env = native_engine.HeadsUpHoldemEngine(
                2_000,
                50,
                100,
                seed=hand,
            )
            py_state = py_env.new_hand(button=button, stacks=stacks, deck=deck)
            native_state = native_env.new_hand(
                button=button,
                stacks=stacks,
                deck=deck,
            )

            for _ in range(50):
                self.assert_states_equal(py_state, native_state)
                py_legal = py_env.legal_actions(py_state)
                native_legal = native_env.legal_actions(native_state)
                self.assertEqual(py_legal, native_legal)
                self.assertEqual(
                    [py_env.action_target(py_state, action) for action in py_legal],
                    [
                        native_env.action_target(native_state, action)
                        for action in native_legal
                    ],
                )
                if py_state.terminal:
                    break
                action = rng.choice(py_legal)
                py_state = py_env.step(py_state, action)
                native_state = native_env.step(native_state, action)
            else:
                self.fail("hand did not terminate within 50 decisions")

    def test_exact_off_tree_path_matches_python_reference(self):
        deck = list(range(52))
        py_env = python_engine.HeadsUpHoldemEngine(10_000, 50, 100, seed=3)
        native_env = native_engine.HeadsUpHoldemEngine(10_000, 50, 100, seed=3)
        py_state = py_env.new_hand(button=0, deck=deck)
        native_state = native_env.new_hand(button=0, deck=deck)

        for kind, target in (
            ("raise_to", 237),
            ("call", None),
            ("check", None),
            ("raise_to", 333),
            ("call", None),
        ):
            py_state = py_env.step_exact(py_state, kind, raise_to=target)
            native_state = native_env.step_exact(
                native_state,
                kind,
                raise_to=target,
            )
            self.assert_states_equal(py_state, native_state)

    def assert_states_equal(self, reference, native):
        for field in (
            "deck",
            "board",
            "burned",
            "hole",
            "stacks",
            "initial_stacks",
            "total_contrib",
            "street_contrib",
            "folded",
            "all_in",
            "pot",
            "current_bet",
            "min_raise",
            "to_act",
            "street",
            "button",
            "sb_player",
            "bb_player",
            "pending_actors",
            "raise_rights",
            "last_action_bet",
            "last_full_raiser",
            "terminal",
            "payoffs",
            "payouts",
            "winners",
        ):
            self.assertEqual(
                getattr(reference, field),
                getattr(native, field),
                msg=f"state field mismatch: {field}",
            )
        self.assertEqual(len(reference.history), len(native.history))
        if reference.history:
            left = reference.history[-1]
            right = native.history[-1]
            for field in (
                "player",
                "street",
                "kind",
                "amount",
                "contribution_after",
                "current_bet_before",
                "current_bet_after",
                "pot_before",
                "pot_after",
                "full_raise",
                "all_in",
            ):
                self.assertEqual(
                    getattr(left, field),
                    getattr(right, field),
                    msg=f"history field mismatch: {field}",
                )


if __name__ == "__main__":
    unittest.main()
