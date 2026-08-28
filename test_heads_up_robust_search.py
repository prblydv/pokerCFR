import unittest

import torch

from heads_up_engine import ACTION_NAMES, NUM_ACTIONS, HeadsUpHoldemEngine
from heads_up_pluribus_search import BlueprintPublicRange, PublicRangeSnapshot
from heads_up_robust_search import (
    RobustHeadsUpSearch,
    action_noise_likelihoods,
    kl_robust_lower_bound,
)
from play_heads_up_gui import parse_args


class UniformPolicy:
    def probabilities(self, env, state):
        result = torch.zeros(NUM_ACTIONS, dtype=torch.float32)
        legal = [int(action) for action in env.legal_actions(state)]
        result[legal] = 1.0 / len(legal)
        return result


def card(rank: str, suit: int) -> int:
    return "23456789TJQKA".index(rank) + 13 * int(suit)


class RobustHeadsUpSearchTests(unittest.TestCase):
    def test_action_noise_retains_non_blueprint_actions(self):
        result = action_noise_likelihoods(
            [0.0, 0.5, 1.0],
            legal_action_count=5,
            epsilon=0.10,
        )
        self.assertAlmostEqual(result[0], 0.02)
        self.assertAlmostEqual(result[1], 0.47)
        self.assertAlmostEqual(result[2], 0.92)

    def test_kl_bound_is_monotone_and_not_above_nominal(self):
        values = [10.0, -100.0]
        weights = [0.95, 0.05]
        nominal = kl_robust_lower_bound(values, weights, radius=0.0)
        robust = kl_robust_lower_bound(values, weights, radius=0.20)
        more_robust = kl_robust_lower_bound(
            values,
            weights,
            radius=0.40,
        )
        self.assertAlmostEqual(nominal, 4.5)
        self.assertLess(robust, nominal)
        self.assertLessEqual(more_robust, robust)
        self.assertGreaterEqual(more_robust, min(values))

    def test_robust_mode_is_a_separate_cli_switch(self):
        robust = parse_args(["--search-mode", "robust"])
        current = parse_args(["--search-mode", "three-player"])
        self.assertEqual(robust.search_mode, "robust")
        self.assertEqual(current.search_mode, "three-player")
        self.assertAlmostEqual(robust.robust_action_noise, 0.10)
        self.assertAlmostEqual(robust.robust_kl_radius, 0.20)

    def test_short_robust_search_returns_a_legal_action(self):
        env = HeadsUpHoldemEngine(
            starting_stack=40,
            small_blind=1,
            big_blind=2,
            seed=17,
        )
        state = env.new_hand(button=0)
        policy = UniformPolicy()
        public_range = BlueprintPublicRange()
        public_range.reset(state.hole[int(state.to_act)])
        search = RobustHeadsUpSearch(
            policy,
            time_budget_ms=1_000,
            max_rollouts=16,
            kl_radius=0.10,
            seed=29,
        )
        result = search.resolve(
            env,
            state,
            policy.probabilities(env, state),
            public_range.snapshot(),
        )
        self.assertIn(result.choice.action, env.legal_actions(state))
        self.assertIn("KL radius", result.convergence_reason)
        self.assertEqual(
            sum(
                estimate.strategy_probability
                for estimate in result.candidates
            ),
            1.0,
        )
        for estimate in result.candidates:
            self.assertAlmostEqual(
                estimate.expected_final_payoff_bb,
                estimate.validation_ev_bb,
            )

    def test_exact_turn_all_in_folds_against_trip_range(self):
        env = HeadsUpHoldemEngine(
            starting_stack=100,
            small_blind=1,
            big_blind=2,
            seed=31,
        )
        state = env.new_hand(button=0)
        state = env.step_exact(state, "call")
        state = env.step_exact(state, "check")
        state = env.step_exact(state, "check")
        state = env.step_exact(state, "check")
        state = env.step_exact(state, "check")
        state = env.step_exact(state, "all_in")
        hero = int(state.to_act)
        self.assertEqual(hero, 1)
        state.hole[hero] = [card("A", 0), card("7", 1)]
        state.board = [
            card("5", 0),
            card("5", 1),
            card("A", 2),
            card("T", 3),
        ]
        opponent_combo = (card("8", 2), card("5", 3))
        public_range = PublicRangeSnapshot(
            combos=(opponent_combo,),
            weights=(1.0,),
            effective_sample_size=1.0,
            updates=4,
        )
        policy = UniformPolicy()
        search = RobustHeadsUpSearch(
            policy,
            time_budget_ms=1_000,
            max_rollouts=16,
            kl_radius=0.20,
            seed=37,
        )
        result = search.resolve(
            env,
            state,
            policy.probabilities(env, state),
            public_range,
        )
        self.assertEqual(ACTION_NAMES[result.choice.action], "fold")
        call = next(
            estimate
            for estimate in result.candidates
            if ACTION_NAMES[estimate.action.action] == "call"
        )
        fold = next(
            estimate
            for estimate in result.candidates
            if ACTION_NAMES[estimate.action.action] == "fold"
        )
        self.assertLess(
            call.expected_final_payoff_bb,
            fold.expected_final_payoff_bb,
        )
        self.assertAlmostEqual(
            call.expected_final_payoff_bb,
            call.validation_ev_bb,
        )
        self.assertIn("exact blocker-compatible", result.convergence_reason)


if __name__ == "__main__":
    unittest.main()
