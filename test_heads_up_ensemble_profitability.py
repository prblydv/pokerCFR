import unittest

import torch

from evaluate_heads_up_ensemble_profitability import (
    ScriptedProvider,
    TrainerProvider,
    ensemble_top_k_from_name,
    run_reciprocal_match,
    top_k_probabilities,
)

from heads_up_cfr import HeadsUpNeuralCFR
from heads_up_engine import HeadsUpHoldemEnv
from heads_up_engine import ACTION_ALL_IN, ACTION_CALL, ACTION_CHECK, NUM_ACTIONS
from heads_up_native import HeadsUpHoldemEngine as NativeHeadsUpHoldemEngine
from heads_up_production import CallingStationOpponent


class _AlwaysShoveProvider:
    name = "always_shove"

    def __init__(self, env):
        self.env = env

    def probabilities_batch(self, states):
        result = torch.zeros((len(states), NUM_ACTIONS), dtype=torch.float32)
        for row, state in enumerate(states):
            legal = self.env.legal_actions(state)
            action = (
                ACTION_ALL_IN
                if ACTION_ALL_IN in legal
                else ACTION_CALL
                if ACTION_CALL in legal
                else ACTION_CHECK
            )
            result[row, action] = 1.0
        return result


class TopKProbabilityTests(unittest.TestCase):
    def test_parses_top_k_ensemble_opponent_name(self):
        self.assertEqual(ensemble_top_k_from_name("ensemble_top4"), 4)
        self.assertIsNone(ensemble_top_k_from_name("ensemble_full"))
        self.assertIsNone(ensemble_top_k_from_name("ensemble_top0"))

    def test_keeps_three_highest_positive_actions_and_renormalizes(self):
        probabilities = torch.tensor(
            [[0.05, 0.00, 0.10, 0.15, 0.20, 0.01, 0.30, 0.07, 0.04, 0.08]],
            dtype=torch.float32,
        )
        result = top_k_probabilities(probabilities, 3)
        self.assertAlmostEqual(float(result.sum()), 1.0, places=6)
        self.assertEqual(torch.nonzero(result[0]).flatten().tolist(), [3, 4, 6])
        expected = probabilities[0, [3, 4, 6]] / probabilities[0, [3, 4, 6]].sum()
        torch.testing.assert_close(result[0, [3, 4, 6]], expected)

    def test_tie_breaks_by_lower_action_index(self):
        probabilities = torch.tensor(
            [[0.2, 0.0, 0.2, 0.2, 0.1, 0.1, 0.1, 0.05, 0.03, 0.02]],
            dtype=torch.float32,
        )
        result = top_k_probabilities(probabilities, 2)
        self.assertEqual(torch.nonzero(result[0]).flatten().tolist(), [0, 2])

    def test_nonpositive_top_k_preserves_distribution(self):
        probabilities = torch.tensor(
            [[0.25, 0.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
        )
        torch.testing.assert_close(top_k_probabilities(probabilities, 0), probabilities)

    def test_live_trainer_provider_uses_its_own_encoder(self):
        env = HeadsUpHoldemEnv(seed=91)
        trainer = HeadsUpNeuralCFR(
            env,
            hidden=16,
            blocks=1,
            advantage_capacity=8,
            policy_capacity=8,
            seed=92,
        )
        state = env.new_hand(button=0)
        probabilities = TrainerProvider(trainer, top_k=3).probabilities_batch([state])
        self.assertEqual(tuple(probabilities.shape), (1, 10))
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=6)
        self.assertLessEqual(int((probabilities > 0.0).sum()), 3)

    def test_reciprocal_match_tracks_all_in_bet_to_pot_ratios(self):
        environment = {
            "starting_stack": 200,
            "small_blind": 1,
            "big_blind": 2,
        }
        candidate_env = NativeHeadsUpHoldemEngine(**environment, seed=5)
        result = run_reciprocal_match(
            _AlwaysShoveProvider(candidate_env),
            lambda env: ScriptedProvider(env, CallingStationOpponent()),
            environment=environment,
            hands=20,
            seed=6,
            inference_batch_size=20,
            simulation_batch_size=10,
        )
        ratio = result["candidate_all_in_bet_to_pot_ratio"]
        self.assertGreater(ratio["count"], 0)
        self.assertGreater(ratio["mean"], 0.0)
        self.assertIn("pot before acting", ratio["definition"])
        self.assertEqual(
            ratio["count"],
            result["candidate_all_in_raise_over_pot_after_call"]["count"],
        )
        spr = result["candidate_all_in_spr_after_call"]
        self.assertEqual(ratio["count"], spr["count"])
        self.assertGreater(spr["median"], 0.0)
        self.assertIn("only when", spr["definition"])


if __name__ == "__main__":
    unittest.main()
