import random
import unittest

import torch

from real_time_search import RealTimeResolver
from three_player_engine import NUM_ACTIONS
from three_player_models import (
    build_policy_network,
    information_state_size,
)
from three_player_native import ThreePlayerHoldemEnv


class UniformPolicy:
    def probabilities(self, env, state):
        result = torch.zeros(NUM_ACTIONS)
        legal = env.legal_actions(state)
        result[legal] = 1.0 / len(legal)
        return result


class UniformOpponent:
    def probabilities(self, env, state, player):
        return UniformPolicy().probabilities(env, state)


def _seat_one_state(env):
    state = env.new_hand(button=0)
    rng = random.Random(7)
    while not state.terminal and state.to_act != 1:
        state = env.step(state, rng.choice(env.legal_actions(state)))
    assert not state.terminal
    return state


class RealTimeSearchTests(unittest.TestCase):
    def test_determinization_preserves_only_hero_private_and_public_cards(self):
        env = ThreePlayerHoldemEnv(seed=3)
        state = _seat_one_state(env)
        resolver = RealTimeResolver(
            UniformPolicy(), UniformOpponent(), tag_seat=2, seed=9
        )

        sampled = resolver._determinize(env, state, hero=1)

        self.assertEqual(sampled.hole[1], state.hole[1])
        self.assertEqual(sampled.board, state.board)
        cards = (
            [card for hand in sampled.hole for card in hand]
            + list(sampled.board)
            + list(sampled.burned)
            + list(sampled.deck)
        )
        self.assertEqual(sorted(cards), list(range(52)))

    def test_resolver_returns_a_normalized_legal_strategy(self):
        env = ThreePlayerHoldemEnv(seed=5)
        state = _seat_one_state(env)
        resolver = RealTimeResolver(
            UniformPolicy(),
            UniformOpponent(),
            tag_seat=2,
            time_budget_ms=100,
            max_rollouts=24,
            seed=11,
        )

        result = resolver.resolve(env, state)
        legal = env.legal_actions(state)

        self.assertIn(result.action, legal)
        self.assertLessEqual(result.rollouts, 24)
        self.assertGreaterEqual(result.iterations, 1)
        self.assertTrue(
            torch.isclose(result.probabilities.sum(), torch.tensor(1.0))
        )
        self.assertTrue(
            all(
                result.probabilities[action] == 0
                for action in range(NUM_ACTIONS)
                if action not in legal
            )
        )

    def test_deep_cfr_v3_policy_architecture_is_loadable(self):
        input_dim = information_state_size(include_tournament_features=True)
        network = build_policy_network("deep_cfr_branch_v3", input_dim, 64, 1)

        output = network(torch.zeros(2, input_dim))

        self.assertEqual(output.shape, (2, NUM_ACTIONS))


if __name__ == "__main__":
    unittest.main()
