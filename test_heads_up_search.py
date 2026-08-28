import copy
import unittest

import torch

from heads_up_engine import ACTION_ALL_IN, NUM_ACTIONS, HeadsUpHoldemEngine
from heads_up_models import (
    build_policy_network,
    encoder_metadata,
    information_state_size,
)
from heads_up_search import (
    HeadsUpNetworkPolicy,
    HeadsUpRealTimeResolver,
    apply_observed_action,
    build_decision_context,
    validate_checkpoint_encoder,
)


class NineOutputLegacyPolicy:
    def probabilities(self, env, state):
        return torch.ones(9) / 9.0


class HeadsUpSearchContractTests(unittest.TestCase):
    def setUp(self):
        self.env = HeadsUpHoldemEngine(seed=29)

    def _off_tree_state(self):
        root = self.env.new_hand(button=0)
        abstract_targets = {
            self.env.action_target(root, action)
            for action in self.env.legal_actions(root)
        }
        self.assertNotIn(7, abstract_targets)
        return root, apply_observed_action(self.env, root, "raise_to", 7)

    def _policy(self):
        network = build_policy_network(
            "residual_mlp",
            information_state_size(),
            hidden=16,
            blocks=0,
        )
        return HeadsUpNetworkPolicy(
            network, checkpoint_encoder=encoder_metadata()
        )

    def test_arbitrary_observed_raise_is_exact_and_root_is_unchanged(self):
        root, state = self._off_tree_state()

        self.assertEqual(root.street_contrib, [1, 2])
        self.assertEqual(state.street_contrib, [7, 2])
        self.assertEqual(state.current_bet, 7)
        self.assertEqual(state.history[-1].raise_to, 7)
        self.assertIsNone(state.history[-1].action)

    def test_decision_context_preserves_exact_state_and_ten_slot_contract(self):
        _, state = self._off_tree_state()
        context = build_decision_context(self.env, state)

        self.assertEqual(context.hero, 1)
        self.assertEqual(context.observation.numel(), information_state_size())
        self.assertEqual(tuple(context.legal_mask.shape), (NUM_ACTIONS,))
        self.assertEqual(
            context.legal_mask.tolist(),
            [float(value) for value in self.env.legal_action_mask(state)],
        )
        self.assertEqual(len(context.action_descriptors), NUM_ACTIONS)
        for action in context.legal_actions:
            descriptor = context.action_descriptors[action]
            self.assertIsNotNone(descriptor)
            self.assertEqual(
                int(descriptor["target"]),
                self.env.action_target(state, action),
            )

    def test_network_policy_masks_illegal_actions_after_off_tree_raise(self):
        _, state = self._off_tree_state()
        probabilities = self._policy().probabilities(self.env, state)
        legal = set(self.env.legal_actions(state))

        self.assertEqual(tuple(probabilities.shape), (NUM_ACTIONS,))
        self.assertTrue(torch.isclose(probabilities.sum(), torch.tensor(1.0)))
        self.assertTrue(
            all(
                float(probabilities[action]) == 0.0
                for action in range(NUM_ACTIONS)
                if action not in legal
            )
        )

    def test_resolver_consumes_off_tree_raise_and_returns_ten_slots(self):
        _, state = self._off_tree_state()
        resolver = HeadsUpRealTimeResolver(
            self._policy(),
            tag_opponent=None,
            tag_seat=None,
            time_budget_ms=100,
            max_rollouts=24,
            seed=41,
        )

        result = resolver.resolve(self.env, state)
        legal = set(self.env.legal_actions(state))

        self.assertIn(result.action, legal)
        self.assertEqual(tuple(result.probabilities.shape), (NUM_ACTIONS,))
        self.assertEqual(tuple(result.blueprint_probabilities.shape), (NUM_ACTIONS,))
        self.assertTrue(torch.isclose(result.probabilities.sum(), torch.tensor(1.0)))
        self.assertIn(ACTION_ALL_IN, legal)
        self.assertTrue(
            all(
                float(result.probabilities[action]) == 0.0
                for action in range(NUM_ACTIONS)
                if action not in legal
            )
        )

    def test_resolver_rejects_three_player_nine_output_policy(self):
        _, state = self._off_tree_state()
        resolver = HeadsUpRealTimeResolver(
            NineOutputLegacyPolicy(),
            tag_opponent=None,
            tag_seat=None,
            max_rollouts=10,
        )
        with self.assertRaisesRegex(ValueError, "must return 10"):
            resolver.resolve(self.env, state)

    def test_checkpoint_schema_is_strict(self):
        metadata = encoder_metadata()
        validate_checkpoint_encoder(metadata)
        broken = copy.deepcopy(metadata)
        broken["action_names"] = tuple(reversed(broken["action_names"]))
        with self.assertRaisesRegex(ValueError, "action_names mismatch"):
            validate_checkpoint_encoder(broken)


if __name__ == "__main__":
    unittest.main()
