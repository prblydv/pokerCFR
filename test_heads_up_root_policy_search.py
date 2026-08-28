import copy
from types import SimpleNamespace
import unittest

import torch

from heads_up_engine import NUM_ACTIONS, HeadsUpHoldemEngine
from heads_up_pluribus_search import BlueprintPublicRange, PublicRangeSnapshot
from heads_up_root_policy_search import (
    HeadsUpRootPolicySearch,
    robust_inferred_range,
)
from play_heads_up_gui import (
    HeadsUpSnapshotPolicy,
    parse_args,
    range_probability_color,
    summarize_public_range,
)


class UniformPolicy:
    def probabilities(self, env, state):
        result = torch.zeros(NUM_ACTIONS, dtype=torch.float32)
        legal = [int(action) for action in env.legal_actions(state)]
        result[legal] = 1.0 / len(legal)
        return result


class BatchedUniformPolicy(UniformPolicy):
    def probabilities_batch(self, env, states):
        return torch.stack([self.probabilities(env, state) for state in states])


class FixedLogitNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer(
            "values",
            torch.linspace(-1.0, 1.0, NUM_ACTIONS),
        )

    def forward(self, observations):
        return self.values.unsqueeze(0).expand(len(observations), -1)


class HeadsUpRootPolicySearchTests(unittest.TestCase):
    def setUp(self):
        self.env = HeadsUpHoldemEngine(
            starting_stack=200,
            small_blind=1,
            big_blind=2,
            seed=13,
        )
        self.state = self.env.new_hand(button=0)
        self.policy = UniformPolicy()

    @staticmethod
    def _fixed_snapshot_policy(*, native_batch_encoding: bool):
        network = FixedLogitNetwork()
        policy = HeadsUpSnapshotPolicy.__new__(HeadsUpSnapshotPolicy)
        policy.device = torch.device("cpu")
        policy.native_batch_encoding = bool(native_batch_encoding)
        policy.snapshot = SimpleNamespace(
            policy_nets=[network, network],
            metadata={"max_history": 32},
        )
        return policy

    def test_defaults_use_inferred_range_and_increased_budget(self):
        search = HeadsUpRootPolicySearch(self.policy)
        self.assertEqual(search.time_budget_ms, 10_000)
        self.assertEqual(search.max_rollouts, 150_000)
        self.assertEqual(search.blueprint_weight, 0.35)
        self.assertEqual(search.max_actions_per_rollout, 64)
        self.assertEqual(search.batch_iterations, 3072)
        self.assertTrue(search.use_native_rollouts)
        self.assertEqual(search.range_mode, "inferred")
        self.assertEqual(search.min_strategy_probability, 0.0)

    def test_minimum_strategy_probability_prunes_before_sampling(self):
        blueprint = self.policy.probabilities(self.env, self.state)
        search = HeadsUpRootPolicySearch(
            self.policy,
            time_budget_ms=1_000,
            max_rollouts=24,
            min_strategy_probability=0.30,
            seed=17,
        )
        public_range = BlueprintPublicRange()
        public_range.reset(self.state.hole[int(self.state.to_act)])
        result = search.resolve(
            self.env,
            self.state,
            blueprint,
            public_range.snapshot(),
        )
        chosen = next(
            row
            for row in result.candidates
            if row.action.action == result.choice.action
        )
        self.assertGreaterEqual(chosen.strategy_probability, 0.30)
        self.assertTrue(any(row.safety_pruned for row in result.candidates))
        self.assertTrue(
            all(
                row.strategy_probability == 0.0
                for row in result.candidates
                if row.safety_pruned
            )
        )
        self.assertAlmostEqual(
            sum(row.strategy_probability for row in result.candidates),
            1.0,
            places=7,
        )

    def test_convergence_text_reports_configured_policy_and_search_weights(self):
        blueprint = self.policy.probabilities(self.env, self.state)
        public_range = BlueprintPublicRange()
        public_range.reset(self.state.hole[int(self.state.to_act)])
        result = HeadsUpRootPolicySearch(
            self.policy,
            time_budget_ms=1_000,
            max_rollouts=24,
            blueprint_weight=0.65,
            seed=17,
        ).resolve(
            self.env,
            self.state,
            blueprint,
            public_range.snapshot(),
        )
        self.assertIn("65% blueprint anchor", result.convergence_reason)
        self.assertIn("35% search weight", result.convergence_reason)

    def test_determinization_does_not_depend_on_opponent_hole_cards(self):
        altered = copy.deepcopy(self.state)
        altered.hole[1] = list(reversed(altered.hole[1]))
        first = HeadsUpRootPolicySearch(self.policy, seed=91)
        second = HeadsUpRootPolicySearch(self.policy, seed=91)
        one = first._determinize(self.env, self.state, hero=0)
        two = second._determinize(self.env, altered, hero=0)
        self.assertEqual(one.hole[1], two.hole[1])
        self.assertEqual(one.deck, two.deck)
        self.assertEqual(one.burned, two.burned)
        self.assertIsNot(one, self.state)
        self.assertIsNot(one.hole, self.state.hole)
        self.assertEqual(self.state.hole[0], altered.hole[0])

    def test_root_search_returns_legal_normalized_mixed_strategy(self):
        blueprint = self.policy.probabilities(self.env, self.state)
        search = HeadsUpRootPolicySearch(
            self.policy,
            time_budget_ms=1_000,
            max_rollouts=24,
            seed=17,
        )
        public_range = BlueprintPublicRange()
        public_range.reset(self.state.hole[int(self.state.to_act)])
        result = search.resolve(
            self.env,
            self.state,
            blueprint,
            public_range.snapshot(),
        )
        self.assertIn(result.choice.action, self.env.legal_actions(self.state))
        self.assertGreater(result.cfr_iterations, 0)
        self.assertGreater(result.terminal_rollouts, 0)
        self.assertAlmostEqual(
            sum(row.strategy_probability for row in result.candidates),
            1.0,
            places=6,
        )

    def test_batched_rollouts_preserve_sequential_resolver_result(self):
        public_range = BlueprintPublicRange()
        public_range.reset(self.state.hole[int(self.state.to_act)])
        snapshot = public_range.snapshot()
        blueprint = self.policy.probabilities(self.env, self.state)
        sequential = HeadsUpRootPolicySearch(
            self.policy,
            time_budget_ms=60_000,
            max_rollouts=48,
            batch_iterations=1,
            use_native_rollouts=False,
            seed=71,
        ).resolve(self.env, self.state, blueprint, snapshot)
        batched = HeadsUpRootPolicySearch(
            BatchedUniformPolicy(),
            time_budget_ms=60_000,
            max_rollouts=48,
            batch_iterations=8,
            use_native_rollouts=False,
            seed=71,
        ).resolve(self.env, self.state, blueprint, snapshot)
        self.assertEqual(sequential.cfr_iterations, batched.cfr_iterations)
        self.assertEqual(sequential.terminal_rollouts, batched.terminal_rollouts)
        for old, new in zip(sequential.candidates, batched.candidates):
            self.assertEqual(old.action.action, new.action.action)
            self.assertEqual(old.samples, new.samples)
            self.assertAlmostEqual(
                old.expected_final_payoff_bb,
                new.expected_final_payoff_bb,
                places=7,
            )
            self.assertAlmostEqual(
                old.strategy_probability,
                new.strategy_probability,
                places=7,
            )

    def test_native_rollouts_match_python_rollouts(self):
        try:
            import heads_up_native  # noqa: F401
        except ImportError:
            self.skipTest("optional native heads-up engine is not built")
        public_range = BlueprintPublicRange()
        public_range.reset(self.state.hole[int(self.state.to_act)])
        snapshot = public_range.snapshot()
        blueprint = self.policy.probabilities(self.env, self.state)
        python_result = HeadsUpRootPolicySearch(
            BatchedUniformPolicy(),
            time_budget_ms=60_000,
            max_rollouts=48,
            batch_iterations=8,
            use_native_rollouts=False,
            seed=79,
        ).resolve(self.env, self.state, blueprint, snapshot)
        native_result = HeadsUpRootPolicySearch(
            BatchedUniformPolicy(),
            time_budget_ms=60_000,
            max_rollouts=48,
            batch_iterations=8,
            use_native_rollouts=True,
            seed=79,
        ).resolve(self.env, self.state, blueprint, snapshot)
        self.assertFalse(python_result.native_backend)
        self.assertTrue(native_result.native_backend)
        for python_row, native_row in zip(
            python_result.candidates, native_result.candidates
        ):
            self.assertEqual(python_row.action.action, native_row.action.action)
            self.assertAlmostEqual(
                python_row.expected_final_payoff_bb,
                native_row.expected_final_payoff_bb,
                places=7,
            )
            self.assertAlmostEqual(
                python_row.strategy_probability,
                native_row.strategy_probability,
                places=7,
            )

    def test_batched_sampler_matches_scalar_threshold_sampling(self):
        import heads_up_native

        env = heads_up_native.HeadsUpHoldemEngine(200, 1, 2, seed=11)
        states = [
            env.new_hand(button=0, deck=list(range(52)))
            for _ in range(5)
        ]
        policy = self._fixed_snapshot_policy(native_batch_encoding=True)
        thresholds = [0.0, 0.1, 0.5, 0.9, 0.999999]
        selected = policy.sample_actions_batch(env, states, thresholds)
        scalar_rows = [
            policy._legacy_single_probabilities(env, state)
            for state in states
        ]
        expected = []
        for state, probabilities, threshold in zip(
            states, scalar_rows, thresholds
        ):
            legal = [int(action) for action in env.legal_actions(state)]
            cumulative = 0.0
            fallback = legal[-1]
            choice = fallback
            for action in legal:
                probability = float(probabilities[action])
                if probability <= 0.0:
                    continue
                fallback = action
                cumulative += probability
                if threshold <= cumulative + 1e-12:
                    choice = action
                    break
            else:
                choice = fallback
            expected.append(choice)
        self.assertEqual(selected, expected)

    def test_new_execution_path_preserves_current_resolver_mathematics(self):
        public_range = BlueprintPublicRange()
        public_range.reset(self.state.hole[int(self.state.to_act)])
        snapshot = public_range.snapshot()
        current_policy = self._fixed_snapshot_policy(
            native_batch_encoding=False
        )
        new_policy = self._fixed_snapshot_policy(native_batch_encoding=True)
        blueprint = current_policy.probabilities(self.env, self.state)
        current = HeadsUpRootPolicySearch(
            current_policy,
            time_budget_ms=60_000,
            max_rollouts=70,
            batch_iterations=10,
            use_native_rollouts=True,
            use_batched_action_sampling=False,
            use_batch_step=False,
            seed=101,
        ).resolve(self.env, self.state, blueprint, snapshot)
        optimized = HeadsUpRootPolicySearch(
            new_policy,
            time_budget_ms=60_000,
            max_rollouts=70,
            batch_iterations=10,
            use_native_rollouts=True,
            use_batched_action_sampling=True,
            use_batch_step=True,
            seed=101,
        ).resolve(self.env, self.state, blueprint, snapshot)
        self.assertEqual(current.cfr_iterations, optimized.cfr_iterations)
        self.assertEqual(current.terminal_rollouts, optimized.terminal_rollouts)
        for old, new in zip(current.candidates, optimized.candidates):
            self.assertEqual(old.action.action, new.action.action)
            self.assertEqual(old.samples, new.samples)
            self.assertAlmostEqual(
                old.expected_final_payoff_bb,
                new.expected_final_payoff_bb,
                places=7,
            )
            self.assertAlmostEqual(
                old.strategy_probability,
                new.strategy_probability,
                places=7,
            )
    def test_cli_switch_preserves_both_search_modes(self):
        self.assertEqual(parse_args(["--search-mode", "three-player"]).search_mode, "three-player")
        self.assertEqual(parse_args(["--search-mode", "cfr"]).search_mode, "cfr")
        self.assertEqual(parse_args([]).root_range_mode, "inferred")
        self.assertEqual(parse_args([]).root_search_ms, 10_000)
        self.assertEqual(parse_args([]).root_blueprint_weight, 0.65)
        self.assertEqual(
            parse_args(
                ["--root-blueprint-weight", "0.80"]
            ).root_blueprint_weight,
            0.80,
        )
        self.assertEqual(
            parse_args(
                ["--root-min-strategy-probability", "0.10"]
            ).root_min_strategy_probability,
            0.10,
        )

    def test_tempering_and_uniform_contamination_keep_range_alive(self):
        source = PublicRangeSnapshot(
            combos=((0, 1), (2, 3), (4, 5)),
            weights=(0.999, 0.001, 0.0),
            effective_sample_size=1.0,
            updates=2,
        )
        robust = robust_inferred_range(source)
        self.assertAlmostEqual(sum(robust.weights), 1.0)
        self.assertTrue(all(weight >= 0.25 / 3 for weight in robust.weights))
        self.assertEqual(robust.updates, 2)
        summary = summarize_public_range(robust, actual_hole=(0, 1))
        self.assertEqual(summary["updates"], 2)
        self.assertTrue(summary["top_classes"])
        self.assertEqual(summary["actual_human_combo_card_ids"], [0, 1])
        self.assertAlmostEqual(
            summary["actual_human_combo_probability"],
            robust.weights[0],
        )
        self.assertEqual(summary["actual_human_combo_rank"], 1)
        self.assertAlmostEqual(
            sum(summary["class_probabilities"].values()),
            1.0,
        )
        self.assertEqual(range_probability_color(0.0, 0.1), "#0e151b")
        self.assertEqual(range_probability_color(0.1, 0.1), "#f4d03f")


if __name__ == "__main__":
    unittest.main()
