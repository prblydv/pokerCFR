import copy
from pathlib import Path
import tempfile
import unittest

import torch

from heads_up_engine import ACTION_NAMES, HeadsUpHoldemEngine
from heads_up_models import (
    build_policy_network,
    encoder_metadata,
    information_state_size,
)
from heads_up_pluribus_search import (
    BlueprintPublicRange,
    MultiprocessPluribusSearch,
    generate_search_actions,
    observed_action_likelihoods,
    sanitize_search_state,
)
from play_heads_up_gui import HeadsUpSnapshotPolicy


def _write_test_snapshot(path: Path) -> None:
    input_dim = information_state_size()
    networks = [
        build_policy_network("residual_mlp", input_dim, hidden=16, blocks=0)
        for _ in range(2)
    ]
    torch.save(
        {
            "version": 2,
            "kind": "heads_up_policy_snapshot",
            "iteration": 1,
            "input_dim": input_dim,
            "hidden": 16,
            "blocks": 0,
            "network_architecture": "residual_mlp",
            "max_history": 32,
            "action_names": tuple(ACTION_NAMES),
            "environment": {
                "starting_stack": 200,
                "small_blind": 1,
                "big_blind": 2,
            },
            "policy_nets": [network.state_dict() for network in networks],
            "metadata": {"encoder": encoder_metadata()},
        },
        path,
    )


class PublicRangeTests(unittest.TestCase):
    def test_range_is_blocker_aware_normalized_and_per_hand(self):
        public_range = BlueprintPublicRange()
        public_range.reset([0, 1])
        self.assertEqual(len(public_range.combos), 1225)
        self.assertAlmostEqual(sum(public_range.weights), 1.0)
        self.assertTrue(all(0 not in combo and 1 not in combo for combo in public_range.combos))
        public_range.filter_known([0, 1, 2, 3, 4])
        self.assertEqual(len(public_range.combos), 1081)
        self.assertAlmostEqual(sum(public_range.weights), 1.0)
        public_range.condition([1.0] * len(public_range.combos))
        self.assertEqual(public_range.snapshot().updates, 1)
        public_range.reset([0, 1])
        self.assertEqual(public_range.snapshot().updates, 0)

    def test_bayesian_conditioning_changes_hand_mass(self):
        public_range = BlueprintPublicRange()
        public_range.reset(range(50))
        self.assertEqual(len(public_range.combos), 1)
        public_range.condition([0.25])
        self.assertEqual(public_range.weights, [1.0])
        self.assertEqual(public_range.snapshot().effective_sample_size, 1.0)

    def test_exact_raise_likelihood_uses_policy_sizing_kernel(self):
        env = HeadsUpHoldemEngine(seed=3)
        state = env.new_hand(button=0)
        legal = env.legal_actions(state)
        probabilities = torch.zeros(2, 10)
        raise_actions = [
            action
            for action in legal
            if ACTION_NAMES[action] not in {"fold", "check", "call"}
        ]
        probabilities[0, raise_actions[0]] = 1.0
        probabilities[1, raise_actions[-1]] = 1.0
        likelihoods = observed_action_likelihoods(
            env,
            state,
            probabilities,
            kind="raise",
            raise_to=7,
        )
        self.assertEqual(len(likelihoods), 2)
        self.assertTrue(all(value > 0.0 for value in likelihoods))
        self.assertNotEqual(likelihoods[0], likelihoods[1])


class PluribusSearchContractTests(unittest.TestCase):
    def setUp(self):
        self.env = HeadsUpHoldemEngine(seed=17)

    def test_worker_input_removes_every_unknown_card_value(self):
        state = self.env.new_hand(button=0)
        hero = int(state.to_act)
        sanitized = sanitize_search_state(state, hero)
        self.assertEqual(sanitized.hole[hero], state.hole[hero])
        self.assertEqual(sanitized.hole[1 - hero], [None, None])
        self.assertTrue(all(card is None for card in sanitized.deck))
        self.assertTrue(all(card is None for card in sanitized.burned))

    def test_sanitized_input_is_invariant_to_actual_opponent_cards(self):
        first = self.env.new_hand(button=0)
        second = copy.deepcopy(first)
        opponent = 1 - int(first.to_act)
        replacement = second.deck[-2:]
        second.deck[-2:] = second.hole[opponent]
        second.hole[opponent] = replacement
        a = sanitize_search_state(first, int(first.to_act))
        b = sanitize_search_state(second, int(second.to_act))
        self.assertEqual(a.hole, b.hole)
        self.assertEqual(a.deck, b.deck)

    def test_root_contains_exact_legal_sizes_outside_blueprint(self):
        state = self.env.new_hand(button=0)
        blueprint = torch.ones(10) / 10.0
        abstract_targets = {
            self.env.action_target(state, action)
            for action in self.env.legal_actions(state)
        }
        candidates = generate_search_actions(self.env, state, blueprint)
        exact = [
            candidate
            for candidate in candidates
            if candidate.kind == "raise_to"
            and candidate.raise_to not in abstract_targets
        ]
        self.assertTrue(exact)
        for candidate in exact:
            child = self.env.step_exact(state, "raise_to", candidate.raise_to)
            self.assertIsNone(child.history[-1].action)

    def test_multiprocess_cfr_returns_legal_mixed_strategy_and_intervals(self):
        state = self.env.new_hand(button=0)
        hero = int(state.to_act)
        public_range = BlueprintPublicRange()
        public_range.reset(state.hole[hero])
        blueprint = torch.ones(10) / 10.0
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "policy.pt"
            _write_test_snapshot(snapshot)
            search = MultiprocessPluribusSearch(
                snapshot,
                workers=2,
                time_budget_seconds=4.0,
                iteration_cap_per_worker=4,
                depth_limit=1,
                seed=19,
            )
            try:
                result = search.resolve(
                    self.env,
                    state,
                    blueprint,
                    public_range.snapshot(),
                )
            finally:
                search.close(wait_for_workers=True)
        self.assertGreater(result.cfr_iterations, 0)
        self.assertGreater(result.terminal_rollouts, 0)
        self.assertAlmostEqual(
            sum(item.strategy_probability for item in result.candidates),
            1.0,
            places=5,
        )
        self.assertTrue(all(item.samples > 0 for item in result.candidates))
        choice = result.choice
        if choice.kind == "abstract":
            child = self.env.step(state, choice.action)
        else:
            child = self.env.step_exact(state, "raise_to", choice.raise_to)
        self.assertEqual(len(child.history), len(state.history) + 1)


class Iteration600SafetyRegressionTests(unittest.TestCase):
    policy_path = Path(
        "artifacts/heads_up_v4_paper3x/snapshots/policy_00000600.pt"
    )

    @unittest.skipUnless(policy_path.is_file(), "iteration-600 snapshot absent")
    def test_a9s_catastrophic_shove_is_safety_pruned(self):
        from benchmark_heads_up_a9_flop import build_state

        policy = HeadsUpSnapshotPolicy(
            self.policy_path,
            mode="sample",
            seed=41,
        )
        env, state, public_range = build_state(policy)
        search = MultiprocessPluribusSearch(
            self.policy_path,
            workers=2,
            time_budget_seconds=4.0,
            validation_rollouts=24,
            all_in_validation_samples=20_000,
            seed=73,
        )
        try:
            result = search.resolve(
                env,
                state,
                policy.probabilities(env, state),
                public_range.snapshot(),
            )
        finally:
            search.close(wait_for_workers=True)
        shove = next(
            row
            for row in result.candidates
            if row.action.kind == "abstract"
            and ACTION_NAMES[int(row.action.action)] == "all_in"
        )
        self.assertEqual(shove.strategy_probability, 0.0)
        self.assertTrue(shove.safety_pruned)
        self.assertLess(shove.validation_ci95_high_bb, 0.0)
        self.assertNotEqual(result.choice, shove.action)
        self.assertFalse(result.used_blueprint_fallback)

    @unittest.skipUnless(policy_path.is_file(), "iteration-600 snapshot absent")
    def test_a7_oversized_flop_shove_requires_positive_proof(self):
        from benchmark_heads_up_search_regressions import build_a7_state

        policy = HeadsUpSnapshotPolicy(
            self.policy_path,
            mode="sample",
            seed=41,
        )
        env, state, public_range = build_a7_state(policy)
        search = MultiprocessPluribusSearch(
            self.policy_path,
            workers=2,
            time_budget_seconds=4.0,
            validation_rollouts=24,
            all_in_validation_samples=20_000,
            seed=73,
        )
        try:
            result = search.resolve(
                env,
                state,
                policy.probabilities(env, state),
                public_range.snapshot(),
            )
        finally:
            search.close(wait_for_workers=True)
        shove = next(
            row
            for row in result.candidates
            if row.action.kind == "abstract"
            and ACTION_NAMES[int(row.action.action)] == "all_in"
        )
        self.assertEqual(shove.strategy_probability, 0.0)
        self.assertTrue(shove.safety_pruned)
        self.assertNotEqual(result.choice, shove.action)
        self.assertFalse(result.used_blueprint_fallback)

    @unittest.skipUnless(policy_path.is_file(), "iteration-600 snapshot absent")
    def test_ajo_button_search_never_returns_the_blueprint_fold_tail(self):
        from benchmark_heads_up_search_regressions import build_aj_state

        policy = HeadsUpSnapshotPolicy(
            self.policy_path,
            mode="sample",
            seed=41,
        )
        env, state, public_range = build_aj_state(policy)
        search = MultiprocessPluribusSearch(
            self.policy_path,
            workers=2,
            time_budget_seconds=4.0,
            validation_rollouts=24,
            seed=73,
        )
        try:
            result = search.resolve(
                env,
                state,
                policy.probabilities(env, state),
                public_range.snapshot(),
            )
        finally:
            search.close(wait_for_workers=True)
        self.assertFalse(result.used_blueprint_fallback)
        self.assertNotEqual(result.choice.label, "fold")

    @unittest.skipUnless(policy_path.is_file(), "iteration-600 snapshot absent")
    def test_hand7_k3_folds_to_flop_shove_under_range_ambiguity(self):
        from benchmark_heads_up_search_regressions import (
            build_hand7_call_state,
        )

        policy = HeadsUpSnapshotPolicy(
            self.policy_path,
            mode="sample",
            seed=41,
        )
        env, state, public_range = build_hand7_call_state(policy)
        search = MultiprocessPluribusSearch(
            self.policy_path,
            workers=2,
            time_budget_seconds=4.0,
            validation_rollouts=24,
            all_in_validation_samples=20_000,
            seed=73,
        )
        try:
            result = search.resolve(
                env,
                state,
                policy.probabilities(env, state),
                public_range.snapshot(),
            )
        finally:
            search.close(wait_for_workers=True)
        call = next(
            row
            for row in result.candidates
            if row.action.kind == "abstract"
            and ACTION_NAMES[int(row.action.action)] == "call"
        )
        self.assertEqual(result.choice.label, "fold")
        self.assertEqual(call.strategy_probability, 0.0)
        self.assertTrue(call.safety_pruned)
        self.assertLess(call.validation_ev_bb, -49.0)
        self.assertFalse(result.used_blueprint_fallback)

    @unittest.skipUnless(policy_path.is_file(), "iteration-600 snapshot absent")
    def test_hand7_earlier_streets_remain_search_authoritative(self):
        from benchmark_heads_up_search_regressions import (
            build_hand7_flop_state,
            build_hand7_preflop_state,
        )

        policy = HeadsUpSnapshotPolicy(
            self.policy_path,
            mode="sample",
            seed=41,
        )
        search = MultiprocessPluribusSearch(
            self.policy_path,
            workers=2,
            time_budget_seconds=4.0,
            validation_rollouts=24,
            seed=73,
        )
        try:
            results = []
            for builder in (
                build_hand7_preflop_state,
                build_hand7_flop_state,
            ):
                env, state, public_range = builder(policy)
                results.append(
                    search.resolve(
                        env,
                        state,
                        policy.probabilities(env, state),
                        public_range.snapshot(),
                    )
                )
        finally:
            search.close(wait_for_workers=True)
        self.assertTrue(
            all(not result.used_blueprint_fallback for result in results)
        )
        self.assertTrue(
            all(
                result.choice.action is not None
                or result.choice.raise_to is not None
                for result in results
            )
        )


if __name__ == "__main__":
    unittest.main()
