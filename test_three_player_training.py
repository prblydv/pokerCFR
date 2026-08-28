import copy
import random
import tempfile
import unittest
from pathlib import Path

import torch

from three_player_cfr import ReservoirBuffer, ThreePlayerNeuralCFR
from three_player_engine import ThreePlayerHoldemEnv
from three_player_models import (
    CARD_FEATURES,
    CARD_STATE_PREFIX_FEATURES,
    CARD_TOKEN_COUNT,
    HISTORY_OFFSET,
    DeepCFRBranchV2Network,
    encode_information_state,
    masked_softmax,
    poker_relational_features,
)


class ThreePlayerTrainingTests(unittest.TestCase):
    def setUp(self):
        self.env = ThreePlayerHoldemEnv(stack_size=12, sb=1, bb=2, seed=7)

    def test_encoder_does_not_expose_opponent_hole_cards(self):
        state = self.env.new_hand(button=0)
        hero = state.to_act
        legal = self.env.legal_actions(state)
        original = encode_information_state(state, hero, legal, self.env.stack_size)

        changed = copy.deepcopy(state)
        opponent = (hero + 1) % 3
        # Encoding privacy should hold even for a deliberately synthetic hand.
        changed.hole[opponent] = [50, 51]
        modified = encode_information_state(changed, hero, legal, self.env.stack_size)
        self.assertTrue(torch.equal(original, modified))

    def test_regret_matching_uses_highest_legal_fallback(self):
        values = torch.tensor([-3.0, -2.0, 99.0, -1.0])
        mask = torch.tensor([1.0, 1.0, 0.0, 1.0])
        strategy = ThreePlayerNeuralCFR.regret_matching(values, mask)
        expected = torch.tensor([0.0, 0.0, 0.0, 1.0])
        self.assertTrue(torch.allclose(strategy, expected))

    def test_vectorized_regret_matching_is_bit_exact(self):
        generator = torch.Generator().manual_seed(72_041)
        values = torch.randn((20_000, 9), generator=generator)
        masks = (torch.rand((20_000, 9), generator=generator) > 0.45).float()
        masks[:, 0] = 1.0
        # Exercise all-negative fallback, exact ties, zeros and positive regrets.
        values[:4000] = -values[:4000].abs() - 0.01
        values[4000:8000] = 0.0
        values[8000:9000, :3] = 2.0
        reference = torch.stack(
            [
                ThreePlayerNeuralCFR.regret_matching(row, mask)
                for row, mask in zip(values, masks)
            ]
        )
        candidate = ThreePlayerNeuralCFR.regret_matching_batch(values, masks)
        self.assertTrue(torch.equal(reference, candidate))

    def test_packed_reservoir_can_bootstrap_to_a_larger_capacity(self):
        original = ReservoirBuffer(4, random.Random(13))
        for index in range(4):
            original.add((torch.tensor([float(index)]), torch.tensor(float(index))))

        restored = ReservoirBuffer(1, random.Random(13))
        restored.load_state_dict(original.state_dict())
        restored.resize_capacity(10)

        self.assertEqual(restored.capacity, 10)
        self.assertEqual(len(restored), 10)
        self.assertGreaterEqual(restored.seen, 10)
        fields = restored.state_dict()["fields"]
        self.assertEqual([int(field.shape[0]) for field in fields], [10, 10])
        retained = set(int(value) for value in fields[0][:4, 0].tolist())
        self.assertEqual(retained, {0, 1, 2, 3})

    def test_packed_reservoir_field_sampling_matches_row_sampling(self):
        items = [
            (
                torch.tensor([index, index + 0.5]),
                torch.tensor([index * 2.0]),
                torch.tensor([index % 2], dtype=torch.bool),
                torch.tensor(float(index + 1)),
            )
            for index in range(8)
        ]
        row_buffer = ReservoirBuffer(8, random.Random(77))
        field_buffer = ReservoirBuffer(8, random.Random(77))
        for item in items:
            row_buffer.add(item)
            field_buffer.add(tuple(value.clone() for value in item))

        rows = row_buffer.sample(5)
        fields = field_buffer.sample_fields(5)
        expected = tuple(torch.stack(field) for field in zip(*rows))
        self.assertEqual(len(fields), len(expected))
        for actual, wanted in zip(fields, expected):
            self.assertTrue(torch.equal(actual, wanted))

        expected_weight_mean = sum(float(item[3]) for item in items) / len(items)
        self.assertAlmostEqual(
            ThreePlayerNeuralCFR._buffer_weight_mean(field_buffer),
            expected_weight_mean,
        )

    def test_packed_merge_matches_row_merge_exactly(self):
        capacity = 7
        initial = [
            (
                torch.tensor([index, index + 0.25]),
                torch.tensor([index * 3.0]),
                torch.tensor(float(index + 1)),
            )
            for index in range(capacity)
        ]
        candidates = [
            (
                torch.tensor([100 + index, 100.25 + index]),
                torch.tensor([300.0 + index]),
                torch.tensor(float(200 + index)),
            )
            for index in range(40)
        ]
        fields = [torch.stack(values) for values in zip(*candidates)]

        row_buffer = ReservoirBuffer(capacity, random.Random(710))
        packed_buffer = ReservoirBuffer(capacity, random.Random(710))
        for item in initial:
            row_buffer.add(item)
            packed_buffer.add(tuple(value.clone() for value in item))
        for item in candidates:
            row_buffer.add(tuple(value.clone() for value in item))
        ThreePlayerNeuralCFR._merge_packed_samples(
            [fields], [packed_buffer]
        )

        self.assertEqual(row_buffer.seen, packed_buffer.seen)
        self.assertEqual(row_buffer.rng.getstate(), packed_buffer.rng.getstate())
        row_fields = row_buffer.state_dict()["fields"]
        packed_fields = packed_buffer.state_dict()["fields"]
        for expected, actual in zip(row_fields, packed_fields):
            self.assertTrue(torch.equal(expected, actual))

    def test_shuffled_reservoir_epoch_covers_every_entry_once(self):
        items = [
            (
                torch.tensor([index], dtype=torch.float32),
                torch.tensor([index * 2.0]),
                torch.ones(1),
                torch.tensor(float(index + 1)),
            )
            for index in range(10)
        ]
        first = ReservoirBuffer(10, random.Random(91))
        second = ReservoirBuffer(10, random.Random(91))
        for item in items:
            first.add(item)
            second.add(tuple(value.clone() for value in item))

        first_batches = list(first.shuffled_field_batches(size=4, steps=3))
        second_batches = list(second.shuffled_field_batches(size=4, steps=3))
        self.assertEqual([len(batch[0]) for batch in first_batches], [4, 4, 2])
        seen = torch.cat([batch[0][:, 0] for batch in first_batches]).to(torch.int64)
        self.assertEqual(sorted(seen.tolist()), list(range(10)))
        for left, right in zip(first_batches, second_batches):
            for left_field, right_field in zip(left, right):
                self.assertTrue(torch.equal(left_field, right_field))

    def test_street_balanced_batches_include_every_available_street(self):
        buffer = ReservoirBuffer(16, random.Random(92))
        for street in range(4):
            for item in range(street + 1):
                observation = torch.zeros(12)
                observation[street] = 1.0
                observation[4] = float(item)
                buffer.add(
                    (
                        observation,
                        torch.zeros(9),
                        torch.ones(9),
                        torch.tensor(1.0),
                    )
                )
        batches = list(buffer.street_balanced_field_batches(size=8, steps=3))
        self.assertEqual(len(batches), 3)
        for observations, *_ in batches:
            counts = torch.bincount(
                observations[:, :4].argmax(dim=1), minlength=4
            )
            self.assertEqual(counts.tolist(), [2, 2, 2, 2])

    def test_deep_cfr_v2_features_and_street_heads(self):
        trainer = ThreePlayerNeuralCFR(
            self.env,
            hidden=128,
            blocks=3,
            network_architecture="deep_cfr_branch_v2",
            max_history=4,
            include_tournament_features=True,
            seed=93,
        )
        state = self.env.new_hand(button=0)
        observation = trainer.encode(state, state.to_act)
        cards = observation[
            CARD_STATE_PREFIX_FEATURES:HISTORY_OFFSET
        ].reshape(1, CARD_TOKEN_COUNT, CARD_FEATURES)
        features = poker_relational_features(
            cards, observation[:4].reshape(1, 4)
        )
        self.assertTrue(torch.isfinite(features).all())
        network = trainer.policy_nets[0]
        self.assertIsInstance(network, DeepCFRBranchV2Network)
        with torch.no_grad():
            for street, head in enumerate(network.street_heads):
                head.weight.zero_()
                head.bias.fill_(float(street + 1))
        for street in range(4):
            synthetic = observation.clone()
            synthetic[:4] = 0.0
            synthetic[street] = 1.0
            output = network(synthetic.unsqueeze(0))[0]
            self.assertTrue(
                torch.allclose(output, torch.full_like(output, float(street + 1)))
            )

    def test_regret_matching_is_invariant_to_positive_target_scaling(self):
        values = torch.tensor([3.0, -7.0, 1.5, 99.0])
        mask = torch.tensor([1.0, 1.0, 1.0, 0.0])
        original = ThreePlayerNeuralCFR.regret_matching(values, mask)
        normalised = values / (values.abs() * mask).amax().clamp(min=1.0)
        self.assertTrue(
            torch.allclose(
                original,
                ThreePlayerNeuralCFR.regret_matching(normalised, mask),
            )
        )

    def test_exploration_does_not_decay_and_keeps_legal_actions_reachable(self):
        trainer = ThreePlayerNeuralCFR(
            self.env,
            hidden=8,
            blocks=0,
            max_history=4,
            exploration=0.2,
            seed=30,
        )
        state = self.env.new_hand(button=0)
        trainer.iteration = 10_000
        _, probabilities, mask = trainer.current_strategy(state)
        legal_count = float(mask.sum())
        self.assertTrue(
            torch.all(probabilities[mask > 0] >= 0.2 / legal_count - 1e-7)
        )
        _, batched_probabilities, batched_mask = trainer._batched_current_strategies(
            [(state, int(state.to_act))]
        )[0]
        self.assertTrue(
            torch.all(
                batched_probabilities[batched_mask > 0]
                >= 0.2 / legal_count - 1e-7
            )
        )

    def test_traversal_worker_skips_policy_networks_and_optimizers(self):
        worker = ThreePlayerNeuralCFR(
            self.env,
            hidden=8,
            blocks=0,
            max_history=4,
            _traversal_worker=True,
            seed=33,
        )
        self.assertEqual(len(worker.advantage_nets), 3)
        self.assertEqual(worker.policy_nets, [])
        self.assertEqual(worker.advantage_optimizers, [])
        self.assertEqual(worker.policy_optimizers, [])

    def test_advantage_refit_interval_boundaries(self):
        trainer = ThreePlayerNeuralCFR(
            self.env,
            hidden=8,
            blocks=0,
            max_history=4,
            advantage_fit_every=25,
            seed=31,
        )
        self.assertFalse(trainer._should_fit_advantage(0))
        self.assertTrue(trainer._should_fit_advantage(1))
        self.assertTrue(trainer._should_fit_advantage(2))
        self.assertTrue(trainer._should_fit_advantage(24))
        self.assertTrue(trainer._should_fit_advantage(25))
        self.assertFalse(trainer._should_fit_advantage(26))
        self.assertFalse(trainer._should_fit_advantage(49))
        self.assertTrue(trainer._should_fit_advantage(50))
        with self.assertRaises(ValueError):
            ThreePlayerNeuralCFR(
                self.env, hidden=8, blocks=0, max_history=4,
                advantage_fit_every=0,
            )

    def test_advantage_reinitialization_can_start_after_bootstrap(self):
        trainer = ThreePlayerNeuralCFR(
            self.env,
            hidden=8,
            blocks=0,
            max_history=4,
            reinitialize_advantage_each_iteration=True,
            advantage_reinitialize_from_iteration=25,
            advantage_fit_every=1,
            seed=32,
        )
        self.assertFalse(trainer._should_reinitialize_advantage(1))
        self.assertFalse(trainer._should_reinitialize_advantage(24))
        self.assertTrue(trainer._should_reinitialize_advantage(25))
        self.assertTrue(trainer._should_reinitialize_advantage(26))

    def test_invalid_encoder_and_policy_calls_fail_loudly(self):
        state = self.env.new_hand(button=0)
        legal = self.env.legal_actions(state)
        with self.assertRaises(ValueError):
            encode_information_state(state, state.to_act, legal, self.env.stack_size, 0)
        with self.assertRaises(ValueError):
            masked_softmax(torch.zeros(9), torch.zeros(9))

        trainer = ThreePlayerNeuralCFR(
            self.env, hidden=8, blocks=0, max_history=4, seed=3
        )
        with self.assertRaises(ValueError):
            trainer.current_strategy(state, (state.to_act + 1) % 3)

    def test_each_traverser_cycles_through_all_three_positions(self):
        trainer = ThreePlayerNeuralCFR(
            self.env,
            hidden=8,
            blocks=0,
            max_history=4,
            max_nodes_per_traversal=10,
            seed=4,
        )
        observed = []

        def record_only(state, traverser, reach, depth):
            observed.append((traverser, state.button))
            return 0.0

        trainer._traverse = record_only
        for _ in range(3):
            trainer.train_iteration(
                traversals_per_player=1,
                advantage_steps=1,
                policy_steps=0,
                batch_size=4,
            )
        for traverser in range(3):
            buttons = [button for player, button in observed if player == traverser]
            self.assertEqual(buttons, [traverser, (traverser - 1) % 3, (traverser - 2) % 3])

    def test_tiny_training_iteration_and_checkpoint_roundtrip(self):
        trainer = ThreePlayerNeuralCFR(
            self.env,
            hidden=16,
            blocks=0,
            max_history=4,
            max_nodes_per_traversal=40,
            max_depth=24,
            advantage_capacity=200,
            policy_capacity=200,
            seed=9,
        )
        metrics = trainer.train_iteration(
            traversals_per_player=1,
            advantage_steps=1,
            policy_steps=1,
            batch_size=8,
        )
        self.assertEqual(trainer.iteration, 1)
        self.assertEqual(trainer.last_fitted_iteration, 1)
        self.assertGreater(metrics["nodes"], 0)
        self.assertTrue(any(len(buffer) for buffer in trainer.advantage_buffers))

        state = self.env.new_hand(button=1)
        probabilities = trainer.average_policy(state)
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=5)
        for action in range(len(probabilities)):
            if action not in self.env.legal_actions(state):
                self.assertEqual(float(probabilities[action]), 0.0)

        dealer_rng_before = trainer.env.rng.getstate()
        button_before = trainer.env._last_button
        evaluation_one = trainer.evaluate_vs_random(games_per_player=3)
        evaluation_two = trainer.evaluate_vs_random(games_per_player=3)
        self.assertEqual(evaluation_one, evaluation_two)
        self.assertEqual(trainer.env.rng.getstate(), dealer_rng_before)
        self.assertEqual(trainer.env._last_button, button_before)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            trainer.save(path, include_buffers=True)
            compact_payload = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(
                compact_payload["advantage_buffers"][0]["format_version"], 2
            )
            self.assertIn("fields", compact_payload["policy_buffers"][0])
            self.assertNotIn("memory", compact_payload["policy_buffers"][0])
            expected_next_hand = trainer.env.new_hand()
            restored_env = ThreePlayerHoldemEnv(stack_size=12, sb=1, bb=2, seed=999)
            restored = ThreePlayerNeuralCFR.load(path, restored_env, device="cpu")
            self.assertEqual(restored.iteration, trainer.iteration)
            self.assertEqual(
                restored.last_fitted_iteration, trainer.last_fitted_iteration
            )
            actual_next_hand = restored.env.new_hand()
            self.assertEqual(actual_next_hand.button, expected_next_hand.button)
            self.assertEqual(actual_next_hand.hole, expected_next_hand.hole)
            self.assertEqual(actual_next_hand.deck, expected_next_hand.deck)
            self.assertEqual(
                [len(buffer) for buffer in restored.advantage_buffers],
                [len(buffer) for buffer in trainer.advantage_buffers],
            )
            for original, loaded in zip(trainer.policy_nets, restored.policy_nets):
                for original_value, loaded_value in zip(
                    original.state_dict().values(), loaded.state_dict().values()
                ):
                    self.assertTrue(torch.equal(original_value.cpu(), loaded_value.cpu()))
            for buffer in restored.advantage_buffers + restored.policy_buffers:
                for item in buffer.memory:
                    self.assertTrue(all(value.device.type == "cpu" for value in item))
            snapshot = trainer.policy_snapshot()
            report = restored.evaluate_vs_snapshot(snapshot, games_per_player=3)
            self.assertIn("mean_ev_bb", report)

            light_path = Path(directory) / "inference_only.pt"
            trainer.save(light_path, include_buffers=False)
            inference_only = ThreePlayerNeuralCFR.load(
                light_path,
                ThreePlayerHoldemEnv(stack_size=12, sb=1, bb=2, seed=1000),
            )
            self.assertFalse(inference_only.can_resume_training)
            with self.assertRaises(RuntimeError):
                inference_only.train_iteration(
                    traversals_per_player=1,
                    advantage_steps=1,
                    policy_steps=1,
                    batch_size=4,
                )

    def test_parallel_cpu_root_collection_and_fitting(self):
        trainer = ThreePlayerNeuralCFR(
            self.env,
            hidden=8,
            blocks=0,
            max_history=4,
            max_nodes_per_traversal=8,
            max_depth=6,
            advantage_capacity=100,
            policy_capacity=100,
            reinitialize_advantage_each_iteration=False,
            seed=17,
        )
        advantage_network_ids = [id(network) for network in trainer.advantage_nets]
        metrics = trainer.train_iteration(
            traversals_per_player=1,
            traversal_workers=2,
            advantage_steps=1,
            policy_steps=1,
            batch_size=4,
        )
        self.assertEqual(metrics["traversal_workers"], 2.0)
        self.assertGreater(metrics["nodes"], 0.0)
        self.assertTrue(any(len(buffer) for buffer in trainer.advantage_buffers))
        self.assertTrue(any(len(buffer) for buffer in trainer.policy_buffers))
        self.assertEqual(trainer.last_fitted_iteration, 1)
        self.assertEqual(
            [id(network) for network in trainer.advantage_nets],
            advantage_network_ids,
        )

    def test_training_forces_autograd_and_recovers_an_incomplete_fit(self):
        trainer = ThreePlayerNeuralCFR(
            self.env,
            hidden=8,
            blocks=0,
            max_history=4,
            max_nodes_per_traversal=12,
            advantage_capacity=100,
            policy_capacity=100,
            reinitialize_advantage_each_iteration=False,
            seed=19,
        )
        with torch.no_grad():
            trainer.train_iteration(
                traversals_per_player=1,
                advantage_steps=1,
                policy_steps=1,
                batch_size=4,
            )
        self.assertEqual(trainer.last_fitted_iteration, 1)
        trainer.iteration = 2
        recovery = trainer.recover_incomplete_fit(
            advantage_steps=1, policy_steps=1, batch_size=4
        )
        self.assertEqual(trainer.last_fitted_iteration, 2)
        self.assertEqual(recovery["previous_last_fitted_iteration"], 1.0)
        self.assertIn("recovery_fit_seconds", recovery)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_training_metrics_and_resume(self):
        env = ThreePlayerHoldemEnv(stack_size=12, sb=1, bb=2, seed=21)
        trainer = ThreePlayerNeuralCFR(
            env,
            device="cuda",
            hidden=8,
            blocks=0,
            max_history=4,
            max_nodes_per_traversal=12,
            advantage_capacity=100,
            policy_capacity=100,
            reinitialize_advantage_each_iteration=False,
            seed=22,
        )
        first = trainer.train_iteration(
            traversals_per_player=1,
            advantage_steps=1,
            policy_steps=1,
            batch_size=4,
        )
        self.assertGreater(first["gpu_peak_memory_mb"], 0.0)
        self.assertTrue(
            all(
                next(network.parameters()).is_cuda
                for network in trainer.advantage_nets + trainer.policy_nets
            )
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cuda_resume.pt"
            trainer.save(path, include_buffers=True)
            restored = ThreePlayerNeuralCFR.load(
                path,
                ThreePlayerHoldemEnv(stack_size=12, sb=1, bb=2, seed=23),
                device="cuda",
            )
            for buffer in restored.advantage_buffers + restored.policy_buffers:
                for item in buffer.memory:
                    self.assertTrue(all(value.device.type == "cpu" for value in item))
            second = restored.train_iteration(
                traversals_per_player=1,
                advantage_steps=1,
                policy_steps=1,
                batch_size=4,
            )
            self.assertGreater(second["gpu_peak_memory_mb"], 0.0)


if __name__ == "__main__":
    unittest.main()
