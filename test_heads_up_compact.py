import copy
import random
import tempfile
import unittest
from pathlib import Path

import torch

from heads_up_compact import (
    COMPACT_ACTION_FEATURES,
    COMPACT_CARD_OFFSET,
    COMPACT_DEFAULT_MAX_HISTORY,
    COMPACT_ENCODER_SCHEMA_VERSION,
    COMPACT_HISTORY_FEATURES,
    COMPACT_HISTORY_OFFSET,
    compact_action_offset,
    compact_encoder_metadata,
    compact_information_state_size,
    encode_compact_information_state,
)
from heads_up_engine import (
    ACTION_CALL,
    ACTION_CHECK,
    ACTION_FOLD,
    ACTION_MIN_RAISE,
    HeadsUpHoldemEngine,
)
from heads_up_cfr import HeadsUpNeuralCFR
from heads_up_models import (
    COMPACT_V6_ARCHITECTURE,
    COMPACT_V6_POLICY_RANGE_ARCHITECTURE,
    build_action_descriptors,
    build_policy_network,
)
from heads_up_production import (
    _collect_independent_range_training_hands,
    load_policy_snapshot,
    range_reservoir_statistics,
    save_policy_snapshot,
)
from heads_up_ranges import opponent_combo_index, valid_combo_mask_from_encoded


def encode(env, state, max_history=COMPACT_DEFAULT_MAX_HISTORY):
    legal = env.legal_actions(state)
    return encode_compact_information_state(
        state,
        int(state.to_act),
        legal,
        env.bb,
        max_history,
        action_descriptors=build_action_descriptors(env, state),
    )


class CompactPythonEncoderTests(unittest.TestCase):
    def setUp(self):
        self.env = HeadsUpHoldemEngine(seed=311)

    def test_locked_width_and_metadata(self):
        self.assertEqual(compact_information_state_size(), 782)
        metadata = compact_encoder_metadata()
        self.assertEqual(metadata["input_dim"], 782)
        self.assertEqual(
            metadata["encoder_schema_version"], COMPACT_ENCODER_SCHEMA_VERSION
        )
        self.assertEqual(metadata["history_policy"], "full_error_on_overflow")

    def test_cards_use_seven_exact_ids_with_zero_padding(self):
        state = self.env.new_hand(button=0)
        result = encode(self.env, state)
        cards = result[COMPACT_CARD_OFFSET : COMPACT_CARD_OFFSET + 7]
        expected = sorted(card + 1 for card in state.hole[int(state.to_act)])
        self.assertEqual(cards[:2].tolist(), [float(value) for value in expected])
        self.assertEqual(cards[2:].tolist(), [0.0] * 5)

    def test_hole_and_flop_order_are_canonical_but_turn_river_are_ordered(self):
        state = self.env.new_hand(button=0)
        state.hole[int(state.to_act)].reverse()
        state.board = [25, 2, 41, 19, 7]
        result = encode(self.env, state)
        cards = result[COMPACT_CARD_OFFSET : COMPACT_CARD_OFFSET + 7].tolist()
        self.assertEqual(cards[:2], sorted(cards[:2]))
        self.assertEqual(cards[2:5], [3.0, 26.0, 42.0])
        self.assertEqual(cards[5:], [20.0, 8.0])

    def test_history_is_chronological_and_exact_target_is_preserved(self):
        state = self.env.new_hand(button=0)
        state = self.env.step_exact(state, "raise_to", raise_to=7)
        result = encode(self.env, state)
        row = result[
            COMPACT_HISTORY_OFFSET : COMPACT_HISTORY_OFFSET
            + COMPACT_HISTORY_FEATURES
        ]
        total = float(sum(state.initial_stacks))
        self.assertEqual(int(row[0]), 1)
        self.assertEqual(int(row[2]), 5)
        self.assertAlmostEqual(float(row[3]), 7.0 / total, places=7)
        self.assertAlmostEqual(float(row[4]), 3.0 / total, places=7)

    def test_100bb_maximum_live_history_fits_without_truncation(self):
        state = self.env.new_hand(button=0)
        while not state.terminal and int(state.street) < 3:
            legal = self.env.legal_actions(state)
            action = ACTION_CHECK if ACTION_CHECK in legal else ACTION_CALL
            state = self.env.step(state, action)
        self.assertEqual(int(state.street), 3)
        state = self.env.step(state, ACTION_CHECK)
        while ACTION_MIN_RAISE in self.env.legal_actions(state):
            state = self.env.step(state, ACTION_MIN_RAISE)
        self.assertFalse(state.terminal)
        self.assertEqual(len(state.history), 106)
        result = encode(self.env, state)
        history = result[
            COMPACT_HISTORY_OFFSET : compact_action_offset()
        ].reshape(COMPACT_DEFAULT_MAX_HISTORY, COMPACT_HISTORY_FEATURES)
        self.assertTrue(torch.all(history[:, 0] > 0))
        with self.assertRaisesRegex(ValueError, "106 > 105"):
            encode(self.env, state, max_history=105)
        final = self.env.step(state, ACTION_CALL)
        self.assertTrue(final.terminal)
        self.assertEqual(len(final.history), 107)

    def test_action_pairs_are_legal_bit_and_exact_target_share(self):
        state = self.env.new_hand(button=0)
        result = encode(self.env, state)
        actions = result[compact_action_offset() :].reshape(10, COMPACT_ACTION_FEATURES)
        total = float(sum(state.initial_stacks))
        for action in range(10):
            if action in self.env.legal_actions(state):
                self.assertEqual(float(actions[action, 0]), 1.0)
                self.assertAlmostEqual(
                    float(actions[action, 1]),
                    self.env.action_target(state, action) / total,
                    places=7,
                )
            else:
                self.assertEqual(actions[action].tolist(), [0.0, 0.0])


class CompactNetworkTests(unittest.TestCase):
    def _trainer(self):
        return HeadsUpNeuralCFR(
            HeadsUpHoldemEngine(seed=700),
            hidden=32,
            blocks=1,
            advantage_capacity=128,
            policy_capacity=128,
            range_capacity=128,
            max_history=COMPACT_DEFAULT_MAX_HISTORY,
            max_nodes_per_traversal=40,
            max_depth=16,
            range_loss_weight=0.01,
            network_architecture=COMPACT_V6_ARCHITECTURE,
            policy_network_architecture=COMPACT_V6_POLICY_RANGE_ARCHITECTURE,
            encoder_schema_version=COMPACT_ENCODER_SCHEMA_VERSION,
            seed=701,
        )

    def test_hidden384_outputs_and_gradients_are_finite(self):
        env = HeadsUpHoldemEngine(seed=91)
        rows = []
        state = env.new_hand(button=0)
        for _ in range(6):
            rows.append(encode(env, state))
            legal = env.legal_actions(state)
            state = env.step(state, ACTION_CHECK if ACTION_CHECK in legal else ACTION_CALL)
            if state.terminal:
                break
        x = torch.stack(rows)
        network = build_policy_network(
            COMPACT_V6_ARCHITECTURE, 782, hidden=384, blocks=2
        )
        output = network(x)
        self.assertEqual(output.shape, (len(rows), 10))
        output.square().mean().backward()
        self.assertTrue(
            all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in network.parameters()
            )
        )

    def test_padding_values_are_masked(self):
        env = HeadsUpHoldemEngine(seed=123)
        state = env.new_hand(button=0)
        original = encode(env, state)
        changed = original.clone()
        history = changed[
            COMPACT_HISTORY_OFFSET : compact_action_offset()
        ].reshape(COMPACT_DEFAULT_MAX_HISTORY, COMPACT_HISTORY_FEATURES)
        history[:, 3:] = torch.randn_like(history[:, 3:])
        network = build_policy_network(
            COMPACT_V6_ARCHITECTURE, 782, hidden=384, blocks=2
        ).eval()
        with torch.inference_mode():
            left = network(original.unsqueeze(0))
            right = network(changed.unsqueeze(0))
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)

    def test_range_policy_contract_is_preserved(self):
        network = build_policy_network(
            COMPACT_V6_POLICY_RANGE_ARCHITECTURE,
            782,
            hidden=384,
            blocks=2,
        )
        actions, ranges = network.forward_with_range(torch.zeros(2, 782))
        self.assertEqual(actions.shape, (2, 10))
        self.assertEqual(ranges.shape, (2, 1326))

    def test_exact_card_ids_drive_blocker_mask(self):
        env = HeadsUpHoldemEngine(seed=444)
        state = env.new_hand(button=0)
        x = encode(env, state).unsqueeze(0)
        valid = valid_combo_mask_from_encoded(x)[0]
        hero = state.hole[int(state.to_act)]
        other = next(card for card in range(52) if card not in hero)
        self.assertFalse(bool(valid[opponent_combo_index((hero[0], other))]))
        expected = torch.tensor(1225, dtype=torch.long)  # C(50, 2), preflop.
        self.assertEqual(valid.sum(), expected)

    def test_padded_cards_do_not_unblock_exact_card_zero(self):
        env = HeadsUpHoldemEngine(seed=445)
        pop_order = [1, 0, 2, 3]
        deck = [card for card in range(52) if card not in pop_order]
        deck.extend(reversed(pop_order))
        state = env.new_hand(button=0, deck=deck)
        actor = int(state.to_act)
        self.assertIn(0, state.hole[actor])
        valid = valid_combo_mask_from_encoded(encode(env, state).unsqueeze(0))[0]
        other = next(card for card in range(1, 52) if card not in state.hole[actor])
        self.assertFalse(bool(valid[opponent_combo_index((0, other))]))
        self.assertEqual(int(valid.sum()), 1225)

    def test_compact_range_collection_training_and_dashboard_statistics(self):
        trainer = self._trainer()
        generated = _collect_independent_range_training_hands(
            trainer,
            profiles=("random",),
            hands=4,
            seed=801,
            reference_policy_nets=None,
            inference_batch_size=16,
            stack_depths_bb=(10, 20),
        )
        self.assertGreater(generated["range_samples_generated"], 0.0)
        row = trainer.train_iteration(
            traversals_per_player=1,
            advantage_steps=1,
            policy_steps=1,
            batch_size=8,
            range_batch_size=4,
        )
        self.assertTrue(torch.isfinite(torch.tensor(row["policy_range_loss"])))
        stats = range_reservoir_statistics(trainer, maximum_rows_per_player=128)
        self.assertEqual(stats["sampled_rows"], int(generated["range_samples_generated"]))
        self.assertAlmostEqual(sum(stats["street_percent"].values()), 100.0, places=4)

    def test_compact_checkpoint_and_policy_snapshot_round_trip(self):
        trainer = self._trainer()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "compact_checkpoint.pt"
            trainer.save(checkpoint)
            restored = HeadsUpNeuralCFR.load(
                checkpoint,
                HeadsUpHoldemEngine(seed=700),
                device="cpu",
            )
            self.assertEqual(restored.input_dim, 782)
            self.assertEqual(
                restored.encoder_schema_version, COMPACT_ENCODER_SCHEMA_VERSION
            )
            snapshot_path = Path(directory) / "compact_policy.pt"
            save_policy_snapshot(restored, snapshot_path)
            snapshot = load_policy_snapshot(snapshot_path)
            self.assertEqual(snapshot.metadata["input_dim"], 782)
            self.assertEqual(
                snapshot.metadata["encoder_schema_version"],
                COMPACT_ENCODER_SCHEMA_VERSION,
            )
            self.assertTrue(snapshot.metadata["has_range_head"])


class CompactNativeParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import heads_up_native_engine as native_module
            from heads_up_native import HeadsUpHoldemEngine as NativeEngine
        except ImportError:
            raise unittest.SkipTest("native extension is unavailable")
        if int(getattr(native_module, "NATIVE_ABI_VERSION", -1)) < 6:
            raise unittest.SkipTest("native ABI 6 is not active")
        cls.NativeEngine = NativeEngine

    def test_python_and_native_match_on_random_legal_states(self):
        chooser = random.Random(8_281)
        for hand in range(40):
            deck = list(range(52))
            random.Random(91_000 + hand).shuffle(deck)
            py_env = HeadsUpHoldemEngine(seed=1)
            native_env = self.NativeEngine(seed=1)
            py_state = py_env.new_hand(button=hand % 2, deck=deck)
            native_state = native_env.new_hand(button=hand % 2, deck=deck)
            while not py_state.terminal:
                py_legal = py_env.legal_actions(py_state)
                native_legal = native_env.legal_actions(native_state)
                self.assertEqual(py_legal, native_legal)
                expected = encode(py_env, py_state)
                actual = encode(native_env, native_state)
                torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-7)
                action = chooser.choice(py_legal)
                py_state = py_env.step(py_state, action)
                native_state = native_env.step(native_state, action)

    def test_native_compact_batch_matches_scalar(self):
        from heads_up_native import encode_compact_information_states_native

        env = self.NativeEngine(seed=7)
        states = [env.new_hand(button=index % 2) for index in range(12)]
        batch, masks = encode_compact_information_states_native(env, states)
        scalar = torch.stack([encode(env, state) for state in states]).numpy()
        self.assertEqual(batch.shape, (12, 782))
        self.assertEqual(masks.shape, (12, 10))
        torch.testing.assert_close(torch.from_numpy(batch), torch.from_numpy(scalar))


if __name__ == "__main__":
    unittest.main()
