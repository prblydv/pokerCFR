import tempfile
import unittest
from pathlib import Path

import torch

from three_player_cfr import ThreePlayerNeuralCFR
from three_player_engine import ThreePlayerHoldemEnv
from three_player_models import (
    TOURNAMENT_FEATURE_NAMES,
    TOURNAMENT_FEATURES,
    encode_information_state,
    information_state_size,
)


class TournamentEncodingTests(unittest.TestCase):
    def test_tournament_suffix_preserves_legacy_prefix_and_describes_heads_up(self):
        env = ThreePlayerHoldemEnv(stack_size=200, sb=1, bb=2, seed=1)
        state = env.new_hand(button=0, stacks=[450, 150, 0])
        hero = int(state.to_act)
        legal = env.legal_actions(state)
        legacy = encode_information_state(state, hero, legal, 200, 32)
        expanded = encode_information_state(
            state,
            hero,
            legal,
            200,
            32,
            include_tournament_features=True,
            tournament_total_chips=600,
        )

        self.assertEqual(legacy.numel(), information_state_size(32))
        self.assertEqual(
            expanded.numel(),
            information_state_size(32, include_tournament_features=True),
        )
        self.assertTrue(torch.equal(expanded[: legacy.numel()], legacy))
        values = dict(
            zip(
                TOURNAMENT_FEATURE_NAMES,
                expanded[-TOURNAMENT_FEATURES:].tolist(),
            )
        )
        self.assertEqual(
            [values[name] for name in TOURNAMENT_FEATURE_NAMES[:3]],
            [1.0, 1.0, 0.0],
        )
        self.assertAlmostEqual(values["hero_starting_chip_share"], 0.75)
        self.assertAlmostEqual(values["clockwise_1_starting_chip_share"], 0.25)
        self.assertAlmostEqual(values["effective_stack_vs_clockwise_1"], 148 / 200)
        self.assertAlmostEqual(values["players_remaining"], 2 / 3)
        self.assertAlmostEqual(values["players_in_hand"], 2 / 3)
        self.assertEqual(values["heads_up"], 1.0)

    def test_explicit_alive_flags_override_state_status(self):
        env = ThreePlayerHoldemEnv(stack_size=20, sb=1, bb=2, seed=2)
        state = env.new_hand(button=0)
        hero = int(state.to_act)
        encoded = encode_information_state(
            state,
            hero,
            env.legal_actions(state),
            20,
            4,
            include_tournament_features=True,
            alive_flags=[True, False, True],
            tournament_total_chips=60,
        )
        values = dict(
            zip(TOURNAMENT_FEATURE_NAMES, encoded[-TOURNAMENT_FEATURES:].tolist())
        )
        # Hero is seat 0's clockwise successor in this three-handed root, so
        # relative status is rotated rather than tied to absolute seat order.
        expected = [float([True, False, True][(hero + i) % 3]) for i in range(3)]
        actual = [values[name] for name in TOURNAMENT_FEATURE_NAMES[:3]]
        self.assertEqual(actual, expected)
        self.assertAlmostEqual(values["players_remaining"], 2 / 3)
        self.assertEqual(values["heads_up"], 1.0)


class TournamentRootTrainingTests(unittest.TestCase):
    def _trainer(self, **kwargs):
        env = ThreePlayerHoldemEnv(stack_size=20, sb=1, bb=2, seed=7)
        options = dict(
            hidden=8,
            blocks=0,
            max_history=4,
            max_nodes_per_traversal=12,
            advantage_capacity=100,
            policy_capacity=100,
            include_tournament_features=True,
            variable_stack_training=True,
            tournament_total_chips=60,
            continuation_root_fraction=0.0,
            seed=9,
        )
        options.update(kwargs)
        return ThreePlayerNeuralCFR(env, **options)

    def test_synthetic_roots_conserve_chips_and_include_zero_for_heads_up(self):
        trainer = self._trainer()
        for live_players in (2, 3):
            for _ in range(20):
                stacks = trainer._sample_tournament_stacks(live_players)
                self.assertAlmostEqual(sum(stacks), 60.0, places=8)
                self.assertEqual(sum(value > 0 for value in stacks), live_players)
                self.assertTrue(
                    all(value == 0 or value >= trainer.minimum_live_stack for value in stacks)
                )

    def test_training_skips_eliminated_traversers_at_heads_up_roots(self):
        trainer = self._trainer(heads_up_root_fraction=1.0)
        observed = []

        def record_only(state, traverser, reach, depth):
            observed.append((traverser, tuple(state.alive), state.button))
            return 0.0

        trainer._traverse = record_only
        metrics = trainer.train_iteration(
            traversals_per_player=8,
            advantage_steps=1,
            policy_steps=0,
            batch_size=4,
        )
        self.assertGreater(len(observed), 0)
        self.assertTrue(all(sum(alive) == 2 for _, alive, _ in observed))
        self.assertTrue(all(alive[traverser] for traverser, alive, _ in observed))
        self.assertEqual(metrics["heads_up_roots"], float(len(observed)))
        self.assertGreater(metrics["eliminated_traversals_skipped"], 0.0)
        self.assertEqual(
            metrics["heads_up_roots"] + metrics["eliminated_traversals_skipped"],
            24.0,
        )

    def test_continuation_roots_are_reused_and_checkpointed(self):
        trainer = self._trainer(continuation_root_fraction=1.0)
        self.assertTrue(trainer._remember_continuation([45, 15, 0]))
        self.assertEqual(trainer._root_stacks(), (45.0, 15.0, 0.0))
        self.assertFalse(trainer._remember_continuation([60, 0, 0]))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "continuation.pt"
            trainer.save(path)
            restored = ThreePlayerNeuralCFR.load(
                path,
                ThreePlayerHoldemEnv(stack_size=20, sb=1, bb=2, seed=100),
            )
            self.assertEqual(restored._continuation_stacks, [(45.0, 15.0, 0.0)])
            self.assertEqual(restored._root_stacks(), (45.0, 15.0, 0.0))


class TournamentCheckpointCompatibilityTests(unittest.TestCase):
    def test_legacy_checkpoint_auto_detect_and_policy_warm_start(self):
        env = ThreePlayerHoldemEnv(stack_size=20, sb=1, bb=2, seed=11)
        legacy = ThreePlayerNeuralCFR(
            env,
            hidden=8,
            blocks=0,
            max_history=4,
            include_tournament_features=False,
            seed=12,
        )
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            original_path = directory / "legacy.pt"
            legacy.save(original_path, include_buffers=False)

            # Reproduce a pre-tournament version-1 checkpoint: it has the old
            # width but no explicit encoder-mode config field.
            payload = torch.load(original_path, map_location="cpu", weights_only=False)
            payload["config"].pop("include_tournament_features")
            old_style_path = directory / "old_style.pt"
            torch.save(payload, old_style_path)
            restored = ThreePlayerNeuralCFR.load(
                old_style_path,
                ThreePlayerHoldemEnv(stack_size=20, sb=1, bb=2, seed=13),
            )
            self.assertFalse(restored.include_tournament_features)
            self.assertEqual(restored.input_dim, information_state_size(4))

            expanded = ThreePlayerNeuralCFR(
                ThreePlayerHoldemEnv(stack_size=20, sb=1, bb=2, seed=14),
                hidden=8,
                blocks=0,
                max_history=4,
                include_tournament_features=True,
                variable_stack_training=True,
                seed=15,
            )
            report = expanded.warm_start_policy(old_style_path)
            self.assertTrue(report["expanded_legacy_input"])
            old_width = legacy.input_dim
            for old_net, new_net in zip(legacy.policy_nets, expanded.policy_nets):
                self.assertTrue(
                    torch.equal(
                        old_net.input_layer.weight,
                        new_net.input_layer.weight[:, :old_width],
                    )
                )
                self.assertTrue(
                    torch.equal(
                        new_net.input_layer.weight[:, old_width:],
                        torch.zeros_like(new_net.input_layer.weight[:, old_width:]),
                    )
                )


if __name__ == "__main__":
    unittest.main()
