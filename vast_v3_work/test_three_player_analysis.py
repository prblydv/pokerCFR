import json
import tempfile
import unittest
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from three_player_analysis import (
    StrategyAnalyzer,
    build_decision_state,
    classify_hole_cards,
    controlled_deck,
    plot_card_sweep,
    plot_network_architecture,
    plot_range_heatmaps,
    postflop_scenarios,
    preflop_scenarios,
)
from three_player_cfr import ThreePlayerNeuralCFR
from three_player_engine import ThreePlayerHoldemEnv
from three_player_production import (
    CampaignConfig,
    CallingStationOpponent,
    ProductionCampaign,
    TightAggressiveOpponent,
    UniformRandomOpponent,
    evaluate_against_profile,
    load_or_create_trainer,
    load_policy_snapshot,
    paired_improvement,
    resolve_latest_checkpoint,
    save_policy_snapshot,
)


class StrategyAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.env = ThreePlayerHoldemEnv(stack_size=40, sb=1, bb=2, seed=31)
        self.trainer = ThreePlayerNeuralCFR(
            self.env,
            hidden=8,
            blocks=0,
            max_history=8,
            max_nodes_per_traversal=20,
            reinitialize_advantage_each_iteration=False,
            seed=32,
        )

    def test_grid_orientation_and_combo_classes(self):
        self.assertEqual(classify_hole_cards((12, 25)), ("AA", 0, 0))
        self.assertEqual(classify_hole_cards((12, 11)), ("AKs", 0, 1))
        self.assertEqual(classify_hole_cards((12, 24)), ("AKo", 1, 0))

    def test_network_architecture_plot_uses_trainer_dimensions(self):
        figure = plot_network_architecture(self.trainer)
        try:
            self.assertEqual(len(figure.axes), 2)
            labels = "\n".join(
                text.get_text()
                for axis in figure.axes
                for text in axis.texts
            )
            self.assertIn("3 advantage + 3 policy = 6 networks", labels)
            self.assertIn(f"{self.trainer.input_dim} features", labels)
            self.assertIn(f"RESIDUAL BLOCK × {self.trainer.blocks}", labels)
        finally:
            plt.close(figure)

    def test_controlled_deck_and_all_builtin_scenarios(self):
        boards = ((), (51, 18, 0), (51, 18, 0, 37), (51, 18, 0, 37, 10))
        for hero in range(3):
            for board in boards:
                button = (hero + 1) % 3
                deck = controlled_deck(
                    button=button, hero=hero, hero_cards=(50, 49), board=board
                )
                self.assertEqual(len(deck), 52)
                self.assertEqual(len(set(deck)), 52)

        scenarios = list(preflop_scenarios()) + list(
            postflop_scenarios(flop=(51, 18, 0), turn=37, river=10)
        )
        for scenario in scenarios:
            for hero in range(3):
                state = build_decision_state(
                    self.env, scenario, hero=hero, hero_cards=(50, 49)
                )
                self.assertEqual(state.to_act, hero)
                self.assertEqual(tuple(state.board), scenario.board)
                self.assertFalse(state.terminal)
                self.assertAlmostEqual(
                    sum(state.stacks) + state.pot, sum(state.initial_stacks)
                )

    def test_full_preflop_and_blocked_flop_range_counts(self):
        analyzer = StrategyAnalyzer(self.trainer, batch_size=2048)
        preflop = analyzer.analyze_range(
            preflop_scenarios()[1], hero_seats=(0,)
        )
        self.assertEqual(len(preflop.combo_table), 1326)
        self.assertEqual(len(preflop.hand_table), 169)
        counts = preflop.hand_table.set_index("hand")["combo_count"]
        self.assertEqual(int(counts["AA"]), 6)
        self.assertEqual(int(counts["AKs"]), 4)
        self.assertEqual(int(counts["AKo"]), 12)

        flop_scenario = postflop_scenarios(flop=(51, 18, 0))[1]
        flop = analyzer.analyze_range(flop_scenario, hero_seats=(0,))
        self.assertEqual(len(flop.combo_table), 1176)
        blocked_counts = flop.hand_table.set_index("hand")["combo_count"]
        self.assertEqual(int(blocked_counts["AA"]), 3)
        self.assertEqual(int(blocked_counts["A7s"]), 2)
        self.assertEqual(int(blocked_counts["A7o"]), 7)

    def test_batch_predictions_charts_and_next_card_sweep(self):
        analyzer = StrategyAnalyzer(self.trainer, batch_size=1024)
        scenario = preflop_scenarios()[2]
        analysis_env = ThreePlayerHoldemEnv(stack_size=40, sb=1, bb=2, seed=90)
        states = [
            build_decision_state(
                analysis_env, scenario, hero=0, hero_cards=cards
            )
            for cards in ((12, 25), (12, 11), (12, 24), (0, 14))
        ]
        batched = self.trainer.average_policy_batch(states)
        for state, batch_probability in zip(states, batched):
            self.assertTrue(
                torch.allclose(
                    batch_probability, self.trainer.average_policy(state), atol=1e-7
                )
            )
        report = analyzer.analyze_range(
            scenario, hero_seats=(0,)
        )
        figure = plot_range_heatmaps(report)
        self.assertEqual(len(figure.axes) >= 3, True)

        turn_scenario = postflop_scenarios(
            flop=(51, 18, 0), turn=37
        )[-1]
        sweep = analyzer.analyze_next_cards(
            turn_scenario, hero_cards=(50, 49), hero_seats=(0,)
        )
        self.assertEqual(len(sweep), 47)
        sweep_figure = plot_card_sweep(sweep, metric="p_aggressive")
        self.assertGreaterEqual(len(sweep_figure.axes), 1)


class ProductionWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.env = ThreePlayerHoldemEnv(stack_size=12, sb=1, bb=2, seed=41)
        self.trainer = ThreePlayerNeuralCFR(
            self.env,
            hidden=8,
            blocks=0,
            max_history=4,
            max_nodes_per_traversal=12,
            advantage_capacity=100,
            policy_capacity=100,
            reinitialize_advantage_each_iteration=False,
            seed=42,
        )

    def test_scripted_profiles_only_choose_legal_actions(self):
        state = self.env.new_hand(button=0)
        for profile in (
            UniformRandomOpponent(),
            CallingStationOpponent(),
            TightAggressiveOpponent(),
        ):
            probabilities = profile.probabilities(self.env, state, state.to_act)
            self.assertTrue(torch.isclose(probabilities.sum(), torch.tensor(1.0)))
            for action, probability in enumerate(probabilities):
                if action not in self.env.legal_actions(state):
                    self.assertEqual(float(probability), 0.0)

    def test_policy_snapshot_and_paired_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.pt"
            save_policy_snapshot(self.trainer, path)
            restored = load_policy_snapshot(path)
            self.assertEqual(restored.iteration, 0)
            state = self.env.new_hand(button=1)
            current = self.trainer.average_policy_batch([state])[0]
            loaded = self.trainer.average_policy_batch(
                [state], policy_nets=restored.policy_nets
            )[0]
            self.assertTrue(torch.allclose(current, loaded))

            baseline = evaluate_against_profile(
                self.trainer,
                "random",
                games_per_player=3,
                seed=123,
                policy_nets=restored.policy_nets,
            )
            candidate = evaluate_against_profile(
                self.trainer, "random", games_per_player=3, seed=123
            )
            delta = paired_improvement(baseline, candidate)
            self.assertAlmostEqual(delta["delta_ev_bb"], 0.0)

    def test_campaign_autosave_resume_and_continue(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact_dir = Path(directory) / "run"
            first_config = CampaignConfig(
                target_iteration=1,
                traversals_per_player=1,
                advantage_steps=1,
                policy_steps=1,
                batch_size=4,
                evaluate_every=1,
                checkpoint_every=1,
                snapshot_every=1,
                evaluation_games_per_player=3,
                league_games_per_player=3,
                opponent_profiles=("random",),
                league_opponents=1,
                keep_full_checkpoints=1,
            )
            campaign = ProductionCampaign(
                self.trainer, artifact_dir, first_config
            )
            campaign.run()
            latest = resolve_latest_checkpoint(artifact_dir)
            self.assertIsNotNone(latest)
            self.assertTrue((artifact_dir / "metrics.jsonl").exists())
            self.assertTrue((artifact_dir / "snapshots" / "initial_policy.pt").exists())

            resumed_env = ThreePlayerHoldemEnv(stack_size=12, sb=1, bb=2, seed=99)
            resumed, was_resumed = load_or_create_trainer(
                resumed_env,
                artifact_dir,
                device="cpu",
                trainer_kwargs={},
            )
            self.assertTrue(was_resumed)
            self.assertEqual(resumed.iteration, 1)
            previous_exploration = resumed.exploration
            previous_max_depth = resumed.max_depth
            resumed.exploration = 0.15
            resumed.max_depth += 1
            second_config = CampaignConfig(
                **{
                    **first_config.__dict__,
                    "target_iteration": 2,
                    "evaluate_every": 2,
                }
            )
            ProductionCampaign(resumed, artifact_dir, second_config).run()
            self.assertEqual(resumed.iteration, 2)
            self.assertEqual(
                len(list((artifact_dir / "checkpoints").glob("step_*.pt"))), 1
            )
            run_config = json.loads(
                (artifact_dir / "run_config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(run_config["campaign"]["target_iteration"], 2)
            self.assertEqual(run_config["campaign"]["evaluate_every"], 2)

            history_rows = [
                json.loads(line)
                for line in (artifact_dir / "run_config_history.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(history_rows), 1)
            self.assertEqual(
                history_rows[0]["campaign_changes"]["evaluate_every"],
                {"previous": 1, "current": 2},
            )
            self.assertEqual(
                history_rows[0]["trainer_changes"]["exploration"],
                {"previous": previous_exploration, "current": 0.15},
            )
            self.assertEqual(
                history_rows[0]["trainer_changes"]["max_depth"],
                {"previous": previous_max_depth, "current": previous_max_depth + 1},
            )

            mutable_config = CampaignConfig(
                **{**second_config.__dict__, "batch_size": 8}
            )
            ProductionCampaign(resumed, artifact_dir, mutable_config)
            updated = json.loads(
                (artifact_dir / "run_config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(updated["campaign"]["batch_size"], 8)

            locked_config = CampaignConfig(
                **{**mutable_config.__dict__, "snapshot_every": 2}
            )
            with self.assertRaisesRegex(ValueError, "snapshot_every"):
                ProductionCampaign(resumed, artifact_dir, locked_config)


if __name__ == "__main__":
    unittest.main()
