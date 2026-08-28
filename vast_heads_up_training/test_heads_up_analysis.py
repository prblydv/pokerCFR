import unittest

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from heads_up_analysis import (
    StrategyAnalyzer,
    build_decision_state,
    classify_hole_cards,
    plot_range_heatmaps,
    postflop_scenarios,
    preflop_scenarios,
)
from heads_up_cfr import HeadsUpNeuralCFR
from heads_up_native import HeadsUpHoldemEngine


class HeadsUpAnalysisTests(unittest.TestCase):
    def test_named_scenarios_are_exact_legal_hero_decisions(self):
        env = HeadsUpHoldemEngine(seed=31)
        scenarios = preflop_scenarios() + postflop_scenarios(
            flop=(51, 18, 0),
            turn=35,
            river=41,
        )
        for scenario in scenarios:
            available = [
                card for card in range(52) if card not in set(scenario.board)
            ]
            for hero in (0, 1):
                state = build_decision_state(
                    env,
                    scenario,
                    hero=hero,
                    hero_cards=available[:2],
                )
                self.assertEqual(state.to_act, hero)
                self.assertEqual(tuple(state.board), scenario.board)
                self.assertTrue(env.legal_actions(state))

    def test_standard_169_hand_coordinates(self):
        self.assertEqual(classify_hole_cards((12, 25))[0], "AA")
        self.assertEqual(classify_hole_cards((12, 11))[0], "AKs")
        self.assertEqual(classify_hole_cards((12, 24))[0], "AKo")

    def test_full_range_report_and_plot(self):
        env = HeadsUpHoldemEngine(starting_stack=40, seed=37)
        trainer = HeadsUpNeuralCFR(
            env,
            device="cpu",
            hidden=8,
            blocks=0,
            advantage_capacity=32,
            policy_capacity=32,
            max_history=16,
            max_nodes_per_traversal=16,
            max_depth=16,
            seed=37,
        )
        report = StrategyAnalyzer(trainer, batch_size=512).analyze_range(
            preflop_scenarios()[1],
            hero_seats=(0,),
        )
        self.assertEqual(len(report.hand_table), 169)
        self.assertEqual(report.state_summary["physical_combos"], 1326)
        self.assertAlmostEqual(
            float(
                report.combo_table[
                    [f"p_{name}" for name in (
                        "fold",
                        "check",
                        "call",
                        "min_raise",
                        "third_pot",
                        "half_pot",
                        "three_quarter_pot",
                        "pot",
                        "overbet",
                        "all_in",
                    )]
                ].sum(axis=1).mean()
            ),
            1.0,
            places=5,
        )
        figure = plot_range_heatmaps(report)
        self.assertEqual(len(figure.axes) >= 3, True)
        plt.close(figure)


if __name__ == "__main__":
    unittest.main()
