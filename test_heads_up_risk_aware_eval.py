"""Focused tests for risk-aware campaign diagnostics."""

from __future__ import annotations

import unittest

import pandas as pd

from heads_up_risk_aware_eval import plot_all_in_spr_trend


class RiskAwareEvaluationTests(unittest.TestCase):
    def test_all_in_spr_plot_uses_only_all_in_summary_columns(self):
        frame = pd.DataFrame(
            {
                "iteration": [25, 50],
                "promotion_policy1025_all_in_spr_median": [12.0, 7.0],
                "promotion_policy1025_all_in_spr_p90": [30.0, 18.0],
                "promotion_ensemble_top3_all_in_spr_median": [10.0, 6.0],
                "promotion_ensemble_top3_all_in_spr_p90": [25.0, 15.0],
            }
        )
        figure = plot_all_in_spr_trend(frame)
        try:
            axis = figure.axes[0]
            self.assertIn("only when", axis.get_title())
            self.assertEqual(axis.get_xlabel(), "Iteration")
            self.assertEqual(len(axis.lines), 5)
        finally:
            import matplotlib.pyplot as plt

            plt.close(figure)


if __name__ == "__main__":
    unittest.main()
