import tempfile
import unittest
from pathlib import Path

from heads_up_action_conditioned_eval import (
    ActionConditionedEvaluationConfig,
    ActionConditionedProductionCampaign,
)
from heads_up_cfr import HeadsUpNeuralCFR
from heads_up_engine import HeadsUpHoldemEngine
from heads_up_models import ACTION_CONDITIONED_ARCHITECTURE
from heads_up_production import CampaignConfig, save_policy_snapshot


def _trainer() -> HeadsUpNeuralCFR:
    env = HeadsUpHoldemEngine(
        starting_stack=8,
        small_blind=1,
        big_blind=2,
        seed=17,
    )
    return HeadsUpNeuralCFR(
        env,
        hidden=16,
        blocks=0,
        advantage_capacity=64,
        policy_capacity=64,
        max_nodes_per_traversal=32,
        max_depth=16,
        range_capacity=1,
        range_loss_weight=0.0,
        network_architecture=ACTION_CONDITIONED_ARCHITECTURE,
        policy_network_architecture=ACTION_CONDITIONED_ARCHITECTURE,
        enable_range_training=False,
        seed=29,
    )


class ActionConditionedEvaluationTests(unittest.TestCase):
    def test_same_state_and_reciprocal_metrics_are_persisted(self):
        trainer = _trainer()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frozen = root / "frozen.pt"
            save_policy_snapshot(trainer, frozen)
            evaluation = ActionConditionedEvaluationConfig(
                policy_1025_path=str(frozen),
                ensemble_policy_paths=(str(frozen), str(frozen), str(frozen)),
                reciprocal_hands=20,
                inference_batch_size=32,
                simulation_batch_size=20,
                same_state_hands=10,
                seed=123,
            )
            config = CampaignConfig(
                target_iteration=1,
                traversals_per_player=1,
                traversal_workers=1,
                advantage_steps=1,
                policy_steps=1,
                batch_size=8,
                evaluate_every=1,
                checkpoint_every=1,
                snapshot_every=1,
                evaluation_games_per_player=2,
                range_evaluation_hands_per_opponent=1,
                range_evaluation_batch_size=8,
                range_training_hands_per_iteration=1,
                range_batch_size=1,
                league_games_per_player=1,
                league_opponents=0,
            )
            campaign = ActionConditionedProductionCampaign(
                trainer,
                root / "campaign",
                config,
                evaluation,
            )
            iteration_dir = root / "campaign" / "evaluations" / "step_00000000"
            iteration_dir.mkdir(parents=True)
            row = {}
            campaign._extend_evaluation(row, iteration_dir)
            self.assertGreater(row["same_state_decisions"], 0.0)
            self.assertIn("promotion_policy1025_ev_bb_per_100", row)
            self.assertIn("promotion_ensemble_top3_all_in_action_rate", row)
            self.assertEqual(row["promotion_gate_no_major_exploitability"], 0.0)
            self.assertEqual(row["promotion_all_required_gates"], 0.0)
            self.assertTrue(
                (iteration_dir / "action_conditioned_benchmarks.json").is_file()
            )
            self.assertTrue(campaign.same_state_holdout_path.is_file())


if __name__ == "__main__":
    unittest.main()
