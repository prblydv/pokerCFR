import json
import tempfile
import unittest
from pathlib import Path

from three_player_cfr import ThreePlayerNeuralCFR
from three_player_native import ThreePlayerHoldemEnv
from three_player_production import CampaignConfig, ProductionCampaign


class QualityResumeOverrideTest(unittest.TestCase):
    def test_importance_cap_is_mutable_but_architecture_remains_locked(self):
        env = ThreePlayerHoldemEnv(stack_size=12, sb=1, bb=2, seed=91)
        trainer = ThreePlayerNeuralCFR(
            env,
            device="cpu",
            hidden=16,
            blocks=1,
            max_strategy_importance=100.0,
            reinitialize_advantage_each_iteration=False,
            seed=91,
        )
        config = CampaignConfig(
            target_iteration=1,
            traversals_per_player=1,
            traversal_workers=1,
            advantage_steps=1,
            policy_steps=1,
            batch_size=4,
            evaluate_every=1,
            checkpoint_every=1,
            snapshot_every=1,
            evaluation_games_per_player=3,
            league_games_per_player=3,
            opponent_profiles=("random",),
            league_opponents=0,
        )
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory)
            ProductionCampaign(trainer, artifact, config)
            trainer.max_strategy_importance = 50.0
            ProductionCampaign(trainer, artifact, config)
            history = [
                json.loads(line)
                for line in (artifact / "run_config_history.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(
                history[-1]["trainer_changes"]["max_strategy_importance"],
                {"previous": 100.0, "current": 50.0},
            )
            trainer.hidden = 32
            with self.assertRaisesRegex(ValueError, "hidden"):
                ProductionCampaign(trainer, artifact, config)


if __name__ == "__main__":
    unittest.main()
