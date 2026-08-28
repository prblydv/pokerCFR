import random
import tempfile
import unittest
from pathlib import Path

import torch

from three_player_cfr_recent_stage import (
    RecentWindowBuffer,
    ReservoirBuffer,
    ThreePlayerNeuralCFR,
)
from three_player_engine import ThreePlayerHoldemEnv


def fields(values):
    values = torch.tensor(values, dtype=torch.float32)
    count = len(values)
    return [
        values[:, None].repeat(1, 4).to(torch.float16),
        values[:, None].repeat(1, 9),
        torch.ones((count, 9)),
        values.clone(),
    ]


class RecentWindowTests(unittest.TestCase):
    def test_uniform_packed_selection_is_deterministic(self):
        left = RecentWindowBuffer(12, 3, random.Random(91))
        right = RecentWindowBuffer(12, 3, random.Random(91))
        chunks = [fields(range(5)), fields(range(5, 11))]
        left.add_packed_iteration(chunks)
        right.add_packed_iteration(chunks)
        self.assertEqual(len(left), 4)
        self.assertTrue(torch.equal(left.memory.fields[3][:4], right.memory.fields[3][:4]))
        self.assertEqual(len(set(left.memory.fields[3][:4].tolist())), 4)

    def test_exact_iteration_blocks_rotate(self):
        buffer = RecentWindowBuffer(12, 3, random.Random(7))
        for iteration in range(1, 5):
            buffer.add_packed_iteration([fields([iteration] * 8)])
        self.assertEqual(len(buffer), 12)
        retained = sorted(set(buffer.memory.fields[3][:12].tolist()))
        self.assertEqual(retained, [2.0, 3.0, 4.0])

    def test_underfilled_iteration_blocks_rotate_without_stale_rows(self):
        buffer = RecentWindowBuffer(12, 3, random.Random(17))
        for iteration in range(1, 5):
            buffer.add_packed_iteration([fields([iteration] * 2)])
        self.assertEqual(len(buffer), 12)
        retained = buffer.memory.fields[3][:12]
        self.assertEqual(sorted(set(retained.tolist())), [2.0, 3.0, 4.0])
        for iteration in (2.0, 3.0, 4.0):
            self.assertEqual(int((retained == iteration).sum()), 4)

    def test_recent_retention_does_not_change_historical_reservoir(self):
        baseline_rng = random.Random(123)
        hybrid_rng = random.Random(123)
        baseline = ReservoirBuffer(20, baseline_rng)
        hybrid = ReservoirBuffer(20, hybrid_rng)
        recent = RecentWindowBuffer(12, 3, random.Random(999))
        packed = fields(range(100))
        for row in range(100):
            baseline.add_packed_row(packed, row)
            hybrid.add_packed_row(packed, row)
        recent.add_packed_iteration([packed])
        baseline._compact()
        hybrid._compact()
        self.assertEqual(baseline.seen, hybrid.seen)
        self.assertEqual(baseline_rng.getstate(), hybrid_rng.getstate())
        for expected, actual in zip(baseline.memory.fields, hybrid.memory.fields):
            self.assertTrue(torch.equal(expected, actual))

    def test_hybrid_batch_has_requested_fraction(self):
        env = ThreePlayerHoldemEnv(stack_size=12, sb=1, bb=2, seed=3)
        trainer = ThreePlayerNeuralCFR(
            env,
            hidden=8,
            blocks=1,
            advantage_capacity=20,
            policy_capacity=20,
            recent_capacity=12,
            recent_window_iterations=3,
            recent_batch_fraction=0.3,
            seed=5,
        )
        historical = trainer.advantage_buffers[0]
        for row in range(20):
            historical.add(tuple(field[row].clone() for field in fields([0] * 20)))
        recent = trainer.recent_advantage_buffers[0]
        recent.add_packed_iteration([fields([1] * 20)])
        batch = next(
            iter(
                trainer._fit_field_batches(
                    historical, recent, 10, 1, shuffled_historical=True
                )
            )
        )
        self.assertEqual(batch[0].shape[0], 10)
        self.assertEqual(int((batch[3] == 1).sum()), 3)

    def test_checkpoint_records_config_but_not_recent_payload(self):
        env = ThreePlayerHoldemEnv(stack_size=12, sb=1, bb=2, seed=8)
        trainer = ThreePlayerNeuralCFR(
            env,
            hidden=8,
            blocks=1,
            advantage_capacity=20,
            policy_capacity=20,
            recent_capacity=12,
            recent_window_iterations=3,
            recent_batch_fraction=0.5,
            seed=9,
        )
        trainer.recent_advantage_buffers[0].add_packed_iteration([fields(range(8))])
        with tempfile.TemporaryDirectory() as directory:
            path = trainer.save(Path(directory) / "checkpoint.pt")
            raw = torch.load(path, map_location="cpu", weights_only=False)
            self.assertNotIn("recent_advantage_buffers", raw)
            restored = ThreePlayerNeuralCFR.load(
                path,
                ThreePlayerHoldemEnv(stack_size=12, sb=1, bb=2, seed=8),
            )
        self.assertEqual(restored.recent_capacity, 12)
        self.assertEqual(restored.recent_window_iterations, 3)
        self.assertEqual(restored.recent_batch_fraction, 0.5)
        self.assertEqual(len(restored.recent_advantage_buffers[0]), 0)


if __name__ == "__main__":
    unittest.main()
