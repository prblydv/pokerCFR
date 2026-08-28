"""Run the multiprocessing regression against the staged CFR implementation."""

import importlib
import importlib.util
import sys
import unittest


def install(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    cfr = install("three_player_cfr", "three_player_cfr_recent_stage.py")
    module = importlib.import_module("test_three_player_training")
    suite = unittest.TestSuite(
        [
            module.ThreePlayerTrainingTests(
                "test_parallel_cpu_root_collection_and_fitting"
            )
        ]
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if result.wasSuccessful():
        from three_player_engine import ThreePlayerHoldemEnv

        trainer = cfr.ThreePlayerNeuralCFR(
            ThreePlayerHoldemEnv(stack_size=12, sb=1, bb=2, seed=17),
            hidden=8,
            blocks=1,
            advantage_capacity=200,
            policy_capacity=200,
            recent_capacity=12,
            recent_window_iterations=3,
            recent_batch_fraction=0.5,
            max_nodes_per_traversal=30,
            max_depth=8,
            reinitialize_advantage_each_iteration=False,
            seed=17,
        )
        trainer.train_iteration(
            traversals_per_player=2,
            advantage_steps=1,
            policy_steps=1,
            batch_size=8,
            traversal_workers=2,
        )
        assert any(len(buffer) for buffer in trainer.recent_advantage_buffers)
        assert any(len(buffer) for buffer in trainer.recent_policy_buffers)
    raise SystemExit(not result.wasSuccessful())
