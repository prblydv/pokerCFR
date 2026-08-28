"""Benchmark the production-shaped four-worker native traversal phase."""

import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from three_player_cfr import ThreePlayerNeuralCFR
from three_player_native import ThreePlayerHoldemEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--traversals", type=int, default=32)
    arguments = parser.parse_args()
    environment = ThreePlayerHoldemEnv(stack_size=200, sb=1, bb=2, seed=442)
    trainer = ThreePlayerNeuralCFR(
        environment,
        hidden=256,
        blocks=1,
        network_architecture="dual_attention_state",
        max_history=32,
        max_nodes_per_traversal=1_000,
        max_depth=16,
        include_tournament_features=True,
        variable_stack_training=True,
        tournament_total_chips=600,
        heads_up_root_fraction=0.25,
        continuation_root_fraction=0.25,
        seed=442,
    )
    trainer.iteration = 450
    trainer._nodes_this_iteration = trainer._rollouts_this_iteration = 0
    trainer._regret_magnitudes = []
    trainer._strategy_weights = []
    trainer._policy_entropies = []
    trainer._raw_strategy_importances = []
    trainer._strategy_cap_hits = 0
    trainer._continuation_hands_this_iteration = 0
    trainer._three_handed_roots_this_iteration = 0
    trainer._heads_up_roots_this_iteration = 0
    trainer._eliminated_traversals_skipped_this_iteration = 0
    started = time.perf_counter()
    workers = trainer._collect_parallel_traversals(
        arguments.traversals, arguments.workers
    )
    seconds = time.perf_counter() - started
    print(
        f"{seconds:.3f}s, {workers} workers, "
        f"{trainer._nodes_this_iteration:.0f} nodes, "
        f"{trainer._rollouts_this_iteration:.0f} rollouts"
    )


if __name__ == "__main__":
    main()
