"""Small A/B traversal benchmark for the opt-in all-in regret adjustment."""

from __future__ import annotations

import argparse
import statistics
import time

from heads_up_cfr import (
    DEFAULT_ROOT_STACK_DEPTHS_BB,
    NETWORK_ARCHITECTURE,
    ROOT_STACK_DISTRIBUTION_MIXED,
    HeadsUpNeuralCFR,
)
from heads_up_engine import HeadsUpHoldemEngine


def run(enabled: bool, traversals: int, repeats: int) -> tuple[int, float]:
    rates = []
    observed_nodes = 0
    for repeat in range(repeats):
        env = HeadsUpHoldemEngine(
            starting_stack=200, small_blind=1, big_blind=2, seed=900 + repeat
        )
        trainer = HeadsUpNeuralCFR(
            env,
            hidden=8,
            blocks=0,
            advantage_capacity=100_000,
            policy_capacity=100_000,
            range_capacity=0,
            range_loss_weight=0.0,
            network_architecture=NETWORK_ARCHITECTURE,
            policy_network_architecture=NETWORK_ARCHITECTURE,
            enable_range_training=False,
            risk_aware_all_in=enabled,
            max_nodes_per_traversal=128,
            max_depth=24,
            seed=700 + repeat,
        )
        started = time.perf_counter()
        trainer._collect_traversals(
            traversals,
            1,
            ROOT_STACK_DISTRIBUTION_MIXED,
            DEFAULT_ROOT_STACK_DEPTHS_BB,
        )
        elapsed = time.perf_counter() - started
        observed_nodes = trainer._nodes_this_iteration
        rates.append(observed_nodes / max(elapsed, 1e-9))
    return observed_nodes, statistics.median(rates)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traversals", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    baseline_nodes, baseline_rate = run(False, args.traversals, args.repeats)
    guarded_nodes, guarded_rate = run(True, args.traversals, args.repeats)
    print(f"baseline_nodes={baseline_nodes} nodes_per_second={baseline_rate:.2f}")
    print(f"risk_aware_nodes={guarded_nodes} nodes_per_second={guarded_rate:.2f}")
    print(f"throughput_ratio={guarded_rate / max(baseline_rate, 1e-9):.4f}")


if __name__ == "__main__":
    main()
