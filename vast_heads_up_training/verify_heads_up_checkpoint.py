"""Read-only full-load verification for a resumable HU Deep CFR checkpoint."""

from __future__ import annotations

import argparse
import json

from heads_up_cfr import HeadsUpNeuralCFR
from heads_up_native import HeadsUpHoldemEngine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    args = parser.parse_args()

    env = HeadsUpHoldemEngine(
        starting_stack=200,
        small_blind=1,
        big_blind=2,
        seed=442,
    )
    trainer = HeadsUpNeuralCFR.load(args.checkpoint, env, device="cpu")
    print(
        json.dumps(
            {
                "iteration": trainer.iteration,
                "last_fitted_iteration": trainer.last_fitted_iteration,
                "can_resume_training": trainer.can_resume_training,
                "advantage_capacity": [
                    buffer.capacity for buffer in trainer.advantage_buffers
                ],
                "advantage_size": [
                    len(buffer) for buffer in trainer.advantage_buffers
                ],
                "advantage_seen": [
                    buffer.seen for buffer in trainer.advantage_buffers
                ],
                "policy_capacity": [
                    buffer.capacity for buffer in trainer.policy_buffers
                ],
                "policy_size": [
                    len(buffer) for buffer in trainer.policy_buffers
                ],
                "policy_seen": [
                    buffer.seen for buffer in trainer.policy_buffers
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
