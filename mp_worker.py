# mp_worker.py
# Multiprocessing worker for advantage traversals.
# IMPORTANT: does NOT import abstraction.py.

import random
import torch
from poker_env import SimpleHoldemEnv
from traversal_only import traverse_once

GLOBAL_ENV = None


def init_worker():
    """Run once per worker process."""
    global GLOBAL_ENV
    GLOBAL_ENV = SimpleHoldemEnv()


def run_traversals(seed: int, num_traversals: int, state_dim: int):
    """
    Run num_traversals advantage traversals for BOTH players.
    Returns: (samples_p0, samples_p1)
    """
    random.seed(seed)
    torch.manual_seed(seed)

    samples0 = []
    samples1 = []

    for _ in range(num_traversals):
        s = GLOBAL_ENV.new_hand()
        samples0.extend(traverse_once(s, 0, GLOBAL_ENV, state_dim))

        s = GLOBAL_ENV.new_hand()
        samples1.extend(traverse_once(s, 1, GLOBAL_ENV, state_dim))

    return samples0, samples1
