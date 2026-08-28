"""Short production-shaped CPU traversal profile for the active Vast code."""

import cProfile
import io
import pstats
import random
import time

import torch

from three_player_cfr import ThreePlayerNeuralCFR
from three_player_native import ThreePlayerHoldemEnv


torch.set_num_threads(1)
env = ThreePlayerHoldemEnv(stack_size=200, sb=1, bb=2, seed=442)
trainer = ThreePlayerNeuralCFR(
    env,
    device="cpu",
    hidden=128,
    blocks=3,
    network_architecture="deep_cfr_branch_v2",
    max_history=32,
    max_nodes_per_traversal=500,
    max_depth=16,
    exploration=0.15,
    include_tournament_features=True,
    variable_stack_training=True,
    tournament_total_chips=600,
    advantage_capacity=150_000,
    policy_capacity=150_000,
    seed=442,
    _traversal_worker=True,
)
contexts = []
for index in range(240):
    stacks = trainer._sample_tournament_stacks()
    traverser = index % 3
    if stacks[traverser] <= 0:
        continue
    alive = [value > 1e-9 for value in stacks]
    button = trainer._button_for_live_role(traverser, alive, index % sum(alive))
    contexts.append(
        {
            "state": env.new_hand(button=button, stacks=stacks),
            "traverser": traverser,
            "rng": random.Random(20_000 + index),
        }
    )

profile = cProfile.Profile()
started = time.perf_counter()
profile.enable()
trainer._run_batched_traversals(contexts)
profile.disable()
elapsed = time.perf_counter() - started
print(
    f"contexts={len(contexts)} nodes={trainer._nodes_this_iteration} "
    f"elapsed={elapsed:.6f} nodes_per_second={trainer._nodes_this_iteration / elapsed:.3f}"
)
for ordering in ("cumtime", "tottime"):
    output = io.StringIO()
    pstats.Stats(profile, stream=output).strip_dirs().sort_stats(ordering).print_stats(40)
    print(f"\nSORT={ordering}\n{output.getvalue()}")
