"""Reproducible old-vs-packed native state benchmark."""

import random
import statistics
import time
import hashlib
import pickle

import torch

from three_player_cfr import ThreePlayerNeuralCFR
from three_player_native import ThreePlayerHoldemEnv


def step_benchmark(repetitions: int = 50_000) -> float:
    env = ThreePlayerHoldemEnv(stack_size=200, sb=1, bb=2, seed=442)
    deck = list(range(52))
    random.Random(88).shuffle(deck)
    state = env.new_hand(button=0, deck=deck)
    action = env.legal_actions(state)[-1]
    started = time.perf_counter()
    for _ in range(repetitions):
        env.step(state, action)
    return time.perf_counter() - started


def traversal_benchmark() -> tuple[float, tuple]:
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
    generator = torch.Generator().manual_seed(83_017)
    with torch.no_grad():
        for network in trainer.advantage_nets:
            for head in network.street_heads:
                head.weight.normal_(mean=0.0, std=0.08, generator=generator)
                head.bias.normal_(mean=0.0, std=0.08, generator=generator)
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
    started = time.perf_counter()
    trainer._run_batched_traversals(contexts)
    elapsed = time.perf_counter() - started
    signature = (
        trainer._nodes_this_iteration,
        trainer._rollouts_this_iteration,
        tuple(trainer._regret_magnitudes),
        tuple(trainer._strategy_weights),
        tuple(trainer._policy_entropies),
        tuple(buffer.seen for buffer in trainer.advantage_buffers + trainer.policy_buffers),
        trainer.rng.getstate(),
    )
    digest = hashlib.sha256(pickle.dumps(signature, protocol=5))
    for buffer in trainer.advantage_buffers + trainer.policy_buffers:
        state = buffer.state_dict()
        digest.update(str(state["seen"]).encode())
        for field in state["fields"]:
            contiguous = field.contiguous()
            digest.update(str(contiguous.dtype).encode())
            digest.update(str(tuple(contiguous.shape)).encode())
            digest.update(contiguous.numpy().tobytes())
    return elapsed, signature, digest.hexdigest()


if __name__ == "__main__":
    step_times = [step_benchmark() for _ in range(2)]
    traversal_times = []
    signature = None
    digest = None
    for _ in range(1):
        elapsed, current_signature, current_digest = traversal_benchmark()
        if signature is not None and signature != current_signature:
            raise AssertionError("benchmark traversal is not deterministic")
        signature = current_signature
        digest = current_digest
        traversal_times.append(elapsed)
    print("step_times", step_times, "median", statistics.median(step_times))
    print(
        "traversal_times", traversal_times,
        "median", statistics.median(traversal_times),
        "nodes", signature[0], "rollouts", signature[1], "digest", digest,
    )
