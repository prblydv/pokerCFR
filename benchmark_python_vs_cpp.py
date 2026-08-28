"""Differential speed benchmark for the Python and packed C++ poker engines."""

from __future__ import annotations

import hashlib
import pickle
import random
import time

import torch

from three_player_cfr import ThreePlayerNeuralCFR
from three_player_engine import ThreePlayerHoldemEnv as PythonEnv
from three_player_native import ThreePlayerHoldemEnv as NativeEnv


def engine_actions(env_type, count: int = 50_000) -> tuple[float, int]:
    env = env_type(stack_size=200, sb=1, bb=2, seed=7)
    chooser = random.Random(9)
    completed = 0
    started = time.perf_counter()
    while completed < count:
        deck = list(range(52))
        random.Random(100_000 + completed).shuffle(deck)
        state = env.new_hand(button=completed % 3, deck=deck)
        while not state.terminal and completed < count:
            legal = env.legal_actions(state)
            state = env.step(state, chooser.choice(legal))
            completed += 1
    return time.perf_counter() - started, completed


def make_trainer(env_type) -> tuple[object, ThreePlayerNeuralCFR]:
    env = env_type(stack_size=200, sb=1, bb=2, seed=442)
    trainer = ThreePlayerNeuralCFR(
        env,
        device="cpu",
        hidden=128,
        blocks=3,
        network_architecture="deep_cfr_branch_v2",
        max_history=32,
        max_nodes_per_traversal=250,
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
    return env, trainer


def traversal(env_type, roots: int = 60) -> tuple[float, int, int, str]:
    env, trainer = make_trainer(env_type)
    contexts = []
    stack_modes = (
        [200.0, 200.0, 200.0],
        [350.0, 0.0, 250.0],
        [20.0, 100.0, 480.0],
        [75.5, 410.25, 114.25],
    )
    for index in range(roots):
        stacks = stack_modes[index % len(stack_modes)]
        traverser = index % 3
        if stacks[traverser] <= 0:
            continue
        live = [value > 1e-9 for value in stacks]
        button = trainer._button_for_live_role(traverser, live, index % sum(live))
        deck = list(range(52))
        random.Random(300_000 + index).shuffle(deck)
        contexts.append(
            {
                "state": env.new_hand(button=button, stacks=stacks, deck=deck),
                "traverser": traverser,
                "rng": random.Random(20_000 + index),
            }
        )
    started = time.perf_counter()
    trainer._run_batched_traversals(contexts)
    elapsed = time.perf_counter() - started
    digest = hashlib.sha256()
    signature = (
        trainer._nodes_this_iteration,
        trainer._rollouts_this_iteration,
        tuple(trainer._regret_magnitudes),
        tuple(trainer._strategy_weights),
        tuple(trainer._policy_entropies),
        tuple(buffer.seen for buffer in trainer.advantage_buffers + trainer.policy_buffers),
        trainer.rng.getstate(),
    )
    digest.update(pickle.dumps(signature, protocol=5))
    for buffer in trainer.advantage_buffers + trainer.policy_buffers:
        state = buffer.state_dict()
        digest.update(str(state["seen"]).encode())
        for field in state["fields"]:
            contiguous = field.contiguous()
            digest.update(str(contiguous.dtype).encode())
            digest.update(str(tuple(contiguous.shape)).encode())
            digest.update(contiguous.numpy().tobytes())
    return (
        elapsed,
        int(trainer._nodes_this_iteration),
        int(trainer._rollouts_this_iteration),
        digest.hexdigest(),
    )


def main() -> None:
    py_action_seconds, actions = engine_actions(PythonEnv)
    cpp_action_seconds, _ = engine_actions(NativeEnv)
    py_traversal = traversal(PythonEnv)
    cpp_traversal = traversal(NativeEnv)
    if py_traversal[1:] != cpp_traversal[1:]:
        raise AssertionError(
            f"training traversal diverged: Python={py_traversal[1:]}, C++={cpp_traversal[1:]}"
        )
    print(
        f"actions={actions} python={py_action_seconds:.6f}s "
        f"cpp={cpp_action_seconds:.6f}s speedup={py_action_seconds / cpp_action_seconds:.2f}x"
    )
    print(
        f"traversal python={py_traversal[0]:.6f}s cpp={cpp_traversal[0]:.6f}s "
        f"speedup={py_traversal[0] / cpp_traversal[0]:.2f}x "
        f"nodes={py_traversal[1]} rollouts={py_traversal[2]} digest={py_traversal[3]}"
    )
    print("PYTHON AND PACKED C++ TRAINING OUTPUTS ARE EXACT")


if __name__ == "__main__":
    main()
