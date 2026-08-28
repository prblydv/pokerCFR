"""Small reproducible engine-only benchmark; run from the repository root."""

import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from three_player_engine import ThreePlayerHoldemEnv as PythonEnv
from three_player_native import ThreePlayerHoldemEnv as NativeEnv


def measure(environment_type, actions=100_000):
    environment = environment_type(seed=7)
    chooser = random.Random(9)
    completed = 0
    started = time.perf_counter()
    while completed < actions:
        state = environment.new_hand()
        while not state.terminal and completed < actions:
            legal = environment.legal_actions(state)
            state = environment.step(state, chooser.choice(legal))
            completed += 1
    seconds = time.perf_counter() - started
    return seconds, completed / seconds


if __name__ == "__main__":
    python_seconds, python_rate = measure(PythonEnv)
    native_seconds, native_rate = measure(NativeEnv)
    print(f"Python: {python_seconds:.3f}s, {python_rate:,.0f} actions/s")
    print(f"C++:    {native_seconds:.3f}s, {native_rate:,.0f} actions/s")
    print(f"Speedup: {python_seconds / native_seconds:.2f}x")
