"""Validate the isolated Vast native encoder before activation."""

from __future__ import annotations

import random
import time
from pathlib import Path

import torch

from three_player_engine import ThreePlayerHoldemEnv as PythonEnv
from three_player_models import encode_information_state
from three_player_native import ThreePlayerHoldemEnv as NativeEnv


def main() -> None:
    chooser = random.Random(91_827)
    stack_modes = (
        [200.0, 200.0, 200.0],
        [0.0, 241.0, 359.0],
        [397.0, 2.0, 201.0],
        [75.5, 410.25, 114.25],
    )
    checked = 0
    for hand in range(250):
        deck = list(range(52))
        chooser.shuffle(deck)
        stacks = stack_modes[hand % len(stack_modes)]
        live = [seat for seat, value in enumerate(stacks) if value > 0]
        button = live[hand % len(live)]
        py_env, native_env = PythonEnv(seed=1), NativeEnv(seed=1)
        py_state = py_env.new_hand(button=button, stacks=stacks, deck=deck)
        native_state = native_env.new_hand(button=button, stacks=stacks, deck=deck)
        while not py_state.terminal:
            legal = py_env.legal_actions(py_state)
            if native_env.legal_actions(native_state) != legal:
                raise AssertionError("native and Python legal actions diverged")
            reference = encode_information_state(
                py_state,
                py_state.to_act,
                legal,
                200.0,
                32,
                include_tournament_features=True,
                tournament_total_chips=600.0,
            )
            candidate = encode_information_state(
                native_state,
                native_state.to_act,
                legal,
                200.0,
                32,
                include_tournament_features=True,
                tournament_total_chips=600.0,
            )
            if not torch.equal(reference, candidate):
                different = torch.nonzero(reference != candidate).flatten()
                raise AssertionError(
                    f"encoder mismatch in hand {hand} at features {different[:10].tolist()}"
                )
            checked += 1
            action = chooser.choice(legal)
            py_state = py_env.step(py_state, action)
            native_state = native_env.step(native_state, action)

    deck = list(range(52))
    chooser.shuffle(deck)
    py_env, native_env = PythonEnv(seed=1), NativeEnv(seed=1)
    py_state = py_env.new_hand(button=0, deck=deck)
    native_state = native_env.new_hand(button=0, deck=deck)
    legal = py_env.legal_actions(py_state)
    repetitions = 100_000
    started = time.perf_counter()
    for _ in range(repetitions):
        encode_information_state(
            py_state, py_state.to_act, legal, 200.0, 32,
            include_tournament_features=True, tournament_total_chips=600.0,
        )
    python_seconds = time.perf_counter() - started
    started = time.perf_counter()
    for _ in range(repetitions):
        encode_information_state(
            native_state, native_state.to_act, legal, 200.0, 32,
            include_tournament_features=True, tournament_total_chips=600.0,
        )
    native_seconds = time.perf_counter() - started

    Path(__file__).with_name("STAGING_VALIDATED").write_text(
        f"states={checked}\npython_seconds={python_seconds:.6f}\n"
        f"native_seconds={native_seconds:.6f}\n"
        f"speedup={python_seconds / native_seconds:.6f}\n",
        encoding="utf-8",
    )
    print(f"Exact equality: {checked} randomized legal states")
    print(
        f"Encoder: Python {python_seconds:.3f}s, native {native_seconds:.3f}s, "
        f"speedup {python_seconds / native_seconds:.2f}x"
    )


if __name__ == "__main__":
    main()
