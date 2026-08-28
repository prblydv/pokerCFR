"""High-volume differential and alias-safety checks for packed native states."""

import pickle
import random

import poker_native_engine
import torch

from test_native_engine import state_view
from three_player_engine import ThreePlayerHoldemEnv as PythonEnv
from three_player_models import encode_information_state
from three_player_native import ThreePlayerHoldemEnv as NativeEnv


def main(hands: int = 5_000) -> None:
    chooser = random.Random(604_177)
    stack_modes = (
        [200.0, 200.0, 200.0],
        [350.0, 0.0, 250.0],
        [20.0, 100.0, 480.0],
        [598.0, 1.0, 1.0],
        [75.5, 410.25, 114.25],
    )
    states = branches = roundtrips = pickle_roundtrips = encoder_rows = 0
    maximum_history = 0
    for hand in range(hands):
        deck = list(range(52))
        random.Random(700_000 + hand).shuffle(deck)
        stacks = stack_modes[hand % len(stack_modes)]
        live = [seat for seat, value in enumerate(stacks) if value > 0]
        button = live[hand % len(live)]
        py_env, native_env = PythonEnv(seed=1), NativeEnv(seed=1)
        py_state = py_env.new_hand(button=button, stacks=stacks, deck=deck)
        native_state = native_env.new_hand(button=button, stacks=stacks, deck=deck)
        while True:
            if state_view(native_state) != state_view(py_state):
                raise AssertionError(f"state mismatch at hand {hand}, state {states}")
            maximum_history = max(maximum_history, len(native_state.history))
            states += 1

            native_dict = poker_native_engine.state_to_dict(native_state)
            if states % 13 == 0:
                restored = poker_native_engine.state_from_dict(native_dict)
                if state_view(restored) != state_view(native_state):
                    raise AssertionError("state dictionary round-trip mismatch")
                roundtrips += 1
            if states % 29 == 0:
                restored = pickle.loads(pickle.dumps(native_state, protocol=5))
                if state_view(restored) != state_view(native_state):
                    raise AssertionError("pickle round-trip mismatch")
                pickle_roundtrips += 1
            if py_state.terminal:
                break

            py_legal = py_env.legal_actions(py_state)
            native_legal = native_env.legal_actions(native_state)
            if native_legal != py_legal:
                raise AssertionError("legal actions diverged")
            for action in py_legal:
                py_target = py_env.action_target(py_state, action)
                native_target = native_env.action_target(native_state, action)
                if py_target != native_target:
                    raise AssertionError("action target diverged")
                before = state_view(native_state)
                py_child = py_env.step(py_state, action)
                native_child = native_env.step(native_state, action)
                if state_view(native_child) != state_view(py_child):
                    raise AssertionError(
                        f"branch mismatch at hand {hand}, action {action}"
                    )
                if state_view(native_state) != before:
                    raise AssertionError("native parent mutated while branching")
                branches += 1

            reference = encode_information_state(
                py_state,
                py_state.to_act,
                py_legal,
                200.0,
                32,
                include_tournament_features=True,
                tournament_total_chips=600.0,
            )
            candidate = encode_information_state(
                native_state,
                native_state.to_act,
                native_legal,
                200.0,
                32,
                include_tournament_features=True,
                tournament_total_chips=600.0,
            )
            if not torch.equal(reference, candidate):
                raise AssertionError("information-state encoder diverged")
            encoder_rows += 1
            action = chooser.choice(py_legal)
            py_state = py_env.step(py_state, action)
            native_state = native_env.step(native_state, action)

    print(f"hands={hands}")
    print(f"states={states}")
    print(f"all_legal_branches={branches}")
    print(f"state_dict_roundtrips={roundtrips}")
    print(f"pickle_roundtrips={pickle_roundtrips}")
    print(f"encoder_rows={encoder_rows}")
    print(f"maximum_natural_history={maximum_history}")
    print("ALL PACKED-STATE CHECKS EXACT")


if __name__ == "__main__":
    main()
