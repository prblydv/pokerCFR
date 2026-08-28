"""Reproducible throughput check for the compact Python and C++ encoders."""

from __future__ import annotations

import argparse
import json
import time

from heads_up_compact import encode_compact_information_state
from heads_up_engine import HeadsUpHoldemEngine as PythonEngine
from heads_up_models import build_action_descriptors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=int, default=50_000)
    args = parser.parse_args()
    count = int(args.states)
    if count <= 0:
        raise ValueError("--states must be positive")

    python_env = PythonEngine(seed=801)
    python_state = python_env.new_hand(button=0)
    legal = python_env.legal_actions(python_state)
    descriptors = build_action_descriptors(python_env, python_state)
    started = time.perf_counter()
    for _ in range(count):
        encode_compact_information_state(
            python_state,
            int(python_state.to_act),
            legal,
            python_env.bb,
            action_descriptors=descriptors,
        )
    python_seconds = time.perf_counter() - started

    import heads_up_native_engine as native_module
    from heads_up_native import (
        HeadsUpHoldemEngine as NativeEngine,
        encode_compact_information_states_native,
    )

    if int(native_module.NATIVE_ABI_VERSION) < 6:
        raise RuntimeError("benchmark requires the compact native ABI 6 build")
    native_env = NativeEngine(seed=801)
    native_state = native_env.new_hand(button=0)
    states = [native_state] * count
    started = time.perf_counter()
    encoded, masks = encode_compact_information_states_native(native_env, states)
    native_seconds = time.perf_counter() - started
    result = {
        "states": count,
        "input_dim": int(encoded.shape[1]),
        "native_abi": int(native_module.NATIVE_ABI_VERSION),
        "python_states_per_second": count / python_seconds,
        "native_batch_states_per_second": count / native_seconds,
        "native_speedup": python_seconds / native_seconds,
        "python_seconds": python_seconds,
        "native_seconds": native_seconds,
        "legal_mask_shape": list(masks.shape),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
