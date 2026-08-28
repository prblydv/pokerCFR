"""Reproducible scalar-versus-batched HU root-search benchmark."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from heads_up_engine import HeadsUpHoldemEngine
from heads_up_pluribus_search import BlueprintPublicRange
from heads_up_root_policy_search import HeadsUpRootPolicySearch
from play_heads_up_gui import HeadsUpSnapshotPolicy


class _ScalarOnlyPolicy:
    """Expose the pre-batching one-state policy contract."""

    def __init__(self, policy: HeadsUpSnapshotPolicy) -> None:
        self.policy = policy

    def probabilities(self, env, state) -> torch.Tensor:
        return self.policy._legacy_single_probabilities(env, state)


def _run(
    name: str,
    policy,
    env,
    state,
    blueprint: torch.Tensor,
    public_range,
    *,
    seconds: float,
    batch_iterations: int,
    native_rollouts: bool,
    batched_action_sampling: bool,
    batch_step: bool,
    seed: int,
) -> dict:
    search = HeadsUpRootPolicySearch(
        policy,
        time_budget_ms=max(1, round(1000.0 * seconds)),
        max_rollouts=150_000,
        batch_iterations=batch_iterations,
        use_native_rollouts=native_rollouts,
        use_batched_action_sampling=batched_action_sampling,
        use_batch_step=batch_step,
        range_mode="inferred",
        seed=seed,
    )
    started = time.perf_counter()
    result = search.resolve(env, state, blueprint, public_range)
    elapsed = time.perf_counter() - started
    return {
        "name": name,
        "elapsed_seconds": elapsed,
        "terminal_rollouts": result.terminal_rollouts,
        "cfr_iterations": result.cfr_iterations,
        "rollouts_per_second": result.terminal_rollouts / elapsed,
        "native_rollouts": result.native_backend,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("policy", type=Path)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--batch-iterations", type=int, default=3072)
    parser.add_argument("--current-batch-iterations", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()
    if args.seconds <= 0.0:
        parser.error("--seconds must be positive")
    if args.batch_iterations <= 0:
        parser.error("--batch-iterations must be positive")
    if args.current_batch_iterations <= 0:
        parser.error("--current-batch-iterations must be positive")

    env = HeadsUpHoldemEngine(200, 1, 2, seed=13)
    state = env.new_hand(button=0)
    public_range = BlueprintPublicRange()
    public_range.reset(state.hole[int(state.to_act)])
    range_snapshot = public_range.snapshot()
    cpu_policy = HeadsUpSnapshotPolicy(args.policy, device="cpu", seed=args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batched_policy = HeadsUpSnapshotPolicy(
        args.policy, device=device, seed=args.seed
    )
    current_policy = HeadsUpSnapshotPolicy(
        args.policy,
        device=device,
        native_batch_encoding=False,
        seed=args.seed,
    )
    blueprint = cpu_policy.probabilities(env, state)

    baseline = _run(
        "legacy_cpu_scalar_python",
        _ScalarOnlyPolicy(cpu_policy),
        env,
        state,
        blueprint,
        range_snapshot,
        seconds=args.seconds,
        batch_iterations=1,
        native_rollouts=False,
        batched_action_sampling=False,
        batch_step=False,
        seed=args.seed,
    )
    current = _run(
        f"current_batched_{device}_native",
        current_policy,
        env,
        state,
        blueprint,
        range_snapshot,
        seconds=args.seconds,
        batch_iterations=args.current_batch_iterations,
        native_rollouts=True,
        batched_action_sampling=False,
        batch_step=False,
        seed=args.seed,
    )
    optimized = _run(
        f"new_native_batch_{device}",
        batched_policy,
        env,
        state,
        blueprint,
        range_snapshot,
        seconds=args.seconds,
        batch_iterations=args.batch_iterations,
        native_rollouts=True,
        batched_action_sampling=True,
        batch_step=True,
        seed=args.seed,
    )
    report = {
        "policy": str(args.policy.resolve()),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "batch_iterations": args.batch_iterations,
        "current_batch_iterations": args.current_batch_iterations,
        "baseline": baseline,
        "current": current,
        "optimized": optimized,
        "versus_original_multiplier": (
            optimized["rollouts_per_second"]
            / baseline["rollouts_per_second"]
        ),
        "versus_current_multiplier": (
            optimized["rollouts_per_second"]
            / current["rollouts_per_second"]
        ),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
