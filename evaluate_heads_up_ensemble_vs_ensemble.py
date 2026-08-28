"""Run a recoverable reciprocal match between two HU policy ensembles."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

from evaluate_heads_up_ensemble_profitability import (
    EnsembleProvider,
    run_reciprocal_match,
)
from heads_up_cfr import HeadsUpNeuralCFR
from heads_up_native import HeadsUpHoldemEngine
from heads_up_production import load_policy_snapshot


Z_SCORES = {"95": 1.959963984540054, "99": 2.5758293035489004, "99_9": 3.2905267314919255}


def _load_ensemble(paths: list[Path], *, device: str):
    return [load_policy_snapshot(path, device=device) for path in paths]


def _common_metadata(snapshots) -> dict:
    first = snapshots[0].metadata
    for snapshot in snapshots[1:]:
        for key in ("input_dim", "max_history", "encoder_schema_version", "environment"):
            if snapshot.metadata.get(key) != first.get(key):
                raise ValueError(f"incompatible policy metadata for {key}")
    return first


def _aggregate(chunks: list[dict], elapsed_seconds: float) -> dict:
    total_hands = sum(int(chunk["hands"]) for chunk in chunks)
    total_pairs = sum(int(chunk["reciprocal_pairs"]) for chunk in chunks)
    weighted_sum = sum(
        float(chunk["mean_ev_bb_per_hand"]) * int(chunk["reciprocal_pairs"])
        for chunk in chunks
    )
    grand_mean = weighted_sum / total_pairs
    sum_squares = 0.0
    for chunk in chunks:
        n = int(chunk["reciprocal_pairs"])
        chunk_mean = float(chunk["mean_ev_bb_per_hand"])
        chunk_stderr = float(chunk["paired_stderr_bb_per_hand"])
        chunk_variance = (chunk_stderr * math.sqrt(n)) ** 2
        sum_squares += (n - 1) * chunk_variance + n * (chunk_mean - grand_mean) ** 2
    variance = sum_squares / (total_pairs - 1)
    stderr = math.sqrt(variance / total_pairs)
    confidence_intervals = {
        label: {
            "low_bb_per_hand": grand_mean - z * stderr,
            "high_bb_per_hand": grand_mean + z * stderr,
            "low_bb_per_100": 100.0 * (grand_mean - z * stderr),
            "high_bb_per_100": 100.0 * (grand_mean + z * stderr),
        }
        for label, z in Z_SCORES.items()
    }
    candidate_actions = Counter()
    opponent_actions = Counter()
    for chunk in chunks:
        candidate_actions.update(chunk["candidate_action_counts"])
        opponent_actions.update(chunk["opponent_action_counts"])
    candidate_total_actions = sum(candidate_actions.values())
    opponent_total_actions = sum(opponent_actions.values())
    return {
        "candidate": chunks[0]["candidate"],
        "opponent": chunks[0]["opponent"],
        "hands": total_hands,
        "reciprocal_pairs": total_pairs,
        "independent_reciprocal_blocks": len(chunks),
        "wins": sum(int(chunk["wins"]) for chunk in chunks),
        "losses": sum(int(chunk["losses"]) for chunk in chunks),
        "ties": sum(int(chunk["ties"]) for chunk in chunks),
        "win_rate": sum(int(chunk["wins"]) for chunk in chunks) / total_hands,
        "mean_ev_bb_per_hand": grand_mean,
        "mean_ev_bb_per_100": 100.0 * grand_mean,
        "paired_stderr_bb_per_hand": stderr,
        "confidence_intervals": confidence_intervals,
        "candidate_action_counts": dict(candidate_actions),
        "opponent_action_counts": dict(opponent_actions),
        "candidate_all_in_actions": int(candidate_actions["all_in"]),
        "candidate_total_actions": candidate_total_actions,
        "candidate_all_in_action_rate": candidate_actions["all_in"] / candidate_total_actions,
        "opponent_all_in_actions": int(opponent_actions["all_in"]),
        "opponent_total_actions": opponent_total_actions,
        "opponent_all_in_action_rate": opponent_actions["all_in"] / opponent_total_actions,
        "candidate_all_in_hands": sum(int(chunk["candidate_all_in_hands"]) for chunk in chunks),
        "candidate_all_in_hand_rate": sum(int(chunk["candidate_all_in_hands"]) for chunk in chunks) / total_hands,
        "candidate_all_in_net_bb": sum(float(chunk["candidate_all_in_net_bb"]) for chunk in chunks),
        "candidate_non_all_in_net_bb": sum(float(chunk["candidate_non_all_in_net_bb"]) for chunk in chunks),
        "elapsed_seconds": elapsed_seconds,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-policies", nargs="+", type=Path, required=True)
    parser.add_argument("--opponent-policies", nargs="+", type=Path, required=True)
    parser.add_argument("--hands", type=int, default=1_000_000)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=1_200_725_950)
    parser.add_argument("--seed-stride", type=int, default=1_000_003)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--inference-batch-size", type=int, default=8192)
    parser.add_argument("--simulation-batch-size", type=int, default=20000)
    parser.add_argument("--chunk-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.hands <= 0 or args.hands % args.blocks or (args.hands // args.blocks) % 2:
        raise ValueError("hands must divide into positive, even-sized blocks")

    candidate_snapshots = _load_ensemble(args.candidate_policies, device=args.device)
    opponent_snapshots = _load_ensemble(args.opponent_policies, device=args.device)
    metadata = _common_metadata(candidate_snapshots + opponent_snapshots)
    environment = dict(metadata["environment"])
    encoder_env = HeadsUpHoldemEngine(
        starting_stack=int(environment["starting_stack"]),
        small_blind=int(environment["small_blind"]),
        big_blind=int(environment["big_blind"]),
        seed=1,
    )
    trainer = HeadsUpNeuralCFR(
        encoder_env,
        device=args.device,
        hidden=int(metadata["hidden"]),
        blocks=int(metadata["blocks"]),
        advantage_capacity=1,
        policy_capacity=1,
        max_history=int(metadata["max_history"]),
        seed=1,
    )
    candidate = EnsembleProvider(trainer, candidate_snapshots, top_k=args.top_k)
    opponent = EnsembleProvider(trainer, opponent_snapshots, top_k=args.top_k)
    args.chunk_dir.mkdir(parents=True, exist_ok=True)
    hands_per_block = args.hands // args.blocks
    chunks = []
    started = time.perf_counter()
    for index in range(args.blocks):
        seed = args.base_seed + index * args.seed_stride
        path = args.chunk_dir / f"chunk_{index + 1:02d}_seed_{seed}.json"
        if path.exists():
            chunk = json.loads(path.read_text(encoding="utf-8"))
        else:
            chunk = run_reciprocal_match(
                candidate,
                lambda env, provider=opponent: provider,
                environment=environment,
                hands=hands_per_block,
                seed=seed,
                inference_batch_size=args.inference_batch_size,
                simulation_batch_size=args.simulation_batch_size,
            )
            path.write_text(json.dumps(chunk, indent=2) + "\n", encoding="utf-8")
        chunks.append(chunk)
        print(
            f"block {index + 1}/{args.blocks}: {chunk['mean_ev_bb_per_100']:+.4f} BB/100; "
            f"cumulative hands {sum(int(item['hands']) for item in chunks):,}",
            flush=True,
        )

    result = {
        "configuration": {
            "hands": args.hands,
            "reciprocal_pairs": args.hands // 2,
            "blocks": args.blocks,
            "hands_per_block": hands_per_block,
            "common_deal_reciprocal_seat_swaps_within_each_block": True,
            "base_seed": args.base_seed,
            "seed_stride": args.seed_stride,
            "sampling": (
                f"equal-weight average probabilities, retain top {args.top_k} legal actions, "
                "renormalize, stochastic sample"
                if args.top_k > 0
                else "equal-weight average probabilities, raw stochastic sample with no top-k filtering"
            ),
            "candidate_components": [snapshot.iteration for snapshot in candidate_snapshots],
            "candidate_weights": [1.0 / len(candidate_snapshots)] * len(candidate_snapshots),
            "opponent_components": [snapshot.iteration for snapshot in opponent_snapshots],
            "opponent_weights": [1.0 / len(opponent_snapshots)] * len(opponent_snapshots),
            "device": args.device,
            "inference_batch_size": args.inference_batch_size,
            "simulation_batch_size": args.simulation_batch_size,
        },
        "aggregate": _aggregate(chunks, time.perf_counter() - started),
        "chunks": chunks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
