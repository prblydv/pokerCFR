"""Multiprocess profile-blind search evaluation against actual TAG opponents."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import time
from pathlib import Path

import torch

from evaluate_search_vs_tag import _deals, _play_mode, _summary
from play_three_player_gui import SnapshotPolicy


def _worker(task):
    (
        worker,
        checkpoint,
        deals,
        seed,
        search_ms,
        search_rollouts,
    ) = task
    torch.set_num_threads(1)
    policy = SnapshotPolicy(Path(checkpoint))
    raw, raw_diagnostics = _play_mode(
        policy,
        deals,
        mode="raw",
        seed=seed,
        search_ms=search_ms,
        search_rollouts=search_rollouts,
        blind_search=True,
    )
    search, search_diagnostics = _play_mode(
        policy,
        deals,
        mode="search",
        seed=seed,
        search_ms=search_ms,
        search_rollouts=search_rollouts,
        blind_search=True,
    )
    return {
        "worker": worker,
        "raw": raw,
        "search": search,
        "raw_diagnostics": raw_diagnostics,
        "search_diagnostics": search_diagnostics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/downloaded_blueprints/policy_00008100.pt"),
    )
    parser.add_argument("--hands", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=28100)
    parser.add_argument("--search-ms", type=int, default=7000)
    parser.add_argument("--search-rollouts", type=int, default=150000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/blind_search_vs_tag_8100_1000.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.hands <= 0:
        raise ValueError("hands must be positive")
    if args.workers <= 0:
        raise ValueError("workers must be positive")

    checkpoint = str(args.checkpoint.resolve())
    deals = _deals(args.hands, args.seed)
    chunks = [deals[index:: args.workers] for index in range(args.workers)]
    tasks = [
        (
            worker,
            checkpoint,
            chunk,
            args.seed + 10_000 * worker,
            args.search_ms,
            args.search_rollouts,
        )
        for worker, chunk in enumerate(chunks)
        if chunk
    ]

    started = time.perf_counter()
    context = multiprocessing.get_context("spawn")
    with context.Pool(processes=len(tasks)) as pool:
        worker_results = pool.map(_worker, tasks)
    wall_seconds = time.perf_counter() - started

    raw = [
        value
        for result in worker_results
        for value in result["raw"]
    ]
    search = [
        value
        for result in worker_results
        for value in result["search"]
    ]
    delta = [
        search_value - raw_value
        for search_value, raw_value in zip(search, raw)
    ]
    search_decisions = sum(
        int(result["search_diagnostics"]["search_decisions"])
        for result in worker_results
    )
    report = {
        "checkpoint": checkpoint,
        "iteration": 8100,
        "hands": args.hands,
        "workers": len(tasks),
        "seed": args.seed,
        "search_ms": args.search_ms,
        "search_rollouts": args.search_rollouts,
        "opponent_knowledge": {
            "actual_opponents": "two scripted tight-aggressive bots",
            "search_opponent_model": "checkpoint blueprint only",
            "tag_identity_supplied_to_search": False,
            "actual_opponent_cards_supplied_to_search": False,
            "opponent_hole_cards": "sampled from unseen cards per determinization",
        },
        "raw_policy_vs_two_tag": _summary(raw),
        "blind_search_policy_vs_two_tag": _summary(search),
        "paired_blind_search_minus_raw": _summary(delta),
        "search_diagnostics": {
            "search_decisions": search_decisions,
            "mean_search_ms": sum(
                int(result["search_diagnostics"]["search_decisions"])
                * float(result["search_diagnostics"]["mean_search_ms"])
                for result in worker_results
            )
            / search_decisions,
            "mean_rollouts": sum(
                int(result["search_diagnostics"]["search_decisions"])
                * float(result["search_diagnostics"]["mean_rollouts"])
                for result in worker_results
            )
            / search_decisions,
            "wall_seconds": wall_seconds,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
