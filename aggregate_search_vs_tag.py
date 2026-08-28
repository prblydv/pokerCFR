"""Pool independent search-vs-TAG worker reports correctly."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


SUMMARY_KEYS = (
    "raw_policy_vs_two_tag",
    "search_policy_vs_two_tag",
    "paired_search_minus_raw",
)


def _pooled_summary(reports: list[dict], key: str) -> dict[str, float]:
    parts = [report[key] for report in reports]
    total = sum(int(part["hands"]) for part in parts)
    mean = sum(
        int(part["hands"]) * float(part["mean_bb_per_hand"])
        for part in parts
    ) / total
    sum_squares = 0.0
    wins = 0.0
    for part in parts:
        count = int(part["hands"])
        part_mean = float(part["mean_bb_per_hand"])
        part_variance = float(part["standard_error"]) ** 2 * count
        sum_squares += (count - 1) * part_variance
        sum_squares += count * (part_mean - mean) ** 2
        wins += count * float(part["win_fraction"])
    variance = sum_squares / (total - 1)
    standard_error = math.sqrt(variance / total)
    return {
        "hands": total,
        "mean_bb_per_hand": mean,
        "standard_error": standard_error,
        "ci95_low": mean - 1.96 * standard_error,
        "ci95_high": mean + 1.96 * standard_error,
        "win_fraction": wins / total,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("workers", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-hands", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reports = [
        json.loads(path.read_text(encoding="utf-8")) for path in args.workers
    ]
    if not reports:
        raise ValueError("at least one worker report is required")
    pooled = {key: _pooled_summary(reports, key) for key in SUMMARY_KEYS}
    hands = int(pooled["search_policy_vs_two_tag"]["hands"])
    if args.expected_hands is not None and hands != args.expected_hands:
        raise ValueError(f"expected {args.expected_hands} hands, found {hands}")
    search_decisions = sum(
        int(report["search_diagnostics"]["search_decisions"])
        for report in reports
    )
    pooled["configuration"] = {
        key: reports[0][key]
        for key in (
            "checkpoint",
            "iteration",
            "search_ms",
            "search_rollouts",
            "torch_threads",
        )
    }
    pooled["workers"] = len(reports)
    pooled["search_diagnostics"] = {
        "search_decisions": search_decisions,
        "mean_search_ms": sum(
            int(report["search_diagnostics"]["search_decisions"])
            * float(report["search_diagnostics"]["mean_search_ms"])
            for report in reports
        )
        / search_decisions,
        "mean_rollouts": sum(
            int(report["search_diagnostics"]["search_decisions"])
            * float(report["search_diagnostics"]["mean_rollouts"])
            for report in reports
        )
        / search_decisions,
        "worker_elapsed_seconds_max": max(
            float(report["search_diagnostics"]["elapsed_seconds"])
            for report in reports
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(pooled, indent=2), encoding="utf-8")
    print(json.dumps(pooled, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
