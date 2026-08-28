"""Atomically activate a verified reservoir-capacity migration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, value: object) -> None:
    _atomic_text(path, json.dumps(value, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--capacity", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    args = parser.parse_args()

    artifact_dir = args.artifact_dir.resolve()
    source = args.source.resolve()
    destination = args.destination.resolve()
    latest_path = artifact_dir / "latest.json"
    metrics_path = artifact_dir / "metrics.jsonl"
    report_path = artifact_dir / "reservoir_migration_3m_to_2m.json"
    archive_path = artifact_dir / "metrics_uncheckpointed_after_275.jsonl"

    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if int(latest["iteration"]) != args.iteration:
        raise ValueError("latest manifest iteration changed during migration")
    if Path(latest["checkpoint"]).resolve() != source:
        raise ValueError("latest manifest no longer points to the source checkpoint")
    if int(report["iteration"]) != args.iteration:
        raise ValueError("migration report iteration does not match")
    if int(report["capacity"]) != args.capacity:
        raise ValueError("migration report capacity does not match")
    if not destination.is_file():
        raise FileNotFoundError(destination)
    if not source.is_file():
        raise FileNotFoundError(source)

    retained_lines: list[str] = []
    uncheckpointed_lines: list[str] = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if int(row["iteration"]) <= args.iteration:
            retained_lines.append(line)
        else:
            uncheckpointed_lines.append(line)
    if uncheckpointed_lines:
        _atomic_text(archive_path, "\n".join(uncheckpointed_lines) + "\n")
    _atomic_text(metrics_path, "\n".join(retained_lines) + "\n")

    latest["checkpoint"] = str(destination)
    latest["iteration"] = args.iteration
    latest["last_fitted_iteration"] = args.iteration
    latest["incomplete_fit"] = False
    latest["emergency"] = False
    latest["failed"] = False
    campaign = dict(latest.get("campaign", {}))
    campaign["batch_size"] = args.batch_size
    campaign["advantage_steps"] = args.steps
    campaign["policy_steps"] = args.steps
    latest["campaign"] = campaign
    latest["reservoir_capacity_migration"] = {
        "previous_capacity": int(report["previous_capacity"]),
        "capacity": args.capacity,
        "selection": "uniform_subset_without_replacement",
        "seen_counters_preserved": True,
        "report": str(report_path),
    }
    _atomic_json(latest_path, latest)

    source.unlink()
    print(
        json.dumps(
            {
                "active_checkpoint": str(destination),
                "deleted_checkpoint": str(source),
                "retained_metrics": len(retained_lines),
                "archived_uncheckpointed_metrics": len(uncheckpointed_lines),
                "metrics_archive": (
                    str(archive_path) if uncheckpointed_lines else None
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
