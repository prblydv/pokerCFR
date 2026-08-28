"""Create the code-only Vast.ai package for compact hidden-384 HU training."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "vast_heads_up_compact_v6_hidden384_20260804.tar.gz"
ARCHIVE_ROOT = "pokerCFR_compact_v6"

FILES = (
    "README_HU.md",
    "README_HU_COMPACT.md",
    "requirements-heads-up.txt",
    "heads_up_compact_training.ipynb",
    "build_heads_up_compact_notebook.py",
    "heads_up_compact_range005_training.ipynb",
    "build_heads_up_compact_range005_notebook.py",
    "setup_vast_compact_v6.sh",
    "benchmark_heads_up_compact_encoder.py",
    "evaluate_heads_up_ensemble_profitability.py",
    "evaluate_heads_up_exploitability.py",
    "heads_up_compact.py",
    "heads_up_cfr.py",
    "heads_up_engine.py",
    "heads_up_models.py",
    "heads_up_native.py",
    "heads_up_production.py",
    "heads_up_reporting.py",
    "heads_up_ranges.py",
    "heads_up_analysis.py",
    "heads_up_pluribus_search.py",
    "heads_up_robust_search.py",
    "heads_up_root_policy_search.py",
    "heads_up_search.py",
    "train_heads_up.py",
    "play_heads_up_gui.py",
    "test_heads_up_compact.py",
    "test_heads_up_engine.py",
    "test_heads_up_models.py",
    "test_heads_up_native_engine.py",
    "test_heads_up_training.py",
    "test_heads_up_analysis.py",
    "test_heads_up_ensemble_profitability.py",
    "test_heads_up_exploitability.py",
    "engine C HU/heads_up_native_engine.cpp",
    "reference_policies/policy_00000725.pt",
    "reference_policies/policy_00000950.pt",
    "reference_policies/policy_00001025.pt",
    "artifacts/heads_up_v4_paper3x/evaluations/ensemble_725_950_1025_top3_vs_top4_100000.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    missing = [name for name in FILES if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError(f"migration input missing: {missing}")
    manifest = {
        "version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": "heads_up_compact_v6_hidden384",
        "fresh_artifact_dir": "artifacts/heads_up_compact_v6_hidden384",
        "encoder_schema_version": "hu_compact_information_state_v1_full_history",
        "physical_input_dim": 782,
        "logical_input_dim": "40 + 7 * public_history_length",
        "max_history": 106,
        "history_overflow_policy": "error_never_truncate",
        "card_storage": "seven_exact_ids_plus_one_zero_pad",
        "architecture": "hu_deep_cfr_compact_v6",
        "policy_architecture": "hu_deep_cfr_compact_v6_policy_range_v1",
        "hidden": 384,
        "range_training": True,
        "range_combinations": 1326,
        "cfr_algorithm_changed": False,
        "worker_rule": "POKER_CPU_WORKERS or min(16, effective_cpu_count())",
        "contains_training_checkpoints_or_reservoirs": False,
        "frozen_benchmarks": [725, 950, 1025],
        "files": {
            name.replace("\\", "/"): {
                "bytes": (ROOT / name).stat().st_size,
                "sha256": sha256(ROOT / name),
            }
            for name in FILES
        },
    }
    manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")
    with tarfile.open(OUTPUT, "w:gz", compresslevel=6) as archive:
        for name in FILES:
            archive_name = name.replace("\\", "/")
            archive.add(
                ROOT / name,
                arcname=f"{ARCHIVE_ROOT}/{archive_name}",
                recursive=False,
            )
        info = tarfile.TarInfo(f"{ARCHIVE_ROOT}/MIGRATION_MANIFEST.json")
        info.size = len(manifest_bytes)
        info.mtime = int(datetime.now(timezone.utc).timestamp())
        info.mode = 0o644
        archive.addfile(info, io.BytesIO(manifest_bytes))
    print(OUTPUT)
    print(f"bytes={OUTPUT.stat().st_size}")
    print(f"sha256={sha256(OUTPUT)}")


if __name__ == "__main__":
    main()
