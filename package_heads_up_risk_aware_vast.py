"""Create a code-only Vast package for the fresh risk-aware campaign."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "vast_heads_up_risk_aware_hidden512_20260804.tar.gz"
ARCHIVE_ROOT = "pokerCFR_risk_aware"

FILES = (
    "README_RISK_AWARE_VAST.md",
    "README_HU.md",
    "requirements-heads-up.txt",
    "heads_up_risk_aware_training.ipynb",
    "build_heads_up_risk_aware_notebook.py",
    "setup_vast_risk_aware.sh",
    "benchmark_heads_up_risk_aware.py",
    "heads_up_risk_aware_eval.py",
    "heads_up_action_conditioned_eval.py",
    "evaluate_heads_up_ensemble_profitability.py",
    "evaluate_heads_up_exploitability.py",
    "heads_up_cfr.py",
    "heads_up_compact.py",
    "heads_up_engine.py",
    "heads_up_models.py",
    "heads_up_native.py",
    "heads_up_production.py",
    "heads_up_reporting.py",
    "heads_up_ranges.py",
    "heads_up_analysis.py",
    "heads_up_search.py",
    "test_heads_up_risk_aware_eval.py",
    "test_heads_up_action_conditioned_eval.py",
    "test_heads_up_ensemble_profitability.py",
    "test_heads_up_engine.py",
    "test_heads_up_models.py",
    "test_heads_up_native_engine.py",
    "test_heads_up_training.py",
    "reference_policies/policy_00000725.pt",
    "reference_policies/policy_00000950.pt",
    "reference_policies/policy_00001025.pt",
)

SPECIAL_FILES = {
    "engine C HU/heads_up_native_engine.cpp": "heads_up_native_engine.cpp",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    sources = (*FILES, *SPECIAL_FILES)
    missing = [name for name in sources if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError(f"migration input missing: {missing}")
    manifest_files = {
        name.replace("\\", "/"): {
            "bytes": (ROOT / name).stat().st_size,
            "sha256": sha256(ROOT / name),
        }
        for name in sources
    }
    manifest = {
        "version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": "heads_up_risk_aware_hidden512_v1",
        "fresh_artifact_dir": "artifacts/heads_up_risk_aware_hidden512_v1",
        "architecture": "hu_deep_cfr_compact_v4",
        "input_dim": 1038,
        "hidden": 512,
        "range_training": False,
        "risk_aware_all_in": True,
        "all_in_risk_threshold": 2.0,
        "advantage_loss": "smooth_l1",
        "reservoir_fit": "one_shuffled_pass_per_iteration",
        "advantage_reset_iterations": "25,26,27,...",
        "contains_training_checkpoints_or_reservoirs": False,
        "frozen_benchmarks": [725, 950, 1025],
        "files": manifest_files,
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
        for source, destination in SPECIAL_FILES.items():
            archive.add(
                ROOT / source,
                arcname=f"{ARCHIVE_ROOT}/{destination}",
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
