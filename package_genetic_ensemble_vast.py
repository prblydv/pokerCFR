"""Create a self-contained Vast package for weighted ensemble optimization."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "vast_genetic_ensemble_optimizer_20260806.tar.gz"
ARCHIVE_ROOT = "pokerCFR_genetic_ensemble"

FILES = (
    "optimize_heads_up_ensemble_genetic.py",
    "evaluate_heads_up_ensemble_profitability.py",
    "heads_up_cfr.py",
    "heads_up_compact.py",
    "heads_up_engine.py",
    "heads_up_models.py",
    "heads_up_native.py",
    "heads_up_production.py",
    "heads_up_reporting.py",
    "heads_up_ranges.py",
    "README_GENETIC_ENSEMBLE.md",
    "requirements-heads-up.txt",
    "setup_genetic_ensemble_vast.sh",
    "run_genetic_ensemble_optimizer_vast.sh",
    "artifacts/heads_up_v4_paper3x/snapshots/policy_00000725.pt",
    "artifacts/heads_up_v4_paper3x/snapshots/policy_00000950.pt",
    "artifacts/heads_up_v4_paper3x/snapshots/policy_00001025.pt",
    "artifacts/downloaded_risk_aware/policy_00000200.pt",
    "artifacts/downloaded_risk_aware/policy_00000275.pt",
    "artifacts/downloaded_risk_aware/policy_00000300.pt",
    "artifacts/downloaded_risk_aware/policy_00000400.pt",
)
SPECIAL_FILES = {
    "engine C HU/heads_up_native_engine.cpp": "heads_up_native_engine.cpp",
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    sources = (*FILES, *SPECIAL_FILES)
    missing = [name for name in sources if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError(f"migration input missing: {missing}")
    manifest = {
        "version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "weighted heads-up ensemble genetic optimization",
        "optimizer_started": False,
        "files": {
            name.replace("\\", "/"): {
                "bytes": (ROOT / name).stat().st_size,
                "sha256": sha256(ROOT / name),
            }
            for name in sources
        },
    }
    manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")
    with tarfile.open(OUTPUT, "w:gz", compresslevel=3) as archive:
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
