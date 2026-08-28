"""Build the memory-safe range-alpha-5% compact HU training notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "heads_up_compact_training.ipynb"
OUTPUT = ROOT / "heads_up_compact_range005_training.ipynb"

notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
cells = notebook["cells"]


def replace(cell: int, old: str, new: str, *, expected: int = 1) -> None:
    text = "".join(cells[cell]["source"])
    found = text.count(old)
    if found != expected:
        raise RuntimeError(
            f"cell {cell} expected {expected} occurrence(s) of {old!r}, found {found}"
        )
    cells[cell]["source"] = text.replace(old, new).splitlines(keepends=True)


replace(
    0,
    "# Compact hidden-384 heads-up Deep CFR production training",
    "# Memory-safe compact hidden-384 HU training with 5% range loss",
)
replace(
    0,
    "This is a fresh, isolated campaign using the lossless-through-100BB compact",
    "This is a fresh, isolated range-alpha-5% campaign using the lossless-through-100BB compact",
)
replace(
    2,
    "import os\n",
    "import os\nos.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')\n",
)
replace(
    8,
    "Path('artifacts/heads_up_compact_v6_hidden384')",
    "Path('artifacts/heads_up_compact_v6_hidden384_range005_memorysafe')",
)
replace(8, "'range_loss_weight': 0.01,", "'range_loss_weight': 0.05,", expected=2)
replace(
    8,
    "'advantage_steps': 977, 'policy_steps': 977,",
    "'advantage_steps': 1_954, 'policy_steps': 1_954,",
)
replace(
    8,
    "'batch_size': 8_192, 'evaluate_every': 25,",
    "'batch_size': 4_096, 'evaluate_every': 25,",
)
replace(
    8,
    "'range_batch_size': 2_048,",
    "'range_batch_size': 1_024,",
)
replace(
    8,
    "# 977 x 8,192 covers every row of an 8M reservoir once.",
    "# 1,954 x 4,096 preserves the same rows-per-iteration with half the peak batch memory.",
)

notebook["metadata"].setdefault("poker_cfr", {}).update(
    {
        "campaign": "heads_up_compact_v6_hidden384_range005_memorysafe",
        "range_loss_weight": 0.05,
        "batch_size": 4096,
        "range_batch_size": 1024,
        "advantage_steps": 1954,
        "policy_steps": 1954,
        "source_notebook": SOURCE.name,
    }
)
OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print(OUTPUT)
