"""Build the isolated hidden-384 compact-encoder notebook.

The established research notebook is treated as an immutable template.  The
generated notebook retains all 36 cells, dashboards, range plots, scripted
opponents and BB confidence intervals, while selecting the compact ABI and a
fresh artifact directory.  It additionally evaluates the live compact policy
against the frozen 725/950/1025 top-three ensemble using each side's own
encoder.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "heads_up_training.ipynb"
OUTPUT = ROOT / "heads_up_compact_training.ipynb"

notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
cells = notebook["cells"]

cells[0]["source"] = """# Compact hidden-384 heads-up Deep CFR production training

This is a fresh, isolated campaign using the lossless-through-100BB compact
encoder and structured hidden-384 network. It preserves the established
training monitor, random/calling-station/TAG evaluations, BB confidence
intervals, opponent-range diagnostics, and every strategy graph. The original
`heads_up_training.ipynb` is not modified.
""".splitlines(keepends=True)


def replace(cell: int, old: str, new: str) -> None:
    text = "".join(cells[cell]["source"])
    if old not in text:
        raise RuntimeError(f"notebook template text was not found in cell {cell}: {old!r}")
    cells[cell]["source"] = text.replace(old, new).splitlines(keepends=True)


replace(
    2,
    "import math\n",
    "import json\nimport math\n",
)
replace(
    2,
    "from heads_up_native import HeadsUpHoldemEngine\n",
    """from heads_up_native import HeadsUpHoldemEngine
from heads_up_compact import (
    COMPACT_DEFAULT_MAX_HISTORY,
    COMPACT_ENCODER_SCHEMA_VERSION,
)
from heads_up_models import (
    COMPACT_V6_ARCHITECTURE,
    COMPACT_V6_POLICY_RANGE_ARCHITECTURE,
)
from evaluate_heads_up_ensemble_profitability import (
    TrainerProvider,
    build_suite,
    run_reciprocal_match,
)
""",
)
replace(
    4,
    "'test_heads_up_training.py', 'test_heads_up_analysis.py'",
    """'test_heads_up_training.py', 'test_heads_up_analysis.py',
         'test_heads_up_compact.py',
         'test_heads_up_ensemble_profitability.py'""",
)
replace(7, "unchanged 1,038-feature HU encoder", "782-value compact HU buffer")
replace(7, "feeds 1,842,512-parameter advantage networks and 2,353,022-parameter\ndual-head policy networks", "feeds structured hidden-384 advantage and exact-range policy networks")
replace(8, "Path('artifacts/heads_up_v4_hidden384')", "Path('artifacts/heads_up_compact_v6_hidden384')")
replace(
    8,
    """'reference_policy_path': os.getenv(
            'POKER_REFERENCE_POLICY',
            'reference_policies/policy_00001025.pt',
        ),""",
    "'reference_policy_path': None,",
)
replace(
    8,
    """'reference_policy_path': os.getenv(
                'POKER_REFERENCE_POLICY',
                'reference_policies/policy_00001025.pt',
            ),""",
    "'reference_policy_path': None,",
)
trainer_marker = """'hidden': 384, 'blocks': 2, 'learning_rate': 1e-3,"""
replace(
    8,
    trainer_marker,
    trainer_marker + """
            'max_history': COMPACT_DEFAULT_MAX_HISTORY,
            'encoder_schema_version': COMPACT_ENCODER_SCHEMA_VERSION,
            'network_architecture': COMPACT_V6_ARCHITECTURE,
            'policy_network_architecture': COMPACT_V6_POLICY_RANGE_ARCHITECTURE,""",
)
validation_marker = """'hidden': 16, 'blocks': 1, 'learning_rate': 1e-3,"""
replace(
    8,
    validation_marker,
    validation_marker + """
            'max_history': COMPACT_DEFAULT_MAX_HISTORY,
            'encoder_schema_version': COMPACT_ENCODER_SCHEMA_VERSION,
            'network_architecture': COMPACT_V6_ARCHITECTURE,
            'policy_network_architecture': COMPACT_V6_POLICY_RANGE_ARCHITECTURE,""",
)
replace(
    8,
    "RANGE_PLOTS_EVERY_EVALS = int(os.getenv('POKER_RANGE_EVERY_EVALS', '1'))",
    """RANGE_PLOTS_EVERY_EVALS = int(os.getenv('POKER_RANGE_EVERY_EVALS', '1'))
ENSEMBLE_EVAL_HANDS = int(os.getenv('POKER_ENSEMBLE_EVAL_HANDS', '10000'))
LEGACY_ENSEMBLE_PATHS = tuple(Path(path) for path in (
    'reference_policies/policy_00000725.pt',
    'reference_policies/policy_00000950.pt',
    'reference_policies/policy_00001025.pt',
))
REFERENCE_TOP3_TOP4_RESULT = Path(
    'artifacts/heads_up_v4_paper3x/evaluations/'
    'ensemble_725_950_1025_top3_vs_top4_100000.json'
)""",
)
replace(
    10,
    "campaign = ProductionCampaign(trainer, ARTIFACT_DIR, CAMPAIGN_CONFIG)\n",
    """campaign = ProductionCampaign(trainer, ARTIFACT_DIR, CAMPAIGN_CONFIG)
legacy_top3, legacy_full, legacy_components, legacy_environment, legacy_snapshots = build_suite(
    LEGACY_ENSEMBLE_PATHS, device=str(DEVICE), top_k=3,
)
live_compact_provider = TrainerProvider(
    trainer, name='live_compact_hidden384',
)
if REFERENCE_TOP3_TOP4_RESULT.is_file():
    frozen_match = json.loads(
        REFERENCE_TOP3_TOP4_RESULT.read_text(encoding='utf-8')
    )['matches'][0]
    frozen_ci = frozen_match['confidence_intervals']['99']
    display(pd.Series({
        'candidate': frozen_match['candidate'],
        'opponent': frozen_match['opponent'],
        'hands': frozen_match['hands'],
        'EV BB/100': frozen_match['mean_ev_bb_per_100'],
        'paired standard error BB/100': 100 * frozen_match['paired_stderr_bb_per_hand'],
        '99% low BB/100': frozen_ci['low_bb_per_100'],
        '99% high BB/100': frozen_ci['high_bb_per_100'],
        '99% verdict': frozen_match['verdict_at_99_percent'],
    }, name='frozen top-3 versus top-4 ensemble result'))
""",
)
replace(10, "'information-state features': trainer.input_dim,", """'information-state features': trainer.input_dim,
    'logical compact length': '40 + 7 x public history events',
    'full-history capacity': trainer.max_history,
    'encoder schema': trainer.encoder_schema_version,""")

ensemble_evaluation = r'''
    ensemble_result = run_reciprocal_match(
        live_compact_provider,
        lambda evaluation_env: legacy_top3,
        environment=legacy_environment,
        hands=ENSEMBLE_EVAL_HANDS,
        seed=CAMPAIGN_CONFIG.validation_seed + 1000003 * trainer.iteration,
        inference_batch_size=2048,
        simulation_batch_size=min(20000, ENSEMBLE_EVAL_HANDS // 2),
    )
    interval = ensemble_result['confidence_intervals']['95']
    all_in_bet_ratio = ensemble_result['candidate_all_in_bet_to_pot_ratio']
    all_in_raise_ratio = ensemble_result[
        'candidate_all_in_raise_over_pot_after_call'
    ]
    display(pd.Series({
        'opponent': ensemble_result['opponent'],
        'hands': ensemble_result['hands'],
        'EV BB/hand': ensemble_result['mean_ev_bb_per_hand'],
        'EV BB/100': ensemble_result['mean_ev_bb_per_100'],
        'paired standard error BB/hand': ensemble_result['paired_stderr_bb_per_hand'],
        '95% low BB/100': interval['low_bb_per_100'],
        '95% high BB/100': interval['high_bb_per_100'],
        'candidate all-in hand rate': ensemble_result['candidate_all_in_hand_rate'],
        'all-in bet/pot mean': all_in_bet_ratio['mean'],
        'all-in bet/pot median': all_in_bet_ratio['median'],
        'all-in bet/pot p90': all_in_bet_ratio['p90'],
        'all-in raise/pot-after-call mean': all_in_raise_ratio['mean'],
        'all-in raise/pot-after-call median': all_in_raise_ratio['median'],
        'all-in raise/pot-after-call p90': all_in_raise_ratio['p90'],
    }, name='live compact policy vs frozen 725/950/1025 top-3 ensemble'))
    ensemble_path = (
        ARTIFACT_DIR / 'evaluations'
        / f'ensemble_top3_step_{trainer.iteration:08d}.json'
    )
    ensemble_path.write_text(
        json.dumps(ensemble_result, indent=2) + '\n', encoding='utf-8'
    )
    print(f'Ensemble evaluation saved to {ensemble_path.resolve()}')
'''
replace(
    14,
    "    if evaluation_number % RANGE_PLOTS_EVERY_EVALS == 0:\n",
    ensemble_evaluation + "\n    if evaluation_number % RANGE_PLOTS_EVERY_EVALS == 0:\n",
)

final_ensemble = r'''
    FINAL_ENSEMBLE_HANDS = int(os.getenv('POKER_FINAL_ENSEMBLE_HANDS', '100000'))
    final_ensemble = run_reciprocal_match(
        live_compact_provider,
        lambda evaluation_env: legacy_top3,
        environment=legacy_environment,
        hands=FINAL_ENSEMBLE_HANDS,
        seed=FINAL_TEST_SEED + 17,
        inference_batch_size=2048,
        simulation_batch_size=min(20000, FINAL_ENSEMBLE_HANDS // 2),
    )
    display(pd.Series(final_ensemble, name='final compact vs top-3 ensemble'))
'''
replace(
    19,
    "    display(pd.Series(final_suite, name='final untouched holdout'))\n",
    "    display(pd.Series(final_suite, name='final untouched holdout'))\n" + final_ensemble,
)
replace(11, "the 1,038-feature input", "the 782-value compact input")
replace(
    34,
    "The full checkpoint is for continuation and includes large reservoirs.",
    "The full checkpoint is for continuation and includes large reservoirs. This compact campaign is a separate rollback boundary and never resumes legacy reservoirs.",
)
replace(
    35,
    "'frozen previous policy': (",
    "'frozen legacy ensemble directory': Path('reference_policies'),\n    'frozen previous policy': (",
)

notebook["metadata"].setdefault("poker_cfr", {}).update(
    {
        "campaign": "heads_up_compact_v6_hidden384",
        "encoder_schema_version": "hu_compact_information_state_v1_full_history",
        "input_dim": 782,
        "hidden": 384,
        "source_notebook": SOURCE.name,
    }
)
OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print(OUTPUT)
