"""Build the isolated hidden-512 risk-aware HU Deep CFR notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "heads_up_action_conditioned_training.ipynb"
OUTPUT = ROOT / "heads_up_risk_aware_training.ipynb"


def replace_all(text: str) -> str:
    replacements = (
        ("action-conditioned", "risk-aware"),
        ("Action-conditioned", "Risk-aware"),
        ("ACTION_CONDITIONED_ARCHITECTURE", "NETWORK_ARCHITECTURE"),
        ("heads_up_action_conditioned_eval", "heads_up_risk_aware_eval"),
        ("ActionConditionedEvaluationConfig", "RiskAwareEvaluationConfig"),
        ("ActionConditionedProductionCampaign", "RiskAwareProductionCampaign"),
        ("ACTION_EVALUATION_CONFIG", "RISK_EVALUATION_CONFIG"),
        ("action_eval_values", "risk_eval_values"),
        ("plot_action_conditioned_dashboard", "plot_action_conditioned_dashboard"),
        (
            "artifacts/heads_up_action_conditioned_hidden512_v1",
            "artifacts/heads_up_risk_aware_hidden512_v1",
        ),
        (
            "artifacts/heads_up_action_conditioned_validation",
            "artifacts/heads_up_risk_aware_validation",
        ),
        ("setup_vast_action_conditioned.sh", "setup_vast_risk_aware.sh"),
    )
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        text = replace_all("".join(cell.get("source", [])))
        text = text.replace(
            "from heads_up_cfr import HeadsUpNeuralCFR",
            "from heads_up_cfr import HeadsUpNeuralCFR, NETWORK_ARCHITECTURE",
        )
        text = text.replace(
            "from heads_up_models import NETWORK_ARCHITECTURE\n",
            "",
        )
        text = text.replace(
            "    plot_action_conditioned_dashboard,\n",
            "    plot_all_in_spr_trend,\n",
        )
        text = text.replace(
            "'network_architecture': NETWORK_ARCHITECTURE,\n"
            "                'policy_network_architecture': NETWORK_ARCHITECTURE,\n"
            "                'enable_range_training': False,",
            "'network_architecture': NETWORK_ARCHITECTURE,\n"
            "                'policy_network_architecture': NETWORK_ARCHITECTURE,\n"
            "                'enable_range_training': False,\n"
            "                'risk_aware_all_in': True,\n"
            "                'all_in_risk_threshold': 2.0,\n"
            "                'all_in_superiority_margin_bb': 0.25,\n"
            "                'robust_advantage_loss': True,\n"
            "                'fit_reservoir_once_per_iteration': True,",
        )
        text = text.replace(
            "'enable_range_training': False,",
            "'enable_range_training': False,\n"
            "        'risk_aware_all_in': True,\n"
            "        'all_in_risk_threshold': 2.0,\n"
            "        'all_in_superiority_margin_bb': 0.25,\n"
            "        'robust_advantage_loss': True,\n"
            "        'fit_reservoir_once_per_iteration': True,",
        )
        text = text.replace(
            "'test_heads_up_risk_aware_eval.py',",
            "'test_heads_up_action_conditioned_eval.py',\n"
            "                    'test_heads_up_ensemble_profitability.py',\n"
            "                    'test_heads_up_risk_aware_eval.py',",
        )
        text = text.replace(
            "os.getenv('POKER_RUN_TESTS', '1')",
            "os.getenv('POKER_RUN_TESTS', '0')",
        )
        text = text.replace(
            "action_conditioned_dashboard.png", "all_in_spr_trend.png"
        )
        text = text.replace(
            "'reinitialize_advantage_each_iteration': False,",
            "'reinitialize_advantage_each_iteration': True,\n"
            "                'advantage_reinitialize_from_iteration': 25,\n"
            "                'advantage_reinitialize_cycle': 25,",
        )
        text = text.replace(
            "'advantage_reinitialize_cycle': 25",
            "'advantage_reinitialize_cycle': 1",
        )
        text = text.replace(
            "action_figure = plot_action_conditioned_dashboard(frame)\n"
            "            action_path = ARTIFACT_DIR / 'action_conditioned_dashboard.png'\n"
            "            action_figure.savefig(action_path, dpi=150, bbox_inches='tight')\n"
            "            display(action_figure)\n"
            "            plt.close(action_figure)",
            "spr_figure = plot_all_in_spr_trend(frame)\n"
            "            spr_path = ARTIFACT_DIR / 'all_in_spr_trend.png'\n"
            "            spr_figure.savefig(spr_path, dpi=150, bbox_inches='tight')\n"
            "            display(spr_figure)\n"
            "            plt.close(spr_figure)",
        )
        text = text.replace(
            "figure = plot_action_conditioned_dashboard(frame)",
            "figure = plot_all_in_spr_trend(frame)",
        )
        text = text.replace(
            "scores each candidate action directly against cards, pot, SPR and only\n"
            "the current hand's action history.",
            "uses the restored 1,038-feature state, including pot-after-call,\n"
            "effective-stack SPR, action payment and resulting-SPR descriptors.\n"
            "Traversal keeps all ten actions legal, but marginal positive regret\n"
            "for high-risk shoves is reduced when smaller bets have comparable value.",
        )
        cell["source"] = text.splitlines(keepends=True)
    OUTPUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
