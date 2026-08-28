"""Build the fresh hidden-512 action-conditioned HU Jupyter notebook."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "heads_up_action_conditioned_training.ipynb"


def source(text: str) -> list[str]:
    return dedent(text).lstrip("\n").splitlines(keepends=True)


def markdown(text: str, cell_id: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": source(text),
    }


def code(text: str, cell_id: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": source(text),
    }


cells = [
    markdown(
        """
        # Fresh hidden-512 action-conditioned heads-up Deep CFR

        This is a new isolated campaign. It never resumes the hidden-128,
        hidden-384, paper3x, policy-1025, or 725/950/1025 reservoirs. Policy
        1025 and the three-policy top-three ensemble are loaded read-only as
        frozen evaluation opponents.

        The poker engine, ten legal action slots, recursive CFR values, raw
        regret equation, and regret matching are unchanged. The new network
        scores each candidate action directly against cards, pot, SPR and only
        the current hand's action history. There is no opponent-hand range head,
        range reservoir, range loss, or cross-hand opponent memory.
        """,
        "overview",
    ),
    markdown(
        """
        ## 1. Imports and reproducible environment

        Keep this notebook in the package root. On Vast.ai, run
        `bash setup_vast_action_conditioned.sh` once before executing cells.
        """,
        "imports-heading",
    ),
    code(
        r"""
        %matplotlib inline
        import json
        import math
        import os
        import subprocess
        import sys
        from pathlib import Path

        import matplotlib.pyplot as plt
        import pandas as pd
        import torch
        from IPython.display import clear_output, display

        from heads_up_native import HeadsUpHoldemEngine
        from heads_up_cfr import HeadsUpNeuralCFR
        from heads_up_models import ACTION_CONDITIONED_ARCHITECTURE
        from heads_up_production import (
            CampaignConfig,
            load_or_create_trainer,
            save_policy_snapshot,
        )
        from heads_up_reporting import (
            metrics_frame,
            plot_training_dashboard,
            save_training_dashboard,
        )
        from heads_up_action_conditioned_eval import (
            ActionConditionedEvaluationConfig,
            ActionConditionedProductionCampaign,
            plot_action_conditioned_dashboard,
        )

        pd.set_option('display.max_columns', 120)
        pd.set_option('display.width', 200)
        """,
        "imports",
    ),
    markdown(
        """
        ## 2. Focused correctness tests

        These tests verify engine parity, direct state-action descriptor
        dependence, absence of the range head and reservoirs, checkpoint
        isolation, and the frozen evaluation harness.
        """,
        "tests-heading",
    ),
    code(
        r"""
        RUN_SELF_TESTS = os.getenv('POKER_RUN_TESTS', '1') == '1'
        if RUN_SELF_TESTS:
            subprocess.run(
                [
                    sys.executable, '-m', 'unittest', '-q',
                    'test_heads_up_engine.py',
                    'test_heads_up_models.py',
                    'test_heads_up_native_engine.py',
                    'test_heads_up_training.py',
                    'test_heads_up_action_conditioned_eval.py',
                ],
                check=True,
            )
        """,
        "tests",
    ),
    markdown(
        """
        ## 3. Hardware and worker count

        The worker rule is intentionally identical to the preceding production
        notebook: `min(16, effective_cpu_count())`, overridable only through the
        same `POKER_CPU_WORKERS` environment variable.
        """,
        "hardware-heading",
    ),
    code(
        r"""
        if not torch.cuda.is_available():
            raise RuntimeError('A CUDA GPU is required for the production profile.')
        DEVICE = torch.device('cuda:0')
        torch.set_float32_matmul_precision('high')

        def effective_cpu_count():
            counts = [os.cpu_count() or 1]
            try:
                topology = subprocess.check_output(
                    ['lscpu', '-p=CORE,SOCKET'], text=True,
                )
                physical = {
                    line for line in topology.splitlines()
                    if line and not line.startswith('#')
                }
                if physical:
                    counts.append(len(physical))
            except (OSError, subprocess.SubprocessError):
                pass
            if hasattr(os, 'sched_getaffinity'):
                counts.append(len(os.sched_getaffinity(0)))
            try:
                quota, period = Path('/sys/fs/cgroup/cpu.max').read_text().split()
                if quota != 'max':
                    counts.append(max(1, int(quota) // int(period)))
            except (OSError, ValueError):
                pass
            return max(1, min(counts))

        DEFAULT_CPU_WORKERS = min(16, effective_cpu_count())
        CPU_WORKERS = int(os.getenv('POKER_CPU_WORKERS', str(DEFAULT_CPU_WORKERS)))
        gpu = torch.cuda.get_device_properties(DEVICE)
        display(pd.Series({
            'device': str(DEVICE),
            'GPU': gpu.name,
            'VRAM GiB': gpu.total_memory / 1024**3,
            'native engine': HeadsUpHoldemEngine.native_backend,
            'CPU traversal workers': CPU_WORKERS,
        }, name='hardware'))
        """,
        "hardware",
    ),
    markdown(
        """
        ## 4. Fresh campaign configuration

        `artifacts/heads_up_action_conditioned_hidden512_v1` is a new rollback
        boundary. Rerunning this notebook may resume only checkpoints created in
        that directory; it cannot load any older campaign.
        """,
        "config-heading",
    ),
    code(
        r"""
        PROFILE = os.getenv('POKER_TRAINING_PROFILE', 'production').strip().lower()
        ARTIFACT_DIR = Path(
            'artifacts/heads_up_action_conditioned_hidden512_v1'
            if PROFILE == 'production'
            else 'artifacts/heads_up_action_conditioned_validation'
        )
        ENVIRONMENT_CONFIG = {
            'starting_stack': 200, 'small_blind': 1, 'big_blind': 2, 'seed': 442,
        }
        if PROFILE == 'production':
            TRAINER_CONFIG = {
                'hidden': 512,
                'blocks': 2,
                'learning_rate': 1e-3,
                'advantage_capacity': 8_000_000,
                'policy_capacity': 8_000_000,
                'range_capacity': 0,
                'max_nodes_per_traversal': 5_000,
                'max_depth': 32,
                'exploration': 0.15,
                'range_loss_weight': 0.0,
                'reservoir_turnover_fraction': 0.18,
                'reinitialize_advantage_each_iteration': True,
                'advantage_reinitialize_from_iteration': 25,
                'advantage_reinitialize_cycle': 25,
                'network_architecture': ACTION_CONDITIONED_ARCHITECTURE,
                'policy_network_architecture': ACTION_CONDITIONED_ARCHITECTURE,
                'enable_range_training': False,
                'seed': 442,
            }
            campaign_values = {
                'target_iteration': 10_000,
                'traversals_per_player': 1_024,
                'advantage_steps': 977,
                'policy_steps': 977,
                'batch_size': 8_192,
                'evaluate_every': 25,
                'checkpoint_every': 25,
                'snapshot_every': 100,
                'evaluation_games_per_player': 10_000,
                'range_evaluation_hands_per_opponent': 1,
                'range_evaluation_batch_size': 4_096,
                'range_training_hands_per_iteration': 1,
                'range_batch_size': 1,
                'league_games_per_player': 99,
                'validation_seed': 402_700,
                'opponent_profiles': ('random', 'calling_station', 'tight_aggressive'),
                'reference_policy_path': None,
                'league_opponents': 3,
                'keep_full_checkpoints': 1,
            }
            action_eval_values = {
                'reciprocal_hands': 20_000,
                'inference_batch_size': 2_048,
                'simulation_batch_size': 20_000,
                'same_state_hands': 2_000,
            }
        elif PROFILE == 'validation':
            TRAINER_CONFIG = {
                'hidden': 32, 'blocks': 1, 'learning_rate': 1e-3,
                'advantage_capacity': 2_000, 'policy_capacity': 2_000,
                'range_capacity': 0, 'max_nodes_per_traversal': 80,
                'max_depth': 24, 'exploration': 0.0,
                'range_loss_weight': 0.0,
                'reservoir_turnover_fraction': 0.18,
                'reinitialize_advantage_each_iteration': False,
                'network_architecture': ACTION_CONDITIONED_ARCHITECTURE,
                'policy_network_architecture': ACTION_CONDITIONED_ARCHITECTURE,
                'enable_range_training': False, 'seed': 542,
            }
            campaign_values = {
                'target_iteration': 2, 'traversals_per_player': 1,
                'advantage_steps': 2, 'policy_steps': 2, 'batch_size': 32,
                'evaluate_every': 1, 'checkpoint_every': 1, 'snapshot_every': 1,
                'evaluation_games_per_player': 4,
                'range_evaluation_hands_per_opponent': 1,
                'range_evaluation_batch_size': 32,
                'range_training_hands_per_iteration': 1, 'range_batch_size': 1,
                'league_games_per_player': 2, 'validation_seed': 502_700,
                'opponent_profiles': ('random', 'calling_station', 'tight_aggressive'),
                'reference_policy_path': None, 'league_opponents': 1,
                'keep_full_checkpoints': 1,
            }
            action_eval_values = {
                'reciprocal_hands': 20, 'inference_batch_size': 64,
                'simulation_batch_size': 20, 'same_state_hands': 10,
            }
        else:
            raise ValueError('POKER_TRAINING_PROFILE must be production or validation')

        campaign_values['traversal_workers'] = CPU_WORKERS
        if os.getenv('POKER_TARGET_ITERATION'):
            campaign_values['target_iteration'] = int(os.environ['POKER_TARGET_ITERATION'])
        CAMPAIGN_CONFIG = CampaignConfig(**campaign_values)

        POLICY_1025 = Path('reference_policies/policy_00001025.pt')
        ENSEMBLE_PATHS = tuple(
            Path(f'reference_policies/policy_{iteration:08d}.pt')
            for iteration in (725, 950, 1025)
        )
        REFERENCE_LBR = (
            ARTIFACT_DIR / 'evaluations' / 'reference_exploitability'
            / 'policy_00001025_lbr.json'
        )
        ACTION_EVALUATION_CONFIG = ActionConditionedEvaluationConfig(
            policy_1025_path=str(POLICY_1025),
            ensemble_policy_paths=tuple(map(str, ENSEMBLE_PATHS)),
            reference_exploitability_path=str(REFERENCE_LBR),
            **action_eval_values,
        )
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        display(pd.Series({
            'profile': PROFILE,
            'artifact directory': str(ARTIFACT_DIR),
            'hidden width': TRAINER_CONFIG['hidden'],
            'architecture': ACTION_CONDITIONED_ARCHITECTURE,
            'range training': TRAINER_CONFIG['enable_range_training'],
            'traversal workers': CAMPAIGN_CONFIG.traversal_workers,
            'evaluate every': CAMPAIGN_CONFIG.evaluate_every,
            'reciprocal hands/evaluation': ACTION_EVALUATION_CONFIG.reciprocal_hands,
        }, name='configuration'))
        """,
        "config",
    ),
    markdown(
        """
        ## 5. Create or resume only this new campaign

        The first run is random initialization. Subsequent runs resume only this
        hidden-512 campaign so interrupted Vast.ai training remains recoverable.
        """,
        "create-heading",
    ),
    code(
        r"""
        env = HeadsUpHoldemEngine(**ENVIRONMENT_CONFIG)
        trainer, resumed = load_or_create_trainer(
            env,
            ARTIFACT_DIR,
            device=DEVICE,
            trainer_kwargs=TRAINER_CONFIG,
        )
        if trainer.network_architecture != ACTION_CONDITIONED_ARCHITECTURE:
            raise RuntimeError('Artifact directory contains another architecture.')
        if trainer.range_training_enabled or trainer.range_buffers:
            raise RuntimeError('Fresh campaign unexpectedly contains range training state.')
        campaign = ActionConditionedProductionCampaign(
            trainer,
            ARTIFACT_DIR,
            CAMPAIGN_CONFIG,
            ACTION_EVALUATION_CONFIG,
        )
        parameter_counts = {
            'advantage network': sum(p.numel() for p in trainer.advantage_nets[0].parameters()),
            'policy network': sum(p.numel() for p in trainer.policy_nets[0].parameters()),
        }
        parameter_counts['four-network total'] = 2 * sum(parameter_counts.values())
        display(pd.Series({
            'resumed new campaign': resumed,
            'iteration': trainer.iteration,
            'input width': trainer.input_dim,
            **parameter_counts,
            'range buffers': len(trainer.range_buffers),
        }, name='active trainer'))
        """,
        "create",
    ),
    markdown(
        """
        ## 6. Live monitoring and promotion evidence

        Every iteration prints throughput. Every 25 production iterations the
        notebook compares the candidate and frozen policies on identical states,
        runs reciprocal matches, plots 99% confidence intervals, and displays
        all four promotion gates. Exploitability remains pending until the
        deliberately expensive learned-best-response cell is run.
        """,
        "monitor-heading",
    ),
    code(
        r"""
        def on_iteration(row):
            print(
                f"iteration {int(row['iteration']):5d} | "
                f"{row['seconds']:8.1f}s | "
                f"{row['traversal_nodes_per_second']:9.0f} nodes/s | "
                f"adv/policy buffers "
                f"{int(row['adv_buffer_p0']) + int(row['adv_buffer_p1']):,}/"
                f"{int(row['policy_buffer_p0']) + int(row['policy_buffer_p1']):,} | "
                f"workers {int(row['traversal_workers'])}"
            )

        def on_evaluation(frame):
            clear_output(wait=True)
            on_iteration(frame.iloc[-1])
            training_figure = plot_training_dashboard(frame)
            display(training_figure)
            plt.close(training_figure)
            action_figure = plot_action_conditioned_dashboard(frame)
            action_path = ARTIFACT_DIR / 'action_conditioned_dashboard.png'
            action_figure.savefig(action_path, dpi=150, bbox_inches='tight')
            display(action_figure)
            plt.close(action_figure)
            evaluated = frame.dropna(
                subset=['same_state_candidate_all_in_probability']
            )
            display(evaluated[[
                'iteration',
                'same_state_candidate_all_in_probability',
                'same_state_policy1025_all_in_probability',
                'same_state_ensemble_top3_all_in_probability',
                'promotion_policy1025_ev_bb_per_100',
                'promotion_policy1025_ci99_low_bb_per_100',
                'promotion_ensemble_top3_ev_bb_per_100',
                'promotion_ensemble_top3_ci99_low_bb_per_100',
                'promotion_policy1025_non_all_in_net_bb_per_100_all_hands',
                'promotion_ensemble_top3_non_all_in_net_bb_per_100_all_hands',
                'promotion_gate_all_in_controlled',
                'promotion_gate_positive_overall_ev_99',
                'promotion_gate_positive_non_all_in_ev',
                'promotion_gate_no_major_exploitability',
                'promoted_to_champion',
            ]].tail(10))
        """,
        "monitor",
    ),
    markdown(
        """
        ## 7. Run the campaign

        Interrupting this cell writes an emergency full checkpoint. Rerun the
        notebook to resume this new campaign, never an older one.
        """,
        "run-heading",
    ),
    code(
        r"""
        metrics = campaign.run(
            on_iteration=on_iteration,
            on_evaluation=on_evaluation,
        )
        """,
        "run",
    ),
    markdown(
        """
        ## 8. Reload saved graphs and evaluation rows
        """,
        "results-heading",
    ),
    code(
        r"""
        frame = metrics_frame(trainer.metrics)
        display(frame.tail(20))
        if 'same_state_candidate_all_in_probability' in frame:
            figure = plot_action_conditioned_dashboard(frame)
            display(figure)
            plt.close(figure)
        """,
        "results",
    ),
    markdown(
        """
        ## 9. Final learned-best-response gate

        This cell is intentionally manual and expensive. Run it only for a
        frozen milestone that has already passed the EV, non-all-in and shove
        gates. It evaluates the candidate and policy 1025 with the same response
        learner configuration. Automatic promotion remains disabled until this
        evidence exists and the periodic evaluation is rerun for the same
        iteration.
        """,
        "exploit-heading",
    ),
    code(
        r"""
        RUN_FINAL_EXPLOITABILITY = False
        if RUN_FINAL_EXPLOITABILITY:
            iteration = trainer.iteration
            candidate_snapshot = (
                ARTIFACT_DIR / 'snapshots' / f'policy_{iteration:08d}.pt'
            )
            save_policy_snapshot(
                trainer,
                candidate_snapshot,
                metadata={'purpose': 'promotion_exploitability_candidate'},
            )
            iteration_dir = (
                ARTIFACT_DIR / 'evaluations' / f'step_{iteration:08d}'
            )
            candidate_output = iteration_dir / 'exploitability'
            reference_output = REFERENCE_LBR.parent
            common = [
                '--device', str(DEVICE),
                '--iterations', '12', '--traversals', '256',
                '--frontier-batch-size', '64', '--advantage-steps', '64',
                '--fit-batch-size', '512', '--reservoir-capacity', '100000',
                '--max-nodes', '512', '--max-depth', '32',
                '--validate-every', '2', '--validation-hands', '4000',
                '--final-hands', '100000', '--inference-batch-size', '2048',
                '--response-hidden', '512',
                '--response-architecture', ACTION_CONDITIONED_ARCHITECTURE,
            ]
            if not REFERENCE_LBR.exists():
                subprocess.run(
                    [sys.executable, 'evaluate_heads_up_exploitability.py',
                     '--policies', str(POLICY_1025),
                     '--output-dir', str(reference_output), *common],
                    check=True,
                )
            subprocess.run(
                [sys.executable, 'evaluate_heads_up_exploitability.py',
                 '--policies', str(candidate_snapshot),
                 '--output-dir', str(candidate_output), *common],
                check=True,
            )
            # Rerun the same frozen iteration's complete evaluation. Promotion
            # can occur only if every relative gate now passes.
            row = trainer.metrics[-1]
            campaign._evaluate(row)
            campaign._append_metric(row)
            on_evaluation(pd.DataFrame(trainer.metrics))
        """,
        "exploit",
    ),
    markdown(
        """
        ## 10. Vast.ai handoff

        Before destroying the instance, copy the complete fresh artifact
        directory back to durable storage. Full checkpoints contain the only
        resumable reservoirs; policy snapshots are inference/evaluation files.
        """,
        "handoff",
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(OUTPUT)
