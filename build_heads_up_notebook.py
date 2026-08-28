"""Build the heads-up notebook from the established 36-cell research layout."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT / "three_player_training.ipynb"
OUTPUTS = (
    ROOT / "heads_up_training.ipynb",
    ROOT / "vast_heads_up_training" / "heads_up_training.ipynb",
)


def lines(value: str) -> list[str]:
    return dedent(value).lstrip("\n").splitlines(keepends=True)


notebook = json.loads(TEMPLATE.read_text(encoding="utf-8"))
for cell in notebook["cells"]:
    if cell["cell_type"] == "markdown":
        text = "".join(cell["source"])
        replacements = (
            ("Three-player Hold'em", "Heads-up Hold'em"),
            ("three-player", "heads-up"),
            ("three player", "heads-up"),
            ("all six networks", "all four networks"),
            ("six-network", "four-network"),
            ("six networks", "four networks"),
            ("three policy networks", "two policy networks"),
            ("three independently trained policy networks", "two independently trained policy networks"),
            ("BTN/SB/BB", "BTN/SB and BB"),
            ("three player networks", "two seat networks"),
            ("all 1,326 physical preflop combinations", "all 1,326 physical preflop combinations"),
        )
        for old, new in replacements:
            text = text.replace(old, new)
        cell["source"] = text.splitlines(keepends=True)

markdown_overrides = {
    0: """# Heads-up Hold'em production training and strategy laboratory

This notebook is the heads-up counterpart of the established three-player
control plane. It trains two advantage and two average-policy networks,
with each policy network also predicting a blocker-masked distribution over
all 1,326 exact opponent-card combinations from the same no-leakage information
set. The policy action loss and 0.01-weighted range loss share only the policy
trunk; advantage/regret networks remain unchanged.
prints every completed iteration, evaluates fixed opponents, checkpoints
resumably, and produces exact 169-hand policy charts for named critical states.
""",
    1: """## What this notebook can—and cannot—prove

The notebook measures optimization health and held-out play against reproducible
scripted opponents and a fixed opponent-range holdout. It does not prove equilibrium convergence or human-beating
strength; those require independent exploitability and untouched match tests.
""",
    3: """## 1. Validate the engine, trainer, production workflow, and range analysis

These tests cover the exact integer-chip engine, native/Python parity, encoder
schema, parallel traversal, checkpoint continuation, and legal critical-state
range construction.
""",
    7: """## 3. Persisted production configuration

The production profile preserves the requested 10,000-iteration Deep CFR
        campaign: 1,024 traversals per player, compact advantage/policy fitting, three-million
entry reservoirs, and cgroup-aware CPU workers capped at 48. The unchanged 1,038-feature HU encoder
feeds 1,842,512-parameter advantage networks and 2,353,022-parameter
dual-head policy networks at hidden 384. The four networks total 8,391,068
parameters; the live phase timings
below make that cost explicit.
""",
    9: """## 4. Create or resume the full four-network trainer

The full checkpoint contains all four networks, optimizers, packed reservoirs,
RNG state, position cycle, and metrics. The initial policy is frozen separately
for range-change and held-out baseline comparisons.
""",
    11: """### 4A. Actual four-network architecture

The diagram is read from the active trainer and shows both advantage networks,
both deployable policy networks, the 1,038-feature input, structured card,
action and ordered-history branches, and the measured parameter count.
""",
    13: """## 5. Live monitoring every training iteration

Every completed iteration prints progress, ETA, traversal/fitting phase time,
node throughput, reservoir sizes, and CUDA memory. At evaluation boundaries the
notebook redraws opponent tables, the nine-panel dashboard, and exact critical-
state preflop/postflop policy maps.
""",
    15: """## 6. Run the resumable production campaign

Interrupting this cell writes an emergency full checkpoint. Rerunning the
notebook resumes the saved iteration and reservoirs.
""",
    20: """## 8. Exact preflop call-frequency maps across 1,326 combinations

Each 13×13 cell averages all physical suit combinations and both role-equivalent
seat networks. These are policy frequencies, not hand equity.
""",
    24: """## 10. How the range changed from the frozen initial policy

Positive movement is descriptive rather than proof of improvement; held-out EV
and its confidence interval remain the strength signal.
""",
    30: """## 13. Detect disagreement between the two seat networks

Large disagreement for the same role-relative critical state can reveal
seat-specific under-training or representation artifacts.
""",
    34: """## 15. Deployment and remaining engineering boundary

The full checkpoint is for continuation and includes large reservoirs. Policy
snapshots are smaller inference artifacts. Copy artifacts off this nonpersistent
instance before destroy/recycle.
""",
}
for index, value in markdown_overrides.items():
    notebook["cells"][index]["source"] = lines(value)

code = {
    2: r"""
        %matplotlib inline
        from pathlib import Path
        import math
        import os
        import subprocess
        import sys

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import torch
        from IPython.display import Image, clear_output, display

        from heads_up_native import HeadsUpHoldemEngine
        from heads_up_production import (
            CampaignConfig,
            ProductionCampaign,
            evaluate_benchmark_suite,
            load_or_create_trainer,
            load_policy_snapshot,
        )
        from heads_up_reporting import (
            metrics_frame,
            plot_training_dashboard,
            plot_range_dashboard,
            save_range_dashboard,
            save_training_dashboard,
        )
        from heads_up_analysis import (
            StrategyAnalyzer,
            compare_ranges,
            plot_call_maps,
            plot_card_sweep,
            plot_network_architecture,
            plot_range_delta,
            plot_range_heatmaps,
            postflop_scenarios,
            preflop_scenarios,
        )

        pd.set_option('display.max_columns', 100)
        pd.set_option('display.width', 180)
    """,
    4: r"""
        RUN_SELF_TESTS = os.getenv('POKER_RUN_TESTS', '1') == '1'
        if RUN_SELF_TESTS:
            subprocess.run(
                [sys.executable, '-m', 'unittest', '-q',
                 'test_heads_up_engine.py', 'test_heads_up_models.py',
                 'test_heads_up_native_engine.py', 'test_heads_up_search.py',
                 'test_heads_up_training.py', 'test_heads_up_analysis.py'],
                check=True,
            )
            print('All engine, CUDA, production, and analysis tests passed.')
        else:
            print('Self-tests skipped by POKER_RUN_TESTS=0.')
    """,
    6: r"""
        if not HeadsUpHoldemEngine.native_backend:
            raise RuntimeError('The compiled heads-up engine is required. Run bash setup_vast.sh.')
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required for the production profile.')
        DEVICE = torch.device('cuda:0')
        torch.set_float32_matmul_precision('high')
        gpu = torch.cuda.get_device_properties(DEVICE)
        display(pd.Series({
            'PyTorch': torch.__version__,
            'device': str(DEVICE),
            'GPU': gpu.name,
            'VRAM GiB': gpu.total_memory / 1024**3,
            'CUDA capability': f'{gpu.major}.{gpu.minor}',
            'native engine': HeadsUpHoldemEngine.native_backend,
        }, name='hardware'))
    """,
    8: r"""
        def effective_cpu_count():
            counts = [os.cpu_count() or 1]
            try:
                topology = subprocess.check_output(
                    ['lscpu', '-p=CORE,SOCKET'], text=True,
                )
                physical_cores = {
                    line for line in topology.splitlines()
                    if line and not line.startswith('#')
                }
                if physical_cores:
                    counts.append(len(physical_cores))
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

        PROFILE = os.getenv('POKER_TRAINING_PROFILE', 'production').strip().lower()
        # This campaign targets the Ryzen 9 7950X instance. Use its 16
        # physical cores for traversal; workers exit before the sequential
        # merge/GPU-fitting phases, and the 30.72-core cgroup quota leaves
        # scheduling headroom for Jupyter and coordination.
        DEFAULT_CPU_WORKERS = min(16, effective_cpu_count())
        CPU_WORKERS = int(os.getenv('POKER_CPU_WORKERS', str(DEFAULT_CPU_WORKERS)))
        PROFILES = {
            'production': {
                # Hidden-384 is a fresh campaign. Never point this at the
                # hidden-128 artifact directory because checkpoints/reservoirs
                # are intentionally architecture-locked.
                'artifact_dir': Path('artifacts/heads_up_v4_hidden384'),
                'reference_policy_path': os.getenv(
                    'POKER_REFERENCE_POLICY',
                    'reference_policies/policy_00001025.pt',
                ),
                'environment': {
                    'starting_stack': 200, 'small_blind': 1,
                    'big_blind': 2, 'seed': 442,
                },
                'trainer': {
                    'hidden': 384, 'blocks': 2, 'learning_rate': 1e-3,
                    'advantage_capacity': 8_000_000,
                    'policy_capacity': 8_000_000,
                    'range_capacity': 500_000,
                    'max_nodes_per_traversal': 5_000, 'max_depth': 32,
                    'exploration': 0.15,
                    'range_loss_weight': 0.01,
                    'reservoir_turnover_fraction': 0.18,
                    'reinitialize_advantage_each_iteration': True,
                    'advantage_reinitialize_from_iteration': 25,
                    'advantage_reinitialize_cycle': 25,
                    'seed': 442,
                },
                'campaign': {
                    'target_iteration': 10_000,
                    'traversals_per_player': 1_024,
                    # 977 x 8,192 covers every row of an 8M reservoir once.
                    'advantage_steps': 977, 'policy_steps': 977,
                    'batch_size': 8_192, 'evaluate_every': 25,
                    'checkpoint_every': 25, 'snapshot_every': 100,
                    'evaluation_games_per_player': 10_000,
                    'range_evaluation_hands_per_opponent': 2_500,
                    'range_evaluation_batch_size': 4_096,
                    'range_training_hands_per_iteration': 2_048,
                    'range_batch_size': 2_048,
                    'league_games_per_player': 99,
                    'validation_seed': 402_700,
                    'opponent_profiles': (
                        'random', 'calling_station', 'tight_aggressive',
                    ),
                    'reference_policy_path': os.getenv(
                        'POKER_REFERENCE_POLICY',
                        'reference_policies/policy_00001025.pt',
                    ),
                    'league_opponents': 3,
                    'keep_full_checkpoints': 1,
                },
            },
            'validation': {
                'artifact_dir': Path('artifacts/heads_up_validation'),
                'reference_policy_path': None,
                'environment': {
                    'starting_stack': 40, 'small_blind': 1,
                    'big_blind': 2, 'seed': 542,
                },
                'trainer': {
                    'hidden': 16, 'blocks': 1, 'learning_rate': 1e-3,
                    'advantage_capacity': 1_000, 'policy_capacity': 1_000,
                    'range_capacity': 1_000,
                    'max_nodes_per_traversal': 80, 'max_depth': 24,
                    'exploration': 0.0,
                    'range_loss_weight': 0.01,
                    'reservoir_turnover_fraction': 0.18,
                    'reinitialize_advantage_each_iteration': False,
                    'seed': 542,
                },
                'campaign': {
                    'target_iteration': 4, 'traversals_per_player': 1,
                    'advantage_steps': 2, 'policy_steps': 2,
                    'batch_size': 32, 'evaluate_every': 2,
                    'checkpoint_every': 2, 'snapshot_every': 2,
                    'evaluation_games_per_player': 3,
                    'range_evaluation_hands_per_opponent': 3,
                    'range_evaluation_batch_size': 32,
                    'range_training_hands_per_iteration': 8,
                    'range_batch_size': 16,
                    'league_games_per_player': 3,
                    'validation_seed': 502_700,
                    'opponent_profiles': (
                        'random', 'calling_station', 'tight_aggressive',
                    ),
                    'league_opponents': 1,
                    'keep_full_checkpoints': 1,
                },
            },
        }
        if PROFILE not in PROFILES:
            raise ValueError(f'Unknown profile {PROFILE!r}; use production or validation.')
        SETTINGS = PROFILES[PROFILE]
        ARTIFACT_DIR = SETTINGS['artifact_dir']
        ENVIRONMENT_CONFIG = dict(SETTINGS['environment'])
        TRAINER_CONFIG = dict(SETTINGS['trainer'])
        campaign_values = dict(SETTINGS['campaign'])
        campaign_values['traversal_workers'] = CPU_WORKERS
        if os.getenv('POKER_TARGET_ITERATION'):
            campaign_values['target_iteration'] = int(os.environ['POKER_TARGET_ITERATION'])
        if os.getenv('POKER_TRAVERSALS'):
            campaign_values['traversals_per_player'] = int(os.environ['POKER_TRAVERSALS'])
        CAMPAIGN_CONFIG = CampaignConfig(**campaign_values)
        RANGE_PLOTS_EVERY_EVALS = int(os.getenv('POKER_RANGE_EVERY_EVALS', '1'))
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

        display(pd.Series({
            'profile': PROFILE,
            'artifacts': str(ARTIFACT_DIR.resolve()),
            'engine backend': 'compiled C++',
            'target iteration': CAMPAIGN_CONFIG.target_iteration,
            'CPU traversal workers': CAMPAIGN_CONFIG.traversal_workers,
            'traversals/player': CAMPAIGN_CONFIG.traversals_per_player,
            'advantage steps': CAMPAIGN_CONFIG.advantage_steps,
            'policy steps': CAMPAIGN_CONFIG.policy_steps,
            'batch size': CAMPAIGN_CONFIG.batch_size,
            'evaluate every': CAMPAIGN_CONFIG.evaluate_every,
            'checkpoint every': CAMPAIGN_CONFIG.checkpoint_every,
            'snapshot every': CAMPAIGN_CONFIG.snapshot_every,
            'validation hands/evaluation': (
                2 * CAMPAIGN_CONFIG.evaluation_games_per_player
                * (
                    len(CAMPAIGN_CONFIG.opponent_profiles)
                    + int(CAMPAIGN_CONFIG.reference_policy_path is not None)
                )
            ),
            'frozen reference policy': (
                CAMPAIGN_CONFIG.reference_policy_path or 'disabled'
            ),
            'range holdout hands/opponent': (
                CAMPAIGN_CONFIG.range_evaluation_hands_per_opponent
            ),
            'range output combinations': 1_326,
            'range loss weight': TRAINER_CONFIG['range_loss_weight'],
            'reservoir turnover when full': (
                f"{100 * TRAINER_CONFIG['reservoir_turnover_fraction']:.0f}% oldest"
            ),
            'strategy plots every evaluations': RANGE_PLOTS_EVERY_EVALS,
        }, name='campaign'))
    """,
    10: r"""
        env = HeadsUpHoldemEngine(**ENVIRONMENT_CONFIG)
        trainer, RESUMED = load_or_create_trainer(
            env,
            ARTIFACT_DIR,
            device=DEVICE,
            trainer_kwargs=TRAINER_CONFIG,
        )
        campaign = ProductionCampaign(trainer, ARTIFACT_DIR, CAMPAIGN_CONFIG)
        initial_policy_nets = campaign.baseline.policy_nets
        all_networks = trainer.advantage_nets + trainer.policy_nets
        assert len(all_networks) == 4
        assert all(next(network.parameters()).is_cuda for network in all_networks)
        display(pd.Series({
            'resumed': RESUMED,
            'current iteration': trainer.iteration,
            'last completely fitted iteration': trainer.last_fitted_iteration,
            'automatic recovery needed': (
                trainer.last_fitted_iteration < trainer.iteration
            ),
            'target iteration': CAMPAIGN_CONFIG.target_iteration,
            'information-state features': trainer.input_dim,
            'network architecture': trainer.network_architecture,
            'policy network architecture': trainer.policy_network_architecture,
            'range output combinations': 1_326,
            'range loss weight': trainer.range_loss_weight,
            'reservoir turnover fraction': trainer.reservoir_turnover_fraction,
            'reservoir merge': 'exact-order bulk tensor copies',
            'player fitting': (
                'two concurrent CUDA streams'
                if trainer.device.type == 'cuda'
                else 'sequential CPU fallback'
            ),
            'advantage parameters/network': sum(
                parameter.numel() for parameter in trainer.advantage_nets[0].parameters()
            ),
            'policy parameters/network': sum(
                parameter.numel() for parameter in trainer.policy_nets[0].parameters()
            ),
            'total trainable parameters': sum(
                parameter.numel() for network in all_networks
                for parameter in network.parameters()
            ),
            'advantage networks': len(trainer.advantage_nets),
            'policy networks': len(trainer.policy_nets),
            'network device': str(next(all_networks[0].parameters()).device),
            'advantage reservoir capacities': [buffer.capacity for buffer in trainer.advantage_buffers],
            'policy reservoir capacities': [buffer.capacity for buffer in trainer.policy_buffers],
            'advantage reservoir sizes': [len(buffer) for buffer in trainer.advantage_buffers],
            'policy reservoir sizes': [len(buffer) for buffer in trainer.policy_buffers],
        }, name='trainer'))
    """,
    12: r"""
        architecture_figure = plot_network_architecture(trainer)
        architecture_path = ARTIFACT_DIR / 'network_architecture.png'
        architecture_figure.savefig(architecture_path, dpi=180, bbox_inches='tight')
        display(architecture_figure)
        plt.close(architecture_figure)
        print(f'Saved architecture diagram to {architecture_path.resolve()}')
    """,
    14: r"""
        PERIODIC_PREFLOP_IDS = (
            'bb_vs_btn_open', 'bb_vs_btn_limp', 'btn_vs_bb_3bet',
        )
        PERIODIC_POSTFLOP_IDS = (
            'bb_flop_vs_halfpot', 'btn_turn_vs_halfpot',
            'bb_river_vs_btn_pot',
        )
        LIVE_FLOP, LIVE_TURN, LIVE_RIVER = (51, 18, 0), 35, 41

        def _display_figure(figure):
            display(figure)
            plt.close(figure)

        def _human_duration(seconds):
            seconds = max(0, int(seconds))
            hours, remainder = divmod(seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            return f'{hours:d}h {minutes:02d}m' if hours else f'{minutes:d}m {seconds:02d}s'

        def save_periodic_strategy_report(evaluation_number):
            print(
                f'Generating critical-state report after evaluation {evaluation_number}...',
                flush=True,
            )
            analyzer = StrategyAnalyzer(trainer, batch_size=4096)
            preflop_catalog = {
                scenario.scenario_id: scenario for scenario in preflop_scenarios()
            }
            postflop_catalog = {
                scenario.scenario_id: scenario
                for scenario in postflop_scenarios(
                    flop=LIVE_FLOP, turn=LIVE_TURN, river=LIVE_RIVER
                )
            }
            preflop_reports = analyzer.analyze_cases(
                [preflop_catalog[key] for key in PERIODIC_PREFLOP_IDS]
            )
            postflop_reports = analyzer.analyze_cases(
                [postflop_catalog[key] for key in PERIODIC_POSTFLOP_IDS]
            )
            reports = preflop_reports + postflop_reports
            report_root = ARTIFACT_DIR / 'strategy_analysis'
            report_dir = report_root / f'step_{trainer.iteration:08d}'
            report_dir.mkdir(parents=True, exist_ok=True)
            by_id = {report.scenario.scenario_id: report for report in reports}
            for report in reports:
                report.hand_table.to_csv(
                    report_dir / f'{report.scenario.scenario_id}_169.csv',
                    index=False,
                )
            display(pd.DataFrame([report.state_summary for report in reports])[
                ['label', 'hero_role', 'street', 'board', 'pot_bb',
                 'to_call_bb', 'pot_odds', 'spr', 'physical_combos']
            ])
            preflop_figure = plot_call_maps(preflop_reports)
            preflop_figure.savefig(
                report_dir / 'preflop_call_maps.png', dpi=160, bbox_inches='tight'
            )
            _display_figure(preflop_figure)
            selected = by_id['bb_vs_btn_open']
            mix_figure = plot_range_heatmaps(
                selected, metrics=('p_fold', 'p_call', 'p_aggressive')
            )
            mix_figure.savefig(
                report_dir / 'live_action_mix.png', dpi=160, bbox_inches='tight'
            )
            _display_figure(mix_figure)
            representative = ['AA', 'KK', 'QQ', 'AKs', 'AQs', 'AJo', 'TT', '76s', '22']
            display(
                selected.hand_table.set_index('hand').reindex(representative)[
                    ['p_fold', 'p_call', 'p_aggressive',
                     'combo_std_p_call', 'between_net_std_p_call']
                ].style.format('{:.1%}')
            )
            postflop_figure = plot_call_maps(postflop_reports)
            postflop_figure.savefig(
                report_dir / 'postflop_call_maps.png', dpi=160, bbox_inches='tight'
            )
            _display_figure(postflop_figure)
            previous_dirs = sorted(
                path for path in report_root.glob('step_*')
                if path.name < report_dir.name
            )
            if previous_dirs:
                previous_path = previous_dirs[-1] / 'bb_vs_btn_open_169.csv'
                if previous_path.exists():
                    previous = pd.read_csv(previous_path)
                    current = selected.hand_table
                    change = previous[['hand', 'row', 'column', 'p_call']].merge(
                        current[['hand', 'row', 'column', 'p_call']],
                        on=['hand', 'row', 'column'],
                        suffixes=('_previous', '_current'),
                    )
                    change['delta_p_call'] = (
                        change['p_call_current'] - change['p_call_previous']
                    )
                    delta_figure = plot_range_delta(
                        change,
                        metric='delta_p_call',
                        title='Call-frequency change since previous report',
                    )
                    delta_figure.savefig(
                        report_dir / 'call_delta_vs_previous_report.png',
                        dpi=160, bbox_inches='tight',
                    )
                    _display_figure(delta_figure)
            print(f'Critical-state report saved to {report_dir.resolve()}')

        def show_existing_metrics():
            frame = metrics_frame(trainer.metrics)
            if frame.empty:
                print('No completed training iterations yet.')
                return
            display(frame.tail(10))
            _display_figure(plot_training_dashboard(frame))
            if 'range_nll' in frame and frame['range_nll'].notna().any():
                _display_figure(plot_range_dashboard(frame))

        def on_iteration(row):
            completed = trainer.iteration
            target = CAMPAIGN_CONFIG.target_iteration
            fraction = min(1.0, completed / max(target, 1))
            width = 28
            progress = '#' * int(width * fraction) + '-' * (
                width - int(width * fraction)
            )
            recent = [
                float(item['seconds']) for item in trainer.metrics[-20:]
                if 'seconds' in item
            ]
            eta = _human_duration(
                (sum(recent) / max(len(recent), 1)) * max(target - completed, 0)
            )
            current_ev = row.get('benchmark_composite_ev_bb')
            if current_ev is not None and math.isfinite(float(current_ev)):
                validation = f'{float(current_ev):+.4f} BB/hand (new evaluation)'
            else:
                latest = next((
                    item for item in reversed(trainer.metrics)
                    if item.get('benchmark_composite_ev_bb') is not None
                    and math.isfinite(float(item['benchmark_composite_ev_bb']))
                ), None)
                until = CAMPAIGN_CONFIG.evaluate_every - (
                    completed % CAMPAIGN_CONFIG.evaluate_every
                )
                validation = (
                    f'pending; first evaluation in {until} iterations'
                    if latest is None
                    else f"{float(latest['benchmark_composite_ev_bb']):+.4f} "
                         f"BB/hand (last iteration {int(latest['iteration'])}; next in {until})"
                )
            print(
                f'[{progress}] {100*fraction:6.2f}% | '
                f'iteration {completed:,}/{target:,} | ETA ~{eta}\n'
                f"step {row['seconds']:.2f}s | traversal "
                f"{row.get('traversal_seconds', float('nan')):.2f}s | "
                f"adv fit {row.get('advantage_fit_seconds', float('nan')):.2f}s | "
                f"policy fit {row.get('policy_fit_seconds', float('nan')):.2f}s\n"
                f"nodes {int(row['nodes']):,} | rollouts {int(row['rollouts']):,} | "
                f"throughput {row.get('traversal_nodes_per_second', 0):,.0f} nodes/s | "
                f"VRAM peak {row.get('gpu_peak_memory_mb', 0):.0f} MiB\n"
                f"adv buffers {[len(buffer) for buffer in trainer.advantage_buffers]} | "
                f"policy buffers {[len(buffer) for buffer in trainer.policy_buffers]}\n"
                f"range buffers {[len(buffer) for buffer in trainer.range_buffers]} | "
                f"range collection {row.get('range_collection_seconds', 0):.2f}s\n"
                f"adv turnover {[buffer.turnover_events for buffer in trainer.advantage_buffers]} | "
                f"policy turnover {[buffer.turnover_events for buffer in trainer.policy_buffers]}\n"
                f"policy action loss {row.get('policy_action_loss', float('nan')):.5f} | "
                f"range loss {row.get('policy_range_loss', float('nan')):.5f} "
                f"(weight {row.get('range_loss_weight', float('nan')):.3f})\n"
                f'validation composite: {validation}',
                flush=True,
            )

        def on_evaluation(frame):
            clear_output(wait=True)
            evaluated = frame.dropna(subset=['benchmark_composite_ev_bb'])
            evaluation_number = len(evaluated)
            current = evaluated.iloc[-1]
            print(
                f'EVALUATION {evaluation_number} COMPLETE — '
                f'iteration {trainer.iteration:,} of {CAMPAIGN_CONFIG.target_iteration:,}'
            )
            rows = []
            for profile in CAMPAIGN_CONFIG.opponent_profiles:
                rows.append({
                    'opponent': profile.replace('_', ' '),
                    'EV BB/hand': current[f'benchmark_{profile}_mean_ev_bb'],
                    '95% low': current[f'benchmark_{profile}_ci95_low_bb'],
                    '95% high': current[f'benchmark_{profile}_ci95_high_bb'],
                    'delta vs initial': current.get(
                        f'benchmark_{profile}_delta_ev_bb', float('nan')
                    ),
                    'P(delta > 0)': current.get(
                        f'benchmark_{profile}_probability_delta_positive',
                        float('nan'),
                    ),
                    'BTN/SB EV': current[f'benchmark_{profile}_ev_BTN_SB_bb'],
                    'BB EV': current[f'benchmark_{profile}_ev_BB_bb'],
                })
            if campaign.reference_policy is not None:
                profile = 'reference_policy'
                rows.append({
                    'opponent': (
                        f'frozen policy '
                        f'{campaign.reference_policy.iteration}'
                    ),
                    'EV BB/hand': current[f'benchmark_{profile}_mean_ev_bb'],
                    '95% low': current[f'benchmark_{profile}_ci95_low_bb'],
                    '95% high': current[f'benchmark_{profile}_ci95_high_bb'],
                    'delta vs initial': current.get(
                        f'benchmark_{profile}_delta_ev_bb', float('nan')
                    ),
                    'P(delta > 0)': current.get(
                        f'benchmark_{profile}_probability_delta_positive',
                        float('nan'),
                    ),
                    'BTN/SB EV': current[f'benchmark_{profile}_ev_BTN_SB_bb'],
                    'BB EV': current[f'benchmark_{profile}_ev_BB_bb'],
                })
            display(pd.DataFrame(rows).style.format({
                'EV BB/hand': '{:+.3f}', '95% low': '{:+.3f}',
                '95% high': '{:+.3f}', 'delta vs initial': '{:+.3f}',
                'P(delta > 0)': '{:.1%}', 'BTN/SB EV': '{:+.3f}',
                'BB EV': '{:+.3f}',
            }))
            display(evaluated[[
                'iteration', 'benchmark_composite_ev_bb',
                'benchmark_composite_lcb95_bb',
                'league_mean_ev_bb', 'league_worst_ev_bb',
                'range_nll', 'range_uniform_nll',
                'range_information_gain', 'range_top10_accuracy',
                'range_top50_accuracy',
                'promoted_to_champion',
            ]].tail(10))
            _display_figure(plot_training_dashboard(frame))
            range_path = save_range_dashboard(
                frame, ARTIFACT_DIR / 'opponent_range_dashboard.png'
            )
            _display_figure(plot_range_dashboard(frame))
            print(f'Opponent-range dashboard saved to {range_path.resolve()}')
            reservoir_dashboard = ARTIFACT_DIR / 'range_reservoir_dashboard.png'
            if reservoir_dashboard.exists():
                display(Image(filename=str(reservoir_dashboard)))
                print(
                    'Range-reservoir composition dashboard saved to '
                    f'{reservoir_dashboard.resolve()}'
                )
            if evaluation_number % RANGE_PLOTS_EVERY_EVALS == 0:
                save_periodic_strategy_report(evaluation_number)
            else:
                remaining = (
                    RANGE_PLOTS_EVERY_EVALS
                    - evaluation_number % RANGE_PLOTS_EVERY_EVALS
                )
                print(
                    f'Full hand/situation plots will refresh in '
                    f'{remaining} evaluation(s).'
                )

        show_existing_metrics()
    """,
    16: r"""
        print(
            f'RESUMING HU TRAINING from completed iteration '
            f'{trainer.iteration:,}; next iteration {trainer.iteration + 1:,}.\n'
            f'Progress will update after each completed iteration. '
            f'The first update can take several minutes.',
            flush=True,
        )
        campaign.run(on_iteration=on_iteration, on_evaluation=on_evaluation)
        print(f'Reached iteration {trainer.iteration}; full state is checkpointed under {ARTIFACT_DIR}.')
    """,
    17: r"""
        training_metrics = metrics_frame(trainer.metrics)
        display(training_metrics.tail(20))
        display(plot_training_dashboard(training_metrics))
        if 'range_nll' in training_metrics and training_metrics['range_nll'].notna().any():
            display(plot_range_dashboard(training_metrics))
    """,
    19: r"""
        FINAL_TEST_GAMES_PER_SEAT = int(os.getenv('POKER_FINAL_TEST_GAMES', '3000'))
        FINAL_TEST_SEED = 909_731
        if trainer.iteration >= CAMPAIGN_CONFIG.target_iteration:
            final_suite = evaluate_benchmark_suite(
                trainer,
                profiles=CAMPAIGN_CONFIG.opponent_profiles,
                games_per_seat=FINAL_TEST_GAMES_PER_SEAT,
                seed=FINAL_TEST_SEED,
                baseline_policy_nets=campaign.baseline.policy_nets,
                reference_policy_nets=(
                    campaign.reference_policy.policy_nets
                    if campaign.reference_policy is not None
                    else None
                ),
            )
            display(pd.Series(final_suite, name='final untouched holdout'))
        else:
            print('Final holdout skipped: finish the configured target first.')
    """,
    21: r"""
        analyzer = StrategyAnalyzer(trainer, batch_size=4096)
        analysis_dir = (
            ARTIFACT_DIR / 'strategy_analysis'
            / f'step_{trainer.iteration:08d}'
        )
        analysis_dir.mkdir(parents=True, exist_ok=True)
        preflop_catalog = {
            scenario.scenario_id: scenario for scenario in preflop_scenarios()
        }
        bet_facing_ids = (
            'bb_vs_btn_open', 'bb_vs_btn_limp', 'btn_vs_bb_3bet',
        )
        preflop_reports = analyzer.analyze_cases(
            [preflop_catalog[key] for key in bet_facing_ids]
        )
        for report in preflop_reports:
            report.hand_table.to_csv(
                analysis_dir / f'{report.scenario.scenario_id}_169.csv',
                index=False,
            )
            report.combo_table.to_csv(
                analysis_dir / f'{report.scenario.scenario_id}_combos.csv',
                index=False,
            )
        display(pd.DataFrame([report.state_summary for report in preflop_reports]))
        preflop_call_figure = plot_call_maps(preflop_reports)
        preflop_call_figure.savefig(
            analysis_dir / 'preflop_call_maps.png',
            dpi=180, bbox_inches='tight',
        )
        display(preflop_call_figure)
    """,
    23: r"""
        selected_preflop = next(
            report for report in preflop_reports
            if report.scenario.scenario_id == 'bb_vs_btn_open'
        )
        mix_figure = plot_range_heatmaps(
            selected_preflop,
            metrics=('p_fold', 'p_call', 'p_aggressive'),
        )
        mix_figure.savefig(
            analysis_dir / 'bb_vs_open_action_mix.png',
            dpi=180, bbox_inches='tight',
        )
        display(mix_figure)
    """,
    25: r"""
        baseline_preflop = analyzer.analyze_range(
            selected_preflop.scenario,
            policy_nets=initial_policy_nets,
        )
        preflop_change = compare_ranges(baseline_preflop, selected_preflop)
        change_figure = plot_range_delta(
            preflop_change,
            metric='delta_p_call',
            title='Call-frequency change: current minus initial policy',
        )
        change_figure.savefig(
            analysis_dir / 'call_delta_vs_initial.png',
            dpi=180, bbox_inches='tight',
        )
        display(change_figure)
        display(preflop_change.nlargest(20, 'strategy_total_variation')[
            ['hand', 'delta_p_fold', 'delta_p_call',
             'delta_p_aggressive', 'strategy_total_variation']
        ])
    """,
    27: r"""
        FLOP = (51, 18, 0)   # As 7d 2c
        TURN = 35            # Jh
        RIVER = 41           # 4s
        postflop_catalog = {
            scenario.scenario_id: scenario
            for scenario in postflop_scenarios(
                flop=FLOP, turn=TURN, river=RIVER
            )
        }
        postflop_ids = (
            'bb_flop_vs_halfpot', 'btn_turn_vs_halfpot',
            'bb_river_vs_btn_pot',
        )
        postflop_reports = analyzer.analyze_cases(
            [postflop_catalog[key] for key in postflop_ids]
        )
        for report in postflop_reports:
            report.hand_table.to_csv(
                analysis_dir / f'{report.scenario.scenario_id}_169.csv',
                index=False,
            )
        display(pd.DataFrame([report.state_summary for report in postflop_reports]))
        postflop_call_figure = plot_call_maps(postflop_reports)
        postflop_call_figure.savefig(
            analysis_dir / 'postflop_call_maps.png',
            dpi=180, bbox_inches='tight',
        )
        display(postflop_call_figure)
    """,
    29: r"""
        turn_scenario = postflop_catalog['btn_turn_vs_halfpot']
        turn_sweep = analyzer.analyze_next_cards(
            turn_scenario, hero_cards=(12, 11)  # Ac Kc
        )
        turn_sweep.to_csv(
            analysis_dir / 'AcKc_turn_card_sweep.csv', index=False
        )
        turn_figure = plot_card_sweep(
            turn_sweep,
            metric='p_call',
            title='AcKc on As-7d-2c: call frequency by turn card',
        )
        turn_figure.savefig(
            analysis_dir / 'AcKc_turn_call_sweep.png',
            dpi=180, bbox_inches='tight',
        )
        display(turn_figure)
    """,
    31: r"""
        divergence_columns = [
            'hand', 'p_call', 'p_aggressive',
            'combo_std_p_call', 'between_net_std_p_call',
            'between_net_std_p_aggressive',
        ]
        display(
            selected_preflop.hand_table
            .sort_values('between_net_std_p_call', ascending=False)
            [divergence_columns].head(30)
        )
    """,
    33: r"""
        HUMAN_MATCH_PATH = Path('artifacts/human_matches.csv')
        if HUMAN_MATCH_PATH.exists():
            human = pd.read_csv(HUMAN_MATCH_PATH)
            required = {
                'hand_id', 'session_id', 'bot_seat', 'button',
                'opponent_pool', 'payoff_bb',
            }
            missing = required - set(human.columns)
            if missing:
                raise ValueError(
                    f'Human match log is missing columns: {sorted(missing)}'
                )
            session_means = human.groupby('session_id')['payoff_bb'].mean()
            mean_ev = float(human['payoff_bb'].mean())
            session_se = (
                float(session_means.std(ddof=1) / math.sqrt(len(session_means)))
                if len(session_means) > 1 else float('nan')
            )
            display(pd.Series({
                'hands': len(human),
                'sessions': human['session_id'].nunique(),
                'mean BB/hand': mean_ev,
                'BB/100': 100 * mean_ev,
                'clustered SE': session_se,
                '95% CI low': mean_ev - 1.96 * session_se,
                '95% CI high': mean_ev + 1.96 * session_se,
            }, name='held-out human match'))
            human.assign(cumulative_bb=human['payoff_bb'].cumsum()).plot(
                x='hand_id', y='cumulative_bb', figsize=(12, 4),
                title='Cumulative held-out human-match result', grid=True,
            )
            plt.show()
        else:
            print(
                f'No human match log yet. Expected schema at '
                f'{HUMAN_MATCH_PATH.resolve()}:'
            )
            display(pd.DataFrame(columns=[
                'hand_id', 'session_id', 'bot_seat', 'button',
                'opponent_pool', 'payoff_bb',
            ]))
    """,
    35: r"""
        artifact_inventory = {
            'latest checkpoint manifest': campaign.latest_manifest,
            'initial frozen policy': campaign.baseline_path,
            'champion policy': campaign.champion_path,
            'append-only metrics': campaign.metrics_path,
            'run configuration': campaign.run_config_path,
            'run configuration history': campaign.run_config_history_path,
            'evaluation hand histories': campaign.evaluation_dir,
            'network architecture': ARTIFACT_DIR / 'network_architecture.png',
            'strategy reports': ARTIFACT_DIR / 'strategy_analysis',
            'milestone policy snapshots': campaign.snapshot_dir,
            'frozen previous policy': (
                Path(CAMPAIGN_CONFIG.reference_policy_path)
                if CAMPAIGN_CONFIG.reference_policy_path is not None
                else Path('disabled')
            ),
        }
        display(pd.Series({
            key: str(path.resolve()) for key, path in artifact_inventory.items()
        }, name='artifacts'))
    """,
}

for index, value in code.items():
    notebook["cells"][index]["source"] = lines(value)
    notebook["cells"][index]["outputs"] = []
    notebook["cells"][index]["execution_count"] = None

notebook["metadata"]["kernelspec"] = {
    "display_name": "Python3 (ipykernel)",
    "language": "python",
    "name": "python3",
}
serialized = json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"
for output in OUTPUTS:
    output.write_text(serialized, encoding="utf-8")
    print(f"Wrote {output} with {len(notebook['cells'])} cells")
