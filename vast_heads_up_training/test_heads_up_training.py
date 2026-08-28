"""Focused correctness tests for the heads-up Deep CFR path."""

from __future__ import annotations

import json
import os
import random
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import torch

from heads_up_cfr import (
    CHECKPOINT_KIND,
    DEFAULT_ROOT_STACK_DEPTHS_BB,
    NETWORK_ARCHITECTURE,
    ROOT_STACK_DISTRIBUTION_MIXED,
    HeadsUpNeuralCFR,
    ReservoirBuffer,
)
from heads_up_engine import (
    ACTION_NAMES,
    ACTION_SCHEMA_VERSION,
    ENGINE_SCHEMA_VERSION,
    NUM_ACTIONS,
    NUM_PLAYERS,
    HeadsUpHoldemEngine,
)
from heads_up_models import ENCODER_SCHEMA_VERSION, encoder_metadata
from heads_up_production import (
    _collect_independent_range_training_hands,
    CampaignConfig,
    ProductionCampaign,
    TightAggressiveOpponent,
    evaluate_benchmark_suite,
    load_or_create_trainer,
    range_reservoir_statistics,
    save_policy_snapshot,
)
from heads_up_search import HeadsUpNetworkPolicy
from train_heads_up import parse_args as parse_training_args


def _small_trainer(
    *,
    max_history: int = 32,
    engine_type=HeadsUpHoldemEngine,
    **trainer_overrides,
) -> HeadsUpNeuralCFR:
    env = engine_type(
        starting_stack=8,
        small_blind=1,
        big_blind=2,
        seed=101,
    )
    config = {
        "hidden": 8,
        "blocks": 0,
        "advantage_capacity": 256,
        "policy_capacity": 256,
        "max_history": max_history,
        "max_nodes_per_traversal": 64,
        "max_depth": 24,
        "exploration": 0.0,
        "seed": 202,
    }
    config.update(trainer_overrides)
    return HeadsUpNeuralCFR(
        env,
        **config,
    )


class HeadsUpTrainingTests(unittest.TestCase):
    def test_parallel_runtime_avoids_low_descriptor_tensor_sharing(self):
        strategies = torch.multiprocessing.get_all_sharing_strategies()
        if "file_system" in strategies:
            self.assertEqual(
                torch.multiprocessing.get_sharing_strategy(),
                "file_system",
            )
        if os.name == "posix":
            import resource

            soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
            self.assertGreaterEqual(
                int(soft_limit),
                min(int(hard_limit), 65_536),
            )

    def test_production_cli_defaults_match_compact_campaign(self):
        args = parse_training_args([])
        campaign = CampaignConfig()
        self.assertEqual(args.traversals, 1_024)
        self.assertLessEqual(args.traversal_workers, 12)
        self.assertGreaterEqual(args.traversal_workers, 1)
        self.assertEqual(args.adv_steps, 245)
        self.assertEqual(args.policy_steps, 245)
        self.assertEqual(args.batch_size, 4_096)
        self.assertEqual(args.hidden, 256)
        self.assertEqual(args.blocks, 6)
        self.assertEqual(args.learning_rate, 1e-3)
        self.assertEqual(args.advantage_capacity, 1_000_000)
        self.assertEqual(args.policy_capacity, 1_000_000)
        self.assertEqual(args.advantage_reinitialize_from_iteration, 25)
        self.assertEqual(args.advantage_reinitialize_cycle, 25)
        self.assertEqual(args.eval_every, 25)
        self.assertEqual(args.eval_games, 10_000)
        self.assertEqual(
            tuple(args.eval_profiles),
            ("random", "calling_station", "tight_aggressive"),
        )
        self.assertEqual(campaign.traversals_per_player, args.traversals)
        self.assertEqual(campaign.advantage_steps, args.adv_steps)
        self.assertEqual(campaign.policy_steps, args.policy_steps)
        self.assertEqual(campaign.batch_size, args.batch_size)
        self.assertEqual(campaign.evaluation_games_per_player, args.eval_games)
        self.assertEqual(
            campaign.root_stack_distribution,
            args.root_stack_distribution,
        )
        self.assertEqual(
            campaign.root_stack_depths_bb,
            tuple(args.root_stack_depths_bb),
        )

    def test_mixed_root_stacks_are_exactly_split_and_reproducible(self):
        first = _small_trainer()
        second = _small_trainer()
        first_schedule = first._root_stack_schedule(
            200,
            ROOT_STACK_DISTRIBUTION_MIXED,
            DEFAULT_ROOT_STACK_DEPTHS_BB,
        )
        second_schedule = second._root_stack_schedule(
            200,
            ROOT_STACK_DISTRIBUTION_MIXED,
            DEFAULT_ROOT_STACK_DEPTHS_BB,
        )
        self.assertEqual(first_schedule, second_schedule)
        equal = [stacks for stacks in first_schedule if stacks[0] == stacks[1]]
        unequal = [stacks for stacks in first_schedule if stacks[0] != stacks[1]]
        self.assertEqual((len(equal), len(unequal)), (100, 100))
        self.assertEqual(
            sum(first_stack > second_stack for first_stack, second_stack in unequal),
            50,
        )
        self.assertEqual(
            sum(first_stack < second_stack for first_stack, second_stack in unequal),
            50,
        )
        allowed = {depth * first.env.bb for depth in DEFAULT_ROOT_STACK_DEPTHS_BB}
        self.assertTrue(
            all(
                first_stack in allowed and second_stack in allowed
                for first_stack, second_stack in first_schedule
            )
        )

    def test_evaluation_game_count_is_a_logged_resume_override(self):
        trainer = _small_trainer()
        initial = CampaignConfig(
            target_iteration=2,
            traversals_per_player=1,
            traversal_workers=1,
            advantage_steps=1,
            policy_steps=1,
            batch_size=8,
            evaluate_every=1,
            checkpoint_every=1,
            snapshot_every=1,
            evaluation_games_per_player=2,
            league_games_per_player=1,
            opponent_profiles=("random",),
            league_opponents=0,
        )
        with tempfile.TemporaryDirectory() as directory:
            artifact_dir = Path(directory)
            ProductionCampaign(trainer, artifact_dir, initial)
            resumed = replace(initial, evaluation_games_per_player=10_000)
            ProductionCampaign(trainer, artifact_dir, resumed)
            history = [
                json.loads(line)
                for line in (
                    artifact_dir / "run_config_history.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                history[-1]["campaign_changes"][
                    "evaluation_games_per_player"
                ],
                {"previous": 2, "current": 10_000},
            )

    def test_mixed_root_stack_upgrade_is_logged_for_existing_campaign(self):
        trainer = _small_trainer()
        config = replace(
            CampaignConfig(),
            target_iteration=2,
            traversals_per_player=1,
            traversal_workers=1,
            advantage_steps=1,
            policy_steps=1,
            batch_size=8,
            evaluate_every=1,
            checkpoint_every=1,
            snapshot_every=1,
            evaluation_games_per_player=2,
            league_games_per_player=1,
            opponent_profiles=("random",),
            league_opponents=0,
        )
        with tempfile.TemporaryDirectory() as directory:
            artifact_dir = Path(directory)
            ProductionCampaign(trainer, artifact_dir, config)
            run_config_path = artifact_dir / "run_config.json"
            old_config = json.loads(run_config_path.read_text(encoding="utf-8"))
            old_config["campaign"].pop("root_stack_distribution")
            old_config["campaign"].pop("root_stack_depths_bb")
            run_config_path.write_text(
                json.dumps(old_config),
                encoding="utf-8",
            )
            ProductionCampaign(trainer, artifact_dir, config)
            history = [
                json.loads(line)
                for line in (
                    artifact_dir / "run_config_history.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]
            changes = history[-1]["campaign_changes"]
            self.assertEqual(
                changes["root_stack_distribution"]["current"],
                ROOT_STACK_DISTRIBUTION_MIXED,
            )
            self.assertEqual(
                changes["root_stack_depths_bb"]["current"],
                list(DEFAULT_ROOT_STACK_DEPTHS_BB),
            )

    def test_reservoir_compacts_and_round_trips_without_row_objects(self):
        buffer = ReservoirBuffer(16, random.Random(17))
        for index in range(40):
            buffer.add(
                (
                    torch.full((5,), float(index), dtype=torch.float16),
                    torch.full((NUM_ACTIONS,), float(index)),
                    torch.ones(NUM_ACTIONS),
                    torch.tensor(float(index + 1)),
                )
            )
        state = buffer.state_dict()
        self.assertEqual(len(buffer), 16)
        self.assertEqual(state["format_version"], 3)
        self.assertEqual(tuple(state["fields"][0].shape), (16, 5))

        restored = ReservoirBuffer(16, random.Random(99))
        restored.load_state_dict(state)
        self.assertEqual(len(restored), len(buffer))
        self.assertEqual(restored.seen, buffer.seen)
        fields = restored.sample_fields(8)
        self.assertEqual(tuple(fields[0].shape), (8, 5))

    def test_reservoir_checkpoint_can_expand_but_not_shrink(self):
        buffer = ReservoirBuffer(16, random.Random(17))
        for index in range(40):
            buffer.add(
                (
                    torch.full((5,), float(index), dtype=torch.float16),
                    torch.full((NUM_ACTIONS,), float(index)),
                    torch.ones(NUM_ACTIONS),
                    torch.tensor(float(index + 1)),
                )
            )
        state = buffer.state_dict()

        expanded = ReservoirBuffer(64, random.Random(99))
        expanded.load_state_dict(state)
        self.assertEqual(len(expanded), 16)
        self.assertEqual(expanded.capacity, 64)
        self.assertEqual(expanded.seen, 40)

        with self.assertRaisesRegex(ValueError, "exceeds configured capacity"):
            ReservoirBuffer(8, random.Random(99)).load_state_dict(state)

    def test_full_reservoir_evicts_oldest_eighteen_percent_in_chunks(self):
        buffer = ReservoirBuffer(10, random.Random(17), turnover_fraction=0.18)
        for index in range(10):
            buffer.add((torch.tensor(index), torch.tensor(float(index + 1))))
        self.assertEqual(
            [int(buffer.memory[index][0]) for index in range(len(buffer))],
            list(range(10)),
        )

        buffer.add((torch.tensor(10), torch.tensor(11.0)))
        self.assertEqual(len(buffer), 9)
        self.assertEqual(buffer.turnover_events, 1)
        self.assertEqual(buffer.evicted_samples, 2)
        self.assertEqual(
            [int(buffer.memory[index][0]) for index in range(len(buffer))],
            list(range(2, 11)),
        )
        buffer.add((torch.tensor(11), torch.tensor(12.0)))
        self.assertEqual(len(buffer), 10)
        self.assertEqual(
            [int(buffer.memory[index][0]) for index in range(len(buffer))],
            list(range(2, 12)),
        )
        buffer.add((torch.tensor(12), torch.tensor(13.0)))
        self.assertEqual(len(buffer), 9)
        self.assertEqual(buffer.turnover_events, 2)
        self.assertEqual(buffer.evicted_samples, 4)
        self.assertEqual(
            [int(buffer.memory[index][0]) for index in range(len(buffer))],
            list(range(4, 13)),
        )

        state = buffer.state_dict()
        restored = ReservoirBuffer(
            10,
            random.Random(99),
            turnover_fraction=0.18,
        )
        restored.load_state_dict(state)
        self.assertEqual(restored.turnover_events, 2)
        self.assertEqual(restored.evicted_samples, 4)
        self.assertEqual(
            [int(restored.memory[index][0]) for index in range(len(restored))],
            list(range(4, 13)),
        )

    def test_packed_row_turnover_matches_row_add_order(self):
        row_buffer = ReservoirBuffer(10, random.Random(7))
        packed_buffer = ReservoirBuffer(10, random.Random(7))
        bulk_buffer = ReservoirBuffer(10, random.Random(7))
        values = torch.arange(27, dtype=torch.int64)
        weights = torch.arange(1, 28, dtype=torch.float32)
        for index in range(10):
            row_buffer.add((values[index], weights[index]))
            packed_buffer.add_packed_row((values, weights), index)
        bulk_buffer.add_packed_fields((values[:10], weights[:10]))
        row_buffer._compact()
        packed_buffer._compact()
        for index in range(10, len(values)):
            row_buffer.add((values[index], weights[index]))
            packed_buffer.add_packed_row((values, weights), index)
        bulk_buffer.add_packed_fields((values[10:], weights[10:]))
        self.assertEqual(len(row_buffer), len(packed_buffer))
        self.assertEqual(len(row_buffer), len(bulk_buffer))
        self.assertEqual(
            [int(row_buffer.memory[index][0]) for index in range(len(row_buffer))],
            [int(packed_buffer.memory[index][0]) for index in range(len(packed_buffer))],
        )
        self.assertEqual(
            [int(row_buffer.memory[index][0]) for index in range(len(row_buffer))],
            [int(bulk_buffer.memory[index][0]) for index in range(len(bulk_buffer))],
        )
        self.assertEqual(
            row_buffer.turnover_events,
            packed_buffer.turnover_events,
        )
        self.assertEqual(
            row_buffer.turnover_events,
            bulk_buffer.turnover_events,
        )
        self.assertEqual(row_buffer.seen, bulk_buffer.seen)

    def test_bulk_merge_is_exact_across_multiple_turnovers_and_wraps(self):
        capacity = 23
        count = 137
        fields = (
            torch.arange(count * 5, dtype=torch.float16).reshape(count, 5),
            torch.arange(count * NUM_ACTIONS, dtype=torch.float32).reshape(
                count, NUM_ACTIONS
            ),
            torch.ones((count, NUM_ACTIONS), dtype=torch.float32),
            torch.arange(1, count + 1, dtype=torch.float32),
        )
        row_buffer = ReservoirBuffer(capacity, random.Random(81))
        bulk_buffer = ReservoirBuffer(capacity, random.Random(81))
        for row in range(count):
            row_buffer.add(tuple(field[row] for field in fields))
        boundaries = (0, 1, 9, 31, 32, 79, 121, count)
        for start, end in zip(boundaries, boundaries[1:]):
            bulk_buffer.add_packed_fields(
                tuple(field[start:end] for field in fields)
            )

        self.assertEqual(len(row_buffer), len(bulk_buffer))
        self.assertEqual(row_buffer.seen, bulk_buffer.seen)
        self.assertEqual(
            row_buffer.turnover_events,
            bulk_buffer.turnover_events,
        )
        self.assertEqual(
            row_buffer.evicted_samples,
            bulk_buffer.evicted_samples,
        )
        for index in range(len(row_buffer)):
            for expected, actual in zip(
                row_buffer.memory[index],
                bulk_buffer.memory[index],
            ):
                self.assertTrue(torch.equal(expected, actual))
        self.assertAlmostEqual(
            row_buffer.mean_weight(),
            bulk_buffer.mean_weight(),
            places=7,
        )

    def test_traverser_regret_target_matches_counterfactual_action_values(self):
        trainer = _small_trainer()

        class State:
            terminal = False
            to_act = 0
            history = []

        class Terminal:
            terminal = True

            def __init__(self, payoff):
                self.payoffs = [payoff, -payoff]

        class TinyTree:
            bb = 2

            @staticmethod
            def legal_actions(state):
                return [0, 1]

            @staticmethod
            def step(state, action):
                return Terminal(4 if action == 0 else -2)

        strategy = torch.zeros(NUM_ACTIONS)
        strategy[0] = 0.25
        strategy[1] = 0.75
        mask = torch.zeros(NUM_ACTIONS)
        mask[:2] = 1.0
        observation = torch.zeros(trainer.input_dim)

        def fixed_strategy(state, player=None):
            return observation, strategy, mask

        trainer.env = TinyTree()
        trainer.current_strategy = fixed_strategy  # type: ignore[method-assign]
        trainer.iteration = 1
        value = trainer._traverse(State(), traverser=0)

        # Values in BB are +2 and -1, so the 25/75 node value is -0.25.
        self.assertAlmostEqual(value, -0.25, places=6)
        _, regrets, stored_mask, _ = trainer.advantage_buffers[0].memory[-1]
        self.assertAlmostEqual(float(regrets[0]), 2.25, places=6)
        self.assertAlmostEqual(float(regrets[1]), -0.75, places=6)
        self.assertTrue(torch.equal(stored_mask, mask))

    def test_policy_output_is_fixed_ten_and_strictly_legal_masked(self):
        trainer = _small_trainer()
        state = trainer.env.new_hand(button=0)
        legal = set(trainer.env.legal_actions(state))

        _, current, mask = trainer.current_strategy(state)
        average = trainer.average_policy(state)

        self.assertEqual(NUM_ACTIONS, 10)
        self.assertEqual(len(ACTION_NAMES), NUM_ACTIONS)
        self.assertEqual(tuple(current.shape), (NUM_ACTIONS,))
        self.assertEqual(tuple(average.shape), (NUM_ACTIONS,))
        self.assertAlmostEqual(float(current.sum()), 1.0, places=6)
        self.assertAlmostEqual(float(average.sum()), 1.0, places=6)
        for action in range(NUM_ACTIONS):
            self.assertEqual(bool(mask[action]), action in legal)
            if action not in legal:
                self.assertEqual(float(current[action]), 0.0)
                self.assertEqual(float(average[action]), 0.0)

    def test_one_iteration_collects_both_players_samples_and_fits(self):
        trainer = _small_trainer()
        row = trainer.train_iteration(
            traversals_per_player=1,
            advantage_steps=1,
            policy_steps=1,
            batch_size=8,
        )

        self.assertEqual(trainer.iteration, 1)
        self.assertEqual(trainer.last_traverser_schedule, (0, 1))
        self.assertEqual(trainer.last_root_buttons, (0, 1))
        self.assertTrue(all(len(buffer) > 0 for buffer in trainer.advantage_buffers))
        self.assertTrue(all(len(buffer) > 0 for buffer in trainer.policy_buffers))
        self.assertGreater(row["nodes"], 0.0)
        self.assertTrue(torch.isfinite(torch.tensor(row["advantage_loss"])))
        self.assertTrue(torch.isfinite(torch.tensor(row["policy_loss"])))
        for buffer in trainer.advantage_buffers:
            observation, target, mask, weight = buffer.memory[0]
            self.assertEqual(observation.numel(), trainer.input_dim)
            self.assertEqual(target.numel(), NUM_ACTIONS)
            self.assertEqual(mask.numel(), NUM_ACTIONS)
            self.assertEqual(float(weight), 1.0)
        for buffer in trainer.policy_buffers:
                observation, target, mask, weight, opponent_combo = (
                    buffer.memory[0]
                )
                self.assertEqual(observation.numel(), trainer.input_dim)
                self.assertEqual(target.numel(), NUM_ACTIONS)
                self.assertEqual(mask.numel(), NUM_ACTIONS)
                self.assertEqual(float(weight), 1.0)
                self.assertGreaterEqual(int(opponent_combo), 0)
                self.assertLess(int(opponent_combo), 1_326)

    def test_independent_range_pool_trains_and_reports_composition(self):
        trainer = _small_trainer(range_capacity=512)
        generated = _collect_independent_range_training_hands(
            trainer,
            profiles=("random", "calling_station", "tight_aggressive"),
            hands=8,
            seed=91_001,
            reference_policy_nets=None,
            inference_batch_size=32,
            stack_depths_bb=(10, 20),
        )
        self.assertEqual(generated["range_hands_generated"], 8.0)
        self.assertGreater(generated["range_samples_generated"], 0.0)
        self.assertTrue(all(len(buffer) > 0 for buffer in trainer.range_buffers))
        row = trainer.train_iteration(
            traversals_per_player=1,
            advantage_steps=1,
            policy_steps=1,
            batch_size=8,
            range_batch_size=4,
        )
        self.assertTrue(torch.isfinite(torch.tensor(row["policy_range_loss"])))
        stats = range_reservoir_statistics(
            trainer,
            maximum_rows_per_player=512,
        )
        self.assertEqual(stats["total_rows"], int(row["range_samples"]))
        self.assertAlmostEqual(
            sum(stats["street_percent"].values()),
            100.0,
            places=4,
        )
        self.assertAlmostEqual(
            sum(sum(row) for row in stats["starting_hand_matrix_percent"]),
            100.0,
            places=4,
        )

    @unittest.skipUnless(
        torch.cuda.is_available(),
        "CUDA stream concurrency requires an NVIDIA GPU",
    )
    def test_cuda_fits_both_players_on_independent_streams(self):
        trainer = _small_trainer(device="cuda:0")
        before = [
            {
                name: value.detach().cpu().clone()
                for name, value in network.state_dict().items()
            }
            for network in trainer.policy_nets
        ]
        row = trainer.train_iteration(
            traversals_per_player=1,
            advantage_steps=2,
            policy_steps=2,
            batch_size=8,
        )
        self.assertEqual(row["parallel_player_fitting"], 1.0)
        self.assertTrue(torch.isfinite(torch.tensor(row["advantage_loss"])))
        self.assertTrue(torch.isfinite(torch.tensor(row["policy_loss"])))
        for player, network in enumerate(trainer.policy_nets):
            self.assertTrue(
                any(
                    not torch.equal(before[player][name], value.detach().cpu())
                    for name, value in network.state_dict().items()
                )
            )

    def test_advantage_reinitialization_uses_configured_cycle(self):
        trainer = _small_trainer(
            reinitialize_advantage_each_iteration=True,
            advantage_reinitialize_from_iteration=25,
            advantage_reinitialize_cycle=25,
        )
        trainer.advantage_buffers[0].add(
            (
                torch.zeros(trainer.input_dim, dtype=torch.float16),
                torch.zeros(NUM_ACTIONS),
                torch.ones(NUM_ACTIONS),
                torch.tensor(1.0),
            )
        )

        original = trainer.advantage_nets[0]
        trainer.iteration = 24
        trainer._fit_advantage(0, steps=1, batch_size=1)
        self.assertIs(trainer.advantage_nets[0], original)

        trainer.iteration = 25
        trainer._fit_advantage(0, steps=1, batch_size=1)
        self.assertIsNot(trainer.advantage_nets[0], original)
        cycle_network = trainer.advantage_nets[0]

        for iteration in (26, 49):
            trainer.iteration = iteration
            trainer._fit_advantage(0, steps=1, batch_size=1)
            self.assertIs(trainer.advantage_nets[0], cycle_network)

        trainer.iteration = 50
        trainer._fit_advantage(0, steps=1, batch_size=1)
        self.assertIsNot(trainer.advantage_nets[0], cycle_network)

    def test_parallel_traversal_workers_collect_both_players(self):
        trainer = _small_trainer()
        row = trainer.train_iteration(
            traversals_per_player=1,
            advantage_steps=1,
            policy_steps=1,
            batch_size=8,
            traversal_workers=2,
            root_stack_distribution=ROOT_STACK_DISTRIBUTION_MIXED,
            root_stack_depths_bb=DEFAULT_ROOT_STACK_DEPTHS_BB,
        )

        self.assertEqual(row["traversal_workers"], 2.0)
        self.assertGreater(row["traversal_nodes_per_second"], 0.0)
        self.assertTrue(all(len(buffer) > 0 for buffer in trainer.advantage_buffers))
        self.assertTrue(all(len(buffer) > 0 for buffer in trainer.policy_buffers))
        self.assertEqual(trainer.last_traverser_schedule, (0, 1))
        self.assertEqual(trainer.last_root_buttons, (0, 1))
        self.assertEqual(row["root_equal_stack_fraction"], 0.5)
        self.assertGreaterEqual(row["root_min_effective_stack_bb"], 10.0)
        self.assertLessEqual(row["root_max_effective_stack_bb"], 100.0)

    @unittest.skipUnless(
        os.getenv("HU_HIGH_WORKER_SMOKE") == "1",
        "set HU_HIGH_WORKER_SMOKE=1 on the target training host",
    )
    def test_twenty_four_worker_descriptor_smoke(self):
        trainer = _small_trainer()
        row = trainer.train_iteration(
            traversals_per_player=12,
            advantage_steps=1,
            policy_steps=1,
            batch_size=8,
            traversal_workers=24,
            root_stack_distribution=ROOT_STACK_DISTRIBUTION_MIXED,
            root_stack_depths_bb=DEFAULT_ROOT_STACK_DEPTHS_BB,
        )
        self.assertEqual(row["traversal_workers"], 24.0)
        self.assertTrue(all(len(buffer) > 0 for buffer in trainer.advantage_buffers))
        self.assertTrue(all(len(buffer) > 0 for buffer in trainer.policy_buffers))

    def test_native_parallel_workers_accept_mixed_root_stacks(self):
        try:
            from heads_up_native import HeadsUpHoldemEngine as NativeEngine
        except ImportError as error:
            self.skipTest(str(error))
        trainer = _small_trainer(engine_type=NativeEngine)
        row = trainer.train_iteration(
            traversals_per_player=1,
            advantage_steps=1,
            policy_steps=1,
            batch_size=8,
            traversal_workers=2,
            root_stack_distribution=ROOT_STACK_DISTRIBUTION_MIXED,
            root_stack_depths_bb=DEFAULT_ROOT_STACK_DEPTHS_BB,
        )
        self.assertEqual(row["root_equal_stack_fraction"], 0.5)
        self.assertTrue(
            all(
                10 <= min(stacks) / trainer.env.bb <= 100
                for stacks in trainer.last_root_stacks
            )
        )

    def test_batched_frontier_strategy_matches_row_strategy(self):
        trainer = _small_trainer()
        states = [
            trainer.env.new_hand(button=0),
            trainer.env.new_hand(button=1),
        ]
        batched = trainer._batched_current_strategies(
            [(state, int(state.to_act)) for state in states]
        )
        rows = [
            trainer.current_strategy(state, int(state.to_act))
            for state in states
        ]
        for batch_row, scalar_row in zip(batched, rows):
            for batch_value, scalar_value in zip(batch_row, scalar_row):
                self.assertTrue(torch.allclose(batch_value, scalar_value))

    def test_tag_benchmark_is_legal_and_position_balanced(self):
        trainer = _small_trainer()
        state = trainer.env.new_hand(button=0)
        probabilities = TightAggressiveOpponent().probabilities(
            trainer.env,
            state,
            int(state.to_act),
        )
        self.assertEqual(tuple(probabilities.shape), (NUM_ACTIONS,))
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=6)
        for action in range(NUM_ACTIONS):
            if action not in trainer.env.legal_actions(state):
                self.assertEqual(float(probabilities[action]), 0.0)

        metrics = evaluate_benchmark_suite(
            trainer,
            profiles=("tight_aggressive",),
            games_per_seat=2,
            seed=909,
            inference_batch_size=4,
            baseline_policy_nets=trainer.policy_nets,
        )
        self.assertIn("benchmark_tight_aggressive_mean_ev_bb", metrics)
        self.assertIn("benchmark_tight_aggressive_ev_BTN_SB_bb", metrics)
        self.assertIn("benchmark_tight_aggressive_ev_BB_bb", metrics)
        self.assertIn("benchmark_composite_lcb95_bb", metrics)
        self.assertIn("benchmark_tight_aggressive_delta_ev_bb", metrics)
        self.assertIn(
            "benchmark_tight_aggressive_probability_delta_positive",
            metrics,
        )

    def test_reference_policy_is_part_of_benchmark_suite_and_campaign(self):
        trainer = _small_trainer()
        reference = _small_trainer(hidden=12)
        reference.iteration = 1_025
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference_path = save_policy_snapshot(
                reference,
                root / "reference_policy.pt",
            )
            metrics = evaluate_benchmark_suite(
                trainer,
                profiles=("random",),
                games_per_seat=2,
                seed=910,
                inference_batch_size=4,
                baseline_policy_nets=trainer.policy_nets,
                reference_policy_nets=reference.policy_nets,
            )
            self.assertIn("benchmark_reference_policy_mean_ev_bb", metrics)
            self.assertIn("benchmark_reference_policy_delta_ev_bb", metrics)

            config = CampaignConfig(
                target_iteration=1,
                traversals_per_player=1,
                traversal_workers=1,
                advantage_steps=1,
                policy_steps=1,
                batch_size=8,
                evaluate_every=1,
                checkpoint_every=1,
                snapshot_every=1,
                evaluation_games_per_player=2,
                range_training_hands_per_iteration=8,
                range_batch_size=4,
                league_games_per_player=1,
                validation_seed=910,
                opponent_profiles=("random",),
                reference_policy_path=str(reference_path),
                league_opponents=0,
                keep_full_checkpoints=1,
            )
            artifact_dir = root / "campaign"
            campaign = ProductionCampaign(trainer, artifact_dir, config)
            self.assertEqual(campaign.reference_policy.iteration, 1_025)
            campaign.run()
            row = json.loads(
                (artifact_dir / "metrics.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[-1]
            )
            self.assertEqual(
                int(row["benchmark_reference_policy_iteration"]),
                1_025,
            )
            self.assertIn("range_nll", row)
            self.assertIn("range_vs_reference_policy_information_gain", row)
            self.assertTrue(
                (
                    artifact_dir
                    / "evaluations"
                    / "step_00000001"
                    / "reference_policy_hands.csv"
                ).is_file()
            )

    def test_traverser_and_button_roots_are_balanced_per_iteration(self):
        trainer = _small_trainer()
        observed: list[tuple[int, int]] = []

        def record_only(state, traverser, depth=0):
            observed.append((int(traverser), int(state.button)))
            return 0.0

        trainer._traverse = record_only  # type: ignore[method-assign]
        trainer._collect_traversals(4)
        for traverser in range(NUM_PLAYERS):
            for button in range(NUM_PLAYERS):
                self.assertEqual(observed.count((traverser, button)), 2)

        observed.clear()
        trainer._collect_traversals(3)
        self.assertEqual(sum(pair[0] == 0 for pair in observed), 3)
        self.assertEqual(sum(pair[0] == 1 for pair in observed), 3)
        self.assertEqual(sum(pair[1] == 0 for pair in observed), 3)
        self.assertEqual(sum(pair[1] == 1 for pair in observed), 3)
        self.assertEqual(
            sorted(observed.count((traverser, button))
                   for traverser in range(NUM_PLAYERS)
                   for button in range(NUM_PLAYERS)),
            [1, 1, 2, 2],
        )

    def test_mixed_stack_roots_preserve_traverser_and_button_balance(self):
        trainer = _small_trainer()
        observed: list[tuple[int, int, tuple[int, int]]] = []

        def record_only(state, traverser, depth=0):
            observed.append(
                (
                    int(traverser),
                    int(state.button),
                    tuple(int(value) for value in state.initial_stacks),
                )
            )
            return 0.0

        trainer._traverse = record_only  # type: ignore[method-assign]
        trainer._collect_traversals(
            4,
            root_stack_distribution=ROOT_STACK_DISTRIBUTION_MIXED,
            root_stack_depths_bb=DEFAULT_ROOT_STACK_DEPTHS_BB,
        )
        self.assertEqual(sum(stacks[0] == stacks[1] for _, _, stacks in observed), 4)
        self.assertEqual(sum(stacks[0] != stacks[1] for _, _, stacks in observed), 4)
        for traverser in range(NUM_PLAYERS):
            for button in range(NUM_PLAYERS):
                self.assertEqual(
                    sum(
                        observed_traverser == traverser
                        and observed_button == button
                        for observed_traverser, observed_button, _ in observed
                    ),
                    2,
                )

    def test_training_uses_configured_recent_history_window(self):
        trainer = _small_trainer(max_history=1)
        state = trainer.env.new_hand(button=0)
        state.history = [object(), object()]
        with mock.patch(
            "heads_up_cfr.encode_information_state",
            return_value=torch.zeros(trainer.input_dim),
        ) as encode:
            result = trainer.encode(state, int(state.to_act))
        self.assertEqual(result.shape, (trainer.input_dim,))
        self.assertEqual(encode.call_args.args[4], 1)

    def test_checkpoint_locks_schema_and_resumes_full_training(self):
        trainer = _small_trainer()
        trainer.train_iteration(
            traversals_per_player=1,
            advantage_steps=1,
            policy_steps=1,
            batch_size=8,
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "hu.pt"
            trainer.save(checkpoint_path)
            payload = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )

            self.assertEqual(payload["kind"], CHECKPOINT_KIND)
            self.assertEqual(
                payload["network_architecture"], NETWORK_ARCHITECTURE
            )
            self.assertEqual(payload["engine_schema_version"], ENGINE_SCHEMA_VERSION)
            self.assertEqual(payload["action_schema_version"], ACTION_SCHEMA_VERSION)
            self.assertEqual(
                payload["encoder"]["encoder_schema_version"],
                ENCODER_SCHEMA_VERSION,
            )
            self.assertEqual(payload["encoder"], encoder_metadata(32))
            self.assertEqual(tuple(payload["action_names"]), tuple(ACTION_NAMES))
            self.assertEqual(payload["num_actions"], NUM_ACTIONS)

            resumed_env = HeadsUpHoldemEngine(
                starting_stack=8,
                small_blind=1,
                big_blind=2,
                seed=999,
            )
            resumed = HeadsUpNeuralCFR.load(checkpoint_path, resumed_env)
            self.assertTrue(resumed.can_resume_training)
            self.assertEqual(resumed.iteration, trainer.iteration)
            self.assertEqual(
                [len(buffer) for buffer in resumed.advantage_buffers],
                [len(buffer) for buffer in trainer.advantage_buffers],
            )
            self.assertEqual(
                [len(buffer) for buffer in resumed.policy_buffers],
                [len(buffer) for buffer in trainer.policy_buffers],
            )
            expected_stacks = trainer._root_stack_schedule(
                20,
                ROOT_STACK_DISTRIBUTION_MIXED,
                DEFAULT_ROOT_STACK_DEPTHS_BB,
            )
            resumed_stacks = resumed._root_stack_schedule(
                20,
                ROOT_STACK_DISTRIBUTION_MIXED,
                DEFAULT_ROOT_STACK_DEPTHS_BB,
            )
            self.assertEqual(resumed_stacks, expected_stacks)
            policy = HeadsUpNetworkPolicy(
                resumed.policy_nets,
                max_history=resumed.max_history,
                checkpoint_encoder=payload["encoder"],
            )
            decision = resumed.env.new_hand(button=0)
            probabilities = policy.probabilities(resumed.env, decision)
            self.assertEqual(tuple(probabilities.shape), (NUM_ACTIONS,))
            self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=6)
            row = resumed.train_iteration(
                traversals_per_player=1,
                advantage_steps=1,
                policy_steps=1,
                batch_size=8,
            )
            self.assertEqual(resumed.iteration, 2)
            self.assertEqual(int(row["iteration"]), 2)
            for key in (
                "advantage_fit_seconds",
                "policy_fit_seconds",
                "adv_loss_p0",
                "adv_loss_p1",
                "policy_loss_p0",
                "policy_loss_p1",
                "adv_buffer_p0",
                "adv_buffer_p1",
                "policy_buffer_p0",
                "policy_buffer_p1",
            ):
                self.assertIn(key, row)

    def test_production_campaign_writes_resumable_parity_artifacts(self):
        trainer = _small_trainer()
        config = CampaignConfig(
            target_iteration=1,
            traversals_per_player=1,
            traversal_workers=1,
            advantage_steps=1,
            policy_steps=1,
            batch_size=8,
            evaluate_every=1,
            checkpoint_every=1,
            snapshot_every=1,
            evaluation_games_per_player=2,
            range_training_hands_per_iteration=8,
            range_batch_size=4,
            league_games_per_player=2,
            validation_seed=1234,
            opponent_profiles=("random",),
            league_opponents=0,
            keep_full_checkpoints=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            artifact_dir = Path(directory)
            campaign = ProductionCampaign(trainer, artifact_dir, config)
            campaign.run()

            self.assertTrue((artifact_dir / "latest.json").is_file())
            self.assertTrue((artifact_dir / "metrics.jsonl").is_file())
            self.assertTrue((artifact_dir / "run_config.json").is_file())
            self.assertTrue((artifact_dir / "snapshots" / "initial_policy.pt").is_file())
            self.assertTrue(
                (artifact_dir / "snapshots" / "champion_policy.pt").is_file()
            )
            self.assertTrue(
                (
                    artifact_dir
                    / "evaluations"
                    / "step_00000001"
                    / "random_hands.csv"
                ).is_file()
            )
            self.assertEqual(
                len(list((artifact_dir / "checkpoints").glob("*.pt"))),
                1,
            )
            self.assertEqual(trainer.last_fitted_iteration, 1)
            metric = json.loads(
                (artifact_dir / "metrics.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[-1]
            )
            self.assertIn("range_nll", metric)
            self.assertTrue(
                (
                    artifact_dir
                    / "evaluations"
                    / "fixed_range_holdout_v1.pt"
                ).is_file()
            )

            resumed, did_resume = load_or_create_trainer(
                trainer.env,
                artifact_dir,
                device="cpu",
                trainer_kwargs={
                    "advantage_capacity": 512,
                    "policy_capacity": 512,
                },
            )
            self.assertTrue(did_resume)
            self.assertEqual(resumed.iteration, 1)
            self.assertEqual(resumed.last_fitted_iteration, 1)
            self.assertEqual(
                [buffer.capacity for buffer in resumed.advantage_buffers],
                [512, 512],
            )
            self.assertEqual(
                [buffer.capacity for buffer in resumed.policy_buffers],
                [512, 512],
            )
            ProductionCampaign(
                resumed,
                artifact_dir,
                replace(config, target_iteration=2),
            )
            self.assertTrue((artifact_dir / "run_config_history.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
