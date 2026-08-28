import random
import tempfile
import unittest
from pathlib import Path

import torch

from three_player_engine import ThreePlayerHoldemEnv as PythonEnv
from three_player_engine import evaluate_5card as python_evaluate_5card
from three_player_engine import evaluate_7card as python_evaluate_7card
from three_player_native import ThreePlayerHoldemEnv as NativeEnv
from three_player_native import evaluate_5card as native_evaluate_5card
from three_player_native import evaluate_7card as native_evaluate_7card
from three_player_native import encode_information_state_native
from three_player_cfr import ThreePlayerNeuralCFR
from three_player_models import encode_information_state
from three_player_models import CARD_FEATURES, poker_relational_features


def state_view(state):
    return {
        "deck": list(state.deck), "board": list(state.board),
        "hole": [list(cards) for cards in state.hole],
        "stacks": list(state.stacks), "initial_stacks": list(state.initial_stacks),
        "total_contrib": list(state.total_contrib),
        "street_contrib": list(state.street_contrib),
        "folded": list(state.folded), "all_in": list(state.all_in),
        "pot": state.pot, "current_bet": state.current_bet,
        "min_raise": state.min_raise, "to_act": state.to_act,
        "street": state.street, "button": state.button,
        "sb_player": state.sb_player, "bb_player": state.bb_player,
        "pending": set(state.pending_actors),
        "raise_rights": list(state.raise_rights),
        "last_action_bet": list(state.last_action_bet),
        "last_full_raiser": state.last_full_raiser,
        "burned": list(state.burned), "alive": list(state.alive),
        "eliminated": list(state.eliminated), "terminal": state.terminal,
        "payoffs": None if state.payoffs is None else list(state.payoffs),
        "payouts": None if state.payouts is None else list(state.payouts),
        "winners": tuple(state.winners),
        "history": [
            (r.player, r.street, r.action, r.amount, r.contribution_after,
             r.current_bet_before, r.current_bet_after, r.pot_after, r.full_raise)
            for r in state.history
        ],
    }


class NativeDifferentialTests(unittest.TestCase):
    def test_native_relational_features_are_bit_exact(self):
        chooser = random.Random(73_991)
        rows = []
        streets = []
        board_sizes = (0, 3, 4, 5)
        for index in range(20_000):
            dealt = chooser.sample(range(52), 7)
            board_size = board_sizes[index % len(board_sizes)]
            cards = torch.zeros((7, CARD_FEATURES), dtype=torch.float32)
            for slot, card in enumerate(dealt[: 2 + board_size]):
                token = slot if slot < 2 else slot
                cards[token, card % 13] = 1.0
                cards[token, 13 + card // 13] = 1.0
                cards[token, 17] = 1.0
            street = torch.zeros(4, dtype=torch.float32)
            street[(0, 1, 2, 3)[board_sizes.index(board_size)]] = 1.0
            rows.append(cards)
            streets.append(street)
        cards = torch.stack(rows)
        street_one_hot = torch.stack(streets)
        reference = poker_relational_features(
            cards, street_one_hot, use_native=False
        )
        candidate = poker_relational_features(cards, street_one_hot)
        self.assertTrue(torch.equal(reference, candidate))

    def test_native_information_encoder_is_bit_exact(self):
        chooser = random.Random(91_827)
        stack_modes = (
            [200.0, 200.0, 200.0],
            [0.0, 241.0, 359.0],
            [397.0, 2.0, 201.0],
            [75.5, 410.25, 114.25],
        )
        checked = 0
        for hand in range(100):
            deck = list(range(52))
            chooser.shuffle(deck)
            stacks = stack_modes[hand % len(stack_modes)]
            live = [seat for seat, value in enumerate(stacks) if value > 0]
            button = live[hand % len(live)]
            py_env, native_env = PythonEnv(seed=1), NativeEnv(seed=1)
            py_state = py_env.new_hand(button=button, stacks=stacks, deck=deck)
            native_state = native_env.new_hand(button=button, stacks=stacks, deck=deck)
            while not py_state.terminal:
                legal = py_env.legal_actions(py_state)
                self.assertEqual(native_env.legal_actions(native_state), legal)
                reference = encode_information_state(
                    py_state,
                    py_state.to_act,
                    legal,
                    200.0,
                    32,
                    include_tournament_features=True,
                    tournament_total_chips=600.0,
                )
                candidate = torch.from_numpy(
                    encode_information_state_native(
                        native_state,
                        native_state.to_act,
                        legal,
                        200.0,
                        32,
                        include_tournament_features=True,
                        tournament_total_chips=600.0,
                    )
                )
                self.assertTrue(torch.equal(reference, candidate))
                checked += 1
                action = chooser.choice(legal)
                py_state = py_env.step(py_state, action)
                native_state = native_env.step(native_state, action)
        self.assertGreater(checked, 300)

    def test_evaluators_match_reference(self):
        rng = random.Random(991)
        for _ in range(20_000):
            seven = rng.sample(range(52), 7)
            self.assertEqual(native_evaluate_5card(seven[:5]), python_evaluate_5card(seven[:5]))
            self.assertEqual(native_evaluate_7card(seven), python_evaluate_7card(seven))

    def test_random_legal_games_match_after_every_action(self):
        chooser = random.Random(8128)
        for hand in range(500):
            deck = list(range(52))
            random.Random(40_000 + hand).shuffle(deck)
            stacks = chooser.choice(([200, 200, 200], [350, 0, 250], [20, 100, 480]))
            live = [i for i, value in enumerate(stacks) if value > 0]
            button = live[hand % len(live)]
            py_env, native_env = PythonEnv(seed=1), NativeEnv(seed=1)
            py_state = py_env.new_hand(button=button, stacks=stacks, deck=deck)
            native_state = native_env.new_hand(button=button, stacks=stacks, deck=deck)
            self.assertEqual(state_view(native_state), state_view(py_state))
            while not py_state.terminal:
                py_legal = py_env.legal_actions(py_state)
                self.assertEqual(native_env.legal_actions(native_state), py_legal)
                action = chooser.choice(py_legal)
                py_state = py_env.step(py_state, action)
                native_state = native_env.step(native_state, action)
                self.assertEqual(state_view(native_state), state_view(py_state))

    def test_native_backend_parallel_training_smoke(self):
        env = NativeEnv(stack_size=20, sb=1, bb=2, seed=77)
        trainer = ThreePlayerNeuralCFR(
            env,
            hidden=16,
            blocks=1,
            max_history=4,
            max_nodes_per_traversal=8,
            max_depth=3,
            reinitialize_advantage_each_iteration=False,
            seed=77,
        )
        row = trainer.train_iteration(
            traversals_per_player=2,
            traversal_workers=2,
            advantage_steps=1,
            policy_steps=1,
            batch_size=4,
        )
        self.assertEqual(row["traversal_workers"], 2.0)
        self.assertGreater(row["nodes"], 0.0)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "native_roundtrip.pt"
            trainer.save(checkpoint)
            restored = ThreePlayerNeuralCFR.load(checkpoint, env, device="cpu")
            self.assertEqual(restored.iteration, trainer.iteration)
            self.assertTrue(getattr(restored.env, "native_backend", False))

    def test_batched_frontiers_preserve_traversal_work_and_targets(self):
        def trainer():
            environment = NativeEnv(stack_size=20, sb=1, bb=2, seed=91)
            return environment, ThreePlayerNeuralCFR(
                environment,
                hidden=16,
                blocks=1,
                max_history=4,
                max_nodes_per_traversal=12,
                max_depth=4,
                reinitialize_advantage_each_iteration=False,
                seed=91,
            )

        sequential_env, sequential = trainer()
        _, batched = trainer()
        for target, source in zip(batched.advantage_nets, sequential.advantage_nets):
            target.load_state_dict(source.state_dict())
        roots = []
        for index in range(6):
            deck = list(range(52))
            random.Random(10_000 + index).shuffle(deck)
            roots.append(
                (
                    sequential_env.new_hand(button=index % 3, deck=deck),
                    index % 3,
                    20_000 + index,
                )
            )
        for state, traverser, seed in roots:
            sequential.rng = random.Random(seed)
            sequential._nodes_this_traversal = 0
            sequential._traverse(state, traverser, [1.0, 1.0, 1.0], 0)
        batched._run_batched_traversals(
            [
                {
                    "state": state,
                    "traverser": traverser,
                    "rng": random.Random(seed),
                }
                for state, traverser, seed in roots
            ]
        )
        self.assertEqual(batched._nodes_this_iteration, sequential._nodes_this_iteration)
        self.assertEqual(batched._rollouts_this_iteration, sequential._rollouts_this_iteration)
        self.assertEqual(
            [buffer.seen for buffer in batched.advantage_buffers],
            [buffer.seen for buffer in sequential.advantage_buffers],
        )
        self.assertEqual(
            [buffer.seen for buffer in batched.policy_buffers],
            [buffer.seen for buffer in sequential.policy_buffers],
        )
        for batched_buffer, sequential_buffer in zip(
            batched.advantage_buffers, sequential.advantage_buffers
        ):
            self.assertAlmostEqual(
                sum(float(item[1].sum()) for item in batched_buffer.memory),
                sum(float(item[1].sum()) for item in sequential_buffer.memory),
                places=4,
            )

    def test_vectorized_frontiers_match_scalar_reference_exactly(self):
        def make_trainer():
            environment = NativeEnv(stack_size=40, sb=1, bb=2, seed=119)
            instance = ThreePlayerNeuralCFR(
                environment,
                hidden=32,
                blocks=2,
                network_architecture="deep_cfr_branch_v2",
                max_history=8,
                max_nodes_per_traversal=40,
                max_depth=8,
                exploration=0.15,
                include_tournament_features=True,
                variable_stack_training=True,
                tournament_total_chips=120,
                advantage_capacity=20_000,
                policy_capacity=20_000,
                reinitialize_advantage_each_iteration=False,
                seed=119,
                _traversal_worker=True,
            )
            return environment, instance

        environment, vectorized = make_trainer()
        _, scalar = make_trainer()
        generator = torch.Generator().manual_seed(83_017)
        with torch.no_grad():
            for network in vectorized.advantage_nets:
                for head in network.street_heads:
                    head.weight.normal_(mean=0.0, std=0.08, generator=generator)
                    head.bias.normal_(mean=0.0, std=0.08, generator=generator)
        for target, source in zip(scalar.advantage_nets, vectorized.advantage_nets):
            target.load_state_dict(source.state_dict())

        original_scalar = scalar._batched_current_strategies
        scalar._batched_current_strategies = lambda requests: original_scalar(
            requests, use_vectorized=False
        )
        vectorized_contexts, scalar_contexts = [], []
        for index in range(72):
            deck = list(range(52))
            random.Random(91_000 + index).shuffle(deck)
            stacks = ([40.0, 40.0, 40.0], [0.0, 55.0, 65.0])[index % 2]
            live = [seat for seat, value in enumerate(stacks) if value > 0]
            traverser = live[index % len(live)]
            button = live[(index // len(live)) % len(live)]
            state = environment.new_hand(button=button, stacks=stacks, deck=deck)
            vectorized_contexts.append(
                {
                    "state": state,
                    "traverser": traverser,
                    "rng": random.Random(120_000 + index),
                }
            )
            scalar_contexts.append(
                {
                    "state": state,
                    "traverser": traverser,
                    "rng": random.Random(120_000 + index),
                }
            )

        vectorized._run_batched_traversals(vectorized_contexts)
        scalar._run_batched_traversals(scalar_contexts)
        self.assertEqual(vectorized._nodes_this_iteration, scalar._nodes_this_iteration)
        self.assertEqual(vectorized._rollouts_this_iteration, scalar._rollouts_this_iteration)
        self.assertEqual(vectorized._regret_magnitudes, scalar._regret_magnitudes)
        self.assertEqual(vectorized._strategy_weights, scalar._strategy_weights)
        self.assertEqual(vectorized._policy_entropies, scalar._policy_entropies)
        self.assertEqual(vectorized._raw_strategy_importances, scalar._raw_strategy_importances)
        self.assertEqual(vectorized.rng.getstate(), scalar.rng.getstate())
        for vector_buffer, scalar_buffer in zip(
            vectorized.advantage_buffers + vectorized.policy_buffers,
            scalar.advantage_buffers + scalar.policy_buffers,
        ):
            vector_state = vector_buffer.state_dict()
            scalar_state = scalar_buffer.state_dict()
            self.assertEqual(vector_state["seen"], scalar_state["seen"])
            self.assertEqual(len(vector_state["fields"]), len(scalar_state["fields"]))
            for vector_field, scalar_field in zip(
                vector_state["fields"], scalar_state["fields"]
            ):
                self.assertTrue(torch.equal(vector_field, scalar_field))


if __name__ == "__main__":
    unittest.main()
