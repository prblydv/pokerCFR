import copy
import unittest
from unittest.mock import patch

import torch

from heads_up_engine import (
    ACTION_MIN_RAISE,
    ACTION_NAMES,
    ACTION_SCHEMA_VERSION,
    ENGINE_SCHEMA_VERSION,
    NUM_ACTIONS,
    HeadsUpHoldemEnv,
)
from heads_up_models import (
    ACTION_DESCRIPTOR_FEATURES,
    ACTION_DESCRIPTOR_FEATURE_NAMES,
    DEFAULT_MAX_HISTORY,
    ENCODER_SCHEMA_VERSION,
    ENCODER_VERSION,
    HISTORY_FEATURE_NAMES,
    HISTORY_FEATURES,
    HISTORY_OFFSET,
    AdvantageNetwork,
    HeadsUpDeepCFRCompactV4Network,
    HeadsUpDeepCFRCompactV4PolicyRangeNetwork,
    POLICY_RANGE_AUX_ARCHITECTURE,
    PolicyNetwork,
    action_descriptor_offset,
    build_action_descriptors,
    build_policy_network,
    encode_information_state,
    encoder_metadata,
    information_state_size,
    legal_mask_offset,
    masked_softmax,
)
from heads_up_ranges import (
    NUM_OPPONENT_COMBOS,
    opponent_combo_index,
    valid_combo_mask_from_encoded,
)
from play_heads_up_gui import (
    HeadsUpManualGUI,
    event_summary,
    fixed_action_label,
    parse_args,
    state_facts,
)


class HeadsUpModelSchemaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = HeadsUpHoldemEnv(
            starting_stack=100,
            small_blind=1,
            big_blind=2,
            seed=71,
        )

    @staticmethod
    def _bb(env) -> float:
        return float(getattr(env, "big_blind", getattr(env, "bb", 0.0)))

    def _encode(self, env, state, hero=None, max_history=DEFAULT_MAX_HISTORY):
        if hero is None:
            hero = int(state.to_act)
        legal = env.legal_actions(state)
        descriptors = build_action_descriptors(env, state)
        return encode_information_state(
            state,
            hero,
            legal,
            self._bb(env),
            max_history,
            action_descriptors=descriptors,
        )

    def test_encoder_width_offsets_and_network_output_contract(self) -> None:
        state = self.env.new_hand(button=0)
        encoded = self._encode(self.env, state)
        self.assertEqual(encoded.shape, (information_state_size(),))
        self.assertEqual(
            action_descriptor_offset(),
            legal_mask_offset() + NUM_ACTIONS,
        )
        self.assertEqual(
            information_state_size(),
            action_descriptor_offset()
            + NUM_ACTIONS * ACTION_DESCRIPTOR_FEATURES,
        )

        for network_type in (AdvantageNetwork, PolicyNetwork):
            network = network_type(encoded.numel(), hidden=32, blocks=1)
            output = network(encoded.unsqueeze(0))
            self.assertEqual(output.shape, (1, NUM_ACTIONS))

    def test_encoder_metadata_locks_engine_actions_and_width(self) -> None:
        self.assertEqual(DEFAULT_MAX_HISTORY, 32)
        self.assertEqual(information_state_size(), 1_038)
        metadata = encoder_metadata(max_history=17)
        self.assertEqual(metadata["engine_schema_version"], ENGINE_SCHEMA_VERSION)
        self.assertEqual(metadata["action_schema_version"], ACTION_SCHEMA_VERSION)
        self.assertEqual(metadata["encoder_version"], ENCODER_VERSION)
        self.assertEqual(
            metadata["encoder_schema_version"],
            ENCODER_SCHEMA_VERSION,
        )
        self.assertEqual(metadata["width"], information_state_size(17))
        self.assertEqual(metadata["input_dim"], information_state_size(17))
        self.assertEqual(metadata["max_history"], 17)
        self.assertEqual(metadata["num_actions"], NUM_ACTIONS)
        self.assertEqual(metadata["action_names"], tuple(ACTION_NAMES))
        self.assertEqual(
            metadata["history_feature_names"],
            HISTORY_FEATURE_NAMES,
        )
        self.assertEqual(
            metadata["action_descriptor_feature_names"],
            ACTION_DESCRIPTOR_FEATURE_NAMES,
        )

    def test_compact_structured_network_size(self) -> None:
        network = build_policy_network(
            "hu_deep_cfr_compact_v4",
            information_state_size(),
            hidden=128,
            blocks=2,
        )
        self.assertIsInstance(network, HeadsUpDeepCFRCompactV4Network)
        parameters = sum(parameter.numel() for parameter in network.parameters())
        self.assertLessEqual(parameters, 3 * 98_948)
        self.assertEqual(parameters, 294_736)

    def test_hidden384_policy_has_action_and_exact_range_heads(self) -> None:
        network = build_policy_network(
            POLICY_RANGE_AUX_ARCHITECTURE,
            information_state_size(),
            hidden=384,
            blocks=2,
        )
        self.assertIsInstance(
            network,
            HeadsUpDeepCFRCompactV4PolicyRangeNetwork,
        )
        parameters = sum(parameter.numel() for parameter in network.parameters())
        self.assertEqual(parameters, 2_353_022)
        state = self.env.new_hand(button=0)
        encoded = self._encode(self.env, state).unsqueeze(0)
        action_logits, range_logits = network.forward_with_range(encoded)
        self.assertEqual(action_logits.shape, (1, NUM_ACTIONS))
        self.assertEqual(range_logits.shape, (1, NUM_OPPONENT_COMBOS))
        self.assertTrue(torch.equal(network(encoded), action_logits))
        valid = valid_combo_mask_from_encoded(encoded)
        self.assertEqual(valid.shape, (1, NUM_OPPONENT_COMBOS))
        target = opponent_combo_index(state.hole[1 - int(state.to_act)])
        self.assertTrue(bool(valid[0, target]))

    def test_dual_head_losses_route_only_through_their_own_head(self) -> None:
        network = build_policy_network(
            POLICY_RANGE_AUX_ARCHITECTURE,
            information_state_size(),
            hidden=32,
            blocks=1,
        )
        with torch.no_grad():
            network.range_head.weight.normal_(std=0.02)
            for head in network.street_heads:
                head.weight.normal_(std=0.02)
        state = self.env.new_hand(button=0)
        encoded = self._encode(self.env, state).unsqueeze(0)

        action_logits, _ = network.forward_with_range(encoded)
        action_logits.sum().backward()
        self.assertIsNone(network.range_head.weight.grad)
        self.assertIsNotNone(network.street_heads[0].weight.grad)
        self.assertTrue(
            any(
                parameter.grad is not None
                and bool(torch.any(parameter.grad != 0))
                for parameter in network.backbone.parameters()
            )
        )

        network.zero_grad(set_to_none=True)
        _, range_logits = network.forward_with_range(encoded)
        range_logits[:, :7].sum().backward()
        self.assertIsNone(network.street_heads[0].weight.grad)
        self.assertIsNotNone(network.range_head.weight.grad)
        self.assertTrue(
            any(
                parameter.grad is not None
                and bool(torch.any(parameter.grad != 0))
                for parameter in network.backbone.parameters()
            )
        )

    def test_encoder_never_reads_opponent_hole_cards(self) -> None:
        state = self.env.new_hand(button=0)
        original = self._encode(self.env, state)
        changed = copy.deepcopy(state)
        hero = int(state.to_act)
        changed.hole[1 - hero] = [50, 51]
        modified = self._encode(self.env, changed)
        self.assertTrue(torch.equal(original, modified))

    def test_legal_mask_and_exact_action_descriptors_match_engine(self) -> None:
        state = self.env.new_hand(button=0)
        legal = self.env.legal_actions(state)
        descriptors = build_action_descriptors(self.env, state)
        encoded = encode_information_state(
            state,
            int(state.to_act),
            legal,
            self._bb(self.env),
            action_descriptors=descriptors,
        )
        mask = encoded[legal_mask_offset() : action_descriptor_offset()]
        self.assertEqual(
            mask.tolist(),
            [float(action in legal) for action in range(NUM_ACTIONS)],
        )

        fields = encoded[action_descriptor_offset() :].reshape(
            NUM_ACTIONS, ACTION_DESCRIPTOR_FEATURES
        )
        target_column = ACTION_DESCRIPTOR_FEATURE_NAMES.index("target_bb")
        payment_column = ACTION_DESCRIPTOR_FEATURE_NAMES.index("payment_bb")
        for action in range(NUM_ACTIONS):
            if action not in legal:
                self.assertTrue(torch.equal(fields[action], torch.zeros_like(fields[action])))
                continue
            expected_target = self.env.action_target(state, action)
            expected_payment = (
                expected_target - state.street_contrib[int(state.to_act)]
            )
            self.assertAlmostEqual(
                float(fields[action, target_column]),
                expected_target / self._bb(self.env),
                places=6,
            )
            self.assertAlmostEqual(
                float(fields[action, payment_column]),
                expected_payment / self._bb(self.env),
                places=6,
            )

    def test_live_encoder_fails_loudly_without_exact_action_effects(self) -> None:
        state = self.env.new_hand(button=0)
        with self.assertRaisesRegex(ValueError, "exact action_descriptors"):
            encode_information_state(
                state,
                int(state.to_act),
                self.env.legal_actions(state),
                self._bb(self.env),
            )

    def test_live_encoder_rejects_nonacting_hero(self) -> None:
        state = self.env.new_hand(button=0)
        with self.assertRaisesRegex(ValueError, "hero == state.to_act"):
            encode_information_state(
                state,
                1 - int(state.to_act),
                self.env.legal_actions(state),
                self._bb(self.env),
                action_descriptors=build_action_descriptors(self.env, state),
            )

    def test_encoder_keeps_most_recent_history_when_capacity_is_exceeded(self) -> None:
        state = self.env.new_hand(button=0)
        for _ in range(9):
            self.assertIn(ACTION_MIN_RAISE, self.env.legal_actions(state))
            state = self.env.step(state, ACTION_MIN_RAISE)
        encoded = encode_information_state(
            state,
            int(state.to_act),
            self.env.legal_actions(state),
            self._bb(self.env),
            max_history=8,
            action_descriptors=build_action_descriptors(self.env, state),
        )
        history = encoded[
            HISTORY_OFFSET : HISTORY_OFFSET + 8 * HISTORY_FEATURES
        ].reshape(8, HISTORY_FEATURES)
        self.assertEqual(encoded.shape, (information_state_size(8),))
        self.assertTrue(torch.all(history[:, 0] == 1.0))
        self.assertAlmostEqual(
            float(history[0, HISTORY_FEATURE_NAMES.index("pot_before_bb")]),
            float(state.history[-8].pot_before) / self._bb(self.env),
        )

    def test_encoder_rejects_old_stack_size_as_normalization_unit(self) -> None:
        state = self.env.new_hand(button=0)
        with self.assertRaisesRegex(ValueError, "does not match state.big_blind"):
            encode_information_state(
                state,
                int(state.to_act),
                self.env.legal_actions(state),
                self.env.starting_stack,
                action_descriptors=build_action_descriptors(self.env, state),
            )

    def test_arbitrary_raise_is_encoded_semantically_with_exact_amounts(self) -> None:
        state = self.env.new_hand(button=0)
        state = self.env.step_exact(state, "raise_to", raise_to=7)
        encoded = self._encode(self.env, state)

        start = HISTORY_OFFSET + (DEFAULT_MAX_HISTORY - 1) * HISTORY_FEATURES
        token = encoded[start : start + HISTORY_FEATURES]
        self.assertEqual(
            float(token[HISTORY_FEATURE_NAMES.index("kind_raise")]),
            1.0,
        )
        self.assertEqual(
            float(token[HISTORY_FEATURE_NAMES.index("kind_bet")]),
            0.0,
        )
        # The button/SB already had one chip in and paid six to raise to seven.
        self.assertAlmostEqual(
            float(token[HISTORY_FEATURE_NAMES.index("amount_added_bb")]),
            3.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(token[HISTORY_FEATURE_NAMES.index("contribution_after_bb")]),
            3.5,
            places=6,
        )
        self.assertAlmostEqual(
            float(token[HISTORY_FEATURE_NAMES.index("target_over_pot_before")]),
            7.0 / 3.0,
            places=6,
        )
        summary = event_summary(state.history[-1])
        self.assertIn("to      7", summary)
        self.assertIn("bet 2->7", summary)

    def test_chip_and_blind_scaling_leaves_observation_unchanged(self) -> None:
        deck = list(range(52))
        small = HeadsUpHoldemEnv(
            starting_stack=100,
            small_blind=1,
            big_blind=2,
            seed=1,
        )
        large = HeadsUpHoldemEnv(
            starting_stack=1000,
            small_blind=10,
            big_blind=20,
            seed=1,
        )
        small_state = small.new_hand(button=0, deck=deck)
        large_state = large.new_hand(button=0, deck=deck)
        # Six makes every fractional candidate an integral chip target at both
        # scales.  If rounding genuinely changes an available target, the
        # descriptor should differ because the engine effect really differs.
        small_state = small.step_exact(small_state, "raise_to", raise_to=6)
        large_state = large.step_exact(large_state, "raise_to", raise_to=60)
        small_observation = self._encode(small, small_state)
        large_observation = self._encode(large, large_state)
        self.assertTrue(torch.allclose(small_observation, large_observation, atol=1e-6))

    def test_masked_softmax_never_assigns_probability_to_illegal_slot(self) -> None:
        logits = torch.arange(NUM_ACTIONS, dtype=torch.float32)
        mask = torch.zeros(NUM_ACTIONS)
        mask[[0, 2, 7]] = 1.0
        probabilities = masked_softmax(logits, mask)
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=6)
        self.assertTrue(torch.equal(probabilities[mask == 0], torch.zeros(7)))

    def test_gui_helpers_report_exact_engine_targets(self) -> None:
        state = self.env.new_hand(button=0)
        legal = self.env.legal_actions(state)
        raise_action = next(
            action
            for action in legal
            if self.env.action_target(state, action) > state.current_bet
        )
        target = self.env.action_target(state, raise_action)
        self.assertIn(f"to {target:g}", fixed_action_label(self.env, state, raise_action))
        facts = state_facts(self.env, state)
        self.assertEqual(facts["actor"], "P0")
        self.assertEqual(facts["pot"], "3")
        self.assertEqual(facts["current_bet"], "2")
        self.assertEqual(facts["to_call"], "1")
        self.assertEqual(facts["minimum_raise_to"], "4")

    def test_gui_passes_integer_chip_configuration_to_engine(self) -> None:
        class FakeRoot:
            def title(self, *_args) -> None:
                pass

            def geometry(self, *_args) -> None:
                pass

            def minsize(self, *_args) -> None:
                pass

            def configure(self, **_kwargs) -> None:
                pass

            def bind(self, *_args, **_kwargs) -> None:
                pass

        args = parse_args(["--stack", "200", "--sb", "1", "--bb", "2"])
        with (
            patch("play_heads_up_gui.tk.StringVar", return_value=object()),
            patch.object(HeadsUpManualGUI, "_build_widgets"),
            patch.object(HeadsUpManualGUI, "reset_match"),
        ):
            gui = HeadsUpManualGUI(
                FakeRoot(),
                starting_stack=args.stack,
                small_blind=args.sb,
                big_blind=args.bb,
                seed=None,
                first_button=0,
            )

        self.assertIs(type(gui.starting_stack), int)
        self.assertIs(type(gui.small_blind), int)
        self.assertIs(type(gui.big_blind), int)
        state = gui.env.new_hand(button=0, stacks=gui.session_stacks)
        self.assertFalse(state.terminal)


if __name__ == "__main__":
    unittest.main()
