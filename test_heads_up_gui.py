import tempfile
import unittest
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from heads_up_engine import ACTION_NAMES, HeadsUpHoldemEngine
from play_heads_up_gui import (
    AveragedHeadsUpSnapshotPolicy,
    POLICY_SHIFT_NEGATIVE_TAG,
    POLICY_SHIFT_NEUTRAL_TAG,
    POLICY_SHIFT_POSITIVE_TAG,
    HUD_HISTORY_VERSION,
    HUD_INITIAL_HANDS,
    HeadsUpManualGUI,
    PolicyResultsLedger,
    calculate_player_hud,
    canonical_hud_hand,
    format_hud_af,
    format_policy_shift,
    is_first_preflop_bot_decision,
    is_premium_preflop_never_fold_hand,
    probabilities_without_fold,
    reconstruct_initial_deck,
    persisted_hud_hands,
    parse_args,
    probabilities_from_top_k,
    should_remove_fold_on_first_preflop_action,
    should_apply_premium_preflop_guard,
    structured_action_history,
    terminal_hand_audit,
)


def result_row(
    *,
    fingerprint: str,
    iteration: int,
    mode: str,
    human_payoff_bb: float,
) -> dict:
    if human_payoff_bb > 0:
        outcome = "win"
    elif human_payoff_bb < 0:
        outcome = "loss"
    else:
        outcome = "tie"
    return {
        "policy_sha256": fingerprint,
        "policy_mode": mode,
        "policy_iteration": iteration,
        "policy_file": f"policy_{iteration}.pt",
        "human_payoff_bb": human_payoff_bb,
        "human_outcome": outcome,
    }


class PolicyShiftDisplayTests(unittest.TestCase):
    def test_formats_positive_policy_shift_in_green(self):
        text, tag = format_policy_shift(0.70, 0.40)
        self.assertEqual(text, "+30.0pp")
        self.assertEqual(tag, POLICY_SHIFT_POSITIVE_TAG)

    def test_formats_negative_policy_shift_in_red(self):
        text, tag = format_policy_shift(0.10, 0.35)
        self.assertEqual(text, "-25.0pp")
        self.assertEqual(tag, POLICY_SHIFT_NEGATIVE_TAG)

    def test_rounds_negligible_policy_shift_to_neutral(self):
        text, tag = format_policy_shift(0.4004, 0.40)
        self.assertEqual(text, " +0.0pp")
        self.assertEqual(tag, POLICY_SHIFT_NEUTRAL_TAG)


class AveragedPolicyTests(unittest.TestCase):
    @staticmethod
    def fake_policy(iteration, fingerprint, probabilities):
        policy = SimpleNamespace(
            iteration=iteration,
            sha256=fingerprint,
            path=Path(f"policy_{iteration:08d}.pt"),
            mode="sample",
            device=torch.device("cpu"),
            snapshot=SimpleNamespace(
                metadata={
                    "model_id": "test_hu",
                    "input_dim": 10,
                    "max_history": 4,
                    "action_schema_version": 1,
                    "engine_schema_version": 1,
                }
            ),
            generator=torch.Generator(device="cpu").manual_seed(123),
        )
        policy.probabilities = Mock(return_value=torch.tensor(probabilities))
        return policy

    def test_three_policy_probabilities_are_equal_weight_average(self):
        policies = [
            self.fake_policy(725, "a" * 64, [0.6, 0.3, 0.1]),
            self.fake_policy(950, "b" * 64, [0.3, 0.6, 0.1]),
            self.fake_policy(1025, "c" * 64, [0.0, 0.3, 0.7]),
        ]
        ensemble = AveragedHeadsUpSnapshotPolicy(policies)

        actual = ensemble.probabilities(Mock(), Mock())

        torch.testing.assert_close(actual, torch.tensor([0.3, 0.4, 0.3]))
        self.assertEqual(ensemble.iterations, (725, 950, 1025))
        self.assertEqual(ensemble.iteration, 1025)
        self.assertEqual(len(ensemble.sha256), 64)

    def test_cli_accepts_two_repeated_secondary_policies(self):
        args = parse_args(
            [
                "--policy",
                "725.pt",
                "--policy-secondary",
                "950.pt",
                "--policy-secondary",
                "1025.pt",
                "--no-search",
            ]
        )
        self.assertEqual(
            args.policy_secondary,
            [Path("950.pt"), Path("1025.pt")],
        )

    def test_top_three_actions_are_kept_and_renormalized(self):
        probabilities = torch.tensor(
            [0.05, 0.0, 0.30, 0.10, 0.0, 0.0, 0.25, 0.20, 0.0, 0.10]
        )
        adjusted = probabilities_from_top_k(
            probabilities,
            [0, 2, 3, 6, 7, 9],
            3,
        )
        expected = torch.zeros(10)
        expected[2] = 0.30 / 0.75
        expected[6] = 0.25 / 0.75
        expected[7] = 0.20 / 0.75
        torch.testing.assert_close(adjusted, expected)


class PlayerHudTests(unittest.TestCase):
    @staticmethod
    def event(player, street, kind, amount=0):
        return SimpleNamespace(
            player=player,
            street=street,
            kind=kind,
            amount=amount,
        )

    def test_vpip_ats_and_postflop_af_use_standard_denominators(self):
        hands = [
            {
                "button": 0,
                "history": (
                    self.event(0, 0, "call", 1),
                    self.event(1, 0, "raise", 2),
                    self.event(0, 0, "call", 2),
                    self.event(1, 1, "bet", 4),
                    self.event(0, 1, "call", 4),
                    self.event(0, 2, "bet", 8),
                ),
            },
            {
                "button": 1,
                "history": (
                    self.event(1, 0, "raise", 3),
                    self.event(0, 0, "fold"),
                ),
            },
        ]

        player_zero = calculate_player_hud(hands, 0)
        self.assertEqual(player_zero["hands"], 2)
        self.assertEqual(player_zero["vpip_hands"], 1)
        self.assertEqual(player_zero["vpip_pct"], 50.0)
        self.assertEqual(player_zero["steal_opportunities"], 1)
        self.assertEqual(player_zero["steal_attempts"], 0)
        self.assertEqual(player_zero["ats_pct"], 0.0)
        self.assertEqual(player_zero["af"], 1.0)

        player_one = calculate_player_hud(hands, 1)
        self.assertEqual(player_one["vpip_pct"], 100.0)
        self.assertEqual(player_one["ats_pct"], 100.0)
        self.assertTrue(torch.isinf(torch.tensor(player_one["af"])))
        self.assertEqual(format_hud_af(player_one["af"]), "\u221e")

    def test_blinds_checks_and_folds_do_not_count_as_vpip(self):
        hands = [
            {
                "button": 0,
                "history": (
                    self.event(0, 0, "fold"),
                ),
            },
            {
                "button": 1,
                "history": (
                    self.event(1, 0, "call", 0),
                    self.event(0, 0, "check"),
                ),
            },
        ]

        stats = calculate_player_hud(hands, 0)

        self.assertEqual(stats["vpip_hands"], 0)
        self.assertEqual(stats["vpip_pct"], 0.0)
        self.assertEqual(stats["af"], 0.0)
        self.assertEqual(format_hud_af(stats["af"]), "0.00")

    def test_persisted_hud_keeps_roles_when_physical_seats_change(self):
        records = [
            {
                "button": 1,
                "human_seat": 1,
                "action_sequence": [
                    {
                        "player": 1,
                        "street": 0,
                        "kind": "raise",
                        "amount": 3,
                    },
                    {
                        "player": 0,
                        "street": 0,
                        "kind": "fold",
                        "amount": 0,
                    },
                ],
            }
        ]

        restored = persisted_hud_hands(records)
        human = calculate_player_hud(restored, 0)
        policy = calculate_player_hud(restored, 1)

        self.assertEqual(human["vpip_pct"], 100.0)
        self.assertEqual(human["ats_pct"], 100.0)
        self.assertEqual(policy["vpip_pct"], 0.0)

    def test_persisted_hud_restores_legacy_text_history(self):
        records = [
            {
                "button": 0,
                "human_seat": 0,
                "public_history": [
                    "preflop   P0 raise              +     4  to      5",
                    "preflop   P1 call               +     3  to      5",
                    "flop      P1 check              +     0  to      0",
                    "flop      P0 bet                +     5  to      5",
                ],
            }
        ]

        restored = persisted_hud_hands(records)
        human = calculate_player_hud(restored, 0)

        self.assertEqual(len(restored), 1)
        self.assertEqual(human["vpip_pct"], 100.0)
        self.assertEqual(human["ats_pct"], 100.0)
        self.assertTrue(torch.isinf(torch.tensor(human["af"])))

    def test_persisted_hud_starts_with_most_recent_1700_hands(self):
        records = [
            {
                "button": index % 2,
                "human_seat": 0,
                "action_sequence": [
                    {
                        "player": 0,
                        "street": 0,
                        "kind": "fold" if index == 0 else "raise",
                        "amount": 0 if index == 0 else 3,
                    }
                ],
            }
            for index in range(HUD_INITIAL_HANDS + 1)
        ]

        restored = persisted_hud_hands(records)
        human = calculate_player_hud(restored, 0)

        self.assertEqual(len(restored), HUD_INITIAL_HANDS)
        self.assertEqual(human["hands"], HUD_INITIAL_HANDS)
        self.assertEqual(human["vpip_pct"], 100.0)

    def test_persisted_hud_keeps_all_hands_after_fixed_boundary(self):
        records = [
            {
                "button": index % 2,
                "human_seat": 0,
                "action_sequence": [
                    {
                        "player": 0,
                        "street": 0,
                        "kind": "raise",
                        "amount": 3,
                    }
                ],
                **(
                    {"hud_history_version": HUD_HISTORY_VERSION}
                    if index >= HUD_INITIAL_HANDS + 2
                    else {}
                ),
            }
            for index in range(HUD_INITIAL_HANDS + 7)
        ]

        restored = persisted_hud_hands(records)

        self.assertEqual(len(restored), HUD_INITIAL_HANDS + 5)

    def test_canonical_hud_hand_maps_bot_button_to_policy_role(self):
        hand = canonical_hud_hand(
            0,
            [self.event(0, 0, "raise", 3)],
            human_seat=1,
        )

        self.assertEqual(hand["button"], 1)
        self.assertEqual(hand["history"][0]["player"], 1)


class FirstPreflopPolicyBypassTests(unittest.TestCase):
    def test_first_bot_preflop_decision_bypasses_search(self):
        state = SimpleNamespace(
            street=0,
            history=[SimpleNamespace(player=0)],
        )
        self.assertTrue(is_first_preflop_bot_decision(state, bot_seat=1))

    def test_second_bot_preflop_decision_uses_search(self):
        state = SimpleNamespace(
            street=0,
            history=[
                SimpleNamespace(player=0),
                SimpleNamespace(player=1),
                SimpleNamespace(player=0),
            ],
        )
        self.assertFalse(is_first_preflop_bot_decision(state, bot_seat=1))

    def test_first_postflop_decision_uses_search(self):
        state = SimpleNamespace(street=1, history=[])
        self.assertFalse(is_first_preflop_bot_decision(state, bot_seat=1))

    def test_hybrid_change_keeps_existing_search_ledger_group(self):
        gui = HeadsUpManualGUI.__new__(HeadsUpManualGUI)
        gui.policy = SimpleNamespace(mode="sample")
        gui.search_enabled = True
        gui.search_mode = "three-player"
        gui.root_range_mode = "inferred"
        self.assertEqual(
            gui._controller_mode(),
            "sample+three_player_root_search_inferred_range",
        )

    def test_unmodified_plain_policy_has_separate_ledger_mode(self):
        gui = HeadsUpManualGUI.__new__(HeadsUpManualGUI)
        gui.policy = SimpleNamespace(mode="sample")
        gui.search_enabled = False
        gui.unmodified_policy_sampling = True

        self.assertEqual(
            gui._controller_mode(),
            "sample+unmodified_policy",
        )


class KeyboardShortcutTests(unittest.TestCase):
    @staticmethod
    def gui_with_legal_actions(*names):
        gui = HeadsUpManualGUI.__new__(HeadsUpManualGUI)
        gui.state = SimpleNamespace(terminal=False, to_act=0)
        gui.policy = None
        gui.bot_seat = 1
        legal = [ACTION_NAMES.index(name) for name in names]
        gui.env = SimpleNamespace(legal_actions=lambda _state: legal)
        gui.apply_fixed_action = Mock()
        return gui

    def test_c_calls_when_call_is_available(self):
        gui = self.gui_with_legal_actions("fold", "call")

        result = gui._handle_keyboard_shortcut(SimpleNamespace(keysym="C"))

        self.assertEqual(result, "break")
        gui.apply_fixed_action.assert_called_once_with(ACTION_NAMES.index("call"))

    def test_f_does_nothing_when_fold_is_unavailable(self):
        gui = self.gui_with_legal_actions("check")

        result = gui._handle_keyboard_shortcut(SimpleNamespace(keysym="f"))

        self.assertIsNone(result)
        gui.apply_fixed_action.assert_not_called()

    def test_space_uses_shared_next_hand_guard(self):
        gui = self.gui_with_legal_actions()
        gui._start_next_hand_if_available = Mock(side_effect=(True, False))

        enabled_result = gui._handle_keyboard_shortcut(
            SimpleNamespace(keysym="space")
        )
        disabled_result = gui._handle_keyboard_shortcut(
            SimpleNamespace(keysym="space")
        )

        self.assertEqual(enabled_result, "break")
        self.assertIsNone(disabled_result)
        self.assertEqual(gui._start_next_hand_if_available.call_count, 2)

    def test_shared_next_hand_guard_rotates_exactly_once(self):
        gui = HeadsUpManualGUI.__new__(HeadsUpManualGUI)
        gui.state = SimpleNamespace(terminal=True)
        gui.session_stacks = [200, 200]
        gui.next_button = 1

        def deal_hand():
            button = gui.next_button
            gui.next_button = 1 - button
            gui.state = SimpleNamespace(terminal=False, button=button)

        gui.deal_hand = Mock(side_effect=deal_hand)

        self.assertTrue(gui._start_next_hand_if_available())
        self.assertEqual(gui.state.button, 1)
        self.assertEqual(gui.next_button, 0)
        gui.deal_hand.assert_called_once_with()

        self.assertFalse(gui._start_next_hand_if_available())
        gui.deal_hand.assert_called_once_with()


class BotCardRevealTests(unittest.TestCase):
    def test_reveal_requires_terminal_hand_and_redraws_table(self):
        gui = HeadsUpManualGUI.__new__(HeadsUpManualGUI)
        gui.policy = SimpleNamespace()
        gui.state = SimpleNamespace(terminal=True)
        gui.bot_cards_revealed = False
        gui.reveal_bot_button = Mock()
        gui.draw_table = Mock()

        gui.reveal_bot_cards()

        self.assertTrue(gui.bot_cards_revealed)
        gui.reveal_bot_button.configure.assert_called_once_with(
            state="disabled"
        )
        gui.draw_table.assert_called_once_with()

    def test_reveal_does_nothing_during_live_hand(self):
        gui = HeadsUpManualGUI.__new__(HeadsUpManualGUI)
        gui.policy = SimpleNamespace()
        gui.state = SimpleNamespace(terminal=False)
        gui.bot_cards_revealed = False
        gui.reveal_bot_button = Mock()
        gui.draw_table = Mock()

        gui.reveal_bot_cards()

        self.assertFalse(gui.bot_cards_revealed)
        gui.reveal_bot_button.configure.assert_not_called()
        gui.draw_table.assert_not_called()


class PremiumPreflopNeverFoldTests(unittest.TestCase):
    @staticmethod
    def state(hole, *, street=0, history=()):
        return SimpleNamespace(
            street=street,
            hole=[[], list(hole)],
            history=list(history),
        )

    def test_protects_aa_through_jj_and_all_ak_combinations(self):
        for hole in ((12, 25), (11, 24), (10, 23), (9, 22), (12, 11), (12, 24)):
            with self.subTest(hole=hole):
                self.assertTrue(
                    is_premium_preflop_never_fold_hand(
                        self.state(hole),
                        seat=1,
                    )
                )

    def test_does_not_protect_tt_or_postflop_aa(self):
        self.assertFalse(
            is_premium_preflop_never_fold_hand(
                self.state((8, 21)),
                seat=1,
            )
        )
        self.assertFalse(
            is_premium_preflop_never_fold_hand(
                self.state((12, 25), street=1),
                seat=1,
            )
        )

    def test_fold_removal_applies_only_to_first_bot_action(self):
        first = self.state(
            (12, 25),
            history=[SimpleNamespace(player=0)],
        )
        second = self.state(
            (12, 25),
            history=[
                SimpleNamespace(player=0),
                SimpleNamespace(player=1),
                SimpleNamespace(player=0),
            ],
        )
        self.assertTrue(
            should_remove_fold_on_first_preflop_action(first, seat=1)
        )
        self.assertFalse(
            should_remove_fold_on_first_preflop_action(second, seat=1)
        )

    def test_raw_policy_fold_is_removed_and_remaining_mass_renormalized(self):
        probabilities = torch.zeros(10)
        probabilities[0] = 0.20
        probabilities[2] = 0.30
        probabilities[9] = 0.50
        adjusted = probabilities_without_fold(
            probabilities,
            [0, 2, 9],
        )
        self.assertEqual(float(adjusted[0]), 0.0)
        self.assertAlmostEqual(float(adjusted[2]), 0.375)
        self.assertAlmostEqual(float(adjusted[9]), 0.625)
        self.assertAlmostEqual(float(adjusted.sum()), 1.0)

    def test_unmodified_campaign_disables_premium_fold_override(self):
        state = self.state(
            (12, 25),
            history=[SimpleNamespace(player=0)],
        )

        self.assertFalse(
            should_apply_premium_preflop_guard(
                state,
                seat=1,
                unmodified_policy_sampling=True,
            )
        )
        self.assertTrue(
            should_apply_premium_preflop_guard(
                state,
                seat=1,
                unmodified_policy_sampling=False,
            )
        )

class PolicyResultsLedgerTests(unittest.TestCase):
    def test_temporary_first_preflop_mode_alias_joins_existing_group(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hands.jsonl"
            ledger = PolicyResultsLedger(path)
            old = result_row(
                fingerprint="a" * 64,
                iteration=1025,
                mode="sample+three_player_root_search_inferred_range",
                human_payoff_bb=1.0,
            )
            temporary = result_row(
                fingerprint="a" * 64,
                iteration=1025,
                mode=(
                    "sample+raw_first_preflop"
                    "+three_player_root_search_inferred_range"
                ),
                human_payoff_bb=-0.5,
            )
            ledger.append(old)
            ledger.append(temporary)
            summaries = ledger.summaries()
            self.assertEqual(len(summaries), 1)
            self.assertEqual(summaries[0]["hands"], 2)
            self.assertEqual(
                summaries[0]["policy_mode"],
                "sample+three_player_root_search_inferred_range",
            )
            self.assertAlmostEqual(summaries[0]["human_net_bb"], 0.5)

    def test_persists_and_aggregates_zero_sum_results_by_policy_and_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hands.jsonl"
            ledger = PolicyResultsLedger(path)
            ledger.append(
                result_row(
                    fingerprint="a" * 64,
                    iteration=600,
                    mode="sample",
                    human_payoff_bb=2.0,
                )
            )
            ledger.append(
                result_row(
                    fingerprint="a" * 64,
                    iteration=600,
                    mode="sample",
                    human_payoff_bb=-1.0,
                )
            )
            ledger.append(
                result_row(
                    fingerprint="b" * 64,
                    iteration=700,
                    mode="argmax",
                    human_payoff_bb=0.0,
                )
            )

            restored = PolicyResultsLedger(path)
            self.assertTrue(
                all(record["version"] == 2 for record in restored.records)
            )
            summaries = restored.summaries()
            self.assertEqual(len(summaries), 2)
            sample = summaries[0]
            self.assertEqual(sample["policy_iteration"], 600)
            self.assertEqual(sample["hands"], 2)
            self.assertEqual((sample["wins"], sample["losses"], sample["ties"]), (1, 1, 0))
            self.assertAlmostEqual(sample["human_net_bb"], 1.0)
            self.assertAlmostEqual(sample["human_bb_per_hand"], 0.5)
            self.assertAlmostEqual(sample["policy_bb_per_hand"], -0.5)

            argmax = summaries[1]
            self.assertEqual(argmax["policy_iteration"], 700)
            self.assertEqual(argmax["hands"], 1)
            self.assertEqual((argmax["wins"], argmax["losses"], argmax["ties"]), (0, 0, 1))

    def test_rejects_a_corrupted_historical_row(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hands.jsonl"
            path.write_text("{broken\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "line 1"):
                PolicyResultsLedger(path)

    def test_accepts_legacy_version_one_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hands.jsonl"
            row = result_row(
                fingerprint="a" * 64,
                iteration=600,
                mode="sample",
                human_payoff_bb=1.0,
            )
            row["version"] = 1
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            ledger = PolicyResultsLedger(path)
            self.assertEqual(len(ledger.records), 1)
            self.assertEqual(ledger.records[0]["version"], 1)

    def test_terminal_audit_reconstructs_and_replays_exact_hand(self):
        deck = list(range(52))
        env = HeadsUpHoldemEngine(200, 1, 2, seed=7)
        state = env.new_hand(button=0, deck=deck)
        while not state.terminal:
            legal = [int(action) for action in env.legal_actions(state)]
            names = {
                int(action): ACTION_NAMES[int(action)]
                for action in legal
            }
            passive = next(
                (
                    action
                    for action, name in names.items()
                    if name in {"call", "check"}
                ),
                legal[0],
            )
            state = env.step(state, passive)

        reconstructed = reconstruct_initial_deck(state)
        self.assertEqual(reconstructed, deck)
        actions = structured_action_history(state)
        self.assertEqual(len(actions), len(state.history))
        self.assertTrue(all("board_card_ids" in row for row in actions))

        audit = terminal_hand_audit(state)
        self.assertEqual(
            audit["record_schema"],
            "heads_up_gui_hand_v2_complete",
        )
        self.assertEqual(len(audit["initial_deck_order_card_ids"]), 52)
        self.assertEqual(len(set(audit["initial_deck_order_card_ids"])), 52)
        self.assertEqual(audit["terminal_reason"], "showdown")

        replay_env = HeadsUpHoldemEngine(200, 1, 2, seed=999)
        replay = replay_env.new_hand(
            button=int(state.button),
            stacks=list(state.initial_stacks),
            deck=audit["initial_deck_order_card_ids"],
        )
        for event in audit["action_sequence"]:
            if event["kind"] in {"bet", "raise"}:
                replay = replay_env.step_exact(
                    replay,
                    "raise_to",
                    raise_to=int(event["raise_to"]),
                )
            else:
                replay = replay_env.step_exact(replay, event["kind"])
        self.assertTrue(replay.terminal)
        self.assertEqual(replay.hole, state.hole)
        self.assertEqual(replay.board, state.board)
        self.assertEqual(replay.payoffs, state.payoffs)

    def test_terminal_audit_handles_preflop_fold_and_all_in_runout(self):
        for starting_stack, actions in (
            (200, (("fold", None),)),
            (20, (("all_in", None), ("call", None))),
        ):
            deck = list(reversed(range(52)))
            env = HeadsUpHoldemEngine(starting_stack, 1, 2, seed=11)
            state = env.new_hand(button=1, deck=deck)
            for kind, raise_to in actions:
                if state.terminal:
                    break
                state = env.step_exact(
                    state,
                    kind,
                    raise_to=raise_to,
                )
            self.assertTrue(state.terminal)
            audit = terminal_hand_audit(state)
            self.assertEqual(
                audit["initial_deck_order_card_ids"],
                deck,
            )
            self.assertEqual(
                audit["terminal_reason"],
                "fold" if starting_stack == 200 else "showdown",
            )


if __name__ == "__main__":
    unittest.main()
