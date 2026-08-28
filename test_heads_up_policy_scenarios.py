import unittest

from evaluate_heads_up_policy_scenarios import (
    ordering_expectations,
    policy_cases,
)
from heads_up_native import HeadsUpHoldemEngine
from heads_up_analysis import build_decision_state


class HeadsUpPolicyScenarioTests(unittest.TestCase):
    def test_catalogue_is_large_unique_and_expectations_resolve(self):
        cases = policy_cases()
        identifiers = {case.case_id for case in cases}
        self.assertEqual(len(cases), 69)
        self.assertEqual(len(identifiers), len(cases))
        self.assertGreaterEqual(len({case.category for case in cases}), 8)
        for expectation in ordering_expectations():
            self.assertIn(expectation.better_case, identifiers)
            self.assertIn(expectation.worse_case, identifiers)
            self.assertIn(
                expectation.metric, {"p_continue", "p_aggressive"}
            )

    def test_every_case_legally_replays_for_both_policy_network_seats(self):
        env = HeadsUpHoldemEngine(
            starting_stack=200,
            small_blind=1,
            big_blind=2,
            seed=812,
        )
        for case in policy_cases():
            with self.subTest(case=case.case_id):
                legal_by_seat = []
                for hero in (0, 1):
                    state = build_decision_state(
                        env,
                        case.scenario,
                        hero=hero,
                        hero_cards=case.hero_cards,
                    )
                    self.assertEqual(state.to_act, hero)
                    self.assertFalse(state.terminal)
                    self.assertEqual(
                        tuple(state.board), tuple(case.scenario.board)
                    )
                    self.assertEqual(
                        sum(state.stacks) + state.pot,
                        sum(state.initial_stacks),
                    )
                    legal_by_seat.append(tuple(env.legal_actions(state)))
                self.assertEqual(legal_by_seat[0], legal_by_seat[1])


if __name__ == "__main__":
    unittest.main()
