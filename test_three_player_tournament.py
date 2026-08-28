import random
import unittest

from three_player_engine import ACTION_ALL_IN, ACTION_CALL, ACTION_CHECK, ThreePlayerHoldemEnv
from three_player_tournament import play_tournament, winner_take_all_rewards


def shove_or_continue(env, state, _player):
    legal = env.legal_actions(state)
    for action in (ACTION_ALL_IN, ACTION_CALL, ACTION_CHECK):
        if action in legal:
            return action
    return legal[0]


class TournamentOrchestrationTests(unittest.TestCase):
    def test_winner_reward_is_zero_sum(self):
        for winner in range(3):
            rewards = winner_take_all_rewards(winner)
            self.assertEqual(rewards[winner], 2.0)
            self.assertEqual(sum(rewards), 0.0)

    def test_continuing_tournament_carries_stacks_and_finishes(self):
        env = ThreePlayerHoldemEnv(stack_size=8, sb=1, bb=2, seed=901)
        result = play_tournament(
            env,
            [shove_or_continue] * 3,
            rng=random.Random(902),
            max_hands=200,
        )
        self.assertGreaterEqual(result.hands_played, 1)
        self.assertEqual(sum(result.final_stacks), 24.0)
        self.assertEqual(sum(stack > 0 for stack in result.final_stacks), 1)
        self.assertEqual(result.rewards[result.winner], 2.0)
        for previous, following in zip(result.hands, result.hands[1:]):
            self.assertEqual(previous.ending_stacks, following.starting_stacks)

    def test_tournament_can_begin_heads_up(self):
        env = ThreePlayerHoldemEnv(stack_size=6, sb=1, bb=2, seed=903)
        result = play_tournament(
            env,
            [shove_or_continue] * 3,
            starting_stacks=[6, 0, 6],
            max_hands=100,
        )
        self.assertEqual(result.eliminated_on_hand[1], 0)
        self.assertNotEqual(result.winner, 1)


if __name__ == "__main__":
    unittest.main()
