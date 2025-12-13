import random
import time
from poker_env import SimpleHoldemEnv

def play_random_hands(num_hands: int = 200330, max_steps: int = 200):
    env = SimpleHoldemEnv()
    start_time = time.perf_counter()

    for hand_idx in range(num_hands):
        state = env.new_hand()
        steps = 0
        flag = False
        while not state.terminal and steps < max_steps:
            if steps == max_steps - 1:
                print(f"Reached max steps ({max_steps}) for hand {hand_idx}")
                flag = True
                break
            legal = env.legal_actions(state)
            if not legal:
                break
            action = random.choice(legal)
            state = env.step(state, action)
            steps += 1

        payoff0 = env.terminal_payoff(state, 0)
        payoff1 = env.terminal_payoff(state, 1)
        print(
            f"Hand {hand_idx:02d}: button={state.sb_player}, "
            # f"steps={steps}, terminal={state.terminal}, "
            f"payoff0={payoff0:.2f}, payoff1={payoff1:.2f}"
        )

    elapsed = time.perf_counter() - start_time
    print(f"\nSimulation completed in {elapsed:.3f} seconds")
    print(flag)
if __name__ == "__main__":
    play_random_hands()
