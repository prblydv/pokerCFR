import random
import time
from collections import defaultdict
from poker_env import SimpleHoldemEnv


def play_random_hands(num_hands: int = 200_330, max_steps: int = 200):
    env = SimpleHoldemEnv()
    start_time = time.perf_counter()

    # Track step statistics
    step_hist = defaultdict(int)
    max_steps_seen = 0
    max_steps_hits = 0
    hard_cap_hits = 0

    for hand_idx in range(num_hands):
        state = env.new_hand()
        steps = 0

        while not state.terminal and steps < max_steps:
            legal = env.legal_actions(state)
            if not legal:
                break
            action = random.choice(legal)
            state = env.step(state, action)
            steps += 1

        # Record stats
        step_hist[steps] += 1

        if steps > max_steps_seen:
            max_steps_seen = steps
            max_steps_hits = 1
        elif steps == max_steps_seen:
            max_steps_hits += 1

        if steps >= max_steps:
            hard_cap_hits += 1

    elapsed = time.perf_counter() - start_time

    # ---- Final Report ----
    print("\n================ SIMULATION REPORT ================\n")
    print(f"Hands played           : {num_hands}")
    print(f"Max steps allowed      : {max_steps}")
    print(f"Max steps observed     : {max_steps_seen}")
    print(f"Hands at max observed  : {max_steps_hits}")
    print(f"Hands hitting hard cap : {hard_cap_hits}")
    print(f"Runtime                : {elapsed:.3f} seconds")

    # Optional: sanity check distribution tail
    print("\nTop 10 highest step counts:")
    for s in sorted(step_hist.keys(), reverse=True)[:10]:
        print(f"  steps={s:3d} → {step_hist[s]} hands")


if __name__ == "__main__":
    play_random_hands()











# runthe folling to detect infinte loops




# import random
# from collections import defaultdict
# from poker_env import SimpleHoldemEnv, GameState

# RNG = random.Random(123)


# # ---------------------------------------------------------
# # State fingerprint (ENGINE-SAFE)
# # ---------------------------------------------------------
# def state_fingerprint(s: GameState):
#     """
#     Fingerprint logical betting state.
#     Uses only fields guaranteed to exist.
#     """
#     return (
#         s.street,
#         s.to_act,
#         round(s.pot, 2),
#         tuple(round(x, 2) for x in s.stacks),
#         tuple(round(x, 2) for x in s.contrib),
#         round(s.current_bet, 2),
#         s.last_aggressor,
#         s.actions_this_street,
#     )


# # ---------------------------------------------------------
# # Infinite loop finder
# # ---------------------------------------------------------
# def find_infinite_loops(
#     num_hands: int = 100_000,
#     max_steps: int = 200,
#     seed: int = 123,
# ):
#     RNG.seed(seed)
#     env = SimpleHoldemEnv()

#     true_loops = 0
#     soft_loops = 0
#     step_cap_hits = 0

#     first_loop_trace = None

#     for hand_id in range(num_hands):
#         state = env.new_hand()
#         seen = {}
#         trace = []

#         for step in range(max_steps):
#             fp = state_fingerprint(state)

#             if fp in seen:
#                 loop_len = step - seen[fp]
#                 if loop_len == 1:
#                     true_loops += 1
#                 else:
#                     soft_loops += 1

#                 if first_loop_trace is None:
#                     first_loop_trace = trace[seen[fp]:]

#                 break

#             seen[fp] = step
#             trace.append(fp)

#             if state.terminal:
#                 break

#             legal = env.legal_actions(state)
#             if not legal:
#                 break

#             action = RNG.choice(legal)
#             state = env.step(state, action)

#         else:
#             step_cap_hits += 1

#     # -----------------------------------------------------
#     # Report
#     # -----------------------------------------------------
#     print("\n================ LOOP SCAN REPORT ================")
#     print(f"Hands scanned           : {num_hands}")
#     print(f"True loops found        : {true_loops}")
#     print(f"Soft loops detected     : {soft_loops}")
#     print(f"Hands hitting step cap  : {step_cap_hits}")
#     print("=================================================")

#     if first_loop_trace:
#         print("\n--- FIRST LOOP TRACE (fingerprints) ---")
#         for i, fp in enumerate(first_loop_trace):
#             print(f"{i:02d}: {fp}")
#         print("-------------------------------------")


# # ---------------------------------------------------------
# # Entry point
# # ---------------------------------------------------------
# if __name__ == "__main__":
#     find_infinite_loops()
