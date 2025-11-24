from copy import deepcopy
from poker_env import SimpleHoldemEnv

env = SimpleHoldemEnv()
s0 = env.new_hand()

# Construct deterministic BTN opening state
s0.street = 0
s0.board = []
s0.pot = 1.5
s0.current_bet = 0
s0.stacks = [200, 200]
s0.last_aggressor = -1

# If your env uses env.current_player instead of acting_player:
# print("env.current_player:", env.current_player)

def inspect_action(a):
    s = deepcopy(s0)

    pot_before = s.pot
    stacks_before = tuple(s.stacks)
    cb_before = s.current_bet
    la_before = s.last_aggressor

    s2 = env.step(s, a)   # NEW STATE

    print(
        f"a={a:>2} | "
        f"pot {pot_before}->{s2.pot} | "
        f"stacks {stacks_before}->{tuple(s2.stacks)} | "
        f"current_bet {cb_before}->{s2.current_bet} | "
        f"last_agg {la_before}->{s2.last_aggressor}"
    )

print("BTN legal actions:", env.legal_actions(s0))
print("\nInspecting each action:\n")

for a in env.legal_actions(s0):
    inspect_action(a)
# from poker_env import SimpleHoldemEnv, STREET_PREFLOP, STREET_FLOP, STREET_TURN, STREET_RIVER
# from copy import deepcopy
# from poker_env import RAISE_MULT

# def assert_close(x, y, tol=1e-6):
#     if abs(x - y) > tol:
#         raise AssertionError(f"{x} != {y} (expected {y})")

# def check_new_hand():
#     print("=== TEST 1: new_hand() blinds correct ===")
#     env = SimpleHoldemEnv()
#     s = env.new_hand()

#     sb, bb = s.sb_player, s.bb_player

#     # blinds deducted
#     assert_close(s.stacks[sb], env.stack_size - env.sb)
#     assert_close(s.stacks[bb], env.stack_size - env.bb)

#     # pot correct
#     assert_close(s.pot, env.sb + env.bb)

#     # to_act must be SB
#     assert s.to_act == sb

#     print("PASS\n")


# def check_fold_awards_pot():
#     print("=== TEST 2: Fold awards pot ===")
#     env = SimpleHoldemEnv()
#     s = env.new_hand()

#     sb, bb = s.sb_player, s.bb_player

#     # SB folds
#     s2 = env.step(s, 0)

#     assert s2.terminal
#     assert s2.winner == bb
#     assert_close(s2.pot, 0.0)
#     assert_close(s2.stacks[bb], env.stack_size - env.bb + env.sb + env.bb)

#     print("PASS\n")


# def check_preflop_call_cost_correct():
#     print("=== TEST 3: Preflop CALL cost correct ===")
#     env = SimpleHoldemEnv()
#     s = env.new_hand()

#     sb, bb = s.sb_player, s.bb_player

#     # SB calls BB open
#     s2 = env.step(s, 1)

#     expected_call_amt = env.bb - env.sb  # 1.0 - 0.5 = 0.5
#     assert_close(s2.pot, env.sb + env.bb + expected_call_amt)
#     assert_close(s2.stacks[sb], env.stack_size - env.sb - expected_call_amt)

#     print("PASS\n")


# def check_raise_sizes():
#     print("=== TEST 4: Preflop raise sizes ===")
#     env = SimpleHoldemEnv()
#     s0 = env.new_hand()
#     sb, bb = s0.sb_player, s0.bb_player

#     for a in [2,3,4,5,6,7,8]:   # 2x .. 6x raises
#         s = deepcopy(s0)
#         s2 = env.step(s, a)

#         print(f"Action {a}: current_bet = {s2.current_bet:.2f} BB")

#         # Skip checking if raise was capped (short-stack all-in)
#         if s2.current_bet < s2.stacks[sb] + env.bb:
#             expected_mult = RAISE_MULT[a]
#             actual_mult = s2.current_bet / env.bb
#             assert_close(actual_mult, expected_mult)

#     print("PASS\n")


# def check_all_in():
#     print("=== TEST 5: ALL-IN logic ===")
#     env = SimpleHoldemEnv()
#     s0 = env.new_hand()
#     sb, bb = s0.sb_player, s0.bb_player

#     s2 = env.step(s0, 9)  # ALL-IN
#     assert s2.stacks[sb] == 0 or s2.stacks[bb] == 0
#     assert s2.current_bet > 0

#     print("PASS\n")


# def check_postflop_transition():
#     print("=== TEST 6: Street transitions ===")
#     env = SimpleHoldemEnv()
#     s = env.new_hand()
#     sb = s.sb_player

#     # SB calls → go to bb → betting ends → flop dealt
#     s2 = env.step(s, 1)  # SB calls BB
#     assert s2.street in [STREET_PREFLOP, STREET_FLOP]

#     # If flop dealt
#     if s2.street == STREET_FLOP:
#         assert len(s2.board) == 3
#         print("PASS\n")
#         return

#     # If still preflop (stack-bottom edge case), call with BB and end round
#     s3 = env.step(s2, 1)
#     if s3.street == STREET_FLOP:
#         assert len(s3.board) == 3
#         print("PASS\n")
#         return

#     raise AssertionError("Flop was not dealt correctly.")


# def check_terminal_payoff():
#     print("=== TEST 7: Terminal payoff stack delta ===")
#     env = SimpleHoldemEnv()
#     s = env.new_hand()

#     sb, bb = s.sb_player, s.bb_player

#     # SB folds → BB wins pot
#     s2 = env.step(s, 0)
#     payoff_bb = env.terminal_payoff(s2, bb)
#     payoff_sb = env.terminal_payoff(s2, sb)

#     # Winner gains blinds
#     delta_bb = (s2.stacks[bb] - s2.initial_stacks[bb])
#     delta_sb = (s2.stacks[sb] - s2.initial_stacks[sb])

#     assert_close(payoff_bb, delta_bb)
#     assert_close(payoff_sb, delta_sb)

#     print("PASS\n")


# if __name__ == "__main__":
#     check_new_hand()
#     check_fold_awards_pot()
#     check_preflop_call_cost_correct()
#     check_raise_sizes()
#     check_all_in()
#     check_postflop_transition()
#     check_terminal_payoff()

#     print("ALL TESTS PASSED SUCCESSFULLY!")
