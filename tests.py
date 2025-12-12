"""
Extended test suite for SimpleHoldemEnv that can be run with
`python tests.py`. Each test prints PASS or FAIL.

IMPORTANT:
- These tests assume heads-up poker.
- Randomness is controlled via RNG_SEED.
"""

from poker_env import (
    SimpleHoldemEnv,
    STREET_PREFLOP,
    STREET_FLOP,
    STREET_TURN,
    STREET_RIVER,
    ACTION_CHECK,
    ACTION_CALL,
    ACTION_FOLD,
    ACTION_RAISE_SMALL,
    ACTION_RAISE_MEDIUM,
    ACTION_ALL_IN,
    GameState,
)

# Helper to print results
# Function metadata:
#   Inputs: name (str) test description  # dtype=str
#   Sample:
#       sample_output = ok("sample test")  # dtype=NoneType (prints "[PASS] sample test")
def ok(name):
    print(f"[PASS] {name}")

# Function metadata:
#   Inputs: name (str), msg (str)  # dtype=str
#   Sample:
#       sample_output = fail("sample test", "oops")  # dtype=NoneType (prints error)
def fail(name, msg):
    print(f"[FAIL] {name}: {msg}")


# ============================================================
# TEST 1 — Blinds posted correctly
# ============================================================
# Function metadata:
#   Inputs: None; uses new SimpleHoldemEnv internally
#   Sample:
#       sample_output = test_blinds()  # dtype=NoneType (prints PASS/FAIL)
def test_blinds():
    env = SimpleHoldemEnv()
    s = env.new_hand()
    sb, bb = s.sb_player, s.bb_player

    if s.stacks[sb] != env.stack_size - env.sb:
        fail("Blinds posted (SB)", "SB blind not deducted")
        return
    if s.stacks[bb] != env.stack_size - env.bb:
        fail("Blinds posted (BB)", "BB blind not deducted")
        return
    if s.pot != env.sb + env.bb:
        fail("Blinds posted (Pot)", "Pot does not equal SB+BB")
        return

    ok("Blinds posted correctly")


# ============================================================
# TEST 2 — SB acts first preflop
# ============================================================
# Function metadata:
#   Inputs: None  # dtype=NoneType
#   Sample:
#       sample_output = test_sb_acts_first()  # dtype=NoneType
def test_sb_acts_first():
    env = SimpleHoldemEnv()
    s = env.new_hand()
    if s.to_act != s.sb_player:
        fail("SB acts first", "to_act does not equal SB")
        return
    ok("SB acts first preflop")


# ============================================================
# TEST 3 — SB CALL returns action to BB (key fix)
# ============================================================
# Function metadata:
#   Inputs: None
#   Sample:
#       sample_output = test_sb_call_returns_to_bb()  # dtype=NoneType
def test_sb_call_returns_to_bb():
    env = SimpleHoldemEnv()
    s = env.new_hand()
    sb = s.sb_player
    bb = s.bb_player

    # SB acts: CALL
    s = env.step(s, ACTION_CALL)

    if s.to_act != bb:
        fail("SB CALL returns to BB", f"Expected to_act = {bb}, got {s.to_act}")
        return
    ok("SB CALL correctly passes action to BB")


# ============================================================
# TEST 4 — BB CHECK closes preflop and deals flop
# ============================================================
# Function metadata:
#   Inputs: None
#   Sample:
#       sample_output = test_bb_call_deals_flop()  # dtype=NoneType
def test_bb_call_deals_flop():
    env = SimpleHoldemEnv()
    s = env.new_hand()
    sb, bb = s.sb_player, s.bb_player

    # SB calls
    s = env.step(s, ACTION_CALL)

    # BB checks
    s = env.step(s, ACTION_CHECK)

    if s.street != STREET_FLOP:
        fail("BB CHECK deals flop", "Street did not advance to FLOP")
        return
    if len(s.board) != 3:
        fail("BB CHECK deals flop", "Board does not have 3 cards")
        return

    ok("BB CHECK ends preflop and deals flop")


# ============================================================
# TEST 5 — Postflop action: BB acts first
# ============================================================
# Function metadata:
#   Inputs: None
#   Sample:
#       sample_output = test_postflop_bb_first()  # dtype=NoneType
def test_postflop_bb_first():
    env = SimpleHoldemEnv()
    s = env.new_hand()

    # SB calls -> BB CHECKs -> flop dealt
    s = env.step(s, ACTION_CALL)
    s = env.step(s, ACTION_CHECK)

    if s.street != STREET_FLOP:
        fail("Postflop BB first", "Not on flop yet")
        return

    if s.to_act != s.bb_player:
        fail("Postflop BB first", "BB did not act first on flop")
        return

    ok("Postflop BB acts first")


# ============================================================
# TEST 6 — Fold immediately ends hand and awards pot
# ============================================================
# Function metadata:
#   Inputs: None
#   Sample:
#       sample_output = test_fold_ends_hand()  # dtype=NoneType
def test_fold_ends_hand():
    env = SimpleHoldemEnv()
    s = env.new_hand()
    sb = s.sb_player

    # SB folds immediately
    s = env.step(s, ACTION_FOLD)

    if not s.terminal:
        fail("Fold ends hand", "Hand not marked terminal after fold")
        return
    if s.winner != (1 - sb):
        fail("Fold winner", "Winner not assigned to opponent")
        return
    ok("Fold ends hand and awards pot")


# ============================================================
# TEST 7 — Simple raise increases pot and changes aggressor
# ============================================================
# Function metadata:
#   Inputs: None
#   Sample:
#       sample_output = test_raise_logic()  # dtype=NoneType
def test_raise_logic():
    env = SimpleHoldemEnv()
    s0 = env.new_hand()
    sb = s0.sb_player

    pot0 = s0.pot
    contrib0_sb = s0.contrib[sb]

    # SB raises small preflop
    s = env.step(s0, ACTION_RAISE_SMALL)

    if s.last_aggressor != sb:
        fail("Raise aggressor", "last_aggressor incorrect")
        return
    if s.to_act != (1 - sb):
        fail("Raise to_act", "Opponent should act after raise")
        return

    # Check contrib and pot changed consistently
    if s.contrib[sb] <= contrib0_sb:
        fail("Raise contrib", "SB contribution did not increase")
        return
    if s.pot <= pot0:
        fail("Raise pot", "Pot did not increase after raise")
        return

    ok("Raise logic correct preflop")


# ============================================================
# TEST 8 — Transition from flop -> turn -> river (check-check)
# ============================================================
# Function metadata:
#   Inputs: None
#   Sample:
#       sample_output = test_street_progression_check_check()  # dtype=NoneType
def test_street_progression_check_check():
    env = SimpleHoldemEnv()
    s = env.new_hand()

    # SB CALL -> BB CHECK -> FLOP
    s = env.step(s, ACTION_CALL)
    s = env.step(s, ACTION_CHECK)

    # Flop betting round: BB CHECK, SB CHECK
    s = env.step(s, ACTION_CHECK)
    s = env.step(s, ACTION_CHECK)

    if s.street != STREET_TURN:
        fail("Flop->Turn", "Turn not dealt after flop checks")
        return

    # Turn betting round: BB CHECK, SB CHECK
    s = env.step(s, ACTION_CHECK)
    s = env.step(s, ACTION_CHECK)

    if s.street != STREET_RIVER:
        fail("Turn->River", "River not dealt after turn checks")
        return

    ok("Street progression (check-check) correct (Flop->Turn->River)")


# ============================================================
# TEST 9 — All-in from both players -> runout to showdown
# ============================================================
# Function metadata:
#   Inputs: None
#   Sample:
#       sample_output = test_allin_runout()  # dtype=NoneType
def test_allin_runout():
    env = SimpleHoldemEnv()
    s = env.new_hand()

    # SB CALL -> BB raises ALL-IN -> SB CALL ALL-IN
    sb = s.sb_player
    bb = s.bb_player

    s = env.step(s, ACTION_CALL)            # SB calls BB
    s = env.step(s, ACTION_ALL_IN)          # BB shoves
    s = env.step(s, ACTION_ALL_IN)          # SB calls all-in

    if not s.terminal:
        fail("All-in runout", "Hand should auto-runout to showdown")
        return
    if s.street != STREET_RIVER:
        fail("All-in street", "Board not fully dealt for showdown")
        return
    if len(s.board) != 5:
        fail("All-in board", "Board should have 5 cards at showdown")
        return

    ok("All-in -> full runout -> showdown works")


# ============================================================
# TEST 10 — terminal_payoff produces positive gain for winner
# ============================================================
# Function metadata:
#   Inputs: None
#   Sample:
#       sample_output = test_terminal_payoff()  # dtype=NoneType
def test_terminal_payoff():
    env = SimpleHoldemEnv()
    s = env.new_hand()

    hero = 0

    # Force terminal result
    s.winner = hero
    s.terminal = True
    s.stacks[hero] += s.pot
    s.pot = 0

    payoff = env.terminal_payoff(s, hero)

    if payoff <= 0:
        fail("terminal_payoff", "Hero should have positive payoff")
        return

    ok("terminal_payoff returns positive value for winner")


# ============================================================
# TEST 11 — legal_actions for SB preflop (facing BB)
# ============================================================
# Function metadata:
#   Inputs: None
#   Sample:
#       sample_output = test_legal_actions_sb_preflop()  # dtype=NoneType
def test_legal_actions_sb_preflop():
    env = SimpleHoldemEnv()
    s = env.new_hand()
    sb = s.sb_player

    actions = env.legal_actions(s)

    if ACTION_CALL not in actions:
        fail("SB preflop actions", "CALL missing")
        return
    # SB is facing a bet (BB), so FOLD must be allowed
    if ACTION_FOLD not in actions:
        fail("SB preflop actions", "FOLD should be allowed facing BB")
        return
    # Some raise or all-in must be allowed if SB has chips
    if all(a not in actions for a in [ACTION_RAISE_SMALL, ACTION_RAISE_MEDIUM, ACTION_ALL_IN]):
        fail("SB preflop actions", "No raise/all-in option found")
        return

    ok("legal_actions SB preflop looks correct")


# ============================================================
# TEST 12 — legal_actions for BB after SB calls (no fold)
# ============================================================
# Function metadata:
#   Inputs: None
#   Sample:
#       sample_output = test_legal_actions_bb_after_sb_call()  # dtype=NoneType
def test_legal_actions_bb_after_sb_call():
    env = SimpleHoldemEnv()
    s = env.new_hand()
    sb, bb = s.sb_player, s.bb_player

    # SB calls first
    s = env.step(s, ACTION_CALL)

    actions_bb = env.legal_actions(s)

    if ACTION_FOLD in actions_bb:
        fail("BB actions after SB call", "BB should NOT have FOLD when no bet outstanding")
        return
    if ACTION_CHECK not in actions_bb:
        fail("BB actions after SB call", "CHECK should be available")
        return

    ok("legal_actions BB after SB call (no fold) correct")


# ============================================================
# TEST 13 — Fold only when facing a bet (postflop)
# ============================================================
# Function metadata:
#   Inputs: None
#   Sample:
#       sample_output = test_no_fold_when_no_bet_postflop()  # dtype=NoneType
def test_no_fold_when_no_bet_postflop():
    env = SimpleHoldemEnv()
    s = env.new_hand()

    # Preflop: SB CALL, BB CHECK -> FLOP
    s = env.step(s, ACTION_CALL)
    s = env.step(s, ACTION_CHECK)

    if s.street != STREET_FLOP:
        fail("No fold postflop", "Not on flop yet")
        return

    # BB acts first on flop, no bet outstanding
    actions_bb = env.legal_actions(s)

    if ACTION_FOLD in actions_bb:
        fail("No fold postflop", "Fold should not be legal when to_call=0")
        return

    ok("Fold not available postflop when no bet is outstanding")


# ============================================================
# TEST 14 — actions_this_street resets on new street
# ============================================================
# Function metadata:
#   Inputs: None
#   Sample:
#       sample_output = test_actions_this_street_reset()  # dtype=NoneType
def test_actions_this_street_reset():
    env = SimpleHoldemEnv()
    s = env.new_hand()

    # SB CALL -> BB CHECK -> FLOP
    s = env.step(s, ACTION_CALL)
    s = env.step(s, ACTION_CHECK)

    if s.actions_this_street != 0:
        fail("actions_this_street reset (flop)", "Should be 0 at start of flop")
        return

    # Flop: BB CHECK
    s = env.step(s, ACTION_CHECK)
    if s.actions_this_street != 1:
        fail("actions_this_street increment", "Should be 1 after first flop action")
        return

    # Flop: SB CHECK -> should move to TURN and reset
    s = env.step(s, ACTION_CHECK)
    if s.street != STREET_TURN:
        fail("Flop->Turn (actions_this_street)", "Turn not dealt")
        return
    if s.actions_this_street != 0:
        fail("actions_this_street reset (turn)", "Should be 0 at start of turn")
        return

    ok("actions_this_street increments and resets correctly")


# ============================================================
# TEST 15 — bet + call ends street postflop (not only check-check)
# ============================================================
# Function metadata:
#   Inputs: None
#   Sample:
#       sample_output = test_bet_call_ends_street_postflop()  # dtype=NoneType
def test_bet_call_ends_street_postflop():
    env = SimpleHoldemEnv()
    s = env.new_hand()

    # Preflop: SB CALL -> BB CHECK -> FLOP
    s = env.step(s, ACTION_CALL)
    s = env.step(s, ACTION_CHECK)

    if s.street != STREET_FLOP:
        fail("bet-call street", "Not on flop yet")
        return

    # On flop: BB RAISE (bet), SB CALL
    s = env.step(s, ACTION_RAISE_SMALL)
    s = env.step(s, ACTION_CALL)

    if s.street != STREET_TURN:
        fail("bet-call street", "Street did not advance to TURN after bet+call")
        return

    ok("bet + call ends street postflop")


# ============================================================
# TEST 16 — legal_actions empty for all-in player
# ============================================================
# Function metadata:
#   Inputs: None
#   Sample:
#       sample_output = test_allin_player_has_no_actions()  # dtype=NoneType
def test_allin_player_has_no_actions():
    env = SimpleHoldemEnv()
    s = env.new_hand()

    # Manually zero out SB stack to simulate all-in / broke state
    s.stacks[s.sb_player] = 0.0

    actions = env.legal_actions(s)
    if actions != []:
        fail("all-in legal_actions", "Player with 0 stack should have no actions")
        return

    ok("all-in player (0 stack) has no legal actions")


# ============================================================
# TEST 17 — random passive hand is zero-sum and stacks non-negative
# ============================================================
# Function metadata:
#   Inputs: None
#   Sample:
#       sample_output = test_zero_sum_and_non_negative()  # dtype=NoneType
def test_zero_sum_and_non_negative():
    env = SimpleHoldemEnv()
    s = env.new_hand()

    # Passive policy: always CALL if possible, else FOLD if allowed, else first action
    while not s.terminal:
        actions = env.legal_actions(s)
        if not actions:
            break
        if ACTION_CALL in actions:
            a = ACTION_CALL
        elif ACTION_CHECK in actions:
            a = ACTION_CHECK
        elif ACTION_FOLD in actions:
            a = ACTION_FOLD
        else:
            a = actions[0]
        s = env.step(s, a)

    # terminal payoff zero-sum
    p0 = env.terminal_payoff(s, 0)
    p1 = env.terminal_payoff(s, 1)
    if abs((p0 + p1)) > 1e-6:
        fail("zero-sum", f"Payoffs not zero-sum: p0={p0}, p1={p1}")
        return

    # stacks and pot non-negative
    if s.stacks[0] < 0 or s.stacks[1] < 0 or s.pot < 0:
        fail("non-negative", "Stacks or pot became negative")
        return

    ok("Random passive hand is zero-sum and non-negative")


# ============================================================
# RUN ALL TESTS
# ============================================================
if __name__ == "__main__":
    print("Running SimpleHoldemEnv extended test suite...\n")

    test_blinds()
    test_sb_acts_first()
    test_sb_call_returns_to_bb()
    test_bb_call_deals_flop()
    test_postflop_bb_first()
    test_fold_ends_hand()
    test_raise_logic()
    test_street_progression_check_check()
    test_allin_runout()
    test_terminal_payoff()
    test_legal_actions_sb_preflop()
    test_legal_actions_bb_after_sb_call()
    test_no_fold_when_no_bet_postflop()
    test_actions_this_street_reset()
    test_bet_call_ends_street_postflop()
    test_allin_player_has_no_actions()
    test_zero_sum_and_non_negative()

    print("\nAll tests completed.")
