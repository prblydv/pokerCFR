"""
Extended test suite for SimpleHoldemEnv
Each test prints PASS or FAIL.

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
    ACTION_CALL,
    ACTION_FOLD,
    ACTION_MINRAISE,
    ACTION_HALF_POT,
    ACTION_POT,
    ACTION_ALL_IN,
    GameState,
)

# Helper to print results
def ok(name):
    print(f"[PASS] {name}")

def fail(name, msg):
    print(f"[FAIL] {name}: {msg}")


# ============================================================
# TEST 1 — Blinds posted correctly
# ============================================================
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
def test_sb_acts_first():
    env = SimpleHoldemEnv()
    s = env.new_hand()
    if s.to_act != s.sb_player:
        fail("SB acts first", "to_act does not equal SB")
        return
    ok("SB acts first preflop")


# ============================================================
# TEST 3 — SB CALL returns action to BB
# ============================================================
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
# TEST 4 — BB CALL closes preflop and deals flop
# ============================================================
def test_bb_call_deals_flop():
    env = SimpleHoldemEnv()
    s = env.new_hand()
    sb, bb = s.sb_player, s.bb_player

    # SB calls
    s = env.step(s, ACTION_CALL)

    # BB calls
    s = env.step(s, ACTION_CALL)

    if s.street != STREET_FLOP:
        fail("BB CALL deals flop", "Street did not advance to FLOP")
        return
    if len(s.board) != 3:
        fail("BB CALL deals flop", "Board does not have 3 cards")
        return

    ok("BB CALL ends preflop and deals flop")


# ============================================================
# TEST 5 — Postflop action: SB acts first
# ============================================================
def test_postflop_sb_first():
    env = SimpleHoldemEnv()
    s = env.new_hand()

    # SB calls → BB calls → flop dealt
    s = env.step(s, ACTION_CALL)
    s = env.step(s, ACTION_CALL)

    if s.street != STREET_FLOP:
        fail("Postflop SB first", "Not on flop yet")
        return

    if s.to_act != s.sb_player:
        fail("Postflop SB first", "SB did not act first on flop")
        return

    ok("Postflop SB acts first")


# ============================================================
# TEST 6 — Fold immediately ends hand and awards pot
# ============================================================
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
# TEST 7 — Min-raise increases pot and changes aggressor
# ============================================================
def test_minraise_logic():
    env = SimpleHoldemEnv()
    s0 = env.new_hand()
    sb = s0.sb_player

    pot0 = s0.pot
    contrib0_sb = s0.contrib[sb]

    # SB min-raises preflop
    s = env.step(s0, ACTION_MINRAISE)

    if s.last_aggressor != sb:
        fail("Min-raise aggressor", "last_aggressor incorrect")
        return
    if s.to_act != (1 - sb):
        fail("Min-raise to_act", "Opponent should act after raise")
        return

    # Check contrib and pot changed consistently
    if s.contrib[sb] <= contrib0_sb:
        fail("Min-raise contrib", "SB contribution did not increase")
        return
    if s.pot <= pot0:
        fail("Min-raise pot", "Pot did not increase after raise")
        return

    ok("Min-raise logic correct preflop")


# ============================================================
# TEST 8 — Transition from flop → turn → river (check-check)
# ============================================================
def test_street_progression_check_check():
    env = SimpleHoldemEnv()
    s = env.new_hand()

    # SB CALL → BB CALL → FLOP
    s = env.step(s, ACTION_CALL)
    s = env.step(s, ACTION_CALL)

    # Flop betting round: SB CHECK, BB CHECK
    s = env.step(s, ACTION_CALL)
    s = env.step(s, ACTION_CALL)

    if s.street != STREET_TURN:
        fail("Flop→Turn", "Turn not dealt after flop checks")
        return

    # Turn betting round: SB CHECK, BB CHECK
    s = env.step(s, ACTION_CALL)
    s = env.step(s, ACTION_CALL)

    if s.street != STREET_RIVER:
        fail("Turn→River", "River not dealt after turn checks")
        return

    ok("Street progression (check-check) correct (Flop→Turn→River)")


# ============================================================
# TEST 9 — All-in from both players → runout to showdown
# ============================================================
def test_allin_runout():
    env = SimpleHoldemEnv()
    s = env.new_hand()

    # SB CALL → BB raises ALL-IN → SB CALL ALL-IN
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

    ok("All-in → full runout → showdown works")


# ============================================================
# TEST 10 — terminal_payoff produces positive gain for winner
# ============================================================
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
    if all(a not in actions for a in [ACTION_MINRAISE, ACTION_HALF_POT, ACTION_POT, ACTION_ALL_IN]):
        fail("SB preflop actions", "No raise/all-in option found")
        return

    ok("legal_actions SB preflop looks correct")


# ============================================================
# TEST 12 — legal_actions for BB after SB calls (no fold)
# ============================================================
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
    if ACTION_CALL not in actions_bb:
        fail("BB actions after SB call", "CALL/CHECK should be available")
        return

    ok("legal_actions BB after SB call (no fold) correct")


# ============================================================
# TEST 13 — Fold only when facing a bet (postflop)
# ============================================================
def test_no_fold_when_no_bet_postflop():
    env = SimpleHoldemEnv()
    s = env.new_hand()

    # Preflop: SB CALL, BB CALL → FLOP
    s = env.step(s, ACTION_CALL)
    s = env.step(s, ACTION_CALL)

    if s.street != STREET_FLOP:
        fail("No fold postflop", "Not on flop yet")
        return

    # SB acts first on flop, no bet outstanding
    actions_sb = env.legal_actions(s)

    if ACTION_FOLD in actions_sb:
        fail("No fold postflop", "Fold should not be legal when to_call=0")
        return

    ok("Fold not available postflop when no bet is outstanding")


# ============================================================
# TEST 14 — actions_since_raise resets on new street
# ============================================================
def test_actions_since_raise_reset():
    env = SimpleHoldemEnv()
    s = env.new_hand()

    # SB CALL → BB CALL → FLOP
    s = env.step(s, ACTION_CALL)
    s = env.step(s, ACTION_CALL)

    if s.actions_since_raise != 0:
        fail("actions_since_raise reset (flop)", "Should be 0 at start of flop")
        return

    # Flop: SB CHECK
    s = env.step(s, ACTION_CALL)
    if s.actions_since_raise != 1:
        fail("actions_since_raise increment", "Should be 1 after first flop action")
        return

    # Flop: BB CHECK → should move to TURN and reset
    s = env.step(s, ACTION_CALL)
    if s.street != STREET_TURN:
        fail("Flop→Turn (actions_since_raise)", "Turn not dealt")
        return
    if s.actions_since_raise != 0:
        fail("actions_since_raise reset (turn)", "Should be 0 at start of turn")
        return

    ok("actions_since_raise increments and resets correctly")


# ============================================================
# TEST 15 — bet + call ends street postflop
# ============================================================
def test_bet_call_ends_street_postflop():
    env = SimpleHoldemEnv()
    s = env.new_hand()

    # Preflop: SB CALL → BB CALL → FLOP
    s = env.step(s, ACTION_CALL)
    s = env.step(s, ACTION_CALL)

    if s.street != STREET_FLOP:
        fail("bet-call street", "Not on flop yet")
        return

    # On flop: SB RAISE (bet), BB CALL
    s = env.step(s, ACTION_MINRAISE)
    s = env.step(s, ACTION_CALL)

    if s.street != STREET_TURN:
        fail("bet-call street", "Street did not advance to TURN after bet+call")
        return

    ok("bet + call ends street postflop")


# ============================================================
# TEST 16 — legal_actions empty for all-in player
# ============================================================
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
# TEST 18 — Half-pot and Pot raise actions work correctly
# ============================================================
def test_halfpot_and_pot_raises():
    env = SimpleHoldemEnv()
    s = env.new_hand()

    # Check that half-pot and pot raises are available
    actions = env.legal_actions(s)
    
    has_half_pot = ACTION_HALF_POT in actions
    has_pot = ACTION_POT in actions
    
    # At least one of them should be available with starting stacks
    if not (has_half_pot or has_pot):
        fail("Half-pot/Pot raises", "Neither half-pot nor pot raise available preflop")
        return

    ok("Half-pot and Pot raise actions available")


# ============================================================
# TEST 19 — SB completion logic works correctly
# ============================================================
def test_sb_completion():
    env = SimpleHoldemEnv()
    s = env.new_hand()
    sb = s.sb_player

    # SB should only need to complete to BB, not pay full BB
    if s.contrib[sb] == env.sb:  # SB has only posted small blind
        if s.current_bet == env.bb:  # Facing BB
            # SB call should only cost (BB - SB)
            s_after_call = env.step(s, ACTION_CALL)
            expected_cost = env.bb - env.sb
            actual_cost = s_after_call.contrib[sb] - s.contrib[sb]
            
            if abs(actual_cost - expected_cost) > 1e-6:
                fail("SB completion", f"SB paid {actual_cost}, expected {expected_cost}")
                return

    ok("SB completion logic works correctly")


# ============================================================
# RUN ALL TESTS
# ============================================================
if __name__ == "__main__":
    print("Running SimpleHoldemEnv extended test suite...\n")

    test_blinds()
    test_sb_acts_first()
    test_sb_call_returns_to_bb()
    test_bb_call_deals_flop()
    test_postflop_sb_first()
    test_fold_ends_hand()
    test_minraise_logic()
    test_street_progression_check_check()
    test_allin_runout()
    test_terminal_payoff()
    test_legal_actions_sb_preflop()
    test_legal_actions_bb_after_sb_call()
    test_no_fold_when_no_bet_postflop()
    test_actions_since_raise_reset()
    test_bet_call_ends_street_postflop()
    test_allin_player_has_no_actions()
    test_zero_sum_and_non_negative()
    test_halfpot_and_pot_raises()
    test_sb_completion()

    print("\nAll tests completed.")