def test_preflop_raise_then_call_closes_round():
    """
    Scenario (HU, blinds 1/2):

    - Blinds posted: SB=1, BB=2 → pot=3, contrib=[1,2], current_bet=2, SB acts first.
    - SB CALLS → contrib=[2,2], pot=4, BB now acts.
    - BB POT-bets → pot=8, contrib=[2,6], current_bet=6.
    - SB HALF-POT raises → pot=18, contrib=[12,6], current_bet=12.
    - BB CALLS → betting round is over, we MUST go straight to FLOP.
    """

    from engine import (
        SimpleHoldemEnv9,
        ACTION_CALL, ACTION_HALF_POT, ACTION_POT,
        STREET_PREFLOP, STREET_FLOP
    )

    env = SimpleHoldemEnv9()

    # New hand with blinds posted: contrib=[1,2], current_bet=2, pot=3
    s = env.new_hand()
    assert s.street == 0  # STREET_PREFLOP is always 0
    assert s.contrib == [1.0, 2.0]
    assert s.pot == 3.0
    assert s.to_act == s.sb_player == 0  # SB acts first

    # 1) SB CALLS → limp
    s = env.step(s, ACTION_CALL)
    assert s.street == 0  # STREET_PREFLOP
    assert s.contrib == [2.0, 2.0]
    assert s.pot == 4.0
    assert s.to_act == s.bb_player == 1

    # 2) BB POT-bets
    s = env.step(s, ACTION_POT)
    assert s.street == 0  # STREET_PREFLOP
    assert s.pot == 8.0        # 4 + 4
    assert s.contrib == [2.0, 6.0]
    assert s.current_bet == 6.0
    assert s.to_act == 0       # back to SB

    # 3) SB HALF-POT raise
    s = env.step(s, ACTION_HALF_POT)
    assert s.street == 0  # STREET_PREFLOP
    assert s.pot == 18.0       # 8 + 10
    assert s.contrib == [12.0, 6.0]
    assert s.current_bet == 12.0
    assert s.to_act == 1       # BB faces the raise

    # 4) BB CALLS → round is over, MUST go to FLOP
    s = env.step(s, ACTION_CALL)

    assert s.street == 1, "After final call, should be on FLOP"  # STREET_FLOP is always 1
    assert s.pot == 24.0,          "Pot should be 24 going to flop"
    assert s.contrib == [0.0, 0.0] # contrib reset for new street
    assert s.to_act == s.sb_player == 0, "SB acts first on flop"

    print("✓ test_preflop_raise_then_call_closes_round passed.")
def test_preflop_call_then_check_advances_to_flop():
    """
    Correct HU preflop logic:

    - Blinds posted: SB=1, BB=2 → contrib=[1,2]
    - SB acts first.
    - SB CALLS → contrib=[2,2], pot=4, current_bet stays 2 (the live bet).
    - BB CHECKS → closes betting.
    - Betting round ends → FLOP is dealt.
    """

    from engine import (
        SimpleHoldemEnv9,
        ACTION_CALL, ACTION_CHECK,
        STREET_PREFLOP, STREET_FLOP
    )

    env = SimpleHoldemEnv9()

    # SB=0, BB=1, contrib=[1,2]
    s = env.new_hand()
    assert s.street == STREET_PREFLOP
    assert s.to_act == 0
    assert s.contrib == [1.0, 2.0]

    # ---- SB CALLS ----
    s2 = env.step(s, ACTION_CALL)

    # Still preflop, BB must act
    assert s2.street == STREET_PREFLOP
    assert s2.to_act == 1
    assert s2.contrib == [2.0, 2.0]
    assert s2.pot == 4.0

    # ---- BB CHECKS ----
    s3 = env.step(s2, ACTION_CHECK)

    # MUST advance to the FLOP
    assert s3.street == STREET_FLOP, \
        f"Expected FLOP but got street={s3.street}"

    # contrib must reset
    assert s3.contrib == [0.0, 0.0]

    # SB acts first on flop
    assert s3.to_act == 0

    print("✓ test_preflop_call_then_check_advances_to_flop passed.")
def test_river_requires_two_checks_before_showdown():
    """
    On the river, betting round must only end after BOTH players check.
    The current bug ends river betting after only one check.
    This test forces correct NLHE behaviour.
    """

    from engine import (
        SimpleHoldemEnv9,
        ACTION_CHECK,
        STREET_RIVER,
    )

    env = SimpleHoldemEnv9()

    # Build a state directly on the river with no current bet
    s = env.new_hand()

    # Force river state manually (safe for testing)
    s.street = STREET_RIVER
    s.board = [1, 2, 3, 4, 5]
    s.current_bet = 0.0
    s.contrib = [0.0, 0.0]
    s.to_act = 0             # SB acts first on river
    s.terminal = False

    # --- FIRST CHECK (SB) ---
    s2 = env.step(s, ACTION_CHECK)

    # Should NOT be terminal yet
    assert s2.terminal is False, "River cannot end after only ONE check"
    assert s2.to_act == 1, "Turn must pass to BB after SB checks"

    # --- SECOND CHECK (BB) ---
    s3 = env.step(s2, ACTION_CHECK)

    # Now the hand MUST end
    assert s3.terminal is True, "River must end after TWO checks"
    assert s3.winner in [-1, 0, 1], "Winner must be decided at showdown"

    print("✓ test_river_requires_two_checks_before_showdown passed.")

import random
from engine import SimpleHoldemEnv9, NUM_ACTIONS
from dataclasses import dataclass

def random_legal_action(env, s):
    legal = env.legal_actions(s)
    if not legal:
        return None
    return random.choice(legal)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def assert_no_negative_stacks(env, s):
    if any(x < -1e-9 for x in s.stacks):
        raise AssertionError(f"Negative stack found: {s.stacks}")

def assert_no_zero_stack_actor(env, s):
    if not s.terminal:
        if s.stacks[s.to_act] <= 0:
            raise AssertionError(
                f"Player {s.to_act} must act with zero chips! "
                f"Stacks={s.stacks}, street={s.street}, pot={s.pot}"
            )

def assert_street_monotonic(last_street, new_street):
    if new_street < last_street:
        raise AssertionError(f"Street regressed {last_street} → {new_street}")

def assert_showdown_valid(env, s):
    if s.terminal:
        if s.pot != 0:
            raise AssertionError(f"Terminal state but pot != 0: {s.pot}")
        if s.winner not in [0, 1, -1]:
            raise AssertionError(f"Illegal winner: {s.winner}")


# ------------------------------------------------------------
# MAIN RANDOM ROLLOUT TEST
# ------------------------------------------------------------

def run_random_rollouts(num_hands=5000, max_actions=200):
    env = SimpleHoldemEnv9()
    print(f"Running {num_hands} random hands…")

    for h in range(num_hands):
        s = env.new_hand()
        last_street = s.street
        action_count = 0

        while not s.terminal:
            action = random_legal_action(env, s)
            if action is None:
                raise AssertionError(f"No legal actions in non-terminal state {s}")

            s = env.step(s, action)
            action_count += 1

            # Basic invariants
            assert_no_negative_stacks(env, s)
            assert_no_zero_stack_actor(env, s)

            # Street progression invariant
            assert_street_monotonic(last_street, s.street)
            last_street = s.street

            if action_count > max_actions:
                raise AssertionError(
                    f"Hand did not terminate in {max_actions} actions.\n"
                    f"State = street={s.street}, stacks={s.stacks}, pot={s.pot}"
                )

        # Terminal-state invariants
        assert_showdown_valid(env, s)

        if (h+1) % 50 == 0:
            print(f"  ✓ Completed {h+1}/{num_hands}")

    print("✓ All invariants passed.")
    print("✓ Engine appears logically correct.")


if __name__ == "__main__":
    test_preflop_raise_then_call_closes_round()
    test_preflop_call_then_check_advances_to_flop()
    test_river_requires_two_checks_before_showdown()
    run_random_rollouts()
