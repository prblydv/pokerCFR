# ============================================================
# manualplay_cpu.py — human-vs-human or vs-random CPU debugger
# ============================================================

from engine import (
    SimpleHoldemEnv9, GameState,
    ACTION_FOLD, ACTION_CHECK, ACTION_CALL,
    ACTION_RAISE_2X, ACTION_RAISE_3X, ACTION_HALF_POT,
    ACTION_POT, ACTION_RAISE_10BB, ACTION_ALL_IN,
)

ACTION_MAP = {
    "fold": ACTION_FOLD,
    "check": ACTION_CHECK,
    "call": ACTION_CALL,
    "r2": ACTION_RAISE_2X,
    "r3": ACTION_RAISE_3X,
    "half": ACTION_HALF_POT,
    "pot": ACTION_POT,
    "10bb": ACTION_RAISE_10BB,
    "allin": ACTION_ALL_IN,
}

ACTION_NAMES = ["FOLD","CHECK","CALL","R2X","R3X","HALF","POT","10BB","ALLIN"]


# ------------------------------------------------------------
# PRINT HUMAN-READABLE STATE
# ------------------------------------------------------------
def print_state(s: GameState):
    p0_hole = tuple(s.hole[0])
    p1_hole = tuple(s.hole[1])
    board   = s.board

    print("\n========== STATE ==========")
    print(f"P0 ({'SB' if s.sb_player == 0 else 'BB'}) hole = {p0_hole}")
    print(f"P1 ({'SB' if s.sb_player == 1 else 'BB'}) hole = {p1_hole}")

    btn = s.sb_player
    bb  = s.bb_player
    btn_hole = p0_hole if btn == 0 else p1_hole
    bb_hole  = p0_hole if bb == 0 else p1_hole

    print(f"BTN hole = {btn_hole}")
    print(f"BB hole  = {bb_hole}")

    print(f"Board    = {board}")
    print(f"Street   = {s.street}")
    print(f"Pot      = {s.pot:.2f}")
    print(f"Stacks   = P0:{s.stacks[0]:.2f}  P1:{s.stacks[1]:.2f}")
    print(f"Current Bet = {s.current_bet:.2f}")
    print(f"Contrib    = {s.contrib}")
    print(f"To act = {'P0' if s.to_act == 0 else 'P1'}")
    print("=================================")


# ------------------------------------------------------------
# HUMAN INPUT
# ------------------------------------------------------------
def ask_human_action(env, s: GameState):
    legal = env.legal_actions(s)

    print("Legal actions:")
    for name, code in ACTION_MAP.items():
        if code in legal:
            print(f"  {name}")

    while True:
        cmd = input("Your action: ").strip().lower()
        if cmd in ACTION_MAP:
            a = ACTION_MAP[cmd]
            if a in legal:
                return a
            else:
                print("Illegal. Try again.")
        else:
            print("Unknown command. Try again.")


# ------------------------------------------------------------
# BOT ACTION (random legal)
# ------------------------------------------------------------
import random
def bot_action(env, s: GameState):
    legal = env.legal_actions(s)
    return random.choice(legal)


# ------------------------------------------------------------
# PLAY ONE HAND (human vs. bot OR human vs. human)
# ------------------------------------------------------------
def play_hand(human_seat="sb", bot=False):
    """
    human_seat: "sb" or "bb"
    bot = False → human vs human
    bot = True  → human vs random bot
    """
    env = SimpleHoldemEnv9()
    s = env.new_hand()

    sb = s.sb_player
    bb = s.bb_player

    if human_seat == "sb":
        human_p = sb
    else:
        human_p = bb

    print(f"\n=== NEW HAND — You are Player {human_p} ({human_seat.upper()}) ===")

    while not s.terminal:
        print_state(s)

        p = s.to_act
        human_turn = (p == human_p)

        if human_turn:
            a = ask_human_action(env, s)
        else:
            if bot:
                a = bot_action(env, s)
                print(f"BOT ACTION = {ACTION_NAMES[a]}")
            else:
                print(f"Player {p}'s turn:")
                a = ask_human_action(env, s)

        s = env.step(s, a)

    # Terminal reached
    print_state(s)

    if s.winner == -1:
        print("RESULT: TIE")
    elif s.winner == human_p:
        print("RESULT: YOU WIN!")
    else:
        print("RESULT: YOU LOSE!")

    print("\n============================\n")


# ------------------------------------------------------------
# MAIN LOOP
# ------------------------------------------------------------
if __name__ == "__main__":
    print("Manual SimpleHoldemEnv9 Tester (CPU Version)")
    print("Type: fold | check | call | r2 | r3 | half | pot | 10bb | allin\n")

    while True:
        seat = input("Choose seat ('sb' or 'bb'): ").strip().lower()
        if seat not in ["sb", "bb"]:
            print("Invalid.")
            continue

        mode = input("Play vs bot? (y/n): ").strip().lower()
        play_hand(human_seat=seat, bot=(mode == "y"))
