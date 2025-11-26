import torch
from typing import Dict

from cash_session import CashSession
from poker_env import (
    SimpleHoldemEnv,
    ACTION_FOLD,
    ACTION_CALL,
    ACTION_RAISE_2X,
    ACTION_RAISE_2_25X,
    ACTION_RAISE_2_5X,
    ACTION_RAISE_3X,
    ACTION_RAISE_3_5X,
    ACTION_RAISE_4_5X,
    ACTION_RAISE_6X,
    ACTION_ALL_IN,
    NUM_ACTIONS,
)
from abstraction import encode_state, card_rank, card_suit
from networks import PolicyNet


# ---------------------------------------------------------
# Colors
# ---------------------------------------------------------
RESET  = "\033[0m"
BOLD   = "\033[1m"
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
WHITE  = "\033[97m"


# ---------------------------------------------------------
# Card formatting
# ---------------------------------------------------------
SUITS = ["♣","♦","♥","♠"]
RANK_MAP = {11:"J", 12:"Q", 13:"K", 14:"A"}

def card_to_str(c: int) -> str:
    r = RANK_MAP.get(card_rank(c), str(card_rank(c)))
    s = SUITS[card_suit(c)]
    if s in ["♦","♥"]:
        return f"{RED}{r}{s}{RESET}"
    return f"{BLUE}{r}{s}{RESET}"

def board_to_str(board):
    if not board:
        return f"{YELLOW}(none){RESET}"
    return " ".join(card_to_str(c) for c in board)


# ---------------------------------------------------------
# Action names
# ---------------------------------------------------------
ACTION_NAMES = {
    ACTION_FOLD:        "FOLD",
    ACTION_CALL:        "CALL/CHECK",
    ACTION_RAISE_2X:    "RAISE 2×",
    ACTION_RAISE_2_25X: "RAISE 2.25×",
    ACTION_RAISE_2_5X:  "RAISE 2.5×",
    ACTION_RAISE_3X:    "RAISE 3×",
    ACTION_RAISE_3_5X:  "RAISE 3.5×",
    ACTION_RAISE_4_5X:  "RAISE 4.5×",
    ACTION_RAISE_6X:    "RAISE 6×",
    ACTION_ALL_IN:      "ALL-IN",
}

USER_INPUT = {
    "f": ACTION_FOLD,
    "c": ACTION_CALL,
    "1": ACTION_RAISE_2X,
    "2": ACTION_RAISE_2_25X,
    "3": ACTION_RAISE_2_5X,
    "4": ACTION_RAISE_3X,
    "5": ACTION_RAISE_3_5X,
    "6": ACTION_RAISE_4_5X,
    "7": ACTION_RAISE_6X,
    "a": ACTION_ALL_IN,
}


# ---------------------------------------------------------
# Load policy
# ---------------------------------------------------------
def load_policy(state_dim: int) -> PolicyNet:
    net = PolicyNet(state_dim)
    net.load_state_dict(torch.load("models/policy.pt", map_location="cpu"))
    net.eval()
    return net


# ---------------------------------------------------------
# Bot action
# ---------------------------------------------------------
def choose_bot_action(policy_net, state, player, legal):
    if not legal:
        return None

    x = encode_state(state, player).float().unsqueeze(0)
    with torch.no_grad():
        logits = policy_net(x).squeeze(0)

    mask = torch.full((NUM_ACTIONS,), -1e9)
    for a in legal:
        mask[a] = 0

    probs = torch.softmax(logits + mask, dim=-1)
    action = torch.multinomial(probs, 1).item()

    if action not in legal:
        action = legal[0]

    return action


# ---------------------------------------------------------
# Play one hand
# ---------------------------------------------------------
def play_one_hand(policy_net, env, state):
    HUMAN = 0
    BOT   = 1

    print(f"\n{YELLOW}{BOLD}========== NEW HAND =========={RESET}")
    print(f"Your Cards: {card_to_str(state.hole[HUMAN][0])} {card_to_str(state.hole[HUMAN][1])}")
    print(f"Stacks: You={state.stacks[HUMAN]}, Bot={state.stacks[BOT]}")
    print(f"Pot: {state.pot}")

    sb = state.sb_player
    bb = state.bb_player
    dealer = sb  # SB = dealer in HU

    print(f"Dealer: {'You' if dealer == HUMAN else 'Bot'}")
    print(f"Big Blind: {'You' if bb == HUMAN else 'Bot'}")
    print(f"{YELLOW}=============================== {RESET}\n")

    # MAIN LOOP
    while not state.terminal:

        legal = env.legal_actions(state)

        # --- No legal actions -> advance street (all-in or street error) ---
        if not legal:
            state = env.step(state, ACTION_CALL)  # safe no-op
            continue

        print(f"{BLUE}{BOLD}--------------------------------{RESET}")
        print(f"STREET {state.street}")
        print(f"Board: {board_to_str(state.board)}")
        print(f"Pot:    {state.pot:.2f}")
        print(f"Stacks: You={state.stacks[HUMAN]:.2f}  Bot={state.stacks[BOT]:.2f}")
        print(f"To act: {('YOU' if state.to_act==HUMAN else 'BOT')}\n")

        # ------------- Human -------------
        if state.to_act == HUMAN:
            print(f"{WHITE}{BOLD}Your options:{RESET}")
            for a in legal:
                print(f"  {a}: {ACTION_NAMES[a]}")
            print("\nInput: f=fold, c=call, 1..7=raises, a=all-in\n")

            action = None
            while action not in legal:
                key = input("Your action → ").strip().lower()
                if key not in USER_INPUT:
                    print(f"{RED}Invalid input.{RESET}")
                    continue
                action = USER_INPUT[key]
                if action not in legal:
                    print(f"{RED}Not legal here.{RESET}")

            print(f"You → {ACTION_NAMES[action]}\n")

        # ------------- Bot -------------
        else:
            action = choose_bot_action(policy_net, state, BOT, legal)
            print(f"Bot → {ACTION_NAMES[action]}\n")

        # Apply action
        state = env.step(state, action)

    # -----------------------------------------------------------
    # FINAL RESULT
    # -----------------------------------------------------------
    print(f"{YELLOW}{BOLD}========== HAND FINISHED =========={RESET}")

    if state.winner == -1:
        print(f"{YELLOW}Result: Split pot{RESET}")
    elif state.winner == HUMAN:
        print(f"{GREEN}You win +{state.stacks[HUMAN] - state.initial_stacks[HUMAN]:.2f}{RESET}")
    else:
        print(f"{RED}Bot wins +{state.stacks[BOT] - state.initial_stacks[BOT]:.2f}{RESET}")

    print(f"Final Board: {board_to_str(state.board)}")

    bc1, bc2 = state.hole[BOT]
    print(f"Bot Cards: {card_to_str(bc1)} {card_to_str(bc2)}")

    print(f"{YELLOW}{BOLD}=============================={RESET}\n")
    return state


# ---------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------
def main():
    env = SimpleHoldemEnv()
    session = CashSession(env, starting_stacks=(200, 200))

    tmp = session.start_hand()
    state_dim = encode_state(tmp, 0).shape[0]
    policy_net = load_policy(state_dim)

    print("Starting cash session...")
    print("Initial stacks:", session.get_stacks())

    while True:
        c = input("\nPlay next hand? (y/n): ").strip().lower()
        if c != "y":
            break

        state = session.start_hand()
        state = play_one_hand(policy_net, env, state)
        session.apply_results(state)

        print("Updated stacks:", session.get_stacks())

    print("Session ended.")
    print("Final stacks:", session.get_stacks())


if __name__ == "__main__":
    main()
