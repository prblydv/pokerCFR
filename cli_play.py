# ---------------------------------------------------------------------------
# cli_play.py
# Clean, robust CLI for playing heads-up NLHE against the trained Deep CFR bot.
# Safe action handling, correct state transitions, better prompts, better UX.
# ---------------------------------------------------------------------------

import torch
import copy
from typing import List, Tuple

from cash_session import CashSession
from poker_env import (
    SimpleHoldemEnv,
    ACTION_FOLD,
    ACTION_CHECK,
    ACTION_CALL,
    ACTION_RAISE_SMALL,
    ACTION_RAISE_MEDIUM,
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
SUITS = ["♣", "♦", "♥", "♠"]
RANK_MAP = {11:"J", 12:"Q", 13:"K", 14:"A"}

def card_to_str(c: int) -> str:
    r = RANK_MAP.get(card_rank(c), str(card_rank(c)))
    s = SUITS[card_suit(c)]
    colored = f"{RED}{r}{s}{RESET}" if s in ["♦","♥"] else f"{BLUE}{r}{s}{RESET}"
    return colored

def board_to_str(board: List[int]) -> str:
    return f"{YELLOW}(none){RESET}" if not board else " ".join(card_to_str(c) for c in board)


# ---------------------------------------------------------
# Action names + mapping from keyboard
# ---------------------------------------------------------
ACTION_NAMES = {
    ACTION_FOLD:        "FOLD",
    ACTION_CHECK:       "CHECK",
    ACTION_CALL:        "CALL",
    ACTION_RAISE_SMALL: "RAISE SMALL",
    ACTION_RAISE_MEDIUM:"RAISE MEDIUM",
    ACTION_ALL_IN:      "ALL-IN",
}

USER_INPUT = {
    "1": ACTION_FOLD,
    "2": ACTION_CHECK,
    "3": ACTION_CALL,
    "4": ACTION_RAISE_SMALL,
    "5": ACTION_RAISE_MEDIUM,
    "6": ACTION_ALL_IN,
}


# ---------------------------------------------------------
# Load policy network
# ---------------------------------------------------------
def load_policy(state_dim: int) -> PolicyNet:
    net = PolicyNet(state_dim)
    try:
        state_dict = torch.load("models/policy.pt", map_location="cpu", weights_only=True)
    except TypeError:
        # weights_only not supported on older torch; fall back to standard load
        state_dict = torch.load("models/policy.pt", map_location="cpu")
    net.load_state_dict(state_dict)
    net.eval()
    return net


# ---------------------------------------------------------
# Bot action (masked softmax)
# ---------------------------------------------------------
def choose_bot_action(policy_net, state, player, legal):
    x = encode_state(state, player).float().unsqueeze(0)

    with torch.no_grad():
        logits = policy_net(x).squeeze(0)

    mask = torch.full((NUM_ACTIONS,), -1e9)
    for a in legal:
        mask[a] = 0.0

    probs = torch.softmax(logits + mask, dim=-1)
    # Stochastic: sample from the masked policy
    action = torch.multinomial(probs, 1).item()

    if action not in legal:  # safety fallback
        action = legal[0]

    return action


def bot_action_probs(policy_net, state, player, legal) -> List[Tuple[int, float]]:
    """Return masked softmax probabilities for legal actions."""
    x = encode_state(state, player).float().unsqueeze(0)
    with torch.no_grad():
        logits = policy_net(x).squeeze(0)
    mask = torch.full((NUM_ACTIONS,), -1e9)
    for a in legal:
        mask[a] = 0.0
    probs = torch.softmax(logits + mask, dim=-1)
    return [(a, float(probs[a])) for a in legal]


# ---------------------------------------------------------
# ONE HAND OF PLAY
# ---------------------------------------------------------
def play_one_hand(policy_net, env, state, debug_bot_probs=False, record_human_states=False):
    """
    Human sits in seat 0; remaining seats are bots driven by the policy net.
    """
    HUMAN = 0
    num_players = getattr(env, "num_players", 2)
    bots = [i for i in range(num_players) if i != HUMAN]
    human_snapshots = []

    print(f"
{YELLOW}{BOLD}========== NEW HAND =========={RESET}")
    print(f"Your Cards: {card_to_str(state.hole[HUMAN][0])} {card_to_str(state.hole[HUMAN][1])}")
    stacks_str = ", ".join(f"P{idx}={state.stacks[idx]:.2f}" for idx in range(num_players))
    print(f"Stacks: {stacks_str}")
    print(f"Pot: {state.pot:.2f}")
    print(f"Dealer/Button: P{state.button_player}")
    print(f"Small Blind:   P{state.sb_player}")
    print(f"Big Blind:     P{state.bb_player}")
    print(f"{YELLOW}=============================== {RESET}
")

    while not state.terminal:
        legal = env.legal_actions(state)

        if not legal:
            print(f"{RED}Engine error: no legal actions. Advancing street...{RESET}")
            state = env.step(state, ACTION_CHECK)
            continue

        print(f"{BLUE}{BOLD}--------------------------------{RESET}")
        print(f"Street: {state.street}")
        print(f"Board:  {board_to_str(state.board)}")
        print(f"Pot:    {state.pot:.2f}")
        stacks_line = ", ".join(f"P{idx}={state.stacks[idx]:.2f}" for idx in range(num_players))
        print(f"Stacks: {stacks_line}")
        print(f"To act: {'YOU' if state.to_act == HUMAN else f'BOT P{state.to_act}'}
")

        if state.to_act == HUMAN:
            if record_human_states:
                human_snapshots.append(copy.deepcopy(state))
            print(f"{WHITE}{BOLD}Your options:{RESET}")
            for a in legal:
                print(f"  {a}: {ACTION_NAMES[a]}")
            print(f"Your Cards: {card_to_str(state.hole[HUMAN][0])} {card_to_str(state.hole[HUMAN][1])}")
            print("
Input: 1=fold, 2=check, 3=call, 4=raise small, 5=raise medium, 6=all-in
")

            action = None
            while action not in legal:
                key = input('Your action -> ').strip().lower()
                if key not in USER_INPUT:
                    print(f"{RED}Invalid key.{RESET}")
                    continue
                mapped = USER_INPUT[key]
                if mapped not in legal:
                    print(f"{RED}Not legal now.{RESET}")
                    continue
                action = mapped

            print(f"You -> {ACTION_NAMES[action]}
")
        else:
            if debug_bot_probs:
                probs_list = bot_action_probs(policy_net, state, state.to_act, legal)
                prob_str = " | ".join(f"{ACTION_NAMES[a]} {p:.2f}" for a, p in probs_list)
                print(f"[Bot P{state.to_act} probs] {prob_str}")
            action = choose_bot_action(policy_net, state, state.to_act, legal)
            print(f"Bot P{state.to_act} -> {ACTION_NAMES[action]}
")

        state = env.step(state, action)

    print(f"{YELLOW}{BOLD}========== HAND FINISHED =========={RESET}")

    human_profit = state.stacks[HUMAN] - state.initial_stacks[HUMAN]
    if state.winner == -1:
        print(f"{YELLOW}Result: Split pot{RESET}")
    elif state.winner == HUMAN:
        print(f"{GREEN}You win {human_profit:+.2f}{RESET}")
    else:
        print(f"{RED}Player {state.winner} wins; your profit {human_profit:+.2f}{RESET}")

    print(f"Final Board: {board_to_str(state.board)}")
    for bot_id in bots:
        bc1, bc2 = state.hole[bot_id]
        print(f"Bot P{bot_id} Cards: {card_to_str(bc1)} {card_to_str(bc2)}")

    print(f"{YELLOW}{BOLD}=============================={RESET}
")
    return state, human_snapshots



# ---------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------
def main():
    env = SimpleHoldemEnv()
    session = CashSession(env)

    tmp = session.start_hand()
    state_dim = encode_state(tmp, 0).shape[0]

    policy_net = load_policy(state_dim)

    print("\nStarting cash session...")
    print("Initial stacks:", session.get_stacks())

    while True:
        c = input("\nPlay next hand? (y/n): ").strip().lower()
        if c != "y":
            break

        state = session.start_hand()
        state, snapshots = play_one_hand(policy_net, env, state, debug_bot_probs=True, record_human_states=True)
        session.apply_results(state)

        print("Updated stacks:", session.get_stacks())

        # Debug replay option
        if snapshots:
            replay = input("Replay last human decision with bot probs? ( - to replay / any other key to skip ): ").strip()
            if replay == "-":
                replay_state = snapshots[-1]
                print(f"{YELLOW}Replaying from your last decision (debug mode, results not applied to session){RESET}")
                _ = play_one_hand(policy_net, env, replay_state, debug_bot_probs=True, record_human_states=False)

    print("\nSession ended.")
    print("Final stacks:", session.get_stacks())


if __name__ == "__main__":
    main()
