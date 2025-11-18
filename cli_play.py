# # # cli_play.py
# # import logging
# # from typing import Dict

# # from poker_env import (
# #     SimpleHoldemEnv,
# #     ACTION_FOLD,
# #     ACTION_CALL,
# #     ACTION_HALF_POT,
# #     ACTION_POT,
# #     ACTION_ALL_IN,
# # )
# # from deep_cfr_trainer import DeepCFRTrainer
# # from abstraction import card_rank, card_suit, encode_state
# # from config import DEVICE

# # logger = logging.getLogger("DeepCFR_CLI")

# # SUIT_SYMBOLS = ["♣", "♦", "♥", "♠"]


# # def card_to_str(card: int) -> str:
# #     r = card_rank(card)
# #     rank_map = {
# #         11: "J",
# #         12: "Q",
# #         13: "K",
# #         14: "A"
# #     }
# #     r_str = rank_map.get(r, str(r))
# #     s_str = SUIT_SYMBOLS[card_suit(card)]
# #     return f"{r_str}{s_str}"


# # def board_to_str(board) -> str:
# #     if not board:
# #         return "(none)"
# #     return " ".join(card_to_str(c) for c in board)


# # ACTION_NAMES = {
# #     ACTION_FOLD: "FOLD",
# #     ACTION_CALL: "CALL",
# #     ACTION_HALF_POT: "BET 0.5 POT",
# #     ACTION_POT: "BET POT",
# #     ACTION_ALL_IN: "ALL-IN",
# # }

# # # For user input mapping
# # USER_INPUT_MAP: Dict[str, int] = {
# #     "f": ACTION_FOLD,
# #     "c": ACTION_CALL,
# #     "h": ACTION_HALF_POT,
# #     "p": ACTION_POT,
# #     "a": ACTION_ALL_IN,
# # }


# # def bot_choose_action(trainer: DeepCFRTrainer, state, player: int):
# #     import torch
# #     x = encode_state(state, player).to(DEVICE)
# #     with torch.no_grad():
# #         logp = trainer.policy_net(x.unsqueeze(0)).squeeze(0)
# #     probs = torch.exp(logp)

# #     legal_actions = trainer.env.legal_actions(state)
# #     import torch
# #     mask = torch.zeros(len(probs), dtype=torch.float32, device=DEVICE)
# #     mask[legal_actions] = 1.0
# #     probs = probs * mask
# #     if probs.sum().item() <= 0:
# #         probs = mask / mask.sum()
# #     else:
# #         probs = probs / probs.sum()

# #     # sample action
# #     probs_np = probs.cpu().numpy()
# #     import random
# #     r = random.random()
# #     cum = 0
# #     for i, p in enumerate(probs_np):
# #         cum += p
# #         if r <= cum:
# #             return i
# #     return legal_actions[-1]


# # def play_cli_hand(trainer: DeepCFRTrainer):
# #     env = trainer.env
# #     s = env.new_hand()

# #     # You = player 0 always (for clarity)
# #     HUMAN = 0
# #     BOT = 1

# #     logger.info("=== NEW HAND ===")
# #     logger.info(f"Your hole cards: {card_to_str(s.hole[HUMAN][0])}  {card_to_str(s.hole[HUMAN][1])}")

# #     while not s.terminal:
# #         print()
# #         print(f"--- STREET {s.street} ---")
# #         print(f"Board: {board_to_str(s.board)}")
# #         print(f"Pot: {s.pot:.2f}")
# #         print(f"Stacks: You={s.stacks[HUMAN]:.2f}, Bot={s.stacks[BOT]:.2f}")
# #         print(f"Current bet facing you: {s.current_bet:.2f}")

# #         if s.to_act == HUMAN:
# #             # Ask the user for input
# #             legal = env.legal_actions(s)
# #             legal_str = ", ".join([f"{code} ({ACTION_NAMES[code]})" for code in legal])
# #             print(f"Your turn. Legal actions: {legal_str}")
# #             print("Type: f = fold, c = call/check, h = half-pot, p = pot, a = all-in")
            
# #             while True:
# #                 inp = input("Your action: ").strip().lower()
# #                 if inp in USER_INPUT_MAP:
# #                     action = USER_INPUT_MAP[inp]
# #                     if action in legal:
# #                         break
# #                     print("Illegal action in this state. Try again.")
# #                 else:
# #                     print("Invalid input. Try: f/c/h/p/a.")

# #             print(f"You choose: {ACTION_NAMES[action]}")
# #             s = env.step(s, action)

# #         else:
# #             # Bot acts
# #             a = bot_choose_action(trainer, s, BOT)
# #             print(f"Bot chooses: {ACTION_NAMES[a]}")
# #             s = env.step(s, a)

# #     # Show result
# #     payoff = env.terminal_payoff(s, HUMAN)
# #     print("\n=== HAND FINISHED ===")
# #     if s.winner == -1:
# #         print("Result: Split pot.")
# #     elif s.winner == HUMAN:
# #         print("Result: YOU WON")
# #     else:
# #         print("Result: BOT WON")

# #     print(f"Your payoff this hand: {payoff:.2f}")
# #     print(f"Final board: {board_to_str(s.board)}")
# #     print()


# # def run_cli():
# #     """
# #     Loads the trained models and launches interactive play.
# #     """
# #     # Load environment + trainer
# #     env = SimpleHoldemEnv()
# #     ex = env.new_hand()
# #     from abstraction import encode_state
# #     state_dim = encode_state(ex, 0).shape[0]

# #     trainer = DeepCFRTrainer(env, state_dim)

# #     # Load models
# #     import torch
# #     trainer.adv_nets[0].load_state_dict(torch.load("models/adv_p0.pt", map_location="cpu"))
# #     trainer.adv_nets[1].load_state_dict(torch.load("models/adv_p1.pt", map_location="cpu"))
# #     trainer.policy_net.load_state_dict(torch.load("models/policy.pt", map_location="cpu"))

# #     print("\nDeep CFR Poker Bot - CLI Mode")
# #     print("------------------------------------")
# #     print("Type your actions each street:")
# #     print("   f = fold")
# #     print("   c = call/check")
# #     print("   h = half-pot bet")
# #     print("   p = pot bet")
# #     print("   a = all-in")

# #     print("\nType 'q' anytime to quit.\n")

# #     # Play infinite hands until user quits
# #     while True:
# #         cmd = input("Play a hand? (y/n): ").strip().lower()
# #         if cmd == "y":
# #             play_cli_hand(trainer)
# #         else:
# #             print("Goodbye.")
# #             break


# # if __name__ == "__main__":
# #     run_cli()
# # cli_play.py

# import torch
# import logging
# from typing import Dict

# from poker_env import (
#     SimpleHoldemEnv,
#     ACTION_FOLD,
#     ACTION_CALL,
#     ACTION_HALF_POT,
#     ACTION_POT,
#     ACTION_ALL_IN,
#     NUM_ACTIONS,
# )
# from abstraction import encode_state, card_rank, card_suit
# from deep_cfr_trainer import DeepCFRTrainer
# from networks import PolicyNet
# from config import DEVICE


# # ===== Card formatting =====

# SUITS = ["♣", "♦", "♥", "♠"]
# RANK_MAP = {11: "J", 12: "Q", 13: "K", 14: "A"}

# def card_to_str(c: int) -> str:
#     r = card_rank(c)
#     r_str = RANK_MAP.get(r, str(r))
#     s_str = SUITS[card_suit(c)]
#     return f"{r_str}{s_str}"

# def board_to_str(board):
#     return " ".join(card_to_str(c) for c in board) if board else "(none)"


# # ===== Action names =====

# ACTION_NAMES = {
#     ACTION_FOLD: "FOLD",
#     ACTION_CALL: "CALL/CHECK",
#     ACTION_HALF_POT: "BET 0.5 POT",
#     ACTION_POT: "BET POT",
#     ACTION_ALL_IN: "ALL-IN",
# }


# # ===== Human input mapping =====

# USER_INPUT = {
#     "f": ACTION_FOLD,
#     "c": ACTION_CALL,
#     "h": ACTION_HALF_POT,
#     "p": ACTION_POT,
#     "a": ACTION_ALL_IN,
# }


# # ===== Load trained policy =====

# def load_policy(state_dim: int):
#     net = PolicyNet(state_dim)
#     net.load_state_dict(torch.load("models/policy.pt", map_location="cpu"))
#     net.eval()
#     return net


# # ===== Bot chooses action =====

# def choose_bot_action(policy_net, state, player, legal_actions):
#     with torch.no_grad():
#         s_vec = encode_state(state, player).float().unsqueeze(0)
#         logits = policy_net(s_vec).squeeze(0)  # log-probs

#     mask = torch.full((NUM_ACTIONS,), -1e9)
#     for a in legal_actions:
#         mask[a] = 0

#     masked_logits = logits + mask
#     probs = torch.softmax(masked_logits, dim=-1)
#     action = torch.multinomial(probs, 1).item()
#     return action


# # ===== Play a single hand =====

# def play_one_hand(policy_net, trainer_env):
#     env = trainer_env
#     state = env.new_hand()

#     HUMAN = 0
#     BOT = 1

#     print("\n==============================")
#     print("          NEW HAND")
#     print("==============================")
#     print(f"Your cards: {card_to_str(state.hole[HUMAN][0])}  {card_to_str(state.hole[HUMAN][1])}")
#     print(f"Stacks: You={state.stacks[HUMAN]}, Bot={state.stacks[BOT]}")
#     print(f"Pot: {state.pot}\n")

#     while not state.terminal:
#         legal = env.legal_actions(state)

#         print("---------------------------------")
#         print(f"STREET: {state.street} | Pot: {state.pot:.2f}")
#         print(f"Board: {board_to_str(state.board)}")
#         print(f"Stacks: You={state.stacks[HUMAN]:.2f}, Bot={state.stacks[BOT]:.2f}")
#         print(f"To act: {'YOU' if state.to_act==HUMAN else 'BOT'}")

#         if state.to_act == HUMAN:

#             print("\nYour legal options:")
#             for a in legal:
#                 print(f"  {a} = {ACTION_NAMES[a]}")
#             print("Input: f=fold, c=call, h=0.5 pot, p=pot, a=all-in")

#             action = None
#             while action not in legal:
#                 key = input("Your action: ").strip().lower()
#                 if key not in USER_INPUT:
#                     print("Invalid. Enter f/c/h/p/a.")
#                     continue
#                 a = USER_INPUT[key]
#                 if a not in legal:
#                     print("Not legal in this state.")
#                     continue
#                 action = a

#             print(f"You → {ACTION_NAMES[action]}")

#         else:
#             action = choose_bot_action(policy_net, state, BOT, legal)
#             print(f"Bot → {ACTION_NAMES[action]}")

#         state = env.step(state, action)

#     print("\n===== HAND FINISHED =====")
#     if state.winner == -1:
#         print("Result: Split pot")
#     elif state.winner == HUMAN:
#         print(f"You win +{state.pot:.2f}")
#     else:
#         print(f"Bot wins +{state.pot:.2f}")

#     print(f"Board: {board_to_str(state.board)}")
#     print(f"Bot’s cards (for info): {card_to_str(state.hole[BOT][0])} {card_to_str(state.hole[BOT][1])}")
#     print("==============================\n")


# # ===== Main loop =====

# def main():
#     env = SimpleHoldemEnv()
#     example = env.new_hand()
#     state_dim = encode_state(example, 0).shape[0]

#     policy_net = load_policy(state_dim)

#     print("\nDeep CFR Poker Bot — CLI Mode")
#     print("--------------------------------")
#     print("Commands: f=fold, c=call, h=half-pot, p=pot, a=all-in")

#     while True:
#         play = input("\nPlay a hand? (y/n): ").strip().lower()
#         if play == "y":
#             play_one_hand(policy_net, env)
#         else:
#             print("Goodbye!")
#             break


# if __name__ == "__main__":
#     main()






































import torch
from typing import Dict
from cash_session import CashSession

from poker_env import (
    SimpleHoldemEnv,
    ACTION_FOLD,
    ACTION_CALL,
    ACTION_HALF_POT,
    ACTION_POT,
    ACTION_ALL_IN,
    NUM_ACTIONS,
)
from abstraction import encode_state, card_rank, card_suit
from networks import PolicyNet


# ---------- COLORS ----------
RESET  = "\033[0m"
BOLD   = "\033[1m"
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
WHITE  = "\033[97m"


# ---------- CARD FORMATTING ----------
SUITS = ["♣", "♦", "♥", "♠"]
RANK_MAP = {11:"J", 12:"Q", 13:"K", 14:"A"}

def card_to_str(c: int) -> str:
    r = RANK_MAP.get(card_rank(c), str(card_rank(c)))
    s = SUITS[card_suit(c)]

    if s in ["♦","♥"]:
        return f"{RED}{r}{s}{RESET}"
    else:
        return f"{BLUE}{r}{s}{RESET}"

def board_to_str(board):
    return " ".join(card_to_str(c) for c in board) if board else f"{YELLOW}(none){RESET}"


# ---------- ACTION NAMES ----------
ACTION_NAMES = {
    ACTION_FOLD: "FOLD",
    ACTION_CALL: "CALL/CHECK",
    ACTION_HALF_POT: "BET 0.5 POT",
    ACTION_POT: "BET POT",
    ACTION_ALL_IN: "ALL-IN",
}

USER_INPUT = {
    "f": ACTION_FOLD,
    "c": ACTION_CALL,
    "h": ACTION_HALF_POT,
    "p": ACTION_POT,
    "a": ACTION_ALL_IN,
}


# ---------- LOAD POLICY ----------
def load_policy(state_dim: int):
    net = PolicyNet(state_dim)
    net.load_state_dict(torch.load("models/policy.pt", map_location="cpu"))
    net.eval()
    return net


# ---------- BOT ACTION ----------
def choose_bot_action(policy_net, state, player, legal_actions):

    with torch.no_grad():
        vec = encode_state(state, player).float().unsqueeze(0)
        logits = policy_net(vec).squeeze(0)

    mask = torch.full((NUM_ACTIONS,), -1e9)
    for a in legal_actions:
        mask[a] = 0

    probs = torch.softmax(logits + mask, dim=-1)
    return torch.multinomial(probs, 1).item()


# ---------- PLAY ONE HAND ----------
def play_one_hand(policy_net, env, state):

    # state = env.new_hand()

    HUMAN = 0
    BOT = 1

    # HEADER
    print(f"\n{YELLOW}{BOLD}=============================================={RESET}")
    print(f"{BOLD}                    NEW HAND                  {RESET}")
    print(f"{YELLOW}=============================================={RESET}\n")

    print(f"{BOLD}Your Cards:{RESET} {card_to_str(state.hole[HUMAN][0])}  {card_to_str(state.hole[HUMAN][1])}")
    # print(f"{BOLD}Stacks:{RESET}  {GREEN}You={state.stacks[HUMAN]}{RESET}   {RED}Bot={state.stacks[BOT]}{RESET}")
    print(f"Stacks: You={state.stacks[HUMAN]}, Bot={state.stacks[BOT]}")

    print(f"{BOLD}Pot:{RESET} {YELLOW}{state.pot}{RESET}\n")

    # MAIN LOOP
    while not state.terminal:

        legal = env.legal_actions(state)

        # SECTION HEADER
        print(f"{BLUE}{BOLD}----------------------------------------------{RESET}")
        print(f"{BOLD}STREET {state.street}{RESET}")
        print(f"{BOLD}Board:{RESET}  {board_to_str(state.board)}")
        print(f"{BOLD}Pot:{RESET}    {YELLOW}{state.pot:.2f}{RESET}")
        print(f"{BOLD}Stacks:{RESET} {GREEN}You={state.stacks[HUMAN]:.2f}{RESET}   {RED}Bot={state.stacks[BOT]:.2f}{RESET}")
        print(f"{BOLD}To act:{RESET}  "
              f"{GREEN+'YOU'+RESET if state.to_act==HUMAN else RED+'BOT'+RESET}\n")

        # ---- HUMAN ----
        if state.to_act == HUMAN:

            print(f"{WHITE}{BOLD}Your options:{RESET}")
            for a in legal:
                print(f"  {a} = {ACTION_NAMES[a]}")
            print(f"{WHITE}Input: f=fold, c=call, h=half-pot, p=pot, a=all-in{RESET}")

            action = None
            while action not in legal:
                key = input(f"{GREEN}Your action → {RESET}").strip().lower()
                if key not in USER_INPUT:
                    print(f"{RED}Invalid. Use f/c/h/p/a.{RESET}")
                    continue
                a = USER_INPUT[key]
                if a not in legal:
                    print(f"{RED}Not legal in this spot.{RESET}")
                    continue
                action = a

            print(f"{GREEN}You → {ACTION_NAMES[action]}{RESET}\n")

        # ---- BOT ----
        else:
            action = choose_bot_action(policy_net, state, BOT, legal)
            print(f"{RED}Bot → {ACTION_NAMES[action]}{RESET}\n")

        # APPLY ACTION
        state = env.step(state, action)


    # ---------- RESULT ----------
    print(f"\n{YELLOW}{BOLD}=============== HAND FINISHED ==============={RESET}")

    if state.winner == -1:
        print(f"{YELLOW}Result: Split pot{RESET}")
    elif state.winner == HUMAN:
        print(f"{GREEN}You win +{state.pot:.2f}{RESET}")
    else:
        print(f"{RED}Bot wins +{state.pot:.2f}{RESET}")

    print(f"{BOLD}Final Board:{RESET} {board_to_str(state.board)}")

    b1, b2 = state.hole[BOT]
    print(f"{BOLD}Bot Cards:{RESET} {card_to_str(b1)}  {card_to_str(b2)}")
    print(f"{YELLOW}{BOLD}=============================================={RESET}\n")
    return state   # <---- CRUCIAL

# ---------- MAIN ----------
def main():
    env = SimpleHoldemEnv()
    session = CashSession(env, starting_stacks=(200, 200))

    # compute state_dim only once
    example = session.start_hand()
    state_dim = encode_state(example, 0).shape[0]

    policy_net = load_policy(state_dim)

    print("\nStarting CASH GAME session...")
    print(f"Initial stacks: {session.get_stacks()}")

    while True:
        play = input("\nPlay next hand? (y/n): ").strip().lower()
        if play != "y":
            print("Session ended.")
            print("Final stacks:", session.get_stacks())
            break

        # PLAY CASH-GAME HAND
        state = session.start_hand()
        state = play_one_hand(policy_net, env, state)   # pass state explicitly

        # APPLY RESULTS
        session.apply_results(state)

        print("Updated stacks:", session.get_stacks())


if __name__ == "__main__":
    main()
