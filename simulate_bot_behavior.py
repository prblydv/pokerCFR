import torch
import numpy as np
import matplotlib.pyplot as plt

from poker_env import SimpleHoldemEnv, STREET_PREFLOP
from cash_session import CashSession
from abstraction import encode_state
from deep_cfr_trainer import DeepCFRTrainer


# =====================================================
# Hole-card canonicalization
# =====================================================
RANK_MAP = "23456789TJQKA"

def card_to_label(card):
    rank = card % 13
    suit = card // 13
    return RANK_MAP[rank], suit

def hand_label(cards):
    r1, s1 = card_to_label(cards[0])
    r2, s2 = card_to_label(cards[1])

    # sort by rank descending
    if RANK_MAP.index(r1) < RANK_MAP.index(r2):
        r1, r2 = r2, r1
        s1, s2 = s2, s1

    suited = "s" if s1 == s2 else "o"
    if r1 == r2:
        return r1 + r2   # pocket pairs
    else:
        return r1 + r2 + suited


# =====================================================
# Logger
# =====================================================
def init_logger():
    return {"hands": [], "actions": []}


# =====================================================
# Load policy
# =====================================================
def load_policy(env):
    example = env.new_hand()
    dim = encode_state(example, 0).shape[0]
    trainer = DeepCFRTrainer(env, dim)
    trainer.policy_net.load_state_dict(torch.load("models/policy.pt", map_location="cpu"))
    return trainer


# =====================================================
# Choose action from policy
# =====================================================
def choose_action(policy_net, state, env):
    legal = env.legal_actions(state)

    x = encode_state(state, state.to_act).unsqueeze(0)
    with torch.no_grad():
        logp = policy_net(x)[0]

    mask = torch.full((5,), -1e9)
    for a in legal:
        mask[a] = 0

    probs = torch.softmax(logp + mask, dim=-1)
    return torch.multinomial(probs, 1).item()


# =====================================================
# PLAY ONE HAND – but LOG ONLY FIRST PREFLOP ACTION OF BOT
# =====================================================
BOT = 1

def play_hand(policy, env, session, logger):
    s = session.start_hand()
    preflop_logged = False

    while not s.terminal:
        to_act = s.to_act
        action = choose_action(policy.policy_net, s, env)

        # ------ LOGGING FILTER -------
        if (not preflop_logged and
            to_act == BOT and
            s.street == STREET_PREFLOP):
            label = hand_label(s.hole[BOT])
            logger["hands"].append(label)
            logger["actions"].append(action)
            preflop_logged = True
        # -----------------------------

        s = env.step(s, action)

    session.apply_results(s)


# =====================================================
# Scatter plot (clean!)
# =====================================================
# =====================================================
# GTO-Wizard Style 13×13 Heatmap
# =====================================================

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

# Ranks sorted highest → lowest for grid
GRID_RANKS = "AKQJT98765432"
GRID_INDEX = {r: i for i, r in enumerate(GRID_RANKS)}

def grid_coords(hand):
    """
    Convert canonical hand label to (row, col) in a 13×13 matrix.
    - AA is (0,0), KK is (1,1), ..., 22 is (12,12)
    - Suited hands: above diagonal (row < col)
    - Offsuit: below diagonal (row > col)
    """
    if len(hand) == 2:      # pocket pair, e.g. "TT"
        r1, r2 = hand[0], hand[1]
        return GRID_INDEX[r1], GRID_INDEX[r2]

    # Suited or offsuit: e.g., "AKs" or "JTo"
    r1, r2, suited_flag = hand[0], hand[1], hand[2]
    i = GRID_INDEX[r1]
    j = GRID_INDEX[r2]
    return i, j


def plot_heatmap(logger):
    hands = logger["hands"]
    actions = logger["actions"]

    # 13×13 buckets storing lists of actions
    buckets = [[[] for _ in range(13)] for _ in range(13)]

    # Fill buckets
    for h, a in zip(hands, actions):
        i, j = grid_coords(h)
        buckets[i][j].append(a)

    # Reduce to the most common action (mode)
    grid = np.full((13, 13), np.nan)
    for i in range(13):
        for j in range(13):
            if buckets[i][j]:
                grid[i, j] = Counter(buckets[i][j]).most_common(1)[0][0]

    plt.figure(figsize=(10, 9))
    cmap = plt.get_cmap("viridis", 5)  # 5 discrete actions

    img = plt.imshow(grid, cmap=cmap, vmin=0, vmax=4)

    plt.xticks(range(13), GRID_RANKS)
    plt.yticks(range(13), GRID_RANKS)
    plt.xlabel("Second card")
    plt.ylabel("First card")
    plt.title("Bot Preflop Strategy (GTO-Style Heatmap)")

    cbar = plt.colorbar(img, ticks=[0, 1, 2, 3, 4])
    cbar.ax.set_yticklabels(["FOLD", "CALL", "0.5P", "POT", "ALL-IN"])

    plt.tight_layout()
    plt.savefig("preflop_heatmap.png", dpi=200)
    plt.show()

# =====================================================
# Simulation
# =====================================================
def simulate(n=3000):
    env = SimpleHoldemEnv()
    session = CashSession(env, starting_stacks=(200, 200))
    policy = load_policy(env)

    logger = init_logger()
    for _ in range(n):
        play_hand(policy, env, session, logger)

    return logger


# =====================================================
# Main
# =====================================================
if __name__ == "__main__":
    print("Running PRE-FLOP simulation and logging bot actions...")
    logger = simulate(5000)
    plot_heatmap(logger)    
    print("Saved: hand_action_preflop_scatter.png")
