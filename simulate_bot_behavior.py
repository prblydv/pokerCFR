# ---------------------------------------------------------------------------
# File overview:
#   simulate_bot_behavior.py rebuilds TRUE preflop action charts by sampling
#   a trained PolicyNet on env-preflop states. Run via
#       `python simulate_bot_behavior.py`
#   after models are saved to `models/`.
# ---------------------------------------------------------------------------

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from poker_env import SimpleHoldemEnv
from abstraction import encode_state
from networks import PolicyNet


# -------------------------------------------------------------------------
# Action Labels (10 actions)
# -------------------------------------------------------------------------
ACTION_LABELS = {
    0: "FOLD",
    1: "CALL",
    2: "2x",
    3: "2.25x",
    4: "2.5x",
    5: "3x",
    6: "3.5x",
    7: "4.5x",
    8: "6x",
    9: "ALL-IN",
}

ACTION_COLORS = {
    "FOLD": "#d3d3d3",
    "CALL": "#87cefa",
    "2x": "#add8e6",
    "2.25x": "#87cefa",
    "2.5x": "#00bfff",
    "3x": "#1e90ff",
    "3.5x": "#4169e1",
    "4.5x": "#0000cd",
    "6x": "#191970",
    "ALL-IN": "#ff4d4d",
}

# 13x13 rank grid
RANKS = ["A","K","Q","J","T","9","8","7","6","5","4","3","2"]


# -------------------------------------------------------------------------
# Build *real* preflop state from env.new_hand()
# -------------------------------------------------------------------------
# Function metadata:
#   Inputs: env, hole  # dtype=varies
#   Sample:
#       sample_output = get_real_preflop_state(env=mock_env, hole=[10, 23])  # dtype=Any
def get_real_preflop_state(env, hole):
    """
    Create a true preflop state exactly like training:
    - blinds already posted by env
    - pot = 1.5
    - current_bet = 1.0 (BB)
    - stacks = [199.5, 199.0]
    - last_aggressor set correctly
    - player_to_act is correct (SB acts first)
    """
    s = env.new_hand()       # This posts blinds and prepares a real statex 
    s.hole[0] = hole[:]      # hero cards
    return s


# -------------------------------------------------------------------------
# Compute average logits over multiple real env preflop samples
# -------------------------------------------------------------------------
# Function metadata:
#   Inputs: policy, env, hole, samples, device  # dtype=varies
#   Sample:
#       sample_output = get_avg_policy_logits(policy=None, env=mock_env, hole=[10, 23], samples=None, device='cpu')  # dtype=Any
def get_avg_policy_logits(policy, env, hole, samples=20, device="cpu"):
    logits_accum = None

    for _ in range(samples):
        s = get_real_preflop_state(env, hole)

        x = encode_state(s, 0).to(device).unsqueeze(0)

        with torch.no_grad():
            logits = policy(x).squeeze(0)

        if logits_accum is None:
            logits_accum = logits
        else:
            logits_accum += logits

    logits_accum /= samples
    probs = torch.softmax(logits_accum, dim=-1)
    a = torch.argmax(probs).item()

    return ACTION_LABELS[a]


# -------------------------------------------------------------------------
# MAIN: Generate true preflop heatmap
# -------------------------------------------------------------------------
if __name__ == "__main__":

    env = SimpleHoldemEnv()

    # Determine state dimension
    dummy = env.new_hand()
    state_dim = encode_state(dummy, 0).numel()

    # Load trained policy
    policy = PolicyNet(state_dim)
    policy.load_state_dict(torch.load("models/policy.pt", map_location="cpu"))
    policy.eval()

    matrix = np.empty((13,13), dtype=object)

    # Build grid
    for i, r1 in enumerate(RANKS):
        for j, r2 in enumerate(RANKS):
            
            hi = 12 - i
            lo = 12 - j

            c1 = hi        # suit 0
            c2 = lo + 13   # suit 1 (offsuit)
            hole = [c1, c2]

            action = get_avg_policy_logits(policy, env, hole, samples=25)
            matrix[i,j] = action


    # ---------------------------------------------------------------------
    # Build color grid
    # ---------------------------------------------------------------------
    color_grid = np.empty((13, 13), dtype=object)

    for i in range(13):
        for j in range(13):
            action = matrix[i, j]
            color_grid[i, j] = ACTION_COLORS[action]

    rgb_grid = np.zeros((13, 13, 3))
    for i in range(13):
        for j in range(13):
            rgb_grid[i, j] = mcolors.to_rgb(color_grid[i, j])

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(rgb_grid)

    # annotate
    for i in range(13):
        for j in range(13):
            ax.text(j, i, matrix[i, j],
                    ha="center", va="center",
                    fontsize=7, color="black")

    ax.set_xticks(range(13))
    ax.set_yticks(range(13))
    ax.set_xticklabels(RANKS)
    ax.set_yticklabels(RANKS)

    ax.set_xlabel("Second card")
    ax.set_ylabel("First card")
    ax.set_title("REAL Preflop GTO-Style Action Chart (Deep CFR)")

    plt.tight_layout()
    plt.savefig("real_preflop_gto_chart50000.png", dpi=300)
    plt.show()
