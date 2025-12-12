# ---------------------------------------------------------------------------
# File overview:
#   heatmaps.py loads a trained policy to produce averaged preflop action
#   heatmaps sampled from randomized preflop states. Run via `python heatmaps.py`
#   after training to regenerate gto_preflop_action_heatmaps5000.png.
# ---------------------------------------------------------------------------

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from poker_env import SimpleHoldemEnv
from abstraction import encode_state
from networks import PolicyNet

# -- Action setup as before --
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
RANKS = ["A","K","Q","J","T","9","8","7","6","5","4","3","2"]

# Function metadata:
#   Inputs: env (SimpleHoldemEnv), hole (List[int] len 2)  # dtype=(SimpleHoldemEnv, List[int])
#   Sample:
#       env = SimpleHoldemEnv()
#       hole = [12, 25]
#       sample_output = get_real_preflop_state(env, hole)  # dtype=GameState
def get_real_preflop_state(env, hole):
    s = env.new_hand()
    s.hole[0] = hole[:]
    return s

# Function metadata:
#   Inputs: policy (PolicyNet), env (SimpleHoldemEnv), hole (List[int]),
#           samples (int), device (str)  # dtype=mixed
#   Sample:
#       sample_output = get_avg_policy_probs(policy, env, [12,25], samples=10)
#       # dtype=np.ndarray shape (10,)
def get_avg_policy_probs(policy, env, hole, samples=20, device="cpu"):
    probs_accum = None
    for _ in range(samples):
        s = get_real_preflop_state(env, hole)
        x = encode_state(s, 0).to(device).unsqueeze(0)
        with torch.no_grad():
            logits = policy(x).squeeze(0)
        p = torch.softmax(logits, dim=-1).cpu().numpy()
        if probs_accum is None:
            probs_accum = p
        else:
            probs_accum += p
    probs_accum /= samples
    return probs_accum  # shape: (10,)

if __name__ == "__main__":
    env = SimpleHoldemEnv()
    dummy = env.new_hand()
    state_dim = encode_state(dummy, 0).numel()
    policy = PolicyNet(state_dim)
    policy.load_state_dict(torch.load("models/policy.pt", map_location="cpu"))
    policy.eval()

    # Build 3D array: (actions, 13, 13)
    action_probs = np.zeros((10, 13, 13))
    for i, r1 in enumerate(RANKS):
        for j, r2 in enumerate(RANKS):
            hi = 12 - i
            lo = 12 - j
            c1 = hi
            c2 = lo + 13
            hole = [c1, c2]
            probs = get_avg_policy_probs(policy, env, hole, samples=25)
            for a in range(10):
                action_probs[a, i, j] = probs[a]

    # Plot a grid of heatmaps, one for each action
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    vmax = np.max(action_probs)  # to keep color scale consistent

    for idx, (a, label) in enumerate(ACTION_LABELS.items()):
        ax = axes[idx // 5, idx % 5]
        im = ax.imshow(action_probs[a], cmap="Reds", vmin=0, vmax=vmax)
        for i in range(13):
            for j in range(13):
                val = action_probs[a, i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color="black")
        ax.set_xticks(range(13))
        ax.set_yticks(range(13))
        ax.set_xticklabels(RANKS)
        ax.set_yticklabels(RANKS)
        ax.set_title(label)

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, label="Action probability")
    plt.suptitle("Preflop GTO-Style Probability Heatmaps (Deep CFR, No Argmax)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("gto_preflop_action_heatmaps5000.png", dpi=300)
    plt.show()
