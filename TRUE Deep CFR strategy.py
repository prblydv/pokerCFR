# ---------------------------------------------------------------------------
# File overview:
#   Generates TRUE Deep CFR preflop action heatmaps by loading a saved policy
#   and sweeping all rank combinations. Run via
#       `python "TRUE Deep CFR strategy.py"`
#   which loads models/policy.pt and emits PNG heatmaps.
# ---------------------------------------------------------------------------

import torch
import numpy as np
import matplotlib.pyplot as plt
from poker_env import SimpleHoldemEnv
from abstraction import encode_state
from networks import PolicyNet

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
RANK_INDEX = {r:i for i,r in enumerate(RANKS)}  # A=0,...2=12

# Function metadata:
#   Inputs: r1_idx (int 0..12), r2_idx (int 0..12)  # dtype=int
#   Sample:
#       sample_input = {'r1_idx': 0, 'r2_idx': 1}
#       sample_output = make_hole(**sample_input)
#       # dtype=List[List[int]] e.g. [[12,11],[12,24]]
def make_hole(r1_idx, r2_idx):
    """
    r1_idx, r2_idx: 0..12 where 0=Ace, 12=2
    We create a generic suited and offsuit version and average them.
    """

    r1 = 12 - r1_idx  # convert A..2 → 12..0 rank index
    r2 = 12 - r2_idx

    # suited version (suit 0)
    suited = [0*13 + r1, 0*13 + r2]

    # offsuit version (suit 0 and 1)
    offsuit = [0*13 + r1, 1*13 + r2]

    return [suited, offsuit]

# Function metadata:
#   Inputs: env (SimpleHoldemEnv), hole (List[int] len 2)  # dtype=(SimpleHoldemEnv, List[int])
#   Sample:
#       env = SimpleHoldemEnv()
#       hole = [12, 11]
#       sample_output = make_preflop_state(env, hole)  # dtype=GameState with hero hole overwritten
def make_preflop_state(env, hole):
    s = env.new_hand()

    # deterministically overwrite everything
    s.street = 0
    s.board = []
    s.pot = 1.5
    s.current_bet = 0
    s.stacks = [200,200]
    s.last_aggressor = -1

    s.hole[0] = hole[:]   # hero cards
    return s

# Function metadata:
#   Inputs: policy (PolicyNet), env (SimpleHoldemEnv), hole (List[int]), device (str)  # dtype=mixed
#   Sample:
#       policy = PolicyNet(input_dim=64); env = SimpleHoldemEnv()
#       hole = [12, 25]
#       sample_output = get_policy_probs(policy, env, hole)
#       # dtype=np.ndarray shape (num_actions,)
def get_policy_probs(policy, env, hole, device="cpu"):
    s = make_preflop_state(env, hole)
    x = encode_state(s, 0).to(device).unsqueeze(0)

    with torch.no_grad():
        logits = policy(x).squeeze(0)

    # legal-action masked softmax
    legal = env.legal_actions(s)
    mask = torch.zeros_like(logits)
    mask[legal] = 1.0

    logits_masked = logits.masked_fill(mask == 0, float('-inf'))
    probs = torch.softmax(logits_masked, dim=-1).cpu().numpy()
    return probs

if __name__ == "__main__":
    env = SimpleHoldemEnv()
    dummy = env.new_hand()
    policy = PolicyNet(encode_state(dummy,0).numel())
    policy.load_state_dict(torch.load("models/policy.pt", map_location="cpu"))
    policy.eval()

    action_probs = np.zeros((10, 13, 13))

    for i, r1 in enumerate(RANKS):
        for j, r2 in enumerate(RANKS):

            holes = make_hole(i, j)

            # average suited + offsuit versions
            total = np.zeros(10)
            for h in holes:
                total += get_policy_probs(policy, env, h)
            total /= len(holes)

            action_probs[:, i, j] = total

    # plot
    fig, axes = plt.subplots(2, 5, figsize=(20, 9))
    vmax = 1.0

    for idx, (a, label) in enumerate(ACTION_LABELS.items()):
        ax = axes[idx//5, idx%5]
        mat = action_probs[a]

        im = ax.imshow(mat, cmap="Reds", vmin=0, vmax=vmax)
        for x in range(13):
            for y in range(13):
                ax.text(y, x, f"{mat[x,y]:.2f}", ha="center", va="center", fontsize=6)

        ax.set_xticks(range(13))
        ax.set_yticks(range(13))
        ax.set_xticklabels(RANKS)
        ax.set_yticklabels(RANKS)
        ax.set_title(label)

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7)
    plt.suptitle("TRUE Preflop Strategy Heatmaps (masked softmax)")
    plt.tight_layout(rect=[0,0.03,1,0.95])
    plt.savefig("fixed_preflop_heatmaps(masked softmax).png", dpi=300)
    plt.show()
