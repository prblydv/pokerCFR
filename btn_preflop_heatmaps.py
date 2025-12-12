# ---------------------------------------------------------------------------
# File overview:
#   btn_preflop_heatmaps.py evaluates policy actions from the button perspective
#   to produce heatmaps. Run via `python btn_preflop_heatmaps.py` after training.
# ---------------------------------------------------------------------------

import torch
import numpy as np
import matplotlib.pyplot as plt
from poker_env import SimpleHoldemEnv
from abstraction import encode_state
from networks import PolicyNet

# Action labels consistent with your net
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
#   Inputs: r1_idx, r2_idx  # dtype=varies
#   Sample:
#       sample_output = make_hole(r1_idx=None, r2_idx=None)  # dtype=Any
def make_hole(r1_idx, r2_idx):
    """
    r1_idx, r2_idx: 0..12 where 0 = A, 12 = 2
    Create suited + offsuit versions.
    """

    # rank index in card deck: A=12, K=11, ..., 2=0
    r1 = 12 - r1_idx
    r2 = 12 - r2_idx

    # suited hand → suit=0
    suited = [0*13 + r1, 0*13 + r2]

    # offsuit hand → suit 0 and 1
    offsuit = [0*13 + r1, 1*13 + r2]

    return [suited, offsuit]


# Function metadata:
#   Inputs: env, hole  # dtype=varies
#   Sample:
#       sample_output = make_btn_preflop_state(env=mock_env, hole=[10, 23])  # dtype=Any
def make_btn_preflop_state(env, hole):
    """
    Construct a deterministic BTN (Player 1) opening state.
    """
    s = env.new_hand()

    s.street = 0
    s.board = []
    s.pot = 1.5
    s.current_bet = 0
    s.stacks = [200, 200]
    s.last_aggressor = -1

    # Button (Player 1) hole cards
    s.hole[1] = hole[:]   # hero = Player 1

    return s


# Function metadata:
#   Inputs: policy, env, hole, device  # dtype=varies
#   Sample:
#       sample_output = get_policy_probs(policy=None, env=mock_env, hole=[10, 23], device='cpu')  # dtype=Any
def get_policy_probs(policy, env, hole, device="cpu"):
    s = make_btn_preflop_state(env, hole)
    x = encode_state(s, 1).to(device).unsqueeze(0)  # NOTE: player=1

    with torch.no_grad():
        logits = policy(x).squeeze(0)

    # mask illegal actions
    legal = env.legal_actions(s)
    mask = torch.zeros_like(logits)
    mask[legal] = 1.0

    logits = logits.masked_fill(mask == 0, float('-inf'))
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    return probs


# -------------------------------------------------------------------

if __name__ == "__main__":
    env = SimpleHoldemEnv()
    dummy = env.new_hand()

    state_dim = encode_state(dummy, 0).numel()

    policy = PolicyNet(state_dim)
    policy.load_state_dict(torch.load("models/policy.pt", map_location="cpu"))
    policy.eval()

    action_probs = np.zeros((10, 13, 13))  # 10 actions, 13x13 grid

    for i, r1 in enumerate(RANKS):
        for j, r2 in enumerate(RANKS):

            holes = make_hole(i, j)
            avg = np.zeros(10)

            for h in holes:
                avg += get_policy_probs(policy, env, h)

            avg /= len(holes)
            action_probs[:, i, j] = avg

    # --- plotting ---
    fig, axes = plt.subplots(2, 5, figsize=(20, 9))
    vmax = 1.0

    for idx, (a, name) in enumerate(ACTION_LABELS.items()):
        ax = axes[idx // 5, idx % 5]
        mat = action_probs[a]

        im = ax.imshow(mat, cmap="Reds", vmin=0, vmax=vmax)

        for x in range(13):
            for y in range(13):
                ax.text(y, x, f"{mat[x,y]:.2f}",
                        ha="center", va="center", fontsize=6)

        ax.set_xticks(range(13))
        ax.set_yticks(range(13))
        ax.set_xticklabels(RANKS)
        ax.set_yticklabels(RANKS)
        ax.set_title(name)
    env = SimpleHoldemEnv()
    s = env.new_hand()

    # construct BTN opening state manually
    s.street = 0
    s.board = []
    s.pot = 1.5
    s.current_bet = 0
    s.stacks = [200,200]
    s.last_aggressor = -1
    s.acting_player = 1
    s.hole[1] = [0, 12]  # example AKs

    print("BTN legal actions:", env.legal_actions(s))

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7)
    plt.suptitle("BTN Preflop Opening Strategy — True Probabilities (Deep CFR)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("btn_preflop_strategy.png", dpi=300)
    plt.show()
