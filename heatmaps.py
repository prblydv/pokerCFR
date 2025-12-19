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
from abstraction import encode_state
from networks import PolicyNet

# -- Action setup matches the new 6-action abstraction --
ACTION_LABELS = {
    ACTION_FOLD: "FOLD",
    ACTION_CHECK: "CHECK",
    ACTION_CALL: "CALL",
    ACTION_RAISE_SMALL: "RAISE SMALL",
    ACTION_RAISE_MEDIUM: "RAISE MEDIUM",
    ACTION_ALL_IN: "ALL-IN",
}
RANKS = ["A","K","Q","J","T","9","8","7","6","5","4","3","2"]


def make_card(rank_idx: int, suit_idx: int) -> int:
    """Encode a card index given rank (0=2 .. 12=A) and suit (0..3)."""
    return suit_idx * 13 + rank_idx


def grid_hole_from_indices(row: int, col: int):
    """
    Return a hole-card pair for the (row, col) entry of the heatmap grid.
    Above diagonal -> suited combos, below -> offsuit, diagonal -> pairs.
    """
    rank_row = 12 - row
    rank_col = 12 - col

    if row == col:
        # Pocket pair: use two different suits for the same rank.
        return [make_card(rank_row, 0), make_card(rank_col, 1)]
    if col > row:
        # Above diagonal => suited combos (same suit, different ranks).
        return [make_card(rank_row, 0), make_card(rank_col, 0)]
    # Below diagonal => offsuit combos (different suits).
    return [make_card(rank_row, 0), make_card(rank_col, 1)]

# Function metadata:
#   Inputs: env (SimpleHoldemEnv), hole (List[int] len 2)  # dtype=(SimpleHoldemEnv, List[int])
#   Sample:
#       env = SimpleHoldemEnv()
#       hole = [12, 25]
#       sample_output = get_real_preflop_state(env, hole)  # dtype=GameState
def get_real_preflop_state(env, hole, button_seat: int, hero_seat: int):
    """
    Generate a preflop state for a specific hero seat and button seat.
    Forces the button to `button_seat` for positional heatmaps, then restores rotation.
    """
    orig_button = getattr(env, "_next_button", 0)
    try:
        env._next_button = button_seat % env.num_players  # type: ignore[attr-defined]
    except Exception:
        pass
    s = env.new_hand()
    # restore rotation so we don't drift
    try:
        env._next_button = orig_button  # type: ignore[attr-defined]
    except Exception:
        pass
    s.hole[hero_seat] = hole[:]
    return s

# Function metadata:
#   Inputs: policy (PolicyNet), env (SimpleHoldemEnv), hole (List[int]),
#           samples (int), device (str)  # dtype=mixed
#   Sample:
#       sample_output = get_avg_policy_probs(policy, env, [12,25], samples=10)
#       # dtype=np.ndarray shape (10,)
def get_avg_policy_probs(policy, env, hole, seat_idx=0, button_seat=0, samples=20, device="cpu"):
    probs_accum = None
    for _ in range(samples):
        s = get_real_preflop_state(env, hole, button_seat, seat_idx)
        x = encode_state(s, seat_idx).to(device).unsqueeze(0)
        with torch.no_grad():
            logits = policy(x).squeeze(0)
        p = torch.softmax(logits, dim=-1).cpu().numpy()
        if probs_accum is None:
            probs_accum = p
        else:
            probs_accum += p
    probs_accum /= samples
    return probs_accum  # shape: (NUM_ACTIONS,)

if __name__ == "__main__":
    env = SimpleHoldemEnv()
    dummy = env.new_hand()
    state_dim = encode_state(dummy, 0).numel()
    policy = PolicyNet(state_dim)
    policy.load_state_dict(torch.load("models/policy.pt", map_location="cpu"))
    policy.eval()

    num_players = getattr(env, "num_players", 6)
    # Position offsets relative to button for 6-max style seating
    position_offsets = [
        ("BTN", 0),  # Button / Dealer
        ("SB", 1),
        ("BB", 2),
        ("UTG", 3),
        ("MP_LJ", 4),  # sanitize for filenames
        ("CO", 5),
    ]
    out_dir = "heatmaps"
    import os
    os.makedirs(out_dir, exist_ok=True)
    for pos_label, offset in position_offsets:
        hero_seat = offset % num_players
        button_seat = 0  # place button at seat 0; hero position determined by offset
        # Build 3D array: (actions, 13, 13) for this seat
        action_probs = np.zeros((NUM_ACTIONS, 13, 13))
        for i, _ in enumerate(RANKS):
            for j, _ in enumerate(RANKS):
                hole = grid_hole_from_indices(i, j)
                probs = get_avg_policy_probs(
                    policy, env, hole, seat_idx=hero_seat, button_seat=button_seat, samples=25
                )
                for a in range(NUM_ACTIONS):
                    action_probs[a, i, j] = probs[a]

        # Plot a grid of heatmaps, one for each action
        rows = 2
        cols = 3
        fig, axes = plt.subplots(rows, cols, figsize=(18, 8))
        vmax = np.max(action_probs)  # keep color scale consistent across actions for this seat

        for idx, (a, label) in enumerate(ACTION_LABELS.items()):
            ax = axes[idx // cols, idx % cols]
            im = ax.imshow(action_probs[a], cmap="Reds", vmin=0, vmax=vmax)
            for r in range(13):
                for c in range(13):
                    val = action_probs[a, r, c]
                    ax.text(c, r, f"{val:.2f}", ha="center", va="center", fontsize=6, color="black")
            ax.set_xticks(range(13))
            ax.set_yticks(range(13))
            ax.set_xticklabels(RANKS)
            ax.set_yticklabels(RANKS)
            ax.set_title(f"{label} ({pos_label})")

        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, label="Action probability")
        plt.suptitle(f"Preflop Probability Heatmaps ({pos_label})")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        safe_label = pos_label.replace("/", "_").replace("\\", "_")
        out_path = os.path.join(out_dir, f"gto_preflop_action_heatmaps_{safe_label}.png")
        plt.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"Saved {out_path}")
