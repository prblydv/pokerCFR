# ============================================================
# CORRECTED EVAL MATCH FOR CPU CFR
# ============================================================

import torch
import torch.nn.functional as F
import numpy as np

from config import DEVICE
from encode_state import encode_state
from engine import SimpleHoldemEnv9, NUM_ACTIONS

ACTIONS = ["FOLD","CHECK","CALL","R2X","R3X","HALF","POT","10BB","ALLIN"]
STREETS = ["PREFLOP","FLOP","TURN","RIVER"]
NUM_STREETS = 4

ACTION_FOLD = 0
ACTION_CALL = 2


# ============================================================
# Masked softmax sampler for deterministic CFR evaluation
# ============================================================
@torch.no_grad()
def sample_action_from_policy(state, legal, net):
    x = encode_state(state, state.to_act).to(DEVICE)
    logits = net(x.unsqueeze(0))[0]

    mask = torch.full((NUM_ACTIONS,), -1e9, device=DEVICE)
    for a in legal:
        mask[a] = 0.0

    probs = F.softmax(logits + mask, dim=-1)

    if torch.isnan(probs).any():
        return np.random.choice(legal)

    a = torch.multinomial(probs, 1).item()
    if a not in legal:
        a = np.random.choice(legal)

    return a


# ============================================================
# Main evaluation
# ============================================================
def eval_match_cpu(env: SimpleHoldemEnv9, bot0, bot1, num_games=200):

    bot0.eval()
    bot1.eval()

    # -------------------------------------
    # Accumulators
    # -------------------------------------
    stats = {
        "hands": 0,
        "ev_sum": 0.0,
        "wins": 0,
        "showdowns": 0,

        "vpip_count": 0,
        "agg_count": 0,
        "pf_fold_count": 0,

        "pos_ev": np.zeros(2),   # 0 = BB, 1 = BTN
        "pos_hands": np.zeros(2),

        "action_counts": np.zeros(NUM_ACTIONS),
        "street_actions": np.zeros((NUM_STREETS, NUM_ACTIONS)),
        "pos_actions": np.zeros((2, NUM_ACTIONS)),
    }


    # ============================================================
    # Helper: play one hand
    # ============================================================
    def play_hand(bot0_SB):

        # --- Deal hand ---
        s = env.new_hand()

        # Swap visual seats if bot0 is BB this hand
        if not bot0_SB:
            s.hole = [s.hole[1], s.hole[0]]
            s.stacks = [s.stacks[1], s.stacks[0]]
            s.contrib = [s.contrib[1], s.contrib[0]]
            s.initial_stacks = [s.initial_stacks[1], s.initial_stacks[0]]

        # Track position mapping
        # pos_map[p] = {0=BB, 1=BTN}
        pos_map = {0: 1 if bot0_SB else 0,
                   1: 0 if bot0_SB else 1}

        history = []

        # =============================================
        # Play hand
        # =============================================
        while not s.terminal:
            p = s.to_act
            legal = env.legal_actions(s)
            net = bot0 if p == 0 else bot1
            a = sample_action_from_policy(s, legal, net)

            history.append((p, pos_map[p], s.street, a))
            s = env.step(s, a)

        payoff0 = env.terminal_payoff(s, 0)
        winner = s.winner
        return payoff0, winner, history, bot0_SB


    # ============================================================
    # Update statistics
    # ============================================================
    def update_stats(stats, payoff0, winner, hist, bot0_SB):

        stats["hands"] += 1
        stats["ev_sum"] += payoff0
        if winner == 0:
            stats["wins"] += 1

        # which position was bot0?
        bot0_pos = 1 if bot0_SB else 0
        stats["pos_ev"][bot0_pos] += payoff0
        stats["pos_hands"][bot0_pos] += 1

        # VPIP / Agg / PF-FOLD
        saw_vpip = False
        saw_agg = False
        saw_pf_fold = False

        # --- detect bot0’s FIRST preflop action ---
        first_pf_action = None
        for (p, pos, street, a) in hist:
            if pos == bot0_pos and street == 0:
                first_pf_action = a
                break

        # PF-FOLD check
        if first_pf_action == ACTION_FOLD:
            saw_pf_fold = True

        # VPIP detection (correct definition)
        # VPIP = voluntarily put $ into pot (CALL or any RAISE)
        for (p, pos, street, a) in hist:
            if pos == bot0_pos:
                if street == 0:
                    if a == ACTION_CALL or a >= 3:  # any raise
                        saw_vpip = True

                if a >= 3:
                    saw_agg = True

                # count actions
                stats["action_counts"][a] += 1
                stats["street_actions"][street][a] += 1
                stats["pos_actions"][pos][a] += 1

        if saw_vpip:
            stats["vpip_count"] += 1
        if saw_agg:
            stats["agg_count"] += 1
        if saw_pf_fold:
            stats["pf_fold_count"] += 1


    # ============================================================
    # MAIN LOOP
    # ============================================================
    for g in range(num_games):
        bot0_SB = (g % 2 == 0)
        payoff0, winner, hist, seatflag = play_hand(bot0_SB)
        update_stats(stats, payoff0, winner, hist, seatflag)


    # ============================================================
    # Final aggregation
    # ============================================================
    H = max(1, stats["hands"])

    out = {}
    out["ev_per_hand"] = stats["ev_sum"] / H
    out["win_rate"]     = stats["wins"] / H
    out["showdown_rate"] = stats["showdowns"] / H

    out["vpip"] = stats["vpip_count"] / H
    out["agg_freq"] = stats["agg_count"] / H
    out["preflop_fold_rate"] = stats["pf_fold_count"] / H

    out["button_ev_per_hand"] = stats["pos_ev"][1] / max(1, stats["pos_hands"][1])
    out["blind_ev_per_hand"]  = stats["pos_ev"][0] / max(1, stats["pos_hands"][0])

    out["action_counts"]  = stats["action_counts"]
    out["street_action"]  = stats["street_actions"]
    out["pos_action"]     = stats["pos_actions"]

    out["hands"] = H
    return out



# =============================================================================
# PRETTY PRINTER (UNCHANGED)
# =============================================================================
def print_eval_stats_colored(stats, iteration=None):

    RESET = '\033[0m'
    BLACK = '\033[30m'
    DARK_RED = '\033[31m'
    RED = '\033[91m'
    BRIGHT_RED = '\033[1;31m'
    YELLOW = '\033[93m'
    BOLD = '\033[1m'

    def colorize(v, lo, hi, pct=False):
        raw = v * 100 if pct else v
        disp = f"{v*100:0.1f}%" if pct else f"{v:0.3f}"

        if lo <= raw <= hi:
            return f"{BLACK}{disp}{RESET}"

        diff = min(abs(raw - lo), abs(raw - hi))
        span = hi - lo

        if diff < span * 0.5:
            return f"{DARK_RED}{disp}{RESET}"
        elif diff < span:
            return f"{RED}{disp}{RESET}"
        else:
            return f"{BRIGHT_RED}{disp}{RESET}"

    if iteration:
        print(f"\n{YELLOW}{BOLD}=== Evaluation @ Iteration {iteration} ==={RESET}")
        print("─"*90)

    print(f"{BOLD}MAIN PERFORMANCE METRICS (Player 0 Only){RESET}")
    print("┌────────────────┬─────────────┬────────────────┬─────────────┬────────────────┬─────────────┬────────────────┬─────────────┐")
    print("│ Metric         │ Value       │ Metric         │ Value       │ Metric         │ Value       │ Metric         │ Value       │")
    print("├────────────────┼─────────────┼────────────────┼─────────────┼────────────────┼─────────────┼────────────────┼─────────────┤")

    r1 = [
        "EV/Hand",   colorize(stats["ev_per_hand"], -0.05, 0.05),
        "Win%",     colorize(stats["win_rate"], 48, 52, pct=True),
        "SD%",      colorize(stats["showdown_rate"], 20, 40, pct=True),
        "VPIP%",    colorize(stats["vpip"], 20, 40, pct=True),
    ]

    r2 = [
        "Btn EV",   colorize(stats["button_ev_per_hand"], 0.01, 0.08),
        "BB EV",    colorize(stats["blind_ev_per_hand"], -0.10, -0.01),
        "PF Fold%", colorize(stats["preflop_fold_rate"], 15, 35, pct=True),
        "Agg%",     colorize(stats["agg_freq"], 25, 45, pct=True),
    ]

    print("│ {:<14} │ {:>11} │ {:<14} │ {:>11} │ {:<14} │ {:>11} │ {:<14} │ {:>11} │".format(*r1))
    print("│ {:<14} │ {:>11} │ {:<14} │ {:>11} │ {:<14} │ {:>11} │ {:<14} │ {:>11} │".format(*r2))
    print("└────────────────┴─────────────┴────────────────┴─────────────┴────────────────┴─────────────┴────────────────┴─────────────┘")

    # -------------------------------------------------------------------------
    # ACTION DISTRIBUTION (PLAYER 0 ONLY)
    # -------------------------------------------------------------------------
    print(f"\n{BOLD}ACTION DISTRIBUTION — Player 0{RESET}")
    print("┌────────┬────────────┐")
    print("│ Action │ Frequency  │")
    print("├────────┼────────────┤")
    for i, name in enumerate(ACTIONS):
        print(f"│ {name:<6} │ {stats['action_counts'][i]:>10.1f} │")
    print("└────────┴────────────┘")

    # -------------------------------------------------------------------------
    # STREET ACTIONS (DYNAMIC FOR NUM_ACTIONS)
    # -------------------------------------------------------------------------
    print(f"\n{BOLD}ACTIONS BY STREET — Player 0{RESET}")

    # header line with all action names
    header_actions = " | ".join(f"{name:>5}" for name in ACTIONS)
    print("┌──────────┬" + "─" * (len(header_actions) + 2) + "┐")
    print(f"│ Street   │ {header_actions} │")
    print("├──────────┼" + "─" * (len(header_actions) + 2) + "┤")

    for i, st in enumerate(STREETS):
        r = stats["street_action"][i]  # shape: [NUM_ACTIONS]
        row_counts = " | ".join(f"{int(r[j]):5d}" for j in range(len(ACTIONS)))
        print(f"│ {st:<8} │ {row_counts} │")

    print("└──────────┴" + "─" * (len(header_actions) + 2) + "┘")

    # -------------------------------------------------------------------------
    # POSITION ACTIONS (PLAYER 0)
    # -------------------------------------------------------------------------
    print(f"\n{BOLD}ACTIONS BY POSITION — Player 0 (0 = BB, 1 = BTN){RESET}")

    print("┌──────────┬" + "─" * (len(header_actions) + 2) + "┐")
    print(f"│ Pos      │ {header_actions} │")
    print("├──────────┼" + "─" * (len(header_actions) + 2) + "┤")

    for pos in range(2):
        r = stats["pos_action"][pos]  # shape: [NUM_ACTIONS]
        row_counts = " | ".join(f"{int(r[j]):5d}" for j in range(len(ACTIONS)))
        print(f"│ {pos:<8} │ {row_counts} │")

    print("└──────────┴" + "─" * (len(header_actions) + 2) + "┘")

    # -------------------------------------------------------------------------
    print("\nNormal | Slightly off | Bad | Critical\n")
