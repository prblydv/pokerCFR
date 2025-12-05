# import torch
# import torch.nn.functional as F
# import numpy as np
# from encode_state import encode_state

# from engine import SimpleHoldemEnv9, NUM_ACTIONS
# from config import DEVICE

# ACTIONS = ["FOLD","CHECK","CALL","R2X","R3X","HALF","POT","10BB","ALLIN"]
# STREETS = ["PREFLOP","FLOP","TURN","RIVER"]
# NUM_STREETS = 4


# # =====================================================================
# # MAIN EVAL MATCH (CPU)
# # =====================================================================
# def eval_match_cpu(env: SimpleHoldemEnv9, bot0, bot1, num_games=200):

#     import logging, time
#     logging.getLogger(__name__).info(f"Starting eval_match_cpu: num_games={num_games}, bot0={bot0.__class__.__name__}, bot1={bot1.__class__.__name__}")
#     t0 = time.perf_counter()

#     bot0.eval()
#     bot1.eval()

#     # ----------------------------------------------------------
#     # LOG INITIALISER
#     # ----------------------------------------------------------
#     def init_log():
#         return {
#             "hands": 0,
#             "ev": 0.0,
#             "ev_pos": np.zeros(2),
#             "hands_pos": np.zeros(2),
#             "win": 0,
#             "vpip": 0,
#             "sd": 0,
#             "pf_fold": 0,
#             "agg": 0,
#             "action_counts": np.zeros(NUM_ACTIONS),
#             "street_action": np.zeros((NUM_STREETS, NUM_ACTIONS)),
#             "pos_action": np.zeros((2, NUM_ACTIONS)),
#         }

#     # ----------------------------------------------------------
#     # SAMPLE ACTION (CPU CFR SOFTMAX)
#     # ----------------------------------------------------------
#     @torch.no_grad()
#     def sample_action(state, legal_actions, net):
#         # encode current player's observation
#         x = encode_state(state, state.to_act).to(DEVICE)   # (feat,)
#         logits = net(x.unsqueeze(0))[0]                    # (NUM_ACTIONS,)

#         # masked softmax over legal actions only
#         mask = torch.full((NUM_ACTIONS,), -1e9, device=DEVICE)
#         for a in legal_actions:
#             mask[a] = 0.0

#         probs = F.softmax(logits + mask, dim=-1)
#         a = torch.multinomial(probs, 1).item()
#         if a not in legal_actions:
#             a = np.random.choice(legal_actions)
#         return a

#     # ----------------------------------------------------------
#     # UPDATE LOG FOR BOT0 ONLY
#     # ----------------------------------------------------------
#     def update(log, payoff, winner, hist, pid):
#         log["hands"] += 1
#         log["ev"] += payoff[pid]
#         log["win"] += int(winner == pid)

#         # position: 1 = SB, 0 = BB
#         first_player = hist[0][0]
#         pos = 1 if first_player == pid else 0

#         log["ev_pos"][pos] += payoff[pid]
#         log["hands_pos"][pos] += 1

#         vpip = False
#         aggressive = False

#         for p, street, pos_, a in hist:
#             if p == pid:
#                 log["action_counts"][a] += 1
#                 log["street_action"][street][a] += 1
#                 log["pos_action"][pos_][a] += 1

#                 if street == 0 and a != 0:  # not fold
#                     vpip = True
#                 if a >= 3:  # raise/bet
#                     aggressive = True

#         if vpip:
#             log["vpip"] += 1
#         if payoff[pid] != 0:
#             log["sd"] += 1
#         if first_player == pid and hist[0][3] == 0:
#             log["pf_fold"] += 1
#         if aggressive:
#             log["agg"] += 1

#     # ----------------------------------------------------------
#     # PLAY ONE GAME CPU SINGLE STATE
#     # ----------------------------------------------------------
#     def play_game(bot0_seat_SB):
#         """
#         bot0_seat_SB = True means bot0 = SB
#                        False means bot0 = BB
#         """

#         # new game
#         s = env.new_hand()

#         # fix seating:
#         # env always sets sb_player=0, bb_player=1
#         # but bot0 must sometimes be BB
#         if not bot0_seat_SB:
#             # Swap hole cards, stacks, positions so bot0 becomes player 1
#             s.hole = [s.hole[1], s.hole[0]]
#             s.stacks = [s.stacks[1], s.stacks[0]]
#             s.contrib = [s.contrib[1], s.contrib[0]]
#             s.initial_stacks = [s.initial_stacks[1], s.initial_stacks[0]]

#         history = []

#         # position array: bot0 is p0
#         pos_map = [1 if bot0_seat_SB else 0, 1 - (1 if bot0_seat_SB else 0)]

#         while not s.terminal:
#             p = s.to_act
#             legal = env.legal_actions(s)

#             # pick the correct bot
#             net = bot0 if p == 0 else bot1
#             a = sample_action(s, legal, net)

#             # record for stats
#             history.append((p, s.street, pos_map[p], a))

#             # advance environment
#             s = env.step(s, a)

#         # compute payoff for bot0
#         payoff0 = env.terminal_payoff(s, 0)
#         payoff = [payoff0, -payoff0]

#         winner = s.winner
#         return payoff, winner, history

#     # ----------------------------------------------------------
#     # MAIN EVAL LOOP
#     # ----------------------------------------------------------
#     log = init_log()

#     for i in range(num_games):
#         bot0_sb = (i % 2 == 0)      # bot0 alternates SB/BB
#         payoff, winner, hist = play_game(bot0_sb)
#         update(log, payoff, winner, hist, pid=0)

#     # finalize metrics
#     H = max(1, log["hands"])
#     log["ev_per_hand"] = log["ev"] / H
#     log["win_rate"] = log["win"] / H
#     log["showdown_rate"] = log["sd"] / H
#     log["vpip"] = log["vpip"] / H
#     log["agg_freq"] = log["agg"] / H
#     log["preflop_fold_rate"] = log["pf_fold"] / H

#     log["button_ev_per_hand"] = log["ev_pos"][1] / max(1, log["hands_pos"][1])
#     log["blind_ev_per_hand"] = log["ev_pos"][0] / max(1, log["hands_pos"][0])

#     elapsed = time.perf_counter() - t0
#     logging.getLogger(__name__).info(f"Finished eval_match_cpu: {num_games} games in {elapsed:0.3f}s — EV={log['ev_per_hand']:.4f}")
#     return log


import torch
import torch.nn.functional as F
import numpy as np

from config import DEVICE
from encode_state import encode_state
from engine import SimpleHoldemEnv9, NUM_ACTIONS


ACTIONS = ["FOLD","CHECK","CALL","R2X","R3X","HALF","POT","10BB","ALLIN"]
STREETS = ["PREFLOP","FLOP","TURN","RIVER"]
NUM_STREETS = 4


# =====================================================================
# MASKED SOFTMAX SAMPLER
# =====================================================================
@torch.no_grad()
def sample_action_from_policy(state, legal, net):
    """Pick action using masked softmax over legal actions only."""
    x = encode_state(state, state.to_act).to(DEVICE)
    logits = net(x.unsqueeze(0))[0]  # shape [A]

    mask = torch.full((NUM_ACTIONS,), -1e9, device=DEVICE)
    for a in legal:
        mask[a] = 0.0

    probs = F.softmax(logits + mask, dim=-1)

    # Numerical safety: fallback
    if torch.isnan(probs).any() or probs.sum() < 1e-12:
        return np.random.choice(legal)

    a = torch.multinomial(probs, 1).item()
    if a not in legal:
        a = np.random.choice(legal)

    return a


# =====================================================================
# POSITION-TRACKING HELPER  (perfect, no swapping bugs)
# =====================================================================
def detect_positions(bot0_seat_SB):
    """
    bot0_seat_SB=True  → bot0=SB (BTN)
    bot0_seat_SB=False → bot0=BB
    return pos_map for stats:
           1 → Button/SB, 0 → BB
    """
    if bot0_seat_SB:
        return {0:1, 1:0}   # bot0=SB, bot1=BB
    else:
        return {0:0, 1:1}   # bot0=BB, bot1=SB


# =====================================================================
# PRETTY PRINTER (unchanged)
# =====================================================================
def print_eval_stats_colored(stats):
    # unchanged — your original pretty printer
    from eval_printer import print_eval_stats_colored_impl
    return print_eval_stats_colored_impl(stats)



# =====================================================================
# MAIN: CLEAN CFR-CONSISTENT EVALUATION LOOP
# =====================================================================
def eval_match_cpu(env: SimpleHoldemEnv9, bot0, bot1, num_games=200):

    bot0.eval()
    bot1.eval()

    # --------------------------
    # init log
    # --------------------------
    log = {
        "hands": 0,
        "ev": 0.0,
        "win": 0,
        "sd": 0,             # showdown
        "vpip": 0,
        "agg": 0,
        "pf_fold": 0,

        "ev_pos": np.zeros(2),
        "hands_pos": np.zeros(2),

        "action_counts": np.zeros(NUM_ACTIONS),
        "street_action": np.zeros((NUM_STREETS, NUM_ACTIONS)),
        "pos_action": np.zeros((2, NUM_ACTIONS)),
    }

    # =====================================================================
    # PLAY SINGLE MATCH
    # =====================================================================
    def play_one_game(bot0_seat_SB):
        """
        bot0_seat_SB=True → bot0=SB
        bot0_seat_SB=False → bot0=BB
        """
        s = env.new_hand()

        # --- IMPORTANT ---
        # Swap seats *visually* by swapping cards+stacks only.
        # Engine still has sb_player=0,bb_player=1 but this is fine,
        # because we track bot positions separately.
        if not bot0_seat_SB:
            s.hole = [s.hole[1], s.hole[0]]
            s.stacks = [s.stacks[1], s.stacks[0]]
            s.contrib = [s.contrib[1], s.contrib[0]]
            s.initial_stacks = [s.initial_stacks[1], s.initial_stacks[0]]

        # Precompute "real seat" mapping for stats
        pos_map = detect_positions(bot0_seat_SB)

        history = []

        # ===== MAIN LOOP =====
        while not s.terminal:
            p = s.to_act
            legal = env.legal_actions(s)

            net = bot0 if p == 0 else bot1
            a = sample_action_from_policy(s, legal, net)

            history.append((p, s.street, pos_map[p], a))

            s = env.step(s, a)

        # payoff for bot0
        payoff0 = env.terminal_payoff(s, 0)
        winner = s.winner

        return payoff0, winner, history, bot0_seat_SB


    # =====================================================================
    # UPDATE LOG
    # =====================================================================
    def update(log, payoff0, winner, hist, bot0_seat_SB):
        pid = 0
        log["hands"] += 1
        log["ev"] += payoff0

        if winner == 0:
            log["win"] += 1

        # Position of bot0 this hand
        pos = 1 if bot0_seat_SB else 0  # 1 = Button/SB, 0 = BB
        log["ev_pos"][pos] += payoff0
        log["hands_pos"][pos] += 1

        saw_vpip = False
        saw_agg = False

        for p, street, pos_, a in hist:
            if p == pid:
                log["action_counts"][a] += 1
                log["street_action"][street][a] += 1
                log["pos_action"][pos_][a] += 1

                if street == 0 and a != 0:
                    saw_vpip = True
                if a >= 3:
                    saw_agg = True

        if saw_vpip: log["vpip"] += 1
        if payoff0 != 0: log["sd"] += 1
        if saw_agg: log["agg"] += 1

        # Preflop fold?
        first_action = hist[0][3]
        if pos == 1 and first_action == 0:
            log["pf_fold"] += 1


    # =====================================================================
    # MAIN LOOP: alternate seats
    # =====================================================================
    for g in range(num_games):
        bot0_SB = (g % 2 == 0)
        payoff0, winner, hist, seatflag = play_one_game(bot0_SB)
        update(log, payoff0, winner, hist, seatflag)


    # =====================================================================
    # FINAL AGGREGATION
    # =====================================================================
    H = max(1, log["hands"])
    log["ev_per_hand"] = log["ev"] / H
    log["win_rate"] = log["win"] / H
    log["showdown_rate"] = log["sd"] / H
    log["vpip"] = log["vpip"] / H
    log["agg_freq"] = log["agg"] / H
    log["preflop_fold_rate"] = log["pf_fold"] / H

    log["button_ev_per_hand"] = log["ev_pos"][1] / max(1, log["hands_pos"][1])
    log["blind_ev_per_hand"]  = log["ev_pos"][0] / max(1, log["hands_pos"][0])

    return log







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
