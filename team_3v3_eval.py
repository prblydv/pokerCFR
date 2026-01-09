import argparse
import logging
import os
import random
from typing import Dict, List, Tuple

import torch

from config import DEVICE, LOG_LEVEL, LOG_FORMAT
from poker_env import (
    SimpleHoldemEnv,
    NUM_ACTIONS,
    ACTION_CALL,
    STREET_PREFLOP,
)
from abstraction import encode_state
from networks import PolicyNet, move_to_device
from deep_cfr_trainer import scripted_eval_action, scripted_tag_action, RAISE_ACTIONS
from opponent_model import ExploitController, update_from_action


logger = logging.getLogger("Team3v3Eval")
RNG = random.Random(1337)


def _parse_seat_list(value: str) -> List[int]:
    if not value:
        return []
    return [int(x.strip()) for x in value.split(",") if x.strip() != ""]


def _sample_policy_action(policy_net: PolicyNet, state, player: int, legal_actions: List[int]) -> int:
    if not legal_actions:
        return 0
    x = encode_state(state, player).to(DEVICE)
    with torch.no_grad():
        logits = policy_net(x.unsqueeze(0)).squeeze(0)

    mask = torch.full((NUM_ACTIONS,), -1e9, device=logits.device)
    for a in legal_actions:
        mask[a] = 0.0

    probs = torch.softmax(logits + mask, dim=-1)
    action = torch.multinomial(probs, 1).item()
    if action not in legal_actions:
        action = RNG.choice(legal_actions)
    return action


def _init_stats(num_players: int) -> Dict[int, Dict]:
    return {
        pid: {
            "hands": 0,
            "wins": 0,
            "showdown_hands": 0,
            "showdown_wins": 0,
            "vpip": 0,
            "pfr": 0,
            "aggr": 0,
            "calls": 0,
            "actions": 0,
            "payoff": 0.0,
        }
        for pid in range(num_players)
    }


def _select_scripted_opponent(state, player: int, scripted_seats: set) -> int:
    last_aggr = getattr(state, "last_aggressor", None)
    if last_aggr is not None and last_aggr in scripted_seats and last_aggr != player:
        if not state.folded[last_aggr] and state.stacks[last_aggr] > 0:
            return last_aggr
    for i in range(state.num_players):
        pid = (player + 1 + i) % state.num_players
        if pid in scripted_seats and pid != player:
            if not state.folded[pid] and state.stacks[pid] > 0:
                return pid
    return None


def _summarize_player(stats: Dict) -> Dict:
    hands = max(stats["hands"], 1)
    win_pct = 100.0 * stats["wins"] / hands
    vpip_pct = 100.0 * stats["vpip"] / hands
    pfr_pct = 100.0 * stats["pfr"] / hands
    calls = stats["calls"]
    aggr = stats["aggr"]
    if calls == 0:
        af = float("inf") if aggr > 0 else 0.0
    else:
        af = aggr / calls
    avg_actions = stats["actions"] / hands
    ev = stats["payoff"] / hands
    sd_hands = max(stats["showdown_hands"], 1)
    sd_win_pct = 100.0 * stats["showdown_wins"] / sd_hands
    sd_seen_pct = 100.0 * stats["showdown_hands"] / hands
    return {
        "hands": hands,
        "win_pct": win_pct,
        "showdown_win_pct": sd_win_pct,
        "showdown_seen_pct": sd_seen_pct,
        "vpip_pct": vpip_pct,
        "pfr_pct": pfr_pct,
        "af": af,
        "avg_actions": avg_actions,
        "ev": ev,
    }


def _team_summary(stats: Dict[int, Dict], seats: List[int], total_hands: int, bb: float) -> Dict:
    if not seats:
        return {}
    total_ev = sum(stats[pid]["payoff"] for pid in seats)
    team_wins = sum(stats[pid]["wins"] for pid in seats)
    team_sd_wins = sum(stats[pid]["showdown_wins"] for pid in seats)
    team_sd_hands = sum(stats[pid]["showdown_hands"] for pid in seats)
    avg_ev = total_ev / max(1, total_hands)
    sd_win_pct = 100.0 * team_sd_wins / max(1, team_sd_hands)
    return {
        "team_seats": seats,
        "hands": total_hands,
        "wins": team_wins,
        "avg_ev": avg_ev,
        "bb_per_100": (avg_ev / max(bb, 1e-9)) * 100.0,
        "showdown_win_pct": sd_win_pct,
    }


def run_match(
    policy_path: str,
    num_hands: int,
    num_players: int,
    scripted_eval_seats: List[int],
    scripted_tag_seats: List[int],
    use_opponent_modeling: bool,
) -> Tuple[Dict[int, Dict], Dict]:
    env = SimpleHoldemEnv(num_players=num_players)

    if not os.path.isfile(policy_path):
        raise FileNotFoundError(f"Policy file not found: {policy_path}")

    state_dim = encode_state(env.new_hand(), 0).shape[0]
    policy = move_to_device(PolicyNet(state_dim))
    policy.load_state_dict(torch.load(policy_path, map_location=DEVICE))
    policy.eval()

    stats = _init_stats(num_players)
    exploit_controller = ExploitController(num_players) if use_opponent_modeling else None

    scripted_eval_seats = set(scripted_eval_seats)
    scripted_tag_seats = set(scripted_tag_seats)
    scripted_seats = scripted_eval_seats | scripted_tag_seats
    policy_seats = [p for p in range(num_players) if p not in scripted_seats]

    for _ in range(num_hands):
        s = env.new_hand()
        hand_flags = {
            pid: {"vpip": False, "pfr": False, "aggr": 0, "calls": 0, "actions": 0}
            for pid in range(num_players)
        }

        while not s.terminal:
            legal = env.legal_actions(s)
            if not legal:
                break
            player = s.to_act

            if player in scripted_eval_seats:
                action = scripted_eval_action(s, player, legal)
            elif player in scripted_tag_seats:
                action = scripted_tag_action(s, player, legal)
            else:
                if exploit_controller is not None:
                    x = encode_state(s, player).to(DEVICE)
                    with torch.no_grad():
                        logits = policy(x.unsqueeze(0)).squeeze(0)
                    opp = _select_scripted_opponent(s, player, scripted_seats)
                    if opp is not None:
                        action = exploit_controller.choose_action(
                            s,
                            player,
                            legal,
                            logits,
                            opponent_override=opp,
                        )
                    else:
                        action = _sample_policy_action(policy, s, player, legal)
                else:
                    action = _sample_policy_action(policy, s, player, legal)

            info = hand_flags[player]
            info["actions"] += 1

            if s.street == STREET_PREFLOP:
                if action == ACTION_CALL or action in RAISE_ACTIONS:
                    info["vpip"] = True
                if action in RAISE_ACTIONS:
                    info["pfr"] = True

            if action in RAISE_ACTIONS:
                info["aggr"] += 1
            elif action == ACTION_CALL:
                info["calls"] += 1

            if exploit_controller is not None and player in scripted_seats:
                to_call = max(0.0, s.current_bet - s.contrib[player])
                facing_bet = to_call > 0
                bet_bucket = action if action in RAISE_ACTIONS else None
                update_from_action(
                    exploit_controller,
                    s,
                    player,
                    action,
                    s.street,
                    facing_bet,
                    bet_bucket,
                )

            s = env.step(s, action)

        showdown = s.terminal and len(s.board) >= 5

        for pid in range(num_players):
            stats[pid]["hands"] += 1
            if s.winner == pid:
                stats[pid]["wins"] += 1
                if showdown:
                    stats[pid]["showdown_wins"] += 1
            if showdown and not s.folded[pid]:
                stats[pid]["showdown_hands"] += 1
            if hand_flags[pid]["vpip"]:
                stats[pid]["vpip"] += 1
            if hand_flags[pid]["pfr"]:
                stats[pid]["pfr"] += 1
            stats[pid]["aggr"] += hand_flags[pid]["aggr"]
            stats[pid]["calls"] += hand_flags[pid]["calls"]
            stats[pid]["actions"] += hand_flags[pid]["actions"]
            stats[pid]["payoff"] += env.terminal_payoff(s, pid)

    summaries = {pid: _summarize_player(stats[pid]) for pid in range(num_players)}
    team_policy = _team_summary(stats, policy_seats, num_hands, env.bb)
    team_scripted = _team_summary(stats, list(scripted_seats), num_hands, env.bb)

    return summaries, {"policy_team": team_policy, "scripted_team": team_scripted}


def _print_results(summaries: Dict[int, Dict], team_summaries: Dict):
    def fmt_af(value):
        return "inf" if value == float("inf") else f"{value:.2f}"

    print("Per-player results")
    for pid, summary in summaries.items():
        print(
            f"P{pid}: hands={summary['hands']}, win%={summary['win_pct']:.1f}, "
            f"ev/hand={summary['ev']:.3f}, showdown_win%={summary['showdown_win_pct']:.1f}, "
            f"showdown_seen%={summary['showdown_seen_pct']:.1f}, VPIP%={summary['vpip_pct']:.1f}, "
            f"PFR%={summary['pfr_pct']:.1f}, AF={fmt_af(summary['af'])}, "
            f"avg_actions={summary['avg_actions']:.2f}"
        )

    print("\nTeam results")
    for label, summary in team_summaries.items():
        if not summary:
            continue
        print(
            f"{label}: seats={summary['team_seats']}, hands={summary['hands']}, "
            f"wins={summary['wins']}, avg_ev/hand={summary['avg_ev']:.3f}, "
            f"bb/100={summary['bb_per_100']:.2f}, showdown_win%={summary['showdown_win_pct']:.1f}"
        )


def main() -> None:
    policy_path = "models/policyev_42.pt"
    hands = 5000
    num_players = 6
    scripted_team_seats = "0,2,4"
    use_opponent_modeling = True

    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)

    scripted_team_seats = _parse_seat_list(scripted_team_seats)
    if len(scripted_team_seats) == 0:
        raise ValueError("At least one scripted seat is required for a 3v3 matchup.")
    if len(scripted_team_seats) * 2 != num_players:
        logger.warning(
            "Non-3v3 setup: scripted seats=%s, num_players=%s",
            sorted(scripted_team_seats),
            num_players,
        )

    print("Match A: scripted_eval_action team vs policy team")
    summaries_a, team_summaries_a = run_match(
        policy_path=policy_path,
        num_hands=hands,
        num_players=num_players,
        scripted_eval_seats=scripted_team_seats,
        scripted_tag_seats=[],
        use_opponent_modeling=use_opponent_modeling,
    )
    _print_results(summaries_a, team_summaries_a)

    print("\nMatch B: scripted_tag_action team vs policy team")
    summaries_b, team_summaries_b = run_match(
        policy_path=policy_path,
        num_hands=hands,
        num_players=num_players,
        scripted_eval_seats=[],
        scripted_tag_seats=scripted_team_seats,
        use_opponent_modeling=use_opponent_modeling,
    )
    _print_results(summaries_b, team_summaries_b)


if __name__ == "__main__":
    main()
# (venv) C:\Users\PRABAL YADAV\Desktop\machine learning iim\pokerbotPlayOnline - Copy (2)>python team_3v3_eval.py
# Match A: scripted_eval_action team vs policy team
# C:\Users\PRABAL YADAV\Desktop\machine learning iim\pokerbotPlayOnline - Copy (2)\team_3v3_eval.py:129: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
#   policy.load_state_dict(torch.load(policy_path, map_location=DEVICE))
# Per-player results
# P0: hands=50000, win%=17.5, ev/hand=-2.342, showdown_win%=30.4, showdown_seen%=57.2, VPIP%=83.3, PFR%=2.2, AF=0.02, avg_actions=3.54
# P1: hands=50000, win%=14.4, ev/hand=2.375, showdown_win%=32.9, showdown_seen%=26.1, VPIP%=32.7, PFR%=23.6, AF=3.65, avg_actions=1.73
# P2: hands=50000, win%=18.1, ev/hand=-2.344, showdown_win%=31.2, showdown_seen%=57.6, VPIP%=83.4, PFR%=2.2, AF=0.02, avg_actions=3.56
# P3: hands=50000, win%=14.4, ev/hand=2.638, showdown_win%=32.3, showdown_seen%=26.4, VPIP%=33.1, PFR%=23.9, AF=3.60, avg_actions=1.74
# P4: hands=50000, win%=17.4, ev/hand=-2.337, showdown_win%=30.1, showdown_seen%=57.1, VPIP%=83.2, PFR%=2.1, AF=0.02, avg_actions=3.55
# P5: hands=50000, win%=14.5, ev/hand=2.010, showdown_win%=32.6, showdown_seen%=26.2, VPIP%=33.0, PFR%=23.9, AF=3.66, avg_actions=1.73

# Team results
# policy_team: seats=[1, 3, 5], hands=50000, wins=21667, avg_ev/hand=7.023, bb/100=351.17, showdown_win%=32.6
# scripted_team: seats=[0, 2, 4], hands=50000, wins=26488, avg_ev/hand=-7.023, bb/100=-351.17, showdown_win%=30.6

# Match B: scripted_tag_action team vs policy team
# Per-player results
# P0: hands=50000, win%=5.9, ev/hand=0.341, showdown_win%=57.5, showdown_seen%=3.5, VPIP%=4.5, PFR%=2.7, AF=0.91, avg_actions=1.16
# P1: hands=50000, win%=27.1, ev/hand=-0.480, showdown_win%=46.7, showdown_seen%=12.1, VPIP%=34.7, PFR%=24.1, AF=2.20, avg_actions=1.35
# P2: hands=50000, win%=6.2, ev/hand=0.553, showdown_win%=60.1, showdown_seen%=3.7, VPIP%=4.8, PFR%=2.9, AF=0.96, avg_actions=1.16
# P3: hands=50000, win%=26.9, ev/hand=-0.037, showdown_win%=47.8, showdown_seen%=12.3, VPIP%=34.9, PFR%=24.1, AF=2.18, avg_actions=1.35
# P4: hands=50000, win%=6.0, ev/hand=0.378, showdown_win%=58.1, showdown_seen%=3.5, VPIP%=4.6, PFR%=2.7, AF=0.94, avg_actions=1.16
# P5: hands=50000, win%=27.1, ev/hand=-0.754, showdown_win%=46.3, showdown_seen%=12.2, VPIP%=34.9, PFR%=24.3, AF=2.20, avg_actions=1.35

# Team results
# policy_team: seats=[1, 3, 5], hands=50000, wins=40528, avg_ev/hand=-1.272, bb/100=-63.61, showdown_win%=46.9
# scripted_team: seats=[0, 2, 4], hands=50000, wins=9081, avg_ev/hand=1.272, bb/100=63.61, showdown_win%=58.6