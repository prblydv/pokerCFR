"""
policy_compare.py
-----------------
Run policy-vs-policy matches for N hands and compare win rates, EV, ranges,
and betting tendencies (c-bets, bluffs, showdowns).

Usage:
    python policy_compare.py --hands 1000
    python policy_compare.py --hands 5000 --models "models/policy.pt" "models/policy phase1.pt"
"""

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import DEVICE, DETERMINISTIC_SEED, NUM_PLAYERS, BIG_BLIND
import poker_env
from poker_env import (
    SimpleHoldemEnv,
    ACTION_FOLD,
    ACTION_CHECK,
    ACTION_CALL,
    ACTION_BET_POT_25,
    ACTION_BET_POT_50,
    ACTION_BET_POT_100,
    ACTION_BET_POT_200,
    ACTION_ALL_IN,
    NUM_ACTIONS,
    STREET_PREFLOP,
    STREET_FLOP,
    STREET_TURN,
    STREET_RIVER,
)
from abstraction import encode_state, card_rank, card_suit, normalized_strength
from networks import PolicyNet

VPIP_ACTIONS = {
    ACTION_CALL,
    ACTION_BET_POT_25,
    ACTION_BET_POT_50,
    ACTION_BET_POT_100,
    ACTION_BET_POT_200,
    ACTION_ALL_IN,
}
RAISE_ACTIONS = {
    ACTION_BET_POT_25,
    ACTION_BET_POT_50,
    ACTION_BET_POT_100,
    ACTION_BET_POT_200,
    ACTION_ALL_IN,
}
RANKS = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]
ACTION_LABELS = {
    ACTION_FOLD: "Fold",
    ACTION_CHECK: "Check",
    ACTION_CALL: "Call",
    ACTION_BET_POT_25: "Bet 25%",
    ACTION_BET_POT_50: "Bet 50%",
    ACTION_BET_POT_100: "Bet 100%",
    ACTION_BET_POT_200: "Bet 200%",
    ACTION_ALL_IN: "All-in",
}
ACTION_COLORS = {
    ACTION_FOLD: "#c7c7c7",
    ACTION_CHECK: "#a6cee3",
    ACTION_CALL: "#1f78b4",
    ACTION_BET_POT_25: "#b2df8a",
    ACTION_BET_POT_50: "#33a02c",
    ACTION_BET_POT_100: "#1b7837",
    ACTION_BET_POT_200: "#e66101",
    ACTION_ALL_IN: "#fb9a99",
}
STREET_ORDER = [STREET_PREFLOP, STREET_FLOP, STREET_TURN, STREET_RIVER]
STREET_NAMES = {
    STREET_PREFLOP: "Preflop",
    STREET_FLOP: "Flop",
    STREET_TURN: "Turn",
    STREET_RIVER: "River",
}
BLUFF_STRENGTH_THRESHOLD = 0.45


@dataclass
class PolicySpec:
    name: str
    path: str
    net: PolicyNet


@dataclass
class PolicyStats:
    name: str
    hands: int = 0
    wins: int = 0
    profit: float = 0.0
    vpip_count: int = 0
    pfr_count: int = 0
    aggression_acts: int = 0
    call_acts: int = 0
    total_actions: int = 0
    showdowns: int = 0
    showdown_wins: int = 0
    cbet_opportunities: int = 0
    cbets: int = 0
    cbet_successes: int = 0
    cbet_faced: int = 0
    fold_to_cbet: int = 0
    bluff_attempts: int = 0
    bluff_caught: int = 0
    bluff_got_through: int = 0
    bluff_called_won: int = 0
    aggressive_postflop_actions: int = 0

    def record_hand(
        self,
        won: bool,
        profit: float,
        vpip: bool,
        pfr: bool,
        aggr_acts: int,
        call_acts: int,
        action_count: int,
    ):
        self.hands += 1
        if won:
            self.wins += 1
        self.profit += profit
        if vpip:
            self.vpip_count += 1
        if pfr:
            self.pfr_count += 1
        self.aggression_acts += aggr_acts
        self.call_acts += call_acts
        self.total_actions += action_count

    def summary(self) -> Dict[str, float]:
        hands = max(self.hands, 1)
        win_pct = 100.0 * self.wins / hands
        vpip_pct = 100.0 * self.vpip_count / hands
        pfr_pct = 100.0 * self.pfr_count / hands
        af = self.aggression_acts / self.call_acts if self.call_acts > 0 else (float("inf") if self.aggression_acts > 0 else 0.0)
        avg_actions = self.total_actions / hands
        avg_profit = self.profit / hands
        bb_per_100 = (avg_profit / max(1e-9, BIG_BLIND)) * 100.0
        showdown_pct = 100.0 * self.showdowns / hands
        showdown_win_pct = 100.0 * self.showdown_wins / max(self.showdowns, 1)
        cbet_pct = 100.0 * self.cbets / max(self.cbet_opportunities, 1)
        cbet_success_pct = 100.0 * self.cbet_successes / max(self.cbets, 1)
        fold_to_cbet_pct = 100.0 * self.fold_to_cbet / max(self.cbet_faced, 1)
        bluff_rate_pct = 100.0 * self.bluff_attempts / max(self.aggressive_postflop_actions, 1)
        bluff_caught_pct = 100.0 * self.bluff_caught / max(self.bluff_attempts, 1)
        bluff_got_through_pct = 100.0 * self.bluff_got_through / max(self.bluff_attempts, 1)
        bluff_called_won_pct = 100.0 * self.bluff_called_won / max(self.bluff_attempts, 1)
        return {
            "hands": self.hands,
            "win_pct": win_pct,
            "vpip_pct": vpip_pct,
            "pfr_pct": pfr_pct,
            "af": af,
            "avg_actions": avg_actions,
            "avg_profit": avg_profit,
            "bb_per_100": bb_per_100,
            "showdown_pct": showdown_pct,
            "showdown_win_pct": showdown_win_pct,
            "cbet_opportunities": self.cbet_opportunities,
            "cbets": self.cbets,
            "cbet_successes": self.cbet_successes,
            "cbet_pct": cbet_pct,
            "cbet_success_pct": cbet_success_pct,
            "cbet_faced": self.cbet_faced,
            "fold_to_cbet": self.fold_to_cbet,
            "fold_to_cbet_pct": fold_to_cbet_pct,
            "bluff_attempts": self.bluff_attempts,
            "bluff_caught": self.bluff_caught,
            "bluff_got_through": self.bluff_got_through,
            "bluff_called_won": self.bluff_called_won,
            "bluff_rate_pct": bluff_rate_pct,
            "bluff_caught_pct": bluff_caught_pct,
            "bluff_got_through_pct": bluff_got_through_pct,
            "bluff_called_won_pct": bluff_called_won_pct,
            "aggressive_postflop_actions": self.aggressive_postflop_actions,
        }


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    poker_env.RNG.seed(seed)


def find_policy_paths(models_dir: str) -> List[str]:
    return sorted(str(p) for p in Path(models_dir).glob("policy*.pt") if p.is_file())


def load_policy(state_dim: int, path: str) -> PolicyNet:
    net = PolicyNet(state_dim)
    try:
        state_dict = torch.load(path, map_location=DEVICE, weights_only=True)
    except TypeError:
        state_dict = torch.load(path, map_location=DEVICE)
    net.load_state_dict(state_dict)
    net.to(DEVICE)
    net.eval()
    return net


def load_policies(state_dim: int, paths: List[str]) -> List[PolicySpec]:
    policies = []
    for path in paths:
        name = Path(path).stem
        net = load_policy(state_dim, path)
        policies.append(PolicySpec(name=name, path=path, net=net))
    return policies


def masked_policy_probs(policy_net: PolicyNet, state, player: int, legal_actions: List[int]) -> torch.Tensor:
    x = encode_state(state, player).to(DEVICE).unsqueeze(0)
    with torch.no_grad():
        logits = policy_net(x).squeeze(0)
    mask = torch.full((NUM_ACTIONS,), -1e9, device=logits.device)
    for a in legal_actions:
        mask[a] = 0.0
    return torch.softmax(logits + mask, dim=-1)


def choose_action(policy_net: PolicyNet, state, player: int, legal_actions: List[int]) -> int:
    probs = masked_policy_probs(policy_net, state, player, legal_actions)
    action = torch.multinomial(probs, 1).item()
    if action not in legal_actions:
        action = random.choice(legal_actions)
    return action


def next_live_actor(state, current: int) -> int:
    for i in range(1, state.num_players + 1):
        nxt = (current + i) % state.num_players
        if not state.folded[nxt] and state.stacks[nxt] > 0:
            return nxt
    return -1


def hole_to_grid_indices(hole: List[int]) -> tuple:
    r1 = card_rank(hole[0])
    r2 = card_rank(hole[1])
    idx1 = 12 - (r1 - 2)
    idx2 = 12 - (r2 - 2)
    if r1 == r2:
        return idx1, idx1
    suited = card_suit(hole[0]) == card_suit(hole[1])
    hi = min(idx1, idx2)
    lo = max(idx1, idx2)
    if suited:
        return hi, lo
    return lo, hi


def run_matches(
    policies: List[PolicySpec],
    num_hands: int,
    num_players: int,
    seed: int,
) -> Dict:
    seed_all(seed)
    env = SimpleHoldemEnv(num_players=num_players)
    seat_to_policy = [i % len(policies) for i in range(num_players)]
    stats = {p.name: PolicyStats(p.name) for p in policies}
    action_counts = {
        p.name: {street: np.zeros(NUM_ACTIONS, dtype=np.int64) for street in STREET_ORDER}
        for p in policies
    }
    strength_by_action = {
        p.name: {
            street: {"aggr_sum": 0.0, "aggr_count": 0, "passive_sum": 0.0, "passive_count": 0}
            for street in STREET_ORDER
        }
        for p in policies
    }
    range_counts = {
        p.name: {"vpip": np.zeros((13, 13), dtype=np.float64), "total": np.zeros((13, 13), dtype=np.float64)}
        for p in policies
    }

    for _ in range(num_hands):
        state = env.new_hand()
        holes = [h[:] for h in state.hole]
        per_hand_flags = {
            pid: {"vpip": False, "pfr": False, "aggr": 0, "calls": 0, "actions": 0}
            for pid in range(num_players)
        }
        last_preflop_aggressor = None
        cbet_active = None
        cbet_opportunity_recorded = set()
        open_bluff_by_player = {}
        bluff_events = []
        last_aggressive_player = None

        while not state.terminal:
            player = state.to_act
            if player is None or player < 0:
                break
            legal = env.legal_actions(state)
            if not legal:
                state.folded[player] = True
                state.players_acted[player] = True
                nxt = next_live_actor(state, player)
                if nxt < 0:
                    break
                state.to_act = nxt
                continue

            policy_idx = seat_to_policy[player]
            policy = policies[policy_idx].net
            policy_name = policies[policy_idx].name
            to_call = max(0.0, state.current_bet - state.contrib[player])

            if (
                state.street == STREET_FLOP
                and player == last_preflop_aggressor
                and player not in cbet_opportunity_recorded
                and state.current_bet == 0
            ):
                stats[policy_name].cbet_opportunities += 1
                cbet_opportunity_recorded.add(player)

            action = choose_action(policy, state, player, legal)
            action_counts[policy_name][state.street][action] += 1
            info = per_hand_flags[player]
            info["actions"] += 1

            if state.street == STREET_PREFLOP:
                if action in VPIP_ACTIONS:
                    info["vpip"] = True
                if action in RAISE_ACTIONS:
                    info["pfr"] = True

            if action in RAISE_ACTIONS:
                info["aggr"] += 1
            elif action == ACTION_CALL:
                info["calls"] += 1

            strength = None
            if action in RAISE_ACTIONS or action in (ACTION_CALL, ACTION_CHECK):
                strength = normalized_strength(state.hole[player], state.board)
                street_stats = strength_by_action[policy_name][state.street]
                if action in RAISE_ACTIONS:
                    street_stats["aggr_sum"] += strength
                    street_stats["aggr_count"] += 1
                else:
                    street_stats["passive_sum"] += strength
                    street_stats["passive_count"] += 1

            if state.street >= STREET_FLOP and action in RAISE_ACTIONS:
                stats[policy_name].aggressive_postflop_actions += 1

            if state.street == STREET_PREFLOP and action in RAISE_ACTIONS:
                last_preflop_aggressor = player

            if (
                state.street == STREET_FLOP
                and player == last_preflop_aggressor
                and state.current_bet == 0
                and action in RAISE_ACTIONS
            ):
                stats[policy_name].cbets += 1
                cbet_active = {"player": player, "faced_resistance": False}

            if (
                cbet_active
                and state.street == STREET_FLOP
                and player != cbet_active["player"]
                and to_call > 0
            ):
                stats[policy_name].cbet_faced += 1
                if action == ACTION_FOLD:
                    stats[policy_name].fold_to_cbet += 1
                if action in ({ACTION_CALL} | RAISE_ACTIONS):
                    cbet_active["faced_resistance"] = True

            if state.street >= STREET_FLOP and action in RAISE_ACTIONS:
                if strength is None:
                    strength = normalized_strength(state.hole[player], state.board)
                if strength <= BLUFF_STRENGTH_THRESHOLD:
                    stats[policy_name].bluff_attempts += 1
                    bluff_events.append(
                        {"player": player, "policy": policy_name, "called": False, "street": state.street}
                    )
                    open_bluff_by_player[player] = len(bluff_events) - 1

            if (
                to_call > 0
                and action in ({ACTION_CALL} | RAISE_ACTIONS)
                and last_aggressive_player is not None
                and last_aggressive_player != player
            ):
                idx = open_bluff_by_player.pop(last_aggressive_player, None)
                if idx is not None:
                    bluff_events[idx]["called"] = True

            if action in RAISE_ACTIONS:
                last_aggressive_player = player

            prev_street = state.street
            state = env.step(state, action)

            if state.street != prev_street:
                if cbet_active and prev_street == STREET_FLOP:
                    cbet_policy = policies[seat_to_policy[cbet_active["player"]]].name
                    if not cbet_active["faced_resistance"]:
                        stats[cbet_policy].cbet_successes += 1
                    cbet_active = None
                for pid, idx in list(open_bluff_by_player.items()):
                    if bluff_events[idx]["street"] == prev_street and not bluff_events[idx]["called"]:
                        bluff_events[idx]["called"] = True
                        open_bluff_by_player.pop(pid, None)

            if state.terminal and cbet_active:
                cbet_policy = policies[seat_to_policy[cbet_active["player"]]].name
                if not cbet_active["faced_resistance"]:
                    stats[cbet_policy].cbet_successes += 1
                cbet_active = None

        for pid in range(num_players):
            policy_name = policies[seat_to_policy[pid]].name
            info = per_hand_flags[pid]
            won = state.winner == pid
            profit = state.stacks[pid] - state.initial_stacks[pid]
            stats[policy_name].record_hand(
                won=won,
                profit=profit,
                vpip=info["vpip"],
                pfr=info["pfr"],
                aggr_acts=info["aggr"],
                call_acts=info["calls"],
                action_count=info["actions"],
            )

            row, col = hole_to_grid_indices(holes[pid])
            range_counts[policy_name]["total"][row, col] += 1.0
            if info["vpip"]:
                range_counts[policy_name]["vpip"][row, col] += 1.0

        live_players = [i for i in range(num_players) if not state.folded[i]]
        if len(live_players) >= 2:
            for pid in live_players:
                policy_name = policies[seat_to_policy[pid]].name
                stats[policy_name].showdowns += 1
                if state.winner == pid:
                    stats[policy_name].showdown_wins += 1

        for event in bluff_events:
            policy_name = event["policy"]
            if event["called"]:
                if state.winner == event["player"]:
                    stats[policy_name].bluff_called_won += 1
                else:
                    stats[policy_name].bluff_caught += 1
            else:
                if state.winner == event["player"]:
                    stats[policy_name].bluff_got_through += 1
                else:
                    stats[policy_name].bluff_caught += 1

    return {
        "stats": stats,
        "seat_to_policy": seat_to_policy,
        "range_counts": range_counts,
        "action_counts": action_counts,
        "strength_by_action": strength_by_action,
    }


def plot_bar(ax, names, values, title, color, y_label=None, allow_negative=False):
    ax.bar(names, values, color=color)
    ax.set_title(title)
    if y_label:
        ax.set_ylabel(y_label)
    ax.tick_params(axis="x", rotation=20)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    if allow_negative:
        low = min(values + [0.0])
        high = max(values + [0.0])
        pad = max((high - low) * 0.2, 1.0)
        ax.set_ylim(low - pad, high + pad)
        ax.axhline(0, color="#666666", linewidth=0.8)
    else:
        ax.set_ylim(0, max(values + [1.0]) * 1.2)


def save_summary_plot(stats: Dict[str, PolicyStats], out_dir: str) -> str:
    names = list(stats.keys())
    win_pcts = [stats[n].summary()["win_pct"] for n in names]
    bb_100 = [stats[n].summary()["bb_per_100"] for n in names]
    vpip_pcts = [stats[n].summary()["vpip_pct"] for n in names]
    pfr_pcts = [stats[n].summary()["pfr_pct"] for n in names]
    wtsd_pcts = [stats[n].summary()["showdown_pct"] for n in names]
    wsd_pcts = [stats[n].summary()["showdown_win_pct"] for n in names]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    plot_bar(axes[0, 0], names, win_pcts, "Win Rate (per seat)", "#4c72b0", "Win %")
    plot_bar(axes[0, 1], names, bb_100, "Profitability", "#55a868", "BB/100", allow_negative=True)
    plot_bar(axes[0, 2], names, vpip_pcts, "VPIP", "#c44e52", "VPIP %")
    plot_bar(axes[1, 0], names, pfr_pcts, "PFR", "#8172b3", "PFR %")
    plot_bar(axes[1, 1], names, wtsd_pcts, "Went To Showdown", "#937860", "WTS %")
    plot_bar(axes[1, 2], names, wsd_pcts, "Won At Showdown", "#da8bc3", "WSD %")

    plt.tight_layout()
    out_path = os.path.join(out_dir, "policy_compare_summary.png")
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def save_tendencies_plot(stats: Dict[str, PolicyStats], out_dir: str) -> str:
    names = list(stats.keys())
    cbet_pcts = [stats[n].summary()["cbet_pct"] for n in names]
    cbet_success = [stats[n].summary()["cbet_success_pct"] for n in names]
    fold_to_cbet = [stats[n].summary()["fold_to_cbet_pct"] for n in names]
    bluff_rate = [stats[n].summary()["bluff_rate_pct"] for n in names]
    bluff_caught = [stats[n].summary()["bluff_caught_pct"] for n in names]
    bluff_got_through = [stats[n].summary()["bluff_got_through_pct"] for n in names]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    plot_bar(axes[0, 0], names, cbet_pcts, "CBet", "#4c72b0", "CBet %")
    plot_bar(axes[0, 1], names, cbet_success, "CBet Success", "#55a868", "Success %")
    plot_bar(axes[0, 2], names, fold_to_cbet, "Fold To CBet", "#c44e52", "Fold %")
    plot_bar(axes[1, 0], names, bluff_rate, "Bluff Rate", "#8172b3", "Bluff %")
    plot_bar(axes[1, 1], names, bluff_caught, "Bluff Caught", "#937860", "Caught %")
    plot_bar(axes[1, 2], names, bluff_got_through, "Bluff Got Through", "#da8bc3", "Got Through %")

    plt.tight_layout()
    out_path = os.path.join(out_dir, "policy_compare_tendencies.png")
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def save_action_distribution_plot(action_counts: Dict[str, Dict[int, np.ndarray]], out_dir: str) -> str:
    names = list(action_counts.keys())
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    for ax, street in zip(axes.flat, STREET_ORDER):
        data = np.array([action_counts[name][street] for name in names], dtype=np.float64)
        totals = data.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = np.where(totals > 0, data / totals, 0.0)

        bottom = np.zeros(len(names))
        for action_id in range(NUM_ACTIONS):
            ax.bar(
                names,
                ratios[:, action_id],
                bottom=bottom,
                label=ACTION_LABELS[action_id],
                color=ACTION_COLORS[action_id],
            )
            bottom += ratios[:, action_id]

        ax.set_title(f"Action Mix - {STREET_NAMES[street]}")
        ax.set_ylim(0, 1.0)
        ax.tick_params(axis="x", rotation=20)
        for label in ax.get_xticklabels():
            label.set_ha("right")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    out_path = os.path.join(out_dir, "policy_compare_action_mix.png")
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def save_strength_plot(strength_by_action: Dict[str, Dict[int, Dict[str, float]]], out_dir: str) -> str:
    names = list(strength_by_action.keys())
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    for ax, street in zip(axes.flat, STREET_ORDER):
        aggr_vals = []
        pass_vals = []
        for name in names:
            stats = strength_by_action[name][street]
            aggr_avg = stats["aggr_sum"] / max(stats["aggr_count"], 1)
            pass_avg = stats["passive_sum"] / max(stats["passive_count"], 1)
            aggr_vals.append(aggr_avg)
            pass_vals.append(pass_avg)

        x = np.arange(len(names))
        width = 0.35
        ax.bar(x - width / 2, aggr_vals, width, label="Aggressive", color="#e78ac3")
        ax.bar(x + width / 2, pass_vals, width, label="Passive", color="#8da0cb")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right")
        ax.set_ylim(0, 1.0)
        ax.set_title(f"Avg Strength - {STREET_NAMES[street]}")
        ax.set_ylabel("Strength (0-1)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    out_path = os.path.join(out_dir, "policy_compare_strength.png")
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def save_range_heatmaps(range_counts: Dict[str, Dict[str, np.ndarray]], out_dir: str) -> List[str]:
    paths = []
    ranges_dir = os.path.join(out_dir, "ranges")
    os.makedirs(ranges_dir, exist_ok=True)
    for name, matrices in range_counts.items():
        totals = matrices["total"]
        vpip = matrices["vpip"]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(totals > 0, vpip / totals, 0.0)

        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(ratio, cmap="YlGn", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(13))
        ax.set_yticks(range(13))
        ax.set_xticklabels(RANKS)
        ax.set_yticklabels(RANKS)
        ax.set_title(f"VPIP Range (empirical) - {name}")
        for r in range(13):
            for c in range(13):
                ax.text(c, r, f"{ratio[r, c]:.2f}", ha="center", va="center", fontsize=6, color="black")
        fig.colorbar(im, ax=ax, shrink=0.8, label="VPIP Probability")
        plt.tight_layout()

        safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)
        out_path = os.path.join(ranges_dir, f"range_{safe}.png")
        plt.savefig(out_path, dpi=250)
        plt.close(fig)
        paths.append(out_path)
    return paths


def print_summary(stats: Dict[str, PolicyStats]) -> None:
    for name, stat in stats.items():
        s = stat.summary()
        af_display = f"{s['af']:.2f}" if s["af"] != float("inf") else "inf"
        print(
            f"{name}: hands={s['hands']}, win%={s['win_pct']:.1f}, bb/100={s['bb_per_100']:.2f}, "
            f"vpip%={s['vpip_pct']:.1f}, pfr%={s['pfr_pct']:.1f}, af={af_display}, "
            f"cbet%={s['cbet_pct']:.1f} (succ {s['cbet_success_pct']:.1f}), "
            f"fold_to_cbet%={s['fold_to_cbet_pct']:.1f}, "
            f"bluff%={s['bluff_rate_pct']:.1f} (caught {s['bluff_caught_pct']:.1f}), "
            f"wtsd%={s['showdown_pct']:.1f}, wsd%={s['showdown_win_pct']:.1f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run policy vs policy comparison matches")
    parser.add_argument("--hands", type=int, default=1000, help="Number of hands to play")
    parser.add_argument("--models", nargs="*", default=None, help="Policy .pt paths; defaults to models/policy*.pt")
    parser.add_argument("--out-dir", type=str, default="policy_compare_results", help="Output directory for plots/json")
    parser.add_argument("--num-players", type=int, default=NUM_PLAYERS, help="Number of seats in the match")
    parser.add_argument("--seed", type=int, default=DETERMINISTIC_SEED, help="Random seed")
    args = parser.parse_args()

    if args.models:
        model_paths = args.models
    else:
        model_paths = find_policy_paths("models")
        if not model_paths:
            raise FileNotFoundError("No policy*.pt files found in models/")

    seed_all(args.seed)
    env = SimpleHoldemEnv(num_players=args.num_players)
    dummy = env.new_hand()
    state_dim = encode_state(dummy, 0).numel()

    policies = load_policies(state_dim, model_paths)
    if len(policies) < 2:
        raise ValueError("Need at least two policies to compare.")

    os.makedirs(args.out_dir, exist_ok=True)

    results = run_matches(
        policies=policies,
        num_hands=args.hands,
        num_players=args.num_players,
        seed=args.seed,
    )

    stats = results["stats"]
    print_summary(stats)

    summary_plot = save_summary_plot(stats, args.out_dir)
    tendencies_plot = save_tendencies_plot(stats, args.out_dir)
    action_mix_plot = save_action_distribution_plot(results["action_counts"], args.out_dir)
    strength_plot = save_strength_plot(results["strength_by_action"], args.out_dir)
    range_paths = save_range_heatmaps(results["range_counts"], args.out_dir)

    payload = {
        "hands": args.hands,
        "num_players": args.num_players,
        "seed": args.seed,
        "models": [p.path for p in policies],
        "seat_to_policy": results["seat_to_policy"],
        "bluff_strength_threshold": BLUFF_STRENGTH_THRESHOLD,
        "plots": {
            "summary": summary_plot,
            "tendencies": tendencies_plot,
            "action_mix": action_mix_plot,
            "strength": strength_plot,
            "ranges": range_paths,
        },
        "stats": {name: stat.summary() for name, stat in stats.items()},
    }
    with open(os.path.join(args.out_dir, "policy_compare_results.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
