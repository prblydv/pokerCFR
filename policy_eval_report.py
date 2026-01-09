
"""
policy_eval_report.py
---------------------
Run a multi-layer evaluation stack for one policy against one or more opponents.
Produces a JSON report plus a PDF summary and basic plots.

Usage:
    python policy_eval_report.py --policy "models/policy phase3_310.pt" --hands 100000 --opponents "models/policy.pt"
    python policy_eval_report.py --policy "policy phase3_310" --hands 50000 --opponents "policy phase1" "policy.pt"
    python policy_eval_report.py --policy "policy phase3_310" --hands 50000 --pool NL10
    python policy_eval_report.py --policy "policy phase3_310" --hands 25000 --pool all
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from textwrap import wrap
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

import poker_env
from config import BIG_BLIND, DETERMINISTIC_SEED, DEVICE
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
from abstraction import encode_state, coarse_strength
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
STREET_ORDER = [STREET_PREFLOP, STREET_FLOP, STREET_TURN, STREET_RIVER]
STREET_NAMES = {
    STREET_PREFLOP: "Preflop",
    STREET_FLOP: "Flop",
    STREET_TURN: "Turn",
    STREET_RIVER: "River",
}
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
BLUFF_STRENGTH_THRESHOLD = 0

STAKE_POOL_DEFS = {
    "NL2": {
        "description": "Loose-passive, under-bluffing",
        "specs": ["script:loose_passive", "script:loose_passive", "script:loose_passive"],
    },
    "NL5": {
        "description": "Loose-aggressive, weak river",
        "specs": ["script:loose_aggro_weak_river", "script:loose_aggro_weak_river", "script:overcall"],
    },
    "NL10": {
        "description": "Reg-ish, weak turn defense",
        "specs": ["script:reg_weak_turn", "models/policy.pt", "script:reg_weak_turn"],
    },
    "NL25": {
        "description": "Semi-solverish, low bluff freq",
        "specs": ["script:semi_solver_low_bluff", "models/policy phase3_120.pt", "script:semi_solver_low_bluff"],
    },
    "NL50+": {
        "description": "Near-balanced, exploit-aware (best available model)",
        "specs": ["best_model", "best_model", "script:semi_solver_low_bluff"],
    },
}


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
    ) -> None:
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
        if self.call_acts > 0:
            af = self.aggression_acts / self.call_acts
        else:
            af = float("inf") if self.aggression_acts > 0 else 0.0
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

@dataclass
class PolicyHandle:
    name: str
    kind: str
    net: Optional[PolicyNet] = None
    chooser: Optional[Callable] = None
    path: Optional[str] = None

    def act(self, state, player: int, legal_actions: List[int]) -> int:
        if self.kind == "net":
            return choose_action(self.net, state, player, legal_actions)
        return int(self.chooser(state, player, legal_actions))


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    poker_env.RNG.seed(seed)


def resolve_policy_path(name_or_path: str) -> str:
    if os.path.isfile(name_or_path):
        return name_or_path
    if not name_or_path.endswith(".pt"):
        candidate = os.path.join("models", f"{name_or_path}.pt")
        if os.path.isfile(candidate):
            return candidate
    models_dir = Path("models")
    if not models_dir.exists():
        raise FileNotFoundError(f"models directory not found: {models_dir}")
    matches = [p for p in models_dir.glob("*.pt") if name_or_path in p.stem]
    if not matches:
        raise FileNotFoundError(f"Could not find model for '{name_or_path}'")
    matches.sort()
    return str(matches[0])


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


def _to_call(state, player: int) -> float:
    return max(0.0, state.current_bet - state.contrib[player])


def _first_legal(legal_actions: List[int], preferred: List[int]) -> int:
    for action in preferred:
        if action in legal_actions:
            return action
    return random.choice(legal_actions)


def scripted_overfold(state, player: int, legal_actions: List[int]) -> int:
    if _to_call(state, player) > 0 and ACTION_FOLD in legal_actions:
        return ACTION_FOLD
    if ACTION_CHECK in legal_actions:
        return ACTION_CHECK
    if ACTION_CALL in legal_actions:
        return ACTION_CALL
    return random.choice(legal_actions)


def scripted_overcall(state, player: int, legal_actions: List[int]) -> int:
    if _to_call(state, player) > 0 and ACTION_CALL in legal_actions:
        return ACTION_CALL
    if ACTION_CHECK in legal_actions:
        return ACTION_CHECK
    if ACTION_FOLD in legal_actions:
        return ACTION_FOLD
    return random.choice(legal_actions)


def scripted_min_raise(state, player: int, legal_actions: List[int]) -> int:
    if ACTION_BET_POT_25 in legal_actions:
        return ACTION_BET_POT_25
    if ACTION_BET_POT_50 in legal_actions:
        return ACTION_BET_POT_50
    if ACTION_CALL in legal_actions:
        return ACTION_CALL
    if ACTION_CHECK in legal_actions:
        return ACTION_CHECK
    if ACTION_FOLD in legal_actions:
        return ACTION_FOLD
    return random.choice(legal_actions)


def scripted_aggro(state, player: int, legal_actions: List[int]) -> int:
    if ACTION_BET_POT_100 in legal_actions:
        return ACTION_BET_POT_100
    if ACTION_BET_POT_50 in legal_actions:
        return ACTION_BET_POT_50
    if ACTION_BET_POT_25 in legal_actions:
        return ACTION_BET_POT_25
    if ACTION_CALL in legal_actions:
        return ACTION_CALL
    if ACTION_CHECK in legal_actions:
        return ACTION_CHECK
    return random.choice(legal_actions)


def scripted_limp_heavy(state, player: int, legal_actions: List[int]) -> int:
    if state.street == STREET_PREFLOP:
        if _to_call(state, player) <= BIG_BLIND and ACTION_CALL in legal_actions:
            return ACTION_CALL
        if ACTION_CHECK in legal_actions:
            return ACTION_CHECK
        if ACTION_FOLD in legal_actions:
            return ACTION_FOLD
    if ACTION_CALL in legal_actions and _to_call(state, player) > 0:
        return ACTION_CALL
    if ACTION_CHECK in legal_actions:
        return ACTION_CHECK
    if ACTION_FOLD in legal_actions:
        return ACTION_FOLD
    return random.choice(legal_actions)


def scripted_river_raise(state, player: int, legal_actions: List[int]) -> int:
    if state.street == STREET_RIVER and ACTION_BET_POT_100 in legal_actions:
        return ACTION_BET_POT_100
    if _to_call(state, player) > 0 and ACTION_CALL in legal_actions:
        return ACTION_CALL
    if ACTION_CHECK in legal_actions:
        return ACTION_CHECK
    if ACTION_FOLD in legal_actions:
        return ACTION_FOLD
    return random.choice(legal_actions)


def scripted_loose_passive(state, player: int, legal_actions: List[int]) -> int:
    strength = coarse_strength(state.hole[player], state.board)
    to_call = _to_call(state, player)
    if state.street == STREET_PREFLOP:
        if strength >= 2 and ACTION_BET_POT_50 in legal_actions and random.random() < 0.15:
            return ACTION_BET_POT_50
        if to_call > 0 and ACTION_CALL in legal_actions:
            return ACTION_CALL
        return _first_legal(legal_actions, [ACTION_CHECK, ACTION_CALL, ACTION_FOLD])
    if to_call > 0:
        if strength >= 1 and ACTION_CALL in legal_actions:
            return ACTION_CALL
        if ACTION_FOLD in legal_actions:
            return ACTION_FOLD
    if strength >= 2 and ACTION_BET_POT_50 in legal_actions and random.random() < 0.1:
        return ACTION_BET_POT_50
    return _first_legal(legal_actions, [ACTION_CHECK, ACTION_CALL, ACTION_FOLD])


def scripted_loose_aggro_weak_river(state, player: int, legal_actions: List[int]) -> int:
    strength = coarse_strength(state.hole[player], state.board)
    to_call = _to_call(state, player)
    if state.street == STREET_RIVER:
        if to_call > 0 and strength < 1 and ACTION_FOLD in legal_actions:
            return ACTION_FOLD
        if strength >= 2 and ACTION_BET_POT_100 in legal_actions:
            return ACTION_BET_POT_100
        return _first_legal(legal_actions, [ACTION_CALL, ACTION_CHECK, ACTION_FOLD])

    if ACTION_BET_POT_50 in legal_actions and random.random() < 0.45:
        return ACTION_BET_POT_50
    if to_call > 0 and ACTION_CALL in legal_actions:
        return ACTION_CALL
    return _first_legal(legal_actions, [ACTION_CHECK, ACTION_CALL, ACTION_FOLD])


def scripted_reg_weak_turn(state, player: int, legal_actions: List[int]) -> int:
    strength = coarse_strength(state.hole[player], state.board)
    to_call = _to_call(state, player)
    if state.street == STREET_PREFLOP:
        if strength >= 1 and ACTION_BET_POT_50 in legal_actions:
            return ACTION_BET_POT_50
        if to_call > 0 and ACTION_CALL in legal_actions:
            return ACTION_CALL
        return _first_legal(legal_actions, [ACTION_CHECK, ACTION_CALL, ACTION_FOLD])

    if state.street == STREET_TURN and to_call > 0 and strength < 1 and ACTION_FOLD in legal_actions:
        return ACTION_FOLD

    if strength >= 1 and ACTION_BET_POT_50 in legal_actions and random.random() < 0.25:
        return ACTION_BET_POT_50
    if to_call > 0 and ACTION_CALL in legal_actions:
        return ACTION_CALL
    return _first_legal(legal_actions, [ACTION_CHECK, ACTION_CALL, ACTION_FOLD])


def scripted_semi_solver_low_bluff(state, player: int, legal_actions: List[int]) -> int:
    strength = coarse_strength(state.hole[player], state.board)
    to_call = _to_call(state, player)
    if state.street == STREET_PREFLOP:
        if strength >= 1 and ACTION_BET_POT_50 in legal_actions:
            return ACTION_BET_POT_50
        if to_call > 0 and ACTION_CALL in legal_actions:
            return ACTION_CALL
        return _first_legal(legal_actions, [ACTION_CHECK, ACTION_CALL, ACTION_FOLD])

    if strength >= 2 and ACTION_BET_POT_100 in legal_actions:
        return ACTION_BET_POT_100
    if strength >= 1 and ACTION_BET_POT_50 in legal_actions:
        return ACTION_BET_POT_50
    if to_call > 0 and strength < 1 and ACTION_FOLD in legal_actions:
        return ACTION_FOLD
    if to_call > 0 and ACTION_CALL in legal_actions:
        return ACTION_CALL
    return _first_legal(legal_actions, [ACTION_CHECK, ACTION_CALL, ACTION_FOLD])


def build_scripted_policies() -> List[PolicyHandle]:
    return [
        PolicyHandle(name="overfold", kind="script", chooser=scripted_overfold),
        PolicyHandle(name="overcall", kind="script", chooser=scripted_overcall),
        PolicyHandle(name="minraise_spam", kind="script", chooser=scripted_min_raise),
        PolicyHandle(name="aggro_raise", kind="script", chooser=scripted_aggro),
        PolicyHandle(name="limp_heavy", kind="script", chooser=scripted_limp_heavy),
        PolicyHandle(name="river_raise", kind="script", chooser=scripted_river_raise),
    ]


SCRIPTED_POOL_MAP = {
    "overfold": scripted_overfold,
    "overcall": scripted_overcall,
    "minraise_spam": scripted_min_raise,
    "aggro_raise": scripted_aggro,
    "limp_heavy": scripted_limp_heavy,
    "river_raise": scripted_river_raise,
    "loose_passive": scripted_loose_passive,
    "loose_aggro_weak_river": scripted_loose_aggro_weak_river,
    "reg_weak_turn": scripted_reg_weak_turn,
    "semi_solver_low_bluff": scripted_semi_solver_low_bluff,
}


def best_model_path() -> Optional[str]:
    preferred = [
        os.path.join("models", "policy phase4_160.pt"),
        os.path.join("models", "policy phase3_310.pt"),
        os.path.join("models", "policy phase3_120.pt"),
        os.path.join("models", "policy phase1.pt"),
        os.path.join("models", "policy.pt"),
    ]
    for path in preferred:
        if os.path.isfile(path):
            return path
    candidates = sorted(str(p) for p in Path("models").glob("policy*.pt") if p.is_file())
    return candidates[-1] if candidates else None


def resolve_pool_key(name: str) -> str:
    raw = name.strip().upper().replace(" ", "")
    raw = raw.replace("PLUS", "+")
    for key in STAKE_POOL_DEFS:
        key_norm = key.upper().replace(" ", "")
        key_norm = key_norm.replace("PLUS", "+")
        if raw == key_norm:
            return key
    raise ValueError(f"Unknown pool '{name}'. Available: {', '.join(STAKE_POOL_DEFS.keys())}")


def build_pool_from_specs(specs: List[str], state_dim: int) -> List[PolicyHandle]:
    handles = []
    for spec in specs:
        if spec.startswith("script:"):
            key = spec.split(":", 1)[1]
            chooser = SCRIPTED_POOL_MAP.get(key)
            if chooser is None:
                raise ValueError(f"Unknown scripted opponent '{key}'")
            handles.append(PolicyHandle(name=key, kind="script", chooser=chooser))
            continue
        if spec == "best_model":
            path = best_model_path()
            if path is None:
                handles.append(
                    PolicyHandle(name="semi_solver_low_bluff", kind="script", chooser=scripted_semi_solver_low_bluff)
                )
            else:
                net = load_policy(state_dim, path)
                handles.append(PolicyHandle(name=Path(path).stem, kind="net", net=net, path=path))
            continue
        path = resolve_policy_path(spec)
        net = load_policy(state_dim, path)
        handles.append(PolicyHandle(name=Path(path).stem, kind="net", net=net, path=path))
    return handles


def expand_handles(handles: List[PolicyHandle], target: int) -> List[PolicyHandle]:
    if not handles:
        raise ValueError("Opponent pool is empty.")
    expanded = []
    idx = 0
    while len(expanded) < target:
        expanded.append(handles[idx % len(handles)])
        idx += 1
    return expanded


def next_live_actor(state, current: int) -> int:
    for i in range(1, state.num_players + 1):
        nxt = (current + i) % state.num_players
        if not state.folded[nxt] and state.stacks[nxt] > 0:
            return nxt
    return -1


def _rotate_seats(base: List[int], shift: int) -> List[int]:
    if not base:
        return base
    shift = shift % len(base)
    return base[-shift:] + base[:-shift]

def run_match(
    hero: PolicyHandle,
    opponents: List[PolicyHandle],
    num_hands: int,
    seed: int,
    num_players: int = 6,
    rotate_seats: bool = True,
    track_actions: bool = True,
) -> Dict[str, object]:
    seed_all(seed)
    env = SimpleHoldemEnv(num_players=num_players)
    if num_players != 6:
        raise ValueError("This evaluator expects 6 seats (3 hero, 3 opponents).")
    opponent_handles = expand_handles(opponents, 3)
    seat_handles_base = [hero, hero, hero] + opponent_handles

    stats = {h.name: PolicyStats(h.name) for h in seat_handles_base}
    action_counts = {hero.name: {street: np.zeros(NUM_ACTIONS, dtype=np.int64) for street in STREET_ORDER}}
    per_hand_bb = []

    for hand_idx in range(num_hands):
        seat_handles = seat_handles_base
        if rotate_seats:
            seat_handles = _rotate_seats(seat_handles_base, hand_idx)
        state = env.new_hand()
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

            policy = seat_handles[player]
            policy_name = policy.name
            to_call = _to_call(state, player)

            if (
                state.street == STREET_FLOP
                and player == last_preflop_aggressor
                and player not in cbet_opportunity_recorded
                and state.current_bet == 0
            ):
                stats[policy_name].cbet_opportunities += 1
                cbet_opportunity_recorded.add(player)

            action = policy.act(state, player, legal)
            info = per_hand_flags[player]
            info["actions"] += 1

            if track_actions and policy == hero:
                action_counts[hero.name][state.street][action] += 1

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
                strength = coarse_strength(state.hole[player], state.board)

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
                    strength = coarse_strength(state.hole[player], state.board)
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
                    cbet_policy = seat_handles[cbet_active["player"]].name
                    if not cbet_active["faced_resistance"]:
                        stats[cbet_policy].cbet_successes += 1
                    cbet_active = None
                for pid, idx in list(open_bluff_by_player.items()):
                    if bluff_events[idx]["street"] == prev_street and not bluff_events[idx]["called"]:
                        bluff_events[idx]["called"] = True
                        open_bluff_by_player.pop(pid, None)

            if state.terminal and cbet_active:
                cbet_policy = seat_handles[cbet_active["player"]].name
                if not cbet_active["faced_resistance"]:
                    stats[cbet_policy].cbet_successes += 1
                cbet_active = None

        hero_profit = 0.0
        hero_seats = 0
        for pid in range(num_players):
            policy_name = seat_handles[pid].name
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
            if seat_handles[pid] is hero:
                hero_profit += profit
                hero_seats += 1

        live_players = [i for i in range(num_players) if not state.folded[i]]
        if len(live_players) >= 2:
            for pid in live_players:
                policy_name = seat_handles[pid].name
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

        hero_seats = max(hero_seats, 1)
        per_hand_bb.append((hero_profit / hero_seats) / max(1e-9, BIG_BLIND))

    return {
        "stats": stats,
        "per_hand_bb": per_hand_bb,
        "action_counts": action_counts,
        "seat_handles_base": [h.name for h in seat_handles_base],
    }

def normal_ci(samples: Sequence[float], confidence: float = 0.95) -> Tuple[float, float]:
    if len(samples) < 2:
        return (0.0, 0.0)
    mean = float(np.mean(samples))
    std = float(np.std(samples, ddof=1))
    z = 1.96 if confidence == 0.95 else 1.0
    se = std / math.sqrt(len(samples))
    return mean - z * se, mean + z * se


def bootstrap_mean_ci(
    samples: Sequence[float],
    num_boot: int,
    seed: int,
    confidence: float = 0.95,
) -> Dict[str, float]:
    if not samples:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "prob_pos": 0.0}
    rng = np.random.default_rng(seed)
    means = []
    n = len(samples)
    for _ in range(num_boot):
        resample = rng.choice(samples, size=n, replace=True)
        means.append(float(np.mean(resample)))
    means = np.array(means, dtype=np.float64)
    low_pct = 100.0 * (1.0 - confidence) / 2.0
    high_pct = 100.0 - low_pct
    ci_low, ci_high = np.percentile(means, [low_pct, high_pct])
    prob_pos = float(np.mean(means > 0.0))
    return {"mean": float(np.mean(means)), "ci_low": float(ci_low), "ci_high": float(ci_high), "prob_pos": prob_pos}


def action_distribution(action_counts: Dict[int, np.ndarray]) -> np.ndarray:
    total = np.zeros(NUM_ACTIONS, dtype=np.float64)
    for street in STREET_ORDER:
        total += action_counts[street]
    denom = total.sum()
    if denom <= 0:
        return np.ones(NUM_ACTIONS, dtype=np.float64) / NUM_ACTIONS
    return total / denom


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-9
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def build_bar_plot(names: List[str], values: List[float], title: str, y_label: str, out_path: str, ci: Optional[List[float]] = None) -> str:
    fig, ax = plt.subplots(figsize=(10, 5))
    if ci:
        ax.bar(names, values, yerr=ci, color="#4c72b0", capsize=5)
    else:
        ax.bar(names, values, color="#4c72b0")
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.tick_params(axis="x", rotation=20)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.axhline(0.0, color="#666666", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def render_pdf_report(out_path: str, title: str, sections: List[Tuple[str, List[str]]]) -> str:
    styles = {
        "title": {"size": 24, "weight": "bold", "color": "#0b3d91", "family": "DejaVu Serif", "wrap": 60},
        "h1": {"size": 14, "weight": "bold", "color": "#0b3d91", "family": "DejaVu Serif", "wrap": 85},
        "body": {"size": 10.5, "weight": "normal", "color": "#111111", "family": "DejaVu Serif", "wrap": 95},
        "code": {"size": 9.5, "weight": "normal", "color": "#111111", "family": "DejaVu Sans Mono", "wrap": 120},
    }
    line_height = {"title": 0.035, "h1": 0.024, "body": 0.018, "code": 0.017}

    def new_page(page_num: int, header: bool = True):
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        if header:
            ax.add_patch(Rectangle((0, 0.94), 1, 0.06, transform=fig.transFigure, color="#0b3d91", zorder=0))
            fig.text(0.05, 0.965, "policy evaluation report", fontsize=11, color="white", fontweight="bold")
            fig.text(0.95, 0.965, f"Page {page_num}", fontsize=9.5, color="white", ha="right")
        return fig

    def draw_lines(fig, lines: List[Tuple[str, str]], start_y: float, page_num: int, pdf: PdfPages):
        y = start_y
        for style, text in lines:
            lh = line_height[style]
            wrapped = []
            if style == "code":
                wrapped = text.split("\n")
            else:
                for para in text.split("\n"):
                    if para.strip() == "":
                        wrapped.append("")
                    else:
                        wrapped.extend(wrap(para, width=styles[style]["wrap"]))
            for line in wrapped:
                if y < 0.06:
                    pdf.savefig(fig)
                    plt.close(fig)
                    page_num += 1
                    fig = new_page(page_num, header=True)
                    y = 0.92
                fig.text(0.06, y, line, fontsize=styles[style]["size"], fontweight=styles[style]["weight"], color=styles[style]["color"], fontfamily=styles[style]["family"])
                y -= lh
            y -= lh * 0.35
        return fig, page_num

    with PdfPages(out_path) as pdf:
        page_num = 1
        fig = new_page(page_num, header=False)
        y = 0.82
        title_lines = [("title", title)]
        fig, page_num = draw_lines(fig, title_lines, y, page_num, pdf)
        pdf.savefig(fig)
        plt.close(fig)

        for header, body_lines in sections:
            page_num += 1
            fig = new_page(page_num, header=True)
            lines = [("h1", header)] + [("body", line) for line in body_lines]
            fig, page_num = draw_lines(fig, lines, 0.92, page_num, pdf)
            pdf.savefig(fig)
            plt.close(fig)

    return out_path

def main() -> int:
    parser = argparse.ArgumentParser(description="Run evaluation stack for one policy vs opponent pools.")
    parser.add_argument("--policy", required=True, help="Hero policy path or name")
    parser.add_argument("--hands", type=int, default=10000, help="Hands per opponent match")
    parser.add_argument("--opponents", nargs="*", default=None, help="Opponent policy paths or names")
    parser.add_argument("--pool", "--stake", dest="pool", default=None, help="Stake pool name (NL2, NL5, NL10, NL25, NL50+) or 'all'")
    parser.add_argument("--out-dir", type=str, default="policy_eval_results", help="Output directory")
    parser.add_argument("--seed", type=int, default=DETERMINISTIC_SEED, help="Random seed")
    parser.add_argument("--num-players", type=int, default=6, help="Number of seats (default 6)")
    parser.add_argument("--bootstrap", type=int, default=5000, help="Bootstrap resamples")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    hero_path = resolve_policy_path(args.policy)

    pool_requests = []
    if args.pool:
        if args.pool.strip().lower() == "all":
            for key in STAKE_POOL_DEFS:
                pool_requests.append(
                    {"name": key, "description": STAKE_POOL_DEFS[key]["description"], "specs": STAKE_POOL_DEFS[key]["specs"]}
                )
        else:
            key = resolve_pool_key(args.pool)
            pool_requests.append(
                {"name": key, "description": STAKE_POOL_DEFS[key]["description"], "specs": STAKE_POOL_DEFS[key]["specs"]}
            )
    elif args.opponents:
        pool_requests.append({"name": "custom", "description": "Custom opponent list", "specs": args.opponents})
    else:
        default_opponent = os.path.join("models", "policy.pt")
        if os.path.isfile(default_opponent) and os.path.abspath(default_opponent) != os.path.abspath(hero_path):
            specs = [default_opponent]
        else:
            candidates = [p for p in Path("models").glob("policy*.pt") if os.path.abspath(str(p)) != os.path.abspath(hero_path)]
            if not candidates:
                raise FileNotFoundError("No opponent policies found in models/")
            specs = [str(sorted(candidates)[0])]
        pool_requests.append({"name": "default", "description": "Default opponent policy", "specs": specs})

    seed_all(args.seed)
    env = SimpleHoldemEnv(num_players=args.num_players)
    dummy = env.new_hand()
    state_dim = encode_state(dummy, 0).numel()

    hero_net = load_policy(state_dim, hero_path)
    hero_name = Path(hero_path).stem
    hero = PolicyHandle(name=hero_name, kind="net", net=hero_net, path=hero_path)

    pools_to_run = []
    for pool in pool_requests:
        opponents = build_pool_from_specs(pool["specs"], state_dim)
        opponents = expand_handles(opponents, 3)
        if any(opp.name == hero.name for opp in opponents):
            for idx, opp in enumerate(opponents):
                if opp.name == hero.name:
                    opp.name = f"{opp.name}_opp{idx + 1}"
        pools_to_run.append(
            {
                "name": pool["name"],
                "description": pool["description"],
                "specs": pool["specs"],
                "opponents": opponents,
            }
        )

    results = []
    for idx, pool in enumerate(pools_to_run):
        match = run_match(
            hero=hero,
            opponents=pool["opponents"],
            num_hands=args.hands,
            seed=args.seed + idx,
            num_players=args.num_players,
            rotate_seats=True,
            track_actions=True,
        )
        per_hand_bb = match["per_hand_bb"]
        mean_bb = float(np.mean(per_hand_bb)) if per_hand_bb else 0.0
        mean_bb100 = mean_bb * 100.0
        ci_low, ci_high = normal_ci(per_hand_bb)
        bootstrap = bootstrap_mean_ci(per_hand_bb, args.bootstrap, args.seed + idx + 77)
        results.append(
            {
                "pool": pool["name"],
                "pool_description": pool["description"],
                "pool_specs": pool["specs"],
                "opponent_names": [h.name for h in pool["opponents"]],
                "hero": hero.name,
                "hands": args.hands,
                "mean_bb100": mean_bb100,
                "normal_ci_bb100": (ci_low * 100.0, ci_high * 100.0),
                "bootstrap_mean_bb100": bootstrap["mean"] * 100.0,
                "bootstrap_ci_bb100": (bootstrap["ci_low"] * 100.0, bootstrap["ci_high"] * 100.0),
                "bootstrap_prob_pos": bootstrap["prob_pos"],
                "stats": {name: stat.summary() for name, stat in match["stats"].items()},
                "action_counts": {
                    hero.name: {street: match["action_counts"][hero.name][street].tolist() for street in STREET_ORDER}
                },
                "seat_handles_base": match["seat_handles_base"],
            }
        )

    self_play = run_match(
        hero=hero,
        opponents=[hero, hero, hero],
        num_hands=args.hands,
        seed=args.seed + 999,
        num_players=args.num_players,
        rotate_seats=True,
        track_actions=False,
    )
    self_play_bb100 = float(np.mean(self_play["per_hand_bb"])) * 100.0 if self_play["per_hand_bb"] else 0.0

    stress_policies = build_scripted_policies()
    stress_results = []
    worst_bb100 = None
    for i, attacker in enumerate(stress_policies):
        match = run_match(
            hero=hero,
            opponents=[attacker, attacker, attacker],
            num_hands=args.hands,
            seed=args.seed + 2000 + i,
            num_players=args.num_players,
            rotate_seats=True,
            track_actions=False,
        )
        mean_bb100 = float(np.mean(match["per_hand_bb"])) * 100.0 if match["per_hand_bb"] else 0.0
        stress_results.append({"opponent": attacker.name, "hero_bb100": mean_bb100})
        if worst_bb100 is None or mean_bb100 < worst_bb100:
            worst_bb100 = mean_bb100

    approx_exploitability = None
    if stress_results:
        attacker_ev = max(-r["hero_bb100"] for r in stress_results)
        approx_exploitability = attacker_ev - self_play_bb100

    stable_pool = pools_to_run[0]
    stable_run_a = run_match(
        hero=hero,
        opponents=stable_pool["opponents"],
        num_hands=args.hands,
        seed=args.seed + 4242,
        num_players=args.num_players,
        rotate_seats=True,
        track_actions=True,
    )
    stable_run_b = run_match(
        hero=hero,
        opponents=stable_pool["opponents"],
        num_hands=args.hands,
        seed=args.seed + 4242,
        num_players=args.num_players,
        rotate_seats=True,
        track_actions=True,
    )
    dist_a = action_distribution(stable_run_a["action_counts"][hero.name])
    dist_b = action_distribution(stable_run_b["action_counts"][hero.name])
    stability = {
        "kl_divergence": kl_divergence(dist_a, dist_b),
        "mean_bb100_a": float(np.mean(stable_run_a["per_hand_bb"])) * 100.0 if stable_run_a["per_hand_bb"] else 0.0,
        "mean_bb100_b": float(np.mean(stable_run_b["per_hand_bb"])) * 100.0 if stable_run_b["per_hand_bb"] else 0.0,
        "action_dist_a": dist_a.tolist(),
        "action_dist_b": dist_b.tolist(),
    }

    head_names = [r["pool"] for r in results]
    head_bb100 = [r["mean_bb100"] for r in results]
    head_ci = [abs(r["normal_ci_bb100"][1] - r["mean_bb100"]) for r in results]
    summary_plot = build_bar_plot(
        head_names,
        head_bb100,
        "Head-to-Head BB/100 (hero per seat)",
        "BB/100",
        os.path.join(args.out_dir, "head_to_head_bb100.png"),
        ci=head_ci,
    )
    stress_plot = build_bar_plot(
        [r["opponent"] for r in stress_results],
        [r["hero_bb100"] for r in stress_results],
        "Stress Tests (hero BB/100)",
        "BB/100",
        os.path.join(args.out_dir, "stress_test_bb100.png"),
    )

    report_payload = {
        "hero_policy": hero.path,
        "pools": [
            {
                "name": pool["name"],
                "description": pool["description"],
                "specs": pool["specs"],
                "opponent_names": [h.name for h in pool["opponents"]],
            }
            for pool in pools_to_run
        ],
        "hands": args.hands,
        "num_players": args.num_players,
        "seed": args.seed,
        "bootstrap_samples": args.bootstrap,
        "head_to_head": results,
        "self_play_bb100": self_play_bb100,
        "stress_tests": stress_results,
        "approx_exploitability_bb100": approx_exploitability,
        "stability": stability,
        "plots": {
            "head_to_head": summary_plot,
            "stress_tests": stress_plot,
        },
    }

    json_path = os.path.join(args.out_dir, "policy_eval_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    sections = []
    pool_lines = []
    for pool in pools_to_run:
        spec_str = ", ".join(pool["specs"])
        pool_lines.append(f"{pool['name']}: {pool['description']} | specs: {spec_str}")

    sections.append(
        (
            "Run configuration",
            [
                f"Hero policy: {hero.path}",
                f"Pools: {'; '.join(pool_lines)}",
                f"Hands per match: {args.hands}",
                f"Seats: {args.num_players} (3 hero, 3 opponent, rotated per hand)",
                f"Seed: {args.seed}",
                f"Bootstrap samples: {args.bootstrap}",
            ],
        )
    )

    head_lines = []
    for r in results:
        head_lines.append(
            f"Pool: {r['pool']} | mean BB/100: {r['mean_bb100']:.2f} | "
            f"normal CI: [{r['normal_ci_bb100'][0]:.2f}, {r['normal_ci_bb100'][1]:.2f}] | "
            f"bootstrap CI: [{r['bootstrap_ci_bb100'][0]:.2f}, {r['bootstrap_ci_bb100'][1]:.2f}] | "
            f"P(bb/100>0): {r['bootstrap_prob_pos']:.3f}"
        )
    sections.append(("Layer 1+2: head-to-head EV and bootstrap", head_lines))

    stats_lines = []
    for r in results:
        opp_list = ", ".join(r["opponent_names"])
        stats_lines.append(f"Pool: {r['pool']} | opponents: {opp_list}")
        for name, stat in r["stats"].items():
            af_display = f"{stat['af']:.2f}" if stat["af"] != float("inf") else "inf"
            stats_lines.append(
                f"  {name}: win%={stat['win_pct']:.1f}, bb/100={stat['bb_per_100']:.2f}, "
                f"vpip={stat['vpip_pct']:.1f}, pfr={stat['pfr_pct']:.1f}, af={af_display}, "
                f"cbet={stat['cbet_pct']:.1f} (succ {stat['cbet_success_pct']:.1f}), "
                f"bluff={stat['bluff_rate_pct']:.1f} (caught {stat['bluff_caught_pct']:.1f}), "
                f"wtsd={stat['showdown_pct']:.1f}, wsd={stat['showdown_win_pct']:.1f}"
            )
    sections.append(("Per-policy stat summary", stats_lines))

    exploit_lines = [
        f"Self-play BB/100: {self_play_bb100:.2f}",
        f"Approx exploitability (restricted BR): {approx_exploitability:.2f} BB/100"
        if approx_exploitability is not None
        else "Approx exploitability: n/a",
    ]
    sections.append(("Layer 3: exploitability (approx)", exploit_lines))

    stress_lines = [f"{r['opponent']}: hero BB/100 {r['hero_bb100']:.2f}" for r in stress_results]
    worst_line = f"Worst case hero BB/100: {worst_bb100:.2f}" if worst_bb100 is not None else "Worst case: n/a"
    stress_lines.append(worst_line)
    sections.append(("Layer 4: stress tests", stress_lines))

    stability_lines = [
        f"KL divergence (action mix): {stability['kl_divergence']:.6f}",
        f"Mean BB/100 run A: {stability['mean_bb100_a']:.2f}",
        f"Mean BB/100 run B: {stability['mean_bb100_b']:.2f}",
    ]
    sections.append(("Layer 5: stability and determinism", stability_lines))

    files_lines = [
        f"JSON report: {json_path}",
        f"Head-to-head plot: {summary_plot}",
        f"Stress plot: {stress_plot}",
    ]
    sections.append(("Artifacts", files_lines))

    pdf_path = os.path.join(args.out_dir, "policy_eval_report.pdf")
    render_pdf_report(
        pdf_path,
        title=f"Policy evaluation report - {hero.name}",
        sections=sections,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
