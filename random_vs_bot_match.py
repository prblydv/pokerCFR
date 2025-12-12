# ---------------------------------------------------------------------------
# File overview:
#   random_vs_bot_match.py plays a configurable number of heads-up hands
#   between a uniform-random policy and the trained Deep CFR policy. The script
#   reports per-player win rate, VPIP, PFR, aggression factor, and total
#   actions, printing interim summaries every MATCH_REPORT_INTERVAL hands.
# ---------------------------------------------------------------------------

import random
from dataclasses import dataclass, field
from typing import Dict, List

import torch
import numpy as np

from config import DEVICE, DETERMINISTIC_SEED
from poker_env import (
    SimpleHoldemEnv,
    ACTION_FOLD,
    ACTION_CHECK,
    ACTION_CALL,
    ACTION_RAISE_SMALL,
    ACTION_RAISE_MEDIUM,
    ACTION_ALL_IN,
    NUM_ACTIONS,
    STREET_PREFLOP,
)
from abstraction import encode_state
from networks import PolicyNet

RAISE_ACTIONS = {ACTION_RAISE_SMALL, ACTION_RAISE_MEDIUM, ACTION_ALL_IN}
MATCH_REPORT_INTERVAL = 50  # default report cadence for this script

random.seed(DETERMINISTIC_SEED)
np.random.seed(DETERMINISTIC_SEED)


@dataclass
class PlayerStats:
    name: str
    wins: int = 0
    hands: int = 0
    vpip_count: int = 0
    pfr_count: int = 0
    aggression_acts: int = 0
    call_acts: int = 0
    total_actions: int = 0

    def record_hand(
        self,
        won: bool,
        vpip: bool,
        pfr: bool,
        aggr_acts: int,
        call_acts: int,
        action_count: int,
    ):
        self.hands += 1
        if won:
            self.wins += 1
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
        af = (
            self.aggression_acts / self.call_acts
            if self.call_acts > 0
            else float("inf") if self.aggression_acts > 0 else 0.0
        )
        avg_actions = self.total_actions / hands
        return {
            "hands": self.hands,
            "win%": win_pct,
            "VPIP%": vpip_pct,
            "PFR%": pfr_pct,
            "AF": af,
            "Avg actions": avg_actions,
        }


def format_summary(stats: PlayerStats) -> str:
    data = stats.summary()
    af_display = (
        f"{data['AF']:.2f}" if data["AF"] != float("inf") else "inf"
    )
    return (
        f"{stats.name}: hands={data['hands']}, "
        f"win%={data['win%']:.1f}, VPIP%={data['VPIP%']:.1f}, "
        f"PFR%={data['PFR%']:.1f}, AF={af_display}, "
        f"avg_actions={data['Avg actions']:.2f}"
    )


def load_policy(env: SimpleHoldemEnv) -> PolicyNet:
    dummy = env.new_hand()
    state_dim = encode_state(dummy, 0).numel()
    policy = PolicyNet(state_dim)
    policy.load_state_dict(
        torch.load("models/policy.pt", map_location=DEVICE)
    )
    policy.to(DEVICE)
    policy.eval()
    return policy


def policy_action(policy: PolicyNet, state, player: int, legal_actions: List[int]) -> int:
    x = encode_state(state, player).to(DEVICE).unsqueeze(0)
    with torch.no_grad():
        logits = policy(x).squeeze(0)
    mask = torch.full((NUM_ACTIONS,), -1e9, device=logits.device)
    for a in legal_actions:
        mask[a] = 0.0
    probs = torch.softmax(logits + mask, dim=-1)
    action = torch.multinomial(probs, 1).item()
    if action not in legal_actions:
        action = random.choice(legal_actions)
    return action


def random_action(legal_actions: List[int]) -> int:
    return random.choice(legal_actions)


def play_match(num_hands: int = 500, report_interval: int = MATCH_REPORT_INTERVAL):
    env = SimpleHoldemEnv()
    policy = load_policy(env)

    random_stats = PlayerStats("Random")
    bot_stats = PlayerStats("Bot")

    for hand in range(1, num_hands + 1):
        state = env.new_hand()
        per_player_flags = {
            0: {"vpip": False, "pfr": False, "aggr": 0, "calls": 0, "actions": 0},
            1: {"vpip": False, "pfr": False, "aggr": 0, "calls": 0, "actions": 0},
        }

        while not state.terminal:
            player = state.to_act
            legal = env.legal_actions(state)
            if not legal:
                break

            if player == 0:
                action = random_action(legal)
            else:
                action = policy_action(policy, state, player, legal)

            info = per_player_flags[player]
            info["actions"] += 1

            if state.street == STREET_PREFLOP:
                if action in {ACTION_CALL} | RAISE_ACTIONS:
                    info["vpip"] = True
                if action in RAISE_ACTIONS:
                    info["pfr"] = True

            if action in RAISE_ACTIONS:
                info["aggr"] += 1
            elif action == ACTION_CALL:
                info["calls"] += 1

            state = env.step(state, action)

        winner = state.winner
        random_stats.record_hand(
            won=(winner == 0),
            vpip=per_player_flags[0]["vpip"],
            pfr=per_player_flags[0]["pfr"],
            aggr_acts=per_player_flags[0]["aggr"],
            call_acts=per_player_flags[0]["calls"],
            action_count=per_player_flags[0]["actions"],
        )
        bot_stats.record_hand(
            won=(winner == 1),
            vpip=per_player_flags[1]["vpip"],
            pfr=per_player_flags[1]["pfr"],
            aggr_acts=per_player_flags[1]["aggr"],
            call_acts=per_player_flags[1]["calls"],
            action_count=per_player_flags[1]["actions"],
        )

        if report_interval > 0 and hand % report_interval == 0:
            print(f"\n--- Interim report after {hand} hands ---")
            print(format_summary(random_stats))
            print(format_summary(bot_stats))

    print("\n=== Final summary ===")
    print(format_summary(random_stats))
    print(format_summary(bot_stats))


if __name__ == "__main__":
    play_match()
