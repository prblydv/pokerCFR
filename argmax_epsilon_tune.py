
"""
argmax_epsilon_tune.py
----------------------
Search for the argmax-epsilon sweet spot that maximizes EV.
Runs 6-max tournaments with 3 hero seats vs 3 opponent seats and
plots EV (BB/100) vs epsilon.

Usage:
    python argmax_epsilon_tune.py --policy "policy phase3_310" --hands 50000 --pool NL10
    python argmax_epsilon_tune.py --policy "models/policy phase3_310.pt" --hands 20000 --opponents "policy.pt"
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import BIG_BLIND, DETERMINISTIC_SEED
from poker_env import SimpleHoldemEnv, NUM_ACTIONS
from abstraction import encode_state

from policy_eval_report import (
    STAKE_POOL_DEFS,
    build_pool_from_specs,
    expand_handles,
    resolve_pool_key,
    resolve_policy_path,
    load_policy,
    seed_all,
)


@dataclass
class HeroHandle:
    name: str
    net: torch.nn.Module
    epsilon: float = 0.0

    def act(self, state, player: int, legal_actions: List[int]) -> int:
        return choose_action_argmax_epsilon(self.net, state, player, legal_actions, self.epsilon)


def masked_policy_probs(policy_net: torch.nn.Module, state, player: int, legal_actions: List[int]) -> torch.Tensor:
    device = next(policy_net.parameters()).device
    x = encode_state(state, player).to(device).unsqueeze(0)
    with torch.no_grad():
        logits = policy_net(x).squeeze(0)
    mask = torch.full((NUM_ACTIONS,), -1e9, device=logits.device)
    for a in legal_actions:
        mask[a] = 0.0
    return torch.softmax(logits + mask, dim=-1)


def choose_action_argmax_epsilon(
    policy_net: torch.nn.Module,
    state,
    player: int,
    legal_actions: List[int],
    epsilon: float,
) -> int:
    probs = masked_policy_probs(policy_net, state, player, legal_actions)
    eps = min(max(epsilon, 0.0), 1.0)
    if eps > 0.0 and random.random() < eps:
        action = torch.multinomial(probs, 1).item()
    else:
        action = torch.argmax(probs, dim=-1).item()
    if action not in legal_actions:
        action = random.choice(legal_actions)
    return action


def next_live_actor(state, current: int) -> int:
    for i in range(1, state.num_players + 1):
        nxt = (current + i) % state.num_players
        if not state.folded[nxt] and state.stacks[nxt] > 0:
            return nxt
    return -1


def rotate_seats(base: List[object], shift: int) -> List[object]:
    if not base:
        return base
    shift = shift % len(base)
    return base[-shift:] + base[:-shift]


def run_match_epsilon(
    hero: HeroHandle,
    opponents: List[object],
    num_hands: int,
    seed: int,
    num_players: int = 6,
) -> List[float]:
    if num_players != 6:
        raise ValueError("This evaluator expects 6 seats (3 hero, 3 opponents).")
    seed_all(seed)
    env = SimpleHoldemEnv(num_players=num_players)
    opponent_handles = expand_handles(opponents, 3)
    seat_handles_base = [hero, hero, hero] + opponent_handles
    per_hand_bb = []

    for hand_idx in range(num_hands):
        seat_handles = rotate_seats(seat_handles_base, hand_idx)
        state = env.new_hand()

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

            handle = seat_handles[player]
            action = handle.act(state, player, legal)
            state = env.step(state, action)

        hero_profit = 0.0
        hero_seats = 0
        for pid in range(num_players):
            if seat_handles[pid] is hero:
                hero_profit += state.stacks[pid] - state.initial_stacks[pid]
                hero_seats += 1

        hero_seats = max(hero_seats, 1)
        per_hand_bb.append((hero_profit / hero_seats) / max(1e-9, BIG_BLIND))

    return per_hand_bb


def normal_ci(samples: Sequence[float], confidence: float = 0.95) -> Tuple[float, float]:
    if len(samples) < 2:
        return (0.0, 0.0)
    mean = float(np.mean(samples))
    std = float(np.std(samples, ddof=1))
    z = 1.96 if confidence == 0.95 else 1.0
    se = std / math.sqrt(len(samples))
    return mean - z * se, mean + z * se


def parse_eps_values(args) -> List[float]:
    if args.eps_list:
        raw = args.eps_list.replace(",", " ")
        values = [float(v) for v in raw.split() if v.strip()]
    else:
        if args.eps_step <= 0:
            raise ValueError("eps-step must be > 0")
        values = []
        cur = args.eps_min
        while cur <= args.eps_max + 1e-9:
            values.append(round(cur, 6))
            cur += args.eps_step
    unique = sorted(set(values))
    return unique


def resolve_default_opponents(hero_path: str) -> List[str]:
    default_opponent = os.path.join("models", "policy.pt")
    if os.path.isfile(default_opponent) and os.path.abspath(default_opponent) != os.path.abspath(hero_path):
        return [default_opponent]
    candidates = [p for p in Path("models").glob("policy*.pt") if os.path.abspath(str(p)) != os.path.abspath(hero_path)]
    if not candidates:
        raise FileNotFoundError("No opponent policies found in models/")
    return [str(sorted(candidates)[0])]


def plot_epsilon_curve(results: List[Dict[str, float]], out_path: str, title: str) -> str:
    eps = [r["epsilon"] for r in results]
    mean = [r["mean_bb100"] for r in results]
    err = [abs(r["ci_high_bb100"] - r["mean_bb100"]) for r in results]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(eps, mean, yerr=err, fmt="-o", color="#2b7bba", ecolor="#8bbbdc", capsize=3)
    ax.axhline(0.0, color="#666666", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Argmax epsilon")
    ax.set_ylabel("EV (BB/100)")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Tune argmax epsilon for max EV")
    parser.add_argument("--policy", required=True, help="Hero policy path or name")
    parser.add_argument("--hands", type=int, default=10000, help="Hands per epsilon")
    parser.add_argument("--opponents", nargs="*", default=None, help="Opponent policy paths or names")
    parser.add_argument("--pool", default=None, help="Stake pool name (NL2, NL5, NL10, NL25, NL50+)")
    parser.add_argument("--eps-min", type=float, default=0.0, help="Minimum epsilon")
    parser.add_argument("--eps-max", type=float, default=1.0, help="Maximum epsilon")
    parser.add_argument("--eps-step", type=float, default=0.005, help="Epsilon step")
    parser.add_argument("--eps-list", type=str, default=None, help="Explicit epsilon list (comma or space separated)")
    parser.add_argument("--seed", type=int, default=DETERMINISTIC_SEED, help="Random seed")
    parser.add_argument("--out-dir", type=str, default="policy_eval_results", help="Output directory")
    parser.add_argument("--num-players", type=int, default=6, help="Number of seats (default 6)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    hero_path = resolve_policy_path(args.policy)
    eps_values = parse_eps_values(args)

    if args.pool:
        pool_key = resolve_pool_key(args.pool)
        pool_def = STAKE_POOL_DEFS[pool_key]
        pool_specs = pool_def["specs"]
        pool_name = pool_key
        pool_desc = pool_def["description"]
    elif args.opponents:
        pool_specs = args.opponents
        pool_name = "custom"
        pool_desc = "Custom opponent list"
    else:
        pool_specs = resolve_default_opponents(hero_path)
        pool_name = "default"
        pool_desc = "Default opponent policy"

    seed_all(args.seed)
    env = SimpleHoldemEnv(num_players=args.num_players)
    dummy = env.new_hand()
    state_dim = encode_state(dummy, 0).numel()

    hero_net = load_policy(state_dim, hero_path)
    hero_name = Path(hero_path).stem
    hero = HeroHandle(name=hero_name, net=hero_net, epsilon=0.0)

    opponents = build_pool_from_specs(pool_specs, state_dim)

    results = []
    for idx, eps in enumerate(eps_values):
        hero.epsilon = eps
        per_hand_bb = run_match_epsilon(
            hero=hero,
            opponents=opponents,
            num_hands=args.hands,
            seed=args.seed + idx,
            num_players=args.num_players,
        )
        mean_bb = float(np.mean(per_hand_bb)) if per_hand_bb else 0.0
        bb100 = mean_bb * 100.0
        ci_low, ci_high = normal_ci(per_hand_bb)
        results.append(
            {
                "epsilon": eps,
                "mean_bb100": bb100,
                "ci_low_bb100": ci_low * 100.0,
                "ci_high_bb100": ci_high * 100.0,
            }
        )

    best = max(results, key=lambda r: r["mean_bb100"], default=None)

    plot_path = os.path.join(args.out_dir, "argmax_epsilon_sweep.png")
    plot_epsilon_curve(
        results,
        plot_path,
        title=f"EV vs Epsilon ({hero_name}) | Pool: {pool_name}",
    )

    payload = {
        "hero_policy": hero_path,
        "pool": pool_name,
        "pool_description": pool_desc,
        "pool_specs": pool_specs,
        "hands": args.hands,
        "num_players": args.num_players,
        "eps_values": eps_values,
        "seed": args.seed,
        "results": results,
        "best": best,
        "plot": plot_path,
    }

    json_path = os.path.join(args.out_dir, "argmax_epsilon_sweep.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    if best:
        print(
            f"Best epsilon: {best['epsilon']:.4f} | mean BB/100: {best['mean_bb100']:.2f} "
            f"(CI {best['ci_low_bb100']:.2f}..{best['ci_high_bb100']:.2f})"
        )
        print(f"Plot: {plot_path}")
        print(f"JSON: {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
