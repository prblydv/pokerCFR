"""Large paired TAG and direct policy-snapshot comparison."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import pandas as pd
import torch

from evaluate_policy_report import _build_trainer
from three_player_production import (
    _clustered_summary,
    _draw_action,
    _role_name,
    evaluate_against_profile,
    load_policy_snapshot,
    paired_improvement,
)


def direct_match(
    trainer,
    hero_policy,
    opponent_policy,
    *,
    games_per_player: int,
    seed: int,
    inference_batch_size: int,
):
    records = []
    for hero in range(3):
        env = type(trainer.env)(
            stack_size=trainer.env.stack_size,
            sb=trainer.env.sb,
            bb=trainer.env.bb,
            seed=seed,
        )
        action_rng = random.Random(seed + 50_000)
        states = [env.new_hand() for _ in range(games_per_player)]
        buttons = [int(state.button) for state in states]
        steps = [0] * games_per_player
        while True:
            live = [index for index, state in enumerate(states) if not state.terminal]
            if not live:
                break
            hero_indices = [i for i in live if int(states[i].to_act) == hero]
            opponent_indices = [i for i in live if int(states[i].to_act) != hero]
            predictions = {}
            if hero_indices:
                values = trainer.average_policy_batch(
                    [states[i] for i in hero_indices],
                    policy_nets=hero_policy.policy_nets,
                    batch_size=inference_batch_size,
                )
                predictions.update(zip(hero_indices, values))
            if opponent_indices:
                values = trainer.average_policy_batch(
                    [states[i] for i in opponent_indices],
                    policy_nets=opponent_policy.policy_nets,
                    batch_size=inference_batch_size,
                )
                predictions.update(zip(opponent_indices, values))
            for index in live:
                state = states[index]
                action = _draw_action(predictions[index], action_rng)
                states[index] = env.step(state, action)
                steps[index] += 1
                if steps[index] > 256:
                    raise RuntimeError("policy match exceeded 256 decisions")
        for deal_index, (state, button) in enumerate(zip(states, buttons)):
            records.append(
                {
                    "hero_seat": hero,
                    "deal_index": deal_index,
                    "button": button,
                    "role": _role_name(button, hero),
                    "payoff_bb": float(state.payoffs[hero]) / float(trainer.env.bb),
                }
            )
    hands = pd.DataFrame.from_records(records)
    return _clustered_summary(hands), hands


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--hands", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=812_700)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    games_per_player = (args.hands + 2) // 3
    actual_hands = 3 * games_per_player
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate = load_policy_snapshot(args.candidate, device=args.device)
    baseline = load_policy_snapshot(args.baseline, device=args.device)
    trainer = _build_trainer(candidate)
    trainer.device = torch.device(args.device)
    # The trainer is only an encoder/batched-inference facade. Snapshot networks
    # already reside on the requested device.
    started = time.perf_counter()
    print(f"TAG baseline iteration {baseline.iteration}: {actual_hands:,} hands", flush=True)
    baseline_tag = evaluate_against_profile(
        trainer,
        "tight_aggressive",
        games_per_player=games_per_player,
        seed=args.seed,
        policy_nets=baseline.policy_nets,
        inference_batch_size=4096,
        hero_action_mode="sample",
    )
    print(f"TAG candidate iteration {candidate.iteration}: {actual_hands:,} hands", flush=True)
    candidate_tag = evaluate_against_profile(
        trainer,
        "tight_aggressive",
        games_per_player=games_per_player,
        seed=args.seed,
        policy_nets=candidate.policy_nets,
        inference_batch_size=4096,
        hero_action_mode="sample",
    )
    tag_delta = paired_improvement(baseline_tag, candidate_tag)
    print("candidate vs two baseline policies", flush=True)
    candidate_direct, candidate_direct_hands = direct_match(
        trainer,
        candidate,
        baseline,
        games_per_player=games_per_player,
        seed=args.seed + 100_000,
        inference_batch_size=4096,
    )
    print("baseline vs two candidate policies", flush=True)
    baseline_direct, baseline_direct_hands = direct_match(
        trainer,
        baseline,
        candidate,
        games_per_player=games_per_player,
        seed=args.seed + 100_000,
        inference_batch_size=4096,
    )
    output = {
        "candidate_iteration": candidate.iteration,
        "baseline_iteration": baseline.iteration,
        "hands_per_matchup": actual_hands,
        "seed": args.seed,
        "device": args.device,
        "candidate_vs_tag": candidate_tag.summary,
        "baseline_vs_tag": baseline_tag.summary,
        "candidate_minus_baseline_vs_tag": tag_delta,
        "candidate_vs_two_baseline": candidate_direct,
        "baseline_vs_two_candidate": baseline_direct,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    candidate_tag.hands.to_csv(args.output_dir / "candidate_vs_tag.csv", index=False)
    baseline_tag.hands.to_csv(args.output_dir / "baseline_vs_tag.csv", index=False)
    candidate_direct_hands.to_csv(
        args.output_dir / "candidate_vs_two_baseline.csv", index=False
    )
    baseline_direct_hands.to_csv(
        args.output_dir / "baseline_vs_two_candidate.csv", index=False
    )
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
