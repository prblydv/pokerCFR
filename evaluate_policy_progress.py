"""Compare a candidate policy with a prior snapshot against TAG and each other."""

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
    EvaluationResult,
    _clustered_summary,
    _draw_action,
    _role_name,
    evaluate_against_profile,
    load_policy_snapshot,
    paired_improvement,
)


def _action(probabilities: torch.Tensor, rng: random.Random, mode: str) -> int:
    if mode == "argmax":
        return int(torch.argmax(probabilities).item())
    return _draw_action(probabilities, rng)


def direct_match(
    trainer,
    hero_policy,
    opponent_policy,
    *,
    games_per_player: int,
    seed: int,
    inference_batch_size: int,
    action_mode: str,
) -> EvaluationResult:
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
                states[index] = env.step(
                    states[index],
                    _action(predictions[index], action_rng, action_mode),
                )
                steps[index] += 1
                if steps[index] > 256:
                    raise RuntimeError("policy match exceeded 256 decisions")
        for deal_index, (state, button) in enumerate(zip(states, buttons)):
            records.append(
                {
                    "profile": "policy_snapshot",
                    "hero_seat": hero,
                    "deal_index": deal_index,
                    "button": button,
                    "role": _role_name(button, hero),
                    "payoff_bb": float(state.payoffs[hero]) / float(trainer.env.bb),
                }
            )
    hands = pd.DataFrame.from_records(records)
    return EvaluationResult("policy_snapshot", _clustered_summary(hands), hands)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--hands", type=int, default=100_002)
    parser.add_argument("--seed", type=int, default=512_700)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    games_per_player = (args.hands + 2) // 3
    actual_hands = 3 * games_per_player
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate = load_policy_snapshot(args.candidate, device=args.device)
    baseline = load_policy_snapshot(args.baseline, device=args.device)
    trainer = _build_trainer(candidate)
    trainer.device = torch.device(args.device)
    output = {
        "candidate_iteration": candidate.iteration,
        "baseline_iteration": baseline.iteration,
        "hands_per_matchup": actual_hands,
        "seed": args.seed,
        "device": args.device,
        "modes": {},
    }
    started = time.perf_counter()
    for mode in ("sample", "argmax"):
        mode_started = time.perf_counter()
        print(f"{mode}: baseline vs TAG", flush=True)
        baseline_tag = evaluate_against_profile(
            trainer,
            "tight_aggressive",
            games_per_player=games_per_player,
            seed=args.seed,
            policy_nets=baseline.policy_nets,
            inference_batch_size=4096,
            hero_action_mode=mode,
        )
        print(f"{mode}: candidate vs TAG", flush=True)
        candidate_tag = evaluate_against_profile(
            trainer,
            "tight_aggressive",
            games_per_player=games_per_player,
            seed=args.seed,
            policy_nets=candidate.policy_nets,
            inference_batch_size=4096,
            hero_action_mode=mode,
        )
        print(f"{mode}: candidate vs two baseline policies", flush=True)
        candidate_direct = direct_match(
            trainer,
            candidate,
            baseline,
            games_per_player=games_per_player,
            seed=args.seed + 100_000,
            inference_batch_size=4096,
            action_mode=mode,
        )
        print(f"{mode}: baseline vs two candidate policies", flush=True)
        baseline_direct = direct_match(
            trainer,
            baseline,
            candidate,
            games_per_player=games_per_player,
            seed=args.seed + 100_000,
            inference_batch_size=4096,
            action_mode=mode,
        )
        output["modes"][mode] = {
            "candidate_vs_tag": candidate_tag.summary,
            "baseline_vs_tag": baseline_tag.summary,
            "candidate_minus_baseline_vs_tag": paired_improvement(
                baseline_tag, candidate_tag
            ),
            "candidate_vs_two_baseline": candidate_direct.summary,
            "baseline_vs_two_candidate": baseline_direct.summary,
            "candidate_minus_baseline_reciprocal_head_to_head": paired_improvement(
                baseline_direct, candidate_direct
            ),
            "elapsed_seconds": time.perf_counter() - mode_started,
        }
        for name, result in (
            ("candidate_vs_tag", candidate_tag),
            ("baseline_vs_tag", baseline_tag),
            ("candidate_vs_two_baseline", candidate_direct),
            ("baseline_vs_two_candidate", baseline_direct),
        ):
            result.hands.to_csv(args.output_dir / f"{mode}_{name}.csv", index=False)
        (args.output_dir / "comparison.json").write_text(
            json.dumps(output, indent=2), encoding="utf-8"
        )
        print(json.dumps(output["modes"][mode], indent=2), flush=True)
    output["elapsed_seconds"] = time.perf_counter() - started
    (args.output_dir / "comparison.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
