"""Command-line training entry point for the three-player poker bot."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from three_player_cfr import ThreePlayerNeuralCFR
from three_player_engine import ThreePlayerHoldemEnv as PythonThreePlayerHoldemEnv
from three_player_native import ThreePlayerHoldemEnv as NativeThreePlayerHoldemEnv
from three_player_production import evaluate_tournaments_against_profile


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--traversals", type=int, default=1, help="per player/iteration")
    parser.add_argument("--adv-steps", type=int, default=16)
    parser.add_argument("--policy-steps", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--eval-games", type=int, default=99, help="per hero seat")
    parser.add_argument(
        "--stack",
        type=float,
        default=40.0,
        help="chips; 40 is a quick 20BB demo, use 200 for 100BB",
    )
    parser.add_argument("--small-blind", type=float, default=1.0)
    parser.add_argument("--big-blind", type=float, default=2.0)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--native-engine",
        action="store_true",
        help="use the compiled C++ hand engine (build with engine C\\build.bat)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-nodes", type=int, default=5000)
    parser.add_argument(
        "--tournament",
        action="store_true",
        help=(
            "train on chip-conserving variable-stack three-handed and heads-up "
            "hand roots with tournament state features"
        ),
    )
    parser.add_argument(
        "--tournament-total-chips",
        type=float,
        help="total chips across all seats (default: three times --stack)",
    )
    parser.add_argument("--heads-up-root-fraction", type=float, default=0.25)
    parser.add_argument("--continuation-root-fraction", type=float, default=0.25)
    parser.add_argument(
        "--minimum-live-stack",
        type=float,
        help="minimum sampled live stack (default: small blind)",
    )
    parser.add_argument("--root-stack-concentration", type=float, default=0.7)
    parser.add_argument("--continuation-capacity", type=int, default=2048)
    parser.add_argument(
        "--warm-start-policy",
        help=(
            "initialize only the average-policy networks from a compatible "
            "policy snapshot or full checkpoint"
        ),
    )
    parser.add_argument(
        "--tournament-eval-games",
        type=int,
        default=0,
        help="complete tournaments versus random per hero seat at evaluation points",
    )
    parser.add_argument("--tournament-eval-max-hands", type=int, default=10000)
    parser.add_argument(
        "--warm-start-advantages",
        action="store_true",
        help="faster approximate mode; strict mode refits fresh advantage nets",
    )
    parser.add_argument(
        "--checkpoint",
        help="output checkpoint (mode-specific artifact path by default)",
    )
    parser.add_argument(
        "--metrics", help="output metrics JSON (mode-specific path by default)"
    )
    parser.add_argument("--resume", help="resume a full checkpoint")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument(
        "--light-checkpoint",
        action="store_true",
        help="omit replay buffers (smaller, but cannot resume CFR faithfully)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.eval_every <= 0 or args.save_every <= 0:
        raise ValueError("eval-every and save-every must be positive")
    if args.tournament_eval_games < 0 or args.tournament_eval_max_hands <= 0:
        raise ValueError(
            "tournament evaluation games cannot be negative and max hands must be positive"
        )
    if args.resume and args.warm_start_policy:
        raise ValueError("--warm-start-policy cannot be combined with --resume")
    default_root = Path(
        "artifacts/three_player_tournament"
        if args.tournament
        else "artifacts"
    )
    checkpoint = (
        Path(args.checkpoint)
        if args.checkpoint
        else (
            Path(args.resume)
            if args.resume
            else default_root / "three_player_cfr.pt"
        )
    )
    metrics_path = (
        Path(args.metrics)
        if args.metrics
        else (
            Path(args.resume).parent / "three_player_metrics.json"
            if args.resume
            else default_root / "three_player_metrics.json"
        )
    )
    engine_type = (
        NativeThreePlayerHoldemEnv if args.native_engine else PythonThreePlayerHoldemEnv
    )
    env = engine_type(
        stack_size=args.stack,
        sb=args.small_blind,
        bb=args.big_blind,
        seed=args.seed,
    )
    if args.resume:
        trainer = ThreePlayerNeuralCFR.load(args.resume, env, device=args.device)
        if not trainer.can_resume_training:
            raise RuntimeError("the supplied light checkpoint is inference-only")
    else:
        trainer = ThreePlayerNeuralCFR(
            env,
            device=args.device,
            hidden=args.hidden,
            blocks=args.blocks,
            max_nodes_per_traversal=args.max_nodes,
            reinitialize_advantage_each_iteration=not args.warm_start_advantages,
            include_tournament_features=args.tournament,
            variable_stack_training=args.tournament,
            tournament_total_chips=args.tournament_total_chips,
            heads_up_root_fraction=args.heads_up_root_fraction,
            continuation_root_fraction=args.continuation_root_fraction,
            minimum_live_stack=args.minimum_live_stack,
            root_stack_concentration=args.root_stack_concentration,
            continuation_capacity=args.continuation_capacity,
            seed=args.seed,
        )
        if args.warm_start_policy:
            report = trainer.warm_start_policy(args.warm_start_policy)
            print(
                "warm-started three policy networks: "
                f"{report['source_input_dim']} -> {report['target_input_dim']} features"
            )

    records: list[dict] = [dict(row) for row in trainer.metrics]
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    def flush_metrics() -> None:
        temporary = metrics_path.with_suffix(metrics_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(_json_safe(records), indent=2), encoding="utf-8"
        )
        temporary.replace(metrics_path)

    for _ in range(args.iterations):
        row = trainer.train_iteration(
            traversals_per_player=args.traversals,
            advantage_steps=args.adv_steps,
            policy_steps=args.policy_steps,
            batch_size=args.batch_size,
        )
        if trainer.iteration % args.eval_every == 0:
            evaluation = trainer.evaluate_vs_random(args.eval_games)
            row.update({key: value for key, value in evaluation.items() if key != "action_counts"})
            if args.tournament_eval_games:
                tournament_evaluation = evaluate_tournaments_against_profile(
                    trainer,
                    "random",
                    tournaments_per_player=args.tournament_eval_games,
                    seed=args.seed + 700_000,
                    max_hands=args.tournament_eval_max_hands,
                )
                row.update(
                    {
                        f"tournament_random_{key}": value
                        for key, value in tournament_evaluation.summary.items()
                    }
                )
        records.append(dict(row))
        flush_metrics()
        ev = row.get("mean_ev_bb")
        ev_text = f", EV vs random={ev:+.3f} BB/hand" if ev is not None else ""
        print(
            f"iteration={trainer.iteration:4d} nodes={int(row['nodes']):6d} "
            f"time={row['seconds']:.2f}s regret={row['mean_abs_regret']:.3f}{ev_text}"
        )
        if trainer.iteration % args.save_every == 0:
            trainer.save(checkpoint, include_buffers=not args.light_checkpoint)

    checkpoint_path = trainer.save(
        checkpoint, include_buffers=not args.light_checkpoint
    )
    flush_metrics()
    print(f"saved checkpoint: {checkpoint_path}")
    print(f"saved metrics:    {metrics_path}")


if __name__ == "__main__":
    main()
