"""Train the fixed-ten-action heads-up poker policy with Deep CFR."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from heads_up_cfr import (
    DEFAULT_ROOT_STACK_DEPTHS_BB,
    ROOT_STACK_DISTRIBUTION_MIXED,
    HeadsUpNeuralCFR,
    TRAINING_DEFAULT_MAX_HISTORY,
)
from heads_up_engine import HeadsUpHoldemEngine as PythonHeadsUpHoldemEngine
from heads_up_production import evaluate_benchmark_suite


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--traversals", type=int, default=1_024, help="per player/iteration")
    parser.add_argument("--adv-steps", type=int, default=245)
    parser.add_argument("--policy-steps", type=int, default=245)
    parser.add_argument("--batch-size", type=int, default=4_096)
    parser.add_argument(
        "--traversal-workers",
        type=int,
        default=min(12, os.cpu_count() or 1),
    )
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument(
        "--eval-games",
        type=int,
        default=10_000,
        help="per hero seat/profile",
    )
    parser.add_argument(
        "--eval-profiles",
        nargs="+",
        default=("random", "calling_station", "tight_aggressive"),
        choices=("random", "calling_station", "tight_aggressive"),
    )
    parser.add_argument("--validation-seed", type=int, default=402_700)
    parser.add_argument("--starting-stack", type=int, default=200)
    parser.add_argument("--small-blind", type=int, default=1)
    parser.add_argument("--big-blind", type=int, default=2)
    parser.add_argument(
        "--root-stack-distribution",
        choices=(ROOT_STACK_DISTRIBUTION_MIXED,),
        default=ROOT_STACK_DISTRIBUTION_MIXED,
    )
    parser.add_argument(
        "--root-stack-depths-bb",
        type=int,
        nargs="+",
        default=DEFAULT_ROOT_STACK_DEPTHS_BB,
    )
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--advantage-capacity", type=int, default=1_000_000)
    parser.add_argument("--policy-capacity", type=int, default=1_000_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--engine",
        choices=("auto", "native", "python"),
        default="auto",
        help="auto prefers the compiled C++ engine and falls back to Python",
    )
    parser.add_argument("--seed", type=int, default=442)
    parser.add_argument("--max-nodes", type=int, default=5_000)
    parser.add_argument("--max-depth", type=int, default=32)
    parser.add_argument(
        "--max-history",
        type=int,
        default=TRAINING_DEFAULT_MAX_HISTORY,
        help=(
            "perfect-recall action capacity; training stops instead of silently "
            "truncating histories that exceed it"
        ),
    )
    parser.add_argument(
        "--exploration",
        type=float,
        default=0.15,
        help=(
            "optional fixed uniform mixture for coverage; nonzero values solve "
            "a perturbed strategy and are therefore approximate"
        ),
    )
    parser.add_argument(
        "--warm-start-advantages",
        action="store_true",
        help="approximate faster mode; never refit fresh advantage networks",
    )
    parser.add_argument(
        "--advantage-reinitialize-from-iteration",
        type=int,
        default=25,
        help=(
            "keep advantage networks persistent before this iteration, then "
            "fresh-fit cumulative advantage memory each iteration"
        ),
    )
    parser.add_argument(
        "--advantage-reinitialize-cycle",
        type=int,
        default=25,
        help="iterations between fresh advantage-network fits",
    )
    parser.add_argument(
        "--checkpoint",
        default="artifacts/heads_up/heads_up_cfr.pt",
    )
    parser.add_argument(
        "--metrics",
        default="artifacts/heads_up/heads_up_metrics.json",
    )
    parser.add_argument(
        "--dashboard",
        default="artifacts/heads_up/heads_up_training_dashboard.png",
        help="PNG training dashboard; use an empty value to disable",
    )
    parser.add_argument("--resume", help="resume a full heads-up checkpoint")
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument(
        "--light-checkpoint",
        action="store_true",
        help="omit reservoirs (smaller, inference-only checkpoint)",
    )
    return parser.parse_args(argv)


def _engine_type(mode: str):
    if mode == "python":
        return PythonHeadsUpHoldemEngine, "python"
    try:
        from heads_up_native import HeadsUpHoldemEngine as NativeHeadsUpHoldemEngine

        return NativeHeadsUpHoldemEngine, "native"
    except ImportError:
        if mode == "native":
            raise
        return PythonHeadsUpHoldemEngine, "python"


def main() -> None:
    args = parse_args()
    if (
        args.iterations <= 0
        or args.traversals <= 0
        or args.traversal_workers <= 0
    ):
        raise ValueError("iterations and traversals must be positive")
    if args.adv_steps <= 0 or args.policy_steps < 0 or args.batch_size <= 0:
        raise ValueError(
            "adv-steps and batch-size must be positive; policy-steps cannot be negative"
        )
    if args.eval_every <= 0 or args.eval_games <= 0 or args.save_every <= 0:
        raise ValueError("evaluation/save intervals and eval-games must be positive")
    if (
        args.learning_rate <= 0.0
        or args.advantage_capacity <= 0
        or args.policy_capacity <= 0
        or args.advantage_reinitialize_from_iteration <= 0
    ):
        raise ValueError(
            "learning rate, reservoir capacities, and advantage reset iteration "
            "must be positive"
        )
    if args.resume and args.warm_start_advantages:
        raise ValueError("--warm-start-advantages cannot be combined with --resume")
    if (
        len(args.root_stack_depths_bb) < 2
        or len(set(args.root_stack_depths_bb)) != len(args.root_stack_depths_bb)
        or any(value <= 0 for value in args.root_stack_depths_bb)
    ):
        raise ValueError(
            "--root-stack-depths-bb requires unique positive integer depths"
        )

    engine_class, engine_name = _engine_type(args.engine)
    env = engine_class(
        starting_stack=args.starting_stack,
        small_blind=args.small_blind,
        big_blind=args.big_blind,
        seed=args.seed,
    )
    checkpoint = Path(args.resume) if args.resume else Path(args.checkpoint)
    metrics_path = (
        Path(args.resume).parent / "heads_up_metrics.json"
        if args.resume and args.metrics == "artifacts/heads_up/heads_up_metrics.json"
        else Path(args.metrics)
    )
    dashboard_path = (
        Path(args.resume).parent / "heads_up_training_dashboard.png"
        if args.resume
        and args.dashboard == "artifacts/heads_up/heads_up_training_dashboard.png"
        else Path(args.dashboard)
        if args.dashboard
        else None
    )
    if args.resume:
        trainer = HeadsUpNeuralCFR.load(args.resume, env, device=args.device)
        if not trainer.can_resume_training:
            raise RuntimeError("the supplied light checkpoint is inference-only")
    else:
        trainer = HeadsUpNeuralCFR(
            env,
            device=args.device,
            hidden=args.hidden,
            blocks=args.blocks,
            learning_rate=args.learning_rate,
            advantage_capacity=args.advantage_capacity,
            policy_capacity=args.policy_capacity,
            max_history=args.max_history,
            max_nodes_per_traversal=args.max_nodes,
            max_depth=args.max_depth,
            exploration=args.exploration,
            reinitialize_advantage_each_iteration=(
                not args.warm_start_advantages
            ),
            advantage_reinitialize_from_iteration=(
                args.advantage_reinitialize_from_iteration
            ),
            advantage_reinitialize_cycle=args.advantage_reinitialize_cycle,
            seed=args.seed,
        )

    records = [dict(row) for row in trainer.metrics]
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    def flush_metrics() -> None:
        temporary = metrics_path.with_suffix(metrics_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(_json_safe(records), indent=2),
            encoding="utf-8",
        )
        temporary.replace(metrics_path)
        if dashboard_path is not None and records:
            from heads_up_reporting import save_training_dashboard

            save_training_dashboard(records, dashboard_path)

    print(
        f"engine={engine_name} actions=10 encoder_width={trainer.input_dim} "
        f"max_history={trainer.max_history} workers={args.traversal_workers}"
    )
    for _ in range(args.iterations):
        row = trainer.train_iteration(
            traversals_per_player=args.traversals,
            advantage_steps=args.adv_steps,
            policy_steps=args.policy_steps,
            batch_size=args.batch_size,
            traversal_workers=args.traversal_workers,
            root_stack_distribution=args.root_stack_distribution,
            root_stack_depths_bb=tuple(args.root_stack_depths_bb),
        )
        if trainer.iteration % args.eval_every == 0:
            evaluation = evaluate_benchmark_suite(
                trainer,
                profiles=tuple(args.eval_profiles),
                games_per_seat=args.eval_games,
                seed=args.validation_seed,
            )
            row.update(evaluation)
            row["mean_ev_bb"] = float(evaluation["benchmark_composite_ev_bb"])
            tag_prefix = "benchmark_tight_aggressive_"
            if "tight_aggressive" in args.eval_profiles:
                for seat in range(2):
                    row[f"seat_{seat}_ev_bb"] = float(
                        evaluation[f"{tag_prefix}seat_{seat}_ev_bb"]
                    )
        # train_iteration records the row before held-out evaluation. Keep the
        # checkpoint copy complete so a resumed run retains historical EV.
        trainer.metrics[-1] = dict(row)
        records.append(dict(row))
        flush_metrics()
        ev = row.get("mean_ev_bb")
        ev_text = (
            f", benchmark EV={float(ev):+.3f} BB/hand"
            if ev is not None
            else ""
        )
        print(
            f"iteration={trainer.iteration:4d} "
            f"nodes={int(row['nodes']):6d} "
            f"time={row['seconds']:.2f}s "
            f"regret={row['mean_abs_regret']:.3f}{ev_text}"
        )
        if trainer.iteration % args.save_every == 0:
            trainer.save(
                checkpoint,
                include_buffers=not args.light_checkpoint,
            )

    checkpoint_path = trainer.save(
        checkpoint,
        include_buffers=not args.light_checkpoint,
    )
    flush_metrics()
    print(f"saved checkpoint: {checkpoint_path}")
    print(f"saved metrics:    {metrics_path}")
    if dashboard_path is not None:
        print(f"saved dashboard:  {dashboard_path}")


if __name__ == "__main__":
    main()
