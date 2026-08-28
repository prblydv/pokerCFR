"""Generate a large-sample strength and critical-situation policy report."""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from three_player_analysis import (
    StrategyAnalyzer,
    compare_ranges,
    plot_range_delta,
    plot_range_heatmaps,
    postflop_scenarios,
    preflop_scenarios,
)
from three_player_cfr import ThreePlayerNeuralCFR
from three_player_engine import ACTION_CALL
from three_player_native import ThreePlayerHoldemEnv
from three_player_production import (
    evaluate_against_profile,
    load_policy_snapshot,
    paired_improvement,
)


PROFILES = ("random", "calling_station", "tight_aggressive")


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_")


def _build_trainer(snapshot) -> ThreePlayerNeuralCFR:
    metadata = snapshot.metadata
    environment = metadata["environment"]
    env = ThreePlayerHoldemEnv(
        stack_size=float(environment["stack_size"]),
        sb=float(environment["sb"]),
        bb=float(environment["bb"]),
        seed=402_700,
    )
    return ThreePlayerNeuralCFR(
        env,
        device="cpu",
        hidden=int(metadata["hidden"]),
        blocks=int(metadata["blocks"]),
        network_architecture=str(metadata["network_architecture"]),
        max_history=int(metadata["max_history"]),
        include_tournament_features=bool(metadata["include_tournament_features"]),
        tournament_total_chips=float(
            environment.get("tournament_total_chips", 600.0)
        ),
        reinitialize_advantage_each_iteration=False,
        seed=442,
    )


def _plot_match_ev(summary: pd.DataFrame, path: Path) -> None:
    profiles = list(PROFILES)
    candidate = summary.set_index("profile")
    x = np.arange(len(profiles), dtype=float)
    means = candidate.loc[profiles, "candidate_mean_ev_bb"].to_numpy(float)
    lows = candidate.loc[profiles, "candidate_ci95_low_bb"].to_numpy(float)
    highs = candidate.loc[profiles, "candidate_ci95_high_bb"].to_numpy(float)
    baseline = candidate.loc[profiles, "baseline_mean_ev_bb"].to_numpy(float)
    fig, axis = plt.subplots(figsize=(9, 5.5))
    axis.errorbar(
        x,
        means,
        yerr=np.vstack((means - lows, highs - means)),
        fmt="o",
        capsize=6,
        linewidth=2,
        markersize=8,
        label="policy 5000 (95% CI)",
    )
    axis.scatter(x, baseline, marker="s", s=65, label="initial policy")
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_xticks(x, [name.replace("_", " ") for name in profiles])
    axis.set_ylabel("EV (BB/hand)")
    axis.set_title("Large fixed-deal evaluation")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_paired_delta(summary: pd.DataFrame, path: Path) -> None:
    profiles = list(PROFILES)
    table = summary.set_index("profile").loc[profiles]
    means = table["delta_ev_bb"].to_numpy(float)
    lows = table["delta_ci95_low_bb"].to_numpy(float)
    highs = table["delta_ci95_high_bb"].to_numpy(float)
    colors = ["tab:green" if value > 0 else "tab:red" for value in means]
    x = np.arange(len(profiles), dtype=float)
    fig, axis = plt.subplots(figsize=(9, 5.5))
    axis.bar(x, means, color=colors, alpha=0.75)
    axis.errorbar(
        x,
        means,
        yerr=np.vstack((means - lows, highs - means)),
        fmt="none",
        ecolor="black",
        capsize=6,
    )
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_xticks(x, [name.replace("_", " ") for name in profiles])
    axis.set_ylabel("Candidate minus initial (BB/hand)")
    axis.set_title("Paired improvement on identical deals (95% CI)")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_position_ev(summary: pd.DataFrame, path: Path) -> None:
    roles = ("BTN", "SB", "BB")
    x = np.arange(len(PROFILES), dtype=float)
    width = 0.24
    fig, axis = plt.subplots(figsize=(10, 5.8))
    for index, role in enumerate(roles):
        values = [
            float(
                summary.loc[summary["profile"] == profile, f"candidate_ev_{role}_bb"].iloc[0]
            )
            for profile in PROFILES
        ]
        axis.bar(x + (index - 1) * width, values, width, label=role)
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_xticks(x, [name.replace("_", " ") for name in PROFILES])
    axis.set_ylabel("EV (BB/hand)")
    axis.set_title("Policy 5000 positional performance")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_payoff_ecdf(candidate_hands: dict[str, pd.DataFrame], path: Path) -> None:
    all_values = np.concatenate(
        [frame["payoff_bb"].to_numpy(float) for frame in candidate_hands.values()]
    )
    lower, upper = np.quantile(all_values, [0.005, 0.995])
    fig, axis = plt.subplots(figsize=(9.5, 5.8))
    for profile, frame in candidate_hands.items():
        values = np.sort(frame["payoff_bb"].to_numpy(float))
        probabilities = np.arange(1, len(values) + 1, dtype=float) / len(values)
        axis.plot(values, probabilities, label=profile.replace("_", " "))
    axis.axvline(0.0, color="black", linewidth=1)
    axis.set_xlim(float(lower), float(upper))
    axis.set_xlabel("Payoff (BB/hand); extreme 0.5% tails clipped from view")
    axis.set_ylabel("Cumulative probability")
    axis.set_title("Policy 5000 payoff-risk distribution")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _payoff_risk(profile: str, frame: pd.DataFrame) -> dict[str, float | str]:
    values = frame["payoff_bb"].to_numpy(float)
    cutoff = float(np.quantile(values, 0.05))
    worst = values[values <= cutoff]
    return {
        "profile": profile,
        "hands": int(len(values)),
        "mean_bb": float(np.mean(values)),
        "median_bb": float(np.median(values)),
        "p05_bb": cutoff,
        "p95_bb": float(np.quantile(values, 0.95)),
        "worst_5pct_mean_bb": float(np.mean(worst)),
        "loss_10bb_or_more_fraction": float(np.mean(values <= -10.0)),
        "win_fraction": float(np.mean(values > 0.0)),
    }


def _critical_reports(
    trainer: ThreePlayerNeuralCFR,
    candidate,
    baseline,
    output_dir: Path,
) -> pd.DataFrame:
    ranges_dir = output_dir / "critical_ranges"
    ranges_dir.mkdir(parents=True, exist_ok=True)
    # Same representative runout used by the production notebook.
    board_scenarios = postflop_scenarios(
        flop=(51, 18, 0),  # As 7d 2c
        turn=35,  # Jh
        river=41,  # 4s
    )
    scenarios = (*preflop_scenarios(), *board_scenarios)
    analyzer = StrategyAnalyzer(trainer, batch_size=4096)
    rows: list[dict[str, float | str]] = []
    for index, scenario in enumerate(scenarios, start=1):
        print(f"critical {index}/{len(scenarios)}: {scenario.label}", flush=True)
        initial_report = analyzer.analyze_range(
            scenario, policy_nets=baseline.policy_nets
        )
        current_report = analyzer.analyze_range(
            scenario, policy_nets=candidate.policy_nets
        )
        comparison = compare_ranges(initial_report, current_report)
        current_report.hand_table.to_csv(
            ranges_dir / f"{scenario.scenario_id}_policy5000.csv", index=False
        )
        comparison.to_csv(
            ranges_dir / f"{scenario.scenario_id}_delta_vs_initial.csv", index=False
        )
        legal_actions = set(current_report.state_summary["legal_actions"])
        metrics = (
            ("p_fold", "p_call", "p_aggressive")
            if "call" in legal_actions
            else ("p_check", "p_aggressive")
        )
        figure = plot_range_heatmaps(current_report, metrics=metrics)
        figure.savefig(
            ranges_dir / f"{scenario.scenario_id}_policy5000.png",
            dpi=150,
        )
        plt.close(figure)
        delta_metric = (
            "delta_p_continue" if "fold" in legal_actions else "delta_p_aggressive"
        )
        figure = plot_range_delta(
            comparison,
            metric=delta_metric,
            title=f"{scenario.label}: policy 5000 minus initial",
        )
        figure.savefig(
            ranges_dir / f"{scenario.scenario_id}_delta_vs_initial.png",
            dpi=150,
        )
        plt.close(figure)
        row: dict[str, float | str] = {
            "scenario_id": scenario.scenario_id,
            "label": scenario.label,
            "street": float(current_report.state_summary["street"]),
            "pot_bb": float(current_report.state_summary["pot_bb"]),
            "to_call_bb": float(current_report.state_summary["to_call_bb"]),
            "mean_strategy_total_variation": float(
                comparison["strategy_total_variation"].mean()
            ),
        }
        for metric in ("p_fold", "p_check", "p_call", "p_aggressive", "p_continue"):
            row[f"initial_{metric}"] = float(
                initial_report.hand_table[metric].mean(skipna=True)
            )
            row[f"policy5000_{metric}"] = float(
                current_report.hand_table[metric].mean(skipna=True)
            )
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "critical_situations_summary.csv", index=False)
    return table


def _plot_critical_summary(table: pd.DataFrame, path: Path) -> None:
    labels = table["label"].tolist()
    x = np.arange(len(labels), dtype=float)
    width = 0.36
    fig, axes = plt.subplots(2, 1, figsize=(13, 10), sharex=True)
    axes[0].bar(
        x - width / 2,
        table["initial_p_aggressive"],
        width,
        label="initial",
    )
    axes[0].bar(
        x + width / 2,
        table["policy5000_p_aggressive"],
        width,
        label="policy 5000",
    )
    axes[0].set_ylabel("Mean raise probability")
    axes[0].set_title("Critical-situation aggression")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(x, table["mean_strategy_total_variation"], color="tab:purple")
    axes[1].set_ylabel("Mean total-variation distance")
    axes[1].set_title("How much the policy changed from initialization")
    axes[1].set_xticks(x, labels, rotation=35, ha="right")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--games-per-player", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=402_700)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.games_per_player <= 0:
        raise ValueError("games-per-player must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate = load_policy_snapshot(args.checkpoint, device="cpu")
    baseline = load_policy_snapshot(args.baseline, device="cpu")
    trainer = _build_trainer(candidate)
    print(
        f"candidate iteration {candidate.iteration}; "
        f"{3 * args.games_per_player:,} hands/profile",
        flush=True,
    )

    summary_path = args.output_dir / "scripted_opponent_summary.csv"
    risk_path = args.output_dir / "payoff_risk_summary.csv"
    hand_paths = {
        profile: args.output_dir / f"{profile}_policy5000_hands.csv"
        for profile in PROFILES
    }
    summary_rows: list[dict[str, float | str]] = []
    risk_rows: list[dict[str, float | str]] = []
    candidate_hands: dict[str, pd.DataFrame] = {}
    started = time.perf_counter()
    if summary_path.exists() and risk_path.exists() and all(
        path.exists() for path in hand_paths.values()
    ):
        print("reusing completed large match simulations", flush=True)
        summary = pd.read_csv(summary_path)
        risk = pd.read_csv(risk_path)
        candidate_hands = {
            profile: pd.read_csv(path) for profile, path in hand_paths.items()
        }
    else:
        for index, profile in enumerate(PROFILES, start=1):
            print(f"match {index}/{len(PROFILES)}: {profile} baseline", flush=True)
            initial_result = evaluate_against_profile(
                trainer,
                profile,
                games_per_player=args.games_per_player,
                seed=args.seed,
                policy_nets=baseline.policy_nets,
                inference_batch_size=1024,
            )
            print(f"match {index}/{len(PROFILES)}: {profile} policy 5000", flush=True)
            current_result = evaluate_against_profile(
                trainer,
                profile,
                games_per_player=args.games_per_player,
                seed=args.seed,
                policy_nets=candidate.policy_nets,
                inference_batch_size=1024,
            )
            paired = paired_improvement(initial_result, current_result)
            candidate_hands[profile] = current_result.hands
            current_result.hands.to_csv(hand_paths[profile], index=False)
            row: dict[str, float | str] = {"profile": profile}
            for key, value in current_result.summary.items():
                row[f"candidate_{key}"] = float(value)
            for key, value in initial_result.summary.items():
                row[f"baseline_{key}"] = float(value)
            row.update({key: float(value) for key, value in paired.items()})
            summary_rows.append(row)
            risk_rows.append(_payoff_risk(profile, current_result.hands))

        summary = pd.DataFrame(summary_rows)
        risk = pd.DataFrame(risk_rows)
        summary.to_csv(summary_path, index=False)
        risk.to_csv(risk_path, index=False)
    _plot_match_ev(summary, args.output_dir / "scripted_opponent_ev.png")
    _plot_paired_delta(summary, args.output_dir / "paired_delta_vs_initial.png")
    _plot_position_ev(summary, args.output_dir / "position_ev.png")
    _plot_payoff_ecdf(candidate_hands, args.output_dir / "payoff_ecdf.png")

    critical = _critical_reports(trainer, candidate, baseline, args.output_dir)
    _plot_critical_summary(critical, args.output_dir / "critical_situations.png")
    metadata = {
        "candidate": str(args.checkpoint.resolve()),
        "candidate_iteration": candidate.iteration,
        "baseline": str(args.baseline.resolve()),
        "baseline_iteration": baseline.iteration,
        "games_per_player": args.games_per_player,
        "hands_per_profile": 3 * args.games_per_player,
        "seed": args.seed,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "report_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)
    print(f"report written to {args.output_dir.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
