"""Evaluate one policy using deterministic argmax actions against scripted bots."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluate_policy_report import PROFILES, _build_trainer
from three_player_production import evaluate_against_profile, load_policy_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hands-per-profile", type=int, default=100_002)
    parser.add_argument("--seed", type=int, default=512_700)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def plot_ev(summary: pd.DataFrame, path: Path) -> None:
    table = summary.set_index("profile").loc[list(PROFILES)]
    means = table["mean_ev_bb"].to_numpy(float)
    lows = table["ci95_low_bb"].to_numpy(float)
    highs = table["ci95_high_bb"].to_numpy(float)
    x = np.arange(len(PROFILES))
    colors = ["tab:green" if value >= 0 else "tab:red" for value in means]
    figure, axis = plt.subplots(figsize=(9, 5.6))
    axis.bar(x, means, color=colors, alpha=0.82)
    axis.errorbar(
        x,
        means,
        yerr=np.vstack((means - lows, highs - means)),
        fmt="none",
        ecolor="black",
        capsize=7,
        linewidth=2,
    )
    axis.axhline(0, color="black", linewidth=1)
    axis.set_xticks(x, [profile.replace("_", " ") for profile in PROFILES])
    axis.set_ylabel("Argmax policy EV (BB/hand)")
    axis.set_title("Policy 5000 deterministic argmax performance (95% CI)")
    axis.grid(axis="y", alpha=0.25)
    for index, value in enumerate(means):
        axis.text(index, value, f"{value:+.3f}", ha="center", va="bottom" if value >= 0 else "top")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_positions(summary: pd.DataFrame, path: Path) -> None:
    x = np.arange(len(PROFILES), dtype=float)
    width = 0.24
    figure, axis = plt.subplots(figsize=(10, 5.8))
    for offset, role in enumerate(("BTN", "SB", "BB")):
        values = summary.set_index("profile").loc[list(PROFILES), f"ev_{role}_bb"].to_numpy(float)
        axis.bar(x + (offset - 1) * width, values, width, label=role)
    axis.axhline(0, color="black", linewidth=1)
    axis.set_xticks(x, [profile.replace("_", " ") for profile in PROFILES])
    axis.set_ylabel("EV (BB/hand)")
    axis.set_title("Argmax performance by table position")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_cumulative(hands: dict[str, pd.DataFrame], path: Path) -> None:
    figure, axes = plt.subplots(len(PROFILES), 1, figsize=(11, 9), sharex=False)
    for axis, profile in zip(axes, PROFILES):
        frame = hands[profile].sort_values(["deal_index", "hero_seat"])
        cumulative = frame["payoff_bb"].to_numpy(float).cumsum()
        axis.plot(np.arange(1, len(cumulative) + 1), cumulative, linewidth=1)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.fill_between(
            np.arange(1, len(cumulative) + 1),
            0,
            cumulative,
            where=cumulative >= 0,
            color="tab:green",
            alpha=0.18,
        )
        axis.fill_between(
            np.arange(1, len(cumulative) + 1),
            0,
            cumulative,
            where=cumulative < 0,
            color="tab:red",
            alpha=0.18,
        )
        axis.set_title(f"{profile.replace('_', ' ')}: final {cumulative[-1]:+.1f} BB")
        axis.set_ylabel("Cumulative BB")
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("Evaluated hands (all three learned seats interleaved by deal)")
    figure.suptitle("Deterministic argmax policy: cumulative wins and losses", y=1.0)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_outcomes(hands: dict[str, pd.DataFrame], path: Path) -> pd.DataFrame:
    rows = []
    for profile, frame in hands.items():
        values = frame["payoff_bb"].to_numpy(float)
        rows.append(
            {
                "profile": profile,
                "winning_hands_fraction": float(np.mean(values > 0)),
                "losing_hands_fraction": float(np.mean(values < 0)),
                "break_even_hands_fraction": float(np.mean(values == 0)),
                "gross_bb_won": float(values[values > 0].sum()),
                "gross_bb_lost": float(values[values < 0].sum()),
                "net_bb": float(values.sum()),
            }
        )
    table = pd.DataFrame(rows)
    x = np.arange(len(PROFILES), dtype=float)
    indexed = table.set_index("profile").loc[list(PROFILES)]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].bar(x - 0.18, indexed["winning_hands_fraction"] * 100, 0.36, label="winning")
    axes[0].bar(x + 0.18, indexed["losing_hands_fraction"] * 100, 0.36, label="losing")
    axes[0].set_ylabel("Hands (%)")
    axes[0].set_title("Winning versus losing hand frequency")
    axes[0].legend()
    axes[1].bar(x, indexed["net_bb"], color=["tab:green" if v >= 0 else "tab:red" for v in indexed["net_bb"]])
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_ylabel("Net BB over complete test")
    axes[1].set_title("Total big blinds won or lost")
    for axis in axes:
        axis.set_xticks(x, [profile.replace("_", " ") for profile in PROFILES])
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return table


def main() -> int:
    args = parse_args()
    if args.hands_per_profile < 3:
        raise ValueError("hands-per-profile must be at least 3")
    games_per_player = (args.hands_per_profile + 2) // 3
    actual_hands = games_per_player * 3
    args.output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = load_policy_snapshot(args.checkpoint, device="cpu")
    trainer = _build_trainer(snapshot)
    summaries = []
    hands: dict[str, pd.DataFrame] = {}
    started = time.perf_counter()
    for index, profile in enumerate(PROFILES, start=1):
        print(f"{index}/{len(PROFILES)} {profile}: {actual_hands:,} argmax-policy hands", flush=True)
        result = evaluate_against_profile(
            trainer,
            profile,
            games_per_player=games_per_player,
            seed=args.seed,
            policy_nets=snapshot.policy_nets,
            inference_batch_size=2048,
            hero_action_mode="argmax",
        )
        row = {"profile": profile, **result.summary}
        row["net_bb"] = float(result.hands["payoff_bb"].sum())
        summaries.append(row)
        hands[profile] = result.hands
        result.hands.to_csv(args.output_dir / f"{profile}_argmax_hands.csv", index=False)
        print(f"  EV {row['mean_ev_bb']:+.4f} BB/hand; net {row['net_bb']:+.1f} BB", flush=True)
    summary = pd.DataFrame(summaries)
    summary.to_csv(args.output_dir / "argmax_summary.csv", index=False)
    outcomes = plot_outcomes(hands, args.output_dir / "argmax_wins_losses.png")
    outcomes.to_csv(args.output_dir / "argmax_outcomes.csv", index=False)
    plot_ev(summary, args.output_dir / "argmax_ev.png")
    plot_positions(summary, args.output_dir / "argmax_position_ev.png")
    plot_cumulative(hands, args.output_dir / "argmax_cumulative_bb.png")
    metadata = {
        "checkpoint": str(args.checkpoint.resolve()),
        "iteration": snapshot.iteration,
        "hero_action_mode": "argmax",
        "scripted_opponent_action_mode": "sample",
        "hands_per_profile": actual_hands,
        "seed": args.seed,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(f"report written to {args.output_dir.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
