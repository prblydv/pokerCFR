"""Training plots for the heads-up Deep CFR campaign."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib

if "inline" not in str(matplotlib.get_backend()).lower():
    matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def metrics_frame(metrics: Iterable[dict[str, Any]]) -> pd.DataFrame:
    frame = (
        metrics.copy()
        if isinstance(metrics, pd.DataFrame)
        else pd.DataFrame(list(metrics))
    )
    if "iteration" in frame:
        frame = frame.sort_values("iteration").reset_index(drop=True)
    return frame


def plot_training_dashboard(
    metrics: Iterable[dict[str, Any]],
) -> plt.Figure:
    """Build the production nine-panel heads-up training dashboard."""

    frame = metrics_frame(metrics)
    if frame.empty:
        raise ValueError("no metrics are available")

    figure, axes = plt.subplots(3, 3, figsize=(20, 14))

    benchmark_columns = [
        column
        for column in (
            "benchmark_random_mean_ev_bb",
            "benchmark_calling_station_mean_ev_bb",
            "benchmark_tight_aggressive_mean_ev_bb",
            "benchmark_reference_policy_mean_ev_bb",
            "benchmark_composite_ev_bb",
        )
        if column in frame
    ]
    evaluation_key = (
        "benchmark_composite_ev_bb"
        if "benchmark_composite_ev_bb" in frame
        else "mean_ev_bb"
    )
    evaluation = (
        frame.dropna(subset=[evaluation_key])
        if evaluation_key in frame
        else pd.DataFrame()
    )
    if evaluation.empty:
        axes[0, 0].text(
            0.5,
            0.5,
            "No held-out evaluation yet",
            ha="center",
            va="center",
        )
    else:
        if benchmark_columns:
            for column in benchmark_columns:
                axes[0, 0].plot(
                    evaluation["iteration"],
                    evaluation[column],
                    marker="o" if column == "benchmark_composite_ev_bb" else None,
                    label=column.removeprefix("benchmark_").removesuffix(
                        "_mean_ev_bb"
                    ).replace("_", " "),
                )
        else:
            axes[0, 0].plot(
                evaluation["iteration"],
                evaluation["mean_ev_bb"],
                marker="o",
                label="Mean",
            )
        axes[0, 0].axhline(0.0, color="black", linewidth=0.8)
        axes[0, 0].legend()
    axes[0, 0].set_title("Held-out EV vs fixed opponents")
    axes[0, 0].set_ylabel("BB / hand")

    delta_columns = [
        column
        for column in (
            "benchmark_random_delta_ev_bb",
            "benchmark_calling_station_delta_ev_bb",
            "benchmark_tight_aggressive_delta_ev_bb",
            "benchmark_reference_policy_delta_ev_bb",
        )
        if column in frame
    ]
    if delta_columns:
        evaluation.plot(
            x="iteration",
            y=delta_columns,
            marker="o",
            ax=axes[0, 1],
        )
        axes[0, 1].axhline(0.0, color="black", linewidth=0.8)
    else:
        axes[0, 1].text(
            0.5,
            0.5,
            "No baseline comparison yet",
            ha="center",
            va="center",
        )
    axes[0, 1].set_title("EV change vs initial policy")

    league_columns = [
        column
        for column in ("league_mean_ev_bb", "league_worst_ev_bb")
        if column in frame
    ]
    if league_columns and not evaluation.empty:
        evaluation.plot(
            x="iteration",
            y=league_columns,
            marker="o",
            ax=axes[0, 2],
        )
        axes[0, 2].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 2].set_title("Current policy vs historical league")

    loss_columns = [
        column
        for column in (
            "adv_loss_p0",
            "adv_loss_p1",
            "policy_action_loss_p0",
            "policy_action_loss_p1",
            "policy_range_loss_p0",
            "policy_range_loss_p1",
            "advantage_loss",
            "policy_action_loss",
            "policy_range_loss",
        )
        if column in frame
    ]
    if loss_columns:
        for column in loss_columns:
            positive = frame[column].where(frame[column] > 0)
            axes[1, 0].plot(frame["iteration"], positive, label=column)
        axes[1, 0].set_yscale("log")
        axes[1, 0].legend()
    axes[1, 0].set_title("Advantage, action-policy, and range losses")

    timing_columns = [
        column
        for column in (
            "traversal_seconds",
            "advantage_fit_seconds",
            "policy_fit_seconds",
        )
        if column in frame
    ]
    for column in timing_columns:
        axes[1, 1].plot(frame["iteration"], frame[column], label=column)
    if timing_columns:
        axes[1, 1].legend()
    axes[1, 1].set_title("Phase timing")
    axes[1, 1].set_ylabel("seconds")

    for column in ("nodes", "rollouts"):
        if column in frame:
            axes[1, 2].plot(frame["iteration"], frame[column], label=column)
    axes[1, 2].legend()
    axes[1, 2].set_title("Traversal work")

    buffer_columns = [
        column
        for column in (
            "adv_buffer_p0",
            "adv_buffer_p1",
            "policy_buffer_p0",
            "policy_buffer_p1",
            "advantage_samples",
            "policy_samples",
        )
        if column in frame
    ]
    for column in buffer_columns:
        axes[2, 0].plot(frame["iteration"], frame[column], label=column)
    if buffer_columns:
        axes[2, 0].legend()
    axes[2, 0].set_title("Per-player reservoir growth")

    memory_columns = [
        column
        for column in (
            "gpu_memory_allocated_mb",
            "gpu_memory_reserved_mb",
            "gpu_peak_memory_mb",
        )
        if column in frame
    ]
    for column in memory_columns:
        axes[2, 1].plot(frame["iteration"], frame[column], label=column)
    if memory_columns:
        axes[2, 1].legend()
    axes[2, 1].set_title("CUDA memory")
    axes[2, 1].set_ylabel("MiB")

    health_columns = [
        column
        for column in (
            "mean_abs_regret",
            "depth_cutoffs",
            "node_cutoffs",
            "traversal_nodes_per_second",
            "adv_turnover_events_p0",
            "adv_turnover_events_p1",
            "policy_turnover_events_p0",
            "policy_turnover_events_p1",
        )
        if column in frame
    ]
    for column in health_columns:
        if column in frame:
            axes[2, 2].plot(frame["iteration"], frame[column], label=column)
    if health_columns:
        axes[2, 2].legend()
    axes[2, 2].set_title("Regret, cutoffs, and throughput")

    for axis in axes.flat:
        axis.set_xlabel("iteration")
        axis.grid(alpha=0.2)
    figure.suptitle("Heads-up Deep CFR training")
    figure.tight_layout()
    return figure


def save_training_dashboard(
    metrics: Iterable[dict[str, Any]],
    path: str | Path,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure = plot_training_dashboard(metrics)
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output


def plot_range_dashboard(
    metrics: Iterable[dict[str, Any]],
) -> plt.Figure:
    """Plot fixed-holdout opponent-range quality at each evaluation."""

    frame = metrics_frame(metrics)
    if "range_nll" not in frame:
        raise ValueError("no opponent-range evaluation is available")
    evaluation = frame.dropna(subset=["range_nll"])
    if evaluation.empty:
        raise ValueError("no opponent-range evaluation is available")

    figure, axes = plt.subplots(2, 3, figsize=(19, 10))

    for column in ("range_nll", "range_uniform_nll", "range_entropy"):
        if column in evaluation:
            axes[0, 0].plot(
                evaluation["iteration"], evaluation[column], marker="o",
                label=column.removeprefix("range_").replace("_", " "),
            )
    axes[0, 0].set_title("Exact-combo cross-entropy")
    axes[0, 0].set_ylabel("nats (lower is better)")

    for column in ("range_top10_accuracy", "range_top50_accuracy"):
        if column in evaluation:
            axes[0, 1].plot(
                evaluation["iteration"], evaluation[column], marker="o",
                label=column.removeprefix("range_").replace("_", " "),
            )
    axes[0, 1].set_title("Exact 1,326-combo retrieval")
    axes[0, 1].set_ylim(0.0, 1.0)

    for column in (
        "range_true_probability",
        "range_correct_class_probability",
    ):
        if column in evaluation:
            axes[0, 2].plot(
                evaluation["iteration"], evaluation[column], marker="o",
                label=column.removeprefix("range_").replace("_", " "),
            )
    axes[0, 2].set_title("Probability on truth")
    axes[0, 2].set_ylim(bottom=0.0)

    street_columns = [
        f"range_{street}_information_gain"
        for street in ("preflop", "flop", "turn", "river")
        if f"range_{street}_information_gain" in evaluation
    ]
    for column in street_columns:
        axes[1, 0].plot(
            evaluation["iteration"], evaluation[column], marker="o",
            label=column.removeprefix("range_").removesuffix(
                "_information_gain"
            ),
        )
    axes[1, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 0].set_title("Information gain by street")
    axes[1, 0].set_ylabel("nats vs blocker-uniform")

    opponent_columns = [
        column
        for column in evaluation.columns
        if column.startswith("range_vs_")
        and column.endswith("_information_gain")
    ]
    for column in opponent_columns:
        axes[1, 1].plot(
            evaluation["iteration"], evaluation[column], marker="o",
            label=column.removeprefix("range_vs_").removesuffix(
                "_information_gain"
            ).replace("_", " "),
        )
    axes[1, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 1].set_title("Information gain by opponent")

    for column in (
        "range_information_gain",
        "range_btn_sb_information_gain",
        "range_bb_information_gain",
    ):
        if column in evaluation:
            axes[1, 2].plot(
                evaluation["iteration"], evaluation[column], marker="o",
                label=column.removeprefix("range_").removesuffix(
                    "_information_gain"
                ).replace("_", " "),
            )
    axes[1, 2].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 2].set_title("Overall and position information gain")

    for axis in axes.flat:
        axis.set_xlabel("iteration")
        axis.grid(alpha=0.2)
        handles, _ = axis.get_legend_handles_labels()
        if handles:
            axis.legend()
    figure.suptitle("Opponent-range head: fixed no-leakage holdout")
    figure.tight_layout()
    return figure


def save_range_dashboard(
    metrics: Iterable[dict[str, Any]],
    path: str | Path,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure = plot_range_dashboard(metrics)
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output


def plot_range_reservoir_dashboard(stats: dict[str, Any]) -> plt.Figure:
    """Plot range replay composition, starting hands, and made hands."""

    figure, axes = plt.subplots(1, 3, figsize=(22, 6))
    streets = ("preflop", "flop", "turn", "river")
    street_values = [
        float(stats.get("street_percent", {}).get(name, 0.0))
        for name in streets
    ]
    axes[0].bar(streets, street_values)
    axes[0].set_ylabel("reservoir rows (%)")
    axes[0].set_title("Range reservoir by street")
    axes[0].set_ylim(0.0, max(100.0, max(street_values, default=0.0) * 1.1))

    matrix = np.asarray(
        stats.get("starting_hand_matrix_percent", np.zeros((13, 13))),
        dtype=float,
    )
    image = axes[1].imshow(matrix, cmap="viridis")
    labels = stats.get("rank_labels", list("AKQJT98765432"))
    axes[1].set_xticks(range(13), labels)
    axes[1].set_yticks(range(13), labels)
    axes[1].set_title("Opponent starting hands (% of reservoir)")
    for row in range(13):
        for column in range(13):
            value = matrix[row, column]
            axes[1].text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=6,
                color="white" if value > matrix.max() * 0.45 else "black",
            )
    figure.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)

    made = stats.get("made_hand_percent", {})
    names = list(made)
    positions = np.arange(len(names))
    axes[2].plot(
        positions,
        [float(made[name]) for name in names],
        marker="o",
        linewidth=2,
        label="overall",
    )
    by_street = stats.get("made_hand_by_street_percent", {})
    for street in streets:
        values = by_street.get(street)
        if values:
            axes[2].plot(
                positions,
                [float(values.get(name, 0.0)) for name in names],
                marker="o",
                label=street,
            )
    axes[2].set_xticks(
        positions,
        [name.replace("_", " ") for name in names],
        rotation=45,
        ha="right",
    )
    axes[2].set_ylabel("rows within street (%)")
    axes[2].set_title("Opponent made-hand category by street")
    axes[2].legend()
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.suptitle(
        "Independent sampled-trajectory range reservoir "
        f"({int(stats.get('sampled_rows', 0)):,} sampled / "
        f"{int(stats.get('total_rows', 0)):,} rows)"
    )
    figure.tight_layout()
    return figure


def save_range_reservoir_dashboard(
    stats: dict[str, Any],
    path: str | Path,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure = plot_range_reservoir_dashboard(stats)
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output


__all__ = [
    "metrics_frame",
    "plot_training_dashboard",
    "plot_range_dashboard",
    "plot_range_reservoir_dashboard",
    "save_range_dashboard",
    "save_range_reservoir_dashboard",
    "save_training_dashboard",
]
