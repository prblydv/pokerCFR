"""Create exact 169-hand critical-condition charts for one HU policy snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
import numpy as np
import torch

from heads_up_analysis import (
    RANK_LABELS,
    StrategyAnalyzer,
    plot_range_heatmaps,
    postflop_scenarios,
    preflop_scenarios,
)
from heads_up_cfr import HeadsUpNeuralCFR
from heads_up_engine import ACTION_NAMES
from heads_up_native import HeadsUpHoldemEngine
from heads_up_production import load_policy_snapshot


DEFAULT_POLICY = Path("artifacts/downloaded_risk_aware/policy_00000200.pt")
DEFAULT_OUTPUT = Path(
    "artifacts/downloaded_risk_aware/policy_00000200_critical_conditions"
)
ACTION_COLUMNS = tuple(f"p_{name}" for name in ACTION_NAMES)


def card(text: str) -> int:
    ranks = "23456789TJQKA"
    suits = "cdhs"
    return suits.index(text[1].lower()) * 13 + ranks.index(text[0].upper())


def metric_matrix(report, metric: str) -> np.ndarray:
    matrix = np.full((13, 13), np.nan, dtype=float)
    for row in report.hand_table.itertuples(index=False):
        matrix[int(row.row), int(row.column)] = float(getattr(row, metric))
    return matrix


def plot_metric_grid(reports, metric: str, title: str, output: Path) -> None:
    figure, axes = plt.subplots(
        2, 2, figsize=(13.5, 11.5), constrained_layout=True
    )
    image = None
    for axis, report in zip(axes.ravel(), reports):
        image = axis.imshow(
            metric_matrix(report, metric),
            vmin=0.0,
            vmax=1.0,
            cmap="magma",
        )
        axis.set_xticks(range(13), RANK_LABELS)
        axis.set_yticks(range(13), RANK_LABELS)
        axis.set_title(report.scenario.label, fontsize=10)
        axis.set_xlabel("suited above diagonal; offsuit below")
    if image is not None:
        figure.colorbar(
            image,
            ax=axes.ravel().tolist(),
            shrink=0.78,
            format=PercentFormatter(1.0),
            label="policy probability",
        )
    figure.suptitle(title, fontsize=15)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_dominant_action_grid(reports, title: str, output: Path) -> None:
    colors = (
        "#d73027",
        "#8c8c8c",
        "#4575b4",
        "#91bfdb",
        "#74add1",
        "#abd9e9",
        "#fee090",
        "#fdae61",
        "#f46d43",
        "#7b3294",
    )
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, len(ACTION_NAMES) + 0.5), cmap.N)
    figure, axes = plt.subplots(2, 2, figsize=(14, 12))
    seen: set[int] = set()
    for axis, report in zip(axes.ravel(), reports):
        matrix = np.full((13, 13), np.nan, dtype=float)
        for row in report.hand_table.itertuples(index=False):
            probabilities = np.asarray(
                [float(getattr(row, column)) for column in ACTION_COLUMNS]
            )
            action = int(np.nanargmax(probabilities))
            matrix[int(row.row), int(row.column)] = action
            seen.add(action)
        axis.imshow(matrix, cmap=cmap, norm=norm)
        axis.set_xticks(range(13), RANK_LABELS)
        axis.set_yticks(range(13), RANK_LABELS)
        axis.set_title(report.scenario.label, fontsize=10)
        axis.set_xlabel("suited above diagonal; offsuit below")
    handles = [
        Patch(facecolor=colors[action], label=ACTION_NAMES[action])
        for action in sorted(seen)
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=min(5, max(1, len(handles))),
        frameon=False,
    )
    figure.suptitle(title, fontsize=15)
    figure.subplots_adjust(top=0.92, bottom=0.1, wspace=0.2, hspace=0.25)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_action_mix(reports, title: str, output: Path) -> None:
    mixes = []
    for report in reports:
        row = []
        for column in ACTION_COLUMNS:
            value = float(report.combo_table[column].mean())
            row.append(value if np.isfinite(value) else 0.0)
        mixes.append(row)
    values = np.asarray(mixes)
    figure, axis = plt.subplots(figsize=(14, 6.5))
    bottom = np.zeros(len(reports), dtype=float)
    palette = plt.get_cmap("tab10").colors
    for action, name in enumerate(ACTION_NAMES):
        axis.bar(
            range(len(reports)),
            values[:, action],
            bottom=bottom,
            label=name,
            color=palette[action],
        )
        bottom += values[:, action]
    axis.set_xticks(
        range(len(reports)),
        [report.scenario.label for report in reports],
        rotation=18,
        ha="right",
    )
    axis.set_ylim(0.0, 1.0)
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_ylabel("combination-weighted policy probability")
    axis.set_title(title)
    axis.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.23))
    figure.subplots_adjust(bottom=0.34)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def build_trainer(snapshot, device: str, seed: int) -> HeadsUpNeuralCFR:
    metadata = snapshot.metadata
    environment = metadata["environment"]
    env = HeadsUpHoldemEngine(
        starting_stack=int(environment["starting_stack"]),
        small_blind=int(environment["small_blind"]),
        big_blind=int(environment["big_blind"]),
        seed=seed,
    )
    return HeadsUpNeuralCFR(
        env,
        device=device,
        hidden=int(metadata["hidden"]),
        blocks=int(metadata["blocks"]),
        advantage_capacity=1,
        policy_capacity=1,
        range_capacity=1,
        max_history=int(metadata["max_history"]),
        network_architecture=str(metadata["network_architecture"]),
        policy_network_architecture=str(metadata["policy_network_architecture"]),
        encoder_schema_version=str(metadata["encoder_schema_version"]),
        enable_range_training=False,
        range_loss_weight=0.0,
        seed=seed,
    )


def report_summary(reports) -> list[dict]:
    rows = []
    for report in reports:
        table = report.hand_table.sort_values("p_all_in", ascending=False)
        mix = {}
        for name in ACTION_NAMES:
            value = float(report.combo_table[f"p_{name}"].mean())
            mix[name] = value if np.isfinite(value) else 0.0
        rows.append(
            {
                **report.state_summary,
                "combination_weighted_action_mix": mix,
                "top_all_in_hand_classes": [
                    {
                        "hand": str(row.hand),
                        "probability": float(row.p_all_in),
                    }
                    for row in table.head(12).itertuples(index=False)
                ],
            }
        )
    return rows


def write_markdown(path: Path, iteration: int, summaries: list[dict]) -> None:
    lines = [
        f"# Policy {iteration} critical-condition report",
        "",
        "These are average-policy action probabilities from exact legal states, not hand equity and not claims of equilibrium play.",
        "",
        "## Scenario summaries",
        "",
    ]
    for row in summaries:
        mix = row["combination_weighted_action_mix"]
        leaders = sorted(mix.items(), key=lambda item: item[1], reverse=True)[:4]
        top_shoves = row["top_all_in_hand_classes"][:5]
        lines.extend(
            [
                f"### {row['label']}",
                "",
                f"Pot {row['pot_bb']:.2f} BB, call {row['to_call_bb']:.2f} BB, SPR {row['spr']:.2f}.",
                "",
                "Largest action masses: "
                + ", ".join(f"{name} {100 * value:.1f}%" for name, value in leaders)
                + ".",
                "",
                "Highest all-in classes: "
                + ", ".join(
                    f"{item['hand']} {100 * item['probability']:.1f}%"
                    for item in top_shoves
                )
                + ".",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    snapshot = load_policy_snapshot(args.policy, device=args.device)
    trainer = build_trainer(snapshot, args.device, seed=200_169)
    analyzer = StrategyAnalyzer(trainer, batch_size=args.batch_size)

    preflop = analyzer.analyze_cases(
        preflop_scenarios(), policy_nets=snapshot.policy_nets
    )
    dry_board = tuple(card(value) for value in ("As", "7d", "2c"))
    all_postflop = postflop_scenarios(
        flop=dry_board,
        turn=card("Qh"),
        river=card("3s"),
    )
    selected_ids = (
        "btn_flop_checked_to",
        "bb_flop_vs_halfpot",
        "btn_turn_vs_halfpot",
        "bb_river_vs_btn_pot",
    )
    postflop_scenario_map = {
        scenario.scenario_id: scenario for scenario in all_postflop
    }
    postflop = analyzer.analyze_cases(
        [postflop_scenario_map[name] for name in selected_ids],
        policy_nets=snapshot.policy_nets,
    )
    reports = [*preflop, *postflop]

    for report in reports:
        report.hand_table.to_csv(
            args.output / f"{report.scenario.scenario_id}_169.csv", index=False
        )
        figure = plot_range_heatmaps(
            report,
            metrics=("p_fold", "p_check", "p_call", "p_aggressive", "p_all_in"),
        )
        figure.savefig(
            args.output / f"{report.scenario.scenario_id}_full_actions.png",
            dpi=160,
            bbox_inches="tight",
        )
        plt.close(figure)

    plot_metric_grid(
        preflop,
        "p_all_in",
        f"Policy {snapshot.iteration}: preflop all-in probability",
        args.output / "preflop_all_in_heatmaps.png",
    )
    plot_metric_grid(
        preflop,
        "p_continue",
        f"Policy {snapshot.iteration}: preflop continue probability",
        args.output / "preflop_continue_heatmaps.png",
    )
    plot_dominant_action_grid(
        preflop,
        f"Policy {snapshot.iteration}: dominant preflop action",
        args.output / "preflop_dominant_action_heatmaps.png",
    )
    plot_action_mix(
        preflop,
        f"Policy {snapshot.iteration}: preflop action mix",
        args.output / "preflop_action_mix.png",
    )
    plot_metric_grid(
        postflop,
        "p_all_in",
        f"Policy {snapshot.iteration}: postflop all-in probability",
        args.output / "postflop_all_in_heatmaps.png",
    )
    plot_dominant_action_grid(
        postflop,
        f"Policy {snapshot.iteration}: dominant postflop action",
        args.output / "postflop_dominant_action_heatmaps.png",
    )
    plot_action_mix(
        postflop,
        f"Policy {snapshot.iteration}: postflop action mix",
        args.output / "postflop_action_mix.png",
    )

    summaries = report_summary(reports)
    with args.policy.open("rb") as stream:
        policy_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    payload = {
        "policy": str(args.policy.resolve()),
        "iteration": int(snapshot.iteration),
        "sha256": policy_hash,
        "scope": "controlled exact-engine strategy fingerprint; probabilities are not equity",
        "scenarios": summaries,
    }
    (args.output / "critical_condition_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    write_markdown(
        args.output / "REPORT.md", int(snapshot.iteration), summaries
    )
    print(json.dumps({"output": str(args.output), "iteration": snapshot.iteration, "scenarios": len(reports)}))


if __name__ == "__main__":
    main()
