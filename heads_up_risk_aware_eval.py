"""Evaluation and plotting for the fresh risk-aware HU Deep CFR campaign."""

from __future__ import annotations

from heads_up_action_conditioned_eval import (
    ActionConditionedEvaluationConfig,
    ActionConditionedProductionCampaign,
    LiveTrainerProvider,
)


RiskAwareEvaluationConfig = ActionConditionedEvaluationConfig


class RiskAwareProductionCampaign(ActionConditionedProductionCampaign):
    """No-range production campaign with frozen reciprocal promotion gates."""

    def __init__(self, trainer, artifact_dir, config, evaluation_config) -> None:
        if not trainer.risk_aware_all_in:
            raise ValueError("risk-aware campaign requires risk_aware_all_in=True")
        if not trainer.robust_advantage_loss:
            raise ValueError("risk-aware campaign requires Smooth-L1 advantage loss")
        if not trainer.fit_reservoir_once_per_iteration:
            raise ValueError("risk-aware campaign requires one-pass reservoir fitting")
        if (
            not trainer.reinitialize_advantage_each_iteration
            or trainer.advantage_reinitialize_from_iteration != 25
            or trainer.advantage_reinitialize_cycle != 1
        ):
            raise ValueError(
                "risk-aware campaign requires fresh advantage fitting at every "
                "iteration from 25 onward"
            )
        super().__init__(trainer, artifact_dir, config, evaluation_config)


def plot_all_in_spr_trend(metrics):
    """Plot SPR exclusively at decisions where the candidate chose all-in."""

    import matplotlib.pyplot as plt
    import pandas as pd

    frame = metrics.copy() if isinstance(metrics, pd.DataFrame) else pd.DataFrame(metrics)
    columns = [
        "promotion_policy1025_all_in_spr_median",
        "promotion_ensemble_top3_all_in_spr_median",
    ]
    available = [name for name in columns if name in frame]
    if not available:
        raise ValueError("no all-in SPR evaluations are available")
    evaluated = frame.dropna(subset=available, how="all").copy()
    if evaluated.empty:
        raise ValueError("no all-in SPR evaluations are available")

    figure, axis = plt.subplots(figsize=(11, 6))
    for prefix, label in (
        ("promotion_policy1025", "vs policy 1025"),
        ("promotion_ensemble_top3", "vs 725/950/1025 top-3"),
    ):
        median = f"{prefix}_all_in_spr_median"
        p90 = f"{prefix}_all_in_spr_p90"
        if median not in evaluated:
            continue
        axis.plot(
            evaluated["iteration"],
            evaluated[median],
            marker="o",
            label=f"{label} median",
        )
        if p90 in evaluated:
            axis.plot(
                evaluated["iteration"],
                evaluated[p90],
                linestyle="--",
                alpha=0.75,
                label=f"{label} p90",
            )
    axis.axhline(2.0, color="black", linewidth=1, linestyle=":", label="SPR 2")
    axis.set_title("SPR only when the candidate chooses all-in")
    axis.set_xlabel("Iteration")
    axis.set_ylabel("SPR after matching the current bet")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    return figure


__all__ = [
    "LiveTrainerProvider",
    "RiskAwareEvaluationConfig",
    "RiskAwareProductionCampaign",
    "plot_all_in_spr_trend",
]
