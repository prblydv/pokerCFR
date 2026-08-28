"""Frozen evaluation gates for the fresh action-conditioned HU campaign.

Training remains ordinary Deep CFR.  This module only evaluates the resulting
average policy against immutable policy-1025 and 725/950/1025 top-three
benchmarks.  Same-state probability comparisons remove deal/state composition
noise; reciprocal matches preserve common deals while swapping physical seats.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from evaluate_heads_up_ensemble_profitability import (
    EnsembleProvider,
    ProbabilityProvider,
    SnapshotProvider,
    _draw_actions,
    run_reciprocal_match,
    top_k_probabilities,
)
from heads_up_engine import ACTION_ALL_IN, ACTION_NAMES, NUM_ACTIONS
from heads_up_models import legal_mask_offset, masked_softmax
from heads_up_production import (
    ProductionCampaign,
    _atomic_json,
    _json_safe,
    load_policy_snapshot,
)


@dataclass(frozen=True)
class ActionConditionedEvaluationConfig:
    policy_1025_path: str
    ensemble_policy_paths: tuple[str, str, str]
    reciprocal_hands: int = 20_000
    inference_batch_size: int = 2_048
    simulation_batch_size: int = 20_000
    same_state_hands: int = 2_000
    seed: int = 725_950_1025
    reference_exploitability_path: str | None = None

    def validate(self) -> None:
        paths = (self.policy_1025_path, *self.ensemble_policy_paths)
        missing = [value for value in paths if not Path(value).is_file()]
        if missing:
            raise FileNotFoundError(f"frozen benchmark snapshot missing: {missing}")
        if self.reciprocal_hands <= 0 or self.reciprocal_hands % 2:
            raise ValueError("reciprocal_hands must be a positive even number")
        if self.same_state_hands <= 0:
            raise ValueError("same_state_hands must be positive")
        if self.inference_batch_size <= 0 or self.simulation_batch_size <= 0:
            raise ValueError("evaluation batch sizes must be positive")


class LiveTrainerProvider:
    def __init__(self, trainer, name: str = "candidate") -> None:
        self.trainer = trainer
        self.name = name

    @torch.inference_mode()
    def probabilities_batch(self, states: Sequence[Any]) -> torch.Tensor:
        if not states:
            return torch.empty((0, NUM_ACTIONS), dtype=torch.float32)
        return torch.stack(
            self.trainer.average_policy_batch(
                states,
                batch_size=max(1, min(len(states), 8_192)),
            )
        )


def _environment_metadata(trainer) -> dict[str, int]:
    return {
        "starting_stack": int(trainer.env.starting_stack),
        "small_blind": int(trainer.env.small_blind),
        "big_blind": int(trainer.env.big_blind),
    }


def _validate_snapshot_compatibility(trainer, snapshots) -> None:
    expected_environment = _environment_metadata(trainer)
    for snapshot in snapshots:
        metadata = snapshot.metadata
        if dict(metadata["environment"]) != expected_environment:
            raise ValueError("frozen benchmark uses another HU environment")
        if int(metadata["input_dim"]) != int(trainer.input_dim):
            raise ValueError("frozen benchmark uses another HU encoder width")
        if int(metadata["max_history"]) != int(trainer.max_history):
            raise ValueError("frozen benchmark uses another history schema")


@torch.inference_mode()
def _probabilities_from_encoded(
    policy_nets: Sequence[torch.nn.Module],
    xs: torch.Tensor,
    players: torch.Tensor,
    *,
    max_history: int,
    batch_size: int,
) -> torch.Tensor:
    result = torch.zeros((len(xs), NUM_ACTIONS), dtype=torch.float32)
    mask_start = legal_mask_offset(max_history)
    for player in range(2):
        indices = torch.nonzero(players == player, as_tuple=False).flatten()
        network = policy_nets[player]
        device = next(network.parameters()).device
        for start in range(0, len(indices), int(batch_size)):
            batch_indices = indices[start : start + int(batch_size)]
            batch = xs.index_select(0, batch_indices).to(device, dtype=torch.float32)
            masks = batch[:, mask_start : mask_start + NUM_ACTIONS]
            probabilities = masked_softmax(network(batch), masks).cpu()
            result.index_copy_(0, batch_indices, probabilities)
    return result


@torch.inference_mode()
def build_fixed_same_state_holdout(
    trainer,
    policy_1025: SnapshotProvider,
    ensemble_top3: EnsembleProvider,
    *,
    hands: int,
    seed: int,
    inference_batch_size: int,
) -> dict[str, Any]:
    """Generate one immutable decision-state set from the frozen benchmarks."""

    env = type(trainer.env)(
        starting_stack=trainer.env.starting_stack,
        small_blind=trainer.env.small_blind,
        big_blind=trainer.env.big_blind,
        seed=int(seed),
    )
    states = [env.new_hand(button=index % 2) for index in range(int(hands))]
    rngs = [random.Random(int(seed) + 1_000_003 * (index + 1)) for index in range(hands)]
    steps = [0] * len(states)
    encoded: list[torch.Tensor] = []
    players: list[int] = []
    streets: list[int] = []
    while True:
        live = [index for index, state in enumerate(states) if not state.terminal]
        if not live:
            break
        chosen: dict[int, int] = {}
        for start in range(0, len(live), int(inference_batch_size)):
            indices = live[start : start + int(inference_batch_size)]
            provider_groups: tuple[tuple[ProbabilityProvider, list[int]], ...] = (
                (
                    policy_1025,
                    [
                        index
                        for index in indices
                        if (index + int(states[index].to_act)) % 2 == 0
                    ],
                ),
                (
                    ensemble_top3,
                    [
                        index
                        for index in indices
                        if (index + int(states[index].to_act)) % 2 == 1
                    ],
                ),
            )
            for provider, group in provider_groups:
                if not group:
                    continue
                probabilities = provider.probabilities_batch(
                    [states[index] for index in group]
                )
                actions = _draw_actions(
                    probabilities,
                    [rngs[index] for index in group],
                    group,
                )
                chosen.update(zip(group, actions))
        for index in live:
            state = states[index]
            actor = int(state.to_act)
            legal = env.legal_actions(state)
            encoded.append(trainer.encode(state, actor, legal).to(torch.float16))
            players.append(actor)
            streets.append(int(state.street))
            states[index] = env.step(state, int(chosen[index]))
            steps[index] += 1
            if steps[index] > 512:
                raise RuntimeError("same-state holdout hand exceeded 512 actions")
    return {
        "version": 1,
        "kind": "heads_up_action_conditioned_same_state_holdout",
        "seed": int(seed),
        "hands": int(hands),
        "input_dim": int(trainer.input_dim),
        "max_history": int(trainer.max_history),
        "environment": _environment_metadata(trainer),
        "information_states": torch.stack(encoded),
        "players": torch.tensor(players, dtype=torch.uint8),
        "streets": torch.tensor(streets, dtype=torch.uint8),
    }


@torch.inference_mode()
def evaluate_same_state_all_in_probabilities(
    trainer,
    holdout: dict[str, Any],
    *,
    policy_1025_nets: Sequence[torch.nn.Module],
    ensemble_component_nets: Sequence[Sequence[torch.nn.Module]],
    batch_size: int,
) -> dict[str, float]:
    xs = holdout["information_states"]
    players = holdout["players"].to(torch.long)
    streets = holdout["streets"].to(torch.long)
    candidate = _probabilities_from_encoded(
        trainer.policy_nets,
        xs,
        players,
        max_history=trainer.max_history,
        batch_size=batch_size,
    )
    reference = _probabilities_from_encoded(
        policy_1025_nets,
        xs,
        players,
        max_history=trainer.max_history,
        batch_size=batch_size,
    )
    components = [
        _probabilities_from_encoded(
            nets,
            xs,
            players,
            max_history=trainer.max_history,
            batch_size=batch_size,
        )
        for nets in ensemble_component_nets
    ]
    ensemble = top_k_probabilities(torch.stack(components).mean(dim=0), 3)
    mask_start = legal_mask_offset(trainer.max_history)
    all_in_legal = xs[:, mask_start + ACTION_ALL_IN] > 0.5
    if not bool(all_in_legal.any()):
        raise RuntimeError("same-state holdout contains no legal all-in decisions")

    metrics: dict[str, float] = {
        "same_state_decisions": float(len(xs)),
        "same_state_all_in_legal_decisions": float(all_in_legal.sum()),
    }
    values = {
        "candidate": candidate[:, ACTION_ALL_IN],
        "policy1025": reference[:, ACTION_ALL_IN],
        "ensemble_top3": ensemble[:, ACTION_ALL_IN],
    }
    for name, probabilities in values.items():
        metrics[f"same_state_{name}_all_in_probability"] = float(
            probabilities[all_in_legal].mean()
        )
        for street in range(4):
            selected = all_in_legal & (streets == street)
            metrics[f"same_state_{name}_all_in_probability_street_{street}"] = (
                float(probabilities[selected].mean())
                if bool(selected.any())
                else float("nan")
            )
    metrics["same_state_all_in_delta_vs_policy1025"] = (
        metrics["same_state_candidate_all_in_probability"]
        - metrics["same_state_policy1025_all_in_probability"]
    )
    metrics["same_state_all_in_delta_vs_ensemble_top3"] = (
        metrics["same_state_candidate_all_in_probability"]
        - metrics["same_state_ensemble_top3_all_in_probability"]
    )
    return metrics


def _flatten_match(prefix: str, result: dict[str, Any]) -> dict[str, float]:
    hands = int(result["hands"])
    action_counts = result["candidate_action_counts"]
    actions = max(1, sum(int(value) for value in action_counts.values()))
    non_all_in_hands = hands - int(result["candidate_all_in_hands"])
    interval = result["confidence_intervals"]["99"]
    all_in_spr = result["candidate_all_in_spr_after_call"]
    spr_value = lambda name: (
        float(all_in_spr[name])
        if all_in_spr.get(name) is not None
        else float("nan")
    )
    return {
        f"{prefix}_ev_bb_per_100": float(result["mean_ev_bb_per_100"]),
        f"{prefix}_ci99_low_bb_per_100": float(interval["low_bb_per_100"]),
        f"{prefix}_ci99_high_bb_per_100": float(interval["high_bb_per_100"]),
        f"{prefix}_all_in_hand_rate": float(result["candidate_all_in_hand_rate"]),
        f"{prefix}_all_in_action_rate": float(action_counts["all_in"]) / actions,
        f"{prefix}_all_in_net_bb_per_100_all_hands": (
            100.0 * float(result["candidate_all_in_net_bb"]) / hands
        ),
        f"{prefix}_non_all_in_net_bb_per_100_all_hands": (
            100.0 * float(result["candidate_non_all_in_net_bb"]) / hands
        ),
        f"{prefix}_non_all_in_conditional_bb_per_100": (
            100.0 * float(result["candidate_non_all_in_net_bb"])
            / max(1, non_all_in_hands)
        ),
        f"{prefix}_all_in_spr_count": float(all_in_spr["count"]),
        f"{prefix}_all_in_spr_mean": spr_value("mean"),
        f"{prefix}_all_in_spr_median": spr_value("median"),
        f"{prefix}_all_in_spr_p90": spr_value("p90"),
    }


def _exploitability_upper(path: Path) -> float | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    final = payload.get("final", {})
    value = final.get("ci95_high_bb_per_hand")
    return None if value is None else float(value)


def plot_action_conditioned_dashboard(metrics):
    """Plot same-state shove probability, reciprocal EV, and promotion gates."""

    import matplotlib.pyplot as plt
    import pandas as pd

    frame = metrics.copy() if isinstance(metrics, pd.DataFrame) else pd.DataFrame(metrics)
    required = "same_state_candidate_all_in_probability"
    if required not in frame:
        raise ValueError("no action-conditioned evaluations are available")
    evaluated = frame.dropna(subset=[required]).copy()
    if evaluated.empty:
        raise ValueError("no action-conditioned evaluations are available")
    x = evaluated["iteration"]
    figure, axes = plt.subplots(1, 3, figsize=(18, 5))

    for name, label in (
        ("candidate", "candidate"),
        ("policy1025", "policy 1025"),
        ("ensemble_top3", "725/950/1025 top-3"),
    ):
        axes[0].plot(
            x,
            100.0 * evaluated[f"same_state_{name}_all_in_probability"],
            marker="o",
            label=label,
        )
    axes[0].set_title("Same-state all-in probability")
    axes[0].set_ylabel("Probability (%)")
    axes[0].set_xlabel("Iteration")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    for name, label in (
        ("policy1025", "vs policy 1025"),
        ("ensemble_top3", "vs top-3 ensemble"),
    ):
        mean = evaluated[f"promotion_{name}_ev_bb_per_100"]
        low = evaluated[f"promotion_{name}_ci99_low_bb_per_100"]
        high = evaluated[f"promotion_{name}_ci99_high_bb_per_100"]
        axes[1].plot(x, mean, marker="o", label=label)
        axes[1].fill_between(x, low, high, alpha=0.15)
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_title("Reciprocal EV with 99% CI")
    axes[1].set_ylabel("BB/100")
    axes[1].set_xlabel("Iteration")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    gate_names = (
        ("promotion_gate_all_in_controlled", "all-in controlled"),
        ("promotion_gate_positive_overall_ev_99", "overall EV 99%"),
        ("promotion_gate_positive_non_all_in_ev", "non-all-in EV"),
        ("promotion_gate_no_major_exploitability", "exploitability"),
    )
    for name, label in gate_names:
        axes[2].step(x, evaluated[name], where="mid", marker="o", label=label)
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_yticks((0, 1), ("fail/pending", "pass"))
    axes[2].set_title("Promotion evidence gates")
    axes[2].set_xlabel("Iteration")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.25)
    figure.tight_layout()
    return figure


class ActionConditionedProductionCampaign(ProductionCampaign):
    """Production campaign with strict, frozen promotion evidence."""

    def __init__(
        self,
        trainer,
        artifact_dir: str | Path,
        config,
        evaluation_config: ActionConditionedEvaluationConfig,
    ) -> None:
        if trainer.range_training_enabled:
            raise ValueError("action-conditioned campaign must not train a range head")
        evaluation_config.validate()
        self.action_evaluation_config = evaluation_config
        super().__init__(trainer, artifact_dir, config)
        self.policy_1025 = load_policy_snapshot(
            evaluation_config.policy_1025_path,
            device=trainer.device,
        )
        self.ensemble_snapshots = tuple(
            load_policy_snapshot(path, device=trainer.device)
            for path in evaluation_config.ensemble_policy_paths
        )
        _validate_snapshot_compatibility(
            trainer,
            (self.policy_1025, *self.ensemble_snapshots),
        )
        self.policy_1025_provider = SnapshotProvider(
            trainer, self.policy_1025, name="policy1025"
        )
        self.ensemble_top3_provider = EnsembleProvider(
            trainer,
            self.ensemble_snapshots,
            top_k=3,
            name="ensemble_725_950_1025_top3",
        )
        self.same_state_holdout_path = (
            self.evaluation_dir / "same_state_holdout_v1.pt"
        )
        self._same_state_holdout: dict[str, Any] | None = None

    def _load_or_create_same_state_holdout(self) -> dict[str, Any]:
        expected = {
            "version": 1,
            "kind": "heads_up_action_conditioned_same_state_holdout",
            "seed": int(self.action_evaluation_config.seed),
            "hands": int(self.action_evaluation_config.same_state_hands),
            "input_dim": int(self.trainer.input_dim),
            "max_history": int(self.trainer.max_history),
            "environment": _environment_metadata(self.trainer),
        }
        if self._same_state_holdout is None:
            if self.same_state_holdout_path.exists():
                holdout = torch.load(
                    self.same_state_holdout_path,
                    map_location="cpu",
                    weights_only=False,
                )
                actual = {key: holdout.get(key) for key in expected}
                if actual != expected:
                    raise ValueError("same-state holdout does not match campaign")
            else:
                holdout = build_fixed_same_state_holdout(
                    self.trainer,
                    self.policy_1025_provider,
                    self.ensemble_top3_provider,
                    hands=self.action_evaluation_config.same_state_hands,
                    seed=self.action_evaluation_config.seed,
                    inference_batch_size=(
                        self.action_evaluation_config.inference_batch_size
                    ),
                )
                temporary = self.same_state_holdout_path.with_suffix(".pt.tmp")
                torch.save(holdout, temporary)
                temporary.replace(self.same_state_holdout_path)
            self._same_state_holdout = holdout
        return self._same_state_holdout

    def _extend_evaluation(
        self,
        row: dict[str, Any],
        iteration_dir: Path,
    ) -> None:
        settings = self.action_evaluation_config
        same_state = evaluate_same_state_all_in_probabilities(
            self.trainer,
            self._load_or_create_same_state_holdout(),
            policy_1025_nets=self.policy_1025.policy_nets,
            ensemble_component_nets=[
                snapshot.policy_nets for snapshot in self.ensemble_snapshots
            ],
            batch_size=settings.inference_batch_size,
        )
        row.update(same_state)
        candidate = LiveTrainerProvider(
            self.trainer, name=f"candidate_{self.trainer.iteration}"
        )
        environment = _environment_metadata(self.trainer)
        results = {}
        for index, (name, opponent) in enumerate(
            (
                ("policy1025", self.policy_1025_provider),
                ("ensemble_top3", self.ensemble_top3_provider),
            )
        ):
            result = run_reciprocal_match(
                candidate,
                lambda _env, provider=opponent: provider,
                environment=environment,
                hands=settings.reciprocal_hands,
                seed=settings.seed + index * 100_000_007,
                inference_batch_size=settings.inference_batch_size,
                simulation_batch_size=settings.simulation_batch_size,
            )
            results[name] = result
            row.update(_flatten_match(f"promotion_{name}", result))
        _atomic_json(
            iteration_dir / "action_conditioned_benchmarks.json",
            _json_safe(
                {"same_state": same_state, "reciprocal_matches": results}
            ),
        )

        row["promotion_gate_all_in_controlled"] = float(
            same_state["same_state_candidate_all_in_probability"]
            <= same_state["same_state_policy1025_all_in_probability"]
            and same_state["same_state_candidate_all_in_probability"]
            <= same_state["same_state_ensemble_top3_all_in_probability"]
        )
        row["promotion_gate_positive_overall_ev_99"] = float(
            all(
                row[f"promotion_{name}_ci99_low_bb_per_100"] > 0.0
                for name in ("policy1025", "ensemble_top3")
            )
        )
        row["promotion_gate_positive_non_all_in_ev"] = float(
            all(
                row[f"promotion_{name}_non_all_in_net_bb_per_100_all_hands"]
                > 0.0
                for name in ("policy1025", "ensemble_top3")
            )
        )

        candidate_reports = sorted(
            (iteration_dir / "exploitability").glob("policy_*_lbr.json")
        )
        candidate_upper = (
            _exploitability_upper(candidate_reports[-1])
            if candidate_reports
            else None
        )
        reference_path = (
            Path(settings.reference_exploitability_path)
            if settings.reference_exploitability_path
            else None
        )
        reference_upper = (
            _exploitability_upper(reference_path)
            if reference_path is not None
            else None
        )
        row["promotion_candidate_lbr_ci95_high_bb_per_hand"] = (
            candidate_upper if candidate_upper is not None else float("nan")
        )
        row["promotion_reference_lbr_ci95_high_bb_per_hand"] = (
            reference_upper if reference_upper is not None else float("nan")
        )
        row["promotion_gate_no_major_exploitability"] = float(
            candidate_upper is not None
            and reference_upper is not None
            and candidate_upper <= reference_upper
        )
        row["promotion_all_required_gates"] = float(
            all(
                row[name] > 0.5
                for name in (
                    "promotion_gate_all_in_controlled",
                    "promotion_gate_positive_overall_ev_99",
                    "promotion_gate_positive_non_all_in_ev",
                    "promotion_gate_no_major_exploitability",
                )
            )
        )

    def _promotion_allowed(
        self,
        row: dict[str, Any],
        previous_lcb: float,
    ) -> bool:
        return bool(
            row.get("promotion_all_required_gates", 0.0) > 0.5
            and super()._promotion_allowed(row, previous_lcb)
        )


__all__ = [
    "ActionConditionedEvaluationConfig",
    "ActionConditionedProductionCampaign",
    "LiveTrainerProvider",
    "build_fixed_same_state_holdout",
    "evaluate_same_state_all_in_probabilities",
    "plot_action_conditioned_dashboard",
]
