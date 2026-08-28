"""Resumable evolutionary search for a weighted HU policy ensemble.

The optimizer changes only how already-trained average-policy probabilities are
weighted.  Cards, engine transitions, legal actions, top-k filtering, and
reciprocal evaluation semantics remain unchanged.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from evaluate_heads_up_ensemble_profitability import (
    ScriptedProvider,
    SnapshotProvider,
    run_reciprocal_match,
    top_k_probabilities,
)
from heads_up_cfr import HeadsUpNeuralCFR
from heads_up_engine import NUM_ACTIONS
from heads_up_models import masked_softmax
from heads_up_native import HeadsUpHoldemEngine
from heads_up_production import OPPONENT_PROFILES, load_policy_snapshot


DEFAULT_MODELS = (
    Path("artifacts/heads_up_v4_paper3x/snapshots/policy_00000725.pt"),
    Path("artifacts/heads_up_v4_paper3x/snapshots/policy_00000950.pt"),
    Path("artifacts/heads_up_v4_paper3x/snapshots/policy_00001025.pt"),
    Path("artifacts/downloaded_risk_aware/policy_00000200.pt"),
    Path("artifacts/downloaded_risk_aware/policy_00000275.pt"),
    Path("artifacts/downloaded_risk_aware/policy_00000300.pt"),
    Path("artifacts/downloaded_risk_aware/policy_00000400.pt"),
)
DEFAULT_FITNESS_LEAGUE = (
    "old3_top3",
    "tag",
)
DEFAULT_REPORT_OPPONENTS = ("random",)
STATE_VERSION = 1


@dataclass(frozen=True)
class FitnessSettings:
    uncertainty_z: float = 1.0
    non_all_in_loss_penalty: float = 0.50
    max_all_in_positive_share: float = 0.70
    all_in_concentration_penalty: float = 0.50
    worst_opponent_loss_penalty: float = 0.20


class WeightedEnsembleProvider:
    """Weighted average of compatible snapshot policies with legal top-k."""

    def __init__(
        self,
        trainer: HeadsUpNeuralCFR,
        snapshots: Sequence[Any],
        weights: Sequence[float],
        *,
        top_k: int,
        name: str,
    ) -> None:
        if len(snapshots) != len(weights) or not snapshots:
            raise ValueError("snapshots and weights must have the same nonzero length")
        positive = [
            (snapshot, float(weight))
            for snapshot, weight in zip(snapshots, weights)
            if float(weight) > 0.0
        ]
        if not positive:
            raise ValueError("weighted ensemble needs positive total weight")
        total = sum(weight for _, weight in positive)
        self.trainer = trainer
        self.snapshots = tuple(snapshot for snapshot, _ in positive)
        self.weights = tuple(weight / total for _, weight in positive)
        self.top_k = int(top_k)
        self.name = str(name)

    @torch.inference_mode()
    def probabilities_batch(self, states: Sequence[Any]) -> torch.Tensor:
        if not states:
            return torch.empty((0, NUM_ACTIONS), dtype=torch.float32)
        outputs: list[torch.Tensor | None] = [None] * len(states)
        grouped: list[list[tuple[int, torch.Tensor, torch.Tensor]]] = [[], []]
        for index, state in enumerate(states):
            if state.terminal or state.to_act is None:
                raise ValueError("every weighted-policy state must be live")
            player = int(state.to_act)
            legal = self.trainer.env.legal_actions(state)
            mask = torch.zeros(NUM_ACTIONS, dtype=torch.float32)
            mask[list(legal)] = 1.0
            grouped[player].append(
                (index, self.trainer.encode(state, player, legal), mask)
            )
        for player, items in enumerate(grouped):
            if not items:
                continue
            network_device = next(
                self.snapshots[0].policy_nets[player].parameters()
            ).device
            encoded = torch.stack([item[1] for item in items]).to(
                network_device, non_blocking=True
            )
            masks = torch.stack([item[2] for item in items]).to(
                network_device, non_blocking=True
            )
            weighted = torch.zeros(
                (len(items), NUM_ACTIONS),
                dtype=torch.float32,
                device=network_device,
            )
            for snapshot, weight in zip(self.snapshots, self.weights):
                network = snapshot.policy_nets[player]
                network.eval()
                weighted.add_(masked_softmax(network(encoded), masks), alpha=float(weight))
            weighted /= weighted.sum(dim=1, keepdim=True)
            weighted = top_k_probabilities(weighted, self.top_k).cpu()
            for (index, _, _), probabilities in zip(items, weighted):
                outputs[index] = probabilities
        if any(output is None for output in outputs):
            raise RuntimeError("failed to evaluate one or more weighted-policy states")
        return torch.stack([output for output in outputs if output is not None])


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_event(path: Path, event: str, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"time_utc": utc_now(), "event": event, **payload}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, separators=(",", ":")) + "\n")


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def candidate_id(weights: Sequence[float]) -> str:
    text = ",".join(f"{float(weight):.10f}" for weight in weights)
    return hashlib.sha256(text.encode("ascii")).hexdigest()[:12]


def normalize_weights(
    weights: Sequence[float],
    *,
    min_active: int,
    max_active: int,
    min_weight: float,
    rng: np.random.Generator,
) -> np.ndarray:
    values = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    if values.ndim != 1:
        raise ValueError("weights must be one-dimensional")
    if np.count_nonzero(values) > max_active:
        keep = np.argsort(values)[-max_active:]
        mask = np.zeros_like(values, dtype=bool)
        mask[keep] = True
        values[~mask] = 0.0
    values[values < float(min_weight)] = 0.0
    while np.count_nonzero(values) < min_active:
        available = np.flatnonzero(values == 0.0)
        if not len(available):
            break
        values[int(rng.choice(available))] = max(float(min_weight), 0.05)
    total = float(values.sum())
    if total <= 0.0:
        chosen = rng.choice(len(values), size=min_active, replace=False)
        values[chosen] = 1.0
        total = float(values.sum())
    values /= total
    return values


def initial_population(
    iterations: Sequence[int],
    *,
    seeded_weights: Sequence[Sequence[float]] = (),
    population_size: int,
    min_active: int,
    max_active: int,
    min_weight: float,
    seed: int,
) -> list[list[float]]:
    count = len(iterations)
    rng = np.random.default_rng(seed)
    seeds: list[np.ndarray] = [
        np.asarray(weights, dtype=np.float64) for weights in seeded_weights
    ]
    old_indices = [
        iterations.index(iteration)
        for iteration in (725, 950, 1025)
        if iteration in iterations
    ]
    if len(old_indices) >= min_active:
        base = np.zeros(count)
        base[old_indices] = 1.0 / len(old_indices)
        seeds.append(base)
        for index in range(count):
            if index in old_indices or len(old_indices) + 1 > max_active:
                continue
            mixed = np.zeros(count)
            mixed[[*old_indices, index]] = 1.0 / (len(old_indices) + 1)
            seeds.append(mixed)
    for left in range(count):
        for right in range(left + 1, count):
            if len(seeds) >= population_size:
                break
            pair = np.zeros(count)
            pair[[left, right]] = 0.5
            seeds.append(pair)
    population: list[list[float]] = []
    seen: set[str] = set()
    for values in seeds:
        values = normalize_weights(
            values,
            min_active=min_active,
            max_active=max_active,
            min_weight=min_weight,
            rng=rng,
        )
        key = candidate_id(values)
        if key not in seen:
            population.append(values.tolist())
            seen.add(key)
        if len(population) >= population_size:
            return population
    while len(population) < population_size:
        active_count = int(rng.integers(min_active, max_active + 1))
        active = rng.choice(count, size=active_count, replace=False)
        values = np.zeros(count)
        values[active] = rng.dirichlet(np.ones(active_count))
        values = normalize_weights(
            values,
            min_active=min_active,
            max_active=max_active,
            min_weight=min_weight,
            rng=rng,
        )
        key = candidate_id(values)
        if key not in seen:
            population.append(values.tolist())
            seen.add(key)
    return population


def evolve_population(
    ranked: Sequence[dict[str, Any]],
    *,
    population_size: int,
    elite_count: int,
    min_active: int,
    max_active: int,
    min_weight: float,
    mutation_scale: float,
    structural_mutation_probability: float,
    random_immigrants: int,
    seed: int,
) -> list[list[float]]:
    rng = np.random.default_rng(seed)
    elites = [np.asarray(item["weights"], dtype=np.float64) for item in ranked[:elite_count]]
    population = [values.tolist() for values in elites]
    seen = {candidate_id(values) for values in elites}

    def tournament() -> np.ndarray:
        size = min(3, len(ranked))
        choices = rng.choice(len(ranked), size=size, replace=False)
        winner = min(choices, key=lambda index: -float(ranked[int(index)]["fitness"]))
        return np.asarray(ranked[int(winner)]["weights"], dtype=np.float64)

    child_target = population_size - int(random_immigrants)
    attempts = 0
    while len(population) < child_target:
        attempts += 1
        first, second = tournament(), tournament()
        alpha = float(rng.uniform(0.15, 0.85))
        child = alpha * first + (1.0 - alpha) * second
        active = child > 0.0
        if np.any(active):
            child[active] *= np.exp(rng.normal(0.0, mutation_scale, size=int(active.sum())))
        if rng.random() < structural_mutation_probability:
            if int(np.count_nonzero(child)) < max_active and rng.random() < 0.55:
                available = np.flatnonzero(child == 0.0)
                if len(available):
                    child[int(rng.choice(available))] = float(rng.uniform(0.04, 0.20))
            elif int(np.count_nonzero(child)) > min_active:
                active_indices = np.flatnonzero(child > 0.0)
                child[int(rng.choice(active_indices))] = 0.0
        child = normalize_weights(
            child,
            min_active=min_active,
            max_active=max_active,
            min_weight=min_weight,
            rng=rng,
        )
        key = candidate_id(child)
        if key not in seen:
            population.append(child.tolist())
            seen.add(key)
        elif attempts > population_size * 100:
            child = normalize_weights(
                rng.random(len(child)),
                min_active=min_active,
                max_active=max_active,
                min_weight=min_weight,
                rng=rng,
            )
            key = candidate_id(child)
            if key not in seen:
                population.append(child.tolist())
                seen.add(key)
    while len(population) < population_size:
        active_count = int(rng.integers(min_active, max_active + 1))
        active = rng.choice(len(elites[0]), size=active_count, replace=False)
        immigrant = np.zeros(len(elites[0]), dtype=np.float64)
        immigrant[active] = rng.dirichlet(np.ones(active_count))
        immigrant = normalize_weights(
            immigrant,
            min_active=min_active,
            max_active=max_active,
            min_weight=min_weight,
            rng=rng,
        )
        key = candidate_id(immigrant)
        if key not in seen:
            population.append(immigrant.tolist())
            seen.add(key)
    return population


def score_matches(
    matches: Sequence[dict[str, Any]], settings: FitnessSettings
) -> dict[str, float]:
    if not matches:
        raise ValueError("fitness requires at least one match")
    evs = [float(match["mean_ev_bb_per_100"]) for match in matches]
    stderrs = [100.0 * float(match["paired_stderr_bb_per_hand"]) for match in matches]
    all_in_evs = [
        100.0 * float(match["candidate_all_in_net_bb"]) / int(match["hands"])
        for match in matches
    ]
    non_all_in_evs = [
        100.0 * float(match["candidate_non_all_in_net_bb"]) / int(match["hands"])
        for match in matches
    ]
    average_ev = mean(evs)
    combined_stderr = math.sqrt(sum(value * value for value in stderrs)) / len(stderrs)
    all_in_ev = mean(all_in_evs)
    non_all_in_ev = mean(non_all_in_evs)
    worst_ev = min(evs)
    positive_all_in = max(0.0, all_in_ev)
    positive_non_all_in = max(0.0, non_all_in_ev)
    positive_components = positive_all_in + positive_non_all_in
    all_in_share = (
        positive_all_in / positive_components if positive_components > 0.0 else 0.0
    )
    uncertainty_penalty = settings.uncertainty_z * combined_stderr
    non_all_in_penalty = (
        settings.non_all_in_loss_penalty * max(0.0, -non_all_in_ev)
    )
    concentration_excess = max(
        0.0, all_in_share - settings.max_all_in_positive_share
    )
    concentration_penalty = (
        settings.all_in_concentration_penalty
        * positive_components
        * concentration_excess
    )
    worst_penalty = (
        settings.worst_opponent_loss_penalty * max(0.0, -worst_ev)
    )
    fitness = (
        average_ev
        - uncertainty_penalty
        - non_all_in_penalty
        - concentration_penalty
        - worst_penalty
    )
    return {
        "fitness": fitness,
        "average_ev_bb_per_100": average_ev,
        "combined_stderr_bb_per_100": combined_stderr,
        "all_in_ev_bb_per_100": all_in_ev,
        "non_all_in_ev_bb_per_100": non_all_in_ev,
        "worst_opponent_ev_bb_per_100": worst_ev,
        "all_in_positive_profit_share": all_in_share,
        "uncertainty_penalty": uncertainty_penalty,
        "non_all_in_loss_penalty": non_all_in_penalty,
        "all_in_concentration_penalty": concentration_penalty,
        "worst_opponent_loss_penalty": worst_penalty,
    }


def write_population_csv(
    path: Path, results: Sequence[dict[str, Any]], labels: Sequence[str]
) -> None:
    fields = [
        "rank",
        "candidate_id",
        "fitness",
        "average_ev_bb_per_100",
        "combined_stderr_bb_per_100",
        "all_in_ev_bb_per_100",
        "non_all_in_ev_bb_per_100",
        "worst_opponent_ev_bb_per_100",
        "all_in_positive_profit_share",
        "elapsed_seconds",
        *[f"weight_{label}" for label in labels],
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for rank, result in enumerate(results, 1):
            row = {key: result.get(key) for key in fields if not key.startswith("weight_")}
            row["rank"] = rank
            for label, weight in zip(labels, result["weights"]):
                row[f"weight_{label}"] = weight
            writer.writerow(row)


def plot_history(
    output: Path,
    generations: Sequence[dict[str, Any]],
    labels: Sequence[str],
) -> None:
    if not generations:
        return
    graph_dir = output / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)
    indices = [int(item["generation"]) for item in generations]
    best = [item["ranked"][0] for item in generations]
    best_fitness = [float(item["fitness"]) for item in best]
    median_fitness = [median(float(row["fitness"]) for row in item["ranked"]) for item in generations]
    best_ev = [float(item["average_ev_bb_per_100"]) for item in best]

    fig, axis = plt.subplots(figsize=(10, 5.5))
    axis.plot(indices, best_fitness, marker="o", label="best fitness")
    axis.plot(indices, median_fitness, marker=".", label="median fitness")
    axis.plot(indices, best_ev, marker="s", label="best raw EV")
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set(xlabel="generation", ylabel="BB/100", title="Evolution progress")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(graph_dir / "fitness_and_ev_by_generation.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 5.5))
    for item in generations:
        generation = int(item["generation"])
        axis.scatter(
            [generation] * len(item["ranked"]),
            [float(row["fitness"]) for row in item["ranked"]],
            alpha=0.65,
            s=24,
        )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set(xlabel="generation", ylabel="fitness (BB/100)", title="Every candidate in every generation")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(graph_dir / "population_fitness_scatter.png", dpi=180)
    plt.close(fig)

    weights = np.asarray([item["weights"] for item in best], dtype=np.float64)
    fig, axis = plt.subplots(figsize=(11, 6))
    axis.stackplot(indices, weights.T, labels=labels, alpha=0.88)
    axis.set(xlabel="generation", ylabel="weight share", title="Best ensemble weights by generation", ylim=(0.0, 1.0))
    axis.legend(loc="upper left", ncol=2, fontsize=8)
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(graph_dir / "best_weights_by_generation.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 5.5))
    axis.plot(indices, [row["average_ev_bb_per_100"] for row in best], marker="o", label="total EV")
    axis.plot(indices, [row["all_in_ev_bb_per_100"] for row in best], marker="o", label="all-in EV contribution")
    axis.plot(indices, [row["non_all_in_ev_bb_per_100"] for row in best], marker="o", label="non-all-in EV contribution")
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set(xlabel="generation", ylabel="BB/100", title="Best candidate EV decomposition")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(graph_dir / "best_ev_components_by_generation.png", dpi=180)
    plt.close(fig)


def weight_text(labels: Sequence[str], weights: Sequence[float]) -> str:
    return " ".join(
        f"{label}={100.0 * float(weight):.1f}%"
        for label, weight in zip(labels, weights)
        if float(weight) > 0.0
    )


class GeneticOptimizer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.output = args.output.resolve()
        self.events_path = self.output / "events.jsonl"
        self.output.mkdir(parents=True, exist_ok=True)
        self.device = (
            "cuda" if args.device == "auto" and torch.cuda.is_available()
            else "cpu" if args.device == "auto"
            else args.device
        )
        self.model_paths = [path.resolve() for path in args.models]
        missing = [str(path) for path in self.model_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError("missing policy snapshots: " + ", ".join(missing))
        print(f"Loading {len(self.model_paths)} policies once on {self.device}...", flush=True)
        self.snapshots = [
            load_policy_snapshot(path, device=self.device) for path in self.model_paths
        ]
        self.iterations = [int(snapshot.iteration) for snapshot in self.snapshots]
        if len(set(self.iterations)) != len(self.iterations):
            raise ValueError("model iterations must be unique")
        self.labels = [f"policy_{iteration}" for iteration in self.iterations]
        self.hashes = [file_sha256(path) for path in self.model_paths]
        self.seeded_weights = self._load_seed_population()
        self.metadata = self._validate_compatibility()
        environment = dict(self.metadata["environment"])
        self.environment = environment
        env = HeadsUpHoldemEngine(
            starting_stack=int(environment["starting_stack"]),
            small_blind=int(environment["small_blind"]),
            big_blind=int(environment["big_blind"]),
            seed=args.seed,
        )
        self.trainer = HeadsUpNeuralCFR(
            env,
            device=self.device,
            hidden=int(self.metadata["hidden"]),
            blocks=int(self.metadata["blocks"]),
            advantage_capacity=1,
            policy_capacity=1,
            max_history=int(self.metadata["max_history"]),
            seed=args.seed,
        )
        self.fitness_league = self._build_opponents(args.fitness_league)
        self.report_opponents = self._build_opponents(args.report_opponents)
        self.settings = FitnessSettings(
            uncertainty_z=args.uncertainty_z,
            non_all_in_loss_penalty=args.non_all_in_loss_penalty,
            max_all_in_positive_share=args.max_all_in_positive_share,
            all_in_concentration_penalty=args.all_in_concentration_penalty,
            worst_opponent_loss_penalty=args.worst_opponent_loss_penalty,
        )
        self.config = self._configuration()
        self.config_hash = hashlib.sha256(
            json.dumps(self.config, sort_keys=True).encode("utf-8")
        ).hexdigest()
        self._validate_or_write_config()

    def _validate_compatibility(self) -> dict[str, Any]:
        first = self.snapshots[0].metadata
        for snapshot in self.snapshots[1:]:
            for key in ("input_dim", "max_history", "encoder_schema_version", "environment"):
                if snapshot.metadata.get(key) != first.get(key):
                    raise ValueError(f"incompatible model metadata for {key}")
        return first

    def _load_seed_population(self) -> list[list[float]]:
        path = self.args.seed_population
        if path is None:
            return []
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"seed population not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        ranked = payload.get("ranked", payload if isinstance(payload, list) else [])
        if not isinstance(ranked, list):
            raise ValueError("seed population must contain a ranked candidate list")
        weights = []
        for item in ranked[: self.args.population_size]:
            values = item.get("weights") if isinstance(item, dict) else item
            if not isinstance(values, list) or len(values) != len(self.model_paths):
                raise ValueError("seed population weight width does not match --models")
            weights.append([float(value) for value in values])
        print(f"Loaded {len(weights)} elite seeds from {path}", flush=True)
        return weights

    def _build_opponents(self, names: Sequence[str]) -> list[tuple[str, Any]]:
        by_iteration = {
            int(snapshot.iteration): snapshot for snapshot in self.snapshots
        }
        opponents: list[tuple[str, Any]] = []
        for name in names:
            if name == "old3_top3":
                required = [by_iteration.get(value) for value in (725, 950, 1025)]
                if any(snapshot is None for snapshot in required):
                    raise ValueError("old3_top3 requires policies 725, 950, and 1025")
                provider = WeightedEnsembleProvider(
                    self.trainer,
                    required,
                    [1.0 / 3.0] * 3,
                    top_k=3,
                    name="old3_top3",
                )
            elif name == "tag":
                provider = ScriptedProvider(
                    self.trainer.env, OPPONENT_PROFILES["tight_aggressive"]()
                )
            elif name == "random":
                provider = ScriptedProvider(
                    self.trainer.env, OPPONENT_PROFILES["random"]()
                )
            elif name.startswith("policy_"):
                iteration = int(name.split("_", 1)[1])
                snapshot = by_iteration.get(iteration)
                if snapshot is None:
                    raise ValueError(f"league opponent {name} is not in --models")
                provider = SnapshotProvider(self.trainer, snapshot, name=name)
            else:
                raise ValueError(f"unsupported opponent {name!r}")
            opponents.append((name, provider))
        return opponents

    def _configuration(self) -> dict[str, Any]:
        args = self.args
        return {
            "version": STATE_VERSION,
            "model_paths": [str(path) for path in self.model_paths],
            "model_hashes": self.hashes,
            "model_iterations": self.iterations,
            "fitness_league": list(args.fitness_league),
            "report_opponents": list(args.report_opponents),
            "seed_population": (
                str(args.seed_population.resolve()) if args.seed_population else None
            ),
            "seed_population_sha256": (
                file_sha256(args.seed_population.resolve())
                if args.seed_population
                else None
            ),
            "top_k": args.top_k,
            "population_size": args.population_size,
            "generations": args.generations,
            "screening_hands": args.screening_hands,
            "validation_hands": args.validation_hands,
            "final_hands": args.final_hands,
            "validation_candidates": args.validation_candidates,
            "final_candidates": args.final_candidates,
            "min_active": args.min_active,
            "max_active": args.max_active,
            "min_weight": args.min_weight,
            "elite_count": args.elite_count,
            "mutation_scale": args.mutation_scale,
            "structural_mutation_probability": args.structural_mutation_probability,
            "random_immigrants": args.random_immigrants,
            "seed": args.seed,
            "seed_stride": args.seed_stride,
            "generation_seed_stride": args.generation_seed_stride,
            "screening_seed_offset": args.screening_seed_offset,
            "validation_seed_offset": args.validation_seed_offset,
            "final_seed_offset": args.final_seed_offset,
            "fitness": asdict(self.settings),
            "device": self.device,
            "inference_batch_size": args.inference_batch_size,
            "simulation_batch_size": args.simulation_batch_size,
            "environment": self.environment,
        }

    def _validate_or_write_config(self) -> None:
        path = self.output / "run_config.json"
        payload = {"config_hash": self.config_hash, **self.config}
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing.get("config_hash") != self.config_hash:
                raise RuntimeError(
                    f"output directory belongs to a different configuration: {path}"
                )
        else:
            atomic_json(path, payload)
            append_event(self.events_path, "run_created", config_hash=self.config_hash)

    def _candidate_provider(self, weights: Sequence[float]) -> WeightedEnsembleProvider:
        identifier = candidate_id(weights)
        return WeightedEnsembleProvider(
            self.trainer,
            self.snapshots,
            weights,
            top_k=self.args.top_k,
            name=f"genetic_{identifier}_top{self.args.top_k}",
        )

    def evaluate_candidate(
        self,
        weights: Sequence[float],
        *,
        hands: int,
        seed_base: int,
        phase: str,
        generation: int | None,
        candidate_index: int,
        candidate_total: int,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        identifier = candidate_id(weights)
        provider = self._candidate_provider(weights)
        fitness_matches = []
        report_matches = []
        print(
            f"[{phase}] gen={generation if generation is not None else '-'} "
            f"candidate={candidate_index}/{candidate_total} id={identifier} "
            f"weights: {weight_text(self.labels, weights)}",
            flush=True,
        )
        opponents = [
            (True, name, opponent) for name, opponent in self.fitness_league
        ] + [
            (False, name, opponent) for name, opponent in self.report_opponents
        ]
        for opponent_index, (in_fitness, opponent_name, opponent) in enumerate(opponents):
            seed = int(seed_base) + opponent_index * int(self.args.seed_stride)
            result = run_reciprocal_match(
                provider,
                lambda env, selected=opponent: selected,
                environment=self.environment,
                hands=hands,
                seed=seed,
                inference_batch_size=self.args.inference_batch_size,
                simulation_batch_size=self.args.simulation_batch_size,
            )
            (fitness_matches if in_fitness else report_matches).append(result)
            all_in_ev = 100.0 * float(result["candidate_all_in_net_bb"]) / hands
            non_all_in_ev = 100.0 * float(result["candidate_non_all_in_net_bb"]) / hands
            print(
                f"  vs {opponent_name:<12} "
                f"[{'FITNESS' if in_fitness else 'REPORT ':7}] "
                f"EV={result['mean_ev_bb_per_100']:+8.3f} "
                f"all-in={all_in_ev:+8.3f} normal={non_all_in_ev:+8.3f} "
                f"shove={100.0 * result['candidate_all_in_hand_rate']:5.2f}%",
                flush=True,
            )
        scored = score_matches(fitness_matches, self.settings)
        elapsed = time.perf_counter() - started
        candidate = {
            "candidate_id": identifier,
            "weights": [float(value) for value in weights],
            "weight_map": {
                label: float(value) for label, value in zip(self.labels, weights)
            },
            **scored,
            "elapsed_seconds": elapsed,
            "phase": phase,
            "generation": generation,
            "hands_per_opponent": hands,
            "fitness_matches": fitness_matches,
            "report_matches": report_matches,
        }
        print(
            f"  FITNESS={scored['fitness']:+8.3f} rawEV={scored['average_ev_bb_per_100']:+8.3f} "
            f"all-inEV={scored['all_in_ev_bb_per_100']:+8.3f} "
            f"normalEV={scored['non_all_in_ev_bb_per_100']:+8.3f} "
            f"worst={scored['worst_opponent_ev_bb_per_100']:+8.3f} "
            f"penalties(u/n/a/w)={scored['uncertainty_penalty']:.2f}/"
            f"{scored['non_all_in_loss_penalty']:.2f}/"
            f"{scored['all_in_concentration_penalty']:.2f}/"
            f"{scored['worst_opponent_loss_penalty']:.2f} "
            f"time={elapsed:.1f}s",
            flush=True,
        )
        append_event(
            self.events_path,
            "candidate_completed",
            phase=phase,
            generation=generation,
            candidate_id=identifier,
            fitness=scored["fitness"],
            average_ev_bb_per_100=scored["average_ev_bb_per_100"],
            all_in_ev_bb_per_100=scored["all_in_ev_bb_per_100"],
            non_all_in_ev_bb_per_100=scored["non_all_in_ev_bb_per_100"],
            elapsed_seconds=elapsed,
        )
        return candidate

    def _generation_files(self) -> list[Path]:
        return sorted((self.output / "generations").glob("generation_*.json"))

    def _load_generations(self) -> list[dict[str, Any]]:
        return [json.loads(path.read_text(encoding="utf-8")) for path in self._generation_files()]

    def run_evolution(self) -> list[dict[str, Any]]:
        state_path = self.output / "evolution_state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("config_hash") != self.config_hash:
                raise RuntimeError("evolution state configuration mismatch")
            generation = int(state["generation"])
            population = state["population"]
            completed = state.get("completed", {})
            print(
                f"Resuming generation {generation}: {len(completed)}/{len(population)} candidates complete.",
                flush=True,
            )
        else:
            generation = 0
            population = initial_population(
                self.iterations,
                seeded_weights=self.seeded_weights,
                population_size=self.args.population_size,
                min_active=self.args.min_active,
                max_active=self.args.max_active,
                min_weight=self.args.min_weight,
                seed=self.args.seed,
            )
            completed = {}
        while self.args.generations == 0 or generation < self.args.generations:
            generation_started = time.perf_counter()
            generation_target = (
                "unbounded" if self.args.generations == 0 else str(self.args.generations)
            )
            print(
                f"\n=== GENERATION {generation + 1}/{generation_target} "
                f"population={len(population)} hands/opponent={self.args.screening_hands:,} ===",
                flush=True,
            )
            for index, weights in enumerate(population, 1):
                identifier = candidate_id(weights)
                if identifier in completed:
                    continue
                result = self.evaluate_candidate(
                    weights,
                    hands=self.args.screening_hands,
                    seed_base=(
                        self.args.seed
                        + generation * self.args.generation_seed_stride
                        + self.args.screening_seed_offset
                    ),
                    phase="evolution",
                    generation=generation,
                    candidate_index=index,
                    candidate_total=len(population),
                )
                completed[identifier] = result
                live_ranked = sorted(
                    completed.values(),
                    key=lambda item: float(item["fitness"]),
                    reverse=True,
                )
                atomic_json(
                    self.output / "best_live_ensemble.json",
                    {
                        "generation": generation,
                        "completed_candidates": len(completed),
                        "population_size": len(population),
                        "best": live_ranked[0],
                    },
                )
                atomic_json(
                    state_path,
                    {
                        "config_hash": self.config_hash,
                        "generation": generation,
                        "population": population,
                        "completed": completed,
                    },
                )
                average_time = mean(
                    float(item["elapsed_seconds"]) for item in completed.values()
                )
                remaining = len(population) - len(completed)
                print(
                    f"  progress {len(completed)}/{len(population)}; "
                    f"generation ETA ~{average_time * remaining / 60.0:.1f} min",
                    flush=True,
                )
            ranked = sorted(
                completed.values(), key=lambda item: float(item["fitness"]), reverse=True
            )
            generation_payload = {
                "generation": generation,
                "completed_at_utc": utc_now(),
                "elapsed_seconds": time.perf_counter() - generation_started,
                "ranked": ranked,
            }
            generation_dir = self.output / "generations"
            generation_dir.mkdir(parents=True, exist_ok=True)
            atomic_json(
                generation_dir / f"generation_{generation:03d}.json",
                generation_payload,
            )
            write_population_csv(
                generation_dir / f"generation_{generation:03d}.csv",
                ranked,
                self.labels,
            )
            generations = self._load_generations()
            plot_history(self.output, generations, self.labels)
            best = ranked[0]
            atomic_json(self.output / "best_screening_ensemble.json", best)
            print("GENERATION RANKING:", flush=True)
            for rank, row in enumerate(ranked, 1):
                print(
                    f"  #{rank:02d} fitness={row['fitness']:+8.3f} "
                    f"EV={row['average_ev_bb_per_100']:+8.3f} "
                    f"all-in={row['all_in_ev_bb_per_100']:+8.3f} "
                    f"normal={row['non_all_in_ev_bb_per_100']:+8.3f} "
                    f"{weight_text(self.labels, row['weights'])}",
                    flush=True,
                )
            print(
                f"GENERATION BEST fitness={best['fitness']:+.3f} "
                f"EV={best['average_ev_bb_per_100']:+.3f} "
                f"all-inEV={best['all_in_ev_bb_per_100']:+.3f} "
                f"normalEV={best['non_all_in_ev_bb_per_100']:+.3f}\n"
                f"  {weight_text(self.labels, best['weights'])}",
                flush=True,
            )
            append_event(
                self.events_path,
                "generation_completed",
                generation=generation,
                best_candidate_id=best["candidate_id"],
                best_fitness=best["fitness"],
            )
            generation += 1
            if self.args.generations > 0 and generation >= self.args.generations:
                atomic_json(
                    state_path,
                    {
                        "config_hash": self.config_hash,
                        "generation": generation,
                        "population": [],
                        "completed": {},
                        "evolution_complete": True,
                    },
                )
                break
            population = evolve_population(
                ranked,
                population_size=self.args.population_size,
                elite_count=self.args.elite_count,
                min_active=self.args.min_active,
                max_active=self.args.max_active,
                min_weight=self.args.min_weight,
                mutation_scale=self.args.mutation_scale,
                structural_mutation_probability=self.args.structural_mutation_probability,
                random_immigrants=self.args.random_immigrants,
                seed=self.args.seed + generation * 97_409,
            )
            completed = {}
            atomic_json(
                state_path,
                {
                    "config_hash": self.config_hash,
                    "generation": generation,
                    "population": population,
                    "completed": completed,
                },
            )
        return self._load_generations()

    def _unique_screening_candidates(
        self, generations: Sequence[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        candidates: dict[str, dict[str, Any]] = {}
        for generation in generations:
            for item in generation["ranked"]:
                current = candidates.get(item["candidate_id"])
                if current is None or float(item["fitness"]) > float(current["fitness"]):
                    candidates[item["candidate_id"]] = item
        return sorted(
            candidates.values(), key=lambda item: float(item["fitness"]), reverse=True
        )

    def run_stage(
        self,
        candidates: Sequence[dict[str, Any]],
        *,
        phase: str,
        hands: int,
        seed_offset: int,
    ) -> list[dict[str, Any]]:
        path = self.output / f"{phase}_results.json"
        existing: dict[str, dict[str, Any]] = {}
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            existing = {item["candidate_id"]: item for item in payload["ranked"]}
        for index, candidate in enumerate(candidates, 1):
            identifier = candidate["candidate_id"]
            if identifier in existing:
                continue
            result = self.evaluate_candidate(
                candidate["weights"],
                hands=hands,
                seed_base=self.args.seed + seed_offset,
                phase=phase,
                generation=None,
                candidate_index=index,
                candidate_total=len(candidates),
            )
            existing[identifier] = result
            ranked = sorted(
                existing.values(), key=lambda item: float(item["fitness"]), reverse=True
            )
            atomic_json(
                path,
                {
                    "phase": phase,
                    "hands_per_opponent": hands,
                    "ranked": ranked,
                    "complete": len(existing) == len(candidates),
                },
            )
            completed_times = [
                float(item["elapsed_seconds"]) for item in existing.values()
            ]
            remaining = len(candidates) - len(existing)
            print(
                f"  {phase} progress {len(existing)}/{len(candidates)}; "
                f"ETA ~{mean(completed_times) * remaining / 60.0:.1f} min",
                flush=True,
            )
        ranked = sorted(
            existing.values(), key=lambda item: float(item["fitness"]), reverse=True
        )
        write_population_csv(self.output / f"{phase}_results.csv", ranked, self.labels)
        return ranked

    def run(self) -> dict[str, Any]:
        started = time.perf_counter()
        opponent_count = len(self.fitness_league) + len(self.report_opponents)
        if self.args.generations == 0:
            per_generation = (
                opponent_count
                * self.args.population_size
                * self.args.screening_hands
            )
            print(
                f"Configured unbounded search: {per_generation:,} reciprocal hands "
                f"per generation across {len(self.fitness_league)} fitness and "
                f"{len(self.report_opponents)} report-only opponents. Ctrl+C stops "
                "safely after preserving completed candidates.",
                flush=True,
            )
        else:
            equivalent_hands = opponent_count * (
                self.args.population_size
                * self.args.generations
                * self.args.screening_hands
                + self.args.validation_candidates * self.args.validation_hands
                + self.args.final_candidates * self.args.final_hands
            )
            print(
                f"Configured workload: {equivalent_hands:,} reciprocal hands across "
                f"{opponent_count} opponents. Completed work is resumable.",
                flush=True,
            )
        generations = self.run_evolution()
        screening = self._unique_screening_candidates(generations)
        validation_candidates = screening[: self.args.validation_candidates]
        print(
            f"\n=== VALIDATION: {len(validation_candidates)} candidates, "
            f"{self.args.validation_hands:,} hands/opponent ===",
            flush=True,
        )
        validation = self.run_stage(
            validation_candidates,
            phase="validation",
            hands=self.args.validation_hands,
            seed_offset=self.args.validation_seed_offset,
        )
        final_candidates = validation[: self.args.final_candidates]
        print(
            f"\n=== FINAL HOLDOUT: {len(final_candidates)} candidates, "
            f"{self.args.final_hands:,} hands/opponent ===",
            flush=True,
        )
        final = self.run_stage(
            final_candidates,
            phase="final",
            hands=self.args.final_hands,
            seed_offset=self.args.final_seed_offset,
        )
        winner = final[0]
        result = {
            "completed_at_utc": utc_now(),
            "elapsed_seconds": time.perf_counter() - started,
            "winner": winner,
            "runner_up": final[1] if len(final) > 1 else None,
            "configuration_hash": self.config_hash,
            "fitness_settings": asdict(self.settings),
        }
        atomic_json(self.output / "best_ensemble.json", result)
        append_event(
            self.events_path,
            "optimization_completed",
            winner=winner["candidate_id"],
            fitness=winner["fitness"],
            average_ev_bb_per_100=winner["average_ev_bb_per_100"],
        )
        print(
            "\n=== OPTIMIZATION COMPLETE ===\n"
            f"winner={winner['candidate_id']} fitness={winner['fitness']:+.3f} "
            f"EV={winner['average_ev_bb_per_100']:+.3f} "
            f"all-inEV={winner['all_in_ev_bb_per_100']:+.3f} "
            f"normalEV={winner['non_all_in_ev_bb_per_100']:+.3f}\n"
            f"weights: {weight_text(self.labels, winner['weights'])}\n"
            f"results: {self.output}",
            flush=True,
        )
        return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resumable genetic optimization of weighted HU policy ensembles"
    )
    parser.add_argument("--models", nargs="+", type=Path, default=list(DEFAULT_MODELS))
    parser.add_argument(
        "--fitness-league", nargs="+", default=list(DEFAULT_FITNESS_LEAGUE)
    )
    parser.add_argument(
        "--report-opponents", nargs="*", default=list(DEFAULT_REPORT_OPPONENTS)
    )
    parser.add_argument(
        "--seed-population",
        type=Path,
        help="generation JSON whose ranked weights seed the initial population",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/downloaded_risk_aware/genetic_ensemble_search_global_v2"
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--population-size", type=int, default=16)
    parser.add_argument(
        "--generations",
        type=int,
        default=0,
        help="generation count; 0 runs until Ctrl+C",
    )
    parser.add_argument("--screening-hands", type=int, default=20_000)
    parser.add_argument("--validation-candidates", type=int, default=5)
    parser.add_argument("--validation-hands", type=int, default=200_000)
    parser.add_argument("--final-candidates", type=int, default=2)
    parser.add_argument("--final-hands", type=int, default=1_000_000)
    parser.add_argument("--min-active", type=int, default=2)
    parser.add_argument("--max-active", type=int, default=5)
    parser.add_argument("--min-weight", type=float, default=0.02)
    parser.add_argument("--elite-count", type=int, default=4)
    parser.add_argument("--mutation-scale", type=float, default=0.35)
    parser.add_argument("--structural-mutation-probability", type=float, default=0.30)
    parser.add_argument("--random-immigrants", type=int, default=2)
    parser.add_argument("--uncertainty-z", type=float, default=1.0)
    parser.add_argument("--non-all-in-loss-penalty", type=float, default=0.50)
    parser.add_argument("--max-all-in-positive-share", type=float, default=0.70)
    parser.add_argument("--all-in-concentration-penalty", type=float, default=0.50)
    parser.add_argument("--worst-opponent-loss-penalty", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=7_259_501_025)
    parser.add_argument("--seed-stride", type=int, default=1_000_003)
    parser.add_argument("--generation-seed-stride", type=int, default=10_000_019)
    parser.add_argument("--screening-seed-offset", type=int, default=0)
    parser.add_argument("--validation-seed-offset", type=int, default=1_000_000_007)
    parser.add_argument("--final-seed-offset", type=int, default=2_000_000_011)
    parser.add_argument("--inference-batch-size", type=int, default=8192)
    parser.add_argument("--simulation-batch-size", type=int, default=20_000)
    args = parser.parse_args(argv)
    if not 2 <= args.population_size:
        parser.error("--population-size must be at least 2")
    if args.generations < 0:
        parser.error("--generations must be nonnegative; 0 means unbounded")
    if not 1 <= args.elite_count < args.population_size:
        parser.error("--elite-count must be in [1, population-size)")
    if not 0 <= args.random_immigrants < args.population_size - args.elite_count:
        parser.error(
            "--random-immigrants must leave room for elites and at least one child"
        )
    if not 1 <= args.min_active <= args.max_active <= len(args.models):
        parser.error("active-model limits must satisfy 1 <= min <= max <= model count")
    if not 0.0 <= args.min_weight < 1.0:
        parser.error("--min-weight must be in [0, 1)")
    if args.min_weight * args.min_active >= 1.0:
        parser.error("--min-weight is too large for --min-active")
    for name in ("screening_hands", "validation_hands", "final_hands"):
        value = int(getattr(args, name))
        if value <= 0 or value % 2:
            parser.error(f"--{name.replace('_', '-')} must be a positive even number")
    if not 1 <= args.final_candidates <= args.validation_candidates:
        parser.error("final candidate count must be positive and no larger than validation count")
    if (
        args.generations > 0
        and args.validation_candidates > args.population_size * args.generations
    ):
        parser.error("validation candidate count exceeds possible screening candidates")
    if args.top_k < 0:
        parser.error("--top-k must be nonnegative")
    if not 0.0 <= args.max_all_in_positive_share <= 1.0:
        parser.error("--max-all-in-positive-share must be in [0, 1]")
    if not args.fitness_league:
        parser.error("--fitness-league needs at least one opponent")
    if len(set(args.fitness_league)) != len(args.fitness_league):
        parser.error("--fitness-league contains duplicates")
    if len(set(args.report_opponents)) != len(args.report_opponents):
        parser.error("--report-opponents contains duplicates")
    overlap = set(args.fitness_league) & set(args.report_opponents)
    if overlap:
        parser.error(f"opponents cannot be both fitness and report-only: {sorted(overlap)}")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    optimizer = GeneticOptimizer(args)
    try:
        optimizer.run()
        return 0
    except KeyboardInterrupt:
        print(
            "\nOptimizer stopped by user. Completed candidates, current population, "
            f"and live best are saved in {optimizer.output}",
            flush=True,
        )
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
