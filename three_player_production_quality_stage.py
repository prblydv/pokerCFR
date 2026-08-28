"""Long-running training, evaluation, snapshots, and dashboards.

This module deliberately separates optimization diagnostics from evidence of
playing strength.  Strength is estimated on fixed, held-out deal suites against
several reproducible opponent profiles and a historical policy league.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from statistics import NormalDist
from typing import Any, Callable, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from three_player_cfr import ThreePlayerNeuralCFR
from three_player_engine import (
    ACTION_ALL_IN,
    ACTION_CALL,
    ACTION_CHECK,
    ACTION_FOLD,
    ACTION_MIN_RAISE,
    ACTION_NAMES,
    NUM_ACTIONS,
    ThreePlayerHoldemEnv,
    evaluate_5card,
)
from three_player_models import PolicyNetwork, build_policy_network
from three_player_tournament import TournamentResult, play_tournament


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _draw_action(probabilities: torch.Tensor, rng: random.Random) -> int:
    threshold = rng.random()
    cumulative = 0.0
    fallback = int(torch.argmax(probabilities).item())
    for action, probability in enumerate(probabilities.tolist()):
        if probability <= 0:
            continue
        fallback = action
        cumulative += float(probability)
        if threshold <= cumulative + 1e-12:
            return action
    return fallback


def _normalised_policy(weights: dict[int, float], legal: Sequence[int]) -> torch.Tensor:
    result = torch.zeros(NUM_ACTIONS, dtype=torch.float32)
    legal_set = set(int(action) for action in legal)
    for action, weight in weights.items():
        if action in legal_set and weight > 0:
            result[action] += float(weight)
    if float(result.sum()) <= 0:
        for action in legal:
            result[int(action)] = 1.0
    return result / result.sum()


class ScriptedOpponent:
    """Stable opponent interface used by held-out benchmark matches."""

    name = "scripted"

    def probabilities(self, env, state, player: int) -> torch.Tensor:
        raise NotImplementedError


class UniformRandomOpponent(ScriptedOpponent):
    name = "random"

    def probabilities(self, env, state, player: int) -> torch.Tensor:
        legal = env.legal_actions(state)
        return _normalised_policy({action: 1.0 for action in legal}, legal)


class CallingStationOpponent(ScriptedOpponent):
    name = "calling_station"

    def probabilities(self, env, state, player: int) -> torch.Tensor:
        legal = env.legal_actions(state)
        if ACTION_CHECK in legal:
            return _normalised_policy({ACTION_CHECK: 0.97, ACTION_MIN_RAISE: 0.03}, legal)
        if ACTION_CALL in legal:
            return _normalised_policy({ACTION_CALL: 0.92, ACTION_FOLD: 0.08}, legal)
        return _normalised_policy({}, legal)


def _preflop_strength(cards: Sequence[int]) -> float:
    ranks = sorted((card % 13 + 2 for card in cards), reverse=True)
    high, low = ranks
    suited = cards[0] // 13 == cards[1] // 13
    pair = high == low
    gap = high - low
    score = 0.45 * ((high - 2) / 12.0) + 0.25 * ((low - 2) / 12.0)
    if pair:
        score += 0.28 + 0.18 * ((high - 2) / 12.0)
    if suited:
        score += 0.07
    if not pair and gap <= 1:
        score += 0.06
    elif gap >= 4:
        score -= 0.06
    return min(1.0, max(0.0, score))


def _postflop_strength(cards: Sequence[int]) -> float:
    if len(cards) < 5:
        return _preflop_strength(cards[:2])
    best = max(evaluate_5card(combo) for combo in combinations(cards, 5))
    category = best // (15**5)
    high_component = (best % (15**5)) / float(15**5)
    return min(1.0, 0.10 + 0.105 * category + 0.08 * high_component)


class TightAggressiveOpponent(ScriptedOpponent):
    """A deterministic, card-aware benchmark—not a claim of optimal play."""

    name = "tight_aggressive"

    def probabilities(self, env, state, player: int) -> torch.Tensor:
        legal = env.legal_actions(state)
        visible = list(state.hole[player]) + list(state.board)
        strength = (
            _preflop_strength(state.hole[player])
            if state.street == 0
            else _postflop_strength(visible)
        )
        raises = [action for action in legal if action >= ACTION_MIN_RAISE]
        preferred_raise = raises[len(raises) // 2] if raises else None
        weights: dict[int, float] = {}
        facing_bet = ACTION_CALL in legal
        if facing_bet:
            if strength >= 0.76:
                weights = {ACTION_FOLD: 0.02, ACTION_CALL: 0.38}
                if preferred_raise is not None:
                    weights[preferred_raise] = 0.60
            elif strength >= 0.48:
                weights = {ACTION_FOLD: 0.18, ACTION_CALL: 0.70}
                if preferred_raise is not None:
                    weights[preferred_raise] = 0.12
            else:
                weights = {ACTION_FOLD: 0.78, ACTION_CALL: 0.20}
                if preferred_raise is not None:
                    weights[preferred_raise] = 0.02
        else:
            check_weight = 0.25 if strength >= 0.68 else 0.82
            weights = {ACTION_CHECK: check_weight}
            if preferred_raise is not None:
                weights[preferred_raise] = 1.0 - check_weight
        return _normalised_policy(weights, legal)


OPPONENT_PROFILES: dict[str, type[ScriptedOpponent]] = {
    UniformRandomOpponent.name: UniformRandomOpponent,
    CallingStationOpponent.name: CallingStationOpponent,
    TightAggressiveOpponent.name: TightAggressiveOpponent,
}


@dataclass
class PolicySnapshot:
    iteration: int
    policy_nets: list[PolicyNetwork]
    metadata: dict[str, Any]


def save_policy_snapshot(
    trainer: ThreePlayerNeuralCFR,
    path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save the three deployable average-policy networks only."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "kind": "three_player_policy_snapshot",
        "iteration": trainer.iteration,
        "input_dim": trainer.input_dim,
        "hidden": trainer.hidden,
        "blocks": trainer.blocks,
        "network_architecture": trainer.network_architecture,
        "max_history": trainer.max_history,
        "action_names": tuple(ACTION_NAMES),
        "environment": {
            "stack_size": trainer.env.stack_size,
            "sb": trainer.env.sb,
            "bb": trainer.env.bb,
            "tournament_total_chips": trainer.tournament_total_chips,
        },
        "include_tournament_features": trainer.include_tournament_features,
        "tournament_features": trainer.include_tournament_features,
        "encoder": {
            "include_tournament_features": trainer.include_tournament_features,
            "tournament_total_chips": trainer.tournament_total_chips,
        },
        "training_mode": {
            "variable_stack_training": trainer.variable_stack_training,
            "heads_up_root_fraction": trainer.heads_up_root_fraction,
            "continuation_root_fraction": trainer.continuation_root_fraction,
            "minimum_live_stack": trainer.minimum_live_stack,
            "root_stack_concentration": trainer.root_stack_concentration,
        },
        "policy_nets": [
            {key: value.detach().cpu() for key, value in net.state_dict().items()}
            for net in trainer.policy_nets
        ],
        "metadata": dict(metadata or {}),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return path


def load_policy_snapshot(
    path: str | Path, *, device: str | torch.device = "cpu"
) -> PolicySnapshot:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("kind") != "three_player_policy_snapshot" or int(
        payload.get("version", -1)
    ) != 1:
        raise ValueError("unsupported policy snapshot")
    if tuple(payload.get("action_names", ())) != tuple(ACTION_NAMES):
        raise ValueError("policy snapshot action space does not match")
    networks = [
        build_policy_network(
            str(payload.get("network_architecture", "residual_mlp")),
            int(payload["input_dim"]),
            int(payload["hidden"]),
            int(payload["blocks"]),
        ).to(device)
        for _ in range(3)
    ]
    if len(payload.get("policy_nets", [])) != 3:
        raise ValueError("policy snapshot must contain three networks")
    for network, state in zip(networks, payload["policy_nets"]):
        network.load_state_dict(state)
        network.eval()
    metadata = dict(payload.get("metadata", {}))
    metadata["environment"] = dict(payload.get("environment", {}))
    metadata["max_history"] = int(payload.get("max_history", 0))
    metadata["input_dim"] = int(payload["input_dim"])
    metadata["hidden"] = int(payload["hidden"])
    metadata["blocks"] = int(payload["blocks"])
    metadata["network_architecture"] = str(
        payload.get("network_architecture", "residual_mlp")
    )
    metadata["encoder"] = dict(payload.get("encoder", {}))
    metadata["training_mode"] = dict(payload.get("training_mode", {}))
    metadata["include_tournament_features"] = bool(
        payload.get(
            "include_tournament_features",
            payload.get("tournament_features", False),
        )
    )
    return PolicySnapshot(int(payload["iteration"]), networks, metadata)


@dataclass
class EvaluationResult:
    profile: str
    summary: dict[str, float]
    hands: pd.DataFrame


@dataclass
class TournamentEvaluationResult:
    profile: str
    summary: dict[str, float]
    tournaments: pd.DataFrame


def _tournament_finish_position(result: TournamentResult, hero: int) -> float:
    """Return 1/2/3, or 2.5 when both losing seats bust together."""

    if result.winner == hero:
        return 1.0
    other_loser = next(
        seat for seat in range(3) if seat not in (hero, result.winner)
    )
    hero_hand = result.eliminated_on_hand[hero]
    other_hand = result.eliminated_on_hand[other_loser]
    if hero_hand is None or other_hand is None:
        raise RuntimeError("completed tournament is missing an elimination hand")
    if hero_hand == other_hand:
        return 2.5
    return 2.0 if hero_hand > other_hand else 3.0


def evaluate_tournaments_against_profile(
    trainer: ThreePlayerNeuralCFR,
    profile: str | ScriptedOpponent,
    *,
    tournaments_per_player: int = 1,
    seed: int = 702_700,
    max_hands: int = 10_000,
) -> TournamentEvaluationResult:
    """Rotate the learned policy through complete persistent-stack tournaments."""

    if tournaments_per_player <= 0:
        raise ValueError("tournaments_per_player must be positive")
    if max_hands <= 0:
        raise ValueError("max_hands must be positive")
    if isinstance(profile, str):
        if profile not in OPPONENT_PROFILES:
            raise ValueError(f"unknown opponent profile {profile!r}")
        opponent = OPPONENT_PROFILES[profile]()
    else:
        opponent = profile
    profile_offset = sum(
        (index + 1) * ord(char) for index, char in enumerate(opponent.name)
    )
    records: list[dict[str, Any]] = []

    for hero in range(3):
        for tournament_index in range(tournaments_per_player):
            tournament_seed = (
                int(seed)
                + profile_offset
                + hero * 100_003
                + tournament_index * 1_009
            )
            evaluation_env = type(trainer.env)(
                stack_size=trainer.env.stack_size,
                sb=trainer.env.sb,
                bb=trainer.env.bb,
                seed=tournament_seed,
            )

            def learned_policy(env, state, player):
                return trainer.average_policy(state, player)

            def opponent_policy(env, state, player):
                return opponent.probabilities(env, state, player)

            policies = [opponent_policy, opponent_policy, opponent_policy]
            policies[hero] = learned_policy
            result = play_tournament(
                evaluation_env,
                policies,
                rng=random.Random(tournament_seed + 50_000),
                max_hands=max_hands,
            )
            records.append(
                {
                    "profile": opponent.name,
                    "hero_seat": hero,
                    "tournament_index": tournament_index,
                    "winner": result.winner,
                    "won": float(result.winner == hero),
                    "busted": float(result.winner != hero),
                    "reward": float(result.rewards[hero]),
                    "finish_position": _tournament_finish_position(result, hero),
                    "eliminated_on_hand": result.eliminated_on_hand[hero],
                    "hands_played": result.hands_played,
                    "actions": sum(hand.actions for hand in result.hands),
                }
            )

    tournaments = pd.DataFrame.from_records(records)
    wins = tournaments["won"].astype(float)
    win_rate = float(wins.mean())
    win_stderr = (
        float(wins.std(ddof=1) / math.sqrt(len(wins))) if len(wins) > 1 else 0.0
    )
    summary = {
        "win_rate": win_rate,
        "win_rate_stderr": win_stderr,
        "win_rate_ci95_low": max(0.0, win_rate - 1.96 * win_stderr),
        "win_rate_ci95_high": min(1.0, win_rate + 1.96 * win_stderr),
        "bust_rate": float(tournaments["busted"].mean()),
        "mean_reward": float(tournaments["reward"].mean()),
        "mean_finish_position": float(tournaments["finish_position"].mean()),
        "mean_hands": float(tournaments["hands_played"].mean()),
        "mean_actions": float(tournaments["actions"].mean()),
        "tournaments": float(len(tournaments)),
    }
    return TournamentEvaluationResult(opponent.name, summary, tournaments)


def _role_name(button: int, seat: int) -> str:
    if seat == button:
        return "BTN"
    if seat == (button + 1) % 3:
        return "SB"
    return "BB"


def _clustered_summary(hands: pd.DataFrame) -> dict[str, float]:
    values = hands["payoff_bb"].astype(float)
    mean = float(values.mean())
    cluster_means = hands.groupby("deal_index")["payoff_bb"].mean()
    stderr = float(cluster_means.std(ddof=1) / math.sqrt(len(cluster_means))) if len(cluster_means) > 1 else 0.0
    output: dict[str, float] = {
        "mean_ev_bb": mean,
        "bb_per_100": 100.0 * mean,
        "clustered_stderr_bb": stderr,
        "ci95_low_bb": mean - 1.96 * stderr,
        "ci95_high_bb": mean + 1.96 * stderr,
        "positive_hand_rate": float((values > 0).mean()),
        "hands": float(len(hands)),
    }
    for role in ("BTN", "SB", "BB"):
        role_values = hands.loc[hands["role"] == role, "payoff_bb"]
        output[f"ev_{role}_bb"] = float(role_values.mean()) if len(role_values) else float("nan")
    return output


def evaluate_against_profile(
    trainer: ThreePlayerNeuralCFR,
    profile: str | ScriptedOpponent,
    *,
    games_per_player: int = 300,
    seed: int = 202_700,
    policy_nets: Sequence[PolicyNetwork] | None = None,
    inference_batch_size: int = 512,
) -> EvaluationResult:
    """Rotate each learned network against two fixed scripted opponents."""
    if games_per_player <= 0:
        raise ValueError("games_per_player must be positive")
    if isinstance(profile, str):
        if profile not in OPPONENT_PROFILES:
            raise ValueError(f"unknown opponent profile {profile!r}")
        opponent = OPPONENT_PROFILES[profile]()
    else:
        opponent = profile
    records: list[dict[str, Any]] = []
    profile_offset = sum((index + 1) * ord(char) for index, char in enumerate(opponent.name))

    for hero in range(3):
        evaluation_env = type(trainer.env)(
            stack_size=trainer.env.stack_size,
            sb=trainer.env.sb,
            bb=trainer.env.bb,
            seed=seed + profile_offset,
        )
        action_rng = random.Random(seed + 50_000 + profile_offset)
        states = [evaluation_env.new_hand() for _ in range(games_per_player)]
        buttons = [int(state.button) for state in states]
        steps = [0] * games_per_player
        while True:
            live = [index for index, state in enumerate(states) if not state.terminal]
            if not live:
                break
            hero_indices = [
                index for index in live if int(states[index].to_act) == hero
            ]
            hero_probabilities: dict[int, torch.Tensor] = {}
            if hero_indices:
                batch_states = [states[index] for index in hero_indices]
                predictions = trainer.average_policy_batch(
                    batch_states,
                    policy_nets=policy_nets,
                    batch_size=inference_batch_size,
                )
                hero_probabilities = dict(zip(hero_indices, predictions))
            for index in live:
                state = states[index]
                player = int(state.to_act)
                if player == hero:
                    probabilities = hero_probabilities[index]
                else:
                    probabilities = opponent.probabilities(
                        evaluation_env, state, player
                    )
                action = _draw_action(probabilities, action_rng)
                states[index] = evaluation_env.step(state, action)
                steps[index] += 1
                if steps[index] > 256:
                    raise RuntimeError("evaluation hand exceeded 256 decisions")
        for deal_index, (state, button) in enumerate(zip(states, buttons)):
            payoff = float(state.payoffs[hero]) / float(trainer.env.bb)
            records.append(
                {
                    "profile": opponent.name,
                    "hero_seat": hero,
                    "deal_index": deal_index,
                    "button": button,
                    "role": _role_name(button, hero),
                    "payoff_bb": payoff,
                }
            )
    hands = pd.DataFrame.from_records(records)
    return EvaluationResult(opponent.name, _clustered_summary(hands), hands)


def paired_improvement(
    baseline: EvaluationResult, candidate: EvaluationResult
) -> dict[str, float]:
    """Clustered common-random-number delta for the same benchmark profile."""
    if baseline.profile != candidate.profile:
        raise ValueError("baseline and candidate profiles must match")
    keys = ["hero_seat", "deal_index", "role"]
    merged = baseline.hands[keys + ["payoff_bb"]].merge(
        candidate.hands[keys + ["payoff_bb"]],
        on=keys,
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
    )
    merged["delta_bb"] = (
        merged["payoff_bb_candidate"] - merged["payoff_bb_baseline"]
    )
    clusters = merged.groupby("deal_index")["delta_bb"].mean()
    mean = float(merged["delta_bb"].mean())
    stderr = float(clusters.std(ddof=1) / math.sqrt(len(clusters))) if len(clusters) > 1 else 0.0
    if stderr <= 1e-12:
        probability_positive = 1.0 if mean > 0 else (0.0 if mean < 0 else 0.5)
    else:
        probability_positive = NormalDist().cdf(mean / stderr)
    return {
        "delta_ev_bb": mean,
        "delta_stderr_bb": stderr,
        "delta_ci95_low_bb": mean - 1.96 * stderr,
        "delta_ci95_high_bb": mean + 1.96 * stderr,
        "probability_delta_positive": float(probability_positive),
    }


@dataclass
class BenchmarkSuiteResult:
    metrics: dict[str, float]
    evaluations: dict[str, EvaluationResult]
    baseline_evaluations: dict[str, EvaluationResult]


def evaluate_benchmark_suite(
    trainer: ThreePlayerNeuralCFR,
    *,
    profiles: Sequence[str] = ("random", "calling_station", "tight_aggressive"),
    games_per_player: int = 300,
    seed: int = 202_700,
    baseline_policy_nets: Sequence[PolicyNetwork] | None = None,
    baseline_results: dict[str, EvaluationResult] | None = None,
    inference_batch_size: int = 512,
) -> BenchmarkSuiteResult:
    """Evaluate current and optional baseline policy on identical fixed suites."""
    metrics: dict[str, float] = {}
    evaluations: dict[str, EvaluationResult] = {}
    baseline_evaluations: dict[str, EvaluationResult] = {}
    for profile in profiles:
        candidate = evaluate_against_profile(
            trainer,
            profile,
            games_per_player=games_per_player,
            seed=seed,
            inference_batch_size=inference_batch_size,
        )
        evaluations[profile] = candidate
        for key, value in candidate.summary.items():
            metrics[f"benchmark_{profile}_{key}"] = value
        if baseline_policy_nets is not None or (
            baseline_results is not None and profile in baseline_results
        ):
            if baseline_results is not None and profile in baseline_results:
                baseline = baseline_results[profile]
            else:
                baseline = evaluate_against_profile(
                    trainer,
                    profile,
                    games_per_player=games_per_player,
                    seed=seed,
                    policy_nets=baseline_policy_nets,
                    inference_batch_size=inference_batch_size,
                )
            baseline_evaluations[profile] = baseline
            for key, value in paired_improvement(baseline, candidate).items():
                metrics[f"benchmark_{profile}_{key}"] = value
    ev_values = [metrics[f"benchmark_{profile}_mean_ev_bb"] for profile in profiles]
    se_values = [
        metrics[f"benchmark_{profile}_clustered_stderr_bb"] for profile in profiles
    ]
    score = float(np.mean(ev_values))
    score_stderr = math.sqrt(sum(value * value for value in se_values)) / len(se_values)
    metrics["benchmark_composite_ev_bb"] = score
    metrics["benchmark_composite_stderr_bb"] = score_stderr
    metrics["benchmark_composite_lcb95_bb"] = score - 1.96 * score_stderr
    return BenchmarkSuiteResult(metrics, evaluations, baseline_evaluations)


@dataclass(frozen=True)
class CampaignConfig:
    target_iteration: int = 10_000
    traversals_per_player: int = 3
    traversal_workers: int = 1
    advantage_steps: int = 64
    policy_steps: int = 32
    batch_size: int = 512
    evaluate_every: int = 25
    checkpoint_every: int = 25
    snapshot_every: int = 100
    evaluation_games_per_player: int = 300
    league_games_per_player: int = 99
    validation_seed: int = 202_700
    opponent_profiles: tuple[str, ...] = (
        "random",
        "calling_station",
        "tight_aggressive",
    )
    league_opponents: int = 3
    keep_full_checkpoints: int = 3
    tournament_evaluation_games_per_player: int = 0
    tournament_opponent_profiles: tuple[str, ...] = ()
    tournament_max_hands: int = 10_000

    def validate(self) -> None:
        positive = {
            "target_iteration": self.target_iteration,
            "traversals_per_player": self.traversals_per_player,
            "traversal_workers": self.traversal_workers,
            "advantage_steps": self.advantage_steps,
            "batch_size": self.batch_size,
            "evaluate_every": self.evaluate_every,
            "checkpoint_every": self.checkpoint_every,
            "snapshot_every": self.snapshot_every,
            "evaluation_games_per_player": self.evaluation_games_per_player,
            "league_games_per_player": self.league_games_per_player,
            "keep_full_checkpoints": self.keep_full_checkpoints,
            "tournament_max_hands": self.tournament_max_hands,
        }
        bad = [name for name, value in positive.items() if value <= 0]
        if bad:
            raise ValueError(f"campaign values must be positive: {', '.join(bad)}")
        if (
            self.policy_steps < 0
            or self.league_opponents < 0
            or self.tournament_evaluation_games_per_player < 0
        ):
            raise ValueError(
                "policy_steps, league_opponents, and tournament evaluation games "
                "cannot be negative"
            )
        unknown = set(self.opponent_profiles) - set(OPPONENT_PROFILES)
        if unknown:
            raise ValueError(f"unknown opponent profiles: {sorted(unknown)}")
        unknown_tournament = set(self.tournament_opponent_profiles) - set(
            OPPONENT_PROFILES
        )
        if unknown_tournament:
            raise ValueError(
                "unknown tournament opponent profiles: "
                f"{sorted(unknown_tournament)}"
            )
        if bool(self.tournament_evaluation_games_per_player) != bool(
            self.tournament_opponent_profiles
        ):
            raise ValueError(
                "tournament evaluation games and opponent profiles must either "
                "both be configured or both be disabled"
            )


class ProductionCampaign:
    """Resumable multi-day runner with versioned checkpoints and evaluation."""

    def __init__(
        self,
        trainer: ThreePlayerNeuralCFR,
        artifact_dir: str | Path,
        config: CampaignConfig,
    ):
        config.validate()
        if not trainer.can_resume_training:
            raise ValueError("production campaign requires a full resumable trainer")
        self.trainer = trainer
        self.artifact_dir = Path(artifact_dir)
        self.config = config
        self.checkpoint_dir = self.artifact_dir / "checkpoints"
        self.snapshot_dir = self.artifact_dir / "snapshots"
        self.evaluation_dir = self.artifact_dir / "evaluations"
        self.metrics_path = self.artifact_dir / "metrics.jsonl"
        self.latest_manifest = self.artifact_dir / "latest.json"
        self.run_config_path = self.artifact_dir / "run_config.json"
        self.run_config_history_path = self.artifact_dir / "run_config_history.jsonl"
        self.warm_start_path = self.artifact_dir / "warm_start.json"
        self.baseline_path = self.snapshot_dir / "initial_policy.pt"
        self.champion_path = self.snapshot_dir / "champion_policy.pt"
        self._last_checkpoint_iteration: int | None = None
        for directory in (
            self.checkpoint_dir,
            self.snapshot_dir,
            self.evaluation_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self._validate_or_create_run_config()
        self._ensure_baseline()

    def _run_config_payload(self) -> dict[str, Any]:
        initialization = (
            json.loads(self.warm_start_path.read_text(encoding="utf-8"))
            if self.warm_start_path.exists()
            else {"kind": "random_initialization"}
        )
        return {
            "version": 1,
            "campaign": _json_safe(asdict(self.config)),
            "environment": {
                "stack_size": self.trainer.env.stack_size,
                "sb": self.trainer.env.sb,
                "bb": self.trainer.env.bb,
                "tournament_total_chips": self.trainer.tournament_total_chips,
            },
            "trainer": {
                "hidden": self.trainer.hidden,
                "blocks": self.trainer.blocks,
                "network_architecture": self.trainer.network_architecture,
                "learning_rate": self.trainer.learning_rate,
                "max_history": self.trainer.max_history,
                "max_nodes_per_traversal": self.trainer.max_nodes_per_traversal,
                "max_depth": self.trainer.max_depth,
                "max_strategy_importance": self.trainer.max_strategy_importance,
                "exploration": self.trainer.exploration,
                "reinitialize_advantage_each_iteration": (
                    self.trainer.reinitialize_advantage_each_iteration
                ),
                "advantage_reinitialize_from_iteration": (
                    self.trainer.advantage_reinitialize_from_iteration
                ),
                "advantage_fit_every": self.trainer.advantage_fit_every,
                "include_tournament_features": (
                    self.trainer.include_tournament_features
                ),
                "variable_stack_training": self.trainer.variable_stack_training,
                "tournament_total_chips": self.trainer.tournament_total_chips,
                "heads_up_root_fraction": self.trainer.heads_up_root_fraction,
                "continuation_root_fraction": (
                    self.trainer.continuation_root_fraction
                ),
                "minimum_live_stack": self.trainer.minimum_live_stack,
                "root_stack_concentration": self.trainer.root_stack_concentration,
                "continuation_capacity": self.trainer.continuation_capacity,
                "advantage_capacity": self.trainer.advantage_buffers[0].capacity,
                "policy_capacity": self.trainer.policy_buffers[0].capacity,
                "recent_capacity": self.trainer.recent_capacity,
                "recent_window_iterations": self.trainer.recent_window_iterations,
                "recent_batch_fraction": self.trainer.recent_batch_fraction,
            },
            "initialization": _json_safe(initialization),
        }

    def _validate_or_create_run_config(self) -> None:
        current = self._run_config_payload()
        if not self.run_config_path.exists():
            _atomic_json(self.run_config_path, current)
            return
        previous = json.loads(self.run_config_path.read_text(encoding="utf-8"))
        previous_campaign = dict(previous.get("campaign", {}))
        current_campaign = dict(current["campaign"])
        # Run configurations created before tournament support lack these
        # fields. Backfill only their legacy meanings so old fixed-stack
        # campaigns remain resumable; tournament settings are locked normally.
        previous_campaign.setdefault("tournament_evaluation_games_per_player", 0)
        previous_campaign.setdefault("tournament_opponent_profiles", [])
        previous_campaign.setdefault("tournament_max_hands", 10_000)
        previous_campaign.setdefault("traversal_workers", 1)
        previous_environment = dict(previous.get("environment", {}))
        previous_environment.setdefault(
            "tournament_total_chips",
            3.0 * float(previous_environment.get("stack_size", self.trainer.env.stack_size)),
        )
        previous_trainer = dict(previous.get("trainer", {}))
        legacy_trainer_defaults = {
            "network_architecture": "residual_mlp",
            "include_tournament_features": False,
            "variable_stack_training": False,
            "tournament_total_chips": previous_environment["tournament_total_chips"],
            "heads_up_root_fraction": 0.25,
            "continuation_root_fraction": 0.25,
            "minimum_live_stack": float(previous_environment.get("sb", self.trainer.env.sb)),
            "root_stack_concentration": 0.7,
            "continuation_capacity": 2_048,
            "recent_capacity": 0,
            "recent_window_iterations": 100,
            "recent_batch_fraction": 0.5,
        }
        for key, value in legacy_trainer_defaults.items():
            previous_trainer.setdefault(key, value)
        if previous_environment != current["environment"]:
            raise ValueError("artifact directory belongs to another game configuration")
        current_trainer = dict(current["trainer"])
        trainer_changes = {
            key: {
                "previous": previous_trainer.get(key),
                "current": current_trainer.get(key),
            }
            for key in sorted(set(previous_trainer) | set(current_trainer))
            if previous_trainer.get(key) != current_trainer.get(key)
        }
        # These controls change future sampling effort without changing tensor
        # shapes or the meaning of an existing information-state encoding.
        # Explicitly allowing and recording them supports an intentional
        # continuation while keeping architectural changes locked.
        resume_mutable_trainer_fields = {
            "learning_rate",
            "exploration",
            "max_strategy_importance",
            "max_depth",
            "max_nodes_per_traversal",
            "advantage_capacity",
            "policy_capacity",
            "recent_capacity",
            "recent_window_iterations",
            "recent_batch_fraction",
        }
        locked_trainer_changes = sorted(
            set(trainer_changes) - resume_mutable_trainer_fields
        )
        if locked_trainer_changes:
            raise ValueError(
                "trainer settings changed on resume for locked field(s): "
                f"{', '.join(locked_trainer_changes)}; use a new artifact "
                "directory or restore the recorded run configuration"
            )
        if (
            "initialization" in previous
            and previous["initialization"] != current["initialization"]
        ):
            raise ValueError("artifact directory uses another policy initialization")

        changes = {
            key: {
                "previous": previous_campaign.get(key),
                "current": current_campaign.get(key),
            }
            for key in sorted(set(previous_campaign) | set(current_campaign))
            if previous_campaign.get(key) != current_campaign.get(key)
        }
        # These values control how often work is reported or when the absolute
        # campaign stops. They do not alter an already-collected CFR sample or
        # the meaning of the six fitted networks, so changing them is safe when
        # resuming. Training and benchmark-identity settings remain locked.
        resume_mutable_fields = {
            "target_iteration",
            "evaluate_every",
            "traversal_workers",
            "traversals_per_player",
            "advantage_steps",
            "policy_steps",
            "batch_size",
            "keep_full_checkpoints",
        }
        locked_changes = sorted(set(changes) - resume_mutable_fields)
        if locked_changes:
            raise ValueError(
                "campaign settings changed on resume for locked field(s): "
                f"{', '.join(locked_changes)}; use a new artifact directory or "
                "restore the recorded run configuration"
            )
        if int(current_campaign["target_iteration"]) < self.trainer.iteration:
            raise ValueError("target_iteration is behind the resumed checkpoint")

        if changes or trainer_changes:
            self.run_config_history_path.parent.mkdir(parents=True, exist_ok=True)
            history_row = {
                "changed_at_utc": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                ),
                "resumed_iteration": int(self.trainer.iteration),
                "campaign_changes": changes,
                "trainer_changes": trainer_changes,
            }
            with self.run_config_history_path.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(_json_safe(history_row), separators=(",", ":")) + "\n"
                )
                stream.flush()
            _atomic_json(self.run_config_path, current)

    def _ensure_baseline(self) -> None:
        if self.baseline_path.exists():
            baseline = load_policy_snapshot(self.baseline_path, device=self.trainer.device)
            environment = baseline.metadata.get("environment", {})
            if environment and not math.isclose(
                float(environment.get("stack_size", self.trainer.env.stack_size)),
                float(self.trainer.env.stack_size),
            ):
                raise ValueError("initial policy snapshot belongs to another game")
            if int(baseline.metadata.get("input_dim", -1)) != self.trainer.input_dim:
                raise ValueError("initial policy snapshot uses another encoder")
            self.baseline = baseline
            return
        if self.trainer.iteration != 0:
            raise FileNotFoundError(
                "cannot reconstruct the initial baseline after training; restore "
                f"{self.baseline_path} or start a new artifact directory"
            )
        save_policy_snapshot(
            self.trainer,
            self.baseline_path,
            metadata={"purpose": "fixed_initial_baseline"},
        )
        self.baseline = load_policy_snapshot(
            self.baseline_path, device=self.trainer.device
        )

    def _cached_baseline_evaluations(self) -> dict[str, EvaluationResult]:
        cached: dict[str, EvaluationResult] = {}
        baseline_dir = (
            self.evaluation_dir
            / "initial_baseline"
            / f"seed_{self.config.validation_seed}_games_{self.config.evaluation_games_per_player}"
        )
        for profile in self.config.opponent_profiles:
            path = baseline_dir / f"{profile}_hands.csv"
            if path.exists():
                hands = pd.read_csv(path)
                cached[profile] = EvaluationResult(
                    profile, _clustered_summary(hands), hands
                )
        return cached

    def _append_metric(self, row: dict[str, Any]) -> None:
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with self.metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(_json_safe(row), separators=(",", ":")) + "\n")
            stream.flush()

    def _save_checkpoint(
        self, *, emergency: bool = False, failed: bool = False
    ) -> Path:
        if emergency and failed:
            raise ValueError("a checkpoint cannot be both emergency and failed")
        suffix = "_emergency" if emergency else ("_failed" if failed else "")
        path = self.checkpoint_dir / f"step_{self.trainer.iteration:08d}{suffix}.pt"
        self.trainer.save(path, include_buffers=True)
        self._last_checkpoint_iteration = self.trainer.iteration
        _atomic_json(
            self.latest_manifest,
            {
                "version": 1,
                "iteration": self.trainer.iteration,
                "checkpoint": str(path.resolve()),
                "emergency": emergency,
                "failed": failed,
                "last_fitted_iteration": self.trainer.last_fitted_iteration,
                "incomplete_fit": (
                    self.trainer.last_fitted_iteration < self.trainer.iteration
                ),
                "saved_at_unix": time.time(),
                "campaign": asdict(self.config),
            },
        )
        self._prune_full_checkpoints()
        return path

    def _prune_full_checkpoints(self) -> None:
        """Retain only the newest configured full checkpoints.

        Policy-only league snapshots are intentionally unaffected.  Every
        deletion target comes from the dedicated checkpoint directory and is
        resolved/validated before removal.
        """
        checkpoint_root = self.checkpoint_dir.resolve()
        paths = sorted(
            self.checkpoint_dir.glob("step_*.pt"),
            key=lambda candidate: (candidate.stat().st_mtime_ns, candidate.name),
        )
        for old_path in paths[: -self.config.keep_full_checkpoints]:
            resolved = old_path.resolve()
            if not resolved.is_relative_to(checkpoint_root):
                raise RuntimeError(f"refusing to remove checkpoint outside {checkpoint_root}")
            resolved.unlink()

    def _milestone_snapshot(self) -> Path:
        path = self.snapshot_dir / f"policy_{self.trainer.iteration:08d}.pt"
        return save_policy_snapshot(
            self.trainer,
            path,
            metadata={"purpose": "historical_league"},
        )

    def _league_paths(self) -> list[Path]:
        paths = sorted(self.snapshot_dir.glob("policy_*.pt"))
        current_name = f"policy_{self.trainer.iteration:08d}.pt"
        paths = [path for path in paths if path.name != current_name]
        return paths[-self.config.league_opponents :]

    def _evaluate(self, row: dict[str, Any]) -> None:
        cached_baselines = self._cached_baseline_evaluations()
        suite = evaluate_benchmark_suite(
            self.trainer,
            profiles=self.config.opponent_profiles,
            games_per_player=self.config.evaluation_games_per_player,
            seed=self.config.validation_seed,
            baseline_policy_nets=self.baseline.policy_nets,
            baseline_results=cached_baselines,
        )
        row.update(suite.metrics)
        iteration_dir = self.evaluation_dir / f"step_{self.trainer.iteration:08d}"
        iteration_dir.mkdir(parents=True, exist_ok=True)
        for profile, result in suite.evaluations.items():
            result.hands.to_csv(iteration_dir / f"{profile}_hands.csv", index=False)
        if self.config.tournament_evaluation_games_per_player:
            for profile in self.config.tournament_opponent_profiles:
                tournament_result = evaluate_tournaments_against_profile(
                    self.trainer,
                    profile,
                    tournaments_per_player=(
                        self.config.tournament_evaluation_games_per_player
                    ),
                    seed=self.config.validation_seed + 500_000,
                    max_hands=self.config.tournament_max_hands,
                )
                for key, value in tournament_result.summary.items():
                    row[f"tournament_{profile}_{key}"] = value
                tournament_result.tournaments.to_csv(
                    iteration_dir / f"tournament_{profile}_results.csv",
                    index=False,
                )
        baseline_dir = (
            self.evaluation_dir
            / "initial_baseline"
            / f"seed_{self.config.validation_seed}_games_{self.config.evaluation_games_per_player}"
        )
        baseline_dir.mkdir(parents=True, exist_ok=True)
        for profile, result in suite.baseline_evaluations.items():
            baseline_path = baseline_dir / f"{profile}_hands.csv"
            if not baseline_path.exists():
                result.hands.to_csv(baseline_path, index=False)
        league_values: list[float] = []
        for path in self._league_paths():
            snapshot = load_policy_snapshot(path, device=self.trainer.device)
            report = self.trainer.evaluate_vs_snapshot(
                snapshot.policy_nets,
                games_per_player=self.config.league_games_per_player,
            )
            league_values.append(float(report["mean_ev_bb"]))
        row["league_opponents"] = float(len(league_values))
        row["league_mean_ev_bb"] = (
            float(np.mean(league_values)) if league_values else float("nan")
        )
        row["league_worst_ev_bb"] = (
            min(league_values) if league_values else float("nan")
        )

        best_manifest = self.artifact_dir / "champion.json"
        previous_lcb = -float("inf")
        if best_manifest.exists():
            previous = json.loads(best_manifest.read_text(encoding="utf-8"))
            previous_lcb = float(previous["benchmark_composite_lcb95_bb"])
        current_lcb = float(row["benchmark_composite_lcb95_bb"])
        promoted = current_lcb > previous_lcb
        row["promoted_to_champion"] = float(promoted)
        if promoted:
            save_policy_snapshot(
                self.trainer,
                self.champion_path,
                metadata={
                    "purpose": "validation_champion",
                    "benchmark_composite_lcb95_bb": current_lcb,
                },
            )
            _atomic_json(
                best_manifest,
                {
                    "iteration": self.trainer.iteration,
                    "benchmark_composite_lcb95_bb": current_lcb,
                    "policy": str(self.champion_path.resolve()),
                },
            )

    def run(
        self,
        *,
        on_iteration: Callable[[dict[str, Any]], None] | None = None,
        on_evaluation: Callable[[pd.DataFrame], None] | None = None,
    ) -> list[dict[str, float]]:
        """Train to the configured absolute target, saving on interruption."""
        if (
            self.trainer.iteration >= self.config.target_iteration
            and self.trainer.last_fitted_iteration >= self.trainer.iteration
        ):
            return self.trainer.metrics
        interrupted = False
        failed = False
        try:
            if self.trainer.last_fitted_iteration < self.trainer.iteration:
                print(
                    "Detected an incomplete network fit at iteration "
                    f"{self.trainer.iteration}; refitting all six networks from "
                    "the saved cumulative reservoirs before traversal resumes...",
                    flush=True,
                )
                recovery = self.trainer.recover_incomplete_fit(
                    advantage_steps=self.config.advantage_steps,
                    policy_steps=self.config.policy_steps,
                    batch_size=self.config.batch_size,
                )
                _atomic_json(
                    self.artifact_dir / "last_recovery.json",
                    _json_safe(recovery),
                )
                self._save_checkpoint()
                print(
                    f"Recovery fit completed in {recovery['recovery_fit_seconds']:.2f}s; "
                    f"iteration {self.trainer.iteration} is safe to continue.",
                    flush=True,
                )
            if self.trainer.iteration >= self.config.target_iteration:
                return self.trainer.metrics
            while self.trainer.iteration < self.config.target_iteration:
                row = self.trainer.train_iteration(
                    traversals_per_player=self.config.traversals_per_player,
                    advantage_steps=self.config.advantage_steps,
                    policy_steps=self.config.policy_steps,
                    batch_size=self.config.batch_size,
                    traversal_workers=self.config.traversal_workers,
                )
                should_snapshot = self.trainer.iteration % self.config.snapshot_every == 0
                if should_snapshot:
                    self._milestone_snapshot()
                should_evaluate = self.trainer.iteration % self.config.evaluate_every == 0
                if should_evaluate:
                    self._evaluate(row)
                self._append_metric(row)
                if on_iteration is not None:
                    on_iteration(row)
                if should_evaluate and on_evaluation is not None:
                    on_evaluation(pd.DataFrame(self.trainer.metrics))
                if self.trainer.iteration % self.config.checkpoint_every == 0:
                    self._save_checkpoint()
        except KeyboardInterrupt:
            interrupted = True
            raise
        except Exception:
            failed = True
            raise
        finally:
            if (
                interrupted
                or failed
                or self._last_checkpoint_iteration != self.trainer.iteration
            ):
                self._save_checkpoint(emergency=interrupted, failed=failed)
        return self.trainer.metrics


def resolve_latest_checkpoint(artifact_dir: str | Path) -> Path | None:
    manifest = Path(artifact_dir) / "latest.json"
    if not manifest.exists():
        return None
    value = json.loads(manifest.read_text(encoding="utf-8"))
    path = Path(value["checkpoint"])
    if not path.exists():
        raise FileNotFoundError(f"latest checkpoint is missing: {path}")
    return path


def load_or_create_trainer(
    env: ThreePlayerHoldemEnv,
    artifact_dir: str | Path,
    *,
    device: str | torch.device,
    trainer_kwargs: dict[str, Any],
    warm_start_policy: str | Path | None = None,
) -> tuple[ThreePlayerNeuralCFR, bool]:
    artifact_dir = Path(artifact_dir)
    latest = resolve_latest_checkpoint(artifact_dir)
    if latest is not None:
        trainer = ThreePlayerNeuralCFR.load(latest, env, device=device)
        if not trainer.can_resume_training:
            raise RuntimeError("latest checkpoint is not resumable")
        return trainer, True
    trainer = ThreePlayerNeuralCFR(env, device=device, **trainer_kwargs)
    if warm_start_policy is not None:
        source = Path(warm_start_policy).resolve()
        report = trainer.warm_start_policy(source)
        provenance = {
            "kind": "policy_warm_start",
            **report,
            "source": str(source),
        }
        _atomic_json(artifact_dir / "warm_start.json", _json_safe(provenance))
    return trainer, False


def metrics_frame(metrics: Iterable[dict[str, Any]]) -> pd.DataFrame:
    frame = metrics.copy() if isinstance(metrics, pd.DataFrame) else pd.DataFrame(list(metrics))
    if frame.empty:
        return frame
    frame["iteration"] = frame["iteration"].astype(int)
    return frame.sort_values("iteration").drop_duplicates("iteration", keep="last")


def plot_training_dashboard(metrics: Iterable[dict[str, Any]]) -> plt.Figure:
    """Nine-panel production dashboard, including held-out strength metrics."""
    frame = metrics_frame(metrics)
    if frame.empty:
        raise ValueError("no metrics are available")
    figure, axes = plt.subplots(3, 3, figsize=(20, 14))

    evaluation = frame.dropna(subset=["benchmark_composite_ev_bb"]) if "benchmark_composite_ev_bb" in frame else pd.DataFrame()
    if not evaluation.empty:
        for profile, color in zip(
            ("random", "calling_station", "tight_aggressive"),
            ("tab:green", "tab:blue", "tab:red"),
        ):
            mean_key = f"benchmark_{profile}_mean_ev_bb"
            se_key = f"benchmark_{profile}_clustered_stderr_bb"
            if mean_key in evaluation:
                axes[0, 0].errorbar(
                    evaluation["iteration"],
                    evaluation[mean_key],
                    yerr=1.96 * evaluation[se_key],
                    marker="o",
                    label=profile.replace("_", " "),
                    color=color,
                )
        axes[0, 0].axhline(0, color="black", lw=1)
        axes[0, 0].legend()
        axes[0, 0].set_title("Held-out EV vs fixed opponents")
        axes[0, 0].set_ylabel("BB / hand (clustered 95% CI)")

        delta_columns = [
            f"benchmark_{profile}_delta_ev_bb"
            for profile in ("random", "calling_station", "tight_aggressive")
            if f"benchmark_{profile}_delta_ev_bb" in evaluation
        ]
        if delta_columns:
            evaluation.plot(
                x="iteration", y=delta_columns, marker="o", ax=axes[0, 1]
            )
        axes[0, 1].axhline(0, color="black", lw=1)
        axes[0, 1].set_title("Paired EV change vs initial policy")

        league_columns = [
            column
            for column in ("league_mean_ev_bb", "league_worst_ev_bb")
            if column in evaluation
        ]
        if league_columns:
            evaluation.plot(
                x="iteration", y=league_columns, marker="o", ax=axes[0, 2]
            )
        axes[0, 2].axhline(0, color="black", lw=1)
        axes[0, 2].set_title("Current policy vs historical league")

    loss_columns = [
        *(f"adv_loss_p{player}" for player in range(3)),
        *(f"policy_loss_p{player}" for player in range(3)),
    ]
    frame.plot(x="iteration", y=loss_columns, logy=True, ax=axes[1, 0])
    axes[1, 0].set_title("All six network losses")

    timing_columns = [
        column
        for column in (
            "traversal_seconds",
            "advantage_fit_seconds",
            "policy_fit_seconds",
        )
        if column in frame
    ]
    frame.plot(x="iteration", y=timing_columns, ax=axes[1, 1])
    axes[1, 1].set_title("Phase timing")
    axes[1, 1].set_ylabel("seconds")

    frame.plot(
        x="iteration",
        y=["nodes", "rollouts"],
        ax=axes[1, 2],
    )
    axes[1, 2].set_title("Tree nodes and rollout cutoffs")

    buffer_columns = [
        *(f"adv_buffer_p{player}" for player in range(3)),
        *(f"policy_buffer_p{player}" for player in range(3)),
        *(
            f"recent_adv_buffer_p{player}"
            for player in range(3)
            if f"recent_adv_buffer_p{player}" in frame
        ),
        *(
            f"recent_policy_buffer_p{player}"
            for player in range(3)
            if f"recent_policy_buffer_p{player}" in frame
        ),
    ]
    frame.plot(x="iteration", y=buffer_columns, ax=axes[2, 0])
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
    frame.plot(x="iteration", y=memory_columns, ax=axes[2, 1])
    axes[2, 1].set_title("CUDA memory")
    axes[2, 1].set_ylabel("MiB")

    health_columns = [
        column
        for column in (
            "strategy_weight_ess_fraction",
            "importance_cap_fraction",
            "mean_policy_entropy",
        )
        if column in frame
    ]
    frame.plot(x="iteration", y=health_columns, ax=axes[2, 2])
    axes[2, 2].set_title("Sampling and policy health")

    for axis in axes.flat:
        axis.grid(alpha=0.25)
        axis.set_xlabel("iteration")
    figure.suptitle("Three-player neural CFR production dashboard", fontsize=16)
    figure.tight_layout()
    return figure


__all__ = [
    "BenchmarkSuiteResult",
    "CallingStationOpponent",
    "CampaignConfig",
    "EvaluationResult",
    "OPPONENT_PROFILES",
    "PolicySnapshot",
    "ProductionCampaign",
    "ScriptedOpponent",
    "TightAggressiveOpponent",
    "TournamentEvaluationResult",
    "UniformRandomOpponent",
    "evaluate_against_profile",
    "evaluate_benchmark_suite",
    "evaluate_tournaments_against_profile",
    "load_or_create_trainer",
    "load_policy_snapshot",
    "metrics_frame",
    "paired_improvement",
    "plot_training_dashboard",
    "resolve_latest_checkpoint",
    "save_policy_snapshot",
]
