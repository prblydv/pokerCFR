"""Held-out heads-up evaluation against reproducible scripted profiles."""

from __future__ import annotations

import json
import math
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from statistics import NormalDist, mean, stdev
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from heads_up_cfr import (
    DEFAULT_ROOT_STACK_DEPTHS_BB,
    ROOT_STACK_DISTRIBUTION_MIXED,
    HeadsUpNeuralCFR,
    POLICY_NETWORK_ARCHITECTURE,
)
from heads_up_engine import (
    ACTION_CALL,
    ACTION_CHECK,
    ACTION_FOLD,
    ACTION_MIN_RAISE,
    ACTION_NAMES,
    NUM_ACTIONS,
    evaluate_5card,
)
from heads_up_models import (
    CARD_FEATURES,
    CARD_STATE_PREFIX_FEATURES,
    HISTORY_OFFSET,
    build_policy_network,
    poker_relational_features,
)
from heads_up_ranges import (
    COMBO_FIRST_CARD,
    COMBO_HAND_CLASS_INDEX,
    COMBO_SECOND_CARD,
    HAND_CLASS_LABELS,
    NUM_OPPONENT_COMBOS,
    masked_range_probabilities,
    opponent_combo_index,
    valid_combo_mask_from_encoded,
)
from heads_up_reporting import save_range_reservoir_dashboard

RANGE_CATEGORY_NAMES = (
    "high_card",
    "one_pair",
    "two_pair",
    "three_of_a_kind",
    "straight",
    "flush",
    "full_house",
    "four_of_a_kind",
    "straight_flush",
)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False),
        encoding="utf-8",
    )
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
        if probability <= 0.0:
            continue
        fallback = action
        cumulative += float(probability)
        if threshold <= cumulative + 1e-12:
            return action
    return fallback


def _normalised_policy(
    weights: dict[int, float],
    legal: Sequence[int],
) -> torch.Tensor:
    result = torch.zeros(NUM_ACTIONS, dtype=torch.float32)
    legal_set = set(int(action) for action in legal)
    for action, weight in weights.items():
        if int(action) in legal_set and float(weight) > 0.0:
            result[int(action)] += float(weight)
    if float(result.sum()) <= 0.0:
        result[list(legal)] = 1.0
    return result / result.sum()


class ScriptedOpponent:
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
            return _normalised_policy(
                {ACTION_CHECK: 0.97, ACTION_MIN_RAISE: 0.03},
                legal,
            )
        if ACTION_CALL in legal:
            return _normalised_policy(
                {ACTION_CALL: 0.92, ACTION_FOLD: 0.08},
                legal,
            )
        return _normalised_policy({}, legal)


def _preflop_strength(cards: Sequence[int]) -> float:
    ranks = sorted((int(card) % 13 + 2 for card in cards), reverse=True)
    high, low = ranks
    suited = int(cards[0]) // 13 == int(cards[1]) // 13
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
    """Stable card-aware TAG benchmark copied from the prior campaign."""

    name = "tight_aggressive"

    def probabilities(self, env, state, player: int) -> torch.Tensor:
        legal = env.legal_actions(state)
        visible = list(state.hole[player]) + list(state.board)
        strength = (
            _preflop_strength(state.hole[player])
            if int(state.street) == 0
            else _postflop_strength(visible)
        )
        raises = [action for action in legal if action >= ACTION_MIN_RAISE]
        preferred_raise = raises[len(raises) // 2] if raises else None
        facing_bet = ACTION_CALL in legal
        weights: dict[int, float]
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
    policy_nets: list[torch.nn.Module]
    metadata: dict[str, Any]


def save_policy_snapshot(
    trainer: HeadsUpNeuralCFR,
    path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save the two deployable average-policy networks only."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 3,
        "kind": "heads_up_policy_snapshot",
        "iteration": trainer.iteration,
        "input_dim": trainer.input_dim,
        "hidden": trainer.hidden,
        "blocks": trainer.blocks,
        "network_architecture": trainer.network_architecture,
        "policy_network_architecture": POLICY_NETWORK_ARCHITECTURE,
        "range_schema_version": "exact_opponent_combos_v1_1326",
        "max_history": trainer.max_history,
        "action_names": tuple(ACTION_NAMES),
        "environment": {
            "starting_stack": trainer.env.starting_stack,
            "small_blind": trainer.env.small_blind,
            "big_blind": trainer.env.big_blind,
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
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> PolicySnapshot:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    version = int(payload.get("version", -1))
    if payload.get("kind") != "heads_up_policy_snapshot" or version not in (2, 3):
        raise ValueError("unsupported heads-up policy snapshot")
    if tuple(payload.get("action_names", ())) != tuple(ACTION_NAMES):
        raise ValueError("policy snapshot action space does not match")
    states = list(payload.get("policy_nets", ()))
    if len(states) != 2:
        raise ValueError("heads-up policy snapshot must contain two networks")
    networks = [
        build_policy_network(
            (
                str(payload["network_architecture"])
                if version == 2
                else str(payload["policy_network_architecture"])
            ),
            int(payload["input_dim"]),
            int(payload["hidden"]),
            int(payload["blocks"]),
        ).to(device)
        for _ in range(2)
    ]
    for network, state in zip(networks, states):
        network.load_state_dict(state)
        network.eval()
    metadata = dict(payload.get("metadata", {}))
    metadata.update(
        {
            "environment": dict(payload.get("environment", {})),
            "input_dim": int(payload["input_dim"]),
            "hidden": int(payload["hidden"]),
            "blocks": int(payload["blocks"]),
            "max_history": int(payload["max_history"]),
            "network_architecture": str(payload["network_architecture"]),
            "policy_network_architecture": (
                str(payload["network_architecture"])
                if version == 2
                else str(payload["policy_network_architecture"])
            ),
            "has_range_head": bool(version >= 3),
            "range_schema_version": payload.get("range_schema_version"),
        }
    )
    return PolicySnapshot(int(payload["iteration"]), networks, metadata)


def _evaluate_against_profile_result(
    trainer,
    profile: str | ScriptedOpponent,
    *,
    games_per_seat: int = 300,
    seed: int = 402_700,
    inference_batch_size: int = 512,
    policy_nets: Sequence[torch.nn.Module] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if games_per_seat <= 0:
        raise ValueError("games_per_seat must be positive")
    if isinstance(profile, str):
        if profile not in OPPONENT_PROFILES:
            raise ValueError(f"unknown opponent profile {profile!r}")
        opponent = OPPONENT_PROFILES[profile]()
    else:
        opponent = profile

    profile_offset = sum(
        (index + 1) * ord(char)
        for index, char in enumerate(opponent.name)
    )
    records: list[tuple[int, int, float]] = []
    action_counts = [0] * NUM_ACTIONS
    for hero in range(2):
        evaluation_env = type(trainer.env)(
            starting_stack=trainer.env.starting_stack,
            small_blind=trainer.env.small_blind,
            big_blind=trainer.env.big_blind,
            seed=seed + profile_offset,
        )
        action_rng = random.Random(seed + 50_000 + profile_offset)
        states = [
            evaluation_env.new_hand(button=(hero + index) % 2)
            for index in range(int(games_per_seat))
        ]
        buttons = [int(state.button) for state in states]
        steps = [0] * len(states)
        while True:
            live = [index for index, state in enumerate(states) if not state.terminal]
            if not live:
                break
            hero_indices = [
                index
                for index in live
                if int(states[index].to_act) == hero
            ]
            hero_probabilities: dict[int, torch.Tensor] = {}
            if hero_indices:
                predictions = trainer.average_policy_batch(
                    [states[index] for index in hero_indices],
                    policy_nets=policy_nets,
                    batch_size=inference_batch_size,
                )
                hero_probabilities = dict(zip(hero_indices, predictions))
            for index in live:
                state = states[index]
                actor = int(state.to_act)
                probabilities = (
                    hero_probabilities[index]
                    if actor == hero
                    else opponent.probabilities(evaluation_env, state, actor)
                )
                action = _draw_action(probabilities, action_rng)
                action_counts[action] += 1
                states[index] = evaluation_env.step(state, action)
                steps[index] += 1
                if steps[index] > 512:
                    raise RuntimeError("evaluation hand exceeded 512 actions")
        for state, button in zip(states, buttons):
            records.append(
                (
                    hero,
                    button,
                    float(state.payoffs[hero]) / float(evaluation_env.bb),
                )
            )

    values = [record[2] for record in records]
    # Pair the two hero seats by deal index for a position-balanced stderr.
    clustered = [
        mean(
            records[hero * int(games_per_seat) + index][2]
            for hero in range(2)
        )
        for index in range(int(games_per_seat))
    ]
    stderr = (
        stdev(clustered) / math.sqrt(len(clustered))
        if len(clustered) > 1
        else 0.0
    )
    summary: dict[str, Any] = {
        "profile": opponent.name,
        "games_per_seat": int(games_per_seat),
        "hands": len(records),
        "mean_ev_bb": mean(values),
        "clustered_stderr_bb": stderr,
        "ci95_low_bb": mean(values) - 1.96 * stderr,
        "ci95_high_bb": mean(values) + 1.96 * stderr,
        "positive_hand_rate": sum(value > 0.0 for value in values) / len(values),
        "action_counts": action_counts,
    }
    for hero in range(2):
        seat_values = [record[2] for record in records if record[0] == hero]
        summary[f"seat_{hero}_ev_bb"] = mean(seat_values)
    for role, is_button in (("BTN_SB", True), ("BB", False)):
        role_values = [
            payoff
            for hero, button, payoff in records
            if bool(hero == button) == is_button
        ]
        summary[f"ev_{role}_bb"] = mean(role_values)
    hands = pd.DataFrame(
        records,
        columns=("hero", "button", "payoff_bb"),
    )
    hands["deal_index"] = hands.groupby("hero").cumcount()
    hands["hero_role"] = np.where(
        hands["hero"] == hands["button"],
        "BTN_SB",
        "BB",
    )
    hands["profile"] = opponent.name
    return summary, hands


def evaluate_against_profile(
    trainer,
    profile: str | ScriptedOpponent,
    *,
    games_per_seat: int = 300,
    seed: int = 402_700,
    inference_batch_size: int = 512,
    policy_nets: Sequence[torch.nn.Module] | None = None,
) -> dict[str, Any]:
    summary, _ = _evaluate_against_profile_result(
        trainer,
        profile,
        games_per_seat=games_per_seat,
        seed=seed,
        inference_batch_size=inference_batch_size,
        policy_nets=policy_nets,
    )
    return summary


def evaluate_benchmark_suite(
    trainer,
    *,
    profiles: Sequence[str] = (
        "random",
        "calling_station",
        "tight_aggressive",
    ),
    games_per_seat: int = 300,
    seed: int = 402_700,
    inference_batch_size: int = 512,
    baseline_policy_nets: Sequence[torch.nn.Module] | None = None,
    baseline_results: dict[str, dict[str, Any]] | None = None,
    reference_policy_nets: Sequence[torch.nn.Module] | None = None,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    means: list[float] = []
    stderrs: list[float] = []
    for profile in profiles:
        result = evaluate_against_profile(
            trainer,
            profile,
            games_per_seat=games_per_seat,
            seed=seed,
            inference_batch_size=inference_batch_size,
        )
        for key, value in result.items():
            if key in {"profile", "action_counts"}:
                continue
            metrics[f"benchmark_{profile}_{key}"] = float(value)
        means.append(float(result["mean_ev_bb"]))
        stderrs.append(float(result["clustered_stderr_bb"]))
        if baseline_policy_nets is not None or baseline_results is not None:
            baseline = (
                baseline_results.get(profile)
                if baseline_results is not None
                else None
            )
            if baseline is None:
                if baseline_policy_nets is None:
                    raise ValueError(
                        f"baseline result for {profile!r} is missing"
                    )
                baseline = evaluate_against_profile(
                    trainer,
                    profile,
                    games_per_seat=games_per_seat,
                    seed=seed,
                    inference_batch_size=inference_batch_size,
                    policy_nets=baseline_policy_nets,
                )
            baseline_mean = float(baseline["mean_ev_bb"])
            baseline_stderr = float(baseline["clustered_stderr_bb"])
            delta = float(result["mean_ev_bb"]) - baseline_mean
            delta_stderr = math.sqrt(
                float(result["clustered_stderr_bb"]) ** 2
                + baseline_stderr**2
            )
            probability = (
                NormalDist().cdf(delta / delta_stderr)
                if delta_stderr > 0.0
                else float(delta > 0.0)
            )
            metrics[f"benchmark_{profile}_baseline_ev_bb"] = baseline_mean
            metrics[f"benchmark_{profile}_delta_ev_bb"] = delta
            metrics[
                f"benchmark_{profile}_probability_delta_positive"
            ] = probability
    if reference_policy_nets is not None:
        result, _ = _evaluate_against_policy_result(
            trainer,
            reference_policy_nets,
            games_per_seat=games_per_seat,
            seed=seed,
            inference_batch_size=inference_batch_size,
        )
        for key, value in result.items():
            if key != "profile":
                metrics[f"benchmark_reference_policy_{key}"] = float(value)
        means.append(float(result["mean_ev_bb"]))
        stderrs.append(float(result["clustered_stderr_bb"]))
        if baseline_policy_nets is not None:
            baseline, _ = _evaluate_against_policy_result(
                trainer,
                reference_policy_nets,
                games_per_seat=games_per_seat,
                seed=seed,
                inference_batch_size=inference_batch_size,
                candidate_policy_nets=baseline_policy_nets,
            )
            baseline_mean = float(baseline["mean_ev_bb"])
            baseline_stderr = float(baseline["clustered_stderr_bb"])
            delta = float(result["mean_ev_bb"]) - baseline_mean
            delta_stderr = math.sqrt(
                float(result["clustered_stderr_bb"]) ** 2
                + baseline_stderr**2
            )
            metrics["benchmark_reference_policy_baseline_ev_bb"] = baseline_mean
            metrics["benchmark_reference_policy_delta_ev_bb"] = delta
            metrics["benchmark_reference_policy_probability_delta_positive"] = (
                NormalDist().cdf(delta / delta_stderr)
                if delta_stderr > 0.0
                else float(delta > 0.0)
            )
    composite = mean(means)
    composite_stderr = math.sqrt(sum(value * value for value in stderrs)) / len(
        stderrs
    )
    metrics["benchmark_composite_ev_bb"] = composite
    metrics["benchmark_composite_stderr_bb"] = composite_stderr
    metrics["benchmark_composite_lcb95_bb"] = (
        composite - 1.96 * composite_stderr
    )
    return metrics


@dataclass(frozen=True)
class CampaignConfig:
    target_iteration: int = 10_000
    traversals_per_player: int = 1_024
    traversal_workers: int = 1
    advantage_steps: int = 245
    policy_steps: int = 245
    batch_size: int = 4_096
    evaluate_every: int = 25
    checkpoint_every: int = 25
    snapshot_every: int = 100
    evaluation_games_per_player: int = 10_000
    range_evaluation_hands_per_opponent: int = 2_500
    range_evaluation_batch_size: int = 4_096
    range_training_hands_per_iteration: int = 2_048
    range_batch_size: int = 2_048
    league_games_per_player: int = 99
    validation_seed: int = 402_700
    opponent_profiles: tuple[str, ...] = (
        "random",
        "calling_station",
        "tight_aggressive",
    )
    reference_policy_path: str | None = None
    league_opponents: int = 3
    keep_full_checkpoints: int = 1
    root_stack_distribution: str = ROOT_STACK_DISTRIBUTION_MIXED
    root_stack_depths_bb: tuple[int, ...] = DEFAULT_ROOT_STACK_DEPTHS_BB

    def validate(self) -> None:
        positive = {
            name: value
            for name, value in asdict(self).items()
            if name
            in {
                "target_iteration",
                "traversals_per_player",
                "traversal_workers",
                "advantage_steps",
                "batch_size",
                "evaluate_every",
                "checkpoint_every",
                "snapshot_every",
                "evaluation_games_per_player",
                "range_evaluation_hands_per_opponent",
                "range_evaluation_batch_size",
                "range_training_hands_per_iteration",
                "range_batch_size",
                "league_games_per_player",
                "keep_full_checkpoints",
            }
        }
        bad = [name for name, value in positive.items() if int(value) <= 0]
        if bad:
            raise ValueError(f"campaign values must be positive: {', '.join(bad)}")
        if self.policy_steps < 0 or self.league_opponents < 0:
            raise ValueError("policy_steps and league_opponents cannot be negative")
        unknown = set(self.opponent_profiles) - set(OPPONENT_PROFILES)
        if unknown:
            raise ValueError(f"unknown opponent profiles: {sorted(unknown)}")
        if self.root_stack_distribution != ROOT_STACK_DISTRIBUTION_MIXED:
            raise ValueError("production requires the versioned mixed root stacks")
        if (
            len(self.root_stack_depths_bb) < 2
            or len(set(self.root_stack_depths_bb))
            != len(self.root_stack_depths_bb)
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in self.root_stack_depths_bb
            )
        ):
            raise ValueError(
                "root_stack_depths_bb must contain unique positive integers"
            )


def _evaluate_against_policy_result(
    trainer: HeadsUpNeuralCFR,
    opponent_policy_nets: Sequence[torch.nn.Module],
    *,
    games_per_seat: int,
    seed: int,
    inference_batch_size: int = 512,
    candidate_policy_nets: Sequence[torch.nn.Module] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if games_per_seat <= 0:
        raise ValueError("games_per_seat must be positive")
    records: list[tuple[int, int, float]] = []
    for hero in range(2):
        env = type(trainer.env)(
            starting_stack=trainer.env.starting_stack,
            small_blind=trainer.env.small_blind,
            big_blind=trainer.env.big_blind,
            seed=seed,
        )
        action_rng = random.Random(seed + 50_000)
        states = [
            # Keep the deal and button schedule identical when policy seats are
            # reversed. This makes the two hero assignments true reciprocal
            # common-random-number matches.
            env.new_hand(button=game % 2)
            for game in range(int(games_per_seat))
        ]
        buttons = [int(state.button) for state in states]
        steps = [0] * len(states)
        while True:
            live = [index for index, state in enumerate(states) if not state.terminal]
            if not live:
                break
            actors = {index: int(states[index].to_act) for index in live}
            for actor_is_hero in (True, False):
                indices = [
                    index
                    for index in live
                    if (actors[index] == hero) == actor_is_hero
                ]
                if not indices:
                    continue
                policy_nets = (
                    candidate_policy_nets
                    if actor_is_hero and candidate_policy_nets is not None
                    else trainer.policy_nets
                    if actor_is_hero
                    else opponent_policy_nets
                )
                probabilities = trainer.average_policy_batch(
                    [states[index] for index in indices],
                    policy_nets=policy_nets,
                    batch_size=inference_batch_size,
                )
                for index, action_probabilities in zip(indices, probabilities):
                    states[index] = env.step(
                        states[index],
                        _draw_action(action_probabilities, action_rng),
                    )
                    steps[index] += 1
                    if steps[index] > 512:
                        raise RuntimeError("evaluation hand exceeded 512 actions")
        for state, button in zip(states, buttons):
            records.append(
                (
                    hero,
                    button,
                    float(state.payoffs[hero]) / float(env.bb),
                )
            )
    values = [record[2] for record in records]
    clustered = [
        mean(
            records[hero * int(games_per_seat) + index][2]
            for hero in range(2)
        )
        for index in range(int(games_per_seat))
    ]
    stderr = (
        stdev(clustered) / math.sqrt(len(clustered))
        if len(clustered) > 1
        else 0.0
    )
    summary: dict[str, Any] = {
        "profile": "reference_policy",
        "games_per_seat": int(games_per_seat),
        "hands": len(records),
        "mean_ev_bb": mean(values),
        "clustered_stderr_bb": stderr,
        "ci95_low_bb": mean(values) - 1.96 * stderr,
        "ci95_high_bb": mean(values) + 1.96 * stderr,
        "positive_hand_rate": sum(value > 0.0 for value in values) / len(values),
    }
    for hero in range(2):
        seat_values = [record[2] for record in records if record[0] == hero]
        summary[f"seat_{hero}_ev_bb"] = mean(seat_values)
    for role, is_button in (("BTN_SB", True), ("BB", False)):
        role_values = [
            payoff
            for hero, button, payoff in records
            if bool(hero == button) == is_button
        ]
        summary[f"ev_{role}_bb"] = mean(role_values)
    hands = pd.DataFrame(
        records,
        columns=("hero", "button", "payoff_bb"),
    )
    hands["deal_index"] = hands.groupby("hero").cumcount()
    hands["hero_role"] = np.where(
        hands["hero"] == hands["button"],
        "BTN_SB",
        "BB",
    )
    hands["profile"] = "reference_policy"
    return summary, hands


def _evaluate_against_policy_nets(
    trainer: HeadsUpNeuralCFR,
    opponent_policy_nets: Sequence[torch.nn.Module],
    *,
    games_per_player: int,
    seed: int,
) -> float:
    summary, _ = _evaluate_against_policy_result(
        trainer,
        opponent_policy_nets,
        games_per_seat=games_per_player,
        seed=seed,
    )
    return float(summary["mean_ev_bb"])


@torch.inference_mode()
def _collect_independent_range_training_hands(
    trainer: HeadsUpNeuralCFR,
    *,
    profiles: Sequence[str],
    hands: int,
    seed: int,
    reference_policy_nets: Sequence[torch.nn.Module] | None,
    inference_batch_size: int,
    stack_depths_bb: Sequence[int],
) -> dict[str, float]:
    """Collect fresh single-trajectory hands without CFR branch duplication."""

    if int(hands) <= 0:
        raise ValueError("range training hands must be positive")
    rng = random.Random(int(seed))
    env = type(trainer.env)(
        starting_stack=trainer.env.starting_stack,
        small_blind=trainer.env.small_blind,
        big_blind=trainer.env.big_blind,
        seed=int(seed) + 1,
    )
    controller_names = ["self_play", *profiles]
    if reference_policy_nets is not None:
        controller_names.append("reference_policy")
    scripted = {
        name: OPPONENT_PROFILES[name]()
        for name in profiles
    }
    depths = tuple(int(value) for value in stack_depths_bb)
    states = []
    controllers: list[tuple[str, int]] = []
    for index in range(int(hands)):
        if index % 2 == 0:
            depth = rng.choice(depths)
            stacks_bb = (depth, depth)
        else:
            stacks_bb = tuple(rng.sample(depths, 2))
        stacks = tuple(int(value * env.bb) for value in stacks_bb)
        states.append(
            env.new_hand(
                button=index % 2,
                stacks=stacks,
            )
        )
        controllers.append(
            (
                controller_names[index % len(controller_names)],
                (index // len(controller_names)) % 2,
            )
        )

    encoded_rows: list[torch.Tensor] = []
    targets: list[int] = []
    players: list[int] = []
    hand_indices: list[int] = []
    action_steps = [0] * len(states)
    while True:
        live = [
            index for index, state in enumerate(states)
            if not state.terminal
        ]
        if not live:
            break
        current_indices: list[int] = []
        reference_indices: list[int] = []
        probabilities: dict[int, torch.Tensor] = {}
        for index in live:
            state = states[index]
            actor = int(state.to_act)
            legal = env.legal_actions(state)
            encoded_rows.append(
                trainer.encode(state, actor, legal).to(torch.float16)
            )
            targets.append(opponent_combo_index(state.hole[1 - actor]))
            players.append(actor)
            hand_indices.append(index)
            controller_name, controller_seat = controllers[index]
            if controller_name == "self_play" or actor != controller_seat:
                current_indices.append(index)
            elif controller_name == "reference_policy":
                reference_indices.append(index)
            else:
                probabilities[index] = scripted[
                    controller_name
                ].probabilities(env, state, actor)
        if current_indices:
            predictions = trainer.average_policy_batch(
                [states[index] for index in current_indices],
                batch_size=inference_batch_size,
            )
            probabilities.update(zip(current_indices, predictions))
        if reference_indices:
            if reference_policy_nets is None:
                raise RuntimeError("reference range controller is unavailable")
            predictions = trainer.average_policy_batch(
                [states[index] for index in reference_indices],
                policy_nets=reference_policy_nets,
                batch_size=inference_batch_size,
            )
            probabilities.update(zip(reference_indices, predictions))
        for index in live:
            states[index] = env.step(
                states[index],
                _draw_action(probabilities[index], rng),
            )
            action_steps[index] += 1
            if action_steps[index] > 512:
                raise RuntimeError("range training hand exceeded 512 actions")

    counts = Counter(zip(hand_indices, players))
    hand_weights = torch.tensor(
        [
            1.0 / float(counts[(hand, player)])
            for hand, player in zip(hand_indices, players)
        ],
        dtype=torch.float32,
    )
    global_ids = torch.tensor(
        [
            int(seed) * 10_000 + int(hand)
            for hand in hand_indices
        ],
        dtype=torch.int64,
    )
    added = trainer.add_range_training_samples(
        torch.stack(encoded_rows),
        torch.tensor(targets, dtype=torch.int16),
        torch.tensor(players, dtype=torch.uint8),
        global_ids,
        hand_weights,
    )
    return {
        "range_hands_generated": float(hands),
        "range_samples_generated": float(added),
        "range_mean_decisions_per_hand": float(added / max(1, hands)),
    }


def _build_fixed_range_holdout(
    trainer: HeadsUpNeuralCFR,
    *,
    profiles: Sequence[str],
    hands_per_opponent: int,
    seed: int,
    behavior_policy_nets: Sequence[torch.nn.Module],
    reference_policy_nets: Sequence[torch.nn.Module] | None,
    inference_batch_size: int,
) -> dict[str, Any]:
    """Generate immutable decision states without exposing hidden cards to inputs."""

    opponent_names = list(profiles)
    if reference_policy_nets is not None:
        opponent_names.append("reference_policy")
    encoded_rows: list[torch.Tensor] = []
    targets: list[int] = []
    players: list[int] = []
    streets: list[int] = []
    positions: list[int] = []
    opponent_ids: list[int] = []

    for opponent_id, opponent_name in enumerate(opponent_names):
        profile_offset = sum(
            (index + 1) * ord(char)
            for index, char in enumerate(opponent_name)
        )
        env = type(trainer.env)(
            starting_stack=trainer.env.starting_stack,
            small_blind=trainer.env.small_blind,
            big_blind=trainer.env.big_blind,
            seed=seed + profile_offset,
        )
        action_rng = random.Random(seed + 50_000 + profile_offset)
        heroes = [index % 2 for index in range(int(hands_per_opponent))]
        states = [
            env.new_hand(button=(index // 2) % 2)
            for index in range(int(hands_per_opponent))
        ]
        steps = [0] * len(states)
        scripted = (
            OPPONENT_PROFILES[opponent_name]()
            if opponent_name in OPPONENT_PROFILES
            else None
        )
        while True:
            live = [index for index, state in enumerate(states) if not state.terminal]
            if not live:
                break
            actors = {index: int(states[index].to_act) for index in live}
            hero_indices = [
                index for index in live if actors[index] == heroes[index]
            ]
            opponent_indices = [
                index for index in live if actors[index] != heroes[index]
            ]
            action_probabilities: dict[int, torch.Tensor] = {}
            if hero_indices:
                for index in hero_indices:
                    state = states[index]
                    hero = heroes[index]
                    legal = env.legal_actions(state)
                    encoded_rows.append(trainer.encode(state, hero, legal).to(torch.float16))
                    targets.append(opponent_combo_index(state.hole[1 - hero]))
                    players.append(hero)
                    streets.append(int(state.street))
                    positions.append(int(hero == int(state.button)))
                    opponent_ids.append(opponent_id)
                predictions = trainer.average_policy_batch(
                    [states[index] for index in hero_indices],
                    policy_nets=behavior_policy_nets,
                    batch_size=inference_batch_size,
                )
                action_probabilities.update(zip(hero_indices, predictions))
            if opponent_indices:
                if scripted is not None:
                    for index in opponent_indices:
                        state = states[index]
                        actor = actors[index]
                        action_probabilities[index] = scripted.probabilities(
                            env,
                            state,
                            actor,
                        )
                elif reference_policy_nets is not None:
                    predictions = trainer.average_policy_batch(
                        [states[index] for index in opponent_indices],
                        policy_nets=reference_policy_nets,
                        batch_size=inference_batch_size,
                    )
                    action_probabilities.update(zip(opponent_indices, predictions))
                else:
                    raise RuntimeError("reference-policy holdout has no reference")
            for index in live:
                states[index] = env.step(
                    states[index],
                    _draw_action(action_probabilities[index], action_rng),
                )
                steps[index] += 1
                if steps[index] > 512:
                    raise RuntimeError("range holdout hand exceeded 512 actions")

    if not encoded_rows:
        raise RuntimeError("fixed range holdout contains no hero decisions")
    return {
        "version": 1,
        "kind": "heads_up_fixed_range_holdout",
        "input_dim": int(trainer.input_dim),
        "max_history": int(trainer.max_history),
        "seed": int(seed),
        "hands_per_opponent": int(hands_per_opponent),
        "opponent_names": tuple(opponent_names),
        "information_states": torch.stack(encoded_rows),
        "opponent_combos": torch.tensor(targets, dtype=torch.int16),
        "players": torch.tensor(players, dtype=torch.uint8),
        "streets": torch.tensor(streets, dtype=torch.uint8),
        "positions": torch.tensor(positions, dtype=torch.uint8),
        "opponent_ids": torch.tensor(opponent_ids, dtype=torch.uint8),
    }


@torch.inference_mode()
def evaluate_fixed_range_holdout(
    trainer: HeadsUpNeuralCFR,
    holdout: dict[str, Any],
    *,
    batch_size: int,
) -> dict[str, float]:
    if holdout.get("kind") != "heads_up_fixed_range_holdout":
        raise ValueError("unsupported opponent-range holdout")
    xs_all = holdout["information_states"]
    targets_all = holdout["opponent_combos"].to(torch.long)
    players_all = holdout["players"].to(torch.long)
    streets_all = holdout["streets"].to(torch.long)
    positions_all = holdout["positions"].to(torch.long)
    opponent_ids_all = holdout["opponent_ids"].to(torch.long)
    count = int(xs_all.shape[0])
    if not all(
        int(values.shape[0]) == count
        for values in (
            targets_all,
            players_all,
            streets_all,
            positions_all,
            opponent_ids_all,
        )
    ):
        raise ValueError("range holdout tensors have inconsistent lengths")

    results: dict[str, list[torch.Tensor]] = {
        name: []
        for name in (
            "nll",
            "uniform_nll",
            "true_probability",
            "top10",
            "top50",
            "rank",
            "entropy",
            "class_probability",
            "indices",
        )
    }
    combo_classes_cpu = COMBO_HAND_CLASS_INDEX
    for player in range(2):
        player_indices = torch.nonzero(
            players_all == player,
            as_tuple=False,
        ).flatten()
        network = trainer.policy_nets[player]
        if not hasattr(network, "forward_with_range"):
            raise ValueError("candidate policy has no opponent-range head")
        network.eval()
        network_device = next(network.parameters()).device
        for start in range(0, len(player_indices), int(batch_size)):
            indices = player_indices[start : start + int(batch_size)]
            xs = xs_all.index_select(0, indices).to(
                network_device,
                dtype=torch.float32,
            )
            targets = targets_all.index_select(0, indices).to(network_device)
            _, range_logits = network.forward_with_range(xs)
            valid = valid_combo_mask_from_encoded(xs)
            target_valid = valid.gather(1, targets.unsqueeze(1)).squeeze(1)
            if not bool(torch.all(target_valid)):
                raise RuntimeError("held-out opponent hand is blocker-invalid")
            masked_logits = range_logits.masked_fill(~valid, -1e9)
            log_probabilities = F.log_softmax(masked_logits, dim=1)
            probabilities = log_probabilities.exp()
            true_log = log_probabilities.gather(
                1,
                targets.unsqueeze(1),
            ).squeeze(1)
            true_probability = true_log.exp()
            true_logits = masked_logits.gather(
                1,
                targets.unsqueeze(1),
            ).squeeze(1)
            greater = ((masked_logits > true_logits.unsqueeze(1)) & valid).sum(
                dim=1
            )
            equal = ((masked_logits == true_logits.unsqueeze(1)) & valid).sum(
                dim=1
            )
            average_rank = greater.to(torch.float32) + (
                equal.to(torch.float32) + 1.0
            ) / 2.0
            top10 = (
                masked_logits.topk(10, dim=1).indices
                == targets.unsqueeze(1)
            ).any(dim=1)
            top50 = (
                masked_logits.topk(50, dim=1).indices
                == targets.unsqueeze(1)
            ).any(dim=1)
            entropy = -(probabilities * log_probabilities).sum(dim=1)
            combo_classes = combo_classes_cpu.to(network_device)
            target_classes = combo_classes.index_select(0, targets)
            class_probability = (
                probabilities
                * (combo_classes.unsqueeze(0) == target_classes.unsqueeze(1))
            ).sum(dim=1)
            values = {
                "nll": -true_log,
                "uniform_nll": valid.sum(dim=1).to(torch.float32).log(),
                "true_probability": true_probability,
                "top10": top10.to(torch.float32),
                "top50": top50.to(torch.float32),
                "rank": average_rank,
                "entropy": entropy,
                "class_probability": class_probability,
                "indices": indices,
            }
            for name, tensor in values.items():
                results[name].append(tensor.detach().cpu())

    ordered: dict[str, torch.Tensor] = {}
    concatenated_indices = torch.cat(results["indices"]).to(torch.long)
    order = torch.argsort(concatenated_indices)
    for name in results:
        if name == "indices":
            continue
        ordered[name] = torch.cat(results[name]).index_select(0, order)

    metrics: dict[str, float] = {}

    def add_group(prefix: str, mask: torch.Tensor) -> None:
        if not bool(mask.any()):
            return
        metrics[f"{prefix}nll"] = float(ordered["nll"][mask].mean())
        metrics[f"{prefix}uniform_nll"] = float(
            ordered["uniform_nll"][mask].mean()
        )
        metrics[f"{prefix}information_gain"] = float(
            (
                ordered["uniform_nll"][mask]
                - ordered["nll"][mask]
            ).mean()
        )
        metrics[f"{prefix}true_probability"] = float(
            ordered["true_probability"][mask].mean()
        )
        metrics[f"{prefix}top10_accuracy"] = float(
            ordered["top10"][mask].mean()
        )
        metrics[f"{prefix}top50_accuracy"] = float(
            ordered["top50"][mask].mean()
        )
        metrics[f"{prefix}mean_rank"] = float(ordered["rank"][mask].mean())
        metrics[f"{prefix}entropy"] = float(ordered["entropy"][mask].mean())
        metrics[f"{prefix}correct_class_probability"] = float(
            ordered["class_probability"][mask].mean()
        )
        metrics[f"{prefix}decisions"] = float(mask.sum())

    add_group("range_", torch.ones(count, dtype=torch.bool))
    for street, name in enumerate(("preflop", "flop", "turn", "river")):
        add_group(f"range_{name}_", streets_all == street)
    for opponent_id, name in enumerate(holdout["opponent_names"]):
        add_group(f"range_vs_{name}_", opponent_ids_all == opponent_id)
    add_group("range_btn_sb_", positions_all == 1)
    add_group("range_bb_", positions_all == 0)
    return metrics


@torch.inference_mode()
def range_reservoir_statistics(
    trainer: HeadsUpNeuralCFR,
    *,
    maximum_rows_per_player: int = 100_000,
) -> dict[str, Any]:
    """Summarise a deterministic representative slice of the range replay."""

    fields = []
    for buffer in trainer.range_buffers:
        count = min(len(buffer), int(maximum_rows_per_player))
        if count <= 0:
            continue
        if count == len(buffer):
            indices = torch.arange(count)
        else:
            indices = torch.linspace(
                0,
                len(buffer) - 1,
                count,
            ).round().to(torch.long)
        fields.append(buffer._fields_at_indices(indices))
    if not fields:
        return {
            "sampled_rows": 0,
            "total_rows": 0,
            "street_percent": {},
            "street_count": {},
            "starting_hand_matrix_percent": [[0.0] * 13 for _ in range(13)],
            "made_hand_percent": {},
            "made_hand_count": {},
            "made_hand_by_street_percent": {},
            "made_hand_by_street_count": {},
        }
    xs = torch.cat([part[0] for part in fields]).to(torch.float32)
    combos = torch.cat([part[1] for part in fields]).to(torch.long)
    streets = xs[:, :4].argmax(dim=1)
    street_names = ("preflop", "flop", "turn", "river")
    street_percent = {
        name: 100.0 * float((streets == index).to(torch.float32).mean())
        for index, name in enumerate(street_names)
    }
    street_count = {
        name: int((streets == index).sum())
        for index, name in enumerate(street_names)
    }

    matrix = torch.zeros((13, 13), dtype=torch.float64)
    first = COMBO_FIRST_CARD.index_select(0, combos)
    second = COMBO_SECOND_CARD.index_select(0, combos)
    first_rank = first % 13
    second_rank = second % 13
    high = torch.maximum(first_rank, second_rank)
    low = torch.minimum(first_rank, second_rank)
    suited = first // 13 == second // 13
    pair = high == low
    for high_rank, low_rank, is_suited, is_pair in zip(
        high.tolist(),
        low.tolist(),
        suited.tolist(),
        pair.tolist(),
    ):
        high_index = 12 - int(high_rank)
        low_index = 12 - int(low_rank)
        if is_pair:
            row, column = high_index, high_index
        elif is_suited:
            row, column = high_index, low_index
        else:
            row, column = low_index, high_index
        matrix[row, column] += 1.0
    matrix *= 100.0 / max(1, len(combos))

    visible = xs[
        :, CARD_STATE_PREFIX_FEATURES:HISTORY_OFFSET
    ].reshape(-1, 7, CARD_FEATURES)
    cards = torch.zeros_like(visible)
    cards[:, 2:] = visible[:, 2:]
    for token, exact in enumerate((first, second)):
        cards[:, token, 17] = 1.0
        cards[
            torch.arange(len(cards)),
            token,
            exact % 13,
        ] = 1.0
        cards[
            torch.arange(len(cards)),
            token,
            13 + exact // 13,
        ] = 1.0
    relational = poker_relational_features(cards, xs[:, :4])
    categories = relational[:, 34:43].argmax(dim=1)

    def category_percent(mask: torch.Tensor) -> dict[str, float]:
        denominator = max(1, int(mask.sum()))
        return {
            name: 100.0
            * float(((categories == index) & mask).sum())
            / denominator
            for index, name in enumerate(RANGE_CATEGORY_NAMES)
        }

    def category_count(mask: torch.Tensor) -> dict[str, int]:
        return {
            name: int(((categories == index) & mask).sum())
            for index, name in enumerate(RANGE_CATEGORY_NAMES)
        }

    all_rows = torch.ones(len(categories), dtype=torch.bool)
    return {
        "sampled_rows": int(len(xs)),
        "total_rows": int(sum(len(buffer) for buffer in trainer.range_buffers)),
        "street_percent": street_percent,
        "street_count": street_count,
        "starting_hand_matrix_percent": matrix.tolist(),
        "rank_labels": list("AKQJT98765432"),
        "made_hand_percent": category_percent(all_rows),
        "made_hand_count": category_count(all_rows),
        "made_hand_by_street_percent": {
            name: category_percent(streets == index)
            for index, name in enumerate(street_names)
            if bool((streets == index).any())
        },
        "made_hand_by_street_count": {
            name: category_count(streets == index)
            for index, name in enumerate(street_names)
            if bool((streets == index).any())
        },
    }


class ProductionCampaign:
    """HU counterpart of the three-player resumable production runner."""

    def __init__(
        self,
        trainer: HeadsUpNeuralCFR,
        artifact_dir: str | Path,
        config: CampaignConfig,
    ) -> None:
        config.validate()
        if not trainer.can_resume_training:
            raise ValueError("production campaign requires a full resumable trainer")
        self.trainer = trainer
        self.artifact_dir = Path(artifact_dir)
        self.config = config
        self.reference_policy: PolicySnapshot | None = None
        if self.config.reference_policy_path is not None:
            reference_path = Path(self.config.reference_policy_path)
            if not reference_path.is_file():
                raise FileNotFoundError(
                    f"reference policy snapshot is missing: {reference_path}"
                )
            self.reference_policy = load_policy_snapshot(
                reference_path,
                device=self.trainer.device,
            )
            expected_environment = {
                "starting_stack": self.trainer.env.starting_stack,
                "small_blind": self.trainer.env.small_blind,
                "big_blind": self.trainer.env.big_blind,
            }
            if self.reference_policy.metadata["environment"] != expected_environment:
                raise ValueError("reference policy uses another HU game")
            if (
                self.reference_policy.metadata["input_dim"] != self.trainer.input_dim
                or self.reference_policy.metadata["max_history"]
                != self.trainer.max_history
                or self.reference_policy.metadata["network_architecture"]
                != self.trainer.network_architecture
            ):
                raise ValueError(
                    "reference policy encoder/architecture is incompatible"
                )
        self.checkpoint_dir = self.artifact_dir / "checkpoints"
        self.snapshot_dir = self.artifact_dir / "snapshots"
        self.evaluation_dir = self.artifact_dir / "evaluations"
        self.metrics_path = self.artifact_dir / "metrics.jsonl"
        self.latest_manifest = self.artifact_dir / "latest.json"
        self.run_config_path = self.artifact_dir / "run_config.json"
        self.run_config_history_path = self.artifact_dir / "run_config_history.jsonl"
        self.baseline_path = self.snapshot_dir / "initial_policy.pt"
        self.champion_path = self.snapshot_dir / "champion_policy.pt"
        self.range_holdout_path = (
            self.evaluation_dir / "fixed_range_holdout_v1.pt"
        )
        self._range_holdout: dict[str, Any] | None = None
        self._last_checkpoint_iteration: int | None = None
        for directory in (
            self.checkpoint_dir,
            self.snapshot_dir,
            self.evaluation_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self._validate_or_create_run_config()
        self._ensure_baseline()

    def _load_or_create_range_holdout(self) -> dict[str, Any]:
        if self._range_holdout is not None:
            return self._range_holdout
        hands_per_opponent = min(
            int(self.config.range_evaluation_hands_per_opponent),
            int(self.config.evaluation_games_per_player),
        )
        opponent_names = list(self.config.opponent_profiles)
        if self.reference_policy is not None:
            opponent_names.append("reference_policy")
        if self.range_holdout_path.exists():
            holdout = torch.load(
                self.range_holdout_path,
                map_location="cpu",
                weights_only=False,
            )
            expected = {
                "version": 1,
                "kind": "heads_up_fixed_range_holdout",
                "input_dim": int(self.trainer.input_dim),
                "max_history": int(self.trainer.max_history),
                "seed": int(self.config.validation_seed),
                "hands_per_opponent": hands_per_opponent,
                "opponent_names": tuple(opponent_names),
            }
            actual = {key: holdout.get(key) for key in expected}
            if actual != expected:
                raise ValueError(
                    "fixed opponent-range holdout does not match this campaign"
                )
        else:
            holdout = _build_fixed_range_holdout(
                self.trainer,
                profiles=self.config.opponent_profiles,
                hands_per_opponent=hands_per_opponent,
                seed=self.config.validation_seed,
                behavior_policy_nets=self.baseline.policy_nets,
                reference_policy_nets=(
                    self.reference_policy.policy_nets
                    if self.reference_policy is not None
                    else None
                ),
                inference_batch_size=self.config.range_evaluation_batch_size,
            )
            temporary = self.range_holdout_path.with_suffix(".pt.tmp")
            torch.save(holdout, temporary)
            temporary.replace(self.range_holdout_path)
        self._range_holdout = holdout
        return holdout

    def _run_config_payload(self) -> dict[str, Any]:
        return {
            "version": 1,
            "campaign": _json_safe(asdict(self.config)),
            "environment": {
                "starting_stack": self.trainer.env.starting_stack,
                "small_blind": self.trainer.env.small_blind,
                "big_blind": self.trainer.env.big_blind,
            },
            "trainer": {
                "hidden": self.trainer.hidden,
                "blocks": self.trainer.blocks,
                "network_architecture": self.trainer.network_architecture,
                "policy_network_architecture": POLICY_NETWORK_ARCHITECTURE,
                "range_loss_weight": self.trainer.range_loss_weight,
                "reservoir_turnover_fraction": (
                    self.trainer.reservoir_turnover_fraction
                ),
                "learning_rate": self.trainer.learning_rate,
                "max_history": self.trainer.max_history,
                "max_nodes_per_traversal": self.trainer.max_nodes_per_traversal,
                "max_depth": self.trainer.max_depth,
                "exploration": self.trainer.exploration,
                "reinitialize_advantage_each_iteration": (
                    self.trainer.reinitialize_advantage_each_iteration
                ),
                "advantage_reinitialize_from_iteration": (
                    self.trainer.advantage_reinitialize_from_iteration
                ),
                "advantage_reinitialize_cycle": (
                    self.trainer.advantage_reinitialize_cycle
                ),
                "advantage_capacity": self.trainer.advantage_buffers[0].capacity,
                "policy_capacity": self.trainer.policy_buffers[0].capacity,
                "range_capacity": self.trainer.range_buffers[0].capacity,
            },
            "initialization": {"kind": "random_initialization"},
        }

    def _validate_or_create_run_config(self) -> None:
        current = self._run_config_payload()
        if not self.run_config_path.exists():
            _atomic_json(self.run_config_path, current)
            return
        previous = json.loads(self.run_config_path.read_text(encoding="utf-8"))
        if previous.get("environment") != current["environment"]:
            raise ValueError("artifact directory belongs to another HU game")
        previous_trainer = dict(previous.get("trainer", {}))
        current_trainer = dict(current["trainer"])
        mutable_trainer = {
            "learning_rate",
            "exploration",
            "max_depth",
            "max_nodes_per_traversal",
            "advantage_capacity",
            "policy_capacity",
            "range_capacity",
            "advantage_reinitialize_cycle",
        }
        trainer_changes = {
            key: {"previous": previous_trainer.get(key), "current": current_trainer.get(key)}
            for key in sorted(set(previous_trainer) | set(current_trainer))
            if previous_trainer.get(key) != current_trainer.get(key)
        }
        locked_trainer = sorted(set(trainer_changes) - mutable_trainer)
        if locked_trainer:
            raise ValueError(
                "trainer settings changed for locked field(s): "
                f"{', '.join(locked_trainer)}"
            )
        previous_campaign = dict(previous.get("campaign", {}))
        current_campaign = dict(current["campaign"])
        mutable_campaign = {
            "target_iteration",
            "evaluate_every",
            "evaluation_games_per_player",
            "range_evaluation_hands_per_opponent",
            "range_evaluation_batch_size",
            "traversal_workers",
            "traversals_per_player",
            "advantage_steps",
            "policy_steps",
            "batch_size",
            "range_training_hands_per_iteration",
            "range_batch_size",
            "keep_full_checkpoints",
            "root_stack_distribution",
            "root_stack_depths_bb",
        }
        campaign_changes = {
            key: {"previous": previous_campaign.get(key), "current": current_campaign.get(key)}
            for key in sorted(set(previous_campaign) | set(current_campaign))
            if previous_campaign.get(key) != current_campaign.get(key)
        }
        locked_campaign = sorted(set(campaign_changes) - mutable_campaign)
        if locked_campaign:
            raise ValueError(
                "campaign settings changed for locked field(s): "
                f"{', '.join(locked_campaign)}"
            )
        if self.config.target_iteration < self.trainer.iteration:
            raise ValueError("target_iteration is behind the resumed checkpoint")
        if trainer_changes or campaign_changes:
            with self.run_config_history_path.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        _json_safe(
                            {
                                "changed_at_utc": time.strftime(
                                    "%Y-%m-%dT%H:%M:%SZ",
                                    time.gmtime(),
                                ),
                                "resumed_iteration": self.trainer.iteration,
                                "campaign_changes": campaign_changes,
                                "trainer_changes": trainer_changes,
                            }
                        ),
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            _atomic_json(self.run_config_path, current)

    def _ensure_baseline(self) -> None:
        if self.baseline_path.exists():
            self.baseline = load_policy_snapshot(
                self.baseline_path,
                device=self.trainer.device,
            )
            if self.baseline.metadata["input_dim"] != self.trainer.input_dim:
                raise ValueError("initial policy snapshot uses another encoder")
            return
        if self.trainer.iteration != 0:
            raise FileNotFoundError(
                "initial baseline is missing after training began"
            )
        save_policy_snapshot(
            self.trainer,
            self.baseline_path,
            metadata={"purpose": "fixed_initial_baseline"},
        )
        self.baseline = load_policy_snapshot(
            self.baseline_path,
            device=self.trainer.device,
        )

    def _append_metric(self, row: dict[str, Any]) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(_json_safe(row), separators=(",", ":")) + "\n"
            )
            stream.flush()

    def _save_checkpoint(
        self,
        *,
        emergency: bool = False,
        failed: bool = False,
    ) -> Path:
        suffix = "_emergency" if emergency else ("_failed" if failed else "")
        path = self.checkpoint_dir / (
            f"step_{self.trainer.iteration:08d}{suffix}.pt"
        )
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
        paths = sorted(
            self.checkpoint_dir.glob("step_*.pt"),
            key=lambda candidate: (candidate.stat().st_mtime_ns, candidate.name),
        )
        for old in paths[: -self.config.keep_full_checkpoints]:
            old.unlink()
        return path

    def _milestone_snapshot(self) -> Path:
        return save_policy_snapshot(
            self.trainer,
            self.snapshot_dir / f"policy_{self.trainer.iteration:08d}.pt",
            metadata={"purpose": "historical_league"},
        )

    def _league_paths(self) -> list[Path]:
        current = f"policy_{self.trainer.iteration:08d}.pt"
        paths = [
            path
            for path in sorted(self.snapshot_dir.glob("policy_*.pt"))
            if path.name != current
        ]
        return paths[-self.config.league_opponents :]

    def _evaluate(self, row: dict[str, Any]) -> None:
        means: list[float] = []
        stderrs: list[float] = []
        iteration_dir = self.evaluation_dir / f"step_{self.trainer.iteration:08d}"
        iteration_dir.mkdir(parents=True, exist_ok=True)
        baseline_dir = (
            self.evaluation_dir
            / "initial_baseline"
            / (
                f"seed_{self.config.validation_seed}_games_"
                f"{self.config.evaluation_games_per_player}"
            )
        )
        baseline_dir.mkdir(parents=True, exist_ok=True)
        for profile in self.config.opponent_profiles:
            candidate, hands = _evaluate_against_profile_result(
                self.trainer,
                profile,
                games_per_seat=self.config.evaluation_games_per_player,
                seed=self.config.validation_seed,
            )
            hands.to_csv(iteration_dir / f"{profile}_hands.csv", index=False)
            for key, value in candidate.items():
                if key not in {"profile", "action_counts"}:
                    row[f"benchmark_{profile}_{key}"] = float(value)
            means.append(float(candidate["mean_ev_bb"]))
            stderrs.append(float(candidate["clustered_stderr_bb"]))
            baseline_json = baseline_dir / f"{profile}_summary.json"
            baseline_csv = baseline_dir / f"{profile}_hands.csv"
            if baseline_json.exists():
                baseline = json.loads(baseline_json.read_text(encoding="utf-8"))
            else:
                baseline, baseline_hands = _evaluate_against_profile_result(
                    self.trainer,
                    profile,
                    games_per_seat=self.config.evaluation_games_per_player,
                    seed=self.config.validation_seed,
                    policy_nets=self.baseline.policy_nets,
                )
                baseline_hands.to_csv(baseline_csv, index=False)
                _atomic_json(baseline_json, _json_safe(baseline))
            delta = float(candidate["mean_ev_bb"]) - float(
                baseline["mean_ev_bb"]
            )
            delta_stderr = math.sqrt(
                float(candidate["clustered_stderr_bb"]) ** 2
                + float(baseline["clustered_stderr_bb"]) ** 2
            )
            row[f"benchmark_{profile}_baseline_ev_bb"] = float(
                baseline["mean_ev_bb"]
            )
            row[f"benchmark_{profile}_delta_ev_bb"] = delta
            row[f"benchmark_{profile}_probability_delta_positive"] = (
                NormalDist().cdf(delta / delta_stderr)
                if delta_stderr > 0
                else float(delta > 0)
            )
        if self.reference_policy is not None:
            candidate, hands = _evaluate_against_policy_result(
                self.trainer,
                self.reference_policy.policy_nets,
                games_per_seat=self.config.evaluation_games_per_player,
                seed=self.config.validation_seed,
            )
            hands.to_csv(
                iteration_dir / "reference_policy_hands.csv",
                index=False,
            )
            for key, value in candidate.items():
                if key != "profile":
                    row[f"benchmark_reference_policy_{key}"] = float(value)
            row["benchmark_reference_policy_iteration"] = float(
                self.reference_policy.iteration
            )
            means.append(float(candidate["mean_ev_bb"]))
            stderrs.append(float(candidate["clustered_stderr_bb"]))
            baseline_json = baseline_dir / "reference_policy_summary.json"
            baseline_csv = baseline_dir / "reference_policy_hands.csv"
            if baseline_json.exists():
                baseline = json.loads(baseline_json.read_text(encoding="utf-8"))
            else:
                baseline, baseline_hands = _evaluate_against_policy_result(
                    self.trainer,
                    self.reference_policy.policy_nets,
                    games_per_seat=self.config.evaluation_games_per_player,
                    seed=self.config.validation_seed,
                    candidate_policy_nets=self.baseline.policy_nets,
                )
                baseline_hands.to_csv(baseline_csv, index=False)
                _atomic_json(baseline_json, _json_safe(baseline))
            delta = float(candidate["mean_ev_bb"]) - float(
                baseline["mean_ev_bb"]
            )
            delta_stderr = math.sqrt(
                float(candidate["clustered_stderr_bb"]) ** 2
                + float(baseline["clustered_stderr_bb"]) ** 2
            )
            row["benchmark_reference_policy_baseline_ev_bb"] = float(
                baseline["mean_ev_bb"]
            )
            row["benchmark_reference_policy_delta_ev_bb"] = delta
            row["benchmark_reference_policy_probability_delta_positive"] = (
                NormalDist().cdf(delta / delta_stderr)
                if delta_stderr > 0
                else float(delta > 0)
            )
        composite = float(np.mean(means))
        composite_stderr = math.sqrt(sum(x * x for x in stderrs)) / len(
            stderrs
        )
        row["benchmark_composite_ev_bb"] = composite
        row["benchmark_composite_stderr_bb"] = composite_stderr
        row["benchmark_composite_lcb95_bb"] = (
            composite - 1.96 * composite_stderr
        )
        league_values = []
        for path in self._league_paths():
            snapshot = load_policy_snapshot(path, device=self.trainer.device)
            league_values.append(
                _evaluate_against_policy_nets(
                    self.trainer,
                    snapshot.policy_nets,
                    games_per_player=self.config.league_games_per_player,
                    seed=self.config.validation_seed + snapshot.iteration,
                )
            )
        row["league_opponents"] = float(len(league_values))
        row["league_mean_ev_bb"] = (
            float(np.mean(league_values)) if league_values else float("nan")
        )
        row["league_worst_ev_bb"] = (
            min(league_values) if league_values else float("nan")
        )
        row.update(
            evaluate_fixed_range_holdout(
                self.trainer,
                self._load_or_create_range_holdout(),
                batch_size=self.config.range_evaluation_batch_size,
            )
        )
        reservoir_stats = range_reservoir_statistics(self.trainer)
        for street, percent in reservoir_stats["street_percent"].items():
            row[f"range_reservoir_{street}_percent"] = float(percent)
        row["range_reservoir_sampled_rows"] = float(
            reservoir_stats["sampled_rows"]
        )
        row["range_reservoir_total_rows"] = float(
            reservoir_stats["total_rows"]
        )
        stats_path = self.evaluation_dir / (
            f"range_reservoir_{self.trainer.iteration:08d}.json"
        )
        _atomic_json(stats_path, reservoir_stats)
        _atomic_json(
            self.artifact_dir / "range_reservoir_latest.json",
            reservoir_stats,
        )
        save_range_reservoir_dashboard(
            reservoir_stats,
            self.artifact_dir / "range_reservoir_dashboard.png",
        )
        champion_manifest = self.artifact_dir / "champion.json"
        previous_lcb = -float("inf")
        if champion_manifest.exists():
            previous_lcb = float(
                json.loads(
                    champion_manifest.read_text(encoding="utf-8")
                )["benchmark_composite_lcb95_bb"]
            )
        promoted = row["benchmark_composite_lcb95_bb"] > previous_lcb
        row["promoted_to_champion"] = float(promoted)
        if promoted:
            save_policy_snapshot(
                self.trainer,
                self.champion_path,
                metadata={
                    "purpose": "validation_champion",
                    "benchmark_composite_lcb95_bb": row[
                        "benchmark_composite_lcb95_bb"
                    ],
                },
            )
            _atomic_json(
                champion_manifest,
                {
                    "iteration": self.trainer.iteration,
                    "benchmark_composite_lcb95_bb": row[
                        "benchmark_composite_lcb95_bb"
                    ],
                    "policy": str(self.champion_path.resolve()),
                },
            )
        self.trainer.metrics[-1] = dict(row)

    def run(
        self,
        *,
        on_iteration: Callable[[dict[str, Any]], None] | None = None,
        on_evaluation: Callable[[pd.DataFrame], None] | None = None,
    ) -> list[dict[str, float]]:
        interrupted = False
        failed = False
        try:
            if self.trainer.last_fitted_iteration < self.trainer.iteration:
                recovery = self.trainer.recover_incomplete_fit(
                    advantage_steps=self.config.advantage_steps,
                    policy_steps=self.config.policy_steps,
                    batch_size=self.config.batch_size,
                    range_batch_size=self.config.range_batch_size,
                )
                _atomic_json(
                    self.artifact_dir / "last_recovery.json",
                    _json_safe(recovery),
                )
                self._save_checkpoint()
            while self.trainer.iteration < self.config.target_iteration:
                target_iteration = self.trainer.iteration + 1
                range_collection: dict[str, float] = {}
                if (
                    self.trainer.range_last_collected_iteration
                    < target_iteration
                ):
                    range_started = time.perf_counter()
                    range_collection = (
                        _collect_independent_range_training_hands(
                            self.trainer,
                            profiles=self.config.opponent_profiles,
                            hands=(
                                self.config.range_training_hands_per_iteration
                            ),
                            seed=(
                                self.trainer.seed
                                + 900_000
                                + target_iteration * 100_003
                            ),
                            reference_policy_nets=(
                                self.reference_policy.policy_nets
                                if self.reference_policy is not None
                                else None
                            ),
                            inference_batch_size=(
                                self.config.range_evaluation_batch_size
                            ),
                            stack_depths_bb=(
                                self.config.root_stack_depths_bb
                            ),
                        )
                    )
                    range_collection["range_collection_seconds"] = float(
                        time.perf_counter() - range_started
                    )
                    self.trainer.range_last_collected_iteration = (
                        target_iteration
                    )
                row = self.trainer.train_iteration(
                    traversals_per_player=self.config.traversals_per_player,
                    advantage_steps=self.config.advantage_steps,
                    policy_steps=self.config.policy_steps,
                    batch_size=self.config.batch_size,
                    range_batch_size=self.config.range_batch_size,
                    traversal_workers=self.config.traversal_workers,
                    root_stack_distribution=(
                        self.config.root_stack_distribution
                    ),
                    root_stack_depths_bb=self.config.root_stack_depths_bb,
                )
                row.update(range_collection)
                self.trainer.metrics[-1] = dict(row)
                if self.trainer.iteration % self.config.snapshot_every == 0:
                    self._milestone_snapshot()
                should_evaluate = (
                    self.trainer.iteration % self.config.evaluate_every == 0
                )
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
    env,
    artifact_dir: str | Path,
    *,
    device: str | torch.device,
    trainer_kwargs: dict[str, Any],
) -> tuple[HeadsUpNeuralCFR, bool]:
    latest = resolve_latest_checkpoint(artifact_dir)
    if latest is not None:
        trainer = HeadsUpNeuralCFR.load(latest, env, device=device)
        if not trainer.can_resume_training:
            raise RuntimeError("latest checkpoint is not resumable")
        trainer.advantage_reinitialize_cycle = int(
            trainer_kwargs.get(
                "advantage_reinitialize_cycle",
                trainer.advantage_reinitialize_cycle,
            )
        )
        if trainer.advantage_reinitialize_cycle <= 0:
            raise ValueError("advantage_reinitialize_cycle must be positive")
        capacity_requests = (
            (
                "advantage_capacity",
                trainer.advantage_buffers,
            ),
            (
                "policy_capacity",
                trainer.policy_buffers,
            ),
            (
                "range_capacity",
                trainer.range_buffers,
            ),
        )
        for name, buffers in capacity_requests:
            requested = int(trainer_kwargs.get(name, buffers[0].capacity))
            current = int(buffers[0].capacity)
            if requested < current:
                raise ValueError(
                    f"{name} cannot shrink a resumed reservoir "
                    f"from {current} to {requested}"
                )
            for buffer in buffers:
                buffer.capacity = requested
        return trainer, True
    return HeadsUpNeuralCFR(env, device=device, **trainer_kwargs), False


__all__ = [
    "CampaignConfig",
    "CallingStationOpponent",
    "OPPONENT_PROFILES",
    "PolicySnapshot",
    "ProductionCampaign",
    "ScriptedOpponent",
    "TightAggressiveOpponent",
    "UniformRandomOpponent",
    "evaluate_against_profile",
    "evaluate_benchmark_suite",
    "evaluate_fixed_range_holdout",
    "load_or_create_trainer",
    "load_policy_snapshot",
    "range_reservoir_statistics",
    "resolve_latest_checkpoint",
    "save_policy_snapshot",
]
