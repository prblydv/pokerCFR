"""Safe bridge between exact heads-up states and finite policy/search actions.

Observed room actions stay exact: callers apply them with
``apply_observed_action``/``env.step_exact`` and never translate them to a
nearby policy bucket.  At a bot decision, :func:`build_decision_context`
encodes that exact state and exposes the ten legal blueprint slots together
with their state-specific chip targets.

``HeadsUpRealTimeResolver`` keeps its bounded-search mechanics local so the
heads-up deployment has no dependency on the three-player action schema.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

import torch
import torch.nn as nn

from heads_up_engine import NUM_ACTIONS
from heads_up_models import (
    DEFAULT_MAX_HISTORY,
    build_action_descriptors,
    encode_information_state,
    encoder_metadata,
    masked_softmax,
)
@dataclass(frozen=True)
class HeadsUpDecisionContext:
    """Everything a ten-output policy needs for one live decision."""

    hero: int
    legal_actions: tuple[int, ...]
    legal_mask: torch.Tensor
    action_descriptors: tuple[object | None, ...]
    observation: torch.Tensor


class _Policy(Protocol):
    def probabilities(self, env, state) -> torch.Tensor: ...


class _Opponent(Protocol):
    def probabilities(self, env, state, player: int) -> torch.Tensor: ...


class SearchCancelled(RuntimeError):
    """Raised when a GUI search is cancelled because its state is obsolete."""


@dataclass(frozen=True)
class SearchResult:
    action: int
    probabilities: torch.Tensor
    blueprint_probabilities: torch.Tensor
    action_values: dict[int, float]
    iterations: int
    rollouts: int
    elapsed_ms: float


class _HeadsUpResolverBase:
    """Bounded-search helpers shared by the heads-up resolver."""

    def __init__(
        self,
        policy: _Policy,
        tag_opponent: _Opponent | None,
        *,
        tag_seat: int | None,
        scripted_opponents: dict[int, _Opponent] | None = None,
        time_budget_ms: int = 900,
        max_rollouts: int = 48,
        blueprint_weight: float = 0.35,
        max_actions_per_rollout: int = 64,
        seed: int | None = None,
    ) -> None:
        if time_budget_ms <= 0:
            raise ValueError("time_budget_ms must be positive")
        if max_rollouts <= 0:
            raise ValueError("max_rollouts must be positive")
        if not 0.0 <= blueprint_weight <= 1.0:
            raise ValueError("blueprint_weight must be in [0, 1]")
        self.policy = policy
        self.tag_opponent = tag_opponent
        self.tag_seat = None if tag_seat is None else int(tag_seat)
        self.scripted_opponents = dict(scripted_opponents or {})
        if self.tag_seat is not None:
            if tag_opponent is None:
                raise ValueError("tag_opponent is required when tag_seat is set")
            self.scripted_opponents.setdefault(self.tag_seat, tag_opponent)
        self.time_budget_ms = int(time_budget_ms)
        self.max_rollouts = int(max_rollouts)
        self.blueprint_weight = float(blueprint_weight)
        self.max_actions_per_rollout = int(max_actions_per_rollout)
        self.rng = random.Random(seed)

    def _determinize(self, env, state, hero: int):
        sampled = env.clone(state)
        hero_cards = [int(card) for card in state.hole[hero]]
        board = [int(card) for card in state.board]
        known = set(hero_cards + board)
        if len(known) != len(hero_cards) + len(board):
            raise ValueError("state contains duplicate known cards")

        pool = [card for card in range(52) if card not in known]
        self.rng.shuffle(pool)
        cursor = 0
        holes: list[list[int]] = []
        for seat, existing in enumerate(state.hole):
            count = len(existing)
            if seat == hero:
                holes.append(hero_cards)
            else:
                holes.append(pool[cursor : cursor + count])
                cursor += count
        burned_count = len(state.burned)
        burned = pool[cursor : cursor + burned_count]
        cursor += burned_count
        deck_count = len(state.deck)
        deck = pool[cursor : cursor + deck_count]
        cursor += deck_count
        if cursor != len(pool):
            raise ValueError("state card zones do not form a complete deck")

        sampled.hole = holes
        sampled.burned = burned
        sampled.deck = deck
        return sampled

    def _sample(
        self,
        probabilities: torch.Tensor,
        rng: random.Random | None = None,
    ) -> int:
        rng = self.rng if rng is None else rng
        values = probabilities.detach().cpu().tolist()
        threshold = rng.random()
        cumulative = 0.0
        fallback = max(range(len(values)), key=values.__getitem__)
        for action, probability in enumerate(values):
            if probability <= 0.0:
                continue
            fallback = action
            cumulative += float(probability)
            if threshold <= cumulative + 1e-12:
                return action
        return fallback

    @staticmethod
    def _regret_strategy(
        legal: list[int],
        regrets: dict[int, float],
        blueprint: torch.Tensor,
    ) -> dict[int, float]:
        positive = {action: max(0.0, regrets[action]) for action in legal}
        total = sum(positive.values())
        if total > 1e-12:
            return {action: positive[action] / total for action in legal}
        blueprint_total = sum(float(blueprint[action]) for action in legal)
        if blueprint_total > 1e-12:
            return {
                action: float(blueprint[action]) / blueprint_total
                for action in legal
            }
        return {action: 1.0 / len(legal) for action in legal}

    @classmethod
    def _average_strategy(
        cls,
        legal: list[int],
        strategy_sum: dict[int, float],
        regrets: dict[int, float],
        blueprint: torch.Tensor,
    ) -> dict[int, float]:
        total = sum(strategy_sum.values())
        if total > 1e-12:
            return {action: strategy_sum[action] / total for action in legal}
        return cls._regret_strategy(legal, regrets, blueprint)


def _state_big_blind(env, state) -> float:
    value = getattr(state, "big_blind", None)
    if value is None:
        value = getattr(env, "bb", getattr(env, "big_blind", None))
    if value is None or isinstance(value, bool) or float(value) <= 0.0:
        raise ValueError("the environment/state must expose a positive big blind")
    return float(value)


def build_decision_context(
    env,
    state,
    *,
    hero: int | None = None,
    max_history: int = DEFAULT_MAX_HISTORY,
) -> HeadsUpDecisionContext:
    """Encode an exact live state without snapping prior bets to action slots."""

    if bool(getattr(state, "terminal", False)) or getattr(state, "to_act", None) is None:
        raise ValueError("a heads-up policy decision requires a live actor")
    actor = int(state.to_act)
    hero = actor if hero is None else int(hero)
    if hero != actor:
        raise ValueError("policy decision hero must be the player currently acting")
    if hero not in (0, 1):
        raise ValueError("hero must be seat 0 or 1")

    legal = tuple(int(action) for action in env.legal_actions(state))
    if not legal:
        raise RuntimeError("live heads-up state has no legal actions")
    if any(action < 0 or action >= NUM_ACTIONS for action in legal):
        raise RuntimeError("engine returned an action outside the HU action schema")

    descriptors = tuple(build_action_descriptors(env, state))
    if len(descriptors) != NUM_ACTIONS:
        raise RuntimeError(
            f"expected {NUM_ACTIONS} action descriptors, got {len(descriptors)}"
        )
    observation = encode_information_state(
        state,
        hero,
        legal,
        _state_big_blind(env, state),
        max_history,
        action_descriptors=descriptors,
    )
    legal_mask = torch.zeros(NUM_ACTIONS, dtype=torch.float32)
    legal_mask[list(legal)] = 1.0
    return HeadsUpDecisionContext(
        hero=hero,
        legal_actions=legal,
        legal_mask=legal_mask,
        action_descriptors=descriptors,
        observation=observation,
    )


def apply_observed_action(env, state, kind: str, raise_to: int | None = None):
    """Apply one real-room action exactly; illegal sizes fail instead of clamp."""

    return env.step_exact(state, kind, raise_to)


def validate_checkpoint_encoder(
    metadata: Mapping[str, object],
    *,
    max_history: int = DEFAULT_MAX_HISTORY,
) -> None:
    """Fail fast when a checkpoint was trained against another action schema."""

    expected = encoder_metadata(max_history)
    keys = (
        "engine_schema_version",
        "action_schema_version",
        "encoder_schema_version",
        "width",
        "max_history",
        "num_actions",
        "action_names",
    )
    missing = [key for key in keys if key not in metadata]
    if missing:
        raise ValueError(f"checkpoint encoder metadata is missing: {missing}")
    for key in keys:
        actual = metadata[key]
        wanted = expected[key]
        if key == "action_names":
            actual = tuple(actual)  # JSON checkpoints normally store a list.
            wanted = tuple(wanted)
        if actual != wanted:
            raise ValueError(
                f"checkpoint encoder {key} mismatch: expected {wanted!r}, "
                f"got {actual!r}"
            )


class HeadsUpNetworkPolicy:
    """Turn one shared, or two seat-specific, ten-logit networks into a policy."""

    def __init__(
        self,
        networks: nn.Module | Sequence[nn.Module],
        *,
        max_history: int = DEFAULT_MAX_HISTORY,
        checkpoint_encoder: Mapping[str, object] | None = None,
    ) -> None:
        if isinstance(networks, nn.Module):
            values = (networks,)
        else:
            values = tuple(networks)
        if len(values) not in (1, 2) or any(
            not isinstance(network, nn.Module) for network in values
        ):
            raise ValueError("networks must be one shared module or two seat modules")
        if isinstance(max_history, bool) or int(max_history) <= 0:
            raise ValueError("max_history must be a positive integer")
        if checkpoint_encoder is not None:
            validate_checkpoint_encoder(
                checkpoint_encoder, max_history=int(max_history)
            )
        self.networks = values
        self.max_history = int(max_history)

    @torch.inference_mode()
    def probabilities(self, env, state) -> torch.Tensor:
        context = build_decision_context(
            env, state, max_history=self.max_history
        )
        network = (
            self.networks[0]
            if len(self.networks) == 1
            else self.networks[context.hero]
        )
        try:
            device = next(network.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        logits = network(context.observation.to(device).unsqueeze(0))
        if logits.shape != (1, NUM_ACTIONS):
            raise ValueError(
                "heads-up policy network must output shape "
                f"(batch, {NUM_ACTIONS}); got {tuple(logits.shape)}"
            )
        probabilities = masked_softmax(
            logits[0], context.legal_mask.to(logits.device)
        )
        return probabilities.detach().cpu()


class HeadsUpRealTimeResolver(_HeadsUpResolverBase):
    """The bounded resolver with a strict ten-action HU policy contract."""

    @staticmethod
    def _checked_probabilities(env, state, values) -> torch.Tensor:
        probabilities = torch.as_tensor(values, dtype=torch.float32).detach().cpu()
        if probabilities.shape != (NUM_ACTIONS,):
            raise ValueError(
                f"heads-up policies must return {NUM_ACTIONS} probabilities; "
                f"got shape {tuple(probabilities.shape)}"
            )
        if not bool(torch.isfinite(probabilities).all()):
            raise ValueError("heads-up policy returned a non-finite probability")
        if bool((probabilities < -1e-8).any()):
            raise ValueError("heads-up policy returned a negative probability")
        probabilities.clamp_(min=0.0)

        legal = [int(action) for action in env.legal_actions(state)]
        mask = torch.zeros(NUM_ACTIONS, dtype=torch.float32)
        mask[legal] = 1.0
        probabilities *= mask
        total = float(probabilities.sum())
        if total <= 1e-12:
            raise ValueError("heads-up policy assigns no mass to any legal action")
        return probabilities / total

    def resolve(self, env, state, cancel_event=None) -> SearchResult:
        if state.terminal or state.to_act is None:
            raise ValueError("search requires a live decision state")
        if cancel_event is not None and cancel_event.is_set():
            raise SearchCancelled()

        started = time.perf_counter()
        deadline = started + self.time_budget_ms / 1000.0
        hero = int(state.to_act)
        legal = [int(action) for action in env.legal_actions(state)]
        if not legal:
            raise RuntimeError("live heads-up state has no legal actions")
        blueprint = self._checked_probabilities(
            env, state, self.policy.probabilities(env, state)
        )
        if len(legal) == 1:
            elapsed = 1000.0 * (time.perf_counter() - started)
            return SearchResult(
                legal[0], blueprint, blueprint, {legal[0]: 0.0}, 0, 0, elapsed
            )

        regrets = {action: 0.0 for action in legal}
        strategy_sum = {action: 0.0 for action in legal}
        value_sum = {action: 0.0 for action in legal}
        value_count = {action: 0 for action in legal}
        rollouts = 0
        iterations = 0

        while rollouts + len(legal) <= self.max_rollouts:
            if cancel_event is not None and cancel_event.is_set():
                raise SearchCancelled()
            if iterations and time.perf_counter() >= deadline:
                break

            strategy = self._regret_strategy(legal, regrets, blueprint)
            determinized = self._determinize(env, state, hero)
            rollout_seed = self.rng.getrandbits(64)
            sampled_values: dict[int, float] = {}
            complete = True
            for action in legal:
                if cancel_event is not None and cancel_event.is_set():
                    raise SearchCancelled()
                if iterations and time.perf_counter() >= deadline:
                    complete = False
                    break
                child = env.step(env.clone(determinized), action)
                value = self._rollout(
                    env,
                    child,
                    hero,
                    random.Random(rollout_seed),
                    cancel_event,
                )
                sampled_values[action] = value
                value_sum[action] += value
                value_count[action] += 1
                rollouts += 1

            if not complete:
                break

            node_value = sum(
                strategy[action] * sampled_values[action] for action in legal
            )
            for action in legal:
                regrets[action] += sampled_values[action] - node_value
                strategy_sum[action] += strategy[action]
            iterations += 1

        search = self._average_strategy(legal, strategy_sum, regrets, blueprint)
        means = {
            action: (
                value_sum[action] / value_count[action]
                if value_count[action]
                else 0.0
            )
            for action in legal
        }
        base = {
            action: (
                self.blueprint_weight * float(blueprint[action])
                + (1.0 - self.blueprint_weight) * search[action]
            )
            for action in legal
        }
        base_value = sum(base[action] * means[action] for action in legal)
        risk_scale = max(
            1.0,
            2.0 * _state_big_blind(env, state),
            0.5 * float(state.pot),
        )
        refined = {
            action: base[action]
            * math.exp(
                max(-3.0, min(3.0, (means[action] - base_value) / risk_scale))
            )
            for action in legal
        }
        refined_total = sum(refined.values())
        mixed = torch.zeros(NUM_ACTIONS, dtype=torch.float32)
        for action in legal:
            mixed[action] = refined[action] / refined_total
        mixed /= mixed.sum()
        action = self._sample(mixed)
        elapsed = 1000.0 * (time.perf_counter() - started)
        return SearchResult(
            action, mixed, blueprint, means, iterations, rollouts, elapsed
        )

    def _rollout(
        self,
        env,
        state,
        hero: int,
        rollout_rng,
        cancel_event=None,
    ) -> float:
        decisions = 0
        while not state.terminal:
            if cancel_event is not None and cancel_event.is_set():
                raise SearchCancelled()
            if state.to_act is None:
                raise RuntimeError("non-terminal rollout state has no actor")
            actor = int(state.to_act)
            if actor in self.scripted_opponents:
                raw = self.scripted_opponents[actor].probabilities(env, state, actor)
            else:
                raw = self.policy.probabilities(env, state)
            probabilities = self._checked_probabilities(env, state, raw)
            state = env.step(state, self._sample(probabilities, rollout_rng))
            decisions += 1
            if decisions > self.max_actions_per_rollout:
                raise RuntimeError("search rollout exceeded the action limit")
        return float(state.payoffs[hero])


__all__ = [
    "HeadsUpDecisionContext",
    "HeadsUpNetworkPolicy",
    "HeadsUpRealTimeResolver",
    "SearchCancelled",
    "SearchResult",
    "apply_observed_action",
    "build_decision_context",
    "validate_checkpoint_encoder",
]
