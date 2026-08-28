"""Bounded real-time re-solving on top of a three-player blueprint policy.

The solver uses public-state-consistent determinizations, paired Monte Carlo
counterfactual values, regret matching at the current information set, and the
blueprint for continuation values.  It is intentionally a small, latency-bound
approximation of Pluribus-style depth-limited search, not a reproduction of the
full Pluribus system.
"""

from __future__ import annotations

import random
import time
import math
from dataclasses import dataclass
from typing import Protocol

import torch

from three_player_engine import NUM_ACTIONS


class Policy(Protocol):
    def probabilities(self, env, state) -> torch.Tensor: ...


class Opponent(Protocol):
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


class RealTimeResolver:
    """Resolve one decision while obeying a wall-clock and rollout budget."""

    def __init__(
        self,
        policy: Policy,
        tag_opponent: Opponent | None,
        *,
        tag_seat: int | None,
        scripted_opponents: dict[int, Opponent] | None = None,
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

    def resolve(self, env, state, cancel_event=None) -> SearchResult:
        if state.terminal or state.to_act is None:
            raise ValueError("search requires a live decision state")
        if cancel_event is not None and cancel_event.is_set():
            raise SearchCancelled()

        started = time.perf_counter()
        deadline = started + self.time_budget_ms / 1000.0
        hero = int(state.to_act)
        legal = [int(action) for action in env.legal_actions(state)]
        blueprint = self.policy.probabilities(env, state).detach().cpu()
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
            # Do not begin another paired iteration when the previous one used
            # the available thinking time.
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
                # Always allow the first paired iteration. Thereafter stop
                # before starting a rollout that is already over budget.
                if iterations and time.perf_counter() >= deadline:
                    complete = False
                    break
                child = env.step(env.clone(determinized), action)
                # Reusing the random stream across root actions is a
                # common-random-numbers variance reduction technique.
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
            action: value_sum[action] / value_count[action]
            if value_count[action]
            else 0.0
            for action in legal
        }
        # Anchor the CFR strategy to the blueprint, then apply a bounded
        # KL-regularized policy-improvement step from the paired rollout EVs.
        # This is much less brittle than selecting the largest noisy mean.
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
            2.0 * float(env.bb),
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
        rollout_rng: random.Random,
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
                probabilities = self.scripted_opponents[actor].probabilities(
                    env, state, actor
                )
            else:
                probabilities = self.policy.probabilities(env, state)
            state = env.step(state, self._sample(probabilities, rollout_rng))
            decisions += 1
            if decisions > self.max_actions_per_rollout:
                raise RuntimeError("search rollout exceeded the action limit")
        return float(state.payoffs[hero])

    def _determinize(self, env, state, hero: int):
        """Randomize every private card unknown to ``hero`` without leakage."""

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
                action: float(blueprint[action]) / blueprint_total for action in legal
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
