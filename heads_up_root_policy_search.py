"""Three-player-style root policy improvement adapted to heads-up Hold'em.

This intentionally preserves the original resolver's algorithm: paired
terminal rollouts, root regret matching, a 35% blueprint anchor, a bounded
KL-like refinement, and mixed-strategy sampling. Unknown cards are uniformly
redeterminized for every paired root comparison.
"""

from __future__ import annotations

import copy
import itertools
import math
import random
import time
from dataclasses import dataclass
from typing import Sequence

import torch

from heads_up_engine import ACTION_NAMES, NUM_ACTIONS
from heads_up_pluribus_search import (
    CandidateEstimate,
    PluribusSearchResult,
    PublicRangeSnapshot,
    SearchAction,
)


def robust_inferred_range(
    snapshot: PublicRangeSnapshot,
    *,
    temperature: float = 0.65,
    uniform_contamination: float = 0.25,
) -> PublicRangeSnapshot:
    """Temper a Bayesian range and retain a uniform model-error component."""
    if not 0.0 < temperature <= 1.0:
        raise ValueError("temperature must be in (0, 1]")
    if not 0.0 <= uniform_contamination < 1.0:
        raise ValueError("uniform_contamination must be in [0, 1)")
    if not snapshot.combos or len(snapshot.combos) != len(snapshot.weights):
        raise ValueError("range snapshot must be non-empty and aligned")
    powered = [max(0.0, float(weight)) ** temperature for weight in snapshot.weights]
    total = sum(powered)
    if total <= 0.0 or not math.isfinite(total):
        raise ValueError("range snapshot has zero or invalid mass")
    uniform = 1.0 / len(powered)
    weights = tuple(
        (1.0 - uniform_contamination) * value / total
        + uniform_contamination * uniform
        for value in powered
    )
    square_sum = sum(value * value for value in weights)
    return PublicRangeSnapshot(
        combos=snapshot.combos,
        weights=weights,
        effective_sample_size=1.0 / square_sum,
        updates=snapshot.updates,
    )


@dataclass
class _RolloutLane:
    iteration: int
    root_action: int
    state: object
    rng: random.Random
    decisions: int = 0


class HeadsUpRootPolicySearch:
    """HU adapter for the original three-player root resolver."""

    workers = 1

    def __init__(
        self,
        policy,
        *,
        time_budget_ms: int = 10_000,
        max_rollouts: int = 150_000,
        blueprint_weight: float = 0.35,
        max_actions_per_rollout: int = 64,
        batch_iterations: int = 3072,
        use_native_rollouts: bool = True,
        use_batched_action_sampling: bool = True,
        use_batch_step: bool = True,
        range_mode: str = "inferred",
        range_temperature: float = 0.65,
        uniform_contamination: float = 0.25,
        min_strategy_probability: float = 0.0,
        seed: int | None = None,
    ) -> None:
        if time_budget_ms <= 0:
            raise ValueError("time_budget_ms must be positive")
        if max_rollouts <= 0:
            raise ValueError("max_rollouts must be positive")
        if not 0.0 <= blueprint_weight <= 1.0:
            raise ValueError("blueprint_weight must be in [0, 1]")
        if batch_iterations <= 0:
            raise ValueError("batch_iterations must be positive")
        if range_mode not in {"uniform", "inferred"}:
            raise ValueError("range_mode must be 'uniform' or 'inferred'")
        if not 0.0 <= float(min_strategy_probability) < 1.0:
            raise ValueError(
                "min_strategy_probability must be in [0, 1)"
            )
        self.policy = policy
        self.time_budget_ms = int(time_budget_ms)
        self.max_rollouts = int(max_rollouts)
        self.blueprint_weight = float(blueprint_weight)
        self.max_actions_per_rollout = int(max_actions_per_rollout)
        self.batch_iterations = int(batch_iterations)
        self.use_native_rollouts = bool(use_native_rollouts)
        self.use_batched_action_sampling = bool(use_batched_action_sampling)
        self.use_batch_step = bool(use_batch_step)
        self.range_mode = str(range_mode)
        self.range_temperature = float(range_temperature)
        self.uniform_contamination = float(uniform_contamination)
        self.min_strategy_probability = float(min_strategy_probability)
        self.rng = random.Random(seed)

    def close(self, *, wait_for_workers: bool = False) -> None:
        del wait_for_workers

    @staticmethod
    def _sample(
        probabilities: Sequence[float] | torch.Tensor,
        rng: random.Random,
    ) -> int:
        values = (
            probabilities.detach().cpu().tolist()
            if isinstance(probabilities, torch.Tensor)
            else list(probabilities)
        )
        threshold = rng.random()
        cumulative = 0.0
        fallback = max(range(len(values)), key=values.__getitem__)
        for index, probability in enumerate(values):
            if probability <= 0.0:
                continue
            fallback = index
            cumulative += float(probability)
            if threshold <= cumulative + 1e-12:
                return index
        return fallback

    @staticmethod
    def _regret_strategy(
        legal: Sequence[int],
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
        legal: Sequence[int],
        strategy_sum: dict[int, float],
        regrets: dict[int, float],
        blueprint: torch.Tensor,
    ) -> dict[int, float]:
        total = sum(strategy_sum.values())
        if total > 1e-12:
            return {action: strategy_sum[action] / total for action in legal}
        return cls._regret_strategy(legal, regrets, blueprint)

    def _determinize(
        self,
        env,
        state,
        hero: int,
        inferred_range: PublicRangeSnapshot | None = None,
        inferred_cumulative_weights: Sequence[float] | None = None,
    ):
        if bool(getattr(env, "native_backend", False)):
            sampled = env.clone(state)
        else:
            # Only card zones are changed during determinization. A full
            # deepcopy of betting history and all scalar/list state fields is
            # unnecessary because the result is converted to an immutable
            # native rollout state before any betting transition.
            sampled = copy.copy(state)
            sampled.hole = [list(cards) for cards in state.hole]
        hero_cards = [int(card) for card in state.hole[hero]]
        board = [int(card) for card in state.board]
        known = set(hero_cards + board)
        if len(known) != len(hero_cards) + len(board):
            raise ValueError("state contains duplicate known cards")
        opponent = 1 - hero
        sampled.hole[hero] = hero_cards
        opponent_count = len(state.hole[opponent])
        if inferred_range is not None:
            if inferred_cumulative_weights is None:
                inferred_cumulative_weights = tuple(
                    itertools.accumulate(inferred_range.weights)
                )
            combo_index = self.rng.choices(
                range(len(inferred_range.combos)),
                cum_weights=inferred_cumulative_weights,
                k=1,
            )[0]
            opponent_cards = list(inferred_range.combos[combo_index])
        else:
            available = [card for card in range(52) if card not in known]
            self.rng.shuffle(available)
            opponent_cards = available[:opponent_count]
        sampled.hole[opponent] = opponent_cards
        pool = [
            card
            for card in range(52)
            if card not in known and card not in set(opponent_cards)
        ]
        self.rng.shuffle(pool)
        cursor = 0
        burned_count = len(state.burned)
        sampled.burned = pool[cursor : cursor + burned_count]
        cursor += burned_count
        deck_count = len(state.deck)
        sampled.deck = pool[cursor : cursor + deck_count]
        cursor += deck_count
        if cursor != len(pool):
            raise ValueError("state card zones do not form a complete deck")
        return sampled

    def _rollout(
        self,
        env,
        state,
        hero: int,
        rollout_rng: random.Random,
    ) -> float:
        decisions = 0
        while not state.terminal:
            if state.to_act is None:
                raise RuntimeError("non-terminal rollout state has no actor")
            probabilities = self.policy.probabilities(env, state)
            legal = [int(action) for action in env.legal_actions(state)]
            selected = self._sample(
                [float(probabilities[action]) for action in legal],
                rollout_rng,
            )
            state = env.step(state, legal[selected])
            decisions += 1
            if decisions > self.max_actions_per_rollout:
                raise RuntimeError("root-policy rollout exceeded action limit")
        return float(state.payoffs[hero])

    def _policy_probabilities_batch(self, env, states: Sequence) -> list[torch.Tensor]:
        """Evaluate live states in actor-homogeneous batches.

        Policies without a batched adapter retain the exact scalar fallback,
        which keeps tests and third-party policy implementations compatible.
        """
        if not states:
            return []
        probabilities_batch = getattr(self.policy, "probabilities_batch", None)
        if probabilities_batch is None:
            return [self.policy.probabilities(env, state) for state in states]

        result: list[torch.Tensor | None] = [None] * len(states)
        actor_groups: dict[int, list[int]] = {}
        for index, state in enumerate(states):
            if state.terminal or state.to_act is None:
                raise ValueError("batched policy evaluation requires live states")
            actor_groups.setdefault(int(state.to_act), []).append(index)
        for indices in actor_groups.values():
            rows = probabilities_batch(env, [states[index] for index in indices])
            if len(rows) != len(indices):
                raise RuntimeError("batched policy returned the wrong row count")
            for index, row in zip(indices, rows):
                result[index] = row
        if any(row is None for row in result):
            raise RuntimeError("batched policy did not return every requested state")
        return [row for row in result if row is not None]

    def _rollouts_batch(
        self,
        env,
        lanes: Sequence[_RolloutLane],
        hero: int,
    ) -> dict[tuple[int, int], float]:
        """Advance independent rollouts in lockstep with batched policy calls."""
        active = list(lanes)
        values: dict[tuple[int, int], float] = {}
        while active:
            live: list[_RolloutLane] = []
            for lane in active:
                if lane.state.terminal:
                    values[(lane.iteration, lane.root_action)] = float(
                        lane.state.payoffs[hero]
                    )
                else:
                    if lane.state.to_act is None:
                        raise RuntimeError("non-terminal rollout state has no actor")
                    live.append(lane)
            if not live:
                break

            sampler = (
                getattr(self.policy, "sample_actions_batch", None)
                if self.use_batched_action_sampling
                else None
            )
            selected_actions: list[int | None] = [None] * len(live)
            if sampler is not None:
                thresholds = [lane.rng.random() for lane in live]
                actor_groups: dict[int, list[int]] = {}
                for index, lane in enumerate(live):
                    actor_groups.setdefault(
                        int(lane.state.to_act), []
                    ).append(index)
                for indices in actor_groups.values():
                    actions = sampler(
                        env,
                        [live[index].state for index in indices],
                        [thresholds[index] for index in indices],
                    )
                    if len(actions) != len(indices):
                        raise RuntimeError(
                            "batched sampler returned the wrong action count"
                        )
                    for index, action in zip(indices, actions):
                        selected_actions[index] = int(action)
            else:
                rows = self._policy_probabilities_batch(
                    env, [lane.state for lane in live]
                )
                for index, (lane, probabilities) in enumerate(zip(live, rows)):
                    legal = [
                        int(action)
                        for action in env.legal_actions(lane.state)
                    ]
                    selected = self._sample(
                        [float(probabilities[action]) for action in legal],
                        lane.rng,
                    )
                    selected_actions[index] = legal[selected]
            if any(action is None for action in selected_actions):
                raise RuntimeError("batched rollout did not select every action")
            actions = [
                int(action)
                for action in selected_actions
                if action is not None
            ]
            step_batch = (
                getattr(env, "step_batch", None)
                if self.use_batch_step
                else None
            )
            if step_batch is not None:
                next_states = step_batch(
                    [lane.state for lane in live],
                    actions,
                )
            else:
                next_states = [
                    env.step(lane.state, action)
                    for lane, action in zip(live, actions)
                ]
            for lane, next_state in zip(live, next_states):
                lane.state = next_state
                lane.decisions += 1
                if lane.decisions > self.max_actions_per_rollout:
                    raise RuntimeError("root-policy rollout exceeded action limit")
            active = live
        return values

    def resolve(
        self,
        env,
        state,
        blueprint: torch.Tensor,
        public_range: PublicRangeSnapshot | None = None,
    ) -> PluribusSearchResult:
        if state.terminal or state.to_act is None:
            raise ValueError("search requires a live decision state")
        started = time.perf_counter()
        deadline = started + self.time_budget_ms / 1000.0
        hero = int(state.to_act)
        inferred_range = None
        if self.range_mode == "inferred":
            if public_range is None:
                raise ValueError("inferred range mode requires a public range")
            inferred_range = robust_inferred_range(
                public_range,
                temperature=self.range_temperature,
                uniform_contamination=self.uniform_contamination,
            )
        inferred_cumulative_weights = (
            tuple(itertools.accumulate(inferred_range.weights))
            if inferred_range is not None
            else None
        )
        legal = [int(action) for action in env.legal_actions(state)]
        if not legal:
            raise RuntimeError("search has no legal root actions")
        rollout_env = env
        convert_rollout_state = lambda value: value
        if self.use_native_rollouts and not bool(
            getattr(env, "native_backend", False)
        ):
            try:
                from heads_up_native import (
                    HeadsUpHoldemEngine as NativeHeadsUpHoldemEngine,
                    reference_state_to_native,
                )

                rollout_env = NativeHeadsUpHoldemEngine(
                    starting_stack=int(env.starting_stack),
                    small_blind=int(env.small_blind),
                    big_blind=int(env.big_blind),
                )
                convert_rollout_state = reference_state_to_native
            except ImportError:
                # The Python reference engine remains a supported fallback on
                # machines where the optional extension has not been built.
                pass

        regrets = {action: 0.0 for action in legal}
        strategy_sum = {action: 0.0 for action in legal}
        value_sum = {action: 0.0 for action in legal}
        value_square_sum = {action: 0.0 for action in legal}
        value_count = {action: 0 for action in legal}
        rollouts = 0
        iterations = 0
        measured_seconds_per_iteration: float | None = None

        while rollouts + len(legal) <= self.max_rollouts:
            now = time.perf_counter()
            if iterations and now >= deadline:
                break
            remaining_iterations = (
                self.max_rollouts - rollouts
            ) // len(legal)
            batch_count = min(self.batch_iterations, remaining_iterations)
            if measured_seconds_per_iteration is not None:
                # Size the final batch to stay close to the wall-clock budget.
                # The safety margin covers fixed launch/encoding overhead that
                # does not scale perfectly with the number of lanes.
                time_remaining = max(0.0, deadline - now)
                budgeted_iterations = int(
                    0.85
                    * time_remaining
                    / max(measured_seconds_per_iteration, 1e-9)
                )
                if budgeted_iterations <= 0:
                    break
                batch_count = min(batch_count, budgeted_iterations)
            batch_started = time.perf_counter()
            lanes: list[_RolloutLane] = []
            for local_iteration in range(batch_count):
                determinized = self._determinize(
                    env,
                    state,
                    hero,
                    inferred_range,
                    inferred_cumulative_weights,
                )
                rollout_determinized = convert_rollout_state(determinized)
                rollout_seed = self.rng.getrandbits(64)
                for action in legal:
                    lanes.append(
                        _RolloutLane(
                            iteration=local_iteration,
                            root_action=action,
                            # The reference engine's step contract already
                            # clones its input, so cloning here duplicated a
                            # full state copy for every root-action rollout.
                            state=rollout_env.step(
                                rollout_determinized, action
                            ),
                            rng=random.Random(rollout_seed),
                        )
                    )
            sampled_batch = self._rollouts_batch(rollout_env, lanes, hero)

            # Rollouts are independent of the regret strategy because every
            # legal root action is evaluated. Replaying updates in iteration
            # order therefore preserves the sequential resolver mathematics.
            for local_iteration in range(batch_count):
                strategy = self._regret_strategy(legal, regrets, blueprint)
                sampled_values = {
                    action: sampled_batch[(local_iteration, action)]
                    for action in legal
                }
                for action, value in sampled_values.items():
                    value_sum[action] += value
                    value_square_sum[action] += value * value
                    value_count[action] += 1
                    rollouts += 1
                node_value = sum(
                    strategy[action] * sampled_values[action]
                    for action in legal
                )
                for action in legal:
                    regrets[action] += sampled_values[action] - node_value
                    strategy_sum[action] += strategy[action]
                iterations += 1
            batch_seconds = time.perf_counter() - batch_started
            current_seconds_per_iteration = batch_seconds / batch_count
            if measured_seconds_per_iteration is None:
                measured_seconds_per_iteration = current_seconds_per_iteration
            else:
                measured_seconds_per_iteration = (
                    0.5 * measured_seconds_per_iteration
                    + 0.5 * current_seconds_per_iteration
                )

        search = self._average_strategy(
            legal, strategy_sum, regrets, blueprint
        )
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
            2.0 * float(env.bb),
            0.5 * float(state.pot),
        )
        refined = {
            action: base[action]
            * math.exp(
                max(
                    -3.0,
                    min(3.0, (means[action] - base_value) / risk_scale),
                )
            )
            for action in legal
        }
        refined_total = sum(refined.values())
        mixed = {
            action: refined[action] / refined_total for action in legal
        }
        pruned_actions: set[int] = set()
        if self.min_strategy_probability > 0.0:
            retained = [
                action
                for action in legal
                if mixed[action] + 1e-12
                >= self.min_strategy_probability
            ]
            if not retained:
                retained = [max(legal, key=mixed.__getitem__)]
            retained_set = set(retained)
            pruned_actions = set(legal) - retained_set
            retained_total = sum(mixed[action] for action in retained)
            mixed = {
                action: (
                    mixed[action] / retained_total
                    if action in retained_set
                    else 0.0
                )
                for action in legal
            }
        selected_position = self._sample(
            [mixed[action] for action in legal],
            self.rng,
        )
        choice_action = legal[selected_position]
        bb = float(state.big_blind)
        estimates = []
        for action in legal:
            count = value_count[action]
            mean = means[action]
            if count > 1:
                variance = max(
                    0.0,
                    (
                        value_square_sum[action]
                        - value_sum[action] * value_sum[action] / count
                    )
                    / (count - 1),
                )
                standard_error = math.sqrt(variance / count)
            else:
                standard_error = float("inf")
            estimates.append(
                CandidateEstimate(
                    action=SearchAction(
                        kind="abstract",
                        action=action,
                        label=ACTION_NAMES[action].replace("_", " "),
                        blueprint_prior=float(blueprint[action]),
                    ),
                    expected_final_payoff_bb=mean / bb,
                    standard_error_bb=standard_error / bb,
                    ci95_low_bb=(mean - 1.96 * standard_error) / bb,
                    ci95_high_bb=(mean + 1.96 * standard_error) / bb,
                    samples=count,
                    strategy_probability=mixed[action],
                    validation_ev_bb=mean / bb,
                    validation_ci95_low_bb=(
                        mean - 1.96 * standard_error
                    ) / bb,
                    validation_ci95_high_bb=(
                        mean + 1.96 * standard_error
                    ) / bb,
                    validation_samples=count,
                    statistically_dominated=False,
                    safety_pruned=action in pruned_actions,
                )
            )
        ordered = sorted(mixed.values(), reverse=True)
        strategy_gap = (
            ordered[0] - ordered[1] if len(ordered) > 1 else 1.0
        )
        unknown_cards = 52 - len(state.hole[hero]) - len(state.board)
        range_combos = (
            len(inferred_range.combos)
            if inferred_range is not None
            else math.comb(unknown_cards, 2)
        )
        range_ess = (
            inferred_range.effective_sample_size
            if inferred_range is not None
            else float(range_combos)
        )
        range_updates = inferred_range.updates if inferred_range is not None else 0
        elapsed_ms = 1000.0 * (time.perf_counter() - started)
        return PluribusSearchResult(
            choice=next(
                estimate.action
                for estimate in estimates
                if estimate.action.action == choice_action
            ),
            candidates=tuple(estimates),
            elapsed_ms=elapsed_ms,
            cfr_iterations=iterations,
            terminal_rollouts=rollouts,
            workers_responded=1,
            range_combos=range_combos,
            range_effective_sample_size=range_ess,
            range_updates=range_updates,
            native_backend=bool(
                getattr(rollout_env, "native_backend", False)
            ),
            converged=iterations > 0,
            used_blueprint_fallback=False,
            convergence_reason=(
                "three-player root resolver: paired terminal rollouts; "
                f"{self.batch_iterations}-sample batched policy inference; "
                f"{100.0 * self.blueprint_weight:g}% blueprint anchor; "
                f"{100.0 * (1.0 - self.blueprint_weight):g}% search weight; "
                "KL-bounded refinement; "
                "mixed-strategy sampling; "
                + (
                    f"{100.0 * self.min_strategy_probability:g}% minimum "
                    "strategy sampling floor; "
                    if self.min_strategy_probability > 0.0
                    else ""
                )
                + (
                    "65% tempered Bayesian likelihoods with 25% uniform "
                    "model-error contamination"
                    if inferred_range is not None
                    else "uniform hidden-card range"
                )
            ),
            validation_samples=min(value_count.values(), default=0),
            worker_agreement=1.0,
            strategy_gap=strategy_gap,
        )


__all__ = ["HeadsUpRootPolicySearch", "robust_inferred_range"]
