"""Distributionally robust heads-up root search.

This module is intentionally separate from ``heads_up_root_policy_search``.
It treats the blueprint-derived public range as a nominal belief, adds
model-error probability in action space, and evaluates root actions against
a KL-bounded adversarial reweighting of paired payoff samples.
"""

from __future__ import annotations

import math
import random
import time
from typing import Sequence

import torch

from heads_up_engine import ACTION_NAMES, evaluate_7card
from heads_up_pluribus_search import (
    CandidateEstimate,
    PluribusSearchResult,
    PublicRangeSnapshot,
    SearchAction,
)
from heads_up_root_policy_search import HeadsUpRootPolicySearch


def action_noise_likelihoods(
    observed_likelihoods: Sequence[float],
    *,
    legal_action_count: int,
    epsilon: float,
) -> list[float]:
    """Mix a blueprint action likelihood with a generic action tremble.

    The contamination is in action space, not hand space. This preserves the
    previous hand prior while allowing every hand to deviate from the
    blueprint without adding a fresh uniform hidden-card range.
    """

    if legal_action_count <= 0:
        raise ValueError("legal_action_count must be positive")
    if not 0.0 <= epsilon < 1.0:
        raise ValueError("epsilon must be in [0, 1)")
    floor = float(epsilon) / int(legal_action_count)
    return [
        (1.0 - float(epsilon)) * max(0.0, float(value)) + floor
        for value in observed_likelihoods
    ]


def kl_robust_lower_bound(
    values: Sequence[float],
    nominal_weights: Sequence[float] | None = None,
    *,
    radius: float,
) -> float:
    """Minimize expected value inside a forward-KL ball around a nominal law."""

    if radius < 0.0 or not math.isfinite(radius):
        raise ValueError("radius must be finite and nonnegative")
    if not values:
        raise ValueError("values must be non-empty")
    clean_values: list[float] = []
    clean_weights: list[float] = []
    if nominal_weights is None:
        nominal_weights = [1.0] * len(values)
    if len(values) != len(nominal_weights):
        raise ValueError("values and nominal_weights must align")
    for value, weight in zip(values, nominal_weights):
        numeric_value = float(value)
        numeric_weight = float(weight)
        if not math.isfinite(numeric_value):
            raise ValueError("values must be finite")
        if not math.isfinite(numeric_weight) or numeric_weight < 0.0:
            raise ValueError("nominal weights must be finite and nonnegative")
        if numeric_weight > 0.0:
            clean_values.append(numeric_value)
            clean_weights.append(numeric_weight)
    if not clean_values:
        raise ValueError("nominal distribution has no positive mass")
    total = sum(clean_weights)
    probabilities = [weight / total for weight in clean_weights]
    nominal_mean = sum(
        probability * value
        for probability, value in zip(probabilities, clean_values)
    )
    if radius <= 1e-15 or max(clean_values) - min(clean_values) <= 1e-15:
        return nominal_mean

    minimum = min(clean_values)
    minimum_mass = sum(
        probability
        for probability, value in zip(probabilities, clean_values)
        if abs(value - minimum) <= 1e-15
    )
    maximum_required_kl = -math.log(max(minimum_mass, 1e-300))
    if radius >= maximum_required_kl - 1e-12:
        return minimum

    def tilted(eta: float) -> tuple[float, float]:
        logits = [
            math.log(probability) - value / eta
            for probability, value in zip(probabilities, clean_values)
        ]
        offset = max(logits)
        unnormalized = [math.exp(logit - offset) for logit in logits]
        normalizer = sum(unnormalized)
        weights = [value / normalizer for value in unnormalized]
        mean = sum(
            weight * value for weight, value in zip(weights, clean_values)
        )
        divergence = sum(
            weight * math.log(weight / probability)
            for weight, probability in zip(weights, probabilities)
            if weight > 0.0
        )
        return divergence, mean

    scale = max(1.0, max(clean_values) - min(clean_values))
    low = scale * 1e-12
    high = scale
    while tilted(high)[0] > radius:
        high *= 2.0
        if high > scale * 1e12:
            return nominal_mean
    for _ in range(80):
        middle = 0.5 * (low + high)
        divergence, _ = tilted(middle)
        if divergence > radius:
            low = middle
        else:
            high = middle
    return tilted(high)[1]


class RobustHeadsUpSearch(HeadsUpRootPolicySearch):
    """Root resolver with action-noise beliefs and KL-robust action values."""

    def __init__(
        self,
        policy,
        *,
        time_budget_ms: int = 10_000,
        max_rollouts: int = 150_000,
        max_actions_per_rollout: int = 64,
        kl_radius: float = 0.20,
        continuation_action_noise: float = 0.04,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            policy,
            time_budget_ms=time_budget_ms,
            max_rollouts=max_rollouts,
            blueprint_weight=0.0,
            max_actions_per_rollout=max_actions_per_rollout,
            range_mode="inferred",
            range_temperature=1.0,
            uniform_contamination=0.0,
            seed=seed,
        )
        if kl_radius < 0.0:
            raise ValueError("kl_radius must be nonnegative")
        if not 0.0 <= continuation_action_noise < 1.0:
            raise ValueError(
                "continuation_action_noise must be in [0, 1)"
            )
        self.kl_radius = float(kl_radius)
        self.continuation_action_noise = float(
            continuation_action_noise
        )

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
            values = [float(probabilities[action]) for action in legal]
            if int(state.to_act) != hero and self.continuation_action_noise:
                floor = self.continuation_action_noise / len(legal)
                values = [
                    (1.0 - self.continuation_action_noise) * value + floor
                    for value in values
                ]
            selected = self._sample(values, rollout_rng)
            state = env.step(state, legal[selected])
            decisions += 1
            if decisions > self.max_actions_per_rollout:
                raise RuntimeError("robust rollout exceeded action limit")
        return float(state.payoffs[hero])

    @staticmethod
    def _weighted_mean(
        values: Sequence[float],
        weights: Sequence[float],
    ) -> float:
        total = sum(float(weight) for weight in weights)
        if total <= 0.0:
            raise ValueError("weights must have positive mass")
        return sum(
            float(value) * float(weight)
            for value, weight in zip(values, weights)
        ) / total

    def _exact_all_in_result(
        self,
        env,
        state,
        blueprint: torch.Tensor,
        public_range: PublicRangeSnapshot,
        started: float,
    ) -> PluribusSearchResult | None:
        """Exactly enumerate turn/river call equity when no betting remains."""

        hero = int(state.to_act)
        opponent = 1 - hero
        legal = [int(action) for action in env.legal_actions(state)]
        by_name = {ACTION_NAMES[action]: action for action in legal}
        if set(by_name) != {"fold", "call"} or len(state.board) not in {4, 5}:
            return None
        call_action = by_name["call"]
        fold_action = by_name["fold"]
        call_payment = int(env.action_payment(state, call_action))
        if not (
            bool(state.all_in[opponent])
            or call_payment == int(state.stacks[hero])
        ):
            return None

        fold_state = env.step(env.clone(state), fold_action)
        fold_value = float(fold_state.payoffs[hero])
        matched_contribution = min(
            int(state.total_contrib[hero]) + call_payment,
            int(state.total_contrib[opponent]),
        )
        known = set(int(card) for card in state.hole[hero])
        known.update(int(card) for card in state.board)
        combo_values: list[float] = []
        combo_weights: list[float] = []
        enumerated_runouts = 0
        for combo, combo_weight in zip(
            public_range.combos,
            public_range.weights,
        ):
            opponent_cards = tuple(int(card) for card in combo)
            if (
                opponent_cards[0] in known
                or opponent_cards[1] in known
                or opponent_cards[0] == opponent_cards[1]
            ):
                continue
            unavailable = known | set(opponent_cards)
            rivers = (
                [None]
                if len(state.board) == 5
                else [card for card in range(52) if card not in unavailable]
            )
            runout_values: list[float] = []
            for river in rivers:
                board = [int(card) for card in state.board]
                if river is not None:
                    board.append(int(river))
                hero_score = evaluate_7card(state.hole[hero], board)
                opponent_score = evaluate_7card(opponent_cards, board)
                if hero_score > opponent_score:
                    payoff = float(matched_contribution)
                elif hero_score < opponent_score:
                    payoff = -float(matched_contribution)
                else:
                    payoff = 0.0
                runout_values.append(payoff)
                enumerated_runouts += 1
            combo_values.append(sum(runout_values) / len(runout_values))
            combo_weights.append(float(combo_weight))
        if not combo_values:
            raise RuntimeError("exact all-in enumeration found no outcomes")

        nominal_call = self._weighted_mean(combo_values, combo_weights)
        robust_call = kl_robust_lower_bound(
            combo_values,
            combo_weights,
            radius=self.kl_radius,
        )
        robust_values = {
            fold_action: fold_value,
            call_action: robust_call,
        }
        nominal_values = {
            fold_action: fold_value,
            call_action: nominal_call,
        }
        choice_action = max(legal, key=robust_values.__getitem__)
        bb = float(state.big_blind)
        estimates = tuple(
            CandidateEstimate(
                action=SearchAction(
                    kind="abstract",
                    action=action,
                    label=ACTION_NAMES[action].replace("_", " "),
                    blueprint_prior=float(blueprint[action]),
                ),
                expected_final_payoff_bb=robust_values[action] / bb,
                standard_error_bb=0.0,
                ci95_low_bb=robust_values[action] / bb,
                ci95_high_bb=robust_values[action] / bb,
                samples=(
                    enumerated_runouts if action == call_action else 1
                ),
                strategy_probability=(
                    1.0 if action == choice_action else 0.0
                ),
                validation_ev_bb=nominal_values[action] / bb,
                validation_ci95_low_bb=nominal_values[action] / bb,
                validation_ci95_high_bb=nominal_values[action] / bb,
                validation_samples=(
                    enumerated_runouts if action == call_action else 1
                ),
            )
            for action in legal
        )
        square_sum = sum(
            float(weight) ** 2 for weight in public_range.weights
        )
        return PluribusSearchResult(
            choice=next(
                estimate.action
                for estimate in estimates
                if estimate.action.action == choice_action
            ),
            candidates=estimates,
            elapsed_ms=1000.0 * (time.perf_counter() - started),
            cfr_iterations=0,
            terminal_rollouts=enumerated_runouts,
            workers_responded=1,
            range_combos=len(public_range.combos),
            range_effective_sample_size=(
                1.0 / square_sum if square_sum > 0.0 else 0.0
            ),
            range_updates=public_range.updates,
            native_backend=bool(getattr(env, "native_backend", False)),
            converged=True,
            used_blueprint_fallback=False,
            convergence_reason=(
                "robust resolver: exact blocker-compatible turn/river "
                f"all-in equity; KL radius {self.kl_radius:.3f}; "
                "deterministic maximin root choice"
            ),
            validation_samples=enumerated_runouts,
            worker_agreement=1.0,
            strategy_gap=1.0,
        )

    def resolve(
        self,
        env,
        state,
        blueprint: torch.Tensor,
        public_range: PublicRangeSnapshot | None = None,
    ) -> PluribusSearchResult:
        if state.terminal or state.to_act is None:
            raise ValueError("search requires a live decision state")
        if public_range is None:
            raise ValueError("robust search requires a public range")
        started = time.perf_counter()
        exact = self._exact_all_in_result(
            env,
            state,
            blueprint,
            public_range,
            started,
        )
        if exact is not None:
            return exact

        deadline = started + self.time_budget_ms / 1000.0
        hero = int(state.to_act)
        legal = [int(action) for action in env.legal_actions(state)]
        if not legal:
            raise RuntimeError("search has no legal root actions")
        regrets = {action: 0.0 for action in legal}
        strategy_sum = {action: 0.0 for action in legal}
        samples = {action: [] for action in legal}
        maximum_iterations = max(1, self.max_rollouts // len(legal))
        particle_total = min(32, max(1, maximum_iterations // 3))
        particle_indices = self.rng.choices(
            range(len(public_range.combos)),
            weights=public_range.weights,
            k=particle_total,
        )
        particle_sums = {
            action: [0.0] * particle_total for action in legal
        }
        particle_counts = {
            action: [0] * particle_total for action in legal
        }
        rollouts = 0
        iterations = 0

        while rollouts + len(legal) <= self.max_rollouts:
            if iterations and time.perf_counter() >= deadline:
                break
            strategy = self._regret_strategy(legal, regrets, blueprint)
            particle_slot = iterations % particle_total
            combo_index = particle_indices[particle_slot]
            particle_range = PublicRangeSnapshot(
                combos=(public_range.combos[combo_index],),
                weights=(1.0,),
                effective_sample_size=1.0,
                updates=public_range.updates,
            )
            determinized = self._determinize(
                env,
                state,
                hero,
                particle_range,
            )
            rollout_seed = self.rng.getrandbits(64)
            sampled_values: dict[int, float] = {}
            complete = True
            for action in legal:
                if iterations and time.perf_counter() >= deadline:
                    complete = False
                    break
                value = self._rollout(
                    env,
                    env.step(env.clone(determinized), action),
                    hero,
                    random.Random(rollout_seed),
                )
                sampled_values[action] = value
                samples[action].append(value)
                particle_sums[action][particle_slot] += value
                particle_counts[action][particle_slot] += 1
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
        if not iterations:
            raise RuntimeError("robust search completed no root iteration")

        nominal = {
            action: sum(samples[action]) / len(samples[action])
            for action in legal
        }
        particle_values: dict[int, list[float]] = {}
        chance_prior_strength = 2.0
        for action in legal:
            values = []
            for total, count in zip(
                particle_sums[action],
                particle_counts[action],
            ):
                if not count:
                    continue
                local_mean = total / count
                values.append(
                    (
                        count * local_mean
                        + chance_prior_strength * nominal[action]
                    )
                    / (count + chance_prior_strength)
                )
            particle_values[action] = values
        robust = {
            action: kl_robust_lower_bound(
                particle_values[action],
                radius=self.kl_radius,
            )
            for action in legal
        }
        choice_action = max(legal, key=robust.__getitem__)
        bb = float(state.big_blind)
        estimates = []
        for action in legal:
            values = samples[action]
            count = len(values)
            mean = nominal[action]
            if count > 1:
                variance = sum(
                    (value - mean) ** 2 for value in values
                ) / (count - 1)
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
                    expected_final_payoff_bb=robust[action] / bb,
                    standard_error_bb=standard_error / bb,
                    ci95_low_bb=(
                        robust[action] - 1.96 * standard_error
                    ) / bb,
                    ci95_high_bb=(
                        robust[action] + 1.96 * standard_error
                    ) / bb,
                    samples=count,
                    strategy_probability=(
                        1.0 if action == choice_action else 0.0
                    ),
                    validation_ev_bb=mean / bb,
                    validation_ci95_low_bb=(
                        mean - 1.96 * standard_error
                    ) / bb,
                    validation_ci95_high_bb=(
                        mean + 1.96 * standard_error
                    ) / bb,
                    validation_samples=count,
                )
            )
        square_sum = sum(
            float(weight) ** 2 for weight in public_range.weights
        )
        ordered = sorted(robust.values(), reverse=True)
        strategy_gap = (
            (ordered[0] - ordered[1]) / bb if len(ordered) > 1 else 1.0
        )
        return PluribusSearchResult(
            choice=next(
                estimate.action
                for estimate in estimates
                if estimate.action.action == choice_action
            ),
            candidates=tuple(estimates),
            elapsed_ms=1000.0 * (time.perf_counter() - started),
            cfr_iterations=iterations,
            terminal_rollouts=rollouts,
            workers_responded=1,
            range_combos=len(public_range.combos),
            range_effective_sample_size=(
                1.0 / square_sum if square_sum > 0.0 else 0.0
            ),
            range_updates=public_range.updates,
            native_backend=bool(getattr(env, "native_backend", False)),
            converged=True,
            used_blueprint_fallback=False,
            convergence_reason=(
                "robust resolver: action-noise Bayesian public range; "
                f"KL radius {self.kl_radius:.3f}; opponent-hand particle "
                "reweighting after chance averaging; opponent continuation "
                "trembles; "
                "deterministic maximin root choice"
            ),
            validation_samples=min(
                (len(values) for values in samples.values()),
                default=0,
            ),
            worker_agreement=1.0,
            strategy_gap=strategy_gap,
        )


__all__ = [
    "RobustHeadsUpSearch",
    "action_noise_likelihoods",
    "kl_robust_lower_bound",
]
