"""Depth-limited, blueprint-guided CFR search for heads-up GUI play.

This is a practical Pluribus-style approximation, not a reproduction of the
published Pluribus system.  It combines:

* a per-hand public Bayesian range derived only from blueprint action
  likelihoods;
* exact, off-tree root bet sizes;
* alternating-traverser external-sampling CFR;
* normal/fold-biased/call-biased/raise-biased continuation strategies; and
* synchronized persistent worker solvers sharing root regrets between waves.

No persistent opponent personality model is used.
"""

from __future__ import annotations

import copy
from concurrent.futures import ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from itertools import combinations
import math
import multiprocessing
import os
from pathlib import Path
import random
import time
from typing import Iterable, Sequence

import torch

from heads_up_engine import ACTION_NAMES, NUM_ACTIONS, HeadsUpHoldemEngine
from heads_up_models import (
    build_action_descriptors,
    encode_information_state,
    masked_softmax,
)
from heads_up_production import load_policy_snapshot

try:
    from heads_up_native import (
        HeadsUpHoldemEngine as NativeHeadsUpHoldemEngine,
        bayesian_condition as native_bayesian_condition,
        estimate_all_in_ev as native_estimate_all_in_ev,
        estimate_terminal_call_scenarios as native_terminal_call_scenarios,
        hierarchical_regret_match_root as native_hierarchical_regret_match_root,
        regret_match_root as native_regret_match_root,
        reference_state_to_native,
    )
except (ImportError, OSError):
    NativeHeadsUpHoldemEngine = None
    native_bayesian_condition = None
    native_estimate_all_in_ev = None
    native_terminal_call_scenarios = None
    native_hierarchical_regret_match_root = None
    native_regret_match_root = None
    reference_state_to_native = None


CONTINUATION_NAMES = ("blueprint", "fold_bias", "call_bias", "raise_bias")


@dataclass(frozen=True)
class SearchAction:
    kind: str
    action: int | None = None
    raise_to: int | None = None
    label: str = ""
    blueprint_prior: float = 0.0


@dataclass(frozen=True)
class PublicRangeSnapshot:
    combos: tuple[tuple[int, int], ...]
    weights: tuple[float, ...]
    effective_sample_size: float
    updates: int


@dataclass(frozen=True)
class CandidateEstimate:
    action: SearchAction
    expected_final_payoff_bb: float
    standard_error_bb: float
    ci95_low_bb: float
    ci95_high_bb: float
    samples: int
    strategy_probability: float
    validation_ev_bb: float = float("nan")
    validation_ci95_low_bb: float = float("nan")
    validation_ci95_high_bb: float = float("nan")
    validation_samples: int = 0
    statistically_dominated: bool = False
    safety_pruned: bool = False


@dataclass(frozen=True)
class PluribusSearchResult:
    choice: SearchAction
    candidates: tuple[CandidateEstimate, ...]
    elapsed_ms: float
    cfr_iterations: int
    terminal_rollouts: int
    workers_responded: int
    range_combos: int
    range_effective_sample_size: float
    range_updates: int
    native_backend: bool = False
    converged: bool = False
    used_blueprint_fallback: bool = False
    convergence_reason: str = ""
    validation_samples: int = 0
    worker_agreement: float = 0.0
    strategy_gap: float = 0.0


def recommended_search_workers() -> int:
    """Use physical cores conservatively and leave one for GUI coordination."""

    physical = None
    try:
        import psutil

        physical = psutil.cpu_count(logical=False)
    except ImportError:
        pass
    logical = os.cpu_count() or 2
    if not physical:
        physical = max(1, logical // 2)
    return max(1, min(12, int(physical) - 1))


class BlueprintPublicRange:
    """Per-hand blocker-aware range with Bayesian public-action updates."""

    def __init__(self) -> None:
        self.combos: list[tuple[int, int]] = []
        self.weights: list[float] = []
        self.updates = 0

    def reset(self, known_cards: Iterable[int]) -> None:
        known = {int(card) for card in known_cards}
        available = [card for card in range(52) if card not in known]
        self.combos = [tuple(combo) for combo in combinations(available, 2)]
        probability = 1.0 / len(self.combos)
        self.weights = [probability] * len(self.combos)
        self.updates = 0

    def filter_known(self, known_cards: Iterable[int]) -> None:
        known = {int(card) for card in known_cards}
        kept = [
            (combo, weight)
            for combo, weight in zip(self.combos, self.weights)
            if combo[0] not in known and combo[1] not in known
        ]
        if not kept:
            raise RuntimeError("public range has no blocker-compatible hands")
        self.combos = [combo for combo, _ in kept]
        self.weights = [float(weight) for _, weight in kept]
        self._normalize()

    def condition(self, likelihoods: Sequence[float]) -> None:
        if len(likelihoods) != len(self.combos):
            raise ValueError("one action likelihood is required per range hand")
        if native_bayesian_condition is not None:
            result = native_bayesian_condition(
                self.weights,
                likelihoods,
                1e-6,
            )
            self.weights = [float(value) for value in result["weights"]]
        else:
            posterior = []
            for prior, likelihood in zip(self.weights, likelihoods):
                value = float(likelihood)
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError(
                        "range likelihoods must be finite and nonnegative"
                    )
                posterior.append(float(prior) * max(1e-6, value))
            self.weights = posterior
            self._normalize()
        self.updates += 1

    def _normalize(self) -> None:
        total = sum(self.weights)
        if total <= 0.0 or not math.isfinite(total):
            raise RuntimeError("public range has zero or invalid mass")
        self.weights = [weight / total for weight in self.weights]

    def snapshot(self) -> PublicRangeSnapshot:
        square_sum = sum(weight * weight for weight in self.weights)
        ess = 1.0 / square_sum if square_sum > 0.0 else 0.0
        return PublicRangeSnapshot(
            combos=tuple(self.combos),
            weights=tuple(self.weights),
            effective_sample_size=ess,
            updates=self.updates,
        )


def _action_family(action: int) -> str:
    name = ACTION_NAMES[int(action)]
    if name == "fold":
        return "fold"
    if name == "check":
        return "check"
    if name == "call":
        return "call"
    return "raise"


def _search_action_family(action: SearchAction) -> str:
    if action.kind != "abstract" or action.action is None:
        return "raise"
    family = _action_family(int(action.action))
    return "passive" if family in {"check", "call"} else family


def _search_action_family_id(action: SearchAction) -> int:
    return {"fold": 0, "passive": 1, "raise": 2}[
        _search_action_family(action)
    ]


def observed_action_likelihoods(
    env,
    state,
    probabilities: torch.Tensor,
    *,
    kind: str,
    raise_to: int | None = None,
) -> list[float]:
    """Map an exact public action to a likelihood for every candidate hand."""

    if probabilities.ndim != 2 or probabilities.shape[1] != NUM_ACTIONS:
        raise ValueError("range probability matrix must have shape [hands, 10]")
    normalized = str(kind).lower().replace("-", "_")
    legal = [int(action) for action in env.legal_actions(state)]
    if normalized in {"fold", "check", "call"}:
        matching = [
            action for action in legal if ACTION_NAMES[action] == normalized
        ]
        if len(matching) != 1:
            raise ValueError(f"observed {normalized} has no unique policy slot")
        return probabilities[:, matching[0]].clamp(min=1e-6).tolist()

    if normalized not in {"bet", "raise", "raise_to", "all_in", "allin"}:
        raise ValueError(f"unsupported observed action kind: {kind!r}")
    if raise_to is None:
        raise ValueError("raise_to is required for an observed aggressive action")
    aggressive = [action for action in legal if _action_family(action) == "raise"]
    if not aggressive:
        raise ValueError("observed raise has no legal policy raise slots")
    target = int(raise_to)
    exact = [
        action for action in aggressive
        if int(env.action_target(state, action)) == target
    ]
    if exact:
        return probabilities[:, exact[0]].clamp(min=1e-6).tolist()

    actor = int(state.to_act)
    contribution = int(state.street_contrib[actor])
    to_call = int(env.amount_to_call(state, actor))
    denominator = max(1.0, float(state.pot + to_call))
    observed_fraction = max(
        1e-3,
        float(target - contribution - to_call) / denominator,
    )
    kernel = []
    for action in aggressive:
        action_target = int(env.action_target(state, action))
        fraction = max(
            1e-3,
            float(action_target - contribution - to_call) / denominator,
        )
        distance = math.log(fraction / observed_fraction)
        kernel.append(math.exp(-0.5 * (distance / 0.32) ** 2))
    kernel_tensor = torch.tensor(kernel, dtype=probabilities.dtype)
    kernel_tensor /= kernel_tensor.sum().clamp(min=1e-12)
    likelihood = probabilities[:, aggressive] @ kernel_tensor
    return likelihood.clamp(min=1e-6).tolist()


def sanitize_search_state(state, hero: int):
    sanitized = copy.deepcopy(state)
    opponent = 1 - int(hero)
    sanitized.hole[opponent] = [None] * len(sanitized.hole[opponent])
    sanitized.burned = [None] * len(sanitized.burned)
    sanitized.deck = [None] * len(sanitized.deck)
    return sanitized


def _candidate_signature(state) -> tuple:
    return (
        int(state.terminal),
        tuple(int(value) for value in state.stacks),
        tuple(int(value) for value in state.street_contrib),
        int(state.pot),
        int(state.current_bet),
        state.to_act,
    )


def generate_search_actions(
    env,
    state,
    blueprint: torch.Tensor,
    *,
    raise_fractions: Iterable[float] = (
        0.25,
        1.0 / 3.0,
        0.5,
        0.6,
        0.71,
        0.75,
        1.0,
        1.25,
        1.5,
        2.0,
    ),
) -> tuple[SearchAction, ...]:
    candidates: list[SearchAction] = []
    signatures: set[tuple] = set()
    legal = [int(action) for action in env.legal_actions(state)]
    for action in legal:
        child = env.step(state, action)
        signature = _candidate_signature(child)
        if signature in signatures:
            continue
        signatures.add(signature)
        candidates.append(
            SearchAction(
                kind="abstract",
                action=action,
                label=ACTION_NAMES[action].replace("_", " "),
                blueprint_prior=float(blueprint[action]),
            )
        )
    actor = int(state.to_act)
    contribution = int(state.street_contrib[actor])
    maximum = contribution + int(state.stacks[actor])
    to_call = int(env.amount_to_call(state, actor))
    pot_after_call = int(state.pot) + to_call
    minimum = int(state.current_bet) + int(state.min_raise)
    targets = {minimum, maximum}
    for fraction in raise_fractions:
        extra = int(math.floor(float(fraction) * pot_after_call + 0.5))
        targets.add(contribution + to_call + max(1, extra))
    aggressive = [action for action in legal if _action_family(action) == "raise"]
    for target in sorted(targets):
        if target <= int(state.current_bet) or target > maximum:
            continue
        try:
            child = env.step_exact(state, "raise_to", target)
        except (TypeError, ValueError):
            continue
        signature = _candidate_signature(child)
        if signature in signatures:
            continue
        signatures.add(signature)
        nearest = (
            min(
                aggressive,
                key=lambda action: abs(
                    int(env.action_target(state, action)) - target
                ),
            )
            if aggressive
            else None
        )
        prior = float(blueprint[nearest]) if nearest is not None else 0.0
        candidates.append(
            SearchAction(
                kind="raise_to",
                raise_to=int(target),
                label=f"raise to {target}",
                blueprint_prior=0.5 * prior,
            )
        )
    return tuple(candidates)


def _apply_action(env, state, action):
    if isinstance(action, SearchAction):
        if action.kind == "abstract":
            return env.step(state, int(action.action))
        if action.kind == "raise_to":
            return env.step_exact(state, "raise_to", int(action.raise_to))
        return env.step_exact(state, action.kind)
    return env.step(state, int(action))


class _SnapshotPolicy:
    def __init__(self, path: str) -> None:
        self.snapshot = load_policy_snapshot(path, device="cpu")
        self.max_history = int(self.snapshot.metadata["max_history"])

    @torch.inference_mode()
    def probabilities(self, env, state) -> torch.Tensor:
        return self.probabilities_batch(env, [state])[0]

    @torch.inference_mode()
    def probabilities_batch(self, env, states: Sequence) -> torch.Tensor:
        if not states:
            return torch.empty((0, NUM_ACTIONS), dtype=torch.float32)
        actor = int(states[0].to_act)
        if any(int(state.to_act) != actor for state in states):
            raise ValueError("batched policy states must share one actor")
        rows = []
        masks = []
        for state in states:
            legal = tuple(int(action) for action in env.legal_actions(state))
            descriptors = build_action_descriptors(env, state)
            rows.append(
                encode_information_state(
                    state,
                    actor,
                    legal,
                    env.bb,
                    self.max_history,
                    action_descriptors=descriptors,
                )
            )
            mask = torch.zeros(NUM_ACTIONS, dtype=torch.float32)
            mask[list(legal)] = 1.0
            masks.append(mask)
        observations = torch.stack(rows)
        legal_masks = torch.stack(masks)
        logits = self.snapshot.policy_nets[actor](observations)
        return masked_softmax(logits, legal_masks).cpu()

    @torch.inference_mode()
    def _legacy_single_probabilities(self, env, state) -> torch.Tensor:
        actor = int(state.to_act)
        legal = tuple(int(action) for action in env.legal_actions(state))
        descriptors = build_action_descriptors(env, state)
        observation = encode_information_state(
            state,
            actor,
            legal,
            env.bb,
            self.max_history,
            action_descriptors=descriptors,
        )
        mask = torch.zeros(NUM_ACTIONS, dtype=torch.float32)
        mask[list(legal)] = 1.0
        logits = self.snapshot.policy_nets[actor](observation.unsqueeze(0))[0]
        return masked_softmax(logits, mask).cpu()


_WORKER_POLICY: _SnapshotPolicy | None = None


def _worker_initializer(policy_path: str) -> None:
    global _WORKER_POLICY
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    _WORKER_POLICY = _SnapshotPolicy(policy_path)


def _worker_ping() -> int:
    if _WORKER_POLICY is None:
        raise RuntimeError("worker policy is not initialized")
    return os.getpid()


def _sample_index(probabilities: Sequence[float], rng: random.Random) -> int:
    threshold = rng.random()
    cumulative = 0.0
    fallback = max(range(len(probabilities)), key=probabilities.__getitem__)
    for index, probability in enumerate(probabilities):
        cumulative += max(0.0, float(probability))
        if threshold <= cumulative + 1e-12:
            return index
    return fallback


def _regret_strategy(regrets: Sequence[float], fallback: Sequence[float]) -> list[float]:
    positive = [max(0.0, float(value)) for value in regrets]
    total = sum(positive)
    if total > 1e-12:
        return [value / total for value in positive]
    base = [max(0.0, float(value)) for value in fallback]
    base_total = sum(base)
    if base_total <= 1e-12:
        return [1.0 / len(base)] * len(base)
    return [value / base_total for value in base]


@dataclass
class _RegretNode:
    actions: tuple
    regrets: list[float]
    strategy_sum: list[float]


def _new_node(actions: tuple) -> _RegretNode:
    return _RegretNode(
        actions=actions,
        regrets=[0.0] * len(actions),
        strategy_sum=[0.0] * len(actions),
    )


def _infoset_key(state, actor: int, suffix: str) -> tuple:
    public_history = tuple(
        (
            int(event.player),
            int(event.street),
            str(event.kind),
            int(event.amount),
            int(event.raise_to),
        )
        for event in state.history
    )
    return (
        suffix,
        actor,
        int(state.street),
        tuple(int(card) for card in state.board),
        tuple(sorted(int(card) for card in state.hole[actor])),
        tuple(int(value) for value in state.stacks),
        tuple(int(value) for value in state.street_contrib),
        int(state.pot),
        int(state.current_bet),
        public_history,
    )


def _biased_probabilities(
    probabilities: torch.Tensor,
    legal: Sequence[int],
    variant: int,
) -> torch.Tensor:
    if variant == 0:
        return probabilities
    desired = ("fold", "call", "raise")[variant - 1]
    adjusted = probabilities.clone()
    for action in legal:
        family = _action_family(int(action))
        adjusted[action] *= 2.75 if family == desired else 0.65
    total = float(adjusted.sum())
    return adjusted / total if total > 1e-12 else probabilities


class _WorkerSolver:
    def __init__(
        self,
        env,
        root_state,
        candidates: tuple[SearchAction, ...],
        public_range: PublicRangeSnapshot,
        *,
        hero: int,
        rng: random.Random,
        deadline: float,
        depth_limit: int,
        initial_root_regrets: Sequence[float] | None = None,
    ) -> None:
        self.env = env
        self.root_state = root_state
        self.candidates = candidates
        self.public_range = public_range
        self.hero = int(hero)
        self.opponent = 1 - self.hero
        self.rng = rng
        self.deadline = deadline
        self.depth_limit = int(depth_limit)
        self.initial_root_regrets = (
            [float(value) for value in initial_root_regrets]
            if initial_root_regrets is not None
            else None
        )
        if (
            self.initial_root_regrets is not None
            and len(self.initial_root_regrets) != len(candidates)
        ):
            raise ValueError("shared root regrets must align with candidates")
        self._root_seeded = False
        self.native_backend = bool(getattr(env, "native_backend", False))
        self.nodes: dict[tuple, _RegretNode] = {}
        self.root_key: tuple | None = None
        self.value_sum = [0.0] * len(candidates)
        self.value_square_sum = [0.0] * len(candidates)
        self.value_count = [0] * len(candidates)
        self.iterations = 0
        self.terminal_rollouts = 0

    def _determinize(self):
        combo_index = self.rng.choices(
            range(len(self.public_range.combos)),
            weights=self.public_range.weights,
            k=1,
        )[0]
        opponent_cards = list(self.public_range.combos[combo_index])
        sampled = (
            copy.deepcopy(self.root_state)
            if self.native_backend
            else self.env.clone(self.root_state)
        )
        hero_cards = [int(card) for card in sampled.hole[self.hero]]
        board = [int(card) for card in sampled.board]
        known = set(hero_cards + opponent_cards + board)
        pool = [card for card in range(52) if card not in known]
        self.rng.shuffle(pool)
        sampled.hole[self.opponent] = opponent_cards
        cursor = 0
        sampled.burned = pool[cursor : cursor + len(sampled.burned)]
        cursor += len(sampled.burned)
        sampled.deck = pool[cursor : cursor + len(sampled.deck)]
        cursor += len(sampled.deck)
        if cursor != len(pool):
            raise RuntimeError("range determinization card zones are inconsistent")
        if self.native_backend:
            if reference_state_to_native is None:
                raise RuntimeError("native state converter is unavailable")
            return reference_state_to_native(sampled)
        return sampled

    def _node(
        self,
        key: tuple,
        actions: tuple,
    ) -> _RegretNode:
        node = self.nodes.get(key)
        if node is None:
            node = _new_node(actions)
            self.nodes[key] = node
        elif node.actions != actions:
            raise RuntimeError("information-set action contract changed")
        return node

    def _fallback(self, state, actions: tuple) -> list[float]:
        if actions and isinstance(actions[0], SearchAction):
            return [max(1e-6, action.blueprint_prior) for action in actions]
        probabilities = _WORKER_POLICY.probabilities(self.env, state)
        return [float(probabilities[int(action)]) for action in actions]

    def _continuation_value(self, state, traverser: int) -> float:
        strategies = {}
        nodes = {}
        for seat in (self.hero, self.opponent):
            key = _infoset_key(state, seat, f"continuation_p{seat}")
            node = self._node(key, tuple(range(len(CONTINUATION_NAMES))))
            nodes[seat] = node
            strategies[seat] = _regret_strategy(node.regrets, [0.25] * 4)
            for index, probability in enumerate(strategies[seat]):
                node.strategy_sum[index] += (
                    (self.iterations + 1) * probability
                )

        other = 1 - traverser
        other_variant = _sample_index(strategies[other], self.rng)
        rollout_seed = self.rng.getrandbits(64)
        variant_profiles = []
        for variant in range(len(CONTINUATION_NAMES)):
            variant_profiles.append(
                {
                    traverser: variant,
                    other: other_variant,
                }
            )
        values = self._rollout_variants(
            state,
            traverser,
            variant_profiles,
            rollout_seed,
        )
        node_value = sum(
            probability * value
            for probability, value in zip(strategies[traverser], values)
        )
        node = nodes[traverser]
        for index, value in enumerate(values):
            node.regrets[index] += value - node_value
        return node_value

    def _rollout_variants(
        self,
        state,
        traverser: int,
        variant_profiles: Sequence[dict[int, int]],
        seed: int,
    ) -> list[float]:
        states = [self.env.clone(state) for _ in variant_profiles]
        rngs = [random.Random(seed) for _ in variant_profiles]
        decisions = [0] * len(states)
        while any(not value.terminal for value in states):
            if time.monotonic() >= self.deadline:
                raise TimeoutError
            for actor in (0, 1):
                indices = [
                    index
                    for index, value in enumerate(states)
                    if not value.terminal and int(value.to_act) == actor
                ]
                if not indices:
                    continue
                matrix = _WORKER_POLICY.probabilities_batch(
                    self.env,
                    [states[index] for index in indices],
                )
                selected_actions = []
                for row, index in enumerate(indices):
                    current = states[index]
                    legal = [
                        int(action)
                        for action in self.env.legal_actions(current)
                    ]
                    probabilities = _biased_probabilities(
                        matrix[row],
                        legal,
                        int(variant_profiles[index][actor]),
                    )
                    selected = _sample_index(
                        [float(probabilities[action]) for action in legal],
                        rngs[index],
                    )
                    selected_actions.append(legal[selected])
                    decisions[index] += 1
                    if decisions[index] > 64:
                        raise RuntimeError(
                            "continuation rollout exceeded action limit"
                        )
                if self.native_backend and hasattr(self.env, "step_batch"):
                    advanced = self.env.step_batch(
                        [states[index] for index in indices],
                        selected_actions,
                    )
                    for index, child in zip(indices, advanced):
                        states[index] = child
                else:
                    for index, action in zip(indices, selected_actions):
                        states[index] = self.env.step(states[index], action)
        self.terminal_rollouts += len(states)
        return [float(value.payoffs[traverser]) for value in states]

    def _cfr(self, state, traverser: int, depth: int) -> float:
        if state.terminal:
            return float(state.payoffs[traverser])
        if time.monotonic() >= self.deadline:
            raise TimeoutError
        if depth >= self.depth_limit:
            return self._continuation_value(state, traverser)
        actor = int(state.to_act)
        actions: tuple = (
            self.candidates
            if depth == 0
            else tuple(int(action) for action in self.env.legal_actions(state))
        )
        key = _infoset_key(state, actor, "root" if depth == 0 else "subgame")
        node = self._node(key, actions)
        if (
            depth == 0
            and not self._root_seeded
            and self.initial_root_regrets is not None
        ):
            node.regrets[:] = self.initial_root_regrets
            self._root_seeded = True
        if any(regret > 0.0 for regret in node.regrets):
            strategy = _regret_strategy(node.regrets, ())
        else:
            strategy = _regret_strategy(
                node.regrets,
                self._fallback(state, actions),
            )
        for index, probability in enumerate(strategy):
            node.strategy_sum[index] += (
                (self.iterations + 1) * probability
            )
        if depth == 0 and actor == self.hero:
            self.root_key = key

        if actor != traverser:
            index = _sample_index(strategy, self.rng)
            child = _apply_action(
                self.env,
                self.env.clone(state),
                actions[index],
            )
            return self._cfr(child, traverser, depth + 1)

        values = []
        paired_rng_state = self.rng.getstate()
        next_rng_state = None
        for action in actions:
            # Counterfactual action values share chance/opponent random numbers
            # wherever their subsequent trees align.  Retain only the first
            # branch's RNG advancement after the paired comparison.
            self.rng.setstate(paired_rng_state)
            child = _apply_action(
                self.env,
                self.env.clone(state),
                action,
            )
            values.append(self._cfr(child, traverser, depth + 1))
            if next_rng_state is None:
                next_rng_state = self.rng.getstate()
        if next_rng_state is not None:
            self.rng.setstate(next_rng_state)
        node_value = sum(
            probability * value
            for probability, value in zip(strategy, values)
        )
        for index, value in enumerate(values):
            node.regrets[index] += value - node_value
        if depth == 0 and traverser == self.hero:
            for index, value in enumerate(values):
                self.value_sum[index] += value
                self.value_square_sum[index] += value * value
                self.value_count[index] += 1
        return node_value

    def run(self, iteration_cap: int) -> dict:
        while self.iterations < iteration_cap and time.monotonic() < self.deadline:
            determinized = self._determinize()
            traverser = self.hero if self.iterations % 2 == 0 else self.opponent
            try:
                self._cfr(determinized, traverser, 0)
            except TimeoutError:
                break
            self.iterations += 1
        root = self.nodes.get(self.root_key) if self.root_key is not None else None
        strategy_sum = (
            list(root.strategy_sum)
            if root is not None
            else [0.0] * len(self.candidates)
        )
        root_regrets = (
            list(root.regrets)
            if root is not None
            else [0.0] * len(self.candidates)
        )
        return {
            "strategy_sum": strategy_sum,
            "root_regrets": root_regrets,
            "value_sum": self.value_sum,
            "value_square_sum": self.value_square_sum,
            "value_count": self.value_count,
            "iterations": self.iterations,
            "terminal_rollouts": self.terminal_rollouts,
        }


def _search_worker(
    sanitized_state,
    candidates: tuple[SearchAction, ...],
    public_range: PublicRangeSnapshot,
    seed: int,
    budget_seconds: float,
    iteration_cap: int,
    depth_limit: int,
    use_native: bool,
    initial_root_regrets: Sequence[float] | None = None,
) -> dict:
    if _WORKER_POLICY is None:
        raise RuntimeError("worker policy is not initialized")
    engine_type = (
        NativeHeadsUpHoldemEngine
        if use_native and NativeHeadsUpHoldemEngine is not None
        else HeadsUpHoldemEngine
    )
    env = engine_type(
        starting_stack=max(int(value) for value in sanitized_state.initial_stacks),
        small_blind=int(sanitized_state.small_blind),
        big_blind=int(sanitized_state.big_blind),
    )
    solver = _WorkerSolver(
        env,
        sanitized_state,
        candidates,
        public_range,
        hero=int(sanitized_state.to_act),
        rng=random.Random(int(seed)),
        deadline=time.monotonic() + max(0.05, float(budget_seconds)),
        depth_limit=depth_limit,
        initial_root_regrets=initial_root_regrets,
    )
    result = solver.run(iteration_cap)
    if initial_root_regrets is not None:
        result["root_regrets"] = [
            float(value) - float(seed_value)
            for value, seed_value in zip(
                result["root_regrets"], initial_root_regrets
            )
        ]
    result["native_backend"] = bool(getattr(env, "native_backend", False))
    return result


class MultiprocessPluribusSearch:
    """Coordinator for synchronized depth-limited CFR worker solvers."""

    def __init__(
        self,
        policy_path: str | Path,
        *,
        workers: int | None = None,
        time_budget_seconds: float = 6.0,
        iteration_cap_per_worker: int = 100_000,
        depth_limit: int = 3,
        validation_rollouts: int = 64,
        all_in_validation_samples: int = 50_000,
        native_backend: bool = True,
        seed: int | None = None,
    ) -> None:
        if workers is None or int(workers) == 0:
            workers = recommended_search_workers()
        if workers < 1:
            raise ValueError("workers must be positive")
        if not 0.1 <= time_budget_seconds <= 12.0:
            raise ValueError("time budget must be in [0.1, 12.0] seconds")
        if depth_limit < 1:
            raise ValueError("depth_limit must be positive")
        if validation_rollouts < 16:
            raise ValueError("validation_rollouts must be at least 16")
        if all_in_validation_samples < 1_000:
            raise ValueError("all_in_validation_samples must be at least 1000")
        self.policy_path = str(Path(policy_path).resolve())
        self.workers = int(workers)
        self.time_budget_seconds = float(time_budget_seconds)
        self.iteration_cap_per_worker = int(iteration_cap_per_worker)
        self.depth_limit = int(depth_limit)
        self.validation_rollouts = int(validation_rollouts)
        self.all_in_validation_samples = int(all_in_validation_samples)
        self.native_backend = bool(
            native_backend and NativeHeadsUpHoldemEngine is not None
        )
        self.rng = random.Random(seed)
        self.coordinator_policy = _SnapshotPolicy(self.policy_path)
        context = multiprocessing.get_context("spawn")
        self.pool = ProcessPoolExecutor(
            max_workers=self.workers,
            mp_context=context,
            initializer=_worker_initializer,
            initargs=(self.policy_path,),
        )
        warm = [self.pool.submit(_worker_ping) for _ in range(self.workers)]
        done, pending = wait(warm, timeout=45.0)
        for future in pending:
            future.cancel()
        if pending or any(future.exception() is not None for future in done):
            self.pool.shutdown(wait=False, cancel_futures=True)
            raise RuntimeError("could not warm all Pluribus-search workers")

    def close(self, *, wait_for_workers: bool = False) -> None:
        self.pool.shutdown(wait=bool(wait_for_workers), cancel_futures=True)

    def _determinize_for_validation(
        self,
        sanitized,
        public_range: PublicRangeSnapshot,
        rng: random.Random,
        *,
        native: bool,
    ):
        combo_index = rng.choices(
            range(len(public_range.combos)),
            weights=public_range.weights,
            k=1,
        )[0]
        opponent = 1 - int(sanitized.to_act)
        sampled = copy.deepcopy(sanitized)
        opponent_cards = list(public_range.combos[combo_index])
        hero_cards = [int(card) for card in sampled.hole[int(sanitized.to_act)]]
        board = [int(card) for card in sampled.board]
        known = set(hero_cards + opponent_cards + board)
        pool = [card for card in range(52) if card not in known]
        rng.shuffle(pool)
        sampled.hole[opponent] = opponent_cards
        cursor = 0
        sampled.burned = pool[cursor : cursor + len(sampled.burned)]
        cursor += len(sampled.burned)
        sampled.deck = pool[cursor : cursor + len(sampled.deck)]
        if native:
            if reference_state_to_native is None:
                raise RuntimeError("native state converter is unavailable")
            return reference_state_to_native(sampled)
        return sampled

    def _validate_candidates(
        self,
        sanitized,
        candidates: tuple[SearchAction, ...],
        public_range: PublicRangeSnapshot,
        *,
        deadline: float,
    ) -> tuple[list[list[float]], bool]:
        """Independently evaluate root actions against frozen blueprint play."""

        use_native = self.native_backend
        engine_type = (
            NativeHeadsUpHoldemEngine if use_native else HeadsUpHoldemEngine
        )
        env = engine_type(
            starting_stack=max(int(value) for value in sanitized.initial_stacks),
            small_blind=int(sanitized.small_blind),
            big_blind=int(sanitized.big_blind),
        )
        values = [[] for _ in candidates]
        for _ in range(self.validation_rollouts):
            if time.monotonic() >= deadline:
                break
            deal_seed = self.rng.getrandbits(64)
            deal_rng = random.Random(deal_seed)
            root = self._determinize_for_validation(
                sanitized,
                public_range,
                deal_rng,
                native=use_native,
            )
            states = [
                _apply_action(env, env.clone(root), candidate)
                for candidate in candidates
            ]
            rngs = [
                random.Random(deal_seed ^ 0x9E3779B97F4A7C15)
                for _ in candidates
            ]
            decisions = [0] * len(states)
            completed = True
            while any(not state.terminal for state in states):
                if time.monotonic() >= deadline:
                    completed = False
                    break
                for actor in (0, 1):
                    indices = [
                        index
                        for index, state in enumerate(states)
                        if not state.terminal and int(state.to_act) == actor
                    ]
                    if not indices:
                        continue
                    matrix = self.coordinator_policy.probabilities_batch(
                        env,
                        [states[index] for index in indices],
                    )
                    for row, index in enumerate(indices):
                        legal = [
                            int(action)
                            for action in env.legal_actions(states[index])
                        ]
                        selected = _sample_index(
                            [float(matrix[row, action]) for action in legal],
                            rngs[index],
                        )
                        states[index] = env.step(
                            states[index],
                            legal[selected],
                        )
                        decisions[index] += 1
                        if decisions[index] > 64:
                            raise RuntimeError(
                                "validation rollout exceeded action limit"
                            )
            if not completed:
                break
            hero = int(sanitized.to_act)
            for index, terminal in enumerate(states):
                values[index].append(float(terminal.payoffs[hero]))
        return values, use_native

    @staticmethod
    def _mean_interval(values: Sequence[float], bb: float) -> tuple:
        count = len(values)
        if not count:
            return float("nan"), float("inf"), float("-inf"), float("inf")
        mean = sum(values) / count
        if count > 1:
            variance = sum((value - mean) ** 2 for value in values) / (count - 1)
            standard_error = math.sqrt(max(0.0, variance) / count)
        else:
            standard_error = float("inf")
        return (
            mean / bb,
            standard_error / bb,
            (mean - 1.96 * standard_error) / bb,
            (mean + 1.96 * standard_error) / bb,
        )

    @staticmethod
    def _blueprint_base(
        candidates: tuple[SearchAction, ...],
    ) -> list[float]:
        # Exact off-tree candidates are search-only. Giving every such size a
        # copy of its nearest abstract prior would multiply total raise mass.
        base = [
            max(0.0, candidate.blueprint_prior)
            if candidate.kind == "abstract"
            else 0.0
            for candidate in candidates
        ]
        total = sum(base)
        if total <= 1e-12:
            return [1.0 / len(candidates)] * len(candidates)
        return [value / total for value in base]

    def _validate_all_in(
        self,
        env,
        state,
        candidate: SearchAction,
        public_range: PublicRangeSnapshot,
    ) -> dict | None:
        """High-precision native veto for a root shove."""

        if not self.native_backend or native_estimate_all_in_ev is None:
            return None
        hero = int(state.to_act)
        opponent = 1 - hero
        hypothetical_children = []
        for combo in public_range.combos:
            hypothetical = copy.deepcopy(state)
            hypothetical.hole[opponent] = list(combo)
            child = _apply_action(env, hypothetical, candidate)
            if child.terminal or child.to_act is None:
                return None
            hypothetical_children.append(child)
        matrix = self.coordinator_policy.probabilities_batch(
            env,
            hypothetical_children,
        )
        legal = [
            int(action)
            for action in env.legal_actions(hypothetical_children[0])
        ]
        call_actions = [
            action for action in legal if ACTION_NAMES[action] == "call"
        ]
        fold_actions = [
            action for action in legal if ACTION_NAMES[action] == "fold"
        ]
        if len(call_actions) != 1 or len(fold_actions) != 1:
            return None
        call_probabilities = [
            float(value)
            for value in matrix[:, call_actions[0]].tolist()
        ]
        folded = env.step_exact(
            copy.deepcopy(hypothetical_children[0]),
            "fold",
        )
        fold_payoff = float(folded.payoffs[hero])
        child = hypothetical_children[0]
        call_amount = min(
            int(env.amount_to_call(child, opponent)),
            int(child.stacks[opponent]),
        )
        matched = min(
            int(child.total_contrib[hero]),
            int(child.total_contrib[opponent]) + call_amount,
        )
        result = native_estimate_all_in_ev(
            state.hole[hero],
            state.board,
            public_range.combos,
            public_range.weights,
            call_probabilities,
            fold_payoff=fold_payoff,
            win_payoff=float(matched),
            tie_payoff=0.0,
            loss_payoff=float(-matched),
            samples=self.all_in_validation_samples,
            seed=self.rng.getrandbits(63),
            robust_best_response=True,
        )
        bb = float(state.big_blind)
        return {
            "mean_bb": float(result["mean"]) / bb,
            "ci_low_bb": float(result["ci95_low"]) / bb,
            "ci_high_bb": float(result["ci95_high"]) / bb,
            "samples": int(result["samples"]),
            "call_rate": float(result["call_rate"]),
            "called_equity": float(result["called_equity"]),
            "robust_best_response": bool(result["robust_best_response"]),
            "robust_call_hands": int(result["robust_call_hands"]),
        }

    def _validate_terminal_call(
        self,
        env,
        state,
        candidates: tuple[SearchAction, ...],
        public_range: PublicRangeSnapshot,
    ) -> tuple[int, dict] | None:
        """Robustly price a terminal call without reading hidden cards."""

        if (
            not self.native_backend
            or native_terminal_call_scenarios is None
            or len(state.board) < 3
        ):
            return None
        fold_indices = [
            index
            for index, candidate in enumerate(candidates)
            if candidate.kind == "abstract"
            and candidate.action is not None
            and ACTION_NAMES[int(candidate.action)] == "fold"
        ]
        call_indices = [
            index
            for index, candidate in enumerate(candidates)
            if candidate.kind == "abstract"
            and candidate.action is not None
            and ACTION_NAMES[int(candidate.action)] == "call"
        ]
        if len(fold_indices) != 1 or len(call_indices) != 1:
            return None
        hero = int(state.to_act)
        opponent = 1 - hero
        if int(state.stacks[opponent]) != 0:
            return None
        fold_child = _apply_action(
            env,
            copy.deepcopy(state),
            candidates[fold_indices[0]],
        )
        if not fold_child.terminal:
            return None
        call_amount = min(
            int(env.amount_to_call(state, hero)),
            int(state.stacks[hero]),
        )
        hero_total = int(state.total_contrib[hero]) + call_amount
        opponent_total = int(state.total_contrib[opponent])
        matched = min(hero_total, opponent_total)
        result = native_terminal_call_scenarios(
            state.hole[hero],
            state.board,
            public_range.combos,
            public_range.weights,
            fold_payoff=float(fold_child.payoffs[hero]),
            win_payoff=float(matched),
            tie_payoff=0.0,
            loss_payoff=float(-matched),
            nominal_samples=self.all_in_validation_samples,
            seed=self.rng.getrandbits(63),
        )
        bb = float(state.big_blind)
        rows = []
        for row in result["scenarios"]:
            rows.append(
                {
                    "name": str(row["name"]),
                    "mean_bb": float(row["mean"]) / bb,
                    "ci_low_bb": float(row["ci95_low"]) / bb,
                    "ci_high_bb": float(row["ci95_high"]) / bb,
                    "equity": float(row["equity"]),
                }
            )
        worst = min(rows, key=lambda row: row["mean_bb"])
        return call_indices[0], {
            "fold_payoff_bb": float(result["fold_payoff"]) / bb,
            "worst_mean_bb": float(result["worst_mean"]) / bb,
            "worst_name": str(result["worst_name"]),
            "worst_ci_low_bb": worst["ci_low_bb"],
            "worst_ci_high_bb": worst["ci_high_bb"],
            "worst_equity": worst["equity"],
            "scenarios": rows,
            "samples": int(result["samples"]),
        }

    def resolve(
        self,
        env,
        state,
        blueprint: torch.Tensor,
        public_range: PublicRangeSnapshot,
    ) -> PluribusSearchResult:
        started = time.monotonic()
        hero = int(state.to_act)
        sanitized = sanitize_search_state(state, hero)
        candidates = generate_search_actions(env, state, blueprint)
        if not candidates:
            raise RuntimeError("search has no legal root candidates")
        compatible = [
            (combo, weight)
            for combo, weight in zip(public_range.combos, public_range.weights)
            if all(card not in set(state.hole[hero] + state.board) for card in combo)
        ]
        if not compatible:
            raise RuntimeError("public range has no root-compatible hands")
        total = sum(weight for _, weight in compatible)
        search_range = PublicRangeSnapshot(
            combos=tuple(combo for combo, _ in compatible),
            weights=tuple(weight / total for _, weight in compatible),
            effective_sample_size=(
                1.0
                / sum((weight / total) ** 2 for _, weight in compatible)
            ),
            updates=public_range.updates,
        )
        validation_reserve = min(
            2.00,
            max(0.75, 0.33 * self.time_budget_seconds),
        )
        worker_budget = max(
            0.05,
            self.time_budget_seconds - validation_reserve - 0.20,
        )
        strategy_sum = [0.0] * len(candidates)
        root_regrets = [0.0] * len(candidates)
        value_sum = [0.0] * len(candidates)
        square_sum = [0.0] * len(candidates)
        counts = [0] * len(candidates)
        iterations = 0
        terminal_rollouts = 0
        responders = 0
        native_responders = 0
        worker_regret_rows = []
        errors = []
        adaptive_depth = {
            0: min(self.depth_limit, 2),
            1: min(self.depth_limit, 2),
            2: min(max(self.depth_limit, 3), 4),
            3: min(max(self.depth_limit, 4), 5),
        }.get(int(state.street), self.depth_limit)

        def run_wave(
            wave_candidates: tuple[SearchAction, ...],
            budget: float,
            shared_regrets: Sequence[float] | None,
        ) -> list[tuple[dict, tuple[int, ...]]]:
            mapping = tuple(candidates.index(action) for action in wave_candidates)
            futures = [
                self.pool.submit(
                    _search_worker,
                    sanitized,
                    wave_candidates,
                    search_range,
                    self.rng.getrandbits(63),
                    budget,
                    self.iteration_cap_per_worker,
                    adaptive_depth,
                    self.native_backend,
                    shared_regrets,
                )
                for _ in range(self.workers)
            ]
            done, pending = wait(futures, timeout=budget + 0.15)
            for future in pending:
                future.cancel()
            rows = []
            for future in done:
                try:
                    rows.append((future.result(), mapping))
                except Exception as exc:
                    errors.append(f"{type(exc).__name__}: {exc}")
            return rows

        def merge_rows(rows: Sequence[tuple[dict, tuple[int, ...]]]) -> None:
            nonlocal responders, native_responders, iterations
            nonlocal terminal_rollouts
            for row, mapping in rows:
                responders += 1
                native_responders += int(bool(row.get("native_backend", False)))
                mapped_regrets = [0.0] * len(candidates)
                for local_index, global_index in enumerate(mapping):
                    mapped_regrets[global_index] = float(
                        row["root_regrets"][local_index]
                    )
                    strategy_sum[global_index] += float(
                        row["strategy_sum"][local_index]
                    )
                    root_regrets[global_index] += mapped_regrets[global_index]
                    value_sum[global_index] += float(
                        row["value_sum"][local_index]
                    )
                    square_sum[global_index] += float(
                        row["value_square_sum"][local_index]
                    )
                    counts[global_index] += int(
                        row["value_count"][local_index]
                    )
                worker_regret_rows.append(mapped_regrets)
                iterations += int(row["iterations"])
                terminal_rollouts += int(row["terminal_rollouts"])

        first_budget = max(0.05, worker_budget * 0.35)
        first_rows = run_wave(candidates, first_budget, None)
        merge_rows(first_rows)
        if not first_rows or not any(counts):
            detail = f" ({'; '.join(errors[:2])})" if errors else ""
            raise TimeoutError(f"no CFR worker completed a root update{detail}")

        # Successive elimination: retain every statistically plausible action,
        # and always retain the current leader of each strategic family.
        wave_means = [
            value_sum[index] / counts[index]
            if counts[index]
            else float("-inf")
            for index in range(len(candidates))
        ]
        wave_errors = []
        for index in range(len(candidates)):
            count = counts[index]
            if count > 1:
                variance = max(
                    0.0,
                    (
                        square_sum[index]
                        - value_sum[index] * value_sum[index] / count
                    )
                    / (count - 1),
                )
                wave_errors.append(math.sqrt(variance / count))
            else:
                wave_errors.append(float("inf"))
        family_leaders = {
            family: max(
                (
                    index
                    for index, action in enumerate(candidates)
                    if _search_action_family(action) == family
                ),
                key=wave_means.__getitem__,
            )
            for family in {"fold", "passive", "raise"}
            if any(
                _search_action_family(action) == family
                for action in candidates
            )
        }
        best_index = max(range(len(candidates)), key=wave_means.__getitem__)
        best_lower = (
            wave_means[best_index] - 1.96 * wave_errors[best_index]
        )
        active_indices = [
            index
            for index in range(len(candidates))
            if (
                index in family_leaders.values()
                or wave_means[index] + 1.96 * wave_errors[index] >= best_lower
            )
        ]
        active_candidates = tuple(candidates[index] for index in active_indices)
        shared_root = [
            root_regrets[index] / max(1, len(first_rows))
            for index in active_indices
        ]
        second_budget = max(0.05, worker_budget - first_budget)
        second_rows = run_wave(
            active_candidates,
            second_budget,
            shared_root,
        )
        merge_rows(second_rows)
        if not responders or not any(counts):
            detail = f" ({'; '.join(errors[:2])})" if errors else ""
            raise TimeoutError(f"no CFR worker completed a root update{detail}")

        statistics = []
        bb = float(state.big_blind)
        for index, candidate in enumerate(candidates):
            count = counts[index]
            mean = value_sum[index] / count if count else float("nan")
            if count > 1:
                variance = max(
                    0.0,
                    (
                        square_sum[index]
                        - value_sum[index] * value_sum[index] / count
                    )
                    / (count - 1),
                )
                standard_error = math.sqrt(variance / count)
            else:
                standard_error = float("inf")
            mean_bb = mean / bb
            se_bb = standard_error / bb
            statistics.append(
                (
                    mean_bb,
                    se_bb,
                    mean_bb - 1.96 * se_bb,
                    mean_bb + 1.96 * se_bb,
                    count,
                )
            )

        # Preserve hard live-play headroom for the postflop native rational-
        # caller veto and result assembly.
        validation_deadline = started + self.time_budget_seconds - 1.45
        validation_values, validation_native = self._validate_candidates(
            sanitized,
            candidates,
            search_range,
            deadline=validation_deadline,
        )
        validation_statistics = [
            self._mean_interval(values, bb)
            for values in validation_values
        ]
        validation_count = min(
            (len(values) for values in validation_values),
            default=0,
        )
        passive_indices = [
            index
            for index, candidate in enumerate(candidates)
            if candidate.kind == "abstract"
            and candidate.action is not None
            and ACTION_NAMES[int(candidate.action)] in {"check", "call", "fold"}
        ]
        reference_pool = passive_indices or list(range(len(candidates)))
        baseline_index = max(
            reference_pool,
            key=lambda index: (
                validation_statistics[index][0]
                if math.isfinite(validation_statistics[index][0])
                else statistics[index][0]
            ),
        )
        baseline_values = validation_values[baseline_index]
        paired_intervals = []
        for values in validation_values:
            paired = [
                value - baseline
                for value, baseline in zip(values, baseline_values)
            ]
            paired_intervals.append(self._mean_interval(paired, bb))

        # A root all-in receives an additional high-precision C++ runout
        # evaluation. This is cheap enough to veto catastrophic noisy shoves.
        all_in_validation = {}
        for index, candidate in enumerate(candidates):
            if (
                time.monotonic() < started + self.time_budget_seconds - 0.08
                and
                candidate.kind == "abstract"
                and candidate.action is not None
                and ACTION_NAMES[int(candidate.action)] == "all_in"
            ):
                result = self._validate_all_in(
                    env,
                    state,
                    candidate,
                    search_range,
                )
                if result is not None:
                    all_in_validation[index] = result

        terminal_call_validation = self._validate_terminal_call(
            env,
            state,
            candidates,
            search_range,
        )
        terminal_call_index = (
            terminal_call_validation[0]
            if terminal_call_validation is not None
            else None
        )
        terminal_call_result = (
            terminal_call_validation[1]
            if terminal_call_validation is not None
            else None
        )

        margin_bb = 0.25
        fold_indices = [
            index
            for index, candidate in enumerate(candidates)
            if candidate.kind == "abstract"
            and candidate.action is not None
            and ACTION_NAMES[int(candidate.action)] == "fold"
        ]
        risk_floor_bb = 0.0
        if fold_indices:
            fold_mean = validation_statistics[fold_indices[0]][0]
            if math.isfinite(fold_mean):
                risk_floor_bb = fold_mean
        dominated = []
        improved = []
        safety_pruned = []
        for index, paired in enumerate(paired_intervals):
            enough = validation_count >= 16
            is_dominated = enough and paired[3] < -margin_bb
            is_improved = enough and paired[2] > margin_bb
            native_all_in = all_in_validation.get(index)
            if native_all_in is not None:
                baseline_mean = validation_statistics[baseline_index][0]
                # The native validator applies a hand-by-hand rational-calling
                # floor. A faulty blueprint caller therefore cannot make a
                # catastrophic shove look profitable.
                if (
                    math.isfinite(baseline_mean)
                    and native_all_in["ci_high_bb"]
                    < baseline_mean - margin_bb
                ):
                    is_dominated = True
                    is_improved = False
            child = _apply_action(env, copy.deepcopy(state), candidates[index])
            committed = max(
                0,
                int(state.stacks[hero]) - int(child.stacks[hero]),
            )
            oversized_unproven = (
                committed >= 4 * max(1, int(state.pot))
                and committed * 2 >= int(state.stacks[hero])
                and (
                    native_all_in is None
                    or native_all_in["ci_low_bb"]
                    <= risk_floor_bb + margin_bb
                )
            )
            terminal_call_unproven = (
                terminal_call_index == index
                and terminal_call_result is not None
                and terminal_call_result["worst_ci_low_bb"]
                <= terminal_call_result["fold_payoff_bb"] + margin_bb
            )
            if terminal_call_unproven:
                is_dominated = True
                is_improved = False
            dominated.append(is_dominated)
            improved.append(is_improved)
            safety_pruned.append(
                oversized_unproven or terminal_call_unproven
            )
        dominated[baseline_index] = False
        if baseline_index != terminal_call_index:
            safety_pruned[baseline_index] = False
        allowed = [
            not dominated[index] and not safety_pruned[index]
            for index in range(len(candidates))
        ]
        if not any(allowed):
            emergency_index = (
                fold_indices[0]
                if fold_indices
                else baseline_index
            )
            allowed[emergency_index] = True

        root_samples = min(counts) if counts else 0
        search_ready = (
            root_samples >= 64
            and responders >= max(1, (self.workers + 1) // 2)
        )
        validation_ready = validation_count >= 16
        value_scores = [
            (
                statistics[index][0] - 0.25 * statistics[index][1]
                if math.isfinite(statistics[index][0])
                and math.isfinite(statistics[index][1])
                else float("-inf")
            )
            for index in range(len(candidates))
        ]
        family_ids = [
            _search_action_family_id(candidate)
            for candidate in candidates
        ]
        if native_hierarchical_regret_match_root is not None:
            hierarchical = native_hierarchical_regret_match_root(
                root_regrets,
                allowed,
                value_scores,
                family_ids,
            )
            strategy = [
                float(value)
                for value in hierarchical["action_strategy"]
            ]
            family_strategy = [
                float(value)
                for value in hierarchical["family_strategy"]
            ]
        else:
            positive = [
                max(0.0, root_regrets[index]) if allowed[index] else 0.0
                for index in range(len(candidates))
            ]
            total = sum(positive)
            if total > 1e-12:
                strategy = [value / total for value in positive]
            else:
                best = max(
                    (
                        index
                        for index in range(len(candidates))
                        if allowed[index]
                    ),
                    key=value_scores.__getitem__,
                )
                strategy = [
                    1.0 if index == best else 0.0
                    for index in range(len(candidates))
                ]
            family_strategy = [
                sum(
                    strategy[index]
                    for index, family_id in enumerate(family_ids)
                    if family_id == target
                )
                for target in range(3)
            ]
        strategy_total = sum(strategy)
        if strategy_total <= 1e-12:
            raise RuntimeError("search produced no admissible root strategy")
        strategy = [value / strategy_total for value in strategy]
        ordered_probabilities = sorted(family_strategy, reverse=True)
        strategy_gap = (
            ordered_probabilities[0] - ordered_probabilities[1]
            if len(ordered_probabilities) > 1
            else 1.0
        )
        chosen_family = max(range(3), key=family_strategy.__getitem__)
        top_probability = max(
            (
                strategy[index]
                for index in range(len(candidates))
                if family_ids[index] == chosen_family and allowed[index]
            ),
            default=0.0,
        )
        near_top = [
            index
            for index, probability in enumerate(strategy)
            if (
                allowed[index]
                and family_ids[index] == chosen_family
                and probability >= top_probability - 0.02
            )
        ]

        def commitment_key(index: int) -> tuple:
            child = _apply_action(
                env,
                copy.deepcopy(state),
                candidates[index],
            )
            committed = max(
                0,
                int(state.stacks[hero]) - int(child.stacks[hero]),
            )
            family = (
                _action_family(int(candidates[index].action))
                if candidates[index].kind == "abstract"
                and candidates[index].action is not None
                else "raise"
            )
            aggression = {
                "check": 0,
                "fold": 1,
                "call": 2,
                "raise": 3,
            }.get(family, 3)
            return committed, aggression, -strategy[index]

        worker_choices = []
        for regrets in worker_regret_rows:
            family_positive = [
                sum(
                    max(0.0, regrets[index])
                    for index in range(len(candidates))
                    if allowed[index] and family_ids[index] == family
                )
                for family in range(3)
            ]
            if sum(family_positive) > 1e-12:
                worker_choices.append(
                    max(range(3), key=family_positive.__getitem__)
                )
        worker_agreement = 0.0
        if worker_choices:
            worker_agreement = max(
                worker_choices.count(index)
                for index in set(worker_choices)
            ) / len(worker_choices)
        converged = bool(
            search_ready
            and validation_ready
            and worker_agreement >= 0.60
            and strategy_gap >= 0.02
        )
        if converged:
            chosen_index = min(near_top, key=commitment_key)
        else:
            family_pool = [
                index
                for index in range(len(candidates))
                if allowed[index] and family_ids[index] == chosen_family
            ]
            conservative_pool = family_pool or [
                index
                for index in range(len(candidates))
                if allowed[index]
            ]

            def conservative_score(index: int) -> float:
                root_mean = statistics[index][0]
                validation_mean = validation_statistics[index][0]
                if (
                    terminal_call_index == index
                    and terminal_call_result is not None
                ):
                    validation_mean = terminal_call_result["worst_mean_bb"]
                values = [
                    value
                    for value in (root_mean, validation_mean)
                    if math.isfinite(value)
                ]
                return min(values) if values else float("-inf")

            chosen_index = max(
                conservative_pool,
                key=lambda index: (
                    conservative_score(index),
                    tuple(-value for value in commitment_key(index)),
                ),
            )
            strategy = [
                1.0 if index == chosen_index else 0.0
                for index in range(len(candidates))
            ]
        reason_parts = [
            "search-authoritative synchronized family-first regret matching",
            f"adaptive depth {adaptive_depth}",
            f"successive set {len(active_candidates)}/{len(candidates)}",
        ]
        if not search_ready:
            reason_parts.append(f"limited root samples {root_samples}/64")
        if not validation_ready:
            reason_parts.append(
                f"limited paired validation {validation_count}/16"
            )
        if worker_agreement < 0.60:
            reason_parts.append(
                f"worker agreement {worker_agreement:.0%}/60%"
            )
        if strategy_gap < 0.02:
            reason_parts.append(
                f"root gap {strategy_gap:.1%}/2%; conservative tie-break"
            )
        if not converged:
            reason_parts.append(
                "unresolved family selected by search EV; no policy fallback"
            )
        if terminal_call_result is not None:
            reason_parts.append(
                "terminal-call worst range "
                f"{terminal_call_result['worst_name']} "
                f"{terminal_call_result['worst_mean_bb']:+.2f} BB"
            )
        if any(safety_pruned):
            reason_parts.append("robust large-bet safety veto applied")
        reason = "; ".join(reason_parts)

        estimates = []
        for index, candidate in enumerate(candidates):
            mean_bb, se_bb, ci_low, ci_high, count = statistics[index]
            validation_mean, _, validation_low, validation_high = (
                validation_statistics[index]
            )
            native_all_in = all_in_validation.get(index)
            if native_all_in is not None:
                validation_mean = native_all_in["mean_bb"]
                validation_low = native_all_in["ci_low_bb"]
                validation_high = native_all_in["ci_high_bb"]
                candidate_validation_samples = native_all_in["samples"]
            elif (
                terminal_call_index == index
                and terminal_call_result is not None
            ):
                validation_mean = terminal_call_result["worst_mean_bb"]
                validation_low = terminal_call_result["worst_ci_low_bb"]
                validation_high = terminal_call_result["worst_ci_high_bb"]
                candidate_validation_samples = terminal_call_result["samples"]
            else:
                candidate_validation_samples = len(validation_values[index])
            estimates.append(
                CandidateEstimate(
                    action=candidate,
                    expected_final_payoff_bb=mean_bb,
                    standard_error_bb=se_bb,
                    ci95_low_bb=ci_low,
                    ci95_high_bb=ci_high,
                    samples=count,
                    strategy_probability=strategy[index],
                    validation_ev_bb=validation_mean,
                    validation_ci95_low_bb=validation_low,
                    validation_ci95_high_bb=validation_high,
                    validation_samples=candidate_validation_samples,
                    statistically_dominated=dominated[index],
                    safety_pruned=safety_pruned[index],
                )
            )
        return PluribusSearchResult(
            choice=candidates[chosen_index],
            candidates=tuple(estimates),
            elapsed_ms=1000.0 * (time.monotonic() - started),
            cfr_iterations=iterations,
            terminal_rollouts=terminal_rollouts,
            workers_responded=responders,
            range_combos=len(search_range.combos),
            range_effective_sample_size=search_range.effective_sample_size,
            range_updates=search_range.updates,
            native_backend=(
                validation_native
                and native_responders == responders
                and responders > 0
            ),
            converged=converged,
            used_blueprint_fallback=False,
            convergence_reason=reason,
            validation_samples=validation_count,
            worker_agreement=worker_agreement,
            strategy_gap=strategy_gap,
        )


__all__ = [
    "BlueprintPublicRange",
    "CONTINUATION_NAMES",
    "CandidateEstimate",
    "MultiprocessPluribusSearch",
    "PluribusSearchResult",
    "PublicRangeSnapshot",
    "SearchAction",
    "generate_search_actions",
    "observed_action_likelihoods",
    "recommended_search_workers",
    "sanitize_search_state",
]
