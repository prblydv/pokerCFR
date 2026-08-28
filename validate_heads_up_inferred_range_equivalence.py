"""Compare current inferred ranges with literal scalar/Python references."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import torch

from heads_up_engine import ACTION_NAMES, HeadsUpHoldemEngine
from heads_up_models import (
    build_action_descriptors,
    encode_information_state,
    masked_softmax,
)
from heads_up_pluribus_search import (
    BlueprintPublicRange,
    PublicRangeSnapshot,
    observed_action_likelihoods,
)
from heads_up_root_policy_search import robust_inferred_range
from play_heads_up_gui import HeadsUpSnapshotPolicy


class PurePythonPublicRange(BlueprintPublicRange):
    """Literal pre-native multiply, floor, sum, and normalize update."""

    def condition(self, likelihoods) -> None:
        if len(likelihoods) != len(self.combos):
            raise ValueError("one action likelihood is required per range hand")
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


@torch.inference_mode()
def _scalar_probabilities_for_holes(
    policy,
    env,
    state,
    actor: int,
    hands,
) -> torch.Tensor:
    actor = int(actor)
    legal = tuple(int(action) for action in env.legal_actions(state))
    descriptors = build_action_descriptors(env, state)
    mask = torch.zeros(len(ACTION_NAMES), dtype=torch.float32)
    mask[list(legal)] = 1.0
    rows = []
    for hand in hands:
        hypothetical = copy.copy(state)
        hypothetical.hole = [list(cards) for cards in state.hole]
        hypothetical.hole[actor] = [int(hand[0]), int(hand[1])]
        observation = encode_information_state(
            hypothetical,
            actor,
            legal,
            env.bb,
            int(policy.snapshot.metadata["max_history"]),
            action_descriptors=descriptors,
        )
        logits = policy.snapshot.policy_nets[actor](
            observation.unsqueeze(0).to(policy.device)
        )
        rows.append(
            masked_softmax(
                logits,
                mask.to(policy.device).unsqueeze(0),
            )[0].cpu()
        )
    return torch.stack(rows)


def _choose_human_action(env, state) -> int:
    legal = [int(action) for action in env.legal_actions(state)]
    by_name = {ACTION_NAMES[action]: action for action in legal}
    preferences = {
        0: ("three_quarter_pot", "min_raise", "call"),
        1: ("half_pot", "third_pot", "check"),
        2: ("check", "third_pot", "half_pot"),
        3: ("overbet", "pot", "three_quarter_pot", "check"),
    }
    for name in preferences[int(state.street)]:
        if name in by_name:
            return by_name[name]
    return legal[0]


def _choose_other_action(env, state) -> int:
    legal = [int(action) for action in env.legal_actions(state)]
    by_name = {ACTION_NAMES[action]: action for action in legal}
    if "call" in by_name:
        return by_name["call"]
    if "check" in by_name:
        return by_name["check"]
    return legal[0]


def _snapshot(range_object) -> PublicRangeSnapshot:
    return range_object.snapshot()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("policy", type=Path)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--scalar-combos", type=int, default=96)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.scalar_combos <= 0:
        parser.error("--scalar-combos must be positive")

    policy = HeadsUpSnapshotPolicy(
        args.policy,
        device="auto",
        seed=args.seed,
    )
    env = HeadsUpHoldemEngine(200, 1, 2, seed=args.seed)
    state = env.new_hand(button=0)
    human_seat = 0
    observer_seat = 1
    known = [*state.hole[observer_seat], *state.board]
    current = BlueprintPublicRange()
    python_reference = PurePythonPublicRange()
    scalar_policy_reference = PurePythonPublicRange()
    current.reset(known)
    python_reference.reset(known)
    scalar_policy_reference.reset(known)

    rows = []
    maximum_scalar_batch_probability_difference = 0.0
    maximum_likelihood_difference = 0.0
    maximum_posterior_weight_difference = 0.0
    maximum_robust_weight_difference = 0.0
    maximum_ess_difference = 0.0
    maximum_scalar_posterior_weight_difference = 0.0
    maximum_scalar_robust_weight_difference = 0.0
    maximum_scalar_ess_difference = 0.0
    actions = 0
    while not state.terminal and actions < 48:
        before = state
        actor = int(before.to_act)
        if actor == human_seat:
            current.filter_known(
                [*before.hole[observer_seat], *before.board]
            )
            python_reference.filter_known(
                [*before.hole[observer_seat], *before.board]
            )
            scalar_policy_reference.filter_known(
                [*before.hole[observer_seat], *before.board]
            )
            if (
                current.combos != python_reference.combos
                or current.combos != scalar_policy_reference.combos
            ):
                raise RuntimeError("range combo order diverged")

            batched = policy.probabilities_for_holes(
                env,
                before,
                actor,
                current.combos,
            )
            stride = max(1, len(current.combos) // args.scalar_combos)
            indices = list(
                range(0, len(current.combos), stride)
            )[: args.scalar_combos]
            scalar_hands = [current.combos[index] for index in indices]
            scalar = _scalar_probabilities_for_holes(
                policy,
                env,
                before,
                actor,
                scalar_hands,
            )
            selected_batched = batched[indices]
            probability_difference = float(
                torch.max(torch.abs(scalar - selected_batched)).item()
            )
            maximum_scalar_batch_probability_difference = max(
                maximum_scalar_batch_probability_difference,
                probability_difference,
            )

            action = _choose_human_action(env, before)
            after = env.step(before, action)
            event = after.history[-1]
            batched_likelihoods = observed_action_likelihoods(
                env,
                before,
                batched,
                kind=str(event.kind),
                raise_to=(
                    int(event.raise_to)
                    if str(event.kind) in {"bet", "raise"}
                    else None
                ),
            )
            scalar_likelihoods = observed_action_likelihoods(
                env,
                before,
                scalar,
                kind=str(event.kind),
                raise_to=(
                    int(event.raise_to)
                    if str(event.kind) in {"bet", "raise"}
                    else None
                ),
            )
            likelihood_difference = max(
                abs(
                    float(scalar_likelihoods[position])
                    - float(batched_likelihoods[index])
                )
                for position, index in enumerate(indices)
            )
            maximum_likelihood_difference = max(
                maximum_likelihood_difference,
                likelihood_difference,
            )

            current.condition(batched_likelihoods)
            python_reference.condition(batched_likelihoods)
            full_scalar_range_checked = len(indices) == len(current.combos)
            if full_scalar_range_checked:
                scalar_policy_reference.condition(scalar_likelihoods)
            else:
                scalar_policy_reference.condition(batched_likelihoods)
            current_snapshot = _snapshot(current)
            python_snapshot = _snapshot(python_reference)
            posterior_difference = max(
                abs(float(one) - float(two))
                for one, two in zip(
                    current_snapshot.weights,
                    python_snapshot.weights,
                )
            )
            maximum_posterior_weight_difference = max(
                maximum_posterior_weight_difference,
                posterior_difference,
            )
            ess_difference = abs(
                float(current_snapshot.effective_sample_size)
                - float(python_snapshot.effective_sample_size)
            )
            maximum_ess_difference = max(
                maximum_ess_difference,
                ess_difference,
            )
            current_robust = robust_inferred_range(
                current_snapshot,
                temperature=0.65,
                uniform_contamination=0.25,
            )
            python_robust = robust_inferred_range(
                python_snapshot,
                temperature=0.65,
                uniform_contamination=0.25,
            )
            robust_difference = max(
                abs(float(one) - float(two))
                for one, two in zip(
                    current_robust.weights,
                    python_robust.weights,
                )
            )
            maximum_robust_weight_difference = max(
                maximum_robust_weight_difference,
                robust_difference,
            )
            scalar_snapshot = _snapshot(scalar_policy_reference)
            scalar_posterior_difference = max(
                abs(float(one) - float(two))
                for one, two in zip(
                    current_snapshot.weights,
                    scalar_snapshot.weights,
                )
            )
            scalar_ess_difference = abs(
                float(current_snapshot.effective_sample_size)
                - float(scalar_snapshot.effective_sample_size)
            )
            scalar_robust = robust_inferred_range(
                scalar_snapshot,
                temperature=0.65,
                uniform_contamination=0.25,
            )
            scalar_robust_difference = max(
                abs(float(one) - float(two))
                for one, two in zip(
                    current_robust.weights,
                    scalar_robust.weights,
                )
            )
            if full_scalar_range_checked:
                maximum_scalar_posterior_weight_difference = max(
                    maximum_scalar_posterior_weight_difference,
                    scalar_posterior_difference,
                )
                maximum_scalar_ess_difference = max(
                    maximum_scalar_ess_difference,
                    scalar_ess_difference,
                )
                maximum_scalar_robust_weight_difference = max(
                    maximum_scalar_robust_weight_difference,
                    scalar_robust_difference,
                )
            actual_combo = tuple(
                sorted(int(card) for card in before.hole[human_seat])
            )
            actual_index = current_robust.combos.index(actual_combo)
            rows.append(
                {
                    "update": current.updates,
                    "street": int(before.street),
                    "history_events": len(before.history),
                    "observed_kind": str(event.kind),
                    "observed_raise_to": int(event.raise_to),
                    "range_combos": len(current.combos),
                    "scalar_combos_checked": len(indices),
                    "max_scalar_vs_batch_probability_difference": (
                        probability_difference
                    ),
                    "max_scalar_vs_batch_likelihood_difference": (
                        likelihood_difference
                    ),
                    "max_native_vs_python_posterior_difference": (
                        posterior_difference
                    ),
                    "native_vs_python_ess_difference": ess_difference,
                    "max_robust_range_weight_difference": robust_difference,
                    "full_scalar_range_checked": full_scalar_range_checked,
                    "max_scalar_policy_posterior_difference": (
                        scalar_posterior_difference
                    ),
                    "scalar_policy_ess_difference": scalar_ess_difference,
                    "max_scalar_policy_robust_weight_difference": (
                        scalar_robust_difference
                    ),
                    "actual_combo_probability": float(
                        current_robust.weights[actual_index]
                    ),
                }
            )
            state = after
        else:
            state = env.step(before, _choose_other_action(env, before))
        actions += 1

    tolerance = 1e-6
    equivalent = (
        len(rows) >= 4
        and maximum_scalar_batch_probability_difference <= tolerance
        and maximum_likelihood_difference <= tolerance
        and maximum_posterior_weight_difference <= tolerance
        and maximum_robust_weight_difference <= tolerance
        and maximum_ess_difference <= tolerance
    )
    report = {
        "policy": str(args.policy.resolve()),
        "policy_iteration": policy.iteration,
        "policy_sha256": policy.sha256,
        "device": str(policy.device),
        "updates": len(rows),
        "streets_observed": [int(row["street"]) for row in rows],
        "maximum_differences": {
            "scalar_vs_batch_policy_probability": (
                maximum_scalar_batch_probability_difference
            ),
            "scalar_vs_batch_action_likelihood": (
                maximum_likelihood_difference
            ),
            "native_vs_python_posterior_weight": (
                maximum_posterior_weight_difference
            ),
            "native_vs_python_robust_weight": (
                maximum_robust_weight_difference
            ),
            "native_vs_python_effective_sample_size": (
                maximum_ess_difference
            ),
            "batched_vs_scalar_policy_posterior_weight": (
                maximum_scalar_posterior_weight_difference
            ),
            "batched_vs_scalar_policy_robust_weight": (
                maximum_scalar_robust_weight_difference
            ),
            "batched_vs_scalar_policy_effective_sample_size": (
                maximum_scalar_ess_difference
            ),
        },
        "tolerance": tolerance,
        "equivalent_within_tolerance": equivalent,
        "updates_detail": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(report, indent=2))
    return 0 if equivalent else 1


if __name__ == "__main__":
    raise SystemExit(main())
