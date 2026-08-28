"""Heads-up GUI for human-vs-policy play and exact manual engine testing.

With ``--policy``, one seat is controlled by a deployable two-network policy
snapshot. Repeated ``--policy-secondary`` options form an equal-weight action
probability ensemble. Without ``--policy``, both seats are controlled manually.
"""

from __future__ import annotations

import argparse
import copy
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tkinter as tk
from tkinter import messagebox, ttk
import uuid

import torch

from heads_up_engine import (
    ACTION_SCHEMA_VERSION,
    ACTION_NAMES,
    ENGINE_SCHEMA_VERSION,
    NUM_ACTIONS,
    STREET_NAMES,
    HeadsUpHoldemEnv,
    card_to_string,
)
from heads_up_models import (
    build_action_descriptors,
    encode_information_state,
    masked_softmax,
)
from heads_up_ranges import (
    OPPONENT_COMBOS,
    masked_range_probabilities,
    valid_combo_mask_from_encoded,
)
from heads_up_production import PolicySnapshot, load_policy_snapshot
from heads_up_robust_search import (
    RobustHeadsUpSearch,
    action_noise_likelihoods,
)
from heads_up_pluribus_search import (
    BlueprintPublicRange,
    MultiprocessPluribusSearch,
    PluribusSearchResult,
    PublicRangeSnapshot,
    observed_action_likelihoods,
    recommended_search_workers,
)
from heads_up_root_policy_search import (
    HeadsUpRootPolicySearch,
    robust_inferred_range,
)


SEAT_NAMES = ("Player 0", "Player 1")
EPSILON = 1e-9
DEFAULT_RESULTS_LOG = Path("artifacts/heads_up_gui_results/hands.jsonl")
RANK_SYMBOLS = "23456789TJQKA"
POLICY_SHIFT_POSITIVE_TAG = "policy_shift_positive"
POLICY_SHIFT_NEGATIVE_TAG = "policy_shift_negative"
POLICY_SHIFT_NEUTRAL_TAG = "policy_shift_neutral"
HUD_INITIAL_HANDS = 1_700
HUD_HISTORY_VERSION = 1


def format_policy_shift(
    strategy_probability: float,
    blueprint_probability: float,
) -> tuple[str, str]:
    """Return percentage-point policy shift text and its GUI colour tag."""
    shift_pp = 100.0 * (
        float(strategy_probability) - float(blueprint_probability)
    )
    if shift_pp > 0.05:
        tag = POLICY_SHIFT_POSITIVE_TAG
    elif shift_pp < -0.05:
        tag = POLICY_SHIFT_NEGATIVE_TAG
    else:
        shift_pp = 0.0
        tag = POLICY_SHIFT_NEUTRAL_TAG
    return f"{shift_pp:+5.1f}pp", tag


def calculate_player_hud(hands, seat: int) -> dict:
    """Calculate session VPIP, HU button-steal frequency, and postflop AF."""

    seat = int(seat)
    total_hands = len(hands)
    vpip_hands = 0
    steal_attempts = 0
    steal_opportunities = 0
    postflop_aggression = 0
    postflop_calls = 0

    for hand in hands:
        button = int(hand["button"])
        history = tuple(hand["history"])
        preflop = [
            event
            for event in history
            if int(_event_attr(event, "street", default=-1)) == 0
        ]
        if any(
            int(_event_attr(event, "player", default=-1)) == seat
            and str(_event_attr(event, "kind", default=""))
            in {"call", "bet", "raise"}
            and int(_event_attr(event, "amount", default=0)) > 0
            for event in preflop
        ):
            vpip_hands += 1

        if button == seat:
            steal_opportunities += 1
            first_button_action = next(
                (
                    event
                    for event in preflop
                    if int(_event_attr(event, "player", default=-1)) == seat
                ),
                None,
            )
            if (
                first_button_action is not None
                and str(_event_attr(first_button_action, "kind", default=""))
                in {"bet", "raise"}
            ):
                steal_attempts += 1

        for event in history:
            if (
                int(_event_attr(event, "player", default=-1)) != seat
                or int(_event_attr(event, "street", default=-1)) <= 0
            ):
                continue
            kind = str(_event_attr(event, "kind", default=""))
            if kind in {"bet", "raise"}:
                postflop_aggression += 1
            elif kind == "call":
                postflop_calls += 1

    return {
        "hands": total_hands,
        "vpip_hands": vpip_hands,
        "vpip_pct": 100.0 * vpip_hands / total_hands if total_hands else 0.0,
        "steal_attempts": steal_attempts,
        "steal_opportunities": steal_opportunities,
        "ats_pct": (
            100.0 * steal_attempts / steal_opportunities
            if steal_opportunities
            else 0.0
        ),
        "postflop_aggression": postflop_aggression,
        "postflop_calls": postflop_calls,
        "af": (
            float(postflop_aggression) / postflop_calls
            if postflop_calls
            else math.inf if postflop_aggression else 0.0
        ),
    }


def canonical_hud_hand(button: int, history, human_seat: int) -> dict:
    """Normalize physical seats to persistent human=0 and policy=1 roles."""

    human_seat = int(human_seat)
    if human_seat not in (0, 1):
        raise ValueError("human_seat must be 0 or 1")
    normalized_history = []
    for event in history:
        physical_seat = int(_event_attr(event, "player", default=-1))
        if physical_seat not in (0, 1):
            continue
        normalized_history.append(
            {
                "player": 0 if physical_seat == human_seat else 1,
                "street": int(_event_attr(event, "street", default=-1)),
                "kind": str(_event_attr(event, "kind", default="")),
                "amount": int(_event_attr(event, "amount", default=0)),
            }
        )
    return {
        "button": 0 if int(button) == human_seat else 1,
        "history": tuple(normalized_history),
    }


def persisted_hud_hands(records) -> list[dict]:
    """Restore the fixed initial sample plus every subsequently recorded hand."""

    records = list(records)
    first_persistent_index = next(
        (
            index
            for index, record in enumerate(records)
            if int(record.get("hud_history_version", 0))
            == HUD_HISTORY_VERSION
        ),
        None,
    )
    if first_persistent_index is None:
        selected_records = records[-HUD_INITIAL_HANDS:]
    else:
        selected_records = records[
            max(0, first_persistent_index - HUD_INITIAL_HANDS) :
        ]
    hands = []
    for record in selected_records:
        history = record.get("action_sequence")
        if not isinstance(history, list):
            history = []
            for line in record.get("public_history", ()):
                match = re.match(
                    r"^\s*(preflop|flop|turn|river)\s+P([01])\s+"
                    r"(fold|check|call|bet|raise)\s+\+\s+([0-9]+)",
                    str(line),
                )
                if match is None:
                    continue
                history.append(
                    {
                        "street": STREET_NAMES.index(match.group(1)),
                        "player": int(match.group(2)),
                        "kind": match.group(3),
                        "amount": int(match.group(4)),
                    }
                )
            if not history and record.get("public_history"):
                continue
        try:
            hands.append(
                canonical_hud_hand(
                    int(record["button"]),
                    history,
                    int(record["human_seat"]),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return hands


def format_hud_af(value: float) -> str:
    return "\u221e" if math.isinf(float(value)) else f"{float(value):.2f}"


def _range_class_label(combo: tuple[int, int]) -> str:
    first, second = (int(combo[0]), int(combo[1]))
    first_rank, second_rank = first % 13, second % 13
    high, low = sorted((first_rank, second_rank), reverse=True)
    if high == low:
        return RANK_SYMBOLS[high] * 2
    suited = first // 13 == second // 13
    return (
        RANK_SYMBOLS[high]
        + RANK_SYMBOLS[low]
        + ("s" if suited else "o")
    )


def summarize_public_range(
    snapshot,
    *,
    top_n: int = 8,
    actual_hole=None,
) -> dict:
    classes: dict[str, float] = {}
    for combo, weight in zip(snapshot.combos, snapshot.weights):
        label = _range_class_label(combo)
        classes[label] = classes.get(label, 0.0) + float(weight)
    top = sorted(classes.items(), key=lambda row: row[1], reverse=True)[:top_n]
    result = {
        "effective_sample_size": float(snapshot.effective_sample_size),
        "updates": int(snapshot.updates),
        "top_classes": top,
        "class_probabilities": dict(sorted(classes.items())),
    }
    if actual_hole is not None:
        actual_combo = tuple(sorted(int(card) for card in actual_hole))
        combo_weights = {
            tuple(sorted(int(card) for card in combo)): float(weight)
            for combo, weight in zip(snapshot.combos, snapshot.weights)
        }
        actual_probability = float(combo_weights.get(actual_combo, 0.0))
        actual_class = _range_class_label(actual_combo)
        result.update(
            {
                "actual_human_combo_card_ids": list(actual_combo),
                "actual_human_combo": [
                    card_to_string(card) for card in actual_combo
                ],
                "actual_human_combo_probability": actual_probability,
                "actual_human_combo_rank": (
                    1
                    + sum(
                        weight > actual_probability + 1e-15
                        for weight in combo_weights.values()
                    )
                    if actual_combo in combo_weights
                    else None
                ),
                "actual_human_class": actual_class,
                "actual_human_class_probability": float(
                    classes.get(actual_class, 0.0)
                ),
                "actual_human_class_rank": (
                    1
                    + sum(
                        probability
                        > float(classes.get(actual_class, 0.0)) + 1e-15
                        for probability in classes.values()
                    )
                    if actual_class in classes
                    else None
                ),
            }
        )
    return result


def range_probability_color(probability: float, maximum: float) -> str:
    """Map one posterior class probability to a readable dark-teal-gold scale."""
    ratio = 0.0 if maximum <= 0.0 else max(0.0, min(1.0, probability / maximum))
    ratio = math.sqrt(ratio)
    low = (14, 21, 27)
    middle = (18, 126, 116)
    high = (244, 208, 63)
    if ratio <= 0.5:
        local = ratio * 2.0
        start, end = low, middle
    else:
        local = (ratio - 0.5) * 2.0
        start, end = middle, high
    values = tuple(
        round(start[index] + local * (end[index] - start[index]))
        for index in range(3)
    )
    return "#" + "".join(f"{value:02x}" for value in values)


class HeadsUpSnapshotPolicy:
    """Legal-action adapter for one deployable two-seat policy snapshot."""

    def __init__(
        self,
        path: str | Path,
        *,
        mode: str = "sample",
        device: str | torch.device = "auto",
        native_batch_encoding: bool = True,
        seed: int | None = None,
    ) -> None:
        if mode not in {"sample", "argmax"}:
            raise ValueError("policy mode must be 'sample' or 'argmax'")
        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"policy snapshot not found: {self.path}")
        if str(device) == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.native_batch_encoding = bool(native_batch_encoding)
        self.snapshot: PolicySnapshot = load_policy_snapshot(
            self.path,
            device=self.device,
        )
        with self.path.open("rb") as stream:
            self.sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        self.mode = mode
        self.generator = torch.Generator(device="cpu")
        if seed is None:
            self.generator.seed()
        else:
            self.generator.manual_seed(int(seed))

    @property
    def iteration(self) -> int:
        return self.snapshot.iteration

    @property
    def iteration_label(self) -> str:
        return str(self.iteration)

    @torch.inference_mode()
    def probabilities(self, env: HeadsUpHoldemEnv, state) -> torch.Tensor:
        return self.probabilities_batch(env, [state])[0]

    @torch.inference_mode()
    def probabilities_batch(
        self,
        env: HeadsUpHoldemEnv,
        states,
    ) -> torch.Tensor:
        probabilities, _ = self._probabilities_device_batch(env, states)
        return probabilities.cpu()

    @torch.inference_mode()
    def _probabilities_device_batch(
        self,
        env: HeadsUpHoldemEnv,
        states,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not states:
            empty = torch.empty(
                (0, NUM_ACTIONS), dtype=torch.float32, device=self.device
            )
            return empty, empty
        actor = int(states[0].to_act)
        if any(
            state.terminal
            or state.to_act is None
            or int(state.to_act) != actor
            for state in states
        ):
            raise ValueError(
                "batched policy states must be live and share one actor"
            )
        native_backend = bool(getattr(env, "native_backend", False))
        if native_backend and getattr(self, "native_batch_encoding", True):
            from heads_up_native import encode_information_states_native

            encoded_array, mask_array = encode_information_states_native(
                env,
                states,
                int(self.snapshot.metadata["max_history"]),
            )
            encoded = torch.from_numpy(encoded_array).to(self.device)
            legal_masks = torch.from_numpy(mask_array).to(self.device)
        else:
            observations = []
            masks = []
            native_encoder = None
            if native_backend:
                from heads_up_native import encode_information_state_native

                native_encoder = encode_information_state_native
            for state in states:
                legal = tuple(
                    int(action) for action in env.legal_actions(state)
                )
                descriptors = build_action_descriptors(env, state)
                if native_encoder is None:
                    observation = encode_information_state(
                        state,
                        actor,
                        legal,
                        env.bb,
                        int(self.snapshot.metadata["max_history"]),
                        action_descriptors=descriptors,
                    )
                else:
                    observation = torch.from_numpy(
                        native_encoder(
                            state,
                            actor,
                            legal,
                            env.bb,
                            int(self.snapshot.metadata["max_history"]),
                            action_descriptors=descriptors,
                        )
                    )
                observations.append(observation)
                mask = torch.zeros(NUM_ACTIONS, dtype=torch.float32)
                mask[list(legal)] = 1.0
                masks.append(mask)
            encoded = torch.stack(observations).to(self.device)
            legal_masks = torch.stack(masks).to(self.device)
        logits = self.snapshot.policy_nets[actor](encoded)
        return masked_softmax(logits, legal_masks), legal_masks

    @torch.inference_mode()
    def sample_actions_batch(
        self,
        env: HeadsUpHoldemEnv,
        states,
        thresholds,
    ) -> list[int]:
        """Sample full-width masked policies using supplied paired uniforms."""
        if len(states) != len(thresholds):
            raise ValueError("states and thresholds must have identical lengths")
        if not states:
            return []
        probabilities, legal_masks = self._probabilities_device_batch(
            env, states
        )
        uniform = torch.as_tensor(
            thresholds,
            dtype=torch.float64,
            device=self.device,
        ).unsqueeze(1)
        cumulative = torch.cumsum(probabilities.to(torch.float64), dim=1)
        hits = cumulative + 1e-12 >= uniform
        first_hit = torch.argmax(hits.to(torch.int8), dim=1)
        action_indices = torch.arange(
            NUM_ACTIONS, device=self.device, dtype=torch.int64
        ).unsqueeze(0)
        last_legal = torch.argmax(
            action_indices * legal_masks.to(torch.int64),
            dim=1,
        )
        selected = torch.where(hits.any(dim=1), first_hit, last_legal)
        return [int(action) for action in selected.cpu().tolist()]

    @torch.inference_mode()
    def _legacy_single_probabilities(
        self, env: HeadsUpHoldemEnv, state
    ) -> torch.Tensor:
        if state.terminal or state.to_act is None:
            raise ValueError("policy requires a live decision state")
        actor = int(state.to_act)
        legal = tuple(int(action) for action in env.legal_actions(state))
        encoded = encode_information_state(
            state,
            actor,
            legal,
            env.bb,
            int(self.snapshot.metadata["max_history"]),
            action_descriptors=build_action_descriptors(env, state),
        )
        mask = torch.zeros(NUM_ACTIONS, dtype=torch.float32)
        mask[list(legal)] = 1.0
        logits = self.snapshot.policy_nets[actor](
            encoded.unsqueeze(0).to(self.device)
        )[0]
        return masked_softmax(logits, mask.to(self.device)).cpu()

    def choose_action(
        self,
        env: HeadsUpHoldemEnv,
        state,
    ) -> tuple[int, torch.Tensor]:
        probabilities = self.probabilities(env, state)
        if self.mode == "argmax":
            action = int(torch.argmax(probabilities).item())
        else:
            action = int(
                torch.multinomial(
                    probabilities,
                    1,
                    generator=self.generator,
                ).item()
            )
        if action not in env.legal_actions(state):
            raise RuntimeError("policy selected an illegal action")
        return action, probabilities

    @torch.inference_mode()
    def probabilities_for_holes(
        self,
        env: HeadsUpHoldemEnv,
        state,
        actor: int,
        hands,
        *,
        batch_size: int = 512,
    ) -> torch.Tensor:
        """Evaluate one public state for hypothetical acting-player hands."""

        actor = int(actor)
        if int(state.to_act) != actor:
            raise ValueError("range likelihood actor must be the live actor")
        legal = tuple(int(action) for action in env.legal_actions(state))
        descriptors = build_action_descriptors(env, state)
        mask = torch.zeros(NUM_ACTIONS, dtype=torch.float32)
        mask[list(legal)] = 1.0
        rows = []
        for start in range(0, len(hands), int(batch_size)):
            observations = []
            for hand in hands[start : start + int(batch_size)]:
                hypothetical = copy.copy(state)
                hypothetical.hole = [
                    list(cards) for cards in state.hole
                ]
                hypothetical.hole[actor] = [
                    int(hand[0]),
                    int(hand[1]),
                ]
                observations.append(
                    encode_information_state(
                        hypothetical,
                        actor,
                        legal,
                        env.bb,
                        int(self.snapshot.metadata["max_history"]),
                        action_descriptors=descriptors,
                    )
                )
            encoded = torch.stack(observations)
            logits = self.snapshot.policy_nets[actor](encoded.to(self.device))
            rows.append(
                masked_softmax(
                    logits,
                    mask.to(self.device).unsqueeze(0).expand_as(logits),
                ).cpu()
            )
        return torch.cat(rows, dim=0)

    @torch.inference_mode()
    def opponent_range(self, env: HeadsUpHoldemEnv, state) -> PublicRangeSnapshot:
        """Read the snapshot's exact opponent-card head at one live decision."""

        if state.terminal or state.to_act is None:
            raise ValueError("policy card head requires a live decision state")
        actor = int(state.to_act)
        network = self.snapshot.policy_nets[actor]
        if not hasattr(network, "forward_with_range"):
            raise ValueError("policy snapshot has no opponent-card head")
        legal = tuple(int(action) for action in env.legal_actions(state))
        encoded = encode_information_state(
            state,
            actor,
            legal,
            env.bb,
            int(self.snapshot.metadata["max_history"]),
            action_descriptors=build_action_descriptors(env, state),
        ).unsqueeze(0).to(self.device)
        _, range_logits = network.forward_with_range(encoded)
        probabilities = masked_range_probabilities(
            range_logits,
            valid_combo_mask_from_encoded(encoded),
        )[0].cpu()
        indices = torch.nonzero(
            probabilities > 0.0,
            as_tuple=False,
        ).flatten().tolist()
        weights = tuple(float(probabilities[index]) for index in indices)
        square_sum = sum(weight * weight for weight in weights)
        return PublicRangeSnapshot(
            combos=tuple(OPPONENT_COMBOS[index] for index in indices),
            weights=weights,
            effective_sample_size=(
                1.0 / square_sum if square_sum > 0.0 else 0.0
            ),
            updates=len(state.history),
        )


class AveragedHeadsUpSnapshotPolicy:
    """Equal-weight probability ensemble of compatible HU snapshots."""

    def __init__(self, policies: list[HeadsUpSnapshotPolicy]) -> None:
        if len(policies) < 2:
            raise ValueError("an averaged policy requires at least two snapshots")
        if len({policy.sha256 for policy in policies}) != len(policies):
            raise ValueError("averaged policy snapshots must be distinct")
        first = policies[0]
        compatibility_keys = (
            "model_id",
            "input_dim",
            "max_history",
            "action_schema_version",
            "engine_schema_version",
        )
        for policy in policies[1:]:
            for key in compatibility_keys:
                if policy.snapshot.metadata.get(key) != first.snapshot.metadata.get(
                    key
                ):
                    raise ValueError(
                        "cannot average incompatible policy snapshots: "
                        f"metadata {key!r} differs"
                    )
            if policy.mode != first.mode:
                raise ValueError("averaged policies must use the same policy mode")
            if policy.device != first.device:
                raise ValueError("averaged policies must use the same device")

        self.policies = tuple(policies)
        self.mode = first.mode
        self.device = first.device
        self.snapshot = first.snapshot
        self.iterations = tuple(policy.iteration for policy in policies)
        self.iteration = max(self.iterations)
        self.iteration_label = "+".join(
            str(iteration) for iteration in self.iterations
        )
        self.paths = tuple(policy.path for policy in policies)
        self.path = " + ".join(str(path) for path in self.paths)
        fingerprint_input = "average_probabilities_v1\0" + "\0".join(
            sorted(policy.sha256 for policy in policies)
        )
        self.sha256 = hashlib.sha256(
            fingerprint_input.encode("ascii")
        ).hexdigest()
        self.generator = torch.Generator(device="cpu")
        self.generator.set_state(first.generator.get_state())

    @staticmethod
    def _average(probability_tensors) -> torch.Tensor:
        averaged = torch.stack(tuple(probability_tensors), dim=0).mean(dim=0)
        denominator = averaged.sum(dim=-1, keepdim=True)
        if torch.any(denominator <= 0.0):
            raise RuntimeError("averaged policy produced zero probability mass")
        return averaged / denominator

    @torch.inference_mode()
    def probabilities(self, env: HeadsUpHoldemEnv, state) -> torch.Tensor:
        averaged, _ = self.probabilities_with_components(env, state)
        return averaged

    @torch.inference_mode()
    def probabilities_with_components(self, env, state):
        components = tuple(
            policy.probabilities(env, state) for policy in self.policies
        )
        return self._average(components), components

    @torch.inference_mode()
    def probabilities_batch(self, env: HeadsUpHoldemEnv, states) -> torch.Tensor:
        return self._average(
            policy.probabilities_batch(env, states) for policy in self.policies
        )

    @torch.inference_mode()
    def sample_actions_batch(self, env, states, thresholds) -> list[int]:
        if len(states) != len(thresholds):
            raise ValueError("states and thresholds must have identical lengths")
        if not states:
            return []
        probabilities = self.probabilities_batch(env, states).to(torch.float64)
        uniforms = torch.as_tensor(thresholds, dtype=torch.float64).unsqueeze(1)
        hits = torch.cumsum(probabilities, dim=1) + 1e-12 >= uniforms
        selected = torch.argmax(hits.to(torch.int8), dim=1)
        return [int(action) for action in selected.tolist()]

    def choose_action(self, env, state) -> tuple[int, torch.Tensor]:
        probabilities = self.probabilities(env, state)
        if self.mode == "argmax":
            action = int(torch.argmax(probabilities).item())
        else:
            action = int(
                torch.multinomial(
                    probabilities, 1, generator=self.generator
                ).item()
            )
        if action not in env.legal_actions(state):
            raise RuntimeError("averaged policy selected an illegal action")
        return action, probabilities

    @torch.inference_mode()
    def probabilities_for_holes(self, env, state, actor, hands, **kwargs):
        return self._average(
            policy.probabilities_for_holes(
                env, state, actor, hands, **kwargs
            )
            for policy in self.policies
        )

    @torch.inference_mode()
    def opponent_range(self, env, state) -> PublicRangeSnapshot:
        snapshots = [
            policy.opponent_range(env, state) for policy in self.policies
        ]
        combined = {}
        for snapshot in snapshots:
            for combo, weight in zip(snapshot.combos, snapshot.weights):
                combined[combo] = combined.get(combo, 0.0) + (
                    float(weight) / len(snapshots)
                )
        total = sum(combined.values())
        if total <= 0.0:
            raise RuntimeError("averaged policy produced an empty opponent range")
        combos = tuple(sorted(combined))
        weights = tuple(combined[combo] / total for combo in combos)
        square_sum = sum(weight * weight for weight in weights)
        return PublicRangeSnapshot(
            combos=combos,
            weights=weights,
            effective_sample_size=(
                1.0 / square_sum if square_sum > 0.0 else 0.0
            ),
            updates=len(state.history),
        )


class PolicyResultsLedger:
    """Append-only hand ledger with aggregates separated by policy and mode."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records: list[dict] = []
        if self.path.exists():
            for line_number, line in enumerate(
                self.path.read_text(encoding="utf-8").splitlines(),
                start=1,
            ):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid results ledger JSON at line {line_number}"
                    ) from exc
                if int(record.get("version", -1)) not in {1, 2}:
                    raise ValueError(
                        f"unsupported results ledger row at line {line_number}"
                    )
                self.records.append(record)

    def append(self, record: dict) -> None:
        row = dict(record)
        row["version"] = 2
        encoded = json.dumps(row, separators=(",", ":"), sort_keys=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(encoded + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.records.append(row)

    @staticmethod
    def _key(record: dict) -> tuple[str, str]:
        mode = str(record["policy_mode"]).replace(
            "+raw_first_preflop+",
            "+",
            1,
        )
        return str(record["policy_sha256"]), mode

    def summaries(self) -> list[dict]:
        grouped: dict[tuple[str, str], dict] = {}
        for record in self.records:
            key = self._key(record)
            if key not in grouped:
                grouped[key] = {
                    "policy_sha256": key[0],
                    "policy_mode": key[1],
                    "policy_iteration": int(record["policy_iteration"]),
                    "policy_file": str(record["policy_file"]),
                    "hands": 0,
                    "human_net_bb": 0.0,
                    "wins": 0,
                    "losses": 0,
                    "ties": 0,
                }
            summary = grouped[key]
            summary["hands"] += 1
            summary["human_net_bb"] += float(record["human_payoff_bb"])
            outcome = str(record["human_outcome"])
            if outcome == "win":
                summary["wins"] += 1
            elif outcome == "loss":
                summary["losses"] += 1
            else:
                summary["ties"] += 1
        for summary in grouped.values():
            hands = int(summary["hands"])
            summary["human_bb_per_hand"] = (
                float(summary["human_net_bb"]) / hands if hands else 0.0
            )
            summary["policy_bb_per_hand"] = -float(
                summary["human_bb_per_hand"]
            )
        return sorted(
            grouped.values(),
            key=lambda item: (
                int(item["policy_iteration"]),
                str(item["policy_mode"]),
                str(item["policy_sha256"]),
            ),
        )

    def summary_for(
        self,
        policy: HeadsUpSnapshotPolicy,
        *,
        mode: str | None = None,
    ) -> dict:
        selected_mode = policy.mode if mode is None else str(mode)
        key = (policy.sha256, selected_mode)
        for summary in self.summaries():
            if (
                str(summary["policy_sha256"]),
                str(summary["policy_mode"]),
            ) == key:
                return summary
        return {
            "policy_sha256": policy.sha256,
            "policy_mode": selected_mode,
            "policy_iteration": policy.iteration,
            "policy_file": str(policy.path),
            "hands": 0,
            "human_net_bb": 0.0,
            "human_bb_per_hand": 0.0,
            "policy_bb_per_hand": 0.0,
            "wins": 0,
            "losses": 0,
            "ties": 0,
        }


def format_chips(value: float) -> str:
    value = float(value)
    if abs(value - round(value)) <= EPSILON:
        return str(int(round(value)))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _event_attr(event, *names: str, default=None):
    if isinstance(event, dict):
        for name in names:
            if name in event:
                return event[name]
    else:
        for name in names:
            if hasattr(event, name):
                return getattr(event, name)
    return default


def event_summary(event) -> str:
    """Return one exact, semantic history line without bucket translation."""

    street = int(_event_attr(event, "street", default=0))
    player = int(_event_attr(event, "player", "actor", default=0))
    kind = str(
        _event_attr(
            event,
            "kind",
            "semantic_action",
            "action_name",
            "name",
            default="action",
        )
    ).replace("_", " ")
    amount = float(
        _event_attr(event, "amount", "amount_added", "payment", default=0.0)
    )
    target = float(
        _event_attr(
            event,
            "contribution_after",
            "target",
            "raise_to",
            default=amount,
        )
    )
    before = float(_event_attr(event, "current_bet_before", default=0.0))
    after = float(_event_attr(event, "current_bet_after", default=before))
    pot_after = float(_event_attr(event, "pot_after", default=amount))
    full_raise = bool(
        _event_attr(event, "full_raise", "is_full_raise", default=False)
    )
    full = " full raise" if full_raise else ""
    return (
        f"{STREET_NAMES[street]:8s}  P{player} {kind:<18s} "
        f"+{format_chips(amount):>6s}  to {format_chips(target):>6s}  "
        f"bet {format_chips(before)}->{format_chips(after)}  "
        f"pot {format_chips(pot_after)}{full}"
    )


def is_first_preflop_bot_decision(state, bot_seat: int) -> bool:
    """Whether this is the bot's first voluntary decision of the hand."""
    if int(state.street) != 0:
        return False
    return not any(
        int(_event_attr(event, "player", "actor", default=-1))
        == int(bot_seat)
        for event in state.history
    )


def is_premium_preflop_never_fold_hand(state, seat: int) -> bool:
    """AA-JJ and every AK combination are protected from preflop folds."""
    if int(state.street) != 0:
        return False
    hole = tuple(int(card) for card in state.hole[int(seat)])
    if len(hole) != 2:
        return False
    ranks = tuple(card % 13 for card in hole)
    return (
        ranks[0] == ranks[1] and ranks[0] >= RANK_SYMBOLS.index("J")
    ) or set(ranks) == {
        RANK_SYMBOLS.index("A"),
        RANK_SYMBOLS.index("K"),
    }


def should_remove_fold_on_first_preflop_action(state, seat: int) -> bool:
    return is_first_preflop_bot_decision(
        state,
        seat,
    ) and is_premium_preflop_never_fold_hand(state, seat)


def should_apply_premium_preflop_guard(
    state,
    seat: int,
    *,
    unmodified_policy_sampling: bool,
) -> bool:
    return (
        not bool(unmodified_policy_sampling)
        and should_remove_fold_on_first_preflop_action(state, seat)
    )


def probabilities_without_fold(
    probabilities: torch.Tensor,
    legal: list[int],
) -> torch.Tensor:
    """Condition a legal policy distribution on taking a non-fold action."""
    adjusted = probabilities.detach().clone().to(dtype=torch.float32)
    fold_actions = [
        action for action in legal if ACTION_NAMES[int(action)] == "fold"
    ]
    adjusted[fold_actions] = 0.0
    total = float(adjusted[legal].sum().item())
    if total <= EPSILON:
        nonfold = [
            action
            for action in legal
            if ACTION_NAMES[int(action)] != "fold"
        ]
        if not nonfold:
            raise RuntimeError("premium preflop hand has no non-fold action")
        adjusted.zero_()
        adjusted[nonfold] = 1.0 / len(nonfold)
        return adjusted
    adjusted /= total
    return adjusted


def probabilities_from_top_k(
    probabilities: torch.Tensor,
    legal: list[int],
    top_k: int,
) -> torch.Tensor:
    """Keep the highest-probability legal actions and renormalize their mass."""
    if int(top_k) <= 0:
        return probabilities
    legal = [int(action) for action in legal]
    keep_count = min(int(top_k), len(legal))
    selected = sorted(
        legal,
        key=lambda action: (-float(probabilities[action]), action),
    )[:keep_count]
    adjusted = torch.zeros_like(probabilities, dtype=torch.float32)
    adjusted[selected] = probabilities[selected].to(dtype=torch.float32)
    total = float(adjusted.sum().item())
    if total <= EPSILON:
        adjusted[selected] = 1.0 / len(selected)
    else:
        adjusted /= total
    return adjusted


def structured_action_history(state) -> list[dict]:
    """Serialize exact engine events with the public board visible at action."""

    board_counts = {0: 0, 1: 3, 2: 4, 3: 5}
    total_contributions = [0, 0]
    total_contributions[int(state.sb_player)] = min(
        int(state.initial_stacks[int(state.sb_player)]),
        int(state.small_blind),
    )
    total_contributions[int(state.bb_player)] = min(
        int(state.initial_stacks[int(state.bb_player)]),
        int(state.big_blind),
    )
    stacks = [
        int(state.initial_stacks[player]) - total_contributions[player]
        for player in range(2)
    ]
    result = []
    for index, event in enumerate(state.history):
        street = int(_event_attr(event, "street", default=0))
        player = int(_event_attr(event, "player", default=0))
        amount = int(_event_attr(event, "amount", default=0))
        action = _event_attr(event, "action", default=None)
        action = None if action is None else int(action)
        board = [
            int(card)
            for card in state.board[: board_counts.get(street, 5)]
        ]
        stacks_before = list(stacks)
        contributions_before = list(total_contributions)
        stacks[player] -= amount
        total_contributions[player] += amount
        result.append(
            {
                "index": index,
                "player": player,
                "street": street,
                "street_name": STREET_NAMES[street],
                "abstract_action": action,
                "abstract_action_name": (
                    ACTION_NAMES[action]
                    if action is not None and 0 <= action < NUM_ACTIONS
                    else None
                ),
                "kind": str(_event_attr(event, "kind", default="")),
                "amount": amount,
                "raise_to": int(
                    _event_attr(event, "raise_to", default=0)
                ),
                "contribution_after": int(
                    _event_attr(event, "contribution_after", default=0)
                ),
                "current_bet_before": int(
                    _event_attr(event, "current_bet_before", default=0)
                ),
                "current_bet_after": int(
                    _event_attr(event, "current_bet_after", default=0)
                ),
                "pot_before": int(
                    _event_attr(event, "pot_before", default=0)
                ),
                "pot_after": int(
                    _event_attr(event, "pot_after", default=0)
                ),
                "to_call_before": int(
                    _event_attr(event, "to_call_before", default=0)
                ),
                "full_raise": bool(
                    _event_attr(event, "full_raise", default=False)
                ),
                "all_in": bool(
                    _event_attr(event, "all_in", default=False)
                ),
                "stacks_before": stacks_before,
                "stacks_after": list(stacks),
                "total_contributions_before": contributions_before,
                "total_contributions_after": list(total_contributions),
                "board_card_ids": board,
                "board": [card_to_string(card) for card in board],
            }
        )
    return result


def reconstruct_initial_deck(state) -> list[int]:
    """Recover the exact 52-card input deck from a completed engine state."""

    deal_order = [
        int(state.hole[state.bb_player][0]),
        int(state.hole[state.sb_player][0]),
        int(state.hole[state.bb_player][1]),
        int(state.hole[state.sb_player][1]),
    ]
    popped = list(deal_order)
    if len(state.board) >= 3:
        popped.extend(
            [
                int(state.burned[0]),
                *[int(card) for card in state.board[:3]],
            ]
        )
    if len(state.board) >= 4:
        popped.extend([int(state.burned[1]), int(state.board[3])])
    if len(state.board) >= 5:
        popped.extend([int(state.burned[2]), int(state.board[4])])
    deck = [int(card) for card in state.deck] + list(reversed(popped))
    if len(deck) != 52 or len(set(deck)) != 52:
        raise RuntimeError("could not reconstruct a complete unique deck")
    return deck


def terminal_hand_audit(state) -> dict:
    """Return lossless terminal data suitable for replay and luck analysis."""

    initial_deck = reconstruct_initial_deck(state)
    action_sequence = structured_action_history(state)
    return {
        "record_schema": "heads_up_gui_hand_v2_complete",
        "engine_schema_version": str(ENGINE_SCHEMA_VERSION),
        "action_schema_version": str(ACTION_SCHEMA_VERSION),
        "initial_deck_order_card_ids": initial_deck,
        "initial_deck_order": [
            card_to_string(card) for card in initial_deck
        ],
        "hole_card_ids": [
            [int(card) for card in cards] for cards in state.hole
        ],
        "board_card_ids": [int(card) for card in state.board],
        "burned_card_ids": [int(card) for card in state.burned],
        "remaining_deck_card_ids": [int(card) for card in state.deck],
        "action_sequence": action_sequence,
        "all_in_action_indices": [
            int(row["index"]) for row in action_sequence if row["all_in"]
        ],
        "terminal_reason": (
            "fold" if any(bool(value) for value in state.folded)
            else "showdown"
        ),
        "folded": [bool(value) for value in state.folded],
        "winners": [int(value) for value in state.winners],
        "total_contributions": [
            int(value) for value in state.total_contrib
        ],
        "uncalled_returns": [
            int(value) for value in state.uncalled_returns
        ],
        "payouts": [int(value) for value in (state.payouts or ())],
        "payoffs": [int(value) for value in (state.payoffs or ())],
    }


def fixed_action_label(env: HeadsUpHoldemEnv, state, action: int) -> str:
    """Label a finite policy slot with the exact effect the engine will apply."""

    if action < 0 or action >= NUM_ACTIONS:
        raise ValueError(f"action must be in 0..{NUM_ACTIONS - 1}")
    actor = state.to_act
    if actor is None:
        return ACTION_NAMES[action].replace("_", " ").title()
    target = float(env.action_target(state, action))
    contribution = float(state.street_contrib[int(actor)])
    payment = max(0.0, target - contribution)
    readable = ACTION_NAMES[action].replace("_", " ").title()
    if payment <= EPSILON:
        return readable
    return (
        f"{readable}\n"
        f"to {format_chips(target)}  (+{format_chips(payment)})"
    )


def state_facts(env: HeadsUpHoldemEnv, state) -> dict[str, str]:
    """GUI-neutral exact values used by the diagnostics panel and tests."""

    actor = state.to_act
    minimum_increment = float(
        getattr(
            state,
            "min_raise",
            getattr(state, "minimum_raise", getattr(state, "min_raise_increment", 0.0)),
        )
    )
    current_bet = float(state.current_bet)
    if actor is None:
        to_call = 0.0
        max_raise_to = 0.0
        actor_text = "-"
    else:
        actor = int(actor)
        to_call = float(env.amount_to_call(state, actor))
        max_raise_to = float(state.street_contrib[actor]) + float(state.stacks[actor])
        actor_text = f"P{actor}"
    return {
        "street": STREET_NAMES[int(state.street)],
        "actor": actor_text,
        "pot": format_chips(state.pot),
        "current_bet": format_chips(current_bet),
        "minimum_raise_increment": format_chips(minimum_increment),
        "minimum_raise_to": format_chips(current_bet + minimum_increment),
        "to_call": format_chips(to_call),
        "maximum_raise_to": format_chips(max_raise_to),
    }


class HeadsUpManualGUI:
    TABLE = "#146b3a"
    FELT_EDGE = "#0a4728"
    PANEL = "#17202a"
    TEXT = "#f4f6f7"
    MUTED = "#bdc3c7"
    GOLD = "#f4d03f"

    def __init__(
        self,
        root: tk.Tk,
        *,
        starting_stack: int,
        small_blind: int,
        big_blind: int,
        seed: int | None,
        first_button: int,
        policy: HeadsUpSnapshotPolicy | None = None,
        human_seat: int = 0,
        bot_delay_ms: int = 450,
        results_log: str | Path = DEFAULT_RESULTS_LOG,
        search_enabled: bool = True,
        search_mode: str = "three-player",
        root_range_mode: str = "inferred",
        search_workers: int = 0,
        search_budget_seconds: float = 6.0,
        root_search_ms: int = 10_000,
        root_search_rollouts: int = 150_000,
        root_search_batch_iterations: int = 3072,
        root_blueprint_weight: float = 0.65,
        root_min_strategy_probability: float = 0.0,
        robust_action_noise: float = 0.10,
        robust_kl_radius: float = 0.20,
        unmodified_policy_sampling: bool = False,
        top_policy_actions: int = 0,
    ) -> None:
        self.root = root
        self.starting_stack = int(starting_stack)
        self.small_blind = int(small_blind)
        self.big_blind = int(big_blind)
        self.seed = seed
        self.first_button = int(first_button)
        self.policy = policy
        self.human_seat = int(human_seat)
        if self.human_seat not in (0, 1):
            raise ValueError("human_seat must be 0 or 1")
        self.bot_seat = 1 - self.human_seat
        self.bot_delay_ms = max(0, int(bot_delay_ms))
        self.unmodified_policy_sampling = bool(unmodified_policy_sampling)
        self.top_policy_actions = max(0, int(top_policy_actions))
        self.bot_job: str | None = None
        self.search_poll_job: str | None = None
        self.search_future: Future | None = None
        self.search_token: tuple | None = None
        self.last_bot_action = ""
        self.search_decision_log: list[dict] = []
        self.policy_decision_log: list[dict] = []
        self.range_log: dict[int, dict] = {}
        self.session_id = uuid.uuid4().hex
        self.logged_hands: set[int] = set()
        self.results_ledger = (
            PolicyResultsLedger(results_log)
            if self.policy is not None
            else None
        )
        self.public_range = (
            BlueprintPublicRange()
            if self.policy is not None
            else None
        )
        self.search_enabled = bool(search_enabled and self.policy is not None)
        if search_mode not in {"three-player", "robust", "cfr"}:
            raise ValueError(
                "search_mode must be 'three-player', 'robust', or 'cfr'"
            )
        self.search_mode = str(search_mode)
        if root_range_mode not in {"uniform", "inferred"}:
            raise ValueError("root_range_mode must be 'uniform' or 'inferred'")
        self.root_range_mode = str(root_range_mode)
        self.root_search_ms = int(root_search_ms)
        self.root_search_rollouts = int(root_search_rollouts)
        self.root_search_batch_iterations = int(
            root_search_batch_iterations
        )
        if not 0.0 <= float(root_blueprint_weight) <= 1.0:
            raise ValueError("root_blueprint_weight must be in [0, 1]")
        self.root_blueprint_weight = float(root_blueprint_weight)
        self.root_min_strategy_probability = float(
            root_min_strategy_probability
        )
        if not 0.0 <= float(robust_action_noise) < 1.0:
            raise ValueError("robust_action_noise must be in [0, 1)")
        if float(robust_kl_radius) < 0.0:
            raise ValueError("robust_kl_radius must be nonnegative")
        self.robust_action_noise = float(robust_action_noise)
        self.robust_kl_radius = float(robust_kl_radius)
        self.search_executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="hu-search")
            if self.search_enabled
            else None
        )
        if not self.search_enabled:
            self.search = None
        elif self.search_mode == "three-player":
            self.search = HeadsUpRootPolicySearch(
                self.policy,
                time_budget_ms=int(root_search_ms),
                max_rollouts=int(root_search_rollouts),
                blueprint_weight=self.root_blueprint_weight,
                max_actions_per_rollout=64,
                batch_iterations=int(root_search_batch_iterations),
                range_mode=self.root_range_mode,
                range_temperature=0.65,
                uniform_contamination=0.25,
                min_strategy_probability=float(
                    root_min_strategy_probability
                ),
                seed=seed,
            )
        elif self.search_mode == "robust":
            self.search = RobustHeadsUpSearch(
                self.policy,
                time_budget_ms=int(root_search_ms),
                max_rollouts=int(root_search_rollouts),
                max_actions_per_rollout=64,
                kl_radius=self.robust_kl_radius,
                continuation_action_noise=0.04,
                seed=seed,
            )
        else:
            self.search = MultiprocessPluribusSearch(
                self.policy.path,
                workers=int(search_workers),
                time_budget_seconds=float(search_budget_seconds),
                seed=seed,
            )
        self.env = self._new_environment()
        self.state = None
        self.hand_number = 0
        self.session_stacks = [self.starting_stack, self.starting_stack]
        self.next_button = self.first_button
        self.action_buttons: list[ttk.Button] = []
        self.hud_hands: list[dict] = persisted_hud_hands(
            self.results_ledger.records
            if self.results_ledger is not None
            else ()
        )
        self.hud_logged_hands: set[int] = set()
        self.bot_cards_revealed = False
        self.exact_raise_var = tk.StringVar()

        if self.policy is None:
            root.title("Heads-Up Hold'em - exact engine manual test")
        else:
            root.title(
                "Heads-Up Hold'em - human vs policy "
                f"(iterations {self.policy.iteration_label})"
            )
        root.geometry("1180x820")
        root.minsize(980, 700)
        root.configure(bg=self.PANEL)

        self._build_widgets()
        self.root.bind("<KeyPress>", self._handle_keyboard_shortcut, add="+")
        if hasattr(self.root, "protocol"):
            self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.reset_match()

    def _new_environment(self) -> HeadsUpHoldemEnv:
        return HeadsUpHoldemEnv(
            starting_stack=self.starting_stack,
            small_blind=self.small_blind,
            big_blind=self.big_blind,
            seed=self.seed,
        )

    def _build_widgets(self) -> None:
        top = tk.Frame(self.root, bg=self.PANEL, padx=12, pady=8)
        top.pack(fill="x")
        self.status = tk.Label(
            top,
            text="",
            bg=self.PANEL,
            fg=self.TEXT,
            font=("Segoe UI", 12, "bold"),
        )
        self.status.pack(anchor="w")
        self.diagnostics = tk.Label(
            top,
            text="",
            bg=self.PANEL,
            fg=self.MUTED,
            justify="left",
            font=("Consolas", 10),
        )
        self.diagnostics.pack(anchor="w", pady=(4, 0))
        self.policy_stats = tk.Label(
            top,
            text="",
            bg=self.PANEL,
            fg=self.GOLD,
            justify="left",
            font=("Consolas", 10, "bold"),
        )
        self.policy_stats.pack(anchor="w", pady=(4, 0))

        middle = tk.Frame(self.root, bg=self.PANEL)
        middle.pack(fill="both", expand=True, padx=12)
        self.canvas = tk.Canvas(
            middle,
            bg=self.FELT_EDGE,
            highlightthickness=0,
            width=780,
            height=510,
        )
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _event: self.draw_table())

        history_panel = tk.Frame(middle, bg=self.PANEL, width=405)
        history_panel.pack(side="right", fill="y", padx=(12, 0))
        history_panel.pack_propagate(False)
        self.range_plot_title = tk.Label(
            history_panel,
            text="Inferred human range — waiting for your action",
            bg=self.PANEL,
            fg=self.GOLD,
            font=("Segoe UI", 10, "bold"),
        )
        self.range_plot_title.pack(anchor="w", pady=(0, 3))
        self.range_plot_canvas = tk.Canvas(
            history_panel,
            width=380,
            height=340,
            bg="#0e151b",
            highlightthickness=0,
        )
        self.range_plot_canvas.pack(fill="x", pady=(0, 5))
        tk.Label(
            history_panel,
            text="Exact public action history",
            bg=self.PANEL,
            fg=self.TEXT,
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w", pady=(0, 3))
        self.history_text = tk.Text(
            history_panel,
            width=57,
            bg="#0e151b",
            fg="#d5dbdb",
            insertbackground="white",
            relief="flat",
            wrap="none",
            font=("Consolas", 9),
            state="disabled",
        )
        self.history_text.tag_configure(
            POLICY_SHIFT_POSITIVE_TAG,
            foreground="#55d68b",
        )
        self.history_text.tag_configure(
            POLICY_SHIFT_NEGATIVE_TAG,
            foreground="#ff6b6b",
        )
        self.history_text.tag_configure(
            POLICY_SHIFT_NEUTRAL_TAG,
            foreground=self.MUTED,
        )
        y_scroll = ttk.Scrollbar(history_panel, command=self.history_text.yview)
        y_scroll.pack(side="right", fill="y")
        x_scroll = ttk.Scrollbar(
            history_panel,
            orient="horizontal",
            command=self.history_text.xview,
        )
        x_scroll.pack(side="bottom", fill="x")
        self.history_text.configure(
            yscrollcommand=y_scroll.set,
            xscrollcommand=x_scroll.set,
        )
        self.history_text.pack(side="left", fill="both", expand=True)

        controls = tk.Frame(self.root, bg=self.PANEL, padx=12, pady=10)
        controls.pack(fill="x")
        self.prompt = tk.Label(
            controls,
            text="",
            bg=self.PANEL,
            fg=self.GOLD,
            font=("Segoe UI", 11, "bold"),
        )
        self.prompt.pack(anchor="w", pady=(0, 6))

        self.actions_frame = tk.Frame(controls, bg=self.PANEL)
        self.actions_frame.pack(fill="x")

        exact_row = tk.Frame(controls, bg=self.PANEL)
        exact_row.pack(fill="x", pady=(8, 0))
        tk.Label(
            exact_row,
            text="Arbitrary integer raise-to:",
            bg=self.PANEL,
            fg=self.TEXT,
        ).pack(side="left")
        self.exact_raise_entry = ttk.Entry(
            exact_row,
            textvariable=self.exact_raise_var,
            width=12,
        )
        self.exact_raise_entry.pack(side="left", padx=(6, 6))
        self.exact_raise_button = ttk.Button(
            exact_row,
            text="Apply exact raise",
            command=self.apply_exact_raise,
        )
        self.exact_raise_button.pack(side="left")
        self.exact_range = tk.Label(
            exact_row,
            text="",
            bg=self.PANEL,
            fg=self.MUTED,
        )
        self.exact_range.pack(side="left", padx=(10, 0))
        self.next_hand_button = ttk.Button(
            exact_row,
            text="Next hand (rotate button)",
            command=self._start_next_hand_if_available,
            state="disabled",
        )
        self.next_hand_button.pack(side="right")
        self.reveal_bot_button = ttk.Button(
            exact_row,
            text="Reveal bot cards",
            command=self.reveal_bot_cards,
            state="disabled",
        )
        self.reveal_bot_button.pack(side="right", padx=(0, 8))
        ttk.Button(
            exact_row,
            text="Reset match",
            command=self.reset_match,
        ).pack(side="right", padx=(0, 8))
        self.policy_history_button = ttk.Button(
            exact_row,
            text="Policy history",
            command=self.show_policy_history,
            state="normal" if self.policy is not None else "disabled",
        )
        self.policy_history_button.pack(side="right", padx=(0, 8))

    def _apply_named_action_if_available(self, action_name: str) -> bool:
        if self.state is None or self.state.terminal:
            return False
        if self.policy is not None and int(self.state.to_act) == self.bot_seat:
            return False
        for action in self.env.legal_actions(self.state):
            if ACTION_NAMES[int(action)] == action_name:
                self.apply_fixed_action(int(action))
                return True
        return False

    def _handle_keyboard_shortcut(self, event):
        key = str(getattr(event, "keysym", "")).casefold()
        if not key and str(getattr(event, "char", "")) == " ":
            key = "space"
        if key == "c":
            invoked = self._apply_named_action_if_available("call")
        elif key == "f":
            invoked = self._apply_named_action_if_available("fold")
        elif key == "space":
            invoked = self._start_next_hand_if_available()
        else:
            invoked = False
        return "break" if invoked else None

    def _start_next_hand_if_available(self) -> bool:
        """Start the next hand through the shared click/Space guard."""

        if (
            self.state is None
            or not self.state.terminal
            or min(self.session_stacks) <= EPSILON
        ):
            return False
        self.deal_hand()
        return True

    def reset_match(self) -> None:
        self._cancel_bot_job()
        self._invalidate_search()
        self.session_id = uuid.uuid4().hex
        self.logged_hands.clear()
        self.hud_logged_hands.clear()
        self.env = self._new_environment()
        self.state = None
        self.hand_number = 0
        self.session_stacks = [self.starting_stack, self.starting_stack]
        self.next_button = self.first_button
        self.last_bot_action = ""
        self.search_decision_log = []
        self.policy_decision_log = []
        self.range_log = {}
        self.deal_hand()

    def reveal_bot_cards(self) -> None:
        if (
            self.policy is None
            or self.state is None
            or not self.state.terminal
            or self.bot_cards_revealed
        ):
            return
        self.bot_cards_revealed = True
        self.reveal_bot_button.configure(state="disabled")
        self.draw_table()

    def deal_hand(self) -> None:
        if self.state is not None and not self.state.terminal:
            return
        if min(self.session_stacks) <= EPSILON:
            messagebox.showinfo(
                "Match complete",
                "One player has no chips. Use Reset match to start again.",
            )
            return
        button = self.next_button
        self.next_button = 1 - button
        self.hand_number += 1
        self.last_bot_action = ""
        self.search_decision_log = []
        self.policy_decision_log = []
        self.range_log = {}
        self.bot_cards_revealed = False
        self.state = self.env.new_hand(button=button, stacks=self.session_stacks)
        if self.public_range is not None:
            self.public_range.reset(
                [
                    *self.state.hole[self.bot_seat],
                    *self.state.board,
                ]
            )
        if self.state.terminal:
            self.session_stacks = [
                0 if int(value) <= 0 else int(value)
                for value in self.state.stacks
            ]
            self._record_hud_hand()
            self._record_completed_hand()
        self.next_hand_button.configure(state="disabled")
        self.reveal_bot_button.configure(state="disabled")
        self.refresh()

    def apply_fixed_action(self, action: int) -> None:
        if self.state is None or self.state.terminal:
            return
        if self.policy is not None and int(self.state.to_act) == self.bot_seat:
            return
        if action not in self.env.legal_actions(self.state):
            return
        before = self.state
        try:
            self.state = self.env.step(before, action)
        except Exception as exc:
            messagebox.showerror("Engine rejected action", str(exc))
            return
        self._observe_human_action(before, self.state)
        self._after_action()

    def apply_exact_raise(self) -> None:
        if self.state is None or self.state.terminal:
            return
        if self.policy is not None and int(self.state.to_act) == self.bot_seat:
            return
        raw = self.exact_raise_var.get().strip()
        try:
            # Explicitly reject decimal/exponential inputs: this control is for
            # real poker-room integer chip targets.
            raise_to = int(raw, 10)
            if str(raise_to) != raw and str(raise_to) != raw.lstrip("+"):
                raise ValueError
            if raise_to < 0:
                raise ValueError
        except ValueError:
            messagebox.showerror(
                "Invalid raise target",
                "Enter one nonnegative integer total street contribution.",
            )
            return
        before = self.state
        try:
            self.state = self.env.step_exact(
                before,
                "raise_to",
                raise_to=raise_to,
            )
        except Exception as exc:
            messagebox.showerror("Illegal exact raise", str(exc))
            return
        self._observe_human_action(before, self.state)
        self._after_action()

    def _observe_human_action(self, before, after) -> None:
        if (
            self.public_range is None
            or self.policy is None
            or before is None
            or before.to_act is None
            or int(before.to_act) != self.human_seat
            or not after.history
        ):
            return
        event = after.history[-1]
        kind = str(_event_attr(event, "kind", default=""))
        target = int(_event_attr(event, "raise_to", default=0))
        self.public_range.filter_known(
            [
                *before.hole[self.bot_seat],
                *before.board,
            ]
        )
        probabilities = self.policy.probabilities_for_holes(
            self.env,
            before,
            self.human_seat,
            self.public_range.combos,
        )
        likelihoods = observed_action_likelihoods(
            self.env,
            before,
            probabilities,
            kind=kind,
            raise_to=target if kind in {"bet", "raise"} else None,
        )
        if self.search_mode == "robust":
            likelihoods = action_noise_likelihoods(
                likelihoods,
                legal_action_count=len(self.env.legal_actions(before)),
                epsilon=self.robust_action_noise,
            )
        self.public_range.condition(likelihoods)
        if (
            not after.terminal
            and after.to_act is not None
            and int(after.to_act) == self.bot_seat
            and bool(self.policy.snapshot.metadata.get("has_range_head"))
        ):
            self.range_log[len(after.history) - 1] = summarize_public_range(
                self.policy.opponent_range(self.env, after),
                actual_hole=after.hole[self.human_seat],
            )
            self.range_log[len(after.history) - 1]["source"] = (
                "policy_card_head"
            )
            return
        if self.search_mode == "robust":
            self.range_log[len(after.history) - 1] = summarize_public_range(
                self.public_range.snapshot(),
                actual_hole=before.hole[self.human_seat],
            )
        elif (
            self.search_mode == "three-player"
            and self.root_range_mode == "inferred"
        ):
            inferred = robust_inferred_range(
                self.public_range.snapshot(),
                temperature=0.65,
                uniform_contamination=0.25,
            )
            self.range_log[len(after.history) - 1] = summarize_public_range(
                inferred,
                actual_hole=before.hole[self.human_seat],
            )

    def _after_action(self) -> None:
        if self.state.terminal:
            self.session_stacks = [
                0 if int(value) <= 0 else int(value)
                for value in self.state.stacks
            ]
            self._record_hud_hand()
            self._record_completed_hand()
        self.refresh()

    def _record_hud_hand(self) -> None:
        if (
            self.state is None
            or not self.state.terminal
            or self.hand_number in self.hud_logged_hands
        ):
            return
        self.hud_hands.append(
            canonical_hud_hand(
                int(self.state.button),
                self.state.history,
                self.human_seat if self.policy is not None else 0,
            )
        )
        self.hud_logged_hands.add(self.hand_number)

    def _hud_hands_with_current(self) -> list[dict]:
        hands = list(self.hud_hands)
        if (
            self.state is not None
            and self.hand_number not in self.hud_logged_hands
        ):
            hands.append(
                canonical_hud_hand(
                    int(self.state.button),
                    self.state.history,
                    self.human_seat if self.policy is not None else 0,
                )
            )
        return hands

    def _record_completed_hand(self) -> None:
        if (
            self.policy is None
            or self.results_ledger is None
            or self.state is None
            or not self.state.terminal
            or self.hand_number in self.logged_hands
        ):
            return
        payoffs = tuple(float(value) for value in self.state.payoffs)
        human_payoff_bb = payoffs[self.human_seat] / float(self.env.bb)
        if human_payoff_bb > EPSILON:
            outcome = "win"
        elif human_payoff_bb < -EPSILON:
            outcome = "loss"
        else:
            outcome = "tie"
        record = {
            "logged_at_utc": datetime.now(timezone.utc).isoformat(),
            "session_id": self.session_id,
            "session_hand_number": self.hand_number,
            "policy_sha256": self.policy.sha256,
            "policy_iteration": self.policy.iteration,
            "policy_file": str(self.policy.path),
            "policy_iterations": list(
                getattr(self.policy, "iterations", (self.policy.iteration,))
            ),
            "policy_files": [
                str(path)
                for path in getattr(self.policy, "paths", (self.policy.path,))
            ],
            "policy_mode": self._controller_mode(),
            "hud_history_version": HUD_HISTORY_VERSION,
            "human_seat": self.human_seat,
            "policy_seat": self.bot_seat,
            "button": int(self.state.button),
            "small_blind": float(self.env.sb),
            "big_blind": float(self.env.bb),
            "initial_stacks": [
                float(value) for value in self.state.initial_stacks
            ],
            "final_stacks": [float(value) for value in self.state.stacks],
            "payoffs_chips": list(payoffs),
            "human_payoff_bb": human_payoff_bb,
            "policy_payoff_bb": -human_payoff_bb,
            "human_outcome": outcome,
            "board": [
                card_to_string(int(card)) for card in self.state.board
            ],
            "human_hole": [
                card_to_string(int(card))
                for card in self.state.hole[self.human_seat]
            ],
            "policy_hole": [
                card_to_string(int(card))
                for card in self.state.hole[self.bot_seat]
            ],
            "public_history": [
                event_summary(event) for event in self.state.history
            ],
            "search_decisions": list(self.search_decision_log),
            "policy_decisions": list(self.policy_decision_log),
            "inferred_range_log": dict(self.range_log),
            "controller_configuration": {
                "search_enabled": bool(self.search_enabled),
                "first_preflop_bot_action_policy_only": True,
                "search_mode": self.search_mode,
                "root_range_mode": self.root_range_mode,
                "root_search_ms_internal": self.root_search_ms,
                "root_search_rollout_cap": self.root_search_rollouts,
                "root_search_batch_iterations": (
                    self.root_search_batch_iterations
                ),
                "root_blueprint_weight": self.root_blueprint_weight,
                "root_search_weight": 1.0 - self.root_blueprint_weight,
                "root_min_strategy_probability": (
                    self.root_min_strategy_probability
                ),
                "unmodified_policy_sampling": (
                    self.unmodified_policy_sampling
                ),
                "top_policy_actions": self.top_policy_actions,
                "range_temperature": (
                    0.65
                    if self.search_mode == "three-player"
                    and self.root_range_mode == "inferred"
                    else None
                ),
                "uniform_contamination": (
                    0.25
                    if self.search_mode == "three-player"
                    and self.root_range_mode == "inferred"
                    else None
                ),
                "robust_action_noise": self.robust_action_noise,
                "robust_kl_radius": self.robust_kl_radius,
                "policy_device": str(self.policy.device),
                "policy_ensemble": {
                    "method": "equal_weight_probability_average",
                    "component_iterations": list(
                        getattr(
                            self.policy,
                            "iterations",
                            (self.policy.iteration,),
                        )
                    ),
                    "component_sha256": [
                        component.sha256
                        for component in getattr(
                            self.policy, "policies", (self.policy,)
                        )
                    ],
                },
                "session_seed": self.seed,
            },
        }
        record.update(terminal_hand_audit(self.state))
        self.results_ledger.append(record)
        self.logged_hands.add(self.hand_number)

    def _policy_stats_text(self) -> str:
        if self.policy is None or self.results_ledger is None:
            return ""
        mode = self._controller_mode()
        summary = self.results_ledger.summary_for(self.policy, mode=mode)
        hands = int(summary["hands"])
        human_rate = float(summary["human_bb_per_hand"])
        policy_rate = float(summary["policy_bb_per_hand"])
        leader = (
            "You are ahead"
            if human_rate > EPSILON
            else (
                "Policy is ahead"
                if policy_rate > EPSILON
                else "Even"
            )
        )
        return (
            f"Lifetime vs iterations {self.policy.iteration_label} "
            f"({mode}, {self.policy.sha256[:10]}): "
            f"{hands} hands | your net {float(summary['human_net_bb']):+.2f} BB | "
            f"you {human_rate:+.4f} BB/hand | "
            f"policy {policy_rate:+.4f} BB/hand | "
            f"W/L/T {summary['wins']}/{summary['losses']}/{summary['ties']} | "
            f"{leader}"
        )

    def _controller_mode(self) -> str:
        if self.policy is None:
            return "manual"
        ensemble = (
            f"+average_{len(self.policy.policies)}_policies"
            if isinstance(self.policy, AveragedHeadsUpSnapshotPolicy)
            else ""
        )
        if not self.search_enabled:
            top_k = int(getattr(self, "top_policy_actions", 0))
            top_k_suffix = f"+top_{top_k}" if top_k > 0 else ""
            guard_suffix = (
                "+no_premium_guard"
                if self.unmodified_policy_sampling and top_k > 0
                else (
                    "+unmodified_policy"
                    if self.unmodified_policy_sampling
                    else ""
                )
            )
            return f"{self.policy.mode}{ensemble}{top_k_suffix}{guard_suffix}"
        if self.search_mode == "three-player":
            return (
                f"{self.policy.mode}+three_player_root_search"
                + (
                    "_inferred_range"
                    if self.root_range_mode == "inferred"
                    else "_uniform_range"
                )
            )
        if self.search_mode == "robust":
            return (
                f"{self.policy.mode}+robust_kl_search"
                f"_eps{self.robust_action_noise:g}"
                f"_rho{self.robust_kl_radius:g}"
            )
        return f"{self.policy.mode}+family_shared_search_v4"

    def show_policy_history(self) -> None:
        if self.results_ledger is None:
            return
        window = tk.Toplevel(self.root)
        window.title("Long-term results by policy")
        window.geometry("1000x420")
        columns = (
            "iteration",
            "mode",
            "fingerprint",
            "hands",
            "record",
            "human_net",
            "human_rate",
            "policy_rate",
            "leader",
        )
        tree = ttk.Treeview(window, columns=columns, show="headings")
        headings = {
            "iteration": "Policy iteration",
            "mode": "Mode",
            "fingerprint": "SHA256",
            "hands": "Hands",
            "record": "Human W/L/T",
            "human_net": "Human net BB",
            "human_rate": "Human BB/hand",
            "policy_rate": "Policy BB/hand",
            "leader": "Ahead",
        }
        widths = {
            "iteration": 105,
            "mode": 70,
            "fingerprint": 110,
            "hands": 65,
            "record": 95,
            "human_net": 100,
            "human_rate": 110,
            "policy_rate": 110,
            "leader": 90,
        }
        for column in columns:
            tree.heading(column, text=headings[column])
            tree.column(column, width=widths[column], anchor="center")
        for summary in self.results_ledger.summaries():
            human_rate = float(summary["human_bb_per_hand"])
            leader = (
                "Human"
                if human_rate > EPSILON
                else ("Policy" if human_rate < -EPSILON else "Even")
            )
            tree.insert(
                "",
                "end",
                values=(
                    summary["policy_iteration"],
                    summary["policy_mode"],
                    str(summary["policy_sha256"])[:10],
                    summary["hands"],
                    (
                        f"{summary['wins']}/{summary['losses']}/"
                        f"{summary['ties']}"
                    ),
                    f"{float(summary['human_net_bb']):+.2f}",
                    f"{human_rate:+.4f}",
                    f"{float(summary['policy_bb_per_hand']):+.4f}",
                    leader,
                ),
            )
        scrollbar = ttk.Scrollbar(window, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        tree.pack(fill="both", expand=True, padx=10, pady=10)
        ttk.Label(
            window,
            text=f"Append-only ledger: {self.results_ledger.path}",
        ).pack(anchor="w", padx=10, pady=(0, 10))

    def _cancel_bot_job(self) -> None:
        if self.bot_job is not None:
            try:
                self.root.after_cancel(self.bot_job)
            except tk.TclError:
                pass
            self.bot_job = None

    def _invalidate_search(self) -> None:
        if self.search_poll_job is not None:
            try:
                self.root.after_cancel(self.search_poll_job)
            except tk.TclError:
                pass
            self.search_poll_job = None
        if self.search_future is not None:
            self.search_future.cancel()
        self.search_future = None
        self.search_token = None

    def close(self) -> None:
        self._cancel_bot_job()
        self._invalidate_search()
        if self.search is not None:
            self.search.close(wait_for_workers=False)
        if self.search_executor is not None:
            self.search_executor.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()

    def _schedule_bot_turn(self) -> None:
        if (
            self.policy is None
            or self.state is None
            or self.state.terminal
            or self.state.to_act is None
            or int(self.state.to_act) != self.bot_seat
            or self.bot_job is not None
            or self.search_future is not None
        ):
            return
        self.bot_job = self.root.after(self.bot_delay_ms, self._bot_turn)

    def _bot_turn(self) -> None:
        self.bot_job = None
        if (
            self.policy is None
            or self.state is None
            or self.state.terminal
            or self.state.to_act is None
            or int(self.state.to_act) != self.bot_seat
        ):
            return
        if is_first_preflop_bot_decision(self.state, self.bot_seat):
            self._apply_blueprint_bot_action(
                controller="raw_policy_first_preflop"
            )
            return
        if not self.search_enabled:
            self._apply_blueprint_bot_action(
                controller="raw_policy_search_disabled"
            )
            return
        if (
            self.search is None
            or self.search_executor is None
            or self.public_range is None
        ):
            self._apply_search_emergency_action("search service unavailable")
            return
        try:
            blueprint = self.policy.probabilities(self.env, self.state)
            self.public_range.filter_known(
                [
                    *self.state.hole[self.bot_seat],
                    *self.state.board,
                ]
            )
            token = self._state_token()
            self.search_token = token
            self.search_future = self.search_executor.submit(
                self.search.resolve,
                self.env,
                self.state,
                blueprint,
                self.public_range.snapshot(),
            )
            self.last_bot_action = (
                f"Bot P{self.bot_seat}: searching with "
                f"{self.search.workers} workers..."
            )
            self.refresh()
            self.search_poll_job = self.root.after(40, self._poll_bot_search)
        except Exception as exc:
            self._apply_search_emergency_action(
                f"search unavailable ({type(exc).__name__})"
            )

    def _state_token(self) -> tuple:
        if self.state is None:
            return ()
        return (
            self.hand_number,
            len(self.state.history),
            self.state.to_act,
            tuple(int(value) for value in self.state.stacks),
            tuple(int(value) for value in self.state.street_contrib),
        )

    def _poll_bot_search(self) -> None:
        self.search_poll_job = None
        future = self.search_future
        if future is None:
            return
        if not future.done():
            self.search_poll_job = self.root.after(40, self._poll_bot_search)
            return
        self.search_future = None
        token = self.search_token
        self.search_token = None
        if token != self._state_token():
            return
        try:
            result: PluribusSearchResult = future.result()
            self._apply_search_result(result)
        except Exception as exc:
            self._apply_search_emergency_action(
                f"search failed ({type(exc).__name__})"
            )

    def _apply_search_result(self, result: PluribusSearchResult) -> None:
        if self.state is None or self.state.terminal:
            return
        estimates = [
            estimate
            for estimate in result.candidates
            if estimate.samples > 0
            and math.isfinite(estimate.expected_final_payoff_bb)
        ]
        effective_probabilities = [
            float(estimate.strategy_probability)
            for estimate in estimates
        ]
        choice = result.choice
        ranked = sorted(
            (
                {
                    "label": estimate.action.label,
                    "estimated_ev_bb": estimate.expected_final_payoff_bb,
                    "standard_error_bb": estimate.standard_error_bb,
                    "ci95_low_bb": estimate.ci95_low_bb,
                    "ci95_high_bb": estimate.ci95_high_bb,
                    "samples": estimate.samples,
                    "strategy_probability": effective_probability,
                    "blueprint_probability": (
                        estimate.action.blueprint_prior
                    ),
                    "validation_ev_bb": estimate.validation_ev_bb,
                    "validation_ci95_low_bb": estimate.validation_ci95_low_bb,
                    "validation_ci95_high_bb": estimate.validation_ci95_high_bb,
                    "validation_samples": estimate.validation_samples,
                    "statistically_dominated": estimate.statistically_dominated,
                    "safety_pruned": estimate.safety_pruned,
                    "chosen": estimate.action == choice,
                }
                for estimate, effective_probability in zip(
                    estimates,
                    effective_probabilities,
                )
            ),
            key=lambda row: float(row["strategy_probability"]),
            reverse=True,
        )
        chosen_row = next(
            (row for row in ranked if bool(row["chosen"])),
            None,
        )
        self.search_decision_log.append(
            {
                "history_index": len(self.state.history),
                "search_mode": self.search_mode,
                "street": int(self.state.street),
                "bot_seat": self.bot_seat,
                "button": int(self.state.button),
                "board_card_ids": [
                    int(card) for card in self.state.board
                ],
                "board": [
                    card_to_string(int(card))
                    for card in self.state.board
                ],
                "policy_hole_card_ids": [
                    int(card) for card in self.state.hole[self.bot_seat]
                ],
                "policy_hole": [
                    card_to_string(int(card))
                    for card in self.state.hole[self.bot_seat]
                ],
                "pot_chips": int(self.state.pot),
                "stacks_chips": [
                    int(value) for value in self.state.stacks
                ],
                "street_contributions_chips": [
                    int(value) for value in self.state.street_contrib
                ],
                "current_bet_chips": int(self.state.current_bet),
                "to_call_chips": int(
                    self.env.amount_to_call(
                        self.state, self.bot_seat
                    )
                ),
                "legal_actions": [
                    {
                        "action": int(action),
                        "name": ACTION_NAMES[int(action)],
                        "target_chips": int(
                            self.env.action_target(self.state, int(action))
                        ),
                    }
                    for action in self.env.legal_actions(self.state)
                ],
                "chosen": choice.label,
                "chosen_action": (
                    int(choice.action)
                    if choice.action is not None
                    else None
                ),
                "chosen_kind": choice.kind,
                "chosen_raise_to": (
                    int(choice.raise_to)
                    if choice.raise_to is not None
                    else None
                ),
                "chosen_estimated_ev_bb": (
                    None
                    if chosen_row is None
                    else float(chosen_row["estimated_ev_bb"])
                ),
                "elapsed_seconds": float(result.elapsed_ms) / 1000.0,
                "cfr_iterations": int(result.cfr_iterations),
                "terminal_rollouts": int(result.terminal_rollouts),
                "workers_responded": int(result.workers_responded),
                "range_combos": int(result.range_combos),
                "range_effective_sample_size": float(
                    result.range_effective_sample_size
                ),
                "range_updates": int(result.range_updates),
                "native_backend": bool(result.native_backend),
                "converged": bool(result.converged),
                "used_blueprint_fallback": bool(
                    result.used_blueprint_fallback
                ),
                "convergence_reason": result.convergence_reason,
                "validation_samples": int(result.validation_samples),
                "worker_agreement": float(result.worker_agreement),
                "strategy_gap": float(result.strategy_gap),
                "candidates": ranked,
            }
        )
        try:
            if choice.kind == "abstract":
                self.state = self.env.step(self.state, int(choice.action))
            elif choice.kind == "raise_to":
                self.state = self.env.step_exact(
                    self.state, "raise_to", int(choice.raise_to)
                )
            else:
                self.state = self.env.step_exact(self.state, choice.kind)
        except Exception as exc:
            self._apply_search_emergency_action(
                f"search action became illegal ({type(exc).__name__})"
            )
            return
        self.last_bot_action = (
            f"Bot P{self.bot_seat}: {choice.label} | search "
            f"{result.elapsed_ms / 1000.0:.2f}s, "
            f"{result.cfr_iterations} CFR iterations, "
            f"{result.terminal_rollouts} continuation rollouts, "
            f"{result.workers_responded}/{self.search.workers} workers, "
            f"range ESS {result.range_effective_sample_size:.0f}, "
            f"agreement {result.worker_agreement:.0%}, "
            f"{'native' if result.native_backend else 'python'} backend, "
            f"{'validated' if result.converged else 'limited-sample'} "
            "search-owned solve"
        )
        self._after_action()

    def _apply_search_emergency_action(self, reason: str) -> None:
        """Keep search mode independent from the neural policy on failures."""

        if self.state is None or self.state.terminal:
            return
        legal = [int(action) for action in self.env.legal_actions(self.state)]
        preferred_names = ("check", "call", "fold")
        selected = next(
            (
                action
                for name in preferred_names
                for action in legal
                if ACTION_NAMES[action] == name
            ),
            None,
        )
        if selected is None:
            raise RuntimeError("search emergency path has no passive legal action")
        self.state = self.env.step(self.state, int(selected))
        self.last_bot_action = (
            f"Bot P{self.bot_seat}: {ACTION_NAMES[selected].replace('_', ' ')} "
            f"| deterministic search emergency ({reason}); policy not used"
        )
        self._after_action()

    def _apply_blueprint_bot_action(
        self,
        *,
        controller: str = "raw_policy",
    ) -> None:
        if self.state is None or self.state.terminal:
            return
        try:
            legal = [int(value) for value in self.env.legal_actions(self.state)]
            if isinstance(self.policy, AveragedHeadsUpSnapshotPolicy):
                raw_probabilities, component_probabilities = (
                    self.policy.probabilities_with_components(
                        self.env, self.state
                    )
                )
            else:
                raw_probabilities = self.policy.probabilities(
                    self.env, self.state
                )
                component_probabilities = (raw_probabilities,)
            premium_never_fold = should_apply_premium_preflop_guard(
                self.state,
                self.bot_seat,
                unmodified_policy_sampling=self.unmodified_policy_sampling,
            )
            probabilities = (
                probabilities_without_fold(raw_probabilities, legal)
                if premium_never_fold
                else raw_probabilities
            )
            probabilities = probabilities_from_top_k(
                probabilities,
                legal,
                self.top_policy_actions,
            )
            if self.policy.mode == "argmax":
                action = int(torch.argmax(probabilities).item())
            else:
                action = int(
                    torch.multinomial(
                        probabilities,
                        1,
                        generator=self.policy.generator,
                    ).item()
                )
            self.policy_decision_log.append(
                {
                    "history_index": len(self.state.history),
                    "street": int(self.state.street),
                    "bot_seat": self.bot_seat,
                    "controller": controller,
                    "search_bypassed": True,
                    "policy_mode": self.policy.mode,
                    "chosen_action": int(action),
                    "chosen": ACTION_NAMES[action].replace("_", " "),
                    "chosen_probability": float(probabilities[action]),
                    "premium_preflop_never_fold": premium_never_fold,
                    "top_policy_actions": self.top_policy_actions,
                    "raw_policy_probabilities": {
                        ACTION_NAMES[value]: float(raw_probabilities[value])
                        for value in legal
                    },
                    "component_policy_probabilities": [
                        {
                            "iteration": int(component.iteration),
                            "probabilities": {
                                ACTION_NAMES[value]: float(values[value])
                                for value in legal
                            },
                        }
                        for component, values in zip(
                            getattr(self.policy, "policies", (self.policy,)),
                            component_probabilities,
                        )
                    ],
                    "legal_policy_probabilities": {
                        ACTION_NAMES[value]: float(probabilities[value])
                        for value in legal
                    },
                }
            )
            prefix = self.last_bot_action + " | " if self.last_bot_action else ""
            self.last_bot_action = (
                f"{prefix}Bot P{self.bot_seat}: "
                f"{ACTION_NAMES[action].replace('_', ' ')} "
                f"({float(probabilities[action]):.1%}) | "
                + (
                    "raw policy first preflop; search bypassed"
                    if controller == "raw_policy_first_preflop"
                    else "raw policy"
                )
                + (
                    "; premium-hand fold removed"
                    if premium_never_fold
                    else ""
                )
            )
            self.state = self.env.step(self.state, action)
        except Exception as exc:
            messagebox.showerror("Policy bot failed", str(exc))
            return
        self._after_action()

    def refresh(self) -> None:
        if self.state is None:
            return
        facts = state_facts(self.env, self.state)
        button = int(self.state.button)
        self.status.configure(
            text=(
                f"Hand {self.hand_number}  |  {facts['street'].title()}  |  "
                f"Button/SB P{button}  |  BB P{1 - button}  |  "
                f"Actor {facts['actor']}"
            )
        )
        self.diagnostics.configure(
            text=(
                f"pot={facts['pot']}   current_bet={facts['current_bet']}   "
                f"min_raise_increment={facts['minimum_raise_increment']}   "
                f"min_raise_to={facts['minimum_raise_to']}\n"
                f"to_call={facts['to_call']}   max_raise_to={facts['maximum_raise_to']}   "
                f"stacks=P0 {format_chips(self.state.stacks[0])}, "
                f"P1 {format_chips(self.state.stacks[1])}   "
                f"street_in=P0 {format_chips(self.state.street_contrib[0])}, "
                f"P1 {format_chips(self.state.street_contrib[1])}"
                + (
                    f"\n{self.last_bot_action}"
                    if self.last_bot_action
                    else ""
                )
            )
        )
        self.policy_stats.configure(text=self._policy_stats_text())
        self._render_range_plot()
        self._render_history()
        self._render_controls()
        self.draw_table()
        self._schedule_bot_turn()

    def _render_range_plot(self) -> None:
        canvas = self.range_plot_canvas
        canvas.delete("all")
        if not self.range_log:
            self.range_plot_title.configure(
                text="Inferred human range — waiting for your action"
            )
            canvas.create_text(
                190,
                155,
                text=(
                    "Your blocker-compatible range will appear here\n"
                    "after the policy observes your first action."
                ),
                fill=self.MUTED,
                justify="center",
                font=("Segoe UI", 10),
            )
            return
        latest_index = max(self.range_log)
        row = self.range_log[latest_index]
        probabilities = {
            str(label): float(value)
            for label, value in row["class_probabilities"].items()
        }
        maximum = max(probabilities.values(), default=0.0)
        source_label = (
            "Policy card-head human range"
            if row.get("source") == "policy_card_head"
            else "Action-inferred human range"
        )
        self.range_plot_title.configure(
            text=(
                f"{source_label} | "
                f"ESS {float(row['effective_sample_size']):.0f}, "
                f"{int(row['updates'])} update"
                f"{'' if int(row['updates']) == 1 else 's'}"
            )
        )
        ranks = list(reversed(RANK_SYMBOLS))
        cell = 24
        origin_x = 7
        origin_y = 4
        for row_index, row_rank in enumerate(ranks):
            for column_index, column_rank in enumerate(ranks):
                if row_index == column_index:
                    label = row_rank + row_rank
                elif row_index < column_index:
                    label = row_rank + column_rank + "s"
                else:
                    label = column_rank + row_rank + "o"
                probability = probabilities.get(label, 0.0)
                fill = range_probability_color(probability, maximum)
                intensity = (
                    0.0
                    if maximum <= 0.0
                    else math.sqrt(probability / maximum)
                )
                text_color = "#101820" if intensity >= 0.72 else "#ecf0f1"
                x0 = origin_x + column_index * cell
                y0 = origin_y + row_index * cell
                canvas.create_rectangle(
                    x0,
                    y0,
                    x0 + cell - 1,
                    y0 + cell - 1,
                    fill=fill,
                    outline="#34495e",
                    width=1,
                )
                canvas.create_text(
                    x0 + cell / 2,
                    y0 + 8,
                    text=label,
                    fill=text_color,
                    font=("Consolas", 7, "bold"),
                )
                percentage = 100.0 * probability
                canvas.create_text(
                    x0 + cell / 2,
                    y0 + 17,
                    text=(
                        f"{percentage:.1f}"
                        if percentage >= 0.1
                        else "<.1"
                    ),
                    fill=text_color,
                    font=("Consolas", 6),
                )
        legend_y = origin_y + 13 * cell + 4
        legend_x = origin_x
        legend_width = 220
        steps = 22
        for index in range(steps):
            x0 = legend_x + index * legend_width / steps
            probability = maximum * index / max(1, steps - 1)
            canvas.create_rectangle(
                x0,
                legend_y,
                legend_x + (index + 1) * legend_width / steps + 1,
                legend_y + 8,
                fill=range_probability_color(probability, maximum),
                outline="",
            )
        canvas.create_text(
            legend_x,
            legend_y + 17,
            text="0%",
            anchor="w",
            fill=self.MUTED,
            font=("Consolas", 7),
        )
        canvas.create_text(
            legend_x + legend_width,
            legend_y + 17,
            text=f"{100.0 * maximum:.1f}% max class probability",
            anchor="e",
            fill=self.MUTED,
            font=("Consolas", 7),
        )

    def _render_history(self) -> None:
        lines = [
            f"Hand {self.hand_number}; button/SB P{self.state.button}",
            (
                f"Starting stacks: P0 {format_chips(self.state.initial_stacks[0])}, "
                f"P1 {format_chips(self.state.initial_stacks[1])}"
            ),
            "",
        ]
        policy_shift_spans: list[tuple[int, int, int, str]] = []
        decisions = {
            int(row["history_index"]): row
            for row in self.search_decision_log
        }
        policy_decisions = {
            int(row["history_index"]): row
            for row in self.policy_decision_log
        }
        for history_index, event in enumerate(self.state.history):
            lines.append(event_summary(event))
            policy_decision = policy_decisions.get(history_index)
            if policy_decision is not None:
                policy_label = (
                    "RAW POLICY FIRST PREFLOP ACTION"
                    if policy_decision["controller"]
                    == "raw_policy_first_preflop"
                    else "RAW POLICY ACTION"
                )
                lines.append(
                    f"  {policy_label}: "
                    f"chose {policy_decision['chosen']} | "
                    f"policy probability "
                    f"{100.0 * float(policy_decision['chosen_probability']):.1f}% "
                    "| search bypassed"
                    + (
                        " | PREMIUM PREFLOP SAFETY: fold removed"
                        if policy_decision.get(
                            "premium_preflop_never_fold",
                            False,
                        )
                        else ""
                    )
                )
            range_row = self.range_log.get(history_index)
            if range_row is not None:
                top_text = ", ".join(
                    f"{label} {100.0 * weight:.1f}%"
                    for label, weight in range_row["top_classes"]
                )
                lines.append(
                    "  INFERRED HUMAN RANGE "
                    + (
                        f"(action-noise Bayesian, epsilon "
                        f"{self.robust_action_noise:.2f}, ESS "
                        if self.search_mode == "robust"
                        else "(tempered + 25% uniform, ESS "
                    )
                    + f"{range_row['effective_sample_size']:.0f}, "
                    f"{range_row['updates']} updates):"
                )
                lines.append(f"    top classes: {top_text}")
            decision = decisions.get(history_index)
            if decision is None:
                continue
            chosen_ev = decision.get("chosen_estimated_ev_bb")
            chosen_text = (
                "n/a"
                if chosen_ev is None
                else f"{float(chosen_ev):+.3f} BB"
            )
            lines.append(
                (
                    "  THREE-PLAYER-STYLE ROOT SEARCH (bot perspective):"
                    if decision.get("search_mode") == "three-player"
                    else (
                        "  KL-ROBUST ROOT SEARCH (bot perspective):"
                        if decision.get("search_mode") == "robust"
                        else "  DEPTH-LIMITED CFR SEARCH (bot perspective):"
                    )
                )
            )
            lines.append(
                f"    chose {decision['chosen']} | EV {chosen_text} | "
                f"{float(decision['elapsed_seconds']):.2f}s | "
                f"{int(decision['cfr_iterations'])} CFR iterations | "
                f"{int(decision['terminal_rollouts'])} continuation rollouts"
            )
            lines.append(
                f"    public range {int(decision['range_combos'])} combos | "
                f"ESS {float(decision['range_effective_sample_size']):.1f} | "
                f"{int(decision['range_updates'])} Bayesian action updates"
            )
            lines.append(
                "    "
                f"{'native' if decision['native_backend'] else 'python'} "
                f"backend | holdout n={int(decision['validation_samples'])} | "
                f"{'CONVERGED' if decision['converged'] else 'LIMITED SEARCH'}: "
                f"{decision['convergence_reason']}"
            )
            for candidate in decision["candidates"]:
                marker = "*" if candidate["chosen"] else " "
                validation = ""
                if math.isfinite(float(candidate["validation_ev_bb"])):
                    validation = (
                        f" | holdout {float(candidate['validation_ev_bb']):+.3f} "
                        f"[{float(candidate['validation_ci95_low_bb']):+.3f},"
                        f"{float(candidate['validation_ci95_high_bb']):+.3f}] "
                        f"n={int(candidate['validation_samples'])}"
                    )
                if candidate["safety_pruned"]:
                    validation += " | SAFETY PRUNED"
                elif candidate["statistically_dominated"]:
                    validation += " | DOMINATED"
                shift_text, shift_tag = format_policy_shift(
                    float(candidate["strategy_probability"]),
                    float(candidate["blueprint_probability"]),
                )
                candidate_line = (
                    f"    {marker} {candidate['label']:<20} "
                    f"strategy {100.0 * float(candidate['strategy_probability']):5.1f}% | "
                    f"raw {100.0 * float(candidate['blueprint_probability']):5.1f}% | "
                    f"change {shift_text} | "
                    f"EV {float(candidate['estimated_ev_bb']):+7.3f} BB | "
                    f"95% [{float(candidate['ci95_low_bb']):+.3f}, "
                    f"{float(candidate['ci95_high_bb']):+.3f}] | "
                    f"n={int(candidate['samples'])}{validation}"
                )
                lines.append(candidate_line)
                shift_start = candidate_line.index(shift_text)
                policy_shift_spans.append(
                    (
                        len(lines),
                        shift_start,
                        shift_start + len(shift_text),
                        shift_tag,
                    )
                )
        if self.state.terminal:
            lines.extend(("", "TERMINAL"))
            payoffs = getattr(self.state, "payoffs", None)
            payouts = getattr(self.state, "payouts", None)
            winners = tuple(getattr(self.state, "winners", ()))
            uncalled = getattr(self.state, "uncalled_returns", None)
            if uncalled is not None and any(float(value) > EPSILON for value in uncalled):
                lines.append(
                    "Uncalled returns: "
                    + ", ".join(
                        f"P{seat} {format_chips(value)}"
                        for seat, value in enumerate(uncalled)
                    )
                )
            if payouts is not None:
                lines.append(
                    "Payouts: "
                    + ", ".join(
                        f"P{seat} {format_chips(value)}"
                        for seat, value in enumerate(payouts)
                    )
                )
            if payoffs is not None:
                lines.append(
                    "Payoffs: "
                    + ", ".join(
                        f"P{seat} {float(value):+g}"
                        for seat, value in enumerate(payoffs)
                    )
                )
            lines.append(
                "Winner(s): "
                + (", ".join(f"P{seat}" for seat in winners) if winners else "none")
            )
        self.history_text.configure(state="normal")
        self.history_text.delete("1.0", "end")
        self.history_text.insert("end", "\n".join(lines))
        for line_number, start, end, tag in policy_shift_spans:
            self.history_text.tag_add(
                tag,
                f"{line_number}.{start}",
                f"{line_number}.{end}",
            )
        self.history_text.see("end")
        self.history_text.configure(state="disabled")

    def _clear_action_buttons(self) -> None:
        for button in self.action_buttons:
            button.destroy()
        self.action_buttons.clear()

    def _render_controls(self) -> None:
        self._clear_action_buttons()
        if self.state.terminal:
            winners = ", ".join(
                f"P{seat}" for seat in getattr(self.state, "winners", ())
            )
            payoffs = getattr(self.state, "payoffs", (0.0, 0.0))
            self.prompt.configure(
                text=(
                    f"Hand complete - winner(s) {winners or 'none'}; "
                    f"payoffs P0 {float(payoffs[0]):+g}, P1 {float(payoffs[1]):+g}"
                )
            )
            can_continue = min(self.session_stacks) > EPSILON
            self.next_hand_button.configure(
                state="normal" if can_continue else "disabled"
            )
            self.reveal_bot_button.configure(
                state=(
                    "normal"
                    if self.policy is not None and not self.bot_cards_revealed
                    else "disabled"
                )
            )
            self.exact_raise_button.configure(state="disabled")
            self.exact_raise_entry.configure(state="disabled")
            self.exact_range.configure(
                text=(
                    "Click Next hand to rotate the button."
                    if can_continue
                    else "Match complete; reset to continue."
                )
            )
            return

        actor = int(self.state.to_act)
        legal = self.env.legal_actions(self.state)
        to_call = float(self.env.amount_to_call(self.state, actor))
        if self.policy is not None and actor == self.bot_seat:
            self.prompt.configure(
                text=(
                    f"Policy iterations {self.policy.iteration_label} are thinking "
                    f"for Player {actor}..."
                )
            )
            self.exact_raise_var.set("")
            self.exact_range.configure(text="Bot turn")
            self.exact_raise_entry.configure(state="disabled")
            self.exact_raise_button.configure(state="disabled")
            self.next_hand_button.configure(state="disabled")
            self.reveal_bot_button.configure(state="disabled")
            return
        self.prompt.configure(
            text=(
                f"Choose for Player {actor} - "
                + (
                    f"{format_chips(to_call)} to call"
                    if to_call > EPSILON
                    else "may check or bet"
                )
            )
        )
        for index, action in enumerate(legal):
            button = ttk.Button(
                self.actions_frame,
                text=fixed_action_label(self.env, self.state, action),
                command=lambda selected=action: self.apply_fixed_action(selected),
                width=19,
            )
            button.grid(
                row=index // 5,
                column=index % 5,
                padx=(0, 6),
                pady=(0, 5),
                sticky="ew",
            )
            self.action_buttons.append(button)
        for column in range(5):
            self.actions_frame.grid_columnconfigure(column, weight=1)

        maximum = (
            float(self.state.street_contrib[actor]) + float(self.state.stacks[actor])
        )
        minimum_increment = float(
            getattr(
                self.state,
                "min_raise",
                getattr(
                    self.state,
                    "minimum_raise",
                    getattr(self.state, "min_raise_increment", 0.0),
                ),
            )
        )
        minimum = float(self.state.current_bet) + minimum_increment
        raise_rights = list(getattr(self.state, "raise_rights", [True, True]))
        can_raise = (
            bool(raise_rights[actor])
            and maximum > float(self.state.current_bet) + EPSILON
        )
        if can_raise:
            suggested = min(maximum, minimum)
            if abs(suggested - round(suggested)) <= EPSILON:
                self.exact_raise_var.set(str(int(round(suggested))))
            else:
                self.exact_raise_var.set(str(math.ceil(suggested)))
            if maximum + EPSILON < minimum:
                range_text = (
                    f"short all-in only: exact target {format_chips(maximum)}"
                )
            else:
                range_text = (
                    f"full raise range {format_chips(minimum)}.."
                    f"{format_chips(maximum)}; short all-in validated by engine"
                )
            self.exact_range.configure(text=range_text)
            self.exact_raise_entry.configure(state="normal")
            self.exact_raise_button.configure(state="normal")
        else:
            self.exact_raise_var.set("")
            self.exact_range.configure(text="raising is not legal")
            self.exact_raise_entry.configure(state="disabled")
            self.exact_raise_button.configure(state="disabled")
        self.next_hand_button.configure(state="disabled")
        self.reveal_bot_button.configure(state="disabled")

    def draw_table(self) -> None:
        if self.state is None:
            return
        canvas = self.canvas
        canvas.delete("all")
        width = max(canvas.winfo_width(), 620)
        height = max(canvas.winfo_height(), 460)
        canvas.create_oval(
            45,
            35,
            width - 45,
            height - 35,
            fill=self.TABLE,
            outline="#c8a951",
            width=5,
        )

        board_y = height * 0.43
        board_width = 5 * 62 + 4 * 8
        board_x = width / 2 - board_width / 2
        for index in range(5):
            card = self.state.board[index] if index < len(self.state.board) else None
            self._draw_card(board_x + index * 70, board_y, card)
        canvas.create_text(
            width / 2,
            board_y - 28,
            text=f"POT  {format_chips(self.state.pot)}",
            fill=self.GOLD,
            font=("Segoe UI", 13, "bold"),
        )

        positions = ((width / 2, height - 115), (width / 2, 92))
        for seat, (x, y) in enumerate(positions):
            self._draw_player(seat, x, y)

    def _draw_player(self, seat: int, x: float, y: float) -> None:
        active = not self.state.terminal and self.state.to_act == seat
        outline = self.GOLD if active else "#8f9aa3"
        fill = "#253746" if not self.state.folded[seat] else "#3d4449"
        self.canvas.create_rectangle(
            x - 175,
            y - 47,
            x + 175,
            y + 68,
            fill=fill,
            outline=outline,
            width=3,
        )
        markers: list[str] = []
        if seat == self.state.button:
            markers.extend(("BTN", "SB"))
        else:
            markers.append("BB")
        if self.state.folded[seat]:
            markers.append("FOLDED")
        elif self.state.all_in[seat]:
            markers.append("ALL-IN")
        if self.policy is None:
            seat_name = SEAT_NAMES[seat]
        else:
            seat_name = "You" if seat == self.human_seat else "Policy bot"
        self.canvas.create_text(
            x - 72,
            y - 28,
            text=f"{seat_name} (P{seat})  {' / '.join(markers)}",
            fill=self.TEXT,
            font=("Segoe UI", 10, "bold"),
        )
        self.canvas.create_text(
            x - 72,
            y - 7,
            text=(
                f"Stack {format_chips(self.state.stacks[seat])}   "
                f"In {format_chips(self.state.total_contrib[seat])}   "
                f"Street {format_chips(self.state.street_contrib[seat])}"
            ),
            fill="#d5dbdb",
            font=("Segoe UI", 9),
        )
        reveal_cards = (
            self.policy is None
            or seat == self.human_seat
            or (
                seat == self.bot_seat
                and self.state.terminal
                and self.bot_cards_revealed
            )
        )
        for offset, card in enumerate(self.state.hole[seat]):
            self._draw_card(
                x + 55 + offset * 53,
                y - 26,
                card if reveal_cards else None,
                small=True,
            )
        self._draw_player_hud(seat, x + 181, y - 43)

    def _draw_player_hud(self, seat: int, x: float, y: float) -> None:
        hud_player = (
            seat
            if self.policy is None
            else 0 if seat == self.human_seat else 1
        )
        stats = calculate_player_hud(
            self._hud_hands_with_current(),
            hud_player,
        )
        width, height = 124, 102
        self.canvas.create_rectangle(
            x,
            y,
            x + width,
            y + height,
            fill="#111b24",
            outline="#d2b24c",
            width=2,
        )
        self.canvas.create_rectangle(
            x + 2,
            y + 2,
            x + width - 2,
            y + 24,
            fill="#263746",
            outline="",
        )
        self.canvas.create_text(
            x + width / 2,
            y + 13,
            text=f"PLAYER HUD   H {int(stats['hands'])}",
            fill=self.GOLD,
            font=("Segoe UI", 8, "bold"),
        )
        rows = (
            ("VPIP", f"{float(stats['vpip_pct']):.1f}%", "#60d394"),
            ("ATS", f"{float(stats['ats_pct']):.1f}%", "#69b7ff"),
            ("AF", format_hud_af(float(stats["af"])), "#ffcc66"),
        )
        for row, (label, value, color) in enumerate(rows):
            row_y = y + 37 + row * 22
            self.canvas.create_text(
                x + 12,
                row_y,
                text=label,
                anchor="w",
                fill="#d5dbdb",
                font=("Consolas", 9, "bold"),
            )
            self.canvas.create_text(
                x + width - 12,
                row_y,
                text=value,
                anchor="e",
                fill=color,
                font=("Consolas", 10, "bold"),
            )

    def _draw_card(
        self,
        x: float,
        y: float,
        card: int | None,
        *,
        small: bool = False,
    ) -> None:
        width, height = (45, 56) if small else (62, 80)
        if card is None:
            self.canvas.create_rectangle(
                x,
                y,
                x + width,
                y + height,
                fill="#1a5837",
                outline="#5d8d72",
            )
            return
        compact = card_to_string(int(card))
        suit = {"c": "clubs", "d": "diamonds", "h": "hearts", "s": "spades"}[
            compact[1]
        ]
        symbol = {
            "clubs": "\u2663",
            "diamonds": "\u2666",
            "hearts": "\u2665",
            "spades": "\u2660",
        }[suit]
        color = "#c62828" if compact[1] in "dh" else "#111111"
        self.canvas.create_rectangle(
            x,
            y,
            x + width,
            y + height,
            fill="white",
            outline="#d0d3d4",
            width=2,
        )
        self.canvas.create_text(
            x + width / 2,
            y + height / 2,
            text=compact[0] + symbol,
            fill=color,
            font=("Segoe UI", 13 if small else 20, "bold"),
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play heads-up manually or against a policy snapshot"
    )
    parser.add_argument("--stack", type=int, default=200, help="starting chips per seat")
    parser.add_argument("--sb", type=int, default=1, help="small blind")
    parser.add_argument("--bb", type=int, default=2, help="big blind")
    parser.add_argument("--seed", type=int, default=None, help="optional deal seed")
    parser.add_argument(
        "--button",
        type=int,
        choices=(0, 1),
        default=0,
        help="button/SB for the first hand",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        help="deployable heads-up policy snapshot; omit for manual two-seat mode",
    )
    parser.add_argument(
        "--policy-secondary",
        type=Path,
        action="append",
        default=[],
        help=(
            "additional compatible HU snapshot; repeat this option to average "
            "three or more policies with equal weight before sampling"
        ),
    )
    parser.add_argument(
        "--human-seat",
        type=int,
        choices=(0, 1),
        default=0,
        help="seat controlled by the human when --policy is supplied",
    )
    parser.add_argument(
        "--policy-mode",
        choices=("sample", "argmax"),
        default="sample",
        help="sample the mixed strategy or choose its highest-probability action",
    )
    parser.add_argument(
        "--policy-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="policy inference device; auto uses CUDA when available",
    )
    parser.add_argument(
        "--bot-delay-ms",
        type=int,
        default=450,
        help="visual delay before automatic bot actions",
    )
    parser.add_argument(
        "--results-log",
        type=Path,
        default=DEFAULT_RESULTS_LOG,
        help="append-only human-vs-policy hand ledger",
    )
    parser.add_argument(
        "--no-search",
        action="store_true",
        help="disable multiprocess local search and use the raw policy",
    )
    parser.add_argument(
        "--unmodified-policy-sampling",
        action="store_true",
        help=(
            "disable the GUI premium-preflop fold override; any explicitly "
            "requested --top-policy-actions pruning still applies"
        ),
    )
    parser.add_argument(
        "--top-policy-actions",
        type=int,
        default=0,
        help=(
            "keep and renormalize only the top K legal policy actions before "
            "sampling; 0 keeps the complete distribution"
        ),
    )
    parser.add_argument(
        "--search-mode",
        choices=("three-player", "robust", "cfr"),
        default="three-player",
        help=(
            "three-player uses the original anchored root resolver; "
            "robust uses action-noise Bayesian ranges and KL-worst-case EV; "
            "cfr uses the current family-shared depth-limited solver"
        ),
    )
    parser.add_argument(
        "--root-range-mode",
        choices=("uniform", "inferred"),
        default="inferred",
        help=(
            "hidden-hand sampling for three-player root search; inferred uses "
            "tempered policy-likelihood Bayesian updates"
        ),
    )
    parser.add_argument(
        "--search-workers",
        type=int,
        default=0,
        help=(
            "persistent search workers; 0 auto-selects a safe physical-core "
            f"count (currently {recommended_search_workers()})"
        ),
    )
    parser.add_argument(
        "--search-budget-seconds",
        type=float,
        default=6.0,
        help="hard coordinator search budget, at most 12 seconds",
    )
    parser.add_argument(
        "--root-search-ms",
        type=int,
        default=10_000,
        help="three-player-style root search budget (default: 10000 ms)",
    )
    parser.add_argument(
        "--root-search-rollouts",
        type=int,
        default=150_000,
        help="three-player-style root rollout cap (default: 150000)",
    )
    parser.add_argument(
        "--root-search-batch-iterations",
        type=int,
        default=3072,
        help="paired CFR samples evaluated together (default: 3072)",
    )
    parser.add_argument(
        "--root-blueprint-weight",
        type=float,
        default=0.65,
        help=(
            "raw-policy anchor for three-player root search "
            "(default: 0.65, leaving 0.35 search weight)"
        ),
    )
    parser.add_argument(
        "--root-min-strategy-probability",
        type=float,
        default=0.0,
        help=(
            "three-player root actions below this final strategy probability "
            "are excluded from GUI sampling (default: 0)"
        ),
    )
    parser.add_argument(
        "--robust-action-noise",
        type=float,
        default=0.10,
        help=(
            "generic action-tremble mass used only by --search-mode robust "
            "(default: 0.10)"
        ),
    )
    parser.add_argument(
        "--robust-kl-radius",
        type=float,
        default=0.20,
        help=(
            "KL ambiguity radius used only by --search-mode robust "
            "(default: 0.20)"
        ),
    )
    args = parser.parse_args(argv)
    if args.stack <= 0:
        parser.error("--stack must be positive")
    if not (0 < args.sb < args.bb):
        parser.error("blinds must satisfy 0 < --sb < --bb")
    if args.bot_delay_ms < 0:
        parser.error("--bot-delay-ms cannot be negative")
    if args.top_policy_actions < 0:
        parser.error("--top-policy-actions cannot be negative")
    if args.policy_secondary and args.policy is None:
        parser.error("--policy-secondary requires --policy")
    if args.policy_secondary and not args.no_search:
        parser.error("multi-policy averaging currently requires --no-search")
    if args.search_workers < 0:
        parser.error("--search-workers cannot be negative")
    if args.root_search_ms <= 0:
        parser.error("--root-search-ms must be positive")
    if args.root_search_rollouts <= 0:
        parser.error("--root-search-rollouts must be positive")
    if args.root_search_batch_iterations <= 0:
        parser.error("--root-search-batch-iterations must be positive")
    if not 0.0 <= args.root_blueprint_weight <= 1.0:
        parser.error("--root-blueprint-weight must be in [0, 1]")
    if not 0.0 <= args.root_min_strategy_probability < 1.0:
        parser.error(
            "--root-min-strategy-probability must be in [0, 1)"
        )
    if not 0.0 <= args.robust_action_noise < 1.0:
        parser.error("--robust-action-noise must be in [0, 1)")
    if args.robust_kl_radius < 0.0:
        parser.error("--robust-kl-radius must be nonnegative")
    if not 0.1 <= args.search_budget_seconds <= 12.0:
        parser.error("--search-budget-seconds must be in [0.1, 12.0]")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        primary_policy = (
            HeadsUpSnapshotPolicy(
                args.policy,
                mode=args.policy_mode,
                device=args.policy_device,
                seed=args.seed,
            )
            if args.policy is not None
            else None
        )
        policy = primary_policy
        if args.policy_secondary:
            policy = AveragedHeadsUpSnapshotPolicy(
                [
                    primary_policy,
                    *[
                        HeadsUpSnapshotPolicy(
                            path,
                            mode=args.policy_mode,
                            device=args.policy_device,
                            seed=args.seed,
                        )
                        for path in args.policy_secondary
                    ],
                ]
            )
        root = tk.Tk()
        HeadsUpManualGUI(
            root,
            starting_stack=args.stack,
            small_blind=args.sb,
            big_blind=args.bb,
            seed=args.seed,
            first_button=args.button,
            policy=policy,
            human_seat=args.human_seat,
            bot_delay_ms=args.bot_delay_ms,
            results_log=args.results_log,
            search_enabled=not args.no_search,
            search_mode=args.search_mode,
            root_range_mode=args.root_range_mode,
            search_workers=args.search_workers,
            search_budget_seconds=args.search_budget_seconds,
            root_search_ms=args.root_search_ms,
            root_search_rollouts=args.root_search_rollouts,
            root_search_batch_iterations=args.root_search_batch_iterations,
            root_blueprint_weight=args.root_blueprint_weight,
            root_min_strategy_probability=(
                args.root_min_strategy_probability
            ),
            robust_action_noise=args.robust_action_noise,
            robust_kl_radius=args.robust_kl_radius,
            unmodified_policy_sampling=args.unmodified_policy_sampling,
            top_policy_actions=args.top_policy_actions,
        )
        root.mainloop()
        return 0
    except Exception as exc:
        print(f"Could not start heads-up GUI: {exc}", file=sys.stderr)
        try:
            messagebox.showerror("Could not start heads-up GUI", str(exc))
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
