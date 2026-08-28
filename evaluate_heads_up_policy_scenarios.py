"""Evaluate deployable HU policies on controlled, reproducible poker decisions.

This suite is a strategy fingerprint, not an equilibrium oracle.  Every case is
legally replayed through the exact heads-up engine, evaluated for both physical
hero seats, and reported as a complete ten-action probability distribution.
The small set of ordering expectations are deliberately broad behavioral
sanity checks (for example, a set should continue more often than air when
facing a bet); passing them does not prove that a strategy is optimal.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import torch

from heads_up_analysis import (
    DecisionScenario,
    build_decision_state,
    postflop_scenarios,
    preflop_scenarios,
)
from heads_up_engine import (
    ACTION_CALL,
    ACTION_CHECK,
    ACTION_FOLD,
    ACTION_NAMES,
    NUM_ACTIONS,
    STREET_NAMES,
    card_to_string,
)
from heads_up_models import (
    build_action_descriptors,
    encode_information_state,
    masked_softmax,
)
from heads_up_native import HeadsUpHoldemEngine
from heads_up_production import load_policy_snapshot


DEFAULT_POLICIES = (
    Path("artifacts/heads_up_v4_paper3x/snapshots/policy_00000350.pt"),
    Path("artifacts/heads_up_v4_paper3x/snapshots/policy_00000725.pt"),
    Path("artifacts/heads_up_v4_paper3x/snapshots/policy_00000950.pt"),
)
DEFAULT_OUTPUT = Path(
    "artifacts/heads_up_v4_paper3x/evaluations/scenario_suite_350_725_950"
)
RAISE_ACTIONS = tuple(range(3, NUM_ACTIONS))


def card(text: str) -> int:
    """Convert a two-character card such as ``As`` to the engine index."""
    if len(text) != 2:
        raise ValueError(f"invalid card {text!r}")
    ranks = "23456789TJQKA"
    suits = "cdhs"
    try:
        return suits.index(text[1].lower()) * 13 + ranks.index(text[0].upper())
    except ValueError as error:
        raise ValueError(f"invalid card {text!r}") from error


@dataclass(frozen=True)
class PolicyCase:
    case_id: str
    category: str
    hand_label: str
    hero_cards: tuple[int, int]
    scenario: DecisionScenario
    importance: str = "core"


@dataclass(frozen=True)
class OrderingExpectation:
    expectation_id: str
    better_case: str
    worse_case: str
    metric: str
    minimum_margin: float
    reason: str


def _named_scenario(
    scenarios: Iterable[DecisionScenario],
    scenario_id: str,
    prefix: str,
) -> DecisionScenario:
    scenario = next(
        item for item in scenarios if item.scenario_id == scenario_id
    )
    return replace(
        scenario,
        scenario_id=f"{prefix}_{scenario.scenario_id}",
        label=f"{prefix}: {scenario.label}",
    )


def _add_hand_cases(
    target: list[PolicyCase],
    *,
    prefix: str,
    category: str,
    scenario: DecisionScenario,
    hands: Sequence[tuple[str, str, str]],
) -> None:
    for slug, label, cards in hands:
        first, second = cards.split()
        target.append(
            PolicyCase(
                case_id=f"{prefix}_{slug}",
                category=category,
                hand_label=label,
                hero_cards=(card(first), card(second)),
                scenario=scenario,
            )
        )


def policy_cases() -> tuple[PolicyCase, ...]:
    """Return the permanent 69-case decision catalogue."""
    cases: list[PolicyCase] = []
    preflop = {item.scenario_id: item for item in preflop_scenarios()}
    preflop_hands = (
        ("AA", "AA premium pair", "Ah Ad"),
        ("AKs", "AK suited", "As Ks"),
        ("A5s", "A5 suited", "Ah 5h"),
        ("76s", "76 suited connector", "7h 6h"),
        ("Q8o", "Q8 offsuit", "Qh 8c"),
        ("72o", "72 offsuit", "7c 2h"),
    )
    for scenario_id, prefix, category in (
        ("btn_unopened", "pre_btn_open", "preflop_open"),
        ("bb_vs_btn_open", "pre_bb_open", "preflop_defence"),
        ("bb_vs_btn_limp", "pre_bb_limp", "preflop_limp_response"),
        ("btn_vs_bb_3bet", "pre_btn_3bet", "preflop_three_bet"),
    ):
        _add_hand_cases(
            cases,
            prefix=prefix,
            category=category,
            scenario=preflop[scenario_id],
            hands=preflop_hands,
        )

    board_families = (
        (
            "dry",
            ("As", "7d", "2c"),
            (
                ("set", "middle set", "7h 7s"),
                ("top_pair", "top pair strong kicker", "Ac Kd"),
                ("underpair", "underpair", "8h 8s"),
                ("low_air", "low disconnected air", "6h 5h"),
                ("broadway_air", "broadway air", "Jc Tc"),
            ),
        ),
        (
            "wet",
            ("Js", "Ts", "9d"),
            (
                ("straight", "flopped straight", "Kh Qc"),
                ("set", "top set", "Jh Jc"),
                ("combo_draw", "nut combo draw", "As Ks"),
                ("top_pair", "top pair", "Ah Jd"),
                ("air", "disconnected air", "7c 2h"),
            ),
        ),
        (
            "paired",
            ("Kh", "Kd", "4c"),
            (
                ("full_house", "flopped full house", "4h 4s"),
                ("trips", "trip kings", "Ks Qc"),
                ("overpair", "pocket aces", "Ah Ad"),
                ("pair", "paired four", "Ac 4d"),
                ("air", "queen-high air", "Qs Js"),
            ),
        ),
    )
    for board_name, board_text, hands in board_families:
        board = tuple(card(value) for value in board_text)
        scenarios = postflop_scenarios(flop=board)
        for scenario_id, decision_name, category in (
            ("btn_flop_checked_to", "checked", "flop_checked_to"),
            ("bb_flop_vs_halfpot", "facing_half", "flop_facing_bet"),
        ):
            scenario = _named_scenario(scenarios, scenario_id, board_name)
            _add_hand_cases(
                cases,
                prefix=f"flop_{board_name}_{decision_name}",
                category=category,
                scenario=scenario,
                hands=hands,
            )

    turn_board = ("As", "7d", "2c", "Qh")
    turn_scenarios = postflop_scenarios(
        flop=tuple(card(value) for value in turn_board[:3]),
        turn=card(turn_board[3]),
    )
    turn_hands = (
        ("set", "turned set", "Qd Qc"),
        ("two_pair", "top two pair", "Ac Qd"),
        ("top_pair", "top pair", "Ah Kd"),
        ("underpair", "underpair", "8h 8s"),
        ("air", "six-high air", "6h 5h"),
    )
    for scenario_id, decision_name, category in (
        ("btn_turn_checked_to", "checked", "turn_checked_to"),
        ("btn_turn_vs_halfpot", "facing_half", "turn_facing_bet"),
    ):
        scenario = _named_scenario(turn_scenarios, scenario_id, "dry")
        _add_hand_cases(
            cases,
            prefix=f"turn_dry_{decision_name}",
            category=category,
            scenario=scenario,
            hands=turn_hands,
        )

    river_board = ("As", "7d", "2c", "Qh", "3s")
    river_scenarios = postflop_scenarios(
        flop=tuple(card(value) for value in river_board[:3]),
        turn=card(river_board[3]),
        river=card(river_board[4]),
    )
    river = _named_scenario(
        river_scenarios, "bb_river_vs_btn_pot", "dry"
    )
    _add_hand_cases(
        cases,
        prefix="river_dry_facing_pot",
        category="river_facing_pot",
        scenario=river,
        hands=(
            ("straight", "wheel straight", "5h 4h"),
            ("two_pair", "top two pair", "Ac Qd"),
            ("top_pair", "top pair", "Ah Kd"),
            ("underpair", "underpair", "8h 8s"),
            ("air", "jack-high air", "Jc Tc"),
        ),
    )
    if len(cases) != 69:
        raise RuntimeError(f"expected 69 policy cases, constructed {len(cases)}")
    if len({item.case_id for item in cases}) != len(cases):
        raise RuntimeError("policy case identifiers must be unique")
    return tuple(cases)


def ordering_expectations() -> tuple[OrderingExpectation, ...]:
    """Broad strength-order checks that avoid prescribing exact frequencies."""
    rows = (
        ("bb_open_continue_AA", "pre_bb_open_AA", "pre_bb_open_72o", "p_continue", .10),
        ("bb_open_raise_AA", "pre_bb_open_AA", "pre_bb_open_72o", "p_aggressive", .05),
        ("bb_open_continue_AKs", "pre_bb_open_AKs", "pre_bb_open_72o", "p_continue", .05),
        ("threebet_continue_AA", "pre_btn_3bet_AA", "pre_btn_3bet_72o", "p_continue", .10),
        ("threebet_raise_AA", "pre_btn_3bet_AA", "pre_btn_3bet_72o", "p_aggressive", .05),
        ("threebet_continue_AKs", "pre_btn_3bet_AKs", "pre_btn_3bet_72o", "p_continue", .05),
        ("dry_set_continue", "flop_dry_facing_half_set", "flop_dry_facing_half_low_air", "p_continue", .10),
        ("dry_top_pair_continue", "flop_dry_facing_half_top_pair", "flop_dry_facing_half_low_air", "p_continue", .05),
        ("wet_straight_continue", "flop_wet_facing_half_straight", "flop_wet_facing_half_air", "p_continue", .10),
        ("wet_set_continue", "flop_wet_facing_half_set", "flop_wet_facing_half_air", "p_continue", .10),
        ("paired_house_continue", "flop_paired_facing_half_full_house", "flop_paired_facing_half_air", "p_continue", .10),
        ("paired_trips_continue", "flop_paired_facing_half_trips", "flop_paired_facing_half_air", "p_continue", .10),
        ("turn_set_continue", "turn_dry_facing_half_set", "turn_dry_facing_half_air", "p_continue", .10),
        ("turn_twopair_continue", "turn_dry_facing_half_two_pair", "turn_dry_facing_half_air", "p_continue", .10),
        ("river_straight_continue", "river_dry_facing_pot_straight", "river_dry_facing_pot_air", "p_continue", .10),
        ("river_twopair_continue", "river_dry_facing_pot_two_pair", "river_dry_facing_pot_air", "p_continue", .05),
        ("river_toppair_continue", "river_dry_facing_pot_top_pair", "river_dry_facing_pot_air", "p_continue", .02),
    )
    return tuple(
        OrderingExpectation(
            expectation_id=name,
            better_case=better,
            worse_case=worse,
            metric=metric,
            minimum_margin=margin,
            reason="stronger made hand should preserve more probability on continuing/aggression",
        )
        for name, better, worse, metric, margin in rows
    )


class SnapshotEvaluator:
    def __init__(self, path: Path):
        self.path = path.resolve()
        self.snapshot = load_policy_snapshot(self.path, device="cpu")
        environment = self.snapshot.metadata["environment"]
        self.env = HeadsUpHoldemEngine(
            starting_stack=int(environment["starting_stack"]),
            small_blind=int(environment["small_blind"]),
            big_blind=int(environment["big_blind"]),
            seed=91_700,
        )
        with self.path.open("rb") as stream:
            self.sha256 = hashlib.file_digest(stream, "sha256").hexdigest()

    @torch.inference_mode()
    def probabilities(self, state) -> torch.Tensor:
        actor = int(state.to_act)
        legal = tuple(int(action) for action in self.env.legal_actions(state))
        observation = encode_information_state(
            state,
            actor,
            legal,
            self.env.bb,
            int(self.snapshot.metadata["max_history"]),
            action_descriptors=build_action_descriptors(self.env, state),
        )
        mask = torch.zeros(NUM_ACTIONS, dtype=torch.float32)
        mask[list(legal)] = 1.0
        logits = self.snapshot.policy_nets[actor](observation.unsqueeze(0))[0]
        return masked_softmax(logits, mask).cpu()


def _entropy(probabilities: Sequence[float]) -> float:
    return -sum(value * math.log(value) for value in probabilities if value > 0)


def evaluate_policy(
    path: Path, cases: Sequence[PolicyCase]
) -> tuple[list[dict], dict]:
    evaluator = SnapshotEvaluator(path)
    rows: list[dict] = []
    hard_failures: list[str] = []
    for case in cases:
        seat_probabilities: list[list[float]] = []
        legal_reference: tuple[int, ...] | None = None
        state_reference = None
        for hero in (0, 1):
            state = build_decision_state(
                evaluator.env,
                case.scenario,
                hero=hero,
                hero_cards=case.hero_cards,
            )
            probabilities = evaluator.probabilities(state).tolist()
            legal = tuple(evaluator.env.legal_actions(state))
            if legal_reference is None:
                legal_reference = legal
                state_reference = state
            elif legal != legal_reference:
                hard_failures.append(
                    f"{case.case_id}: legal actions differ by physical seat"
                )
            if abs(sum(probabilities) - 1.0) > 1e-5:
                hard_failures.append(
                    f"{case.case_id}/seat{hero}: probabilities do not sum to one"
                )
            illegal_mass = sum(
                probabilities[action]
                for action in range(NUM_ACTIONS)
                if action not in legal
            )
            if illegal_mass > 1e-7:
                hard_failures.append(
                    f"{case.case_id}/seat{hero}: illegal mass {illegal_mass}"
                )
            seat_probabilities.append(probabilities)
        assert legal_reference is not None and state_reference is not None
        probabilities = [
            (seat_probabilities[0][index] + seat_probabilities[1][index]) / 2
            for index in range(NUM_ACTIONS)
        ]
        seat_tv = 0.5 * sum(
            abs(left - right)
            for left, right in zip(*seat_probabilities)
        )
        row = {
            "iteration": evaluator.snapshot.iteration,
            "policy_file": str(evaluator.path),
            "policy_sha256": evaluator.sha256,
            "case_id": case.case_id,
            "category": case.category,
            "importance": case.importance,
            "hand_label": case.hand_label,
            "hero_cards": " ".join(
                card_to_string(value) for value in case.hero_cards
            ),
            "scenario_id": case.scenario.scenario_id,
            "scenario_label": case.scenario.label,
            "street": STREET_NAMES[int(state_reference.street)],
            "board": " ".join(
                card_to_string(value) for value in state_reference.board
            ),
            "pot_bb": state_reference.pot / evaluator.env.bb,
            "to_call_bb": evaluator.env.amount_to_call(
                state_reference, int(state_reference.to_act)
            )
            / evaluator.env.bb,
            "legal_actions": " ".join(
                ACTION_NAMES[action] for action in legal_reference
            ),
            "argmax_action": ACTION_NAMES[
                max(range(NUM_ACTIONS), key=probabilities.__getitem__)
            ],
            "p_continue": 1.0 - probabilities[ACTION_FOLD],
            "p_aggressive": sum(
                probabilities[action] for action in RAISE_ACTIONS
            ),
            "p_passive": probabilities[ACTION_CHECK]
            + probabilities[ACTION_CALL],
            "entropy_nats": _entropy(probabilities),
            "seat_strategy_tv": seat_tv,
        }
        row.update(
            {
                f"p_{name}": probabilities[action]
                for action, name in enumerate(ACTION_NAMES)
            }
        )
        rows.append(row)
    summary = {
        "iteration": evaluator.snapshot.iteration,
        "policy_file": str(evaluator.path),
        "policy_sha256": evaluator.sha256,
        "case_count": len(rows),
        "hard_failure_count": len(hard_failures),
        "hard_failures": hard_failures,
        "mean_entropy_nats": sum(row["entropy_nats"] for row in rows) / len(rows),
        "mean_seat_strategy_tv": sum(
            row["seat_strategy_tv"] for row in rows
        )
        / len(rows),
        "max_seat_strategy_tv": max(row["seat_strategy_tv"] for row in rows),
    }
    return rows, summary


def score_expectations(
    rows: Sequence[dict], expectations: Sequence[OrderingExpectation]
) -> tuple[list[dict], dict]:
    by_iteration: dict[int, dict[str, dict]] = {}
    for row in rows:
        by_iteration.setdefault(int(row["iteration"]), {})[row["case_id"]] = row
    details: list[dict] = []
    summaries: dict[str, dict] = {}
    for iteration, by_case in sorted(by_iteration.items()):
        passed = 0
        for expectation in expectations:
            better = float(by_case[expectation.better_case][expectation.metric])
            worse = float(by_case[expectation.worse_case][expectation.metric])
            observed_margin = better - worse
            success = observed_margin >= expectation.minimum_margin
            passed += int(success)
            details.append(
                {
                    "iteration": iteration,
                    **asdict(expectation),
                    "better_value": better,
                    "worse_value": worse,
                    "observed_margin": observed_margin,
                    "passed": success,
                }
            )
        summaries[str(iteration)] = {
            "passed": passed,
            "total": len(expectations),
            "pass_rate": passed / len(expectations),
        }
    return details, summaries


def compare_policies(rows: Sequence[dict]) -> list[dict]:
    by_iteration: dict[int, dict[str, dict]] = {}
    for row in rows:
        by_iteration.setdefault(int(row["iteration"]), {})[row["case_id"]] = row
    iterations = sorted(by_iteration)
    comparisons: list[dict] = []
    for earlier, later in zip(iterations, iterations[1:]):
        values: list[tuple[str, float]] = []
        for case_id in sorted(by_iteration[earlier]):
            left = by_iteration[earlier][case_id]
            right = by_iteration[later][case_id]
            tv = 0.5 * sum(
                abs(float(left[f"p_{name}"]) - float(right[f"p_{name}"]))
                for name in ACTION_NAMES
            )
            values.append((case_id, tv))
        maximum_case, maximum = max(values, key=lambda item: item[1])
        ordered = sorted(value for _, value in values)
        comparisons.append(
            {
                "earlier_iteration": earlier,
                "later_iteration": later,
                "case_count": len(values),
                "mean_strategy_tv": sum(ordered) / len(ordered),
                "median_strategy_tv": ordered[len(ordered) // 2],
                "large_shift_count_tv_ge_0_25": sum(
                    value >= 0.25 for value in ordered
                ),
                "max_strategy_tv": maximum,
                "max_shift_case": maximum_case,
            }
        )
    return comparisons


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_suite(policy_paths: Sequence[Path], output_dir: Path) -> dict:
    cases = policy_cases()
    expectations = ordering_expectations()
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    policy_summaries: list[dict] = []
    for path in policy_paths:
        rows, summary = evaluate_policy(path, cases)
        all_rows.extend(rows)
        policy_summaries.append(summary)
    expectation_rows, expectation_summaries = score_expectations(
        all_rows, expectations
    )
    comparisons = compare_policies(all_rows)
    for summary in policy_summaries:
        summary["behavioral_expectations"] = expectation_summaries[
            str(summary["iteration"])
        ]
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "controlled strategy fingerprint; behavioral expectations are "
            "sanity checks, not an equilibrium or exploitability measurement"
        ),
        "case_count_per_policy": len(cases),
        "expectation_count": len(expectations),
        "policies": policy_summaries,
        "comparisons": comparisons,
    }
    _write_csv(output_dir / "scenario_probabilities.csv", all_rows)
    _write_csv(output_dir / "behavioral_expectations.csv", expectation_rows)
    _write_csv(output_dir / "checkpoint_strategy_drift.csv", comparisons)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policies",
        nargs="+",
        type=Path,
        default=list(DEFAULT_POLICIES),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    missing = [path for path in args.policies if not path.is_file()]
    if missing:
        parser.error(
            "policy snapshot(s) not found: "
            + ", ".join(str(path) for path in missing)
        )
    report = run_suite(args.policies, args.output_dir)
    print(
        f"Wrote {report['case_count_per_policy']} controlled cases per policy "
        f"to {args.output_dir}"
    )
    for policy in report["policies"]:
        checks = policy["behavioral_expectations"]
        print(
            f"iteration {policy['iteration']}: "
            f"hard_failures={policy['hard_failure_count']} "
            f"behavior={checks['passed']}/{checks['total']} "
            f"mean_seat_TV={policy['mean_seat_strategy_tv']:.3f}"
        )
    for comparison in report["comparisons"]:
        print(
            f"{comparison['earlier_iteration']}->{comparison['later_iteration']}: "
            f"mean strategy TV={comparison['mean_strategy_tv']:.3f}, "
            f"large shifts={comparison['large_shift_count_tv_ge_0_25']}/"
            f"{comparison['case_count']}, "
            f"largest={comparison['max_shift_case']} "
            f"({comparison['max_strategy_tv']:.3f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
