"""Production strategy diagnostics for the three-player poker bot.

The charts in this module report *policy frequencies*, not hand equity or win
probability.  Every decision state is produced by a legal engine replay with a
controlled deck; no betting fields are patched by hand.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from itertools import combinations
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import torch

from three_player_engine import (
    ACTION_ALL_IN,
    ACTION_CALL,
    ACTION_CHECK,
    ACTION_FOLD,
    ACTION_HALF_POT,
    ACTION_MIN_RAISE,
    ACTION_NAMES,
    ACTION_POT,
    ACTION_RAISE_2X,
    ACTION_RAISE_3X,
    NUM_ACTIONS,
    STREET_FLOP,
    STREET_PREFLOP,
    STREET_RIVER,
    STREET_TURN,
    ThreePlayerHoldemEnv,
    card_to_string,
)


ROLE_BUTTON = "BTN"
ROLE_SMALL_BLIND = "SB"
ROLE_BIG_BLIND = "BB"
ROLES = (ROLE_BUTTON, ROLE_SMALL_BLIND, ROLE_BIG_BLIND)
RANK_LABELS = tuple("AKQJT98765432")
SUIT_LABELS = ("clubs", "diamonds", "hearts", "spades")
RAISE_ACTIONS = tuple(range(ACTION_MIN_RAISE, ACTION_ALL_IN + 1))


@dataclass(frozen=True)
class LineAction:
    """One role-relative action in a public betting line."""

    actor_role: str
    action: int


@dataclass(frozen=True)
class DecisionScenario:
    """A reproducible public situation ending at the hero's decision."""

    scenario_id: str
    label: str
    hero_role: str
    line: tuple[LineAction, ...] = ()
    expected_street: int = STREET_PREFLOP
    board: tuple[int, ...] = ()
    description: str = ""


@dataclass
class RangeReport:
    """Physical-combo predictions and their standard 169-cell aggregation."""

    scenario: DecisionScenario
    combo_table: pd.DataFrame
    hand_table: pd.DataFrame
    state_summary: dict[str, Any]


def _seat_for_role(button: int, role: str) -> int:
    if role == ROLE_BUTTON:
        return button
    if role == ROLE_SMALL_BLIND:
        return (button + 1) % 3
    if role == ROLE_BIG_BLIND:
        return (button + 2) % 3
    raise ValueError(f"unknown role {role!r}; expected one of {ROLES}")


def _role_for_seat(button: int, seat: int) -> str:
    for role in ROLES:
        if _seat_for_role(button, role) == seat:
            return role
    raise ValueError(f"seat {seat} is not valid")


def _street_for_board(board: Sequence[int]) -> int:
    return {0: STREET_PREFLOP, 3: STREET_FLOP, 4: STREET_TURN, 5: STREET_RIVER}[
        len(board)
    ]


def controlled_deck(
    *, button: int, hero: int, hero_cards: Sequence[int], board: Sequence[int]
) -> list[int]:
    """Return a full deck whose pop order deals the requested visible cards."""
    hero_cards = tuple(int(card) for card in hero_cards)
    board = tuple(int(card) for card in board)
    if len(hero_cards) != 2:
        raise ValueError("hero_cards must contain exactly two cards")
    if len(board) not in (0, 3, 4, 5):
        raise ValueError("board must contain 0, 3, 4, or 5 cards")
    visible = hero_cards + board
    if any(card < 0 or card >= 52 for card in visible):
        raise ValueError("cards must be integers in [0, 51]")
    if len(set(visible)) != len(visible):
        raise ValueError("hero cards and board must be unique")
    if hero not in (0, 1, 2) or button not in (0, 1, 2):
        raise ValueError("hero and button must be seats 0, 1, or 2")

    sb = (button + 1) % 3
    bb = (button + 2) % 3
    deal_order = (sb, bb, button, sb, bb, button)
    available = [card for card in range(52) if card not in set(visible)]

    def filler() -> int:
        if not available:
            raise RuntimeError("ran out of filler cards")
        return available.pop(0)

    controlled: list[int] = []
    hero_card_index = 0
    for seat in deal_order:
        if seat == hero:
            controlled.append(hero_cards[hero_card_index])
            hero_card_index += 1
        else:
            controlled.append(filler())
    if len(board) >= 3:
        controlled.extend((filler(), board[0], board[1], board[2]))
    if len(board) >= 4:
        controlled.extend((filler(), board[3]))
    if len(board) >= 5:
        controlled.extend((filler(), board[4]))

    controlled_set = set(controlled)
    if len(controlled_set) != len(controlled):
        raise RuntimeError("controlled draw sequence contains duplicate cards")
    remaining = [card for card in range(52) if card not in controlled_set]
    deck = remaining + list(reversed(controlled))
    if len(deck) != 52 or len(set(deck)) != 52:
        raise RuntimeError("controlled deck is not a complete permutation")
    return deck


def build_decision_state(
    env: ThreePlayerHoldemEnv,
    scenario: DecisionScenario,
    *,
    hero: int,
    hero_cards: Sequence[int],
):
    """Deal and legally replay a scenario, validating every public invariant."""
    if scenario.hero_role not in ROLES:
        raise ValueError(f"invalid hero role {scenario.hero_role!r}")
    if _street_for_board(scenario.board) != scenario.expected_street:
        raise ValueError("scenario board length does not match expected street")
    button = {
        ROLE_BUTTON: hero,
        ROLE_SMALL_BLIND: (hero - 1) % 3,
        ROLE_BIG_BLIND: (hero - 2) % 3,
    }[scenario.hero_role]
    deck = controlled_deck(
        button=button, hero=hero, hero_cards=hero_cards, board=scenario.board
    )
    state = env.new_hand(button=button, deck=deck)
    if set(state.hole[hero]) != set(hero_cards):
        raise RuntimeError("controlled deck did not deal the requested hero cards")

    for index, line_action in enumerate(scenario.line):
        if state.terminal or state.to_act is None:
            raise ValueError(f"scenario ended before line action {index}")
        actual_role = _role_for_seat(state.button, int(state.to_act))
        if actual_role != line_action.actor_role:
            raise ValueError(
                f"line action {index} expects {line_action.actor_role}, "
                f"but {actual_role} acts"
            )
        legal = env.legal_actions(state)
        if line_action.action not in legal:
            names = ", ".join(ACTION_NAMES[action] for action in legal)
            raise ValueError(
                f"line action {index} ({ACTION_NAMES[line_action.action]}) is "
                f"illegal for {actual_role}; legal actions: {names}"
            )
        state = env.step(state, line_action.action)

    if state.terminal or state.to_act != hero:
        raise ValueError("scenario must end at a nonterminal hero decision")
    if state.street != scenario.expected_street:
        raise ValueError(
            f"scenario reached street {state.street}, expected {scenario.expected_street}"
        )
    if tuple(state.board) != tuple(scenario.board):
        raise RuntimeError("controlled deck produced the wrong public board")
    expected_chips = sum(state.initial_stacks)
    if not math.isclose(sum(state.stacks) + state.pot, expected_chips, abs_tol=1e-8):
        raise RuntimeError("scenario violates chip conservation")
    return state


def preflop_scenarios() -> tuple[DecisionScenario, ...]:
    """Named preflop spots used by the production notebook."""
    return (
        DecisionScenario(
            "btn_unopened",
            "BTN unopened pot",
            ROLE_BUTTON,
            description="Button acts first with both blinds behind.",
        ),
        DecisionScenario(
            "sb_vs_btn_open",
            "SB facing BTN min-open",
            ROLE_SMALL_BLIND,
            (LineAction(ROLE_BUTTON, ACTION_MIN_RAISE),),
        ),
        DecisionScenario(
            "bb_vs_btn_open_sb_fold",
            "BB facing BTN open, SB folds",
            ROLE_BIG_BLIND,
            (
                LineAction(ROLE_BUTTON, ACTION_MIN_RAISE),
                LineAction(ROLE_SMALL_BLIND, ACTION_FOLD),
            ),
        ),
        DecisionScenario(
            "bb_vs_btn_open_sb_call",
            "BB facing BTN open + SB call",
            ROLE_BIG_BLIND,
            (
                LineAction(ROLE_BUTTON, ACTION_MIN_RAISE),
                LineAction(ROLE_SMALL_BLIND, ACTION_CALL),
            ),
        ),
        DecisionScenario(
            "bb_vs_two_limpers",
            "BB facing two limpers",
            ROLE_BIG_BLIND,
            (
                LineAction(ROLE_BUTTON, ACTION_CALL),
                LineAction(ROLE_SMALL_BLIND, ACTION_CALL),
            ),
        ),
        DecisionScenario(
            "btn_vs_bb_3bet",
            "BTN facing BB three-bet",
            ROLE_BUTTON,
            (
                LineAction(ROLE_BUTTON, ACTION_MIN_RAISE),
                LineAction(ROLE_SMALL_BLIND, ACTION_FOLD),
                LineAction(ROLE_BIG_BLIND, ACTION_RAISE_2X),
            ),
        ),
    )


def postflop_scenarios(
    *,
    flop: Sequence[int],
    turn: int | None = None,
    river: int | None = None,
) -> tuple[DecisionScenario, ...]:
    """Representative single-raised-pot postflop decisions on a fixed board."""
    flop = tuple(int(card) for card in flop)
    if len(flop) != 3:
        raise ValueError("flop must contain exactly three cards")
    if len(set(flop)) != 3:
        raise ValueError("flop cards must be unique")
    scenarios: list[DecisionScenario] = [
        DecisionScenario(
            "btn_flop_checked_to",
            "BTN on flop after SB/BB check",
            ROLE_BUTTON,
            (
                LineAction(ROLE_BUTTON, ACTION_MIN_RAISE),
                LineAction(ROLE_SMALL_BLIND, ACTION_CALL),
                LineAction(ROLE_BIG_BLIND, ACTION_CALL),
                LineAction(ROLE_SMALL_BLIND, ACTION_CHECK),
                LineAction(ROLE_BIG_BLIND, ACTION_CHECK),
            ),
            STREET_FLOP,
            flop,
        ),
        DecisionScenario(
            "btn_flop_vs_bet_call",
            "BTN facing flop half-pot bet + call",
            ROLE_BUTTON,
            (
                LineAction(ROLE_BUTTON, ACTION_MIN_RAISE),
                LineAction(ROLE_SMALL_BLIND, ACTION_CALL),
                LineAction(ROLE_BIG_BLIND, ACTION_CALL),
                # In a 12-chip pot the six-chip half-pot target aliases the
                # earlier raise_3x action ID; the engine intentionally keeps
                # one canonical action per resulting state.
                LineAction(ROLE_SMALL_BLIND, ACTION_RAISE_3X),
                LineAction(ROLE_BIG_BLIND, ACTION_CALL),
            ),
            STREET_FLOP,
            flop,
        ),
        DecisionScenario(
            "bb_hu_flop_vs_halfpot",
            "BB heads-up facing BTN half-pot c-bet",
            ROLE_BIG_BLIND,
            (
                LineAction(ROLE_BUTTON, ACTION_MIN_RAISE),
                LineAction(ROLE_SMALL_BLIND, ACTION_FOLD),
                LineAction(ROLE_BIG_BLIND, ACTION_CALL),
                LineAction(ROLE_BIG_BLIND, ACTION_CHECK),
                LineAction(ROLE_BUTTON, ACTION_HALF_POT),
            ),
            STREET_FLOP,
            flop,
        ),
    ]
    if turn is not None:
        turn_board = flop + (int(turn),)
        scenarios.extend(
            (
                DecisionScenario(
                "btn_turn_checked_to",
                "BTN checked to on turn",
                ROLE_BUTTON,
                (
                    LineAction(ROLE_BUTTON, ACTION_MIN_RAISE),
                    LineAction(ROLE_SMALL_BLIND, ACTION_CALL),
                    LineAction(ROLE_BIG_BLIND, ACTION_CALL),
                    LineAction(ROLE_SMALL_BLIND, ACTION_CHECK),
                    LineAction(ROLE_BIG_BLIND, ACTION_CHECK),
                    LineAction(ROLE_BUTTON, ACTION_CHECK),
                    LineAction(ROLE_SMALL_BLIND, ACTION_CHECK),
                    LineAction(ROLE_BIG_BLIND, ACTION_CHECK),
                ),
                STREET_TURN,
                turn_board,
                ),
                DecisionScenario(
                    "btn_turn_vs_halfpot",
                    "BTN facing SB half-pot turn bet",
                    ROLE_BUTTON,
                    (
                        LineAction(ROLE_BUTTON, ACTION_MIN_RAISE),
                        LineAction(ROLE_SMALL_BLIND, ACTION_CALL),
                        LineAction(ROLE_BIG_BLIND, ACTION_CALL),
                        LineAction(ROLE_SMALL_BLIND, ACTION_CHECK),
                        LineAction(ROLE_BIG_BLIND, ACTION_CHECK),
                        LineAction(ROLE_BUTTON, ACTION_CHECK),
                        # As on the flop, six chips is both half-pot and the
                        # canonical raise_3x target in this abstraction.
                        LineAction(ROLE_SMALL_BLIND, ACTION_RAISE_3X),
                        LineAction(ROLE_BIG_BLIND, ACTION_FOLD),
                    ),
                    STREET_TURN,
                    turn_board,
                ),
            )
        )
    if river is not None:
        if turn is None:
            raise ValueError("river requires a turn card")
        river_board = flop + (int(turn), int(river))
        scenarios.append(
            DecisionScenario(
                "bb_river_vs_btn_pot",
                "BB facing BTN pot-size river bet",
                ROLE_BIG_BLIND,
                (
                    LineAction(ROLE_BUTTON, ACTION_MIN_RAISE),
                    LineAction(ROLE_SMALL_BLIND, ACTION_FOLD),
                    LineAction(ROLE_BIG_BLIND, ACTION_CALL),
                    LineAction(ROLE_BIG_BLIND, ACTION_CHECK),
                    LineAction(ROLE_BUTTON, ACTION_CHECK),
                    LineAction(ROLE_BIG_BLIND, ACTION_CHECK),
                    LineAction(ROLE_BUTTON, ACTION_CHECK),
                    LineAction(ROLE_BIG_BLIND, ACTION_CHECK),
                    LineAction(ROLE_BUTTON, ACTION_POT),
                ),
                STREET_RIVER,
                river_board,
            )
        )
    return tuple(scenarios)


def _rank_grid_index(rank: int) -> int:
    return 12 - int(rank)


def classify_hole_cards(cards: Sequence[int]) -> tuple[str, int, int]:
    """Return standard hand label and 13x13 row/column coordinates."""
    if len(cards) != 2 or cards[0] == cards[1]:
        raise ValueError("cards must contain two distinct cards")
    rank_a, rank_b = cards[0] % 13, cards[1] % 13
    if rank_a == rank_b:
        index = _rank_grid_index(rank_a)
        label = RANK_LABELS[index] * 2
        return label, index, index
    high, low = max(rank_a, rank_b), min(rank_a, rank_b)
    high_index, low_index = _rank_grid_index(high), _rank_grid_index(low)
    suited = cards[0] // 13 == cards[1] // 13
    label = f"{RANK_LABELS[high_index]}{RANK_LABELS[low_index]}{'s' if suited else 'o'}"
    if suited:
        return label, high_index, low_index
    return label, low_index, high_index


def _scenario_summary(env: ThreePlayerHoldemEnv, state, scenario: DecisionScenario) -> dict[str, Any]:
    hero = int(state.to_act)
    legal = env.legal_actions(state)
    to_call = env.amount_to_call(state, hero)
    pot_odds = to_call / (state.pot + to_call) if to_call > 0 else 0.0
    active_opponent_stacks = [
        state.stacks[player]
        for player in range(3)
        if player != hero and not state.folded[player]
    ]
    effective_stack = min([state.stacks[hero], *active_opponent_stacks])
    targets = {
        ACTION_NAMES[action]: env.action_target(state, action) for action in legal
    }
    return {
        "scenario_id": scenario.scenario_id,
        "label": scenario.label,
        "hero_role": scenario.hero_role,
        "street": state.street,
        "board": " ".join(card_to_string(card) for card in state.board),
        "pot_bb": state.pot / env.bb,
        "to_call_bb": to_call / env.bb,
        "pot_odds": pot_odds,
        "effective_stack_bb": effective_stack / env.bb,
        "spr": effective_stack / max(state.pot, 1e-9),
        "legal_actions": tuple(ACTION_NAMES[action] for action in legal),
        "action_targets_bb": {
            name: target / env.bb for name, target in targets.items()
        },
    }


class StrategyAnalyzer:
    """Batched range and board-card inspection for average-policy networks."""

    def __init__(self, trainer, *, batch_size: int = 4096):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.trainer = trainer
        self.batch_size = int(batch_size)

    def _analysis_env(self) -> ThreePlayerHoldemEnv:
        return type(self.trainer.env)(
            stack_size=self.trainer.env.stack_size,
            sb=self.trainer.env.sb,
            bb=self.trainer.env.bb,
            seed=self.trainer.seed + 90_000,
        )

    def analyze_range(
        self,
        scenario: DecisionScenario,
        *,
        policy_nets: Sequence[torch.nn.Module] | None = None,
        hero_seats: Sequence[int] = (0, 1, 2),
    ) -> RangeReport:
        """Evaluate every unblocked physical hole-card combination."""
        hero_seats = tuple(int(seat) for seat in hero_seats)
        if not hero_seats or any(seat not in (0, 1, 2) for seat in hero_seats):
            raise ValueError("hero_seats must contain seats from 0, 1, 2")
        board_set = set(scenario.board)
        hole_combos = [
            cards
            for cards in combinations(range(52), 2)
            if cards[0] not in board_set and cards[1] not in board_set
        ]
        env = self._analysis_env()
        states: list[Any] = []
        metadata: list[tuple[int, tuple[int, int], str, int, int]] = []
        for hero in hero_seats:
            for cards in hole_combos:
                label, row, column = classify_hole_cards(cards)
                state = build_decision_state(
                    env, scenario, hero=hero, hero_cards=cards
                )
                states.append(state)
                metadata.append((hero, cards, label, row, column))

        probabilities = self.trainer.average_policy_batch(
            states, policy_nets=policy_nets, batch_size=self.batch_size
        )
        records: list[dict[str, Any]] = []
        for state, probability, meta in zip(states, probabilities, metadata):
            hero, cards, label, row, column = meta
            legal = set(env.legal_actions(state))
            record: dict[str, Any] = {
                "scenario_id": scenario.scenario_id,
                "hero_seat": hero,
                "cards": " ".join(card_to_string(card) for card in cards),
                "combo_key": f"{cards[0]:02d}-{cards[1]:02d}",
                "hand": label,
                "row": row,
                "column": column,
                "call_available": ACTION_CALL in legal,
            }
            for action, name in enumerate(ACTION_NAMES):
                record[f"p_{name}"] = float(probability[action])
            record["p_fold"] = float(probability[ACTION_FOLD])
            record["p_check"] = float(probability[ACTION_CHECK])
            record["p_call"] = (
                float(probability[ACTION_CALL]) if ACTION_CALL in legal else float("nan")
            )
            record["p_aggressive"] = float(
                sum(float(probability[action]) for action in RAISE_ACTIONS)
            )
            record["p_continue"] = 1.0 - record["p_fold"]
            records.append(record)

        combo_table = pd.DataFrame.from_records(records)
        probability_columns = [
            *(f"p_{name}" for name in ACTION_NAMES),
            "p_fold",
            "p_check",
            "p_call",
            "p_aggressive",
            "p_continue",
        ]
        probability_columns = list(dict.fromkeys(probability_columns))
        grouped = combo_table.groupby(
            ["scenario_id", "hand", "row", "column"], sort=False
        )
        hand_table = grouped[probability_columns].mean().reset_index()
        combo_std = (
            grouped[probability_columns]
            .std(ddof=0)
            .add_prefix("combo_std_")
            .reset_index()
        )
        hand_table = hand_table.merge(
            combo_std,
            on=["scenario_id", "hand", "row", "column"],
        )
        combo_counts = grouped["combo_key"].nunique().rename("combo_count").reset_index()
        hand_table = hand_table.merge(
            combo_counts, on=["scenario_id", "hand", "row", "column"]
        )

        by_network = (
            combo_table.groupby(["hand", "hero_seat"])[probability_columns]
            .mean()
            .reset_index()
        )
        network_std = (
            by_network.groupby("hand")[probability_columns]
            .std(ddof=0)
            .add_prefix("between_net_std_")
            .reset_index()
        )
        hand_table = hand_table.merge(network_std, on="hand", how="left")
        sample_state = states[0]
        summary = _scenario_summary(env, sample_state, scenario)
        summary["physical_combos"] = len(hole_combos)
        summary["network_samples"] = len(states)
        return RangeReport(scenario, combo_table, hand_table, summary)

    def analyze_cases(
        self,
        scenarios: Iterable[DecisionScenario],
        *,
        policy_nets: Sequence[torch.nn.Module] | None = None,
        hero_seats: Sequence[int] = (0, 1, 2),
    ) -> list[RangeReport]:
        return [
            self.analyze_range(
                scenario, policy_nets=policy_nets, hero_seats=hero_seats
            )
            for scenario in scenarios
        ]

    def analyze_next_cards(
        self,
        scenario: DecisionScenario,
        *,
        hero_cards: Sequence[int],
        policy_nets: Sequence[torch.nn.Module] | None = None,
        hero_seats: Sequence[int] = (0, 1, 2),
    ) -> pd.DataFrame:
        """Vary the final turn/river card in a scenario and report its policy."""
        if len(scenario.board) not in (4, 5):
            raise ValueError("next-card sweep requires a turn or river scenario")
        prefix = tuple(scenario.board[:-1])
        blocked = set(prefix) | set(hero_cards)
        states: list[Any] = []
        metadata: list[tuple[int, int]] = []
        env = self._analysis_env()
        for next_card in range(52):
            if next_card in blocked:
                continue
            varied = replace(scenario, board=prefix + (next_card,))
            for hero in hero_seats:
                states.append(
                    build_decision_state(
                        env, varied, hero=int(hero), hero_cards=hero_cards
                    )
                )
                metadata.append((next_card, int(hero)))
        predictions = self.trainer.average_policy_batch(
            states, policy_nets=policy_nets, batch_size=self.batch_size
        )
        rows: list[dict[str, Any]] = []
        for state, probability, (card, hero) in zip(states, predictions, metadata):
            legal = set(env.legal_actions(state))
            rows.append(
                {
                    "card": card,
                    "card_label": card_to_string(card),
                    "rank": _rank_grid_index(card % 13),
                    "suit": card // 13,
                    "hero_seat": hero,
                    "p_fold": float(probability[ACTION_FOLD]),
                    "p_check": float(probability[ACTION_CHECK]),
                    "p_call": (
                        float(probability[ACTION_CALL])
                        if ACTION_CALL in legal
                        else float("nan")
                    ),
                    "p_aggressive": float(
                        sum(float(probability[action]) for action in RAISE_ACTIONS)
                    ),
                }
            )
        return (
            pd.DataFrame(rows)
            .groupby(["card", "card_label", "rank", "suit"], as_index=False)
            .mean(numeric_only=True)
        )


def compare_ranges(previous: RangeReport, current: RangeReport) -> pd.DataFrame:
    """Return strategy movement; positive movement is not necessarily better."""
    if previous.scenario.scenario_id != current.scenario.scenario_id:
        raise ValueError("range reports must describe the same scenario")
    action_columns = [f"p_{name}" for name in ACTION_NAMES]
    keys = ["hand", "row", "column"]
    merged = previous.hand_table[keys + action_columns].merge(
        current.hand_table[keys + action_columns],
        on=keys,
        suffixes=("_previous", "_current"),
    )
    for metric in ("p_fold", "p_call", "p_aggressive", "p_continue"):
        previous_values = previous.hand_table[keys + [metric]]
        current_values = current.hand_table[keys + [metric]]
        delta = previous_values.merge(
            current_values, on=keys, suffixes=("_previous", "_current")
        )
        merged[f"delta_{metric}"] = delta[f"{metric}_current"] - delta[
            f"{metric}_previous"
        ]
    merged["strategy_total_variation"] = 0.5 * sum(
        (
            merged[f"{column}_current"] - merged[f"{column}_previous"]
        ).abs().fillna(0.0)
        for column in action_columns
    )
    return merged


def _metric_matrix(table: pd.DataFrame, metric: str) -> np.ndarray:
    matrix = np.full((13, 13), np.nan, dtype=float)
    for row in table.itertuples(index=False):
        matrix[int(row.row), int(row.column)] = float(getattr(row, metric))
    return matrix


def plot_range_heatmaps(
    report: RangeReport,
    *,
    metrics: Sequence[str] = ("p_fold", "p_call", "p_aggressive"),
    annotate: bool = False,
) -> plt.Figure:
    """Plot standard 13x13 policy-frequency charts for one scenario."""
    metrics = tuple(metrics)
    if not metrics:
        raise ValueError("at least one metric is required")
    fig, axes = plt.subplots(1, len(metrics), figsize=(6.2 * len(metrics), 5.5))
    axes_array = np.atleast_1d(axes)
    titles = {
        "p_fold": "Fold frequency",
        "p_check": "Check frequency",
        "p_call": "Call frequency",
        "p_aggressive": "Total raise frequency",
        "p_continue": "Continue frequency",
    }
    for axis, metric in zip(axes_array, metrics):
        matrix = _metric_matrix(report.hand_table, metric)
        image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis")
        axis.set_xticks(range(13), RANK_LABELS)
        axis.set_yticks(range(13), RANK_LABELS)
        axis.set_xlabel("second rank (suited above diagonal)")
        axis.set_ylabel("first rank (offsuit below diagonal)")
        axis.set_title(titles.get(metric, metric))
        if annotate:
            for row in range(13):
                for column in range(13):
                    value = matrix[row, column]
                    if math.isfinite(value):
                        axis.text(
                            column,
                            row,
                            f"{100 * value:.0f}",
                            ha="center",
                            va="center",
                            fontsize=6,
                            color="white" if value < 0.55 else "black",
                        )
        fig.colorbar(
            image,
            ax=axis,
            fraction=0.046,
            pad=0.04,
            format=PercentFormatter(1.0),
        )
    fig.suptitle(
        f"{report.scenario.label} — average policy frequency",
        fontsize=14,
    )
    fig.tight_layout()
    return fig


def plot_call_maps(reports: Sequence[RangeReport]) -> plt.Figure:
    """Plot call frequency for several bet-facing situations."""
    if not reports:
        raise ValueError("at least one report is required")
    columns = min(3, len(reports))
    rows = math.ceil(len(reports) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(5.5 * columns, 5 * rows))
    flat_axes = np.atleast_1d(axes).ravel()
    last_image = None
    for axis, report in zip(flat_axes, reports):
        matrix = _metric_matrix(report.hand_table, "p_call")
        last_image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="magma")
        axis.set_xticks(range(13), RANK_LABELS)
        axis.set_yticks(range(13), RANK_LABELS)
        axis.set_title(report.scenario.label)
    for axis in flat_axes[len(reports) :]:
        axis.axis("off")
    if last_image is not None:
        fig.colorbar(
            last_image,
            ax=list(flat_axes[: len(reports)]),
            shrink=0.72,
            format=PercentFormatter(1.0),
        )
    fig.suptitle("Call frequency by starting hand (policy probability, not equity)")
    fig.subplots_adjust(top=0.9, wspace=0.22, hspace=0.3)
    return fig


def plot_range_delta(
    comparison: pd.DataFrame,
    *,
    metric: str = "delta_p_call",
    title: str = "Policy-frequency change",
) -> plt.Figure:
    """Plot a symmetric 13x13 change map between two policy snapshots."""
    matrix = _metric_matrix(comparison, metric)
    finite = np.abs(matrix[np.isfinite(matrix)])
    limit = max(float(finite.max()) if finite.size else 0.0, 0.01)
    fig, axis = plt.subplots(figsize=(6.5, 5.5))
    image = axis.imshow(matrix, vmin=-limit, vmax=limit, cmap="coolwarm")
    axis.set_xticks(range(13), RANK_LABELS)
    axis.set_yticks(range(13), RANK_LABELS)
    axis.set_xlabel("second rank (suited above diagonal)")
    axis.set_ylabel("first rank (offsuit below diagonal)")
    axis.set_title(title)
    fig.colorbar(image, ax=axis, label="candidate minus baseline")
    fig.tight_layout()
    return fig


def plot_card_sweep(
    table: pd.DataFrame, *, metric: str = "p_call", title: str = "Next-card policy"
) -> plt.Figure:
    """Plot how each possible turn/river card changes one policy frequency."""
    matrix = np.full((4, 13), np.nan, dtype=float)
    for row in table.itertuples(index=False):
        matrix[int(row.suit), int(row.rank)] = float(getattr(row, metric))
    fig, axis = plt.subplots(figsize=(12, 3.8))
    image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="plasma", aspect="auto")
    axis.set_xticks(range(13), RANK_LABELS)
    axis.set_yticks(range(4), SUIT_LABELS)
    axis.set_title(title)
    for suit in range(4):
        for rank in range(13):
            value = matrix[suit, rank]
            if math.isfinite(value):
                axis.text(
                    rank,
                    suit,
                    f"{100 * value:.0f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if value < 0.55 else "black",
                )
    fig.colorbar(
        image,
        ax=axis,
        label="policy frequency",
        format=PercentFormatter(1.0),
    )
    fig.tight_layout()
    return fig


def plot_network_architecture(trainer: Any) -> plt.Figure:
    """Draw the actual six-network trainer and one network's residual stack."""
    input_dim = int(trainer.input_dim)
    hidden = int(trainer.hidden)
    blocks = int(trainer.blocks)
    advantage_parameters = sum(
        parameter.numel() for parameter in trainer.advantage_nets[0].parameters()
    )
    policy_parameters = sum(
        parameter.numel() for parameter in trainer.policy_nets[0].parameters()
    )
    total_parameters = sum(
        parameter.numel()
        for network in trainer.advantage_nets + trainer.policy_nets
        for parameter in network.parameters()
    )

    figure, axes = plt.subplots(1, 2, figsize=(18, 9))
    system_axis, detail_axis = axes
    for axis in axes:
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.axis("off")

    def box(
        axis,
        x: float,
        y: float,
        width: float,
        height: float,
        label: str,
        color: str,
        *,
        fontsize: float = 10,
        edgecolor: str = "#243447",
    ) -> None:
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.015",
            linewidth=1.5,
            edgecolor=edgecolor,
            facecolor=color,
        )
        axis.add_patch(patch)
        axis.text(
            x + width / 2,
            y + height / 2,
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            wrap=True,
        )

    def arrow(axis, start: tuple[float, float], end: tuple[float, float]) -> None:
        axis.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={"arrowstyle": "-|>", "lw": 1.8, "color": "#34495e"},
        )

    system_axis.set_title("Six-network Neural CFR system", fontsize=15, pad=16)
    box(
        system_axis,
        0.02,
        0.40,
        0.25,
        0.20,
        f"Hero-visible information state\n{input_dim} numeric features\n+ legal-action mask",
        "#d9edf7",
        fontsize=10,
    )
    box(
        system_axis,
        0.36,
        0.65,
        0.31,
        0.20,
        "ADVANTAGE NETWORKS\nPlayer 0  |  Player 1  |  Player 2\n3 independently fitted networks",
        "#f8d7da",
        fontsize=10,
    )
    box(
        system_axis,
        0.36,
        0.15,
        0.31,
        0.20,
        "AVERAGE-POLICY NETWORKS\nPlayer 0  |  Player 1  |  Player 2\n3 independently fitted networks",
        "#d4edda",
        fontsize=10,
    )
    box(
        system_axis,
        0.76,
        0.65,
        0.22,
        0.20,
        f"{NUM_ACTIONS} advantage values\nRegret matching\n→ traversal strategy",
        "#fde2e4",
        fontsize=9.5,
    )
    box(
        system_axis,
        0.76,
        0.15,
        0.22,
        0.20,
        f"{NUM_ACTIONS} policy logits\nLegal mask + softmax\n→ action probabilities",
        "#e2f0d9",
        fontsize=9.5,
    )
    arrow(system_axis, (0.27, 0.52), (0.36, 0.75))
    arrow(system_axis, (0.27, 0.48), (0.36, 0.25))
    arrow(system_axis, (0.67, 0.75), (0.76, 0.75))
    arrow(system_axis, (0.67, 0.25), (0.76, 0.25))
    system_axis.text(
        0.50,
        0.94,
        "3 advantage + 3 policy = 6 networks",
        ha="center",
        fontsize=11,
        weight="bold",
    )
    system_axis.text(
        0.50,
        0.04,
        (
            f"Parameters: {advantage_parameters:,} per advantage network, "
            f"{policy_parameters:,} per policy network\n"
            f"{total_parameters:,} trainable parameters across all six"
        ),
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#34495e",
    )

    architecture = getattr(trainer, "network_architecture", "residual_mlp")
    detail_axis.set_title("Architecture shared by every network", fontsize=15, pad=16)
    if architecture == "dual_attention_state":
        box(detail_axis, 0.03, 0.79, 0.28, 0.14,
            "7 card tokens\n2 private + 5 board\n4-head self-attention (128d)",
            "#d9edf7", fontsize=9)
        box(detail_axis, 0.36, 0.79, 0.28, 0.14,
            "32 betting tokens\n4-head self-attention (128d)\n+ GRU state memory",
            "#e8daef", fontsize=9)
        box(detail_axis, 0.69, 0.79, 0.28, 0.14,
            "Stack, position, legal action\nand tournament features\nMLP (128d)",
            "#d4edda", fontsize=9)
        box(detail_axis, 0.20, 0.57, 0.60, 0.11,
            f"Concatenate 3 representations  →  Linear(384 → {hidden})  →  SiLU",
            "#fff3cd", fontsize=9.5)
        box(detail_axis, 0.16, 0.36, 0.68, 0.13,
            f"ONE RESIDUAL HIDDEN BLOCK ({hidden}d)\nLayerNorm → SiLU → Linear → SiLU → Linear + skip",
            "#e8daef", fontsize=9.5)
        box(detail_axis, 0.22, 0.21, 0.56, 0.08,
            f"LayerNorm({hidden}) → Linear({hidden} → {NUM_ACTIONS})",
            "#fff3cd", fontsize=9.5)
        box(detail_axis, 0.04, 0.04, 0.42, 0.10,
            f"Advantage head\n{NUM_ACTIONS} unconstrained values", "#f8d7da", fontsize=9)
        box(detail_axis, 0.54, 0.04, 0.42, 0.10,
            f"Policy head\n{NUM_ACTIONS} raw logits", "#d4edda", fontsize=9)
        arrow(detail_axis, (0.17, 0.79), (0.36, 0.68))
        arrow(detail_axis, (0.50, 0.79), (0.50, 0.68))
        arrow(detail_axis, (0.83, 0.79), (0.64, 0.68))
        arrow(detail_axis, (0.50, 0.57), (0.50, 0.49))
        arrow(detail_axis, (0.50, 0.36), (0.50, 0.29))
        arrow(detail_axis, (0.48, 0.21), (0.27, 0.14))
        arrow(detail_axis, (0.52, 0.21), (0.73, 0.14))
    elif architecture == "deep_cfr_branch":
        box(detail_axis, 0.03, 0.78, 0.43, 0.16,
            "CARD BRANCH\nrank + suit + exact-card embeddings (64d)\nsum within hole/flop/turn/river groups\n256 -> 192 -> 192 -> 64",
            "#d9edf7", fontsize=8.7)
        box(detail_axis, 0.54, 0.78, 0.43, 0.16,
            "PUBLIC STATE + BETTING BRANCH\nordered 32-event history, stacks, position,\npot, legal actions, tournament features\nLayerNorm -> Linear(64) + residual",
            "#f8d7da", fontsize=8.7)
        box(detail_axis, 0.18, 0.56, 0.64, 0.12,
            "Concatenate [64 + 64] -> Linear(128 -> 64) + ReLU",
            "#fff3cd", fontsize=9.5)
        box(detail_axis, 0.16, 0.34, 0.68, 0.14,
            f"DEEP CFR TRUNK: RESIDUAL BLOCK x {blocks}\nLinear(64 -> 64) + ReLU + skip connection",
            "#e8daef", fontsize=9.5)
        box(detail_axis, 0.25, 0.22, 0.50, 0.07,
            "L2 feature normalization", "#fff3cd", fontsize=9)
        box(detail_axis, 0.04, 0.04, 0.42, 0.11,
            f"Advantage head\nLinear(64 -> {NUM_ACTIONS}) values", "#f8d7da", fontsize=9)
        box(detail_axis, 0.54, 0.04, 0.42, 0.11,
            f"Policy head\nLinear(64 -> {NUM_ACTIONS}) logits", "#d4edda", fontsize=9)
        arrow(detail_axis, (0.25, 0.78), (0.39, 0.68))
        arrow(detail_axis, (0.75, 0.78), (0.61, 0.68))
        arrow(detail_axis, (0.50, 0.56), (0.50, 0.48))
        arrow(detail_axis, (0.50, 0.34), (0.50, 0.29))
        arrow(detail_axis, (0.48, 0.22), (0.27, 0.15))
        arrow(detail_axis, (0.52, 0.22), (0.73, 0.15))
    else:
        box(
            detail_axis,
            0.22,
            0.86,
            0.56,
            0.09,
            f"Input information state  •  {input_dim} features",
            "#d9edf7",
        )
        box(
        detail_axis,
        0.22,
        0.86,
        0.56,
        0.09,
        f"Input information state  •  {input_dim} features",
        "#d9edf7",
    )
        box(
        detail_axis,
        0.22,
        0.70,
        0.56,
        0.10,
        f"LayerNorm({input_dim})  →  Linear({input_dim} → {hidden})  →  SiLU",
        "#fff3cd",
    )
        box(
        detail_axis,
        0.12,
        0.39,
        0.76,
        0.23,
        (
            f"RESIDUAL BLOCK × {blocks}\n\n"
            f"LayerNorm({hidden}) → SiLU → Linear({hidden} → {hidden})\n"
            f"→ SiLU → Linear({hidden} → {hidden})\n"
            "output + unchanged skip connection"
        ),
        "#e8daef",
        fontsize=10,
    )
        box(
        detail_axis,
        0.22,
        0.23,
        0.56,
        0.09,
        f"LayerNorm({hidden})  →  Linear({hidden} → {NUM_ACTIONS})",
        "#fff3cd",
    )
        box(
        detail_axis,
        0.04,
        0.04,
        0.42,
        0.11,
        f"Advantage head\n{NUM_ACTIONS} unconstrained values",
        "#f8d7da",
        fontsize=9.5,
    )
        box(
        detail_axis,
        0.54,
        0.04,
        0.42,
        0.11,
        f"Policy head\n{NUM_ACTIONS} raw logits",
        "#d4edda",
        fontsize=9.5,
    )
        arrow(detail_axis, (0.50, 0.86), (0.50, 0.80))
        arrow(detail_axis, (0.50, 0.70), (0.50, 0.62))
        arrow(detail_axis, (0.50, 0.39), (0.50, 0.32))
        arrow(detail_axis, (0.48, 0.23), (0.27, 0.15))
        arrow(detail_axis, (0.52, 0.23), (0.73, 0.15))

    figure.suptitle(
        "Three-player poker bot network architecture",
        fontsize=18,
        weight="bold",
        y=0.995,
    )
    figure.subplots_adjust(top=0.90, bottom=0.04, wspace=0.10)
    return figure


__all__ = [
    "DecisionScenario",
    "LineAction",
    "RangeReport",
    "StrategyAnalyzer",
    "build_decision_state",
    "classify_hole_cards",
    "compare_ranges",
    "controlled_deck",
    "plot_call_maps",
    "plot_card_sweep",
    "plot_network_architecture",
    "plot_range_delta",
    "plot_range_heatmaps",
    "postflop_scenarios",
    "preflop_scenarios",
]
