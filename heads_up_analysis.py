"""Strategy diagnostics and critical-state plots for the heads-up bot.

Every plotted decision is constructed by dealing a controlled deck and legally
replaying the public betting line through the exact heads-up engine.  The
resulting charts are policy frequencies, not hand equity.
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

from heads_up_engine import (
    ACTION_ALL_IN,
    ACTION_CALL,
    ACTION_CHECK,
    ACTION_FOLD,
    ACTION_HALF_POT,
    ACTION_MIN_RAISE,
    ACTION_NAMES,
    ACTION_POT,
    NUM_ACTIONS,
    STREET_FLOP,
    STREET_PREFLOP,
    STREET_RIVER,
    STREET_TURN,
    card_to_string,
)
from heads_up_native import HeadsUpHoldemEngine


ROLE_BUTTON = "BTN/SB"
ROLE_BIG_BLIND = "BB"
ROLES = (ROLE_BUTTON, ROLE_BIG_BLIND)
RANK_LABELS = tuple("AKQJT98765432")
SUIT_LABELS = ("clubs", "diamonds", "hearts", "spades")
RAISE_ACTIONS = tuple(range(ACTION_MIN_RAISE, ACTION_ALL_IN + 1))


@dataclass(frozen=True)
class LineAction:
    actor_role: str
    action: int


@dataclass(frozen=True)
class DecisionScenario:
    scenario_id: str
    label: str
    hero_role: str
    line: tuple[LineAction, ...] = ()
    expected_street: int = STREET_PREFLOP
    board: tuple[int, ...] = ()
    description: str = ""


@dataclass
class RangeReport:
    scenario: DecisionScenario
    combo_table: pd.DataFrame
    hand_table: pd.DataFrame
    state_summary: dict[str, Any]


def _seat_for_role(button: int, role: str) -> int:
    if role == ROLE_BUTTON:
        return int(button)
    if role == ROLE_BIG_BLIND:
        return 1 - int(button)
    raise ValueError(f"unknown role {role!r}; expected one of {ROLES}")


def _role_for_seat(button: int, seat: int) -> str:
    return ROLE_BUTTON if int(seat) == int(button) else ROLE_BIG_BLIND


def _street_for_board(board: Sequence[int]) -> int:
    try:
        return {
            0: STREET_PREFLOP,
            3: STREET_FLOP,
            4: STREET_TURN,
            5: STREET_RIVER,
        }[len(board)]
    except KeyError as error:
        raise ValueError("board must contain 0, 3, 4, or 5 cards") from error


def controlled_deck(
    *, button: int, hero: int, hero_cards: Sequence[int], board: Sequence[int]
) -> list[int]:
    """Return a full deck whose pop order deals the requested visible cards."""
    hero_cards = tuple(int(card) for card in hero_cards)
    board = tuple(int(card) for card in board)
    if len(hero_cards) != 2:
        raise ValueError("hero_cards must contain exactly two cards")
    _street_for_board(board)
    visible = hero_cards + board
    if any(card < 0 or card >= 52 for card in visible):
        raise ValueError("cards must be integers in [0, 51]")
    if len(set(visible)) != len(visible):
        raise ValueError("hero cards and board must be unique")
    if hero not in (0, 1) or button not in (0, 1):
        raise ValueError("hero and button must be seat 0 or 1")

    bb = 1 - button
    deal_order = (bb, button, bb, button)
    blocked = set(visible)
    available = [card for card in range(52) if card not in blocked]

    def filler() -> int:
        if not available:
            raise RuntimeError("ran out of filler cards")
        return available.pop(0)

    controlled: list[int] = []
    hero_index = 0
    for seat in deal_order:
        if seat == hero:
            controlled.append(hero_cards[hero_index])
            hero_index += 1
        else:
            controlled.append(filler())
    if len(board) >= 3:
        controlled.extend((filler(), board[0], board[1], board[2]))
    if len(board) >= 4:
        controlled.extend((filler(), board[3]))
    if len(board) >= 5:
        controlled.extend((filler(), board[4]))
    if len(set(controlled)) != len(controlled):
        raise RuntimeError("controlled draw sequence contains duplicate cards")
    remaining = [card for card in range(52) if card not in set(controlled)]
    deck = remaining + list(reversed(controlled))
    if len(deck) != 52 or len(set(deck)) != 52:
        raise RuntimeError("controlled deck is not a complete permutation")
    return deck


def build_decision_state(
    env: HeadsUpHoldemEngine,
    scenario: DecisionScenario,
    *,
    hero: int,
    hero_cards: Sequence[int],
):
    """Legally replay a named public line and end at the hero decision."""
    if scenario.hero_role not in ROLES:
        raise ValueError(f"invalid hero role {scenario.hero_role!r}")
    if _street_for_board(scenario.board) != scenario.expected_street:
        raise ValueError("scenario board length does not match expected street")
    button = hero if scenario.hero_role == ROLE_BUTTON else 1 - hero
    state = env.new_hand(
        button=button,
        deck=controlled_deck(
            button=button,
            hero=hero,
            hero_cards=hero_cards,
            board=scenario.board,
        ),
    )
    if set(state.hole[hero]) != set(hero_cards):
        raise RuntimeError("controlled deck did not deal the requested cards")
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
            f"scenario reached street {state.street}, "
            f"expected {scenario.expected_street}"
        )
    if tuple(state.board) != tuple(scenario.board):
        raise RuntimeError("controlled deck produced the wrong public board")
    if sum(state.stacks) + state.pot != sum(state.initial_stacks):
        raise RuntimeError("scenario violates chip conservation")
    return state


def preflop_scenarios() -> tuple[DecisionScenario, ...]:
    return (
        DecisionScenario(
            "btn_unopened",
            "BTN/SB unopened pot",
            ROLE_BUTTON,
            description="Button/small blind acts first before the flop.",
        ),
        DecisionScenario(
            "bb_vs_btn_open",
            "BB facing BTN/SB min-open",
            ROLE_BIG_BLIND,
            (LineAction(ROLE_BUTTON, ACTION_MIN_RAISE),),
        ),
        DecisionScenario(
            "bb_vs_btn_limp",
            "BB facing BTN/SB limp",
            ROLE_BIG_BLIND,
            (LineAction(ROLE_BUTTON, ACTION_CALL),),
        ),
        DecisionScenario(
            "btn_vs_bb_3bet",
            "BTN/SB facing BB three-bet",
            ROLE_BUTTON,
            (
                LineAction(ROLE_BUTTON, ACTION_MIN_RAISE),
                LineAction(ROLE_BIG_BLIND, ACTION_MIN_RAISE),
            ),
        ),
    )


def postflop_scenarios(
    *,
    flop: Sequence[int],
    turn: int | None = None,
    river: int | None = None,
) -> tuple[DecisionScenario, ...]:
    flop = tuple(int(card) for card in flop)
    if len(flop) != 3 or len(set(flop)) != 3:
        raise ValueError("flop must contain three unique cards")
    open_call = (
        LineAction(ROLE_BUTTON, ACTION_MIN_RAISE),
        LineAction(ROLE_BIG_BLIND, ACTION_CALL),
    )
    flop_checks = (
        LineAction(ROLE_BIG_BLIND, ACTION_CHECK),
        LineAction(ROLE_BUTTON, ACTION_CHECK),
    )
    scenarios: list[DecisionScenario] = [
        DecisionScenario(
            "btn_flop_checked_to",
            "BTN/SB checked to on flop",
            ROLE_BUTTON,
            open_call + (LineAction(ROLE_BIG_BLIND, ACTION_CHECK),),
            STREET_FLOP,
            flop,
        ),
        DecisionScenario(
            "bb_flop_vs_halfpot",
            "BB facing BTN/SB half-pot c-bet",
            ROLE_BIG_BLIND,
            open_call
            + (
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
                    "BTN/SB checked to on turn",
                    ROLE_BUTTON,
                    open_call
                    + flop_checks
                    + (LineAction(ROLE_BIG_BLIND, ACTION_CHECK),),
                    STREET_TURN,
                    turn_board,
                ),
                DecisionScenario(
                    "btn_turn_vs_halfpot",
                    "BTN/SB facing BB half-pot turn bet",
                    ROLE_BUTTON,
                    open_call
                    + flop_checks
                    + (LineAction(ROLE_BIG_BLIND, ACTION_HALF_POT),),
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
                "BB facing BTN/SB pot-size river bet",
                ROLE_BIG_BLIND,
                open_call
                + flop_checks
                + (
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
    if len(cards) != 2 or cards[0] == cards[1]:
        raise ValueError("cards must contain two distinct cards")
    rank_a, rank_b = cards[0] % 13, cards[1] % 13
    if rank_a == rank_b:
        index = _rank_grid_index(rank_a)
        return RANK_LABELS[index] * 2, index, index
    high, low = max(rank_a, rank_b), min(rank_a, rank_b)
    high_index, low_index = _rank_grid_index(high), _rank_grid_index(low)
    suited = cards[0] // 13 == cards[1] // 13
    label = (
        f"{RANK_LABELS[high_index]}{RANK_LABELS[low_index]}"
        f"{'s' if suited else 'o'}"
    )
    return (
        (label, high_index, low_index)
        if suited
        else (label, low_index, high_index)
    )


def _scenario_summary(
    env: HeadsUpHoldemEngine, state, scenario: DecisionScenario
) -> dict[str, Any]:
    hero = int(state.to_act)
    legal = env.legal_actions(state)
    to_call = env.amount_to_call(state, hero)
    effective_stack = min(state.stacks)
    targets = {
        ACTION_NAMES[action]: env.action_target(state, action) for action in legal
    }
    return {
        "scenario_id": scenario.scenario_id,
        "label": scenario.label,
        "hero_role": scenario.hero_role,
        "street": state.street,
        "board": " ".join(card_to_string(card) for card in state.board),
        "pot_bb": state.pot / env.big_blind,
        "to_call_bb": to_call / env.big_blind,
        "pot_odds": to_call / (state.pot + to_call) if to_call else 0.0,
        "effective_stack_bb": effective_stack / env.big_blind,
        "spr": effective_stack / max(state.pot, 1),
        "legal_actions": tuple(ACTION_NAMES[action] for action in legal),
        "action_targets_bb": {
            name: target / env.big_blind for name, target in targets.items()
        },
    }


class StrategyAnalyzer:
    def __init__(self, trainer, *, batch_size: int = 4096):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.trainer = trainer
        self.batch_size = int(batch_size)

    def _analysis_env(self) -> HeadsUpHoldemEngine:
        return type(self.trainer.env)(
            starting_stack=self.trainer.env.starting_stack,
            small_blind=self.trainer.env.small_blind,
            big_blind=self.trainer.env.big_blind,
            seed=self.trainer.seed + 90_000,
        )

    def analyze_range(
        self,
        scenario: DecisionScenario,
        *,
        policy_nets: Sequence[torch.nn.Module] | None = None,
        hero_seats: Sequence[int] = (0, 1),
    ) -> RangeReport:
        hero_seats = tuple(int(seat) for seat in hero_seats)
        if not hero_seats or any(seat not in (0, 1) for seat in hero_seats):
            raise ValueError("hero_seats must contain seat 0 or 1")
        board = set(scenario.board)
        hole_combos = [
            cards
            for cards in combinations(range(52), 2)
            if cards[0] not in board and cards[1] not in board
        ]
        env = self._analysis_env()
        states: list[Any] = []
        metadata: list[tuple[int, tuple[int, int], str, int, int]] = []
        for hero in hero_seats:
            for cards in hole_combos:
                label, row, column = classify_hole_cards(cards)
                states.append(
                    build_decision_state(
                        env, scenario, hero=hero, hero_cards=cards
                    )
                )
                metadata.append((hero, cards, label, row, column))
        predictions = self.trainer.average_policy_batch(
            states, policy_nets=policy_nets, batch_size=self.batch_size
        )
        records: list[dict[str, Any]] = []
        for state, probability, meta in zip(states, predictions, metadata):
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
                float(probability[ACTION_CALL])
                if ACTION_CALL in legal
                else float("nan")
            )
            record["p_aggressive"] = sum(
                float(probability[action]) for action in RAISE_ACTIONS
            )
            record["p_continue"] = 1.0 - record["p_fold"]
            records.append(record)
        combo_table = pd.DataFrame.from_records(records)
        probability_columns = list(
            dict.fromkeys(
                [
                    *(f"p_{name}" for name in ACTION_NAMES),
                    "p_fold",
                    "p_check",
                    "p_call",
                    "p_aggressive",
                    "p_continue",
                ]
            )
        )
        keys = ["scenario_id", "hand", "row", "column"]
        grouped = combo_table.groupby(keys, sort=False)
        hand_table = grouped[probability_columns].mean().reset_index()
        hand_table = hand_table.merge(
            grouped[probability_columns]
            .std(ddof=0)
            .add_prefix("combo_std_")
            .reset_index(),
            on=keys,
        )
        hand_table = hand_table.merge(
            grouped["combo_key"].nunique().rename("combo_count").reset_index(),
            on=keys,
        )
        by_seat = (
            combo_table.groupby(["hand", "hero_seat"])[probability_columns]
            .mean()
            .reset_index()
        )
        seat_std = (
            by_seat.groupby("hand")[probability_columns]
            .std(ddof=0)
            .add_prefix("between_net_std_")
            .reset_index()
        )
        hand_table = hand_table.merge(seat_std, on="hand", how="left")
        summary = _scenario_summary(env, states[0], scenario)
        summary["physical_combos"] = len(hole_combos)
        summary["network_samples"] = len(states)
        return RangeReport(scenario, combo_table, hand_table, summary)

    def analyze_cases(
        self,
        scenarios: Iterable[DecisionScenario],
        *,
        policy_nets: Sequence[torch.nn.Module] | None = None,
        hero_seats: Sequence[int] = (0, 1),
    ) -> list[RangeReport]:
        return [
            self.analyze_range(
                scenario,
                policy_nets=policy_nets,
                hero_seats=hero_seats,
            )
            for scenario in scenarios
        ]

    def analyze_next_cards(
        self,
        scenario: DecisionScenario,
        *,
        hero_cards: Sequence[int],
        policy_nets: Sequence[torch.nn.Module] | None = None,
        hero_seats: Sequence[int] = (0, 1),
    ) -> pd.DataFrame:
        if len(scenario.board) not in (4, 5):
            raise ValueError("next-card sweep requires a turn or river scenario")
        prefix = tuple(scenario.board[:-1])
        blocked = set(prefix) | set(hero_cards)
        env = self._analysis_env()
        states: list[Any] = []
        metadata: list[tuple[int, int]] = []
        for card in range(52):
            if card in blocked:
                continue
            varied = replace(scenario, board=prefix + (card,))
            for hero in hero_seats:
                states.append(
                    build_decision_state(
                        env, varied, hero=int(hero), hero_cards=hero_cards
                    )
                )
                metadata.append((card, int(hero)))
        predictions = self.trainer.average_policy_batch(
            states, policy_nets=policy_nets, batch_size=self.batch_size
        )
        rows: list[dict[str, Any]] = []
        for state, probability, (card, hero) in zip(
            states, predictions, metadata
        ):
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
                    "p_aggressive": sum(
                        float(probability[action]) for action in RAISE_ACTIONS
                    ),
                }
            )
        return (
            pd.DataFrame(rows)
            .groupby(["card", "card_label", "rank", "suit"], as_index=False)
            .mean(numeric_only=True)
        )


def compare_ranges(previous: RangeReport, current: RangeReport) -> pd.DataFrame:
    if previous.scenario.scenario_id != current.scenario.scenario_id:
        raise ValueError("range reports must describe the same scenario")
    keys = ["hand", "row", "column"]
    action_columns = [f"p_{name}" for name in ACTION_NAMES]
    merged = previous.hand_table[keys + action_columns].merge(
        current.hand_table[keys + action_columns],
        on=keys,
        suffixes=("_previous", "_current"),
    )
    for metric in ("p_fold", "p_call", "p_aggressive", "p_continue"):
        values = previous.hand_table[keys + [metric]].merge(
            current.hand_table[keys + [metric]],
            on=keys,
            suffixes=("_previous", "_current"),
        )
        merged[f"delta_{metric}"] = (
            values[f"{metric}_current"] - values[f"{metric}_previous"]
        )
    merged["strategy_total_variation"] = 0.5 * sum(
        (
            merged[f"{column}_current"]
            - merged[f"{column}_previous"]
        )
        .abs()
        .fillna(0.0)
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
    if not metrics:
        raise ValueError("at least one metric is required")
    figure, axes = plt.subplots(
        1, len(metrics), figsize=(6.2 * len(metrics), 5.5)
    )
    titles = {
        "p_fold": "Fold frequency",
        "p_check": "Check frequency",
        "p_call": "Call frequency",
        "p_aggressive": "Total raise frequency",
        "p_continue": "Continue frequency",
    }
    for axis, metric in zip(np.atleast_1d(axes), metrics):
        matrix = _metric_matrix(report.hand_table, metric)
        image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis")
        axis.set_xticks(range(13), RANK_LABELS)
        axis.set_yticks(range(13), RANK_LABELS)
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
        figure.colorbar(
            image,
            ax=axis,
            fraction=0.046,
            pad=0.04,
            format=PercentFormatter(1.0),
        )
    figure.suptitle(
        f"{report.scenario.label} — average policy frequency", fontsize=14
    )
    figure.tight_layout()
    return figure


def plot_call_maps(reports: Sequence[RangeReport]) -> plt.Figure:
    if not reports:
        raise ValueError("at least one report is required")
    columns = min(3, len(reports))
    rows = math.ceil(len(reports) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(5.5 * columns, 5 * rows))
    flat = np.atleast_1d(axes).ravel()
    image = None
    for axis, report in zip(flat, reports):
        image = axis.imshow(
            _metric_matrix(report.hand_table, "p_call"),
            vmin=0.0,
            vmax=1.0,
            cmap="magma",
        )
        axis.set_xticks(range(13), RANK_LABELS)
        axis.set_yticks(range(13), RANK_LABELS)
        axis.set_title(report.scenario.label)
    for axis in flat[len(reports) :]:
        axis.axis("off")
    if image is not None:
        figure.colorbar(
            image,
            ax=list(flat[: len(reports)]),
            shrink=0.72,
            format=PercentFormatter(1.0),
        )
    figure.suptitle(
        "Call frequency by starting hand (policy probability, not equity)"
    )
    figure.subplots_adjust(top=0.9, wspace=0.22, hspace=0.3)
    return figure


def plot_range_delta(
    comparison: pd.DataFrame,
    *,
    metric: str = "delta_p_call",
    title: str = "Policy-frequency change",
) -> plt.Figure:
    matrix = _metric_matrix(comparison, metric)
    finite = np.abs(matrix[np.isfinite(matrix)])
    limit = max(float(finite.max()) if finite.size else 0.0, 0.01)
    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    image = axis.imshow(matrix, vmin=-limit, vmax=limit, cmap="coolwarm")
    axis.set_xticks(range(13), RANK_LABELS)
    axis.set_yticks(range(13), RANK_LABELS)
    axis.set_title(title)
    figure.colorbar(image, ax=axis, label="current minus previous")
    figure.tight_layout()
    return figure


def plot_card_sweep(
    table: pd.DataFrame,
    *,
    metric: str = "p_call",
    title: str = "Next-card policy",
) -> plt.Figure:
    matrix = np.full((4, 13), np.nan, dtype=float)
    for row in table.itertuples(index=False):
        matrix[int(row.suit), int(row.rank)] = float(getattr(row, metric))
    figure, axis = plt.subplots(figsize=(12, 3.8))
    image = axis.imshow(
        matrix, vmin=0.0, vmax=1.0, cmap="plasma", aspect="auto"
    )
    axis.set_xticks(range(13), RANK_LABELS)
    axis.set_yticks(range(4), SUIT_LABELS)
    axis.set_title(title)
    figure.colorbar(
        image, ax=axis, label="policy frequency", format=PercentFormatter(1.0)
    )
    figure.tight_layout()
    return figure


def plot_network_architecture(trainer: Any) -> plt.Figure:
    """Draw the actual four-network compact structured architecture."""
    figure, axes = plt.subplots(1, 2, figsize=(17, 7))
    left, right = axes
    left.axis("off")
    right.axis("off")

    network_rows = (
        ("Advantage P0", "fresh regret fit"),
        ("Advantage P1", "fresh regret fit"),
        ("Policy P0", "average strategy"),
        ("Policy P1", "average strategy"),
    )
    for index, (name, purpose) in enumerate(network_rows):
        y = 0.82 - index * 0.2
        box = FancyBboxPatch(
            (0.1, y),
            0.8,
            0.12,
            boxstyle="round,pad=0.02",
            facecolor="#d9edf7" if index < 2 else "#dff0d8",
            edgecolor="#365f91",
        )
        left.add_patch(box)
        left.text(0.5, y + 0.075, name, ha="center", weight="bold")
        left.text(0.5, y + 0.035, purpose, ha="center", fontsize=9)
    left.set_xlim(0, 1)
    left.set_ylim(0, 1)
    left.set_title("Four independently trained networks")

    stages = [
        (f"Information state\n{trainer.input_dim:,} floats", "#fce5cd"),
        ("Card + poker\nrelation branches", "#fff2cc"),
        ("Public + exact-action\nbranches", "#d9ead3"),
        ("Attention + GRU\nhistory branch", "#d9d2e9"),
        (f"Fusion + {trainer.blocks}\nresidual blocks", "#cfe2f3"),
        (f"4 street-specific\n{NUM_ACTIONS}-action heads", "#ead1dc"),
    ]
    for index, (label, color) in enumerate(stages):
        x = 0.02 + index * 0.16
        box = FancyBboxPatch(
            (x, 0.39),
            0.135,
            0.2,
            boxstyle="round,pad=0.02",
            facecolor=color,
            edgecolor="#555555",
        )
        right.add_patch(box)
        right.text(x + 0.0675, 0.49, label, ha="center", va="center", fontsize=8)
        if index < len(stages) - 1:
            right.annotate(
                "",
                xy=(x + 0.16, 0.49),
                xytext=(x + 0.135, 0.49),
                arrowprops={"arrowstyle": "->", "lw": 1.5},
            )
    parameters = sum(
        parameter.numel()
        for network in trainer.advantage_nets + trainer.policy_nets
        for parameter in network.parameters()
    )
    per_network = parameters // 4
    right.text(
        0.5,
        0.22,
        (
            f"{per_network:,} parameters / "
            f"{per_network * 4 / 1024**2:.1f} MiB per network\n"
            f"{parameters:,} parameters across four networks"
        ),
        ha="center",
        fontsize=11,
    )
    right.set_xlim(0, 1)
    right.set_ylim(0, 1)
    right.set_title("Actual structured HU Deep-CFR V3 path")
    figure.tight_layout()
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
