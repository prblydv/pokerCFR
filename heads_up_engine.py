"""Exact integer-chip heads-up no-limit Texas Hold'em reference engine.

The engine deliberately separates two interfaces:

* :meth:`HeadsUpHoldemEngine.step_exact` accepts real room actions, including
  any strictly legal integer ``raise_to`` amount.
* :meth:`HeadsUpHoldemEngine.step` exposes ten fixed policy slots.  The slots
  are converted to exact state-dependent targets by the same code used for the
  legal mask and are effect-deduplicated.

Every transition deep-copies its input.  Invalid actions raise; targets are
never clamped.  Cards use ``suit * 13 + rank`` (2..A, clubs/diamonds/hearts/
spades), and a supplied deck is dealt by popping from its end.
"""

from __future__ import annotations

import copy
import itertools
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


NUM_PLAYERS = 2
NUM_ACTIONS = 10
ENGINE_SCHEMA_VERSION = "hu_nlhe_engine_v1"
ACTION_SCHEMA_VERSION = "hu_nlhe_actions_v1_10"

ACTION_FOLD = 0
ACTION_CHECK = 1
ACTION_CALL = 2
ACTION_MIN_RAISE = 3
ACTION_THIRD_POT = 4
ACTION_HALF_POT = 5
ACTION_THREE_QUARTER_POT = 6
ACTION_POT = 7
ACTION_OVERBET = 8
ACTION_ALL_IN = 9

# Discoverable aliases used by callers and prose specifications.
ACTION_RAISE_MIN = ACTION_MIN_RAISE
ACTION_33_POT = ACTION_THIRD_POT
ACTION_50_POT = ACTION_HALF_POT
ACTION_75_POT = ACTION_THREE_QUARTER_POT
ACTION_100_POT = ACTION_POT
ACTION_150_POT = ACTION_OVERBET

ACTION_NAMES = (
    "fold",
    "check",
    "call",
    "min_raise",
    "third_pot",
    "half_pot",
    "three_quarter_pot",
    "pot",
    "overbet",
    "all_in",
)

STREET_PREFLOP = 0
STREET_FLOP = 1
STREET_TURN = 2
STREET_RIVER = 3
STREET_NAMES = ("preflop", "flop", "turn", "river")

DEFAULT_STACK = 200
DEFAULT_SMALL_BLIND = 1
DEFAULT_BIG_BLIND = 2

HIGH_CARD = 0
ONE_PAIR = 1
TWO_PAIR = 2
THREE_OF_A_KIND = 3
STRAIGHT = 4
FLUSH = 5
FULL_HOUSE = 6
FOUR_OF_A_KIND = 7
STRAIGHT_FLUSH = 8


@dataclass(frozen=True)
class ActionRecord:
    """Lossless semantic history event for encoders and off-tree search."""

    player: int
    street: int
    action: Optional[int]
    kind: str
    amount: int
    raise_to: int
    contribution_after: int
    current_bet_before: int
    current_bet_after: int
    pot_before: int
    pot_after: int
    to_call_before: int
    full_raise: bool
    all_in: bool

    @property
    def action_name(self) -> str:
        return self.kind

    @property
    def amount_added(self) -> int:
        return self.amount

    @property
    def target(self) -> int:
        return self.raise_to


@dataclass
class HeadsUpState:
    deck: List[int]
    board: List[int]
    burned: List[int]
    hole: List[List[int]]
    stacks: List[int]
    initial_stacks: List[int]
    total_contrib: List[int]
    street_contrib: List[int]
    folded: List[bool]
    all_in: List[bool]
    raise_rights: List[bool]
    last_action_bet: List[Optional[int]]
    uncalled_returns: List[int]
    small_blind: int
    big_blind: int
    pot: int
    current_bet: int
    min_raise: int
    to_act: Optional[int]
    street: int
    button: int
    sb_player: int
    bb_player: int
    last_full_raiser: Optional[int]
    pending_actors: Set[int]
    history: List[ActionRecord]
    terminal: bool = False
    payoffs: Optional[List[int]] = None
    payouts: Optional[List[int]] = None
    winners: Tuple[int, ...] = ()

    @property
    def contrib(self) -> List[int]:
        return self.street_contrib

    @property
    def sb(self) -> int:
        return self.small_blind

    @property
    def bb(self) -> int:
        return self.big_blind

    @property
    def players_remaining(self) -> int:
        return sum(not folded for folded in self.folded)


GameState = HeadsUpState


@dataclass(frozen=True)
class _ActionOption:
    action: int
    kind: str
    semantic: str
    payment: int
    target: int


def _require_chip(value: object, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer chip amount")
    if value < (1 if positive else 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be {qualifier}")
    return value


def _validate_cards(cards: Sequence[int], expected: int) -> None:
    if len(cards) != expected:
        raise ValueError(f"expected {expected} cards, received {len(cards)}")
    if any(isinstance(card, bool) or not isinstance(card, int) for card in cards):
        raise TypeError("cards must be integer indices")
    if any(card < 0 or card >= 52 for card in cards):
        raise ValueError("card indices must be in the range 0..51")
    if len(set(cards)) != len(cards):
        raise ValueError("duplicate cards are not valid")


def _pack_score(category: int, kickers: Iterable[int]) -> int:
    fields = [category, *kickers]
    fields.extend([0] * (6 - len(fields)))
    score = 0
    for value in fields[:6]:
        score = score * 15 + value
    return score


def evaluate_5card(cards: Sequence[int]) -> int:
    """Return a high-is-better score for exactly five cards."""

    cards = list(cards)
    _validate_cards(cards, 5)
    ranks = [card % 13 + 2 for card in cards]
    suits = [card // 13 for card in cards]
    counts: Dict[int, int] = {}
    for rank in ranks:
        counts[rank] = counts.get(rank, 0) + 1

    unique = sorted(counts, reverse=True)
    straight_high = 0
    straight_ranks = sorted(set(ranks), reverse=True)
    if len(straight_ranks) == 5:
        if straight_ranks == [14, 5, 4, 3, 2]:
            straight_high = 5
        elif straight_ranks[0] - straight_ranks[-1] == 4:
            straight_high = straight_ranks[0]

    flush = len(set(suits)) == 1
    if flush and straight_high:
        return _pack_score(STRAIGHT_FLUSH, [straight_high])
    groups = sorted(((count, rank) for rank, count in counts.items()), reverse=True)
    if groups[0][0] == 4:
        quad = groups[0][1]
        return _pack_score(FOUR_OF_A_KIND, [quad, max(r for r in unique if r != quad)])
    if groups[0][0] == 3 and groups[1][0] == 2:
        return _pack_score(FULL_HOUSE, [groups[0][1], groups[1][1]])
    if flush:
        return _pack_score(FLUSH, sorted(ranks, reverse=True))
    if straight_high:
        return _pack_score(STRAIGHT, [straight_high])
    if groups[0][0] == 3:
        trip = groups[0][1]
        return _pack_score(
            THREE_OF_A_KIND,
            [trip, *sorted((r for r in ranks if r != trip), reverse=True)],
        )
    pairs = sorted((rank for rank, count in counts.items() if count == 2), reverse=True)
    if len(pairs) == 2:
        kicker = next(rank for rank, count in counts.items() if count == 1)
        return _pack_score(TWO_PAIR, [pairs[0], pairs[1], kicker])
    if len(pairs) == 1:
        pair = pairs[0]
        return _pack_score(
            ONE_PAIR,
            [pair, *sorted((r for r in ranks if r != pair), reverse=True)],
        )
    return _pack_score(HIGH_CARD, sorted(ranks, reverse=True))


def evaluate_7card(
    cards_or_hole: Sequence[int], board: Optional[Sequence[int]] = None
) -> int:
    cards = list(cards_or_hole)
    if board is not None:
        if len(cards) != 2 or len(board) != 5:
            raise ValueError("hole/board form requires exactly 2 and 5 cards")
        cards.extend(board)
    _validate_cards(cards, 7)
    return max(evaluate_5card(combo) for combo in itertools.combinations(cards, 5))


def card_to_string(card: int) -> str:
    _validate_cards([card], 1)
    return "23456789TJQKA"[card % 13] + "cdhs"[card // 13]


def _rounded_fraction(value: int, numerator: int, denominator: int) -> int:
    """Round a nonnegative rational chip amount half-up, using integers."""

    if value < 0 or numerator < 0 or denominator <= 0:
        raise ValueError("invalid nonnegative chip fraction")
    return (value * numerator + denominator // 2) // denominator


class HeadsUpHoldemEngine:
    """Pure-Python, immutable-transition heads-up Hold'em engine."""

    def __init__(
        self,
        starting_stack: int = DEFAULT_STACK,
        small_blind: int = DEFAULT_SMALL_BLIND,
        big_blind: int = DEFAULT_BIG_BLIND,
        seed: Optional[int] = None,
        **aliases: int,
    ) -> None:
        if "stack_size" in aliases:
            starting_stack = aliases.pop("stack_size")
        if "sb" in aliases:
            small_blind = aliases.pop("sb")
        if "bb" in aliases:
            big_blind = aliases.pop("bb")
        if aliases:
            raise TypeError(f"unexpected constructor argument(s): {', '.join(sorted(aliases))}")
        self.starting_stack = _require_chip(starting_stack, "starting_stack", positive=True)
        self.small_blind = _require_chip(small_blind, "small_blind", positive=True)
        self.big_blind = _require_chip(big_blind, "big_blind", positive=True)
        if self.small_blind >= self.big_blind:
            raise ValueError("blinds must satisfy 0 < small_blind < big_blind")
        self.stack_size = self.starting_stack
        self.sb = self.small_blind
        self.bb = self.big_blind
        self.rng = random.Random(seed)
        self._last_button = 1

    def new_hand(
        self,
        button: Optional[int] = None,
        *,
        stacks: Optional[Sequence[int]] = None,
        deck: Optional[Sequence[int]] = None,
    ) -> HeadsUpState:
        if stacks is None:
            initial = [self.starting_stack, self.starting_stack]
        else:
            if len(stacks) != NUM_PLAYERS:
                raise ValueError("stacks must contain exactly two values")
            initial = [
                _require_chip(value, f"stacks[{index}]", positive=True)
                for index, value in enumerate(stacks)
            ]
        if button is None:
            button = 1 - self._last_button
        if isinstance(button, bool) or not isinstance(button, int) or button not in (0, 1):
            raise ValueError("button must be seat 0 or 1")
        self._last_button = button
        sb_player = button
        bb_player = 1 - button

        if deck is None:
            working_deck = list(range(52))
            self.rng.shuffle(working_deck)
        else:
            working_deck = list(deck)
            _validate_cards(working_deck, 52)

        hole = [[], []]
        # Cards are dealt left of the dealer first: the BB receives first card
        # even though the button is the SB in heads-up play.
        for _ in range(2):
            for player in (bb_player, sb_player):
                hole[player].append(working_deck.pop())

        live_stacks = initial[:]
        total = [0, 0]
        street_total = [0, 0]
        for player, blind in ((sb_player, self.small_blind), (bb_player, self.big_blind)):
            posted = min(live_stacks[player], blind)
            live_stacks[player] -= posted
            total[player] += posted
            street_total[player] += posted
        all_in = [stack == 0 for stack in live_stacks]

        state = HeadsUpState(
            deck=working_deck,
            board=[],
            burned=[],
            hole=hole,
            stacks=live_stacks,
            initial_stacks=initial,
            total_contrib=total,
            street_contrib=street_total,
            folded=[False, False],
            all_in=all_in,
            raise_rights=[not value for value in all_in],
            last_action_bet=[None, None],
            uncalled_returns=[0, 0],
            small_blind=self.small_blind,
            big_blind=self.big_blind,
            pot=sum(total),
            current_bet=max(street_total),
            min_raise=self.big_blind,
            to_act=None,
            street=STREET_PREFLOP,
            button=button,
            sb_player=sb_player,
            bb_player=bb_player,
            last_full_raiser=None,
            pending_actors=set(),
            history=[],
        )
        can_act = self._can_act_players(state)
        if len(can_act) == NUM_PLAYERS:
            state.pending_actors = set(can_act)
            state.to_act = state.sb_player
        elif can_act:
            player = next(iter(can_act))
            if state.street_contrib[player] < state.current_bet:
                state.pending_actors = {player}
                state.to_act = player
            else:
                self._runout_and_showdown(state)
        else:
            self._runout_and_showdown(state)
        self._assert_chip_conservation(state)
        return state

    @staticmethod
    def clone(state: HeadsUpState) -> HeadsUpState:
        return copy.deepcopy(state)

    def amount_to_call(self, state: HeadsUpState, player: Optional[int] = None) -> int:
        if player is None:
            if state.to_act is None:
                return 0
            player = state.to_act
        if isinstance(player, bool) or not isinstance(player, int) or player not in (0, 1):
            raise ValueError("player must be seat 0 or 1")
        return max(0, state.current_bet - state.street_contrib[player])

    def _checked_actor(self, state: HeadsUpState) -> int:
        if state.terminal:
            raise RuntimeError("cannot act on a terminal state")
        player = state.to_act
        if player is None or player not in state.pending_actors:
            raise RuntimeError("non-terminal state has no valid pending actor")
        if state.folded[player] or state.all_in[player] or state.stacks[player] <= 0:
            raise RuntimeError("folded or all-in player cannot act")
        return player

    def _action_options(self, state: HeadsUpState) -> Dict[int, _ActionOption]:
        if state.terminal:
            return {}
        player = self._checked_actor(state)
        contribution = state.street_contrib[player]
        stack = state.stacks[player]
        to_call = self.amount_to_call(state, player)
        options: Dict[int, _ActionOption] = {}
        seen: Set[Tuple[str, int]] = set()

        def add(action: int, kind: str, semantic: str, payment: int, target: int) -> None:
            effect = (kind, target)
            if effect not in seen:
                seen.add(effect)
                options[action] = _ActionOption(action, kind, semantic, payment, target)

        if to_call:
            add(ACTION_FOLD, "fold", "fold", 0, contribution)
            payment = min(stack, to_call)
            add(ACTION_CALL, "commit", "call", payment, contribution + payment)
        else:
            add(ACTION_CHECK, "check", "check", 0, contribution)

        opponent = 1 - player
        max_target = contribution + stack
        opponent_can_respond = (
            not state.folded[opponent]
            and not state.all_in[opponent]
            and state.stacks[opponent] > 0
        )
        may_raise = (
            state.raise_rights[player]
            and opponent_can_respond
            and max_target > state.current_bet
        )
        if not may_raise:
            return options

        minimum_target = state.current_bet + state.min_raise
        called_payment = min(stack, to_call)
        called_target = contribution + called_payment
        pot_after_call = state.pot + called_payment

        def add_full_template(action: int, target: int) -> None:
            if minimum_target <= target <= max_target:
                add(action, "commit", ACTION_NAMES[action], target - contribution, target)

        add_full_template(ACTION_MIN_RAISE, minimum_target)
        for action, numerator, denominator in (
            (ACTION_THIRD_POT, 1, 3),
            (ACTION_HALF_POT, 1, 2),
            (ACTION_THREE_QUARTER_POT, 3, 4),
            (ACTION_POT, 1, 1),
            (ACTION_OVERBET, 3, 2),
        ):
            add_full_template(
                action,
                called_target + _rounded_fraction(pot_after_call, numerator, denominator),
            )
        # All-in is the only abstract slot allowed to be a short raise.
        add(ACTION_ALL_IN, "commit", "all_in", stack, max_target)
        return options

    def legal_actions(self, state: HeadsUpState) -> List[int]:
        return sorted(self._action_options(state))

    def legal_action_mask(self, state: HeadsUpState) -> List[int]:
        legal = set(self.legal_actions(state))
        return [int(action in legal) for action in range(NUM_ACTIONS)]

    def action_target(self, state: HeadsUpState, action: int) -> int:
        options = self._action_options(state)
        if action not in options:
            raise ValueError("illegal action")
        return options[action].target

    def action_payment(self, state: HeadsUpState, action: int) -> int:
        options = self._action_options(state)
        if action not in options:
            raise ValueError("illegal action")
        return options[action].payment

    def action_descriptors(self, state: HeadsUpState) -> List[Optional[dict]]:
        options = self._action_options(state)
        descriptors: List[Optional[dict]] = []
        for action in range(NUM_ACTIONS):
            option = options.get(action)
            if option is None:
                descriptors.append(None)
                continue
            actor = self._checked_actor(state)
            remaining = state.stacks[actor] - option.payment
            aggressive = option.target > state.current_bet
            full_raise = aggressive and option.target - state.current_bet >= state.min_raise
            descriptors.append(
                {
                    "action": action,
                    "target": option.target,
                    "payment": option.payment,
                    "resulting_pot": state.pot + option.payment,
                    "remaining_stack": remaining,
                    "resulting_effective_stack": min(remaining, state.stacks[1 - actor]),
                    "is_all_in": remaining == 0,
                    "is_aggressive": aggressive,
                    "is_full_raise": full_raise,
                    # In heads-up play a raise is actually available afterward
                    # only if this is a full wager and the aggressor retained
                    # chips with which to respond.  A short opening all-in gives
                    # the opponent a call/fold decision, not a raise branch.
                    "reopens_betting": full_raise
                    and remaining > 0
                    and state.stacks[1 - actor] > 0,
                }
            )
        return descriptors

    def step(self, old: HeadsUpState, action: int) -> HeadsUpState:
        if isinstance(action, bool) or not isinstance(action, int) or not 0 <= action < NUM_ACTIONS:
            raise ValueError("action must be an integer in 0..9")
        options = self._action_options(old)
        if action not in options:
            raise ValueError(f"illegal action {action} ({ACTION_NAMES[action]})")
        return self._apply_option(old, options[action], action)

    def step_exact(
        self, old: HeadsUpState, kind: str, raise_to: Optional[int] = None
    ) -> HeadsUpState:
        if old.terminal:
            raise RuntimeError("cannot act on a terminal state")
        if not isinstance(kind, str):
            raise TypeError("kind must be a string")
        normalized = kind.lower().replace("-", "_")
        if normalized in {"fold", "check", "call"}:
            if raise_to is not None:
                raise ValueError(f"raise_to is only valid with an exact raise, not {normalized}")
            option = self._exact_passive_option(old, normalized)
        elif normalized in {"all_in", "allin"}:
            if raise_to is not None:
                raise ValueError("raise_to is only valid with an exact raise, not all_in")
            option = self._exact_all_in_option(old)
        elif normalized in {"raise", "bet", "raise_to"}:
            if raise_to is None:
                raise ValueError("raise_to is required for an exact raise")
            target = _require_chip(raise_to, "raise_to", positive=True)
            option = self._exact_raise_option(old, target)
        else:
            raise ValueError("kind must be fold, check, call, raise_to, or all_in")
        return self._apply_option(old, option, None)

    def _exact_passive_option(self, state: HeadsUpState, requested: str) -> _ActionOption:
        player = self._checked_actor(state)
        contribution = state.street_contrib[player]
        to_call = self.amount_to_call(state, player)
        if requested == "fold":
            return _ActionOption(-1, "fold", "fold", 0, contribution)
        if requested == "check":
            if to_call:
                raise ValueError("check is illegal when facing a bet")
            return _ActionOption(-1, "check", "check", 0, contribution)
        if not to_call:
            raise ValueError("call is illegal when checking is available")
        payment = min(state.stacks[player], to_call)
        return _ActionOption(-1, "commit", "call", payment, contribution + payment)

    def _exact_all_in_option(self, state: HeadsUpState) -> _ActionOption:
        player = self._checked_actor(state)
        contribution = state.street_contrib[player]
        stack = state.stacks[player]
        target = contribution + stack
        if target <= state.current_bet:
            return _ActionOption(-1, "commit", "call", stack, target)
        self._validate_exact_raise(state, target)
        return _ActionOption(-1, "commit", "all_in", stack, target)

    def _exact_raise_option(self, state: HeadsUpState, target: int) -> _ActionOption:
        player = self._checked_actor(state)
        self._validate_exact_raise(state, target)
        semantic = "bet" if state.current_bet == 0 else "raise"
        return _ActionOption(
            -1, "commit", semantic, target - state.street_contrib[player], target
        )

    def _validate_exact_raise(self, state: HeadsUpState, target: int) -> None:
        player = self._checked_actor(state)
        opponent = 1 - player
        contribution = state.street_contrib[player]
        maximum = contribution + state.stacks[player]
        if target <= state.current_bet:
            raise ValueError("raise_to must exceed the current bet; use call instead")
        if target <= contribution:
            raise ValueError("raise_to must add chips")
        if target > maximum:
            raise ValueError("raise_to exceeds the acting player's stack")
        if not state.raise_rights[player]:
            raise ValueError("raising has not been reopened")
        if state.folded[opponent] or state.all_in[opponent] or state.stacks[opponent] <= 0:
            raise ValueError("cannot raise when the opponent cannot respond")
        minimum = state.current_bet + state.min_raise
        if target < minimum and target != maximum:
            raise ValueError("a sub-minimum raise is legal only when it is all-in")

    def _apply_option(
        self, old: HeadsUpState, option: _ActionOption, recorded_action: Optional[int]
    ) -> HeadsUpState:
        state = copy.deepcopy(old)
        player = self._checked_actor(state)
        current_before = state.current_bet
        pot_before = state.pot
        to_call_before = self.amount_to_call(state, player)
        full_raise = False

        if option.kind == "fold":
            state.folded[player] = True
            state.raise_rights[player] = False
            state.last_action_bet[player] = state.current_bet
            state.pending_actors.discard(player)
        elif option.kind == "check":
            state.raise_rights[player] = False
            state.last_action_bet[player] = state.current_bet
            state.pending_actors.discard(player)
        else:
            if option.payment < 0 or option.payment > state.stacks[player]:
                raise RuntimeError("internal action target exceeds the acting stack")
            state.stacks[player] -= option.payment
            state.street_contrib[player] += option.payment
            state.total_contrib[player] += option.payment
            state.pot += option.payment
            state.all_in[player] = state.stacks[player] == 0
            new_total = state.street_contrib[player]
            state.pending_actors.discard(player)
            if new_total > current_before:
                increment = new_total - current_before
                old_min_raise = state.min_raise
                full_raise = increment >= old_min_raise
                state.current_bet = new_total
                if full_raise:
                    state.min_raise = increment
                    state.last_full_raiser = player
                    opponent = 1 - player
                    if opponent in self._can_act_players(state):
                        state.raise_rights[opponent] = True
                opponent = 1 - player
                state.pending_actors = (
                    {opponent}
                    if opponent in self._can_act_players(state)
                    and state.street_contrib[opponent] < state.current_bet
                    else set()
                )
            state.raise_rights[player] = False
            state.last_action_bet[player] = state.current_bet

        if option.kind == "fold":
            semantic = "fold"
        elif option.kind == "check":
            semantic = "check"
        elif state.current_bet > current_before:
            semantic = "bet" if current_before == 0 else "raise"
        else:
            semantic = "call"

        state.history.append(
            ActionRecord(
                player=player,
                street=state.street,
                action=recorded_action,
                kind=semantic,
                amount=option.payment,
                raise_to=option.target,
                contribution_after=state.street_contrib[player],
                current_bet_before=current_before,
                current_bet_after=state.current_bet,
                pot_before=pot_before,
                pot_after=state.pot,
                to_call_before=to_call_before,
                full_raise=full_raise,
                all_in=state.all_in[player],
            )
        )

        opponent = 1 - player
        if state.folded[player]:
            self._award_uncontested(state, opponent)
            return state
        if state.folded[opponent]:
            self._award_uncontested(state, player)
            return state
        can_act = self._can_act_players(state)
        state.pending_actors.intersection_update(can_act)
        # Once only one player still has chips, that player has no meaningful
        # action when contributions are already matched.  In particular, a
        # short-stacked SB completing the blind all-in must run out immediately
        # instead of forcing the BB to make a synthetic check.
        if (
            len(can_act) < NUM_PLAYERS
            and all(
                state.street_contrib[player] >= state.current_bet
                for player in can_act
            )
        ):
            state.pending_actors.clear()
        if not state.pending_actors:
            self._close_betting_round(state)
        else:
            state.to_act = 1 - player
        self._assert_chip_conservation(state)
        return state

    def _close_betting_round(self, state: HeadsUpState) -> None:
        if state.street == STREET_RIVER:
            self._resolve_showdown_in_place(state)
        elif len(self._can_act_players(state)) < NUM_PLAYERS:
            self._runout_and_showdown(state)
        else:
            self._advance_street(state)

    def _advance_street(self, state: HeadsUpState) -> None:
        if state.street == STREET_PREFLOP:
            self._burn(state)
            self._deal_board(state, 3)
            state.street = STREET_FLOP
        elif state.street == STREET_FLOP:
            self._burn(state)
            self._deal_board(state, 1)
            state.street = STREET_TURN
        elif state.street == STREET_TURN:
            self._burn(state)
            self._deal_board(state, 1)
            state.street = STREET_RIVER
        else:
            raise RuntimeError("cannot advance beyond the river")
        state.street_contrib = [0, 0]
        state.current_bet = 0
        state.min_raise = self.big_blind
        state.last_full_raiser = None
        state.last_action_bet = [None, None]
        active = self._can_act_players(state)
        state.raise_rights = [player in active for player in range(NUM_PLAYERS)]
        state.pending_actors = set(active)
        state.to_act = state.bb_player  # non-button acts first postflop

    def _runout_and_showdown(self, state: HeadsUpState) -> None:
        while len(state.board) < 5:
            if not state.board:
                self._burn(state)
                self._deal_board(state, 3)
                state.street = STREET_FLOP
            elif len(state.board) == 3:
                self._burn(state)
                self._deal_board(state, 1)
                state.street = STREET_TURN
            elif len(state.board) == 4:
                self._burn(state)
                self._deal_board(state, 1)
                state.street = STREET_RIVER
            else:
                raise RuntimeError("board has an invalid number of cards")
        state.street = STREET_RIVER
        self._resolve_showdown_in_place(state)

    def resolve_showdown(self, old: HeadsUpState) -> HeadsUpState:
        if old.terminal:
            raise RuntimeError("state is already terminal")
        if len(old.board) != 5:
            raise ValueError("showdown requires a five-card board")
        state = copy.deepcopy(old)
        self._resolve_showdown_in_place(state)
        return state

    def _refund_uncalled(self, state: HeadsUpState) -> None:
        if state.total_contrib[0] == state.total_contrib[1]:
            return
        player = 0 if state.total_contrib[0] > state.total_contrib[1] else 1
        refund = state.total_contrib[player] - state.total_contrib[1 - player]
        if refund <= 0 or refund > state.pot:
            raise RuntimeError("invalid uncalled-bet refund")
        if refund > state.street_contrib[player]:
            raise RuntimeError("uncalled excess is not on the current street")
        state.total_contrib[player] -= refund
        state.street_contrib[player] -= refund
        state.current_bet = max(state.street_contrib)
        state.stacks[player] += refund
        state.pot -= refund
        state.uncalled_returns[player] += refund

    def _resolve_showdown_in_place(self, state: HeadsUpState) -> None:
        if len(state.board) != 5:
            raise RuntimeError("showdown requires five board cards")
        all_cards = state.hole[0] + state.hole[1] + state.board
        if len(all_cards) != 9 or len(set(all_cards)) != 9:
            raise ValueError("duplicate or missing cards in showdown state")
        self._refund_uncalled(state)
        scores = [
            evaluate_7card(state.hole[player], state.board)
            for player in range(NUM_PLAYERS)
        ]
        awards = [0, 0]
        if scores[0] > scores[1]:
            winners = (0,)
            awards[0] = state.pot
        elif scores[1] > scores[0]:
            winners = (1,)
            awards[1] = state.pot
        else:
            winners = (0, 1)
            awards[0] = state.pot // 2
            awards[1] = state.pot // 2
            awards[state.bb_player] += state.pot % 2
        for player in range(NUM_PLAYERS):
            state.stacks[player] += awards[player]
        self._finish(state, awards, winners)

    def _award_uncontested(self, state: HeadsUpState, winner: int) -> None:
        # Poker rooms return the unmatched part of the winner's last wager
        # before pushing the matched pot.  Final stacks are the same either
        # way, but keeping the refund and payout separate makes hand histories,
        # GUI accounting, and training diagnostics exact.
        self._refund_uncalled(state)
        awards = [0, 0]
        awards[winner] = state.pot
        state.stacks[winner] += state.pot
        self._finish(state, awards, (winner,))

    def _finish(
        self, state: HeadsUpState, awards: List[int], winners: Tuple[int, ...]
    ) -> None:
        state.pot = 0
        state.terminal = True
        state.to_act = None
        state.pending_actors.clear()
        state.payouts = awards
        state.winners = winners
        state.payoffs = [
            state.stacks[player] - state.initial_stacks[player]
            for player in range(NUM_PLAYERS)
        ]
        self._assert_chip_conservation(state)

    def terminal_payoff(self, state: HeadsUpState, player: int) -> int:
        if isinstance(player, bool) or not isinstance(player, int) or player not in (0, 1):
            raise ValueError("player must be seat 0 or 1")
        if not state.terminal or state.payoffs is None:
            raise RuntimeError("payoff is available only at a terminal state")
        return state.payoffs[player]

    @staticmethod
    def _burn(state: HeadsUpState) -> None:
        if not state.deck:
            raise RuntimeError("deck exhausted while burning")
        state.burned.append(state.deck.pop())

    @staticmethod
    def _deal_board(state: HeadsUpState, count: int) -> None:
        if len(state.deck) < count:
            raise RuntimeError("deck exhausted while dealing board")
        for _ in range(count):
            state.board.append(state.deck.pop())

    @staticmethod
    def _can_act_players(state: HeadsUpState) -> Set[int]:
        return {
            player
            for player in range(NUM_PLAYERS)
            if not state.folded[player]
            and not state.all_in[player]
            and state.stacks[player] > 0
        }

    @staticmethod
    def _assert_chip_conservation(state: HeadsUpState) -> None:
        expected = sum(state.initial_stacks)
        actual = sum(state.stacks) + state.pot
        if actual != expected:
            raise RuntimeError(
                f"chip conservation failed: expected {expected}, found {actual}"
            )
        if any(stack < 0 for stack in state.stacks):
            raise RuntimeError("negative stack detected")
        if sum(state.total_contrib) != state.pot and not state.terminal:
            raise RuntimeError("pot does not equal total contributions")
        if state.payoffs is not None and sum(state.payoffs) != 0:
            raise RuntimeError("terminal payoffs are not zero-sum")


HeadsUpHoldemEnv = HeadsUpHoldemEngine
HeadsUpPokerEnv = HeadsUpHoldemEngine


__all__ = [
    "NUM_PLAYERS",
    "NUM_ACTIONS",
    "ENGINE_SCHEMA_VERSION",
    "ACTION_SCHEMA_VERSION",
    "ACTION_FOLD",
    "ACTION_CHECK",
    "ACTION_CALL",
    "ACTION_MIN_RAISE",
    "ACTION_RAISE_MIN",
    "ACTION_THIRD_POT",
    "ACTION_HALF_POT",
    "ACTION_THREE_QUARTER_POT",
    "ACTION_POT",
    "ACTION_OVERBET",
    "ACTION_ALL_IN",
    "ACTION_33_POT",
    "ACTION_50_POT",
    "ACTION_75_POT",
    "ACTION_100_POT",
    "ACTION_150_POT",
    "ACTION_NAMES",
    "STREET_PREFLOP",
    "STREET_FLOP",
    "STREET_TURN",
    "STREET_RIVER",
    "STREET_NAMES",
    "ActionRecord",
    "HeadsUpState",
    "GameState",
    "HeadsUpHoldemEngine",
    "HeadsUpHoldemEnv",
    "HeadsUpPokerEnv",
    "evaluate_5card",
    "evaluate_7card",
    "card_to_string",
]
