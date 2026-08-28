"""A small, self-contained three-player no-limit Texas Hold'em engine.

The engine is intended for game-tree traversal and reinforcement-learning code:
``step`` never mutates its input, invalid actions raise instead of being silently
replaced, and the same action-option calculation drives both the legal mask and
execution.  Cards use the repository's compact encoding::

    card = suit * 13 + rank
    rank: 0..12 == 2..A, suit: 0..3 == clubs, diamonds, hearts, spades

Monetary values are floats so half-pot actions remain exact even when the pot is
odd.  With integral actions they remain integral apart from tied-pot division.
"""

from __future__ import annotations

import copy
import itertools
import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

NUM_PLAYERS = 3
NUM_ACTIONS = 9

ACTION_FOLD = 0
ACTION_CHECK = 1
ACTION_CALL = 2
ACTION_MIN_RAISE = 3
ACTION_RAISE_2X = 4
ACTION_RAISE_3X = 5
ACTION_HALF_POT = 6
ACTION_POT = 7
ACTION_ALL_IN = 8

# Convenient spelling used by some callers.
ACTION_RAISE_MIN = ACTION_MIN_RAISE

ACTION_NAMES = (
    "fold",
    "check",
    "call",
    "min_raise",
    "raise_2x",
    "raise_3x",
    "half_pot",
    "pot",
    "all_in",
)

STREET_PREFLOP = 0
STREET_FLOP = 1
STREET_TURN = 2
STREET_RIVER = 3
STREET_NAMES = ("preflop", "flop", "turn", "river")

DEFAULT_STACK = 200.0
DEFAULT_SMALL_BLIND = 1.0
DEFAULT_BIG_BLIND = 2.0

EPSILON = 1e-9


# Hand categories.  A larger category (and a larger returned score) is better.
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
    """One player decision, retained for encoders and notebook inspection."""

    player: int
    street: int
    action: int
    action_name: str
    amount: float
    contribution_after: float
    current_bet_before: float
    current_bet_after: float
    pot_after: float
    full_raise: bool = False


@dataclass(frozen=True)
class SidePot:
    """A contribution tranche and the players allowed to win it."""

    amount: float
    cap: float
    contributors: Tuple[int, ...]
    eligible: Tuple[int, ...]


@dataclass
class ThreePlayerState:
    """Complete public state for a hand.

    ``pending_actors`` and ``raise_rights`` make multiway betting-round state
    explicit.  A player whose raise right is false can still fold or call; one
    or more short all-ins reopen raising only after their cumulative increase
    reaches ``min_raise``.
    """

    deck: List[int]
    board: List[int]
    hole: List[List[int]]
    stacks: List[float]
    initial_stacks: List[float]
    total_contrib: List[float]
    street_contrib: List[float]
    folded: List[bool]
    all_in: List[bool]
    pot: float
    current_bet: float
    min_raise: float
    to_act: Optional[int]
    street: int
    button: int
    sb_player: int
    bb_player: int
    pending_actors: Set[int]
    raise_rights: List[bool]
    last_action_bet: List[Optional[float]]
    last_full_raiser: Optional[int]
    history: List[ActionRecord]
    burned: List[int]
    # Tournament membership is distinct from hand status.  A live player can
    # be all-in (and temporarily have a zero stack) without being eliminated;
    # elimination is known only once the hand has settled.
    alive: List[bool]
    eliminated: List[bool]
    terminal: bool = False
    payoffs: Optional[List[float]] = None
    payouts: Optional[List[float]] = None
    winners: Tuple[int, ...] = ()

    @property
    def contrib(self) -> List[float]:
        """Compatibility alias for per-street contributions."""

        return self.street_contrib

    @property
    def players_remaining(self) -> int:
        """Number of seats still alive in the surrounding tournament."""

        return sum(self.alive)


# A shorter name is useful in type annotations and mirrors the old engine.
GameState = ThreePlayerState


# ---------------------------------------------------------------------------
# Pure-Python hand evaluator
# ---------------------------------------------------------------------------

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
    """Pack a category and rank kickers into a high-is-better integer."""

    fields = [category, *kickers]
    fields.extend([0] * (6 - len(fields)))
    score = 0
    for value in fields[:6]:
        score = score * 15 + value
    return score


def evaluate_5card(cards: Sequence[int]) -> int:
    """Return a high-is-better integer score for exactly five cards."""

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
        kicker = max(rank for rank in unique if rank != quad)
        return _pack_score(FOUR_OF_A_KIND, [quad, kicker])

    if groups[0][0] == 3 and groups[1][0] == 2:
        return _pack_score(FULL_HOUSE, [groups[0][1], groups[1][1]])

    if flush:
        return _pack_score(FLUSH, sorted(ranks, reverse=True))

    if straight_high:
        return _pack_score(STRAIGHT, [straight_high])

    if groups[0][0] == 3:
        trip = groups[0][1]
        kickers = sorted((rank for rank in ranks if rank != trip), reverse=True)
        return _pack_score(THREE_OF_A_KIND, [trip, *kickers])

    pairs = sorted((rank for rank, count in counts.items() if count == 2), reverse=True)
    if len(pairs) == 2:
        kicker = next(rank for rank, count in counts.items() if count == 1)
        return _pack_score(TWO_PAIR, [pairs[0], pairs[1], kicker])

    if len(pairs) == 1:
        pair = pairs[0]
        kickers = sorted((rank for rank in ranks if rank != pair), reverse=True)
        return _pack_score(ONE_PAIR, [pair, *kickers])

    return _pack_score(HIGH_CARD, sorted(ranks, reverse=True))


def evaluate_7card(
    cards_or_hole: Sequence[int], board: Optional[Sequence[int]] = None
) -> int:
    """Return the best high-is-better five-card score from seven cards.

    Both ``evaluate_7card(seven_cards)`` and the convenient repository-style
    ``evaluate_7card(two_hole_cards, five_board_cards)`` are accepted.
    """

    cards = list(cards_or_hole)
    if board is not None:
        if len(cards) != 2 or len(board) != 5:
            raise ValueError("hole/board form requires exactly 2 and 5 cards")
        cards.extend(board)
    _validate_cards(cards, 7)
    return max(evaluate_5card(combo) for combo in itertools.combinations(cards, 5))


def card_to_string(card: int) -> str:
    """Return a compact human-readable card such as ``As``."""

    _validate_cards([card], 1)
    return "23456789TJQKA"[card % 13] + "cdhs"[card // 13]


def calculate_side_pots(
    contributions: Sequence[float], folded: Sequence[bool]
) -> List[SidePot]:
    """Build main/side-pot tranches, including one-player refund tranches."""

    if len(contributions) != NUM_PLAYERS or len(folded) != NUM_PLAYERS:
        raise ValueError("side-pot inputs must contain exactly three players")
    values = [float(value) for value in contributions]
    if any(value < -EPSILON for value in values):
        raise ValueError("contributions cannot be negative")

    levels = sorted({value for value in values if value > EPSILON})
    pots: List[SidePot] = []
    previous = 0.0
    for level in levels:
        contributors = tuple(
            player for player, value in enumerate(values) if value + EPSILON >= level
        )
        amount = (level - previous) * len(contributors)
        eligible = tuple(player for player in contributors if not folded[player])
        if amount > EPSILON:
            pots.append(SidePot(amount, level, contributors, eligible))
        previous = level
    return pots


@dataclass(frozen=True)
class _ActionOption:
    action: int
    kind: str
    payment: float
    target: float


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class ThreePlayerHoldemEnv:
    """Three-handed discrete-action no-limit Hold'em environment."""

    def __init__(
        self,
        starting_stack: float = DEFAULT_STACK,
        small_blind: float = DEFAULT_SMALL_BLIND,
        big_blind: float = DEFAULT_BIG_BLIND,
        seed: Optional[int] = None,
        **legacy_kwargs: float,
    ) -> None:
        # Friendly aliases make notebook/trainer construction less brittle.
        if "stack_size" in legacy_kwargs:
            starting_stack = legacy_kwargs.pop("stack_size")
        if "sb" in legacy_kwargs:
            small_blind = legacy_kwargs.pop("sb")
        if "bb" in legacy_kwargs:
            big_blind = legacy_kwargs.pop("bb")
        if legacy_kwargs:
            unknown = ", ".join(sorted(legacy_kwargs))
            raise TypeError(f"unexpected constructor argument(s): {unknown}")

        self.starting_stack = float(starting_stack)
        self.small_blind = float(small_blind)
        self.big_blind = float(big_blind)
        if self.starting_stack <= 0:
            raise ValueError("starting_stack must be positive")
        if not (0 < self.small_blind < self.big_blind):
            raise ValueError("blinds must satisfy 0 < small_blind < big_blind")
        self.rng = random.Random(seed)
        self._last_button = NUM_PLAYERS - 1

        # Common shorthand attributes.
        self.stack_size = self.starting_stack
        self.sb = self.small_blind
        self.bb = self.big_blind

    def new_hand(
        self,
        button: Optional[int] = None,
        *,
        stacks: Optional[Sequence[float]] = None,
        deck: Optional[Sequence[int]] = None,
    ) -> ThreePlayerState:
        """Deal a hand, rotating the button unless an explicit seat is given.

        ``stacks`` and ``deck`` are optional deterministic-test conveniences;
        ordinary training should omit both.  A zero stack marks an eliminated
        tournament seat, and at least two seats must be live.  A supplied deck
        is popped from its end, just like the internally shuffled deck.
        """

        if stacks is None:
            initial = [self.starting_stack] * NUM_PLAYERS
        else:
            if len(stacks) != NUM_PLAYERS:
                raise ValueError("stacks must contain exactly three values")
            initial = [float(value) for value in stacks]
            if any(not math.isfinite(value) or value < 0 for value in initial):
                raise ValueError("starting stacks must be finite and nonnegative")
            # Terminal settlement already snaps nanoscopic residues to zero.
            # Do the same for caller-supplied tournament stacks so such a seat
            # cannot retain chips while being treated as eliminated.
            initial = [0.0 if value <= EPSILON else value for value in initial]

        alive = [value > EPSILON for value in initial]
        eliminated = [not value for value in alive]
        live_players = {player for player, value in enumerate(alive) if value}
        if len(live_players) < 2:
            raise ValueError("at least two players must have a positive stack")

        if button is None:
            button = self._next_clockwise(self._last_button, live_players)
            assert button is not None
        if isinstance(button, bool) or not isinstance(button, int) or not 0 <= button < 3:
            raise ValueError("button must be seat 0, 1, or 2")
        if button not in live_players:
            raise ValueError("button must be assigned to a live player")
        self._last_button = button

        if deck is None:
            working_deck = list(range(52))
            self.rng.shuffle(working_deck)
        else:
            working_deck = list(deck)
            _validate_cards(working_deck, 52)

        if len(live_players) == 2:
            # In heads-up Hold'em the dealer posts the small blind.  Starting
            # action clockwise from the big blind below therefore also makes
            # the button act first preflop.
            sb_player = button
            bb_player = self._next_clockwise(button, live_players)
            assert bb_player is not None
        else:
            sb_player = self._next_clockwise(button, live_players)
            assert sb_player is not None
            bb_player = self._next_clockwise(sb_player, live_players)
            assert bb_player is not None

        # Deal clockwise from the small blind, one card at a time.
        hole = [[] for _ in range(NUM_PLAYERS)]
        deal_order = []
        player = sb_player
        for _ in range(len(live_players)):
            deal_order.append(player)
            next_player = self._next_clockwise(player, live_players)
            assert next_player is not None
            player = next_player
        for _ in range(2):
            for player in deal_order:
                hole[player].append(working_deck.pop())

        live_stacks = initial[:]
        total = [0.0] * NUM_PLAYERS
        street_total = [0.0] * NUM_PLAYERS
        for player, blind in ((sb_player, self.small_blind), (bb_player, self.big_blind)):
            posted = min(live_stacks[player], blind)
            live_stacks[player] -= posted
            total[player] += posted
            street_total[player] += posted

        all_in = [
            alive[player] and stack <= EPSILON
            for player, stack in enumerate(live_stacks)
        ]
        for player in range(NUM_PLAYERS):
            if all_in[player]:
                live_stacks[player] = 0.0
        pending = {player for player in live_players if not all_in[player]}
        first_to_act = self._next_clockwise(bb_player, pending)

        state = ThreePlayerState(
            deck=working_deck,
            board=[],
            hole=hole,
            stacks=live_stacks,
            initial_stacks=initial,
            total_contrib=total,
            street_contrib=street_total,
            # Marking non-participants folded preserves compatibility with
            # encoders and side-pot code that historically used this flag as
            # the only eligibility test.  ``eliminated`` retains the semantic
            # distinction between a bust-out and a fold made in this hand.
            folded=eliminated[:],
            all_in=all_in,
            pot=sum(total),
            # The full big blind remains the preflop bring-in even when the
            # BB happened to post a short all-in blind.
            current_bet=self.big_blind,
            min_raise=self.big_blind,
            to_act=first_to_act,
            street=STREET_PREFLOP,
            button=button,
            sb_player=sb_player,
            bb_player=bb_player,
            pending_actors=pending,
            raise_rights=[
                alive[player] and not all_in[player]
                for player in range(NUM_PLAYERS)
            ],
            last_action_bet=[None] * NUM_PLAYERS,
            last_full_raiser=None,
            history=[],
            burned=[],
            alive=alive,
            eliminated=eliminated,
        )

        # This matters only for deliberately tiny test stacks.
        if state.to_act is None:
            self._runout_and_showdown(state)
        return state

    @staticmethod
    def clone(state: ThreePlayerState) -> ThreePlayerState:
        """Return an independent deep copy suitable for a game-tree branch."""

        return copy.deepcopy(state)

    def amount_to_call(self, state: ThreePlayerState, player: Optional[int] = None) -> float:
        if player is None:
            if state.to_act is None:
                return 0.0
            player = state.to_act
        return max(0.0, state.current_bet - state.street_contrib[player])

    def legal_actions(self, state: ThreePlayerState) -> List[int]:
        """Return legal, effect-deduplicated action IDs in ascending order."""

        if state.terminal:
            return []
        return sorted(self._action_options(state))

    def legal_action_mask(self, state: ThreePlayerState) -> List[int]:
        legal = set(self.legal_actions(state))
        return [int(action in legal) for action in range(NUM_ACTIONS)]

    def action_target(self, state: ThreePlayerState, action: int) -> float:
        """Return the resulting street contribution for a legal action."""

        options = self._action_options(state)
        if action not in options:
            raise ValueError(f"illegal action {action} ({self._action_label(action)})")
        return options[action].target

    def _action_options(self, state: ThreePlayerState) -> Dict[int, _ActionOption]:
        if state.terminal:
            return {}
        player = state.to_act
        if player is None or player not in state.pending_actors:
            raise RuntimeError("non-terminal state has no valid pending actor")
        if state.folded[player] or state.all_in[player] or state.stacks[player] <= EPSILON:
            raise RuntimeError("folded or all-in player cannot be to_act")

        contribution = state.street_contrib[player]
        stack = state.stacks[player]
        to_call = self.amount_to_call(state, player)
        options: Dict[int, _ActionOption] = {}
        seen_effects: Set[Tuple[str, int]] = set()

        def add(action: int, kind: str, payment: float, target: float) -> None:
            # A nanodollar-scale quantization is used only to identify aliases;
            # execution retains the original float target.
            effect = (kind, round(float(target) * 1_000_000_000))
            if effect not in seen_effects:
                seen_effects.add(effect)
                options[action] = _ActionOption(action, kind, float(payment), float(target))

        if to_call > EPSILON:
            add(ACTION_FOLD, "fold", 0.0, contribution)
            call_payment = min(stack, to_call)
            add(ACTION_CALL, "commit", call_payment, contribution + call_payment)
        else:
            add(ACTION_CHECK, "check", 0.0, contribution)

        max_target = contribution + stack
        may_raise = state.raise_rights[player] and max_target > state.current_bet + EPSILON
        if not may_raise:
            return options

        minimum_target = state.current_bet + state.min_raise
        base = state.current_bet if state.current_bet > EPSILON else self.big_blind
        candidates = (
            (ACTION_MIN_RAISE, minimum_target),
            (ACTION_RAISE_2X, 2.0 * base),
            (ACTION_RAISE_3X, 3.0 * base),
            (
                ACTION_HALF_POT,
                contribution + to_call + max(0.5 * (state.pot + to_call), self.big_blind if state.current_bet <= EPSILON else 0.0),
            ),
            (
                ACTION_POT,
                contribution + to_call + max(state.pot + to_call, self.big_blind if state.current_bet <= EPSILON else 0.0),
            ),
        )
        for action, target in candidates:
            # Named raises are never silently truncated to all-in and must be a
            # full legal raise.  This makes mask and execution identical.
            if target <= max_target + EPSILON and target >= minimum_target - EPSILON:
                target = min(target, max_target)
                add(action, "commit", target - contribution, target)

        # All-in may be a short raise.  If it is exactly a call or named target,
        # the earlier, lower action ID remains the single canonical action.
        add(ACTION_ALL_IN, "commit", stack, max_target)
        return options

    def step(self, old: ThreePlayerState, action: int) -> ThreePlayerState:
        """Apply one strictly legal action to a deep copy of ``old``."""

        if old.terminal:
            raise RuntimeError("cannot act on a terminal state")
        if isinstance(action, bool) or not isinstance(action, int) or not 0 <= action < NUM_ACTIONS:
            raise ValueError(f"action must be an integer in 0..{NUM_ACTIONS - 1}")
        options = self._action_options(old)
        if action not in options:
            legal = ", ".join(ACTION_NAMES[value] for value in sorted(options))
            raise ValueError(f"illegal action {action} ({ACTION_NAMES[action]}); legal: {legal}")

        state = copy.deepcopy(old)
        option = options[action]
        player = state.to_act
        assert player is not None
        current_before = state.current_bet
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
            payment = option.payment
            if payment < -EPSILON or payment > state.stacks[player] + EPSILON:
                raise RuntimeError("internal action target exceeds stack")
            payment = min(payment, state.stacks[player])
            state.stacks[player] -= payment
            if state.stacks[player] <= EPSILON:
                state.stacks[player] = 0.0
                state.all_in[player] = True
            state.street_contrib[player] += payment
            state.total_contrib[player] += payment
            state.pot += payment
            new_total = state.street_contrib[player]

            state.pending_actors.discard(player)
            if new_total > current_before + EPSILON:
                increment = new_total - current_before
                full_raise = increment + EPSILON >= state.min_raise
                state.current_bet = new_total
                old_min_raise = state.min_raise
                if full_raise:
                    state.min_raise = increment
                    state.last_full_raiser = player
                    for other in self._can_act_players(state):
                        if other != player:
                            state.raise_rights[other] = True
                elif current_before <= EPSILON:
                    # A sub-minimum opening all-in opens betting to checkers.
                    for other in self._can_act_players(state):
                        if other != player:
                            state.raise_rights[other] = True
                else:
                    # Cumulative short all-ins can eventually amount to a full
                    # raise relative to a player's previous decision.
                    for other in self._can_act_players(state):
                        prior = state.last_action_bet[other]
                        if (
                            other != player
                            and not state.raise_rights[other]
                            and prior is not None
                            and new_total - prior + EPSILON >= old_min_raise
                        ):
                            state.raise_rights[other] = True

                state.pending_actors = {
                    other
                    for other in self._can_act_players(state)
                    if other != player
                    and state.street_contrib[other] + EPSILON < state.current_bet
                }

            state.raise_rights[player] = False
            state.last_action_bet[player] = state.current_bet

        state.history.append(
            ActionRecord(
                player=player,
                street=state.street,
                action=action,
                action_name=ACTION_NAMES[action],
                amount=option.payment,
                contribution_after=state.street_contrib[player],
                current_bet_before=current_before,
                current_bet_after=state.current_bet,
                pot_after=state.pot,
                full_raise=full_raise,
            )
        )

        remaining = [
            seat
            for seat in range(NUM_PLAYERS)
            if state.alive[seat] and not state.folded[seat]
        ]
        if len(remaining) == 1:
            self._award_uncontested(state, remaining[0])
            return state

        # Remove actors who became unable to act, then close or continue.
        state.pending_actors.intersection_update(self._can_act_players(state))
        if not state.pending_actors:
            self._close_betting_round(state)
        else:
            state.to_act = self._next_clockwise(player, state.pending_actors)

        self._assert_chip_conservation(state)
        return state

    def resolve_showdown(self, old: ThreePlayerState) -> ThreePlayerState:
        """Resolve a complete-board state on a copy (useful in tests/tools)."""

        state = copy.deepcopy(old)
        if state.terminal:
            raise RuntimeError("state is already terminal")
        if len(state.board) != 5:
            raise ValueError("showdown requires a five-card board")
        self._resolve_showdown_in_place(state)
        return state

    def terminal_payoff(self, state: ThreePlayerState, player: int) -> float:
        if not 0 <= player < NUM_PLAYERS:
            raise ValueError("player must be seat 0, 1, or 2")
        if not state.terminal or state.payoffs is None:
            raise RuntimeError("payoff is available only at a terminal state")
        return state.payoffs[player]

    def _close_betting_round(self, state: ThreePlayerState) -> None:
        if state.street == STREET_RIVER:
            self._resolve_showdown_in_place(state)
            return

        if len(self._can_act_players(state)) < 2:
            self._runout_and_showdown(state)
            return

        self._advance_street(state)

    def _advance_street(self, state: ThreePlayerState) -> None:
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

        state.street_contrib = [0.0] * NUM_PLAYERS
        state.current_bet = 0.0
        state.min_raise = self.big_blind
        state.last_full_raiser = None
        state.last_action_bet = [None] * NUM_PLAYERS
        can_act = self._can_act_players(state)
        state.raise_rights = [player in can_act for player in range(NUM_PLAYERS)]
        state.pending_actors = set(can_act)
        state.to_act = self._next_clockwise(state.button, state.pending_actors)

    def _runout_and_showdown(self, state: ThreePlayerState) -> None:
        while len(state.board) < 5:
            if len(state.board) == 0:
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

    def _resolve_showdown_in_place(self, state: ThreePlayerState) -> None:
        if len(state.board) != 5:
            raise RuntimeError("showdown requires five board cards")
        all_cards = [card for cards in state.hole for card in cards] + state.board
        if len(all_cards) != len(set(all_cards)):
            raise ValueError("duplicate cards in showdown state")

        scores = {
            player: evaluate_7card(state.hole[player], state.board)
            for player in range(NUM_PLAYERS)
            if state.alive[player] and not state.folded[player]
        }
        awards = [0.0] * NUM_PLAYERS
        all_winners: Set[int] = set()
        side_pots = calculate_side_pots(state.total_contrib, state.folded)
        side_total = sum(side.amount for side in side_pots)
        if abs(side_total - state.pot) > 1e-7:
            raise RuntimeError("pot does not match total contributions")

        for side in side_pots:
            if not side.eligible:
                raise RuntimeError("invalid side pot has no eligible player")
            best = max(scores[player] for player in side.eligible)
            winners = tuple(player for player in side.eligible if scores[player] == best)
            share = side.amount / len(winners)
            for player in winners:
                awards[player] += share
                all_winners.add(player)

        for player, award in enumerate(awards):
            state.stacks[player] += award
        self._finish_terminal(state, awards, tuple(sorted(all_winners)))

    def _award_uncontested(self, state: ThreePlayerState, winner: int) -> None:
        awards = [0.0] * NUM_PLAYERS
        awards[winner] = state.pot
        state.stacks[winner] += state.pot
        self._finish_terminal(state, awards, (winner,))

    def _finish_terminal(
        self, state: ThreePlayerState, awards: List[float], winners: Tuple[int, ...]
    ) -> None:
        state.pot = 0.0
        state.terminal = True
        state.to_act = None
        state.pending_actors.clear()
        state.payouts = awards
        state.winners = winners
        state.payoffs = [
            state.stacks[player] - state.initial_stacks[player]
            for player in range(NUM_PLAYERS)
        ]
        # A zero stack during betting means all-in, not eliminated.  Once all
        # pots are awarded the result is final and can seed the next hand.
        state.alive = [stack > EPSILON for stack in state.stacks]
        state.eliminated = [not value for value in state.alive]
        self._assert_chip_conservation(state)

    def _burn(self, state: ThreePlayerState) -> None:
        if not state.deck:
            raise RuntimeError("deck exhausted while burning")
        state.burned.append(state.deck.pop())

    @staticmethod
    def _deal_board(state: ThreePlayerState, count: int) -> None:
        if len(state.deck) < count:
            raise RuntimeError("deck exhausted while dealing board")
        for _ in range(count):
            state.board.append(state.deck.pop())

    @staticmethod
    def _next_clockwise(start: int, candidates: Set[int]) -> Optional[int]:
        for distance in range(1, NUM_PLAYERS + 1):
            player = (start + distance) % NUM_PLAYERS
            if player in candidates:
                return player
        return None

    @staticmethod
    def _can_act_players(state: ThreePlayerState) -> Set[int]:
        return {
            player
            for player in range(NUM_PLAYERS)
            if state.alive[player]
            and not state.folded[player]
            and not state.all_in[player]
            and state.stacks[player] > EPSILON
        }

    @staticmethod
    def _action_label(action: object) -> str:
        if isinstance(action, int) and 0 <= action < NUM_ACTIONS:
            return ACTION_NAMES[action]
        return "unknown"

    @staticmethod
    def _assert_chip_conservation(state: ThreePlayerState) -> None:
        if len(state.alive) != NUM_PLAYERS or len(state.eliminated) != NUM_PLAYERS:
            raise RuntimeError("tournament status must contain exactly three seats")
        if any(
            alive == eliminated
            for alive, eliminated in zip(state.alive, state.eliminated)
        ):
            raise RuntimeError("alive and eliminated status must be complementary")
        expected = sum(state.initial_stacks)
        actual = sum(state.stacks) + state.pot
        if abs(actual - expected) > 1e-7:
            raise RuntimeError(
                f"chip conservation failed: expected {expected}, found {actual}"
            )
        if any(stack < -EPSILON for stack in state.stacks):
            raise RuntimeError("negative stack detected")
        if state.terminal and state.payoffs is not None and abs(sum(state.payoffs)) > 1e-7:
            raise RuntimeError("terminal payoffs are not zero-sum")


# Alternative names make intent discoverable without forcing one spelling.
ThreePlayerPokerEnv = ThreePlayerHoldemEnv
ThreePlayerHoldemEngine = ThreePlayerHoldemEnv


__all__ = [
    "NUM_PLAYERS",
    "NUM_ACTIONS",
    "ACTION_FOLD",
    "ACTION_CHECK",
    "ACTION_CALL",
    "ACTION_MIN_RAISE",
    "ACTION_RAISE_MIN",
    "ACTION_RAISE_2X",
    "ACTION_RAISE_3X",
    "ACTION_HALF_POT",
    "ACTION_POT",
    "ACTION_ALL_IN",
    "ACTION_NAMES",
    "STREET_PREFLOP",
    "STREET_FLOP",
    "STREET_TURN",
    "STREET_RIVER",
    "STREET_NAMES",
    "ActionRecord",
    "SidePot",
    "ThreePlayerState",
    "GameState",
    "ThreePlayerHoldemEnv",
    "ThreePlayerPokerEnv",
    "ThreePlayerHoldemEngine",
    "evaluate_5card",
    "evaluate_7card",
    "card_to_string",
    "calculate_side_pots",
]
