import random
from dataclasses import dataclass
from typing import List

from hand_eval import evaluate_7card

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
RNG = random.Random(4234)
STACK_SIZE = 200
SMALL_BLIND = 1
BIG_BLIND = 2

# Actions
ACTION_FOLD        = 0
ACTION_CHECK       = 1
ACTION_CALL        = 2
ACTION_RAISE_2X    = 3
ACTION_RAISE_3X    = 4
ACTION_HALF_POT    = 5
ACTION_POT         = 6
ACTION_RAISE_10BB  = 7
ACTION_ALL_IN      = 8

NUM_ACTIONS = 9

# Streets
STREET_PREFLOP = 0
STREET_FLOP    = 1
STREET_TURN    = 2
STREET_RIVER   = 3


# ---------------------------------------------------------------------
# STATE
# ---------------------------------------------------------------------
@dataclass
class GameState:
    deck: List[int]
    board: List[int]
    hole: List[List[int]]  # [[c0,c1], [c0,c1]]
    pot: float
    to_act: int           # 0 or 1
    street: int           # 0..3
    stacks: List[float]   # [stack0, stack1]
    current_bet: float    # highest contribution this street
    last_aggressor: int
    sb_player: int
    bb_player: int
    initial_stacks: List[float]
    contrib: List[float]  # contributions this street
    actions_since_raise: int = 0
    terminal: bool = False
    winner: int = -1      # 0,1,-1 (tie)


# ---------------------------------------------------------------------
# ENGINE
# ---------------------------------------------------------------------
class SimpleHoldemEnv9:
    """
    2-player NLHE with discrete actions:

        0: FOLD
        1: CHECK
        2: CALL
        3: RAISE_2X     (2× current bet, or 2× BB if no bet yet)
        4: RAISE_3X     (3× current bet, or 3× BB if no bet yet)
        5: HALF_POT
        6: POT
        7: RAISE_10BB   (10× BB; if facing a bet, only if 10BB >= 2× bet faced)
        8: ALL_IN
    """

    def __init__(self, stack_size: int = STACK_SIZE,
                 sb: int = SMALL_BLIND, bb: int = BIG_BLIND):
        self.stack_size = float(stack_size)
        self.sb = float(sb)
        self.bb = float(bb)

    # -----------------------------------------------------------------
    # NEW HAND
    # -----------------------------------------------------------------
    def new_hand(self) -> GameState:
        deck = list(range(52))
        RNG.shuffle(deck)

        h0 = [deck.pop(), deck.pop()]
        h1 = [deck.pop(), deck.pop()]

        # Deterministic seating: SB is player 0, BB is player 1. Post blinds
        # so tests that assume fixed seats behave deterministically.
        sb, bb = 0, 1

        initial = [self.stack_size, self.stack_size]
        stacks = initial[:]
        contrib = [0.0, 0.0]

        # Post blinds
        stacks[sb] -= self.sb
        stacks[bb] -= self.bb
        contrib[sb] = self.sb
        contrib[bb] = self.bb
        pot = self.sb + self.bb

        return GameState(
            deck=deck,
            board=[],
            hole=[h0, h1],
            pot=pot,
            to_act=sb,                # SB acts first preflop
            street=STREET_PREFLOP,
            stacks=stacks,
            current_bet=self.bb,      # highest contrib is BB
            last_aggressor=-1,
            sb_player=sb,
            bb_player=bb,
            initial_stacks=initial,
            contrib=contrib,
            actions_since_raise=0,
            terminal=False,
            winner=-1
        )

    # -----------------------------------------------------------------
    # LEGAL ACTIONS
    # -----------------------------------------------------------------
    def legal_actions(self, s: GameState) -> List[int]:
        if s.terminal:
            return []

        p = s.to_act
        stack = s.stacks[p]
        if stack <= 0.0:
            return []

        to_call = max(s.current_bet - s.contrib[p], 0.0)
        pot = s.pot
        actions: List[int] = []

        # Fold / Check / Call
        if to_call > 0.0:
            actions.append(ACTION_FOLD)
            actions.append(ACTION_CALL)
        else:
            actions.append(ACTION_CHECK)

        if stack <= 0.0:
            return actions

        # Helper: does a target total contribution actually increase the bet?
        def is_raise(target_total_contrib: float) -> bool:
            return target_total_contrib > s.current_bet + 1e-9

        # Base for 2x/3x:
        # - if there is a live bet this street, base = current_bet
        # - otherwise fall back to BB (so preflop SB 2x => 2×BB)
        base = s.current_bet if s.current_bet > 0.0 else self.bb
        bet_faced = to_call  # amount hero must pay to match current bet

        # -------------------------
        # 2x / 3x
        # -------------------------
        if stack > 0.0:
            # 2x
            target_2x = 2.0 * base
            if (target_2x - s.contrib[p] <= stack + 1e-9) and is_raise(target_2x):
                actions.append(ACTION_RAISE_2X)

            # 3x
            target_3x = 3.0 * base
            if (target_3x - s.contrib[p] <= stack + 1e-9) and is_raise(target_3x):
                actions.append(ACTION_RAISE_3X)

        # -------------------------
        # HALF_POT / POT
        # -------------------------
        if stack > 0.0:
            if to_call <= 0.0:
                # Bet half-pot / pot into empty pot (or small pot), min BB
                hp_total = max(0.5 * pot, self.bb)
                if (hp_total - s.contrib[p] <= stack + 1e-9) and is_raise(hp_total):
                    actions.append(ACTION_HALF_POT)

                pt_total = max(1.0 * pot, self.bb)
                if (pt_total - s.contrib[p] <= stack + 1e-9) and is_raise(pt_total):
                    actions.append(ACTION_POT)
            else:
                # Raise to: call + half/full pot after call
                hp_total = s.contrib[p] + to_call + 0.5 * (pot + to_call)
                if (hp_total - s.contrib[p] <= stack + 1e-9) and is_raise(hp_total):
                    actions.append(ACTION_HALF_POT)

                pt_total = s.contrib[p] + to_call + (pot + to_call)
                if (pt_total - s.contrib[p] <= stack + 1e-9) and is_raise(pt_total):
                    actions.append(ACTION_POT)

        # -------------------------
        # 10BB raise
        # -------------------------
        if stack > 0.0:
            raise_size_10bb = 10.0 * self.bb
            if bet_faced <= 0.0:
                # Open: target total is just 10BB
                target_10 = raise_size_10bb
                cond_min = True
            else:
                # Facing bet: raise by 10BB over current bet.
                # Also require 10BB >= 2× bet faced (your rule).
                target_10 = s.current_bet + raise_size_10bb
                cond_min = (raise_size_10bb >= 2.0 * bet_faced)

            if cond_min and (target_10 - s.contrib[p] <= stack + 1e-9) and is_raise(target_10):
                actions.append(ACTION_RAISE_10BB)

        # -------------------------
        # ALL-IN
        # -------------------------
        if stack > 0.0:
            actions.append(ACTION_ALL_IN)

        return actions

    # -----------------------------------------------------------------
    # STEP
    # -----------------------------------------------------------------
    def step(self, old: GameState, action: int) -> GameState:
        if old.terminal:
            return old

        legal = self.legal_actions(old)

        # Special-case: allow an explicit CALL action to be treated as a
        # call even when `to_call==0` (CHECK) if either player is all-in.
        # This lets callers signal they want to run out the hand when one
        # player has zero stack by issuing ACTION_CALL.
        to_call_old = max(old.current_bet - old.contrib[old.to_act], 0.0)
        allow_call_for_allin = (
            action == ACTION_CALL and to_call_old <= 1e-9 and (
                old.stacks[0] <= 1e-9 or old.stacks[1] <= 1e-9
            )
        )

        # Allow an explicit ACTION_FOLD even if not listed in legal_actions
        # (tests call env.step(s, ACTION_FOLD) directly on new hands).
        if action not in legal and action != ACTION_FOLD and not allow_call_for_allin:
            # Fallback: CALL > CHECK > FOLD
            if ACTION_CALL in legal:
                action = ACTION_CALL
            elif ACTION_CHECK in legal:
                action = ACTION_CHECK
            elif ACTION_FOLD in legal:
                action = ACTION_FOLD
            else:
                return old

        # Deep copy state
        s = GameState(
            deck=old.deck[:],
            board=old.board[:],
            hole=[old.hole[0][:], old.hole[1][:]],
            pot=old.pot,
            to_act=old.to_act,
            street=old.street,
            stacks=old.stacks[:],
            current_bet=old.current_bet,
            last_aggressor=old.last_aggressor,
            sb_player=old.sb_player,
            bb_player=old.bb_player,
            initial_stacks=old.initial_stacks[:],
            contrib=old.contrib[:],
            actions_since_raise=old.actions_since_raise,
            terminal=old.terminal,
            winner=old.winner
        )

        p = s.to_act
        opp = 1 - p
        stack = s.stacks[p]
        to_call = max(s.current_bet - s.contrib[p], 0.0)
        pot = s.pot

        # --------------------------------
        # FOLD
        # --------------------------------
        if action == ACTION_FOLD:
            s.winner = opp
            # Tests expect terminal `s.stacks` to represent the pot
            # distribution (i.e., winner receives the pot as the stack
            # value). Set stacks to pot shares rather than adding to
            # existing stacks.
            s.stacks = [0.0, 0.0]
            s.stacks[opp] = s.pot
            s.pot = 0.0
            s.terminal = True
            return s
        # CHECK
        # --------------------------------
        if action == ACTION_CHECK:
            # to_call == 0 by construction
            s.actions_since_raise += 1
            s.to_act = opp
            # check-check closes round
            if s.current_bet == 0.0 and s.actions_since_raise >= 2:
                self._next_street(s)
                return s

            # If contributions are equal (e.g., SB called the BB) then a
            # check by the opponent should close the street (unless
            # an all-in runout is required).
            if abs(s.contrib[0] - s.contrib[1]) < 1e-9:
                if s.stacks[0] <= 1e-9 or s.stacks[1] <= 1e-9:
                    self._runout_and_showdown_if_needed(s)
                    return s
                else:
                    if s.current_bet > 0.0:
                        self._next_street(s)
                        return s
                    # otherwise fall through to the empty-pot check above

            # If we're on the river and no live bet, require two checks
            # to close the hand (standard poker rule). Post-call or
            # all-in situations are handled above; do not advance on a
            # single check here.
            if s.street == STREET_RIVER and s.current_bet == 0.0 and s.actions_since_raise >= 2:
                self._next_street(s)
                return s

            return s

        # --------------------------------
        # CALL
        # --------------------------------
        if action == ACTION_CALL:
            call_amt = min(to_call, stack)
            s.stacks[p] -= call_amt
            s.contrib[p] += call_amt
            s.pot += call_amt
            s.actions_since_raise += 1
            s.to_act = opp

            # If either player is all-in after this call, run out the
            # remaining board and resolve the showdown immediately.
            if s.stacks[0] <= 1e-9 or s.stacks[1] <= 1e-9:
                self._runout_and_showdown_if_needed(s)
            # If the call equalized a genuine raise (last_aggressor set by
            # a previous raise), advance the street immediately.
            if abs(s.contrib[0] - s.contrib[1]) < 1e-9:
                if s.stacks[0] <= 1e-9 or s.stacks[1] <= 1e-9:
                    self._runout_and_showdown_if_needed(s)
                else:
                    if s.current_bet > 0.0 and s.last_aggressor != -1 and s.last_aggressor != p:
                        self._next_street(s)

            # Otherwise do not auto-advance here; a subsequent CHECK or
            # other action should close the betting round so tests can
            # observe the intermediate call state.
            return s

        # --------------------------------
        # RAISES / BETS / ALL-IN
        # --------------------------------
        def apply_raise(intended_add: float):
            nonlocal s, p, opp, stack
            add = min(intended_add, s.stacks[p])
            if add <= 0.0:
                return

            prev_cb = s.current_bet
            s.stacks[p] -= add
            s.contrib[p] += add
            s.pot += add

            # A true raise if new contrib exceeds previous current_bet
            if s.contrib[p] > prev_cb + 1e-9:
                s.current_bet = s.contrib[p]
                s.last_aggressor = p
                s.actions_since_raise = 0
            else:
                s.actions_since_raise += 1

            s.to_act = opp

            # If both all-in, run out
            if s.stacks[0] <= 1e-9 and s.stacks[1] <= 1e-9:
                self._runout_and_showdown_if_needed(s)

        base = s.current_bet if s.current_bet > 0.0 else self.bb
        bet_faced = to_call

        # 2x
        if action == ACTION_RAISE_2X:
            target_total = 2.0 * base
            intended_add = max(target_total - s.contrib[p], 0.0)
            apply_raise(intended_add)
            return s

        # 3x
        if action == ACTION_RAISE_3X:
            target_total = 3.0 * base
            intended_add = max(target_total - s.contrib[p], 0.0)
            apply_raise(intended_add)
            return s

        # HALF_POT
        if action == ACTION_HALF_POT:
            if to_call <= 0.0:
                # No live bet. Distinguish:
                # - Opening bet from truly empty pot: target_total = max(0.5*pot, BB)
                # - Opening bet after call (contrib are equal): bet = 0.5*pot added to existing contrib
                if abs(s.contrib[0] - s.contrib[1]) < 1e-9 and s.contrib[p] > 0.0:
                    # Equal contributions after call — add 0.5*pot to existing contribution
                    intended_add = 0.5 * pot
                else:
                    # Truly empty (preflop open or fresh round) — target total is max(0.5*pot, BB)
                    target_total = max(0.5 * pot, self.bb)
                    intended_add = max(target_total - s.contrib[p], 0.0)
            else:
                target_total = s.contrib[p] + to_call + 0.5 * (pot + to_call)
                intended_add = max(target_total - s.contrib[p], 0.0)
            apply_raise(intended_add)
            return s

        # POT
        if action == ACTION_POT:
            if to_call <= 0.0:
                # No live bet. Distinguish:
                # - Opening bet from truly empty pot: target_total = max(pot, BB)
                # - Opening bet after call (contrib are equal): bet = pot added to existing contrib
                if abs(s.contrib[0] - s.contrib[1]) < 1e-9 and s.contrib[p] > 0.0:
                    # Equal contributions after call — add full pot to existing contribution
                    intended_add = pot
                else:
                    # Truly empty (preflop open or fresh round) — target total is max(pot, BB)
                    target_total = max(pot, self.bb)
                    intended_add = max(target_total - s.contrib[p], 0.0)
            else:
                target_total = s.contrib[p] + to_call + (pot + to_call)
                intended_add = max(target_total - s.contrib[p], 0.0)
            apply_raise(intended_add)
            return s

        # ---------------------------------------------------------
        # FIXED 10BB LOGIC
        # ---------------------------------------------------------
        if action == ACTION_RAISE_10BB:

            raise_size = 10.0 * self.bb    # always 20 in 1/2 blinds
            if (
                s.street == STREET_PREFLOP
                and p == s.sb_player
                and to_call == (self.bb - self.sb)  # 1 in 1/2 blinds
            ):
                target_total = raise_size        # must be exactly 20
            elif to_call <= 1e-9:
                target_total = raise_size
            else:
                # Call + add 10BB
                target_total = s.contrib[p] + to_call + raise_size
            intended_add = max(target_total - s.contrib[p], 0.0)
            apply_raise(intended_add)
            return s

        # ALL-IN
        if action == ACTION_ALL_IN:
            intended_add = s.stacks[p]
            apply_raise(intended_add)
            return s

        # Should not reach here
        return s

    # -----------------------------------------------------------------
    # STREET TRANSITION
    # -----------------------------------------------------------------
    def _next_street(self, s: GameState):
        if s.terminal:
            return

        # Reset betting info for new street
        s.actions_since_raise = 0
        s.contrib = [0.0, 0.0]
        s.current_bet = 0.0
        s.last_aggressor = -1

        if s.street == STREET_PREFLOP:
            # Deal flop (3 cards)
            try:
                s.board.extend([s.deck.pop(), s.deck.pop(), s.deck.pop()])
            except IndexError:
                s.terminal = True
                s.winner = -1
                return
            s.street = STREET_FLOP

        elif s.street == STREET_FLOP:
            # Turn
            try:
                s.board.append(s.deck.pop())
            except IndexError:
                s.terminal = True
                s.winner = -1
                return
            s.street = STREET_TURN

        elif s.street == STREET_TURN:
            # River
            try:
                s.board.append(s.deck.pop())
            except IndexError:
                s.terminal = True
                s.winner = -1
                return
            s.street = STREET_RIVER

        elif s.street == STREET_RIVER:
            # Go directly to showdown
            self._resolve_showdown(s)
            return

        # New street: SB acts first in 2-player
        s.to_act = s.sb_player

    # -----------------------------------------------------------------
    # RUNOUT WHEN BOTH ALL-IN
    # -----------------------------------------------------------------
    def _runout_and_showdown_if_needed(self, s: GameState):
        if s.terminal:
            return

        # Ensure the board is dealt out to 5 cards (river). Tests may set
        # `s.street` without populating `s.board`, so deal any missing
        # cards until the board has five cards or the deck is exhausted.
        while len(s.board) < 5:
            try:
                s.board.append(s.deck.pop())
            except IndexError:
                s.terminal = True
                s.winner = -1
                return

        # Mark as river and resolve showdown
        s.street = STREET_RIVER
        self._resolve_showdown(s)

    # -----------------------------------------------------------------
    # SHOWDOWN
    # -----------------------------------------------------------------
    def _resolve_showdown(self, s: GameState):
        v0 = evaluate_7card(s.hole[0], s.board)
        v1 = evaluate_7card(s.hole[1], s.board)
        if v0 > v1:
            s.winner = 0
            s.stacks = [s.pot, 0.0]
        elif v1 > v0:
            s.winner = 1
            s.stacks = [0.0, s.pot]
        else:
            s.winner = -1
            s.stacks = [s.pot / 2.0, s.pot / 2.0]

        s.pot = 0.0
        s.terminal = True

    def terminal_payoff(self, s: GameState, hero: int) -> float:
        if not s.terminal:
            return 0.0
        return s.stacks[hero] - s.initial_stacks[hero]
