import random
from dataclasses import dataclass
from typing import List

from config import STACK_SIZE, SMALL_BLIND, BIG_BLIND, RNG_SEED
from abstraction import evaluate_7card

RNG = random.Random(RNG_SEED)

ACTION_FOLD     = 0
ACTION_CALL     = 1
ACTION_MINRAISE = 2
ACTION_HALF_POT = 3
ACTION_POT      = 4
ACTION_ALL_IN   = 5

NUM_ACTIONS = 6

STREET_PREFLOP = 0
STREET_FLOP    = 1
STREET_TURN    = 2
STREET_RIVER   = 3

@dataclass
class GameState:
    deck: List[int]
    board: List[int]
    hole: List[List[int]]
    pot: float
    to_act: int
    street: int
    stacks: List[float]
    current_bet: float
    last_aggressor: int
    sb_player: int
    bb_player: int
    initial_stacks: List[float]
    contrib: List[float]
    actions_since_raise: int = 0
    last_raise_amt: float = 0.0
    terminal: bool = False
    winner: int = -1

class SimpleHoldemEnv:
    def __init__(self, stack_size=STACK_SIZE, sb=SMALL_BLIND, bb=BIG_BLIND):
        self.stack_size = stack_size
        self.sb = sb
        self.bb = bb

    def new_hand(self) -> GameState:
        deck = list(range(52))
        RNG.shuffle(deck)
        h0 = [deck.pop(), deck.pop()]
        h1 = [deck.pop(), deck.pop()]
        if RNG.random() < 0.5:
            sb, bb = 0, 1
        else:
            sb, bb = 1, 0
        initial = [self.stack_size, self.stack_size]
        stacks = initial[:]
        contrib = [0.0, 0.0]
        stacks[sb] -= self.sb
        stacks[bb] -= self.bb
        contrib[sb] = self.sb
        contrib[bb] = self.bb
        return GameState(
            deck=deck,
            board=[],
            hole=[h0, h1],
            pot=self.sb + self.bb,
            to_act=sb,
            street=STREET_PREFLOP,
            stacks=stacks,
            current_bet=self.bb,
            last_aggressor=bb,
            sb_player=sb,
            bb_player=bb,
            initial_stacks=initial,
            contrib=contrib,
            actions_since_raise=0,
            last_raise_amt=self.bb
        )

    def legal_actions(self, s: GameState) -> List[int]:
        if s.terminal or s.stacks[s.to_act] <= 0:
            return []
        actions = []
        p = s.to_act
        opp = 1 - p
        to_call = s.current_bet - s.contrib[p]
        stack = s.stacks[p]
        can_raise = (s.stacks[opp] > 0 and stack > to_call)
        min_raise = self._min_raise_amt(s)
        # Fold/Call/Check
        if to_call > 0:
            actions.append(ACTION_FOLD)
        actions.append(ACTION_CALL)
        # Min-raise
        minraise_total = s.current_bet + min_raise
        if can_raise and stack + s.contrib[p] >= minraise_total and min_raise > 0:
            actions.append(ACTION_MINRAISE)
        # Half-pot
        halfpot_total = s.current_bet + (s.pot + to_call) * 0.5
        min_legal_total = s.current_bet + min_raise
        halfpot_total = max(halfpot_total, min_legal_total)
        if (can_raise and halfpot_total > s.current_bet and
                stack + s.contrib[p] >= halfpot_total and
                (halfpot_total - s.current_bet) + 1e-6 >= min_raise):
            actions.append(ACTION_HALF_POT)
        # Pot
        pot_total = s.current_bet + (s.pot + to_call)
        pot_total = max(pot_total, min_legal_total)
        if (can_raise and pot_total > s.current_bet and
                stack + s.contrib[p] >= pot_total and
                (pot_total - s.current_bet) + 1e-6 >= min_raise):
            actions.append(ACTION_POT)
        if stack > 0:
            actions.append(ACTION_ALL_IN)
        return actions

    def _min_raise_amt(self, s: GameState):
        if s.street == STREET_PREFLOP and s.current_bet == self.bb:
            return self.bb
        return max(s.last_raise_amt, self.bb)

    def _end_betting_round(self, s: GameState) -> bool:
        bets_equal = abs(s.contrib[0] - s.contrib[1]) < 1e-9
        both_acted = s.actions_since_raise >= 2
        if s.current_bet == 0:
            return both_acted
        return bets_equal and both_acted

    def _check_all_in_and_run_out(self, s: GameState):
        if (s.stacks[0] <= 1e-9 and s.stacks[1] <= 1e-9 and not s.terminal):
            while s.street < STREET_RIVER:
                self._next_street(s)
            if not s.terminal:
                self.resolve_showdown(s)

    def step(self, old: GameState, action: int) -> GameState:
        if old.terminal:
            return old
        legal_actions = self.legal_actions(old)
        if action not in legal_actions:
            if ACTION_CALL in legal_actions:
                action = ACTION_CALL
            else:
                action = ACTION_FOLD
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
            last_raise_amt=old.last_raise_amt,
            terminal=old.terminal,
            winner=old.winner
        )
        p = s.to_act
        opp = 1 - p
        stack = s.stacks[p]
        to_call = s.current_bet - s.contrib[p]
        if to_call < 0: to_call = 0

        if action == ACTION_FOLD:
            s.winner = opp
            s.stacks[opp] += s.pot
            s.pot = 0
            s.terminal = True
            return s

        if action == ACTION_CALL:
            if (s.street == STREET_PREFLOP and p == s.sb_player and 
                s.current_bet == self.bb and s.contrib[p] < self.bb):
                amount_needed = self.bb - s.contrib[p]
                call_amt = min(amount_needed, stack)
            else:
                call_amt = min(to_call, stack)
            s.stacks[p] -= call_amt
            s.contrib[p] += call_amt
            s.pot += call_amt
            s.actions_since_raise += 1
            s.to_act = opp
            self._check_all_in_and_run_out(s)
            if self._end_betting_round(s):
                self._next_street(s)
            return s

        if action == ACTION_MINRAISE:
            min_raise = self._min_raise_amt(s)
            target = s.current_bet + min_raise
            raise_amt = target - s.contrib[p]
            amt = min(raise_amt, stack)
            s.stacks[p] -= amt
            s.contrib[p] += amt
            s.pot += amt
            s.last_raise_amt = target - s.current_bet
            s.current_bet = target
            s.last_aggressor = p
            s.actions_since_raise = 1
            s.to_act = opp
            self._check_all_in_and_run_out(s)
            if s.stacks[opp] <= 0:
                if self._end_betting_round(s):
                    self._next_street(s)
            elif self._end_betting_round(s):
                self._next_street(s)
            return s

        if action == ACTION_HALF_POT:
            min_raise = self._min_raise_amt(s)
            halfpot_total = s.current_bet + (s.pot + to_call) * 0.5
            min_legal_total = s.current_bet + min_raise
            halfpot_total = max(halfpot_total, min_legal_total)
            raise_amt = halfpot_total - s.contrib[p]
            amt = min(raise_amt, stack)
            s.stacks[p] -= amt
            s.contrib[p] += amt
            s.pot += amt
            s.last_raise_amt = halfpot_total - s.current_bet
            s.current_bet = halfpot_total
            s.last_aggressor = p
            s.actions_since_raise = 1
            s.to_act = opp
            self._check_all_in_and_run_out(s)
            if s.stacks[opp] <= 0:
                if self._end_betting_round(s):
                    self._next_street(s)
            elif self._end_betting_round(s):
                self._next_street(s)
            return s

        if action == ACTION_POT:
            min_raise = self._min_raise_amt(s)
            pot_total = s.current_bet + (s.pot + to_call)
            min_legal_total = s.current_bet + min_raise
            pot_total = max(pot_total, min_legal_total)
            raise_amt = pot_total - s.contrib[p]
            amt = min(raise_amt, stack)
            s.stacks[p] -= amt
            s.contrib[p] += amt
            s.pot += amt
            s.last_raise_amt = pot_total - s.current_bet
            s.current_bet = pot_total
            s.last_aggressor = p
            s.actions_since_raise = 1
            s.to_act = opp
            self._check_all_in_and_run_out(s)
            if s.stacks[opp] <= 0:
                if self._end_betting_round(s):
                    self._next_street(s)
            elif self._end_betting_round(s):
                self._next_street(s)
            return s

        if action == ACTION_ALL_IN:
            new_total = s.contrib[p] + stack
            min_raise = self._min_raise_amt(s)
            if new_total >= s.current_bet + min_raise:
                raise_amt = new_total - s.current_bet
                amt = stack
                s.stacks[p] -= amt
                s.contrib[p] += amt
                s.pot += amt
                s.last_raise_amt = raise_amt
                s.current_bet = new_total
                s.last_aggressor = p
                s.actions_since_raise = 1
            else:
                amt = stack
                s.stacks[p] -= amt
                s.contrib[p] += amt
                s.pot += amt
                s.actions_since_raise += 1
            s.to_act = opp
            self._check_all_in_and_run_out(s)
            if self._end_betting_round(s):
                self._next_street(s)
            return s

        return s

    def _next_street(self, s: GameState):
        s.actions_since_raise = 0
        s.contrib = [0.0, 0.0]
        s.current_bet = 0.0
        s.last_raise_amt = 0.0
        s.last_aggressor = -1
        try:
            if s.street == STREET_PREFLOP:
                s.board.extend([s.deck.pop(), s.deck.pop(), s.deck.pop()])
                s.street = STREET_FLOP
            elif s.street == STREET_FLOP:
                s.board.append(s.deck.pop())
                s.street = STREET_TURN
            elif s.street == STREET_TURN:
                s.board.append(s.deck.pop())
                s.street = STREET_RIVER
            elif s.street == STREET_RIVER:
                self.resolve_showdown(s)
                return
        except IndexError:
            s.terminal = True
            s.winner = -1
            return
        s.to_act = s.sb_player

    def resolve_showdown(self, s: GameState):
        v0 = evaluate_7card(s.hole[0], s.board)
        v1 = evaluate_7card(s.hole[1], s.board)
        if v0 > v1:
            s.winner = 0
            s.stacks[0] += s.pot
        elif v1 > v0:
            s.winner = 1
            s.stacks[1] += s.pot
        else:
            s.winner = -1
            s.stacks[0] += s.pot / 2
            s.stacks[1] += s.pot / 2
        s.pot = 0
        s.terminal = True

    def terminal_payoff(self, s: GameState, hero: int) -> float:
        if not s.terminal:
            return 0.0
        return s.stacks[hero] - s.initial_stacks[hero]
