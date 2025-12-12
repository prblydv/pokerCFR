import random

from dataclasses import dataclass

from typing import List



from config import STACK_SIZE, SMALL_BLIND, BIG_BLIND, RNG_SEED

from abstraction import evaluate_7card



RNG = random.Random(RNG_SEED)



# ---------------------------------------------------------

# Actions

# ---------------------------------------------------------

ACTION_FOLD = 0
ACTION_CHECK = 1
ACTION_CALL = 2
ACTION_RAISE_SMALL = 3   # PF: 2Ã— BB, Postflop: 0.5 pot
ACTION_RAISE_MEDIUM = 4  # PF: 4Ã— BB, Postflop: 1.0 pot
ACTION_ALL_IN = 5
NUM_ACTIONS = 6
RAISE_ACTIONS = [ACTION_RAISE_SMALL, ACTION_RAISE_MEDIUM]
PREFLOP_RAISE_MULT = {
    ACTION_RAISE_SMALL: 2.0,
    ACTION_RAISE_MEDIUM: 4.0,
}


POSTFLOP_RAISE_FRAC = {
    ACTION_RAISE_SMALL: 0.5,
    ACTION_RAISE_MEDIUM: 1.0,
}



# ---------------------------------------------------------

# Streets``

# ---------------------------------------------------------

STREET_PREFLOP = 0
STREET_FLOP = 1
STREET_TURN = 2
STREET_RIVER = 3





# ---------------------------------------------------------

# GameState (same API)

# ---------------------------------------------------------

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
    actions_this_street: int = 0
    terminal: bool = False
    winner: int = -1

# ---------------------------------------------------------

# ---------------------------------------------------------
# SimpleHoldemEnv (FAST + CORRECT)
# ---------------------------------------------------------
class SimpleHoldemEnv:

    def __init__(self,
                 stack_size: float = STACK_SIZE,
                 sb: float = SMALL_BLIND,
                 bb: float = BIG_BLIND):
        self.stack_size = stack_size
        self.sb = sb
        self.bb = bb
        self._next_sb = RNG.randint(0, 1)



    def new_hand(self) -> GameState:

        deck = list(range(52))

        RNG.shuffle(deck)



        h0 = [deck.pop(), deck.pop()]
        h1 = [deck.pop(), deck.pop()]

        sb = self._next_sb
        bb = 1 - sb
        self._next_sb = bb



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

            actions_this_street=0

        )



    def legal_actions(self, s: GameState) -> List[int]:

        if s.terminal:

            return []



        p = s.to_act

        opp = 1 - p

        if s.stacks[p] <= 0:

            return []



        to_call = self._amount_to_call(s, p)

        actions: List[int] = []



        if to_call > 0:

            actions.append(ACTION_FOLD)

        else:

            actions.append(ACTION_CHECK)



        if to_call > 0 and s.stacks[p] > 0:

            actions.append(ACTION_CALL)



        if s.stacks[opp] > 0:

            for a in RAISE_ACTIONS:

                if self._can_raise(s, a):

                    actions.append(a)

            if s.stacks[p] > 0:

                actions.append(ACTION_ALL_IN)



        return actions



    def deal_next_street(self, s: GameState) -> None:

        if s.street == STREET_PREFLOP:

            s.board.extend([s.deck.pop(), s.deck.pop(), s.deck.pop()])

            s.street = STREET_FLOP

        elif s.street == STREET_FLOP:

            s.board.append(s.deck.pop())

            s.street = STREET_TURN

        elif s.street == STREET_TURN:

            s.board.append(s.deck.pop())

            s.street = STREET_RIVER

        else:

            self.resolve_showdown(s)

            return



        s.current_bet = 0.0

        s.last_aggressor = -1

        s.contrib = [0.0, 0.0]

        s.actions_this_street = 0

        s.to_act = s.bb_player



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



    def step(self, old: GameState, action: int) -> GameState:

        if old.terminal:

            return old



        s = GameState(

            deck=old.deck[:],

            board=old.board[:],

            hole=old.hole,

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

            actions_this_street=old.actions_this_street,

            terminal=False,

            winner=-1

        )



        p = s.to_act

        opp = 1 - p

        s.actions_this_street += 1

        to_call = self._amount_to_call(s, p)



        if action == ACTION_FOLD:

            s.winner = opp

            s.stacks[opp] += s.pot

            s.pot = 0

            s.terminal = True

            return s



        if action == ACTION_CHECK:

            if to_call > 0:

                return s

            s.to_act = opp

            self._maybe_advance_round(s)

            return s



        if action == ACTION_CALL:

            if to_call <= 0:

                return s

            self._invest(s, p, min(to_call, s.stacks[p]))

            if s.current_bet > 0 and self._amount_to_call(s, opp) <= 1e-9:

                s.current_bet = 0.0

                s.last_aggressor = -1

            s.to_act = opp

            if s.stacks[p] <= 0 and s.stacks[opp] <= 0:

                return self.run_out_board_all_in(s)

            self._maybe_advance_round(s)

            return s



        if action in RAISE_ACTIONS:

            amount = self._raise_amount(s, action)

            if amount <= to_call:

                return s

            amount = min(amount, s.stacks[p])

            if amount <= to_call:

                return s

            self._invest(s, p, amount)

            s.current_bet = max(s.current_bet, s.contrib[p])

            s.last_aggressor = p

            s.to_act = opp

            if s.stacks[p] <= 0 and s.stacks[opp] <= 0:

                return self.run_out_board_all_in(s)

            return s



        if action == ACTION_ALL_IN:

            amount = s.stacks[p]

            if amount <= 0:

                return s

            self._invest(s, p, amount)

            if s.contrib[p] > s.current_bet:

                s.current_bet = s.contrib[p]

                s.last_aggressor = p

            s.to_act = opp

            if s.stacks[p] <= 0 and s.stacks[opp] <= 0:

                return self.run_out_board_all_in(s)

            return s



        return s



    def run_out_board_all_in(self, s: GameState) -> GameState:

        while not s.terminal and s.street < STREET_RIVER:

            self.deal_next_street(s)

        if not s.terminal:

            self.resolve_showdown(s)

        return s



    def terminal_payoff(self, s: GameState, hero: int) -> float:

        if not s.terminal:

            return 0.0

        return s.stacks[hero] - s.initial_stacks[hero]



    def _amount_to_call(self, s: GameState, p: int) -> float:

        return max(0.0, s.current_bet - s.contrib[p])



    def _maybe_advance_round(self, s: GameState) -> None:

        if s.current_bet == 0.0 and s.actions_this_street >= 2:

            s.actions_this_street = 0

            s.contrib = [0.0, 0.0]

            if s.street == STREET_RIVER:

                self.resolve_showdown(s)

            else:

                self.deal_next_street(s)



    def _can_raise(self, s: GameState, action: int) -> bool:

        if action not in RAISE_ACTIONS:

            return False

        p = s.to_act

        opp = 1 - p

        if s.stacks[p] <= 0 or s.stacks[opp] <= 0:

            return False

        amount = self._raise_amount(s, action)

        to_call = self._amount_to_call(s, p)

        if amount <= to_call or amount > s.stacks[p]:

            return False

        return True



    def _raise_amount(self, s: GameState, action: int) -> float:

        p = s.to_act

        to_call = self._amount_to_call(s, p)

        if s.street == STREET_PREFLOP:

            base_size = PREFLOP_RAISE_MULT[action] * self.bb

        else:

            pot_base = max(self.bb, s.pot)

            base_size = POSTFLOP_RAISE_FRAC[action] * pot_base

        if base_size <= 0.0:

            return 0.0

        if to_call <= 0.0:

            return base_size

        return to_call + base_size



    def _invest(self, s: GameState, p: int, amount: float) -> float:

        amount = max(0.0, min(amount, s.stacks[p]))

        s.stacks[p] -= amount

        s.contrib[p] += amount

        s.pot += amount

        return amount



