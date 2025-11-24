# # poker_env.py — no-limit holdem env with 10 size-based actions on all streets

# import random
# from dataclasses import dataclass
# from typing import List

# from config import STACK_SIZE, SMALL_BLIND, BIG_BLIND, RNG_SEED
# from abstraction import evaluate_7card

# RNG = random.Random(RNG_SEED)

# # ---------------------------------------------------------
# # Actions
# # ---------------------------------------------------------
# ACTION_FOLD = 0
# ACTION_CALL = 1          # also check
# ACTION_RAISE_2X = 2
# ACTION_RAISE_2_25X = 3
# ACTION_RAISE_2_5X = 4
# ACTION_RAISE_3X = 5
# ACTION_RAISE_3_5X = 6
# ACTION_RAISE_4_5X = 7
# ACTION_RAISE_6X = 8
# ACTION_ALL_IN = 9

# NUM_ACTIONS = 10

# # Raise actions (all except fold/call/all-in)
# RAISE_ACTIONS = [
#     ACTION_RAISE_2X,
#     ACTION_RAISE_2_25X,
#     ACTION_RAISE_2_5X,
#     ACTION_RAISE_3X,
#     ACTION_RAISE_3_5X,
#     ACTION_RAISE_4_5X,
#     ACTION_RAISE_6X,
# ]

# # Multiplier for each raise action (× pot, capped by stack)
# RAISE_MULT = {
#     ACTION_RAISE_2X: 2.0,
#     ACTION_RAISE_2_25X: 2.25,
#     ACTION_RAISE_2_5X: 2.5,
#     ACTION_RAISE_3X: 3.0,
#     ACTION_RAISE_3_5X: 3.5,
#     ACTION_RAISE_4_5X: 4.5,
#     ACTION_RAISE_6X: 6.0,
# }

# # ---------------------------------------------------------
# # Streets
# # ---------------------------------------------------------
# STREET_PREFLOP = 0
# STREET_FLOP = 1
# STREET_TURN = 2
# STREET_RIVER = 3


# @dataclass
# class GameState:
#     deck: List[int]
#     board: List[int]
#     hole: List[List[int]]
#     pot: float
#     to_act: int
#     street: int
#     stacks: List[float]
#     current_bet: float            # amount last aggressor put in this street
#     last_aggressor: int
#     initial_stacks: List[float] = None
#     terminal: bool = False
#     winner: int = -1


# class SimpleHoldemEnv:
#     def __init__(self,
#                  stack_size: float = STACK_SIZE,
#                  sb: float = SMALL_BLIND,
#                  bb: float = BIG_BLIND):
#         self.stack_size = stack_size
#         self.sb = sb
#         self.bb = bb

#     # --------------------------------------------------
#     # Hand start helpers
#     # --------------------------------------------------
#     def _deal_new_hand(self, stacks: List[float], use_existing: bool) -> GameState:
#         deck = list(range(52))
#         RNG.shuffle(deck)

#         hole0 = [deck.pop(), deck.pop()]
#         hole1 = [deck.pop(), deck.pop()]

#         if RNG.random() < 0.5:
#             sb_player, bb_player = 0, 1
#         else:
#             sb_player, bb_player = 1, 0

#         if use_existing:
#             initial_stacks = list(stacks)
#             new_stacks = list(stacks)
#         else:
#             initial_stacks = [self.stack_size, self.stack_size]
#             new_stacks = [self.stack_size, self.stack_size]

#         # post blinds
#         new_stacks[sb_player] -= self.sb
#         new_stacks[bb_player] -= self.bb
#         pot = self.sb + self.bb

#         return GameState(
#             deck=deck,
#             board=[],
#             hole=[hole0, hole1],
#             pot=pot,
#             to_act=sb_player,
#             street=STREET_PREFLOP,
#             stacks=new_stacks,
#             current_bet=self.bb,      # amount facing SB
#             last_aggressor=bb_player,
#             initial_stacks=initial_stacks,
#         )

#     def new_hand(self) -> GameState:
#         return self._deal_new_hand(stacks=None, use_existing=False)

#     def new_hand_with_stacks(self, stacks: List[float]) -> GameState:
#         return self._deal_new_hand(stacks=stacks, use_existing=True)

#     # --------------------------------------------------
#     # Auto runout when both all-in
#     # --------------------------------------------------
#     def run_out_board_all_in(self, s: GameState) -> GameState:
#         while not s.terminal and s.street < STREET_RIVER:
#             self.deal_next_street(s)
#         if not s.terminal:
#             self.resolve_showdown(s)
#         return s

#     # --------------------------------------------------
#     # Legal actions
#     # --------------------------------------------------
#     def legal_actions(self, s: GameState) -> List[int]:
#         if s.terminal:
#             return []

#         p = s.to_act
#         opp = 1 - p

#         if s.stacks[p] <= 0:
#             return []  # player already all-in

#         actions = [ACTION_FOLD, ACTION_CALL]

#         # if opponent is already all-in, no more raises
#         if s.stacks[opp] <= 0:
#             return actions

#         # allow all raise sizes that have positive intended size
#         base = max(s.pot, self.bb)  # pot-based sizing; at least 1 BB
#         for a in RAISE_ACTIONS:
#             mult = RAISE_MULT[a]
#             intended = mult * base
#             if intended > 0.0:
#                 actions.append(a)

#         # all-in is always legal if hero has chips and opp has chips
#         actions.append(ACTION_ALL_IN)
#         return actions

#     # --------------------------------------------------
#     # Street helpers
#     # --------------------------------------------------
#     @staticmethod
#     def is_betting_round_over(s: GameState) -> bool:
#         return s.current_bet == 0.0 and s.last_aggressor == -1

#     def deal_next_street(self, s: GameState) -> None:
#         if s.street == STREET_PREFLOP:
#             s.board.extend([s.deck.pop(), s.deck.pop(), s.deck.pop()])
#             s.street = STREET_FLOP
#         elif s.street == STREET_FLOP:
#             s.board.append(s.deck.pop())
#             s.street = STREET_TURN
#         elif s.street == STREET_TURN:
#             s.board.append(s.deck.pop())
#             s.street = STREET_RIVER
#         else:  # river → showdown
#             self.resolve_showdown(s)
#             return

#         s.current_bet = 0.0
#         s.last_aggressor = -1

#     def resolve_showdown(self, s: GameState) -> None:
#         v0 = evaluate_7card(s.hole[0], s.board)
#         v1 = evaluate_7card(s.hole[1], s.board)
#         if v0 > v1:
#             s.winner = 0
#         elif v1 > v0:
#             s.winner = 1
#         else:
#             s.winner = -1
#         s.terminal = True

#     # --------------------------------------------------
#     # Step
#     # --------------------------------------------------
#     def step(self, old: GameState, action: int) -> GameState:
#         if old.terminal:
#             return old

#         s = GameState(
#             deck=list(old.deck),
#             board=list(old.board),
#             hole=[list(old.hole[0]), list(old.hole[1])],
#             pot=old.pot,
#             to_act=old.to_act,
#             street=old.street,
#             stacks=list(old.stacks),
#             current_bet=old.current_bet,
#             last_aggressor=old.last_aggressor,
#             initial_stacks=list(old.initial_stacks),
#             terminal=old.terminal,
#             winner=old.winner,
#         )

#         p = s.to_act
#         opp = 1 - p

#         # ---------------- FOLD ----------------
#         if action == ACTION_FOLD:
#             s.terminal = True
#             s.winner = opp
#             s.stacks[opp] += s.pot       # give pot to winner
#             s.pot = 0.0
#             return s

#         # ---------------- CALL / CHECK ----------------
#         if action == ACTION_CALL:
#             call_amt = min(s.current_bet, s.stacks[p])
            
#             s.stacks[p] -= call_amt
#             s.pot += call_amt
#             s.current_bet = 0.0
#             s.last_aggressor = -1

#             # if both all-in after call → run out board
#             if s.stacks[p] == 0 and s.stacks[opp] == 0:
#                 return self.run_out_board_all_in(s)

#             s.to_act = opp
#             if self.is_betting_round_over(s):
#                 if s.street == STREET_RIVER:
#                     self.resolve_showdown(s)
#                 else:
#                     self.deal_next_street(s)
#             return s

#         # ---------------- RAISES / ALL-IN ----------------
#         if action == ACTION_ALL_IN:
#             amount = s.stacks[p]
#         elif action in RAISE_ACTIONS:
#             base = max(s.pot, self.bb)
#             mult = RAISE_MULT[action]
#             intended = mult * base
#             # cap by stack → if 3x etc exceeds stack, becomes all-in
#             amount = min(intended, s.stacks[p])
#         else:
#             # invalid action index -> no-op (should not happen if masks are correct)
#             return s

#         if amount <= 0.0:
#             return s  # nothing to do

#         s.stacks[p] -= amount
#         s.pot += amount
#         s.current_bet = amount
#         s.last_aggressor = p
#         s.to_act = opp

#         # if both players are now all-in -> run out board
#         if s.stacks[p] == 0 and s.stacks[opp] == 0:
#             return self.run_out_board_all_in(s)

#         return s

#     # --------------------------------------------------
#     # Payoff for CFR
#     # --------------------------------------------------
#     def terminal_payoff(self, s: GameState, hero: int) -> float:
#         if not s.terminal or s.winner == -1:
#             return 0.0
#         return s.pot if s.winner == hero else -s.pot
# poker_env.py — clean HU no-limit holdem with 10 discrete actions

import random
from dataclasses import dataclass
from typing import List, Optional

from config import STACK_SIZE, SMALL_BLIND, BIG_BLIND, RNG_SEED
from abstraction import evaluate_7card

RNG = random.Random(RNG_SEED)

# ---------------------------------------------------------
# Actions
# ---------------------------------------------------------
ACTION_FOLD = 0
ACTION_CALL = 1          # also check when to_call=0
ACTION_RAISE_2X = 2
ACTION_RAISE_2_25X = 3
ACTION_RAISE_2_5X = 4
ACTION_RAISE_3X = 5
ACTION_RAISE_3_5X = 6
ACTION_RAISE_4_5X = 7
ACTION_RAISE_6X = 8
ACTION_ALL_IN = 9

NUM_ACTIONS = 10

RAISE_ACTIONS = [
    ACTION_RAISE_2X,
    ACTION_RAISE_2_25X,
    ACTION_RAISE_2_5X,
    ACTION_RAISE_3X,
    ACTION_RAISE_3_5X,
    ACTION_RAISE_4_5X,
    ACTION_RAISE_6X,
]

RAISE_MULT = {
    ACTION_RAISE_2X: 2.0,
    ACTION_RAISE_2_25X: 2.25,
    ACTION_RAISE_2_5X: 2.5,
    ACTION_RAISE_3X: 3.0,
    ACTION_RAISE_3_5X: 3.5,
    ACTION_RAISE_4_5X: 4.5,
    ACTION_RAISE_6X: 6.0,
}

# ---------------------------------------------------------
# Streets
# ---------------------------------------------------------
STREET_PREFLOP = 0
STREET_FLOP = 1
STREET_TURN = 2
STREET_RIVER = 3


@dataclass
class GameState:
    deck: List[int]
    board: List[int]
    hole: List[List[int]]
    pot: float
    to_act: int
    street: int
    stacks: List[float]

    # total bet to match on this street (0 means no bet outstanding)
    current_bet: float
    last_aggressor: int

    # who posted blinds this hand
    sb_player: int
    bb_player: int

    initial_stacks: List[float]
    terminal: bool = False
    winner: int = -1  # 0/1 or -1 for tie


class SimpleHoldemEnv:
    def __init__(self,
                 stack_size: float = STACK_SIZE,
                 sb: float = SMALL_BLIND,
                 bb: float = BIG_BLIND):
        self.stack_size = stack_size
        self.sb = sb
        self.bb = bb

    # --------------------------------------------------
    # Hand start
    # --------------------------------------------------
    def new_hand(self) -> GameState:
        deck = list(range(52))
        RNG.shuffle(deck)

        hole0 = [deck.pop(), deck.pop()]
        hole1 = [deck.pop(), deck.pop()]

        # random dealer each hand
        if RNG.random() < 0.5:
            sb_player, bb_player = 0, 1
        else:
            sb_player, bb_player = 1, 0

        stacks = [self.stack_size, self.stack_size]
        initial_stacks = stacks[:]

        # post blinds (real poker)
        stacks[sb_player] -= self.sb
        stacks[bb_player] -= self.bb
        pot = self.sb + self.bb

        return GameState(
            deck=deck,
            board=[],
            hole=[hole0, hole1],
            pot=pot,
            to_act=sb_player,                 # SB/BTN acts first preflop
            street=STREET_PREFLOP,
            stacks=stacks,
            current_bet=self.bb,             # bet to match is BB
            last_aggressor=bb_player,
            sb_player=sb_player,
            bb_player=bb_player,
            initial_stacks=initial_stacks,
        )

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------
    def is_betting_round_over(self, s: GameState) -> bool:
        # round over when no one is facing a bet and last aggressor cleared
        return s.current_bet == 0.0 and s.last_aggressor == -1

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
        s.to_act = s.bb_player            # BB acts first postflop HU

    def resolve_showdown(self, s: GameState) -> None:
        v0 = evaluate_7card(s.hole[0], s.board)
        v1 = evaluate_7card(s.hole[1], s.board)

        if v0 > v1:
            s.winner = 0
        elif v1 > v0:
            s.winner = 1
        else:
            s.winner = -1

        # award pot
        if s.winner != -1:
            s.stacks[s.winner] += s.pot
        else:
            # split on tie
            s.stacks[0] += s.pot / 2.0
            s.stacks[1] += s.pot / 2.0

        s.pot = 0.0
        s.terminal = True

    def run_out_board_all_in(self, s: GameState) -> GameState:
        while not s.terminal and s.street < STREET_RIVER:
            self.deal_next_street(s)
        if not s.terminal:
            self.resolve_showdown(s)
        return s

    # --------------------------------------------------
    # Legal actions
    # --------------------------------------------------
    def legal_actions(self, s: GameState) -> List[int]:
        if s.terminal:
            return []

        p = s.to_act
        opp = 1 - p

        if s.stacks[p] <= 0:
            return []

        actions = []

        # ---------- FIX ----------
        # FOLD is ONLY legal if hero is facing a bet
        if s.current_bet > 0:
            actions.append(ACTION_FOLD)

        # CALL becomes CHECK automatically when current_bet == 0
        actions.append(ACTION_CALL)
        # -------------------------

        # If opponent is all-in, raises are illegal
        if s.stacks[opp] <= 0:
            return actions

        # Raise sizing rules
        if s.street == STREET_PREFLOP:
            base_total = self.bb
        else:
            base_total = max(s.pot, self.bb)

        for a in RAISE_ACTIONS:
            target_total = RAISE_MULT[a] * base_total
            if target_total > s.current_bet and target_total > 0:
                actions.append(a)

        actions.append(ACTION_ALL_IN)
        return actions


    # --------------------------------------------------
    # Step
    # --------------------------------------------------
    def step(self, old: GameState, action: int) -> GameState:
        if old.terminal:
            return old

        # copy state
        s = GameState(
            deck=list(old.deck),
            board=list(old.board),
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
            terminal=old.terminal,
            winner=old.winner,
        )

        p = s.to_act
        opp = 1 - p

        # ---------- FOLD ----------
        if action == ACTION_FOLD:
            s.winner = opp
            s.stacks[opp] += s.pot   # award pot
            s.pot = 0.0
            s.terminal = True
            return s

        # ---------- CALL / CHECK ----------
        if action == ACTION_CALL:
            # amount needed to match current_bet
            to_call = s.current_bet

            # preflop special case: SB calling BB with no raise
            if s.street == STREET_PREFLOP and to_call == self.bb and p == s.sb_player:
                to_call = self.bb - self.sb  # 0.5 in 0.5/1 game

            to_call = min(to_call, s.stacks[p])
            s.stacks[p] -= to_call
            s.pot += to_call

            s.current_bet = 0.0
            s.last_aggressor = -1

            # all-in runout
            if s.stacks[p] == 0 and s.stacks[opp] == 0:
                return self.run_out_board_all_in(s)

            s.to_act = opp

            if self.is_betting_round_over(s):
                if s.street == STREET_RIVER:
                    self.resolve_showdown(s)
                else:
                    self.deal_next_street(s)
            return s

        # ---------- RAISE / ALL-IN ----------
        if action == ACTION_ALL_IN:
            target_total = s.stacks[p] + (self.bb if s.street == STREET_PREFLOP else 0.0)
            amount = s.stacks[p]
        elif action in RAISE_ACTIONS:
            if s.street == STREET_PREFLOP:
                base_total = self.bb
            else:
                base_total = max(s.pot, self.bb)

            target_total = RAISE_MULT[action] * base_total
            # amount to add beyond what hero has already put in this street
            amount = min(max(0.0, target_total - 0.0), s.stacks[p])
        else:
            return s  # invalid index

        if amount <= 0:
            return s

        s.stacks[p] -= amount
        s.pot += amount
        s.current_bet = target_total
        s.last_aggressor = p
        s.to_act = opp

        if s.stacks[p] == 0 and s.stacks[opp] == 0:
            return self.run_out_board_all_in(s)

        return s

    # --------------------------------------------------
    # Payoff for CFR (robust)
    # --------------------------------------------------
    def terminal_payoff(self, s: GameState, hero: int) -> float:
        if not s.terminal:
            return 0.0
        return s.stacks[hero] - s.initial_stacks[hero]
