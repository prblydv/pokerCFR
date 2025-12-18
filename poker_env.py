import random

from dataclasses import dataclass

from typing import List



from config import STACK_SIZE, SMALL_BLIND, BIG_BLIND, RNG_SEED, NUM_PLAYERS, numPlayer

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
ACTION_SEQ_LEN = 6  # number of recent actions to keep for sequence encoding
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
    button_player: int
    initial_stacks: List[float]
    contrib: List[float]
    folded: List[bool]
    players_acted: List[bool]
    num_players: int
    actions_this_street: int = 0
    terminal: bool = False
    winner: int = -1
    action_seq: List = None

# ---------------------------------------------------------

# ---------------------------------------------------------
# SimpleHoldemEnv (FAST + CORRECT)
# ---------------------------------------------------------
class SimpleHoldemEnv:

    def __init__(self,
                 stack_size: float = STACK_SIZE,
                 sb: float = SMALL_BLIND,
                 bb: float = BIG_BLIND,
                 num_players: int = None):
        self.stack_size = stack_size
        self.sb = sb
        self.bb = bb
        configured = num_players if num_players is not None else numPlayer if numPlayer else NUM_PLAYERS
        self.num_players = max(2, int(configured))
        # Track dealer/button rotation across hands
        self._next_button = RNG.randint(0, self.num_players - 1)



    def new_hand(self) -> GameState:
        deck = list(range(52))
        RNG.shuffle(deck)

        n = self.num_players
        hole = [[deck.pop(), deck.pop()] for _ in range(n)]

        button = self._next_button
        self._next_button = (button + 1) % n

        if n == 2:
            sb = button
            bb = (button + 1) % n
        else:
            sb = (button + 1) % n
            bb = (button + 2) % n

        initial = [self.stack_size for _ in range(n)]
        stacks = initial[:]
        contrib = [0.0 for _ in range(n)]

        pot = 0.0
        pot += self._post_blind(stacks, contrib, sb, self.sb)
        pot += self._post_blind(stacks, contrib, bb, self.bb)

        current_bet = max(contrib)
        last_aggressor = bb

        folded = [False] * n
        players_acted = [False if stacks[i] > 0 else True for i in range(n)]

        # Preflop action starts left of the big blind (UTG)
        to_act = self._find_next_player(contrib, folded, stacks, start=(bb + 1) % n, include_start=True)
        if to_act is None:
            to_act = bb  # fallback if everyone else is all-in/absent

        return GameState(
            deck=deck,
            board=[],
            hole=hole,
            pot=pot,
            to_act=to_act,
            street=STREET_PREFLOP,
            stacks=stacks,
            current_bet=current_bet,
            last_aggressor=last_aggressor,
            sb_player=sb,
            bb_player=bb,
            button_player=button,
            initial_stacks=initial,
            contrib=contrib,
            folded=folded,
            players_acted=players_acted,
            num_players=n,
            actions_this_street=0,
            action_seq=[]
        )

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------
    def _post_blind(self, stacks: List[float], contrib: List[float], player: int, amount: float) -> float:
        paid = max(0.0, min(amount, stacks[player]))
        stacks[player] -= paid
        contrib[player] += paid
        return paid

    def _find_next_player(self, contrib: List[float], folded: List[bool], stacks: List[float],
                          start: int, include_start: bool = False) -> int:
        n = len(contrib)
        idx = start if include_start else (start + 1) % n
        for _ in range(n):
            if not folded[idx] and stacks[idx] > 0:
                return idx
            idx = (idx + 1) % n
        return None

    def _reset_players_acted(self, s: GameState, start_player: int) -> None:
        s.players_acted = []
        for i in range(s.num_players):
            acted = s.folded[i] or s.stacks[i] <= 0
            s.players_acted.append(acted)
        if start_player is not None and start_player >= 0 and not s.folded[start_player] and s.stacks[start_player] > 0:
            s.players_acted[start_player] = False
        s.to_act = start_player

    def _active_players(self, s: GameState) -> List[int]:
        return [i for i in range(s.num_players) if not s.folded[i]]



    def legal_actions(self, s: GameState) -> List[int]:
        if s.terminal:
            return []

        p = s.to_act
        if p is None or p < 0:
            return []

        if s.folded[p] or s.stacks[p] <= 0:
            return []

        to_call = self._amount_to_call(s, p)
        actions: List[int] = []

        if to_call > 0:
            actions.append(ACTION_FOLD)
        else:
            actions.append(ACTION_CHECK)

        if to_call > 0 and s.stacks[p] > 0:
            actions.append(ACTION_CALL)

        # Allow raises only if another live player can continue
        any_live_opponent = any(
            (i != p and not s.folded[i] and s.stacks[i] > 0)
            for i in range(s.num_players)
        )
        if any_live_opponent:
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
        s.contrib = [0.0 for _ in range(s.num_players)]
        s.actions_this_street = 0
        start = self._find_next_player(s.contrib, s.folded, s.stacks, (s.button_player + 1) % s.num_players, include_start=True)
        if start is None:
            s.to_act = -1
            s.players_acted = [True for _ in range(s.num_players)]
        else:
            self._reset_players_acted(s, start)

    def resolve_showdown(self, s: GameState):
        live_players = [i for i in range(s.num_players) if not s.folded[i]]
        if not live_players:
            s.winner = -1
            s.terminal = True
            return

        strengths = {pid: evaluate_7card(s.hole[pid], s.board) for pid in live_players}
        best = max(strengths.values())
        winners = [pid for pid, val in strengths.items() if val == best]

        payout = s.pot / len(winners) if winners else 0.0
        for pid in winners:
            s.stacks[pid] += payout

        s.winner = winners[0] if len(winners) == 1 else -1
        s.pot = 0
        s.to_act = -1
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
            button_player=old.button_player,
            initial_stacks=old.initial_stacks[:],
            contrib=old.contrib[:],
            folded=old.folded[:],
            players_acted=old.players_acted[:],
            num_players=old.num_players,
            actions_this_street=old.actions_this_street,
            terminal=False,
            winner=-1,
            action_seq=old.action_seq[:] if old.action_seq is not None else []
        )

        pot_before = s.pot
        p = s.to_act
        if p is None or p < 0 or s.folded[p] or s.stacks[p] < 0:
            return s

        s.actions_this_street += 1
        to_call = self._amount_to_call(s, p)

        if action == ACTION_FOLD:
            s.folded[p] = True
            self._record_action(s, p, action, 0.0, pot_before)
            if self._count_active(s) == 1:
                winner = self._active_players(s)[0]
                s.winner = winner
                s.stacks[winner] += s.pot
                s.pot = 0.0
                s.terminal = True
                s.to_act = -1
                return s
            self._handle_post_action_rotation(s, p)
            return s

        if action == ACTION_CHECK:
            if to_call > 0:
                return s
            s.players_acted[p] = True
            self._record_action(s, p, action, 0.0, pot_before)
            self._handle_post_action_rotation(s, p)
            return s

        if action == ACTION_CALL:
            if to_call <= 0:
                return s
            invested = self._invest(s, p, min(to_call, s.stacks[p]))
            s.players_acted[p] = True
            self._record_action(s, p, action, invested, pot_before)
            if self._all_active_all_in(s):
                return self.run_out_board_all_in(s)
            self._handle_post_action_rotation(s, p)
            return s

        if action in RAISE_ACTIONS:
            amount = self._raise_amount(s, action)
            if amount <= to_call:
                return s
            invest_amt = min(amount, s.stacks[p])
            if invest_amt <= to_call:
                return s
            self._invest(s, p, invest_amt)
            s.current_bet = max(s.current_bet, s.contrib[p])
            s.last_aggressor = p
            s.players_acted = [s.folded[i] or s.stacks[i] <= 0 for i in range(s.num_players)]
            s.players_acted[p] = True
            self._record_action(s, p, action, invest_amt, pot_before)
            if self._all_active_all_in(s):
                return self.run_out_board_all_in(s)
            self._handle_post_action_rotation(s, p)
            return s

        if action == ACTION_ALL_IN:
            amount = s.stacks[p]
            if amount <= 0:
                return s
            invested = self._invest(s, p, amount)
            if s.contrib[p] > s.current_bet:
                s.current_bet = s.contrib[p]
                s.last_aggressor = p
                s.players_acted = [s.folded[i] or s.stacks[i] <= 0 for i in range(s.num_players)]
                s.players_acted[p] = True
            else:
                s.players_acted[p] = True
            self._record_action(s, p, action, invested, pot_before)
            if self._all_active_all_in(s):
                return self.run_out_board_all_in(s)
            self._handle_post_action_rotation(s, p)
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
        if self._betting_round_complete(s):
            self._advance_street_or_showdown(s)



    def _can_raise(self, s: GameState, action: int) -> bool:

        if action not in RAISE_ACTIONS:

            return False

        p = s.to_act

        if p is None or p < 0 or s.folded[p] or s.stacks[p] <= 0:
            return False

        if not any((i != p and not s.folded[i] and s.stacks[i] > 0) for i in range(s.num_players)):
            return False

        amount = self._raise_amount(s, action)

        to_call = self._amount_to_call(s, p)

        if amount <= to_call or amount > s.stacks[p]:

            return False

        return True



    def _raise_amount(self, s: GameState, action: int) -> float:

        p = s.to_act

        to_call = self._amount_to_call(s, p)
        # print("[DEBUG]: S.TOACT, ACTION, TOCALL",s.to_act,action, to_call)

        if s.street == STREET_PREFLOP:
            if to_call <= 1 + 1e-9:
                base_size = PREFLOP_RAISE_MULT[action] * self.bb
            else:
                base_size = PREFLOP_RAISE_MULT[action] * to_call

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

    def _record_action(self, s: GameState, player: int, action: int, invested: float, pot_before: float) -> None:
        """Append an action tuple for downstream encoders (last K actions)."""
        denom = max(1.0, pot_before)
        size_norm = 0.0
        if invested > 0.0:
            size_norm = min(invested / denom, 4.0) / 4.0  # clip extreme shoves
        if s.action_seq is None:
            s.action_seq = []
        s.action_seq.append((player, action, size_norm))
        if len(s.action_seq) > ACTION_SEQ_LEN:
            s.action_seq = s.action_seq[-ACTION_SEQ_LEN:]

    # --- additional helpers for multi-way play ---
    def _count_active(self, s: GameState) -> int:
        return len([1 for i in range(s.num_players) if not s.folded[i]])

    def _all_active_all_in(self, s: GameState) -> bool:
        active = [i for i in range(s.num_players) if not s.folded[i]]
        return len(active) > 0 and all(s.stacks[i] <= 0 for i in active)

    def _betting_round_complete(self, s: GameState) -> bool:
        for pid in range(s.num_players):
            if s.folded[pid] or s.stacks[pid] <= 0:
                continue
            if not s.players_acted[pid]:
                return False
            if self._amount_to_call(s, pid) > 1e-9:
                return False
        return True

    def _advance_street_or_showdown(self, s: GameState) -> None:
        if s.terminal:
            return
        s.actions_this_street = 0
        s.contrib = [0.0 for _ in range(s.num_players)]
        s.current_bet = 0.0
        s.last_aggressor = -1
        if s.street == STREET_RIVER or self._all_active_all_in(s):
            self.run_out_board_all_in(s)
        else:
            self.deal_next_street(s)

    def _next_player_to_act(self, s: GameState, actor: int):
        n = s.num_players
        idx = (actor + 1) % n
        for _ in range(n - 1):
            if s.folded[idx] or s.stacks[idx] <= 0:
                idx = (idx + 1) % n
                continue
            needs_action = (not s.players_acted[idx]) or (self._amount_to_call(s, idx) > 1e-9)
            if needs_action:
                return idx
            idx = (idx + 1) % n
        return None

    def _handle_post_action_rotation(self, s: GameState, actor: int) -> None:
        if s.terminal:
            return
        if self._count_active(s) <= 1:
            winner = self._active_players(s)[0]
            s.winner = winner
            s.stacks[winner] += s.pot
            s.pot = 0.0
            s.terminal = True
            s.to_act = -1
            return
        if self._all_active_all_in(s):
            self.run_out_board_all_in(s)
            return
        if self._betting_round_complete(s):
            self._advance_street_or_showdown(s)
            return
        nxt = self._next_player_to_act(s, actor)
        if nxt is None:
            self._advance_street_or_showdown(s)
        else:
            s.to_act = nxt
