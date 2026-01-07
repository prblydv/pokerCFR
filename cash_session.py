# ---------------------------------------------------------------------------
# CashSession: Correct version for SimpleHoldemEnv
# ---------------------------------------------------------------------------

class CashSession:
    def __init__(self, env, starting_stacks=None):
        self.env = env
        n = getattr(env, "num_players", 2)
        if starting_stacks is None:
            starting_stacks = tuple(env.stack_size for _ in range(n))
        self.session_stacks = list(starting_stacks)
        self._button_seat = None

    def _live_seats(self):
        return [i for i, stack in enumerate(self.session_stacks) if stack > 0]

    def _next_live_seat(self, live, start):
        if not live:
            return None
        n = len(self.session_stacks)
        for i in range(1, n + 1):
            idx = (start + i) % n
            if idx in live:
                return idx
        return live[0]

    def start_hand(self):
        """
        Start a new cash-game hand.
        We override stacks and reassign blinds based on live seats to keep
        button/SB/BB rotation among players with chips.
        """
        pre_hand_stacks = self.session_stacks[:]
        live = self._live_seats()
        if live:
            if self._button_seat is None:
                base = getattr(self.env, "_next_button", 0)
                if base not in live:
                    base = self._next_live_seat(live, base) if live else 0
                self._button_seat = base
            else:
                self._button_seat = self._next_live_seat(live, self._button_seat)
            if self._button_seat is not None:
                self.env._next_button = self._button_seat  # type: ignore[attr-defined]

        s = self.env.new_hand()

        if live:
            next_button = self._next_live_seat(live, self._button_seat if self._button_seat is not None else 0)
            if next_button is not None:
                self.env._next_button = next_button  # type: ignore[attr-defined]

        # Overwrite stacks for chip continuity
        s.stacks = pre_hand_stacks[:]

        # Reassign button/SB/BB to live seats
        if live:
            button = self._button_seat if self._button_seat is not None else s.button_player
            n = getattr(s, "num_players", 2)
            if len(live) == 2:
                sbp = button
                bbp = self._next_live_seat(live, sbp)
                if bbp is None:
                    bbp = sbp
            else:
                sbp = self._next_live_seat(live, button)
                if sbp is None:
                    sbp = button
                bbp = self._next_live_seat(live, sbp)
                if bbp is None:
                    bbp = sbp

            s.button_player = button
            s.sb_player = sbp
            s.bb_player = bbp

        sb = self.env.sb
        bb = self.env.bb
        s.contrib = [0.0 for _ in range(getattr(s, "num_players", 2))]

        sbp = s.sb_player
        bbp = s.bb_player
        sb_post = min(sb, s.stacks[sbp])
        bb_post = min(bb, s.stacks[bbp])
        s.contrib[sbp] = sb_post
        s.contrib[bbp] = bb_post
        s.stacks[sbp] -= sb_post
        s.stacks[bbp] -= bb_post
        s.pot = sb_post + bb_post
        s.current_bet = max(sb_post, bb_post)
        s.last_aggressor = bbp

        s.folded = [stack <= 0 for stack in pre_hand_stacks]

        try:
            to_act = self.env._find_next_player(
                s.contrib, s.folded, s.stacks, start=(bbp + 1) % len(s.stacks), include_start=True
            )
        except Exception:
            to_act = None
        if to_act is None:
            to_act = bbp
        s.to_act = to_act

        # Save for payoff reference
        s.initial_stacks = self.session_stacks[:]
        s.players_acted = [s.folded[i] or s.stacks[i] <= 0 for i in range(getattr(s, "num_players", 2))]

        return s

    def apply_results(self, final_state):
        """Directly adopt the final stacks from environment."""
        self.session_stacks = final_state.stacks[:]

    def get_stacks(self):
        return tuple(self.session_stacks)
