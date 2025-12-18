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

    def start_hand(self):
        """
        Start a new cash-game hand.
        We override ONLY the stacks inside the new GameState,
        but DO NOT repost blinds, DO NOT recalc pot, DO NOT touch to_act.
        The environment already handles blinds & setup correctly.
        """
        s = self.env.new_hand()

        # Overwrite stacks for chip continuity
        s.stacks = self.session_stacks[:]

        # Re-apply blinds EXACTLY the way env.new_hand() already did:
        # SB and BB have already been deducted in new_hand(),
        # so we must simply recompute contrib/pot to match stacks.
        sbp = s.sb_player
        bbp = s.bb_player
        sb = self.env.sb
        bb = self.env.bb

        # Reset contributions consistent with new stacks
        s.contrib = [0.0 for _ in range(getattr(s, "num_players", 2))]
        sb_post = min(sb, s.stacks[sbp])
        bb_post = min(bb, s.stacks[bbp])
        s.contrib[sbp] = sb_post
        s.contrib[bbp] = bb_post

        # Deduct blinds properly (cap at available stack to avoid negatives)
        s.stacks[sbp] -= sb_post
        s.stacks[bbp] -= bb_post

        # Recompute pot + betting state (current bet is the larger posted blind)
        s.pot = sb_post + bb_post
        s.current_bet = max(sb_post, bb_post)

        # Save for payoff reference
        s.initial_stacks = self.session_stacks[:]
        s.players_acted = [s.folded[i] or s.stacks[i] <= 0 for i in range(getattr(s, "num_players", 2))]

        return s

    def apply_results(self, final_state):
        """Directly adopt the final stacks from environment."""
        self.session_stacks = final_state.stacks[:]

    def get_stacks(self):
        return tuple(self.session_stacks)
