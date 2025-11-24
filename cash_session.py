# cash_session.py — works with new SimpleHoldemEnv

class CashSession:
    def __init__(self, env, starting_stacks=(200,200)):
        self.env = env
        self.session_stacks = list(starting_stacks)

    def start_hand(self):
        """
        Start a new cash-game hand.
        We force the environment to use our session stacks.
        """
        # Let env create a fresh hand
        s = self.env.new_hand()

        # Overwrite stacks to use session stacks
        s.stacks = self.session_stacks[:]

        # Re-post blinds using correct players
        sb = self.env.sb
        bb = self.env.bb

        sbp = s.sb_player
        bbp = s.bb_player

        s.stacks[sbp] -= sb
        s.stacks[bbp] -= bb

        s.pot = sb + bb
        s.current_bet = bb
        s.last_aggressor = bbp
        s.to_act = sbp

        # Save for payoff calculation
        s.initial_stacks = self.session_stacks[:]

        return s

    def apply_results(self, final_state):
        """
        Update cash session stacks at the end of each hand.
        session_stacks = final_state.stacks
        """
        self.session_stacks = final_state.stacks[:]

    def get_stacks(self):
        return tuple(self.session_stacks)
