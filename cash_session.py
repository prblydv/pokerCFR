# ---------------------------------------------------------------------------
# File overview:
#   cash_session.py tracks stacks for repeated hands with SimpleHoldemEnv.
#   Import CashSession from CLI or interactive play scripts.
# ---------------------------------------------------------------------------

class CashSession:
    # Function metadata:
    #   Inputs: env, starting_stacks  # dtype=varies
    #   Sample:
    #       sample_output = __init__(env=mock_env, starting_stacks=None)  # dtype=Any
    def __init__(self, env, starting_stacks=(200,200)):
        self.env = env
        self.session_stacks = list(starting_stacks)

    # Function metadata:
    #   Inputs: no explicit parameters  # dtype=varies
    #   Sample:
    #       sample_output = start_hand()  # dtype=Any
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

    # Function metadata:
    #   Inputs: final_state  # dtype=varies
    #   Sample:
    #       sample_output = apply_results(final_state=None)  # dtype=Any
    def apply_results(self, final_state):
        """
        Update cash session stacks at the end of each hand.
        session_stacks = final_state.stacks
        """
        self.session_stacks = final_state.stacks[:]

    # Function metadata:
    #   Inputs: no explicit parameters  # dtype=varies
    #   Sample:
    #       sample_output = get_stacks()  # dtype=Any
    def get_stacks(self):
        return tuple(self.session_stacks)
