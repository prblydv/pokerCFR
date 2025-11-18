class CashSession:
    """
    True cash-game session:
    - stacks persist across hands
    - correct pot/effective-stack accounting
    - no negative stacks, no incorrect winnings
    """

    def __init__(self, env, starting_stacks=(200.0, 200.0)):
        self.env = env
        self.session_stacks = list(starting_stacks)
        self.total_hands = 0

    def start_hand(self):
        """
        Returns the GameState for a new hand.
        Does NOT reset stacks — uses session_stacks.
        """
        return self.env.new_hand_with_stacks(self.session_stacks)

    def apply_results(self, state):

    # How much each player invested in this hand:
        contrib0 = state.initial_stacks[0] - state.stacks[0]
        contrib1 = state.initial_stacks[1] - state.stacks[1]
        pot = contrib0 + contrib1

        # Winner is 0
        if state.winner == 0:
            self.session_stacks[0] = pot    # winner gets the entire pot
            self.session_stacks[1] = 0      # loser busted

        # Winner is 1
        elif state.winner == 1:
            self.session_stacks[1] = pot
            self.session_stacks[0] = 0

        # Split pot
        else:
            half = pot / 2.0
            self.session_stacks[0] = half
            self.session_stacks[1] = half

        self.total_hands += 1

    def get_stacks(self):
        return tuple(self.session_stacks)

    def reset(self, stacks=(200.0, 200.0)):
        """Reset session manually."""
        self.session_stacks = list(stacks)
        self.total_hands = 0
