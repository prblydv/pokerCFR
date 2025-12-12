# ---------------------------------------------------------------------------
# File overview:
#   interactive_play.py runs a CLI two-player (manual vs manual) session.
#   Run via `python interactive_play.py --hands 5`.
# ---------------------------------------------------------------------------

"""Interactive manual poker session where you control both seats."""

import logging
import argparse
import sys
from config import LOG_LEVEL, LOG_FORMAT, STACK_SIZE
from poker_env import SimpleHoldemEnv, NUM_ACTIONS, GameState, RAISE_ACTIONS
from poker_env import (
    ACTION_FOLD,
    ACTION_CHECK,
    ACTION_CALL,
    ACTION_RAISE_SMALL,
    ACTION_RAISE_MEDIUM,
    ACTION_ALL_IN,
)
from abstraction import card_rank, card_suit

logger = logging.getLogger("InteractivePlay")

STREET_NAMES = ["Preflop", "Flop", "Turn", "River"]
ACTION_NAMES = {
    ACTION_FOLD: "FOLD",
    ACTION_CHECK: "CHECK",
    ACTION_CALL: "CALL",
    ACTION_RAISE_SMALL: "RAISE SMALL (2× BB / 0.5 pot)",
    ACTION_RAISE_MEDIUM: "RAISE MEDIUM (4× BB / 1.0 pot)",
    ACTION_ALL_IN: "ALL-IN",
}


class InteractivePokerGame:
    """Interactive poker game allowing manual control of both players."""

    def __init__(self):
        """Initialize interactive game with a fresh environment."""
        self.env = SimpleHoldemEnv()

        # Game statistics
        self.p0_wins = 0
        self.p1_wins = 0
        self.ties = 0
        self.total_payoff_p0 = 0.0
        self.total_payoff_p1 = 0.0

        logger.info("[Game] Manual vs. manual interactive session")
    
    # (bot loading/state dimension helpers removed; both seats are manual)
    
    # Function metadata:
    #   Inputs: card  # dtype=varies
    #   Sample:
    #       sample_output = _card_to_str(card=None)  # dtype=Any
    def _card_to_str(self, card: int) -> str:
        """Convert card 0..51 to string representation."""
        rank = card_rank(card)
        suit = card_suit(card)
        
        rank_str = {2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 
                    10: 'T', 11: 'J', 12: 'Q', 13: 'K', 14: 'A'}[rank]
        suit_str = ['♠', '♥', '♦', '♣'][suit]
        
        return f"{rank_str}{suit_str}"
    
    # Function metadata:
    #   Inputs: cards  # dtype=varies
    #   Sample:
    #       sample_output = _cards_to_str(cards=None)  # dtype=Any
    def _cards_to_str(self, cards: list) -> str:
        """Convert list of cards to string."""
        return " ".join(self._card_to_str(c) for c in cards)
    
    # Function metadata:
    #   Inputs: state  # dtype=varies
    #   Sample:
    #       sample_output = _print_game_state(state=mock_state)  # dtype=Any
    def _print_game_state(self, state: GameState):
        """Print current game state."""
        print("\n" + "="*60)
        print(f"Street: {STREET_NAMES[state.street]} | Pot: ${state.pot:.2f}")
        print(f"Stacks: P0=${state.stacks[0]:.2f} | P1=${state.stacks[1]:.2f}")
        print("="*60)
        
        if state.board:
            print(f"Board: {self._cards_to_str(state.board)}")
        
        print(f"Player 0 cards: {self._cards_to_str(state.hole[0])}")
        print(f"Player 1 cards: {self._cards_to_str(state.hole[1])}")

        if not state.terminal:
            to_act = state.to_act
            print(f"\nTo act: Player {to_act}")
    
    # Function metadata:
    #   Inputs: state  # dtype=varies
    #   Sample:
    #       sample_output = _get_legal_actions(state=mock_state)  # dtype=Any
    def _get_legal_actions(self, state: GameState) -> list:
        """Get legal actions for current player."""
        return self.env.legal_actions(state)
    
    # Function metadata:
    #   Inputs: state  # dtype=varies
    #   Sample:
    #       sample_output = _get_human_action(state=mock_state)  # dtype=Any
    def _describe_action(self, state: GameState, player: int, action: int) -> str:
        name = ACTION_NAMES[action]
        contrib = state.contrib[player]
        stack = state.stacks[player]
        to_call = self.env._amount_to_call(state, player)

        if action == ACTION_FOLD:
            return name
        if action == ACTION_CHECK:
            return f"{name} (no chips)"
        if action == ACTION_CALL:
            invest = min(to_call, stack)
            total = contrib + invest
            return f"{name} (+${invest:.2f}, total ${total:.2f})"
        if action in RAISE_ACTIONS:
            invest = min(self.env._raise_amount(state, action), stack)
            extra = max(0.0, invest - to_call)
            total = contrib + invest
            return f"{name} (+${extra:.2f} beyond call, total ${total:.2f})"
        if action == ACTION_ALL_IN:
            invest = stack
            total = contrib + invest
            return f"{name} (push ${invest:.2f}, total ${total:.2f})"
        return name

    def _get_human_action(self, state: GameState, player: int) -> int:
        """Get action from human player via CLI."""
        legal_actions = self._get_legal_actions(state)
        
        while True:
            print(f"\nAvailable actions for Player {player}:")
            for i, action in enumerate(legal_actions, 1):
                desc = self._describe_action(state, player, action)
                print(f"  {i}. {desc}")
            
            try:
                choice = input("\nEnter action number (or 'q' to quit): ").strip()
                
                if choice.lower() == 'q':
                    logger.info("Game quit by player")
                    sys.exit(0)
                
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(legal_actions):
                    return legal_actions[choice_idx]
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a number.")
    
    
    # Function metadata:
    #   Inputs: no explicit parameters  # dtype=varies
    #   Sample:
    #       sample_output = play_hand()  # dtype=Any
    def play_hand(self) -> float:
        """
        Play a single hand.
        
        Returns:
            Payoff for human player (positive = won money)
        """
        state = self.env.new_hand()
        
        logger.info("Starting new hand...")
        
        while not state.terminal:
            self._print_game_state(state)
            
            player = state.to_act
            action = self._get_human_action(state, player)
            state = self.env.step(state, action)
        
        self._print_game_state(state)
        
        payoff_p0 = self.env.terminal_payoff(state, 0)
        payoff_p1 = self.env.terminal_payoff(state, 1)
        
        print("\n" + "="*60)
        if payoff_p0 > payoff_p1:
            print(f"Player 0 wins +${payoff_p0:.2f}")
            self.p0_wins += 1
        elif payoff_p1 > payoff_p0:
            print(f"Player 1 wins +${payoff_p1:.2f}")
            self.p1_wins += 1
        else:
            print("Hand ends in a tie")
            self.ties += 1
        print("="*60)
        
        self.total_payoff_p0 += payoff_p0
        self.total_payoff_p1 += payoff_p1
        return payoff_p0, payoff_p1
    
    # Function metadata:
    #   Inputs: num_hands  # dtype=varies
    #   Sample:
    #       sample_output = play_session(num_hands=100)  # dtype=Any
    def play_session(self, num_hands: int):
        """Play a session of poker."""
        logger.info(f"Starting session: {num_hands} hands")
        print(f"\n{'='*60}")
        print(f"POKER SESSION - {num_hands} HANDS")
        print(f"{'='*60}\n")
        
        try:
            for hand_num in range(1, num_hands + 1):
                print(f"\n--- HAND {hand_num}/{num_hands} ---")
                self.play_hand()
                
                print(f"\nSession Stats:")
                print(f"  Player0 Wins: {self.p0_wins} | Player1 Wins: {self.p1_wins} | Ties: {self.ties}")
                print(f"  Total P0: ${self.total_payoff_p0:.2f} | Total P1: ${self.total_payoff_p1:.2f}")
                
                play_again = input("\nContinue? (y/n): ").strip().lower()
                if play_again != 'y':
                    break
        
        except KeyboardInterrupt:
            print("\n\nSession interrupted by user.")
        
        self._print_session_summary()
    
    # Function metadata:
    #   Inputs: no explicit parameters  # dtype=varies
    #   Sample:
    #       sample_output = _print_session_summary()  # dtype=Any
    def _print_session_summary(self):
        """Print session summary."""
        print(f"\n{'='*60}")
        print("SESSION SUMMARY")
        print(f"{'='*60}")
        total_hands = self.p0_wins + self.p1_wins + self.ties
        print(f"Total hands:      {total_hands}")
        print(f"Player0 wins:     {self.p0_wins}")
        print(f"Player1 wins:     {self.p1_wins}")
        print(f"Ties:             {self.ties}")
        print(f"Total P0 payoff:  ${self.total_payoff_p0:.2f}")
        print(f"Total P1 payoff:  ${self.total_payoff_p1:.2f}")
        if total_hands > 0:
            avg0 = self.total_payoff_p0 / total_hands
            avg1 = self.total_payoff_p1 / total_hands
            print(f"Avg per hand P0:  ${avg0:.4f}")
            print(f"Avg per hand P1:  ${avg1:.4f}")
        print(f"{'='*60}\n")


# Function metadata:
#   Inputs: no explicit parameters  # dtype=varies
#   Sample:
#       sample_output = setup_logging()  # dtype=Any
def setup_logging():
    """Configure logging."""
    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)


# Function metadata:
#   Inputs: no explicit parameters  # dtype=varies
#   Sample:
#       sample_output = main()  # dtype=Any
def main():
    parser = argparse.ArgumentParser(description="Play poker manually (Player 0 vs Player 1)")
    parser.add_argument("--hands", type=int, default=10,
                        help="Number of hands to play (default: 10)")
    
    args = parser.parse_args()
    
    setup_logging()
    
    try:
        game = InteractivePokerGame()
        game.play_session(args.hands)
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
