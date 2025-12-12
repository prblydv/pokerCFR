# ---------------------------------------------------------------------------
# File overview:
#   bot_match_engine.py loads two checkpoints and pits them head-to-head.
#   Run via `python bot_match_engine.py --model1 models --model2 models`.
# ---------------------------------------------------------------------------
"""
Bot vs Bot Match Engine

Allows playing poker hands between two bots loaded from checkpoint files.
Supports match statistics, logging, and detailed hand history.

Usage:
    python bot_match_engine.py --model1 models/checkpoint1 --model2 models/checkpoint2 --hands 100
"""

import logging
import os
import argparse
from pathlib import Path
from typing import Tuple, List
import torch

from config import DEVICE, LOG_LEVEL, LOG_FORMAT
from poker_env import SimpleHoldemEnv, NUM_ACTIONS
from abstraction import encode_state
from networks import AdvantageNet, PolicyNet, move_to_device
from deep_cfr_trainer import DeepCFRTrainer

logger = logging.getLogger("BotMatch")


class BotMatchEngine:
    """Engine for playing poker between two bot checkpoints."""
    
    # Function metadata:
    #   Inputs: model_path1 (str), model_path2 (str)
    #   Sample:
    #       engine = BotMatchEngine("models/botA", "models/botB")
    def __init__(self, model_path1: str, model_path2: str):
        """
        Initialize bot match engine.
        
        Args:
            model_path1: Path to first bot's checkpoint directory
            model_path2: Path to second bot's checkpoint directory
        """
        self.env = SimpleHoldemEnv()
        self.model_path1 = model_path1
        self.model_path2 = model_path2
        
        # Load bots
        self.state_dim = self._get_state_dim()
        self.bot1 = self._load_bot_from_checkpoint(model_path1)
        self.bot2 = self._load_bot_from_checkpoint(model_path2)
        
        # Match statistics
        self.p0_wins = 0
        self.p1_wins = 0
        self.ties = 0
        self.total_hands = 0
        self.p0_total_payoff = 0.0
        self.p1_total_payoff = 0.0
        
        logger.info(f"[BotMatch] Loaded Bot1 from {model_path1}")
        logger.info(f"[BotMatch] Loaded Bot2 from {model_path2}")
    
    # Function metadata:
    #   Inputs: None (uses env)
    #   Sample:
    #       dim = engine._get_state_dim()  # dtype=int
    def _get_state_dim(self) -> int:
        """Get state dimension by encoding an example state."""
        env = SimpleHoldemEnv()
        state = env.new_hand()
        encoded = encode_state(state, 0)
        return encoded.shape[0]
    
    # Function metadata:
    #   Inputs: checkpoint_path (str)
    #   Sample:
    #       bot = engine._load_bot_from_checkpoint("models/botA")  # dtype=dict
    def _load_bot_from_checkpoint(self, checkpoint_path: str) -> dict:
        """Load bot networks from checkpoint directory."""
        if not os.path.isdir(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")
        
        # Load networks
        adv_p0 = move_to_device(AdvantageNet(self.state_dim))
        adv_p1 = move_to_device(AdvantageNet(self.state_dim))
        policy = move_to_device(PolicyNet(self.state_dim))
        
        # Load weights
        adv_p0.load_state_dict(torch.load(f"{checkpoint_path}/adv_p0.pt", map_location=DEVICE))
        adv_p1.load_state_dict(torch.load(f"{checkpoint_path}/adv_p1.pt", map_location=DEVICE))
        policy.load_state_dict(torch.load(f"{checkpoint_path}/policy.pt", map_location=DEVICE))
        
        adv_p0.eval()
        adv_p1.eval()
        policy.eval()
        
        return {
            "adv_nets": [adv_p0, adv_p1],
            "policy_net": policy
        }
    
    # Function metadata:
    #   Inputs: state (GameState), player (int), bot_index (int)
    #   Sample:
    #       action = engine.choose_action(state, player=0, bot_index=0)  # dtype=int
    def choose_action(self, state, player: int, bot_index: int) -> int:
        """
        Choose action for bot using policy network.
        
        Args:
            state: Current game state
            player: Player to act (0 or 1)
            bot_index: Which bot (0 or 1)
        
        Returns:
            Action index
        """
        bot = self.bot1 if bot_index == 0 else self.bot2
        policy_net = bot["policy_net"]
        
        x = encode_state(state, player).to(DEVICE)
        
        with torch.no_grad():
            logp = policy_net(x.unsqueeze(0)).squeeze(0)
        
        legal_actions = self.env.legal_actions(state)
        
        # Apply mask to illegal actions
        mask = torch.full((NUM_ACTIONS,), -1e9, device=DEVICE)
        for a in legal_actions:
            mask[a] = 0
        
        probs = torch.softmax(logp + mask, dim=-1)
        action = torch.multinomial(probs, 1).item()
        
        # Fallback to random legal action if needed
        if action not in legal_actions:
            import random
            action = random.choice(legal_actions)
        
        return action
    
    # Function metadata:
    #   Inputs: verbose (bool)
    #   Sample:
    #       winner, p0, p1 = engine.play_hand(verbose=True)
    #       # dtype=(int,float,float)
    def play_hand(self, verbose: bool = False) -> Tuple[int, float, float]:
        """
        Play a single poker hand.
        
        Returns:
            (winner, p0_payoff, p1_payoff)
        """
        state = self.env.new_hand()
        hand_history = []
        
        while not state.terminal:
            player = state.to_act
            # Determine which bot is playing (account for button rotations)
            bot_idx = player
            
            action = self.choose_action(state, player, bot_idx)
            
            if verbose:
                action_names = ["FOLD", "CALL/CHECK", "2×", "2.25×", "2.5×", "3×", "3.5×", "4.5×", "6×", "ALL-IN"]
                hand_history.append(f"P{player} → {action_names[action]}")
            
            state = self.env.step(state, action)
        
        p0_payoff = self.env.terminal_payoff(state, 0)
        p1_payoff = -p0_payoff  # Zero-sum game
        
        if verbose:
            logger.info(f"Hand: {' | '.join(hand_history)} → P0: {p0_payoff:.2f}, P1: {p1_payoff:.2f}")
        
        return state.winner, p0_payoff, p1_payoff
    
    # Function metadata:
    #   Inputs: num_hands (int), verbose (bool)
    #   Sample:
    #       stats = engine.run_match(num_hands=100)  # dtype=dict
    def run_match(self, num_hands: int, verbose: bool = False) -> dict:
        """
        Run a match between two bots.
        
        Args:
            num_hands: Number of hands to play
            verbose: Print detailed hand information
        
        Returns:
            Match statistics dictionary
        """
        logger.info(f"[BotMatch] Starting match: {num_hands} hands")
        
        for hand_num in range(1, num_hands + 1):
            winner, p0_payoff, p1_payoff = self.play_hand(verbose=verbose)
            
            self.total_hands += 1
            self.p0_total_payoff += p0_payoff
            self.p1_total_payoff += p1_payoff
            
            if p0_payoff > 0:
                self.p0_wins += 1
            elif p0_payoff < 0:
                self.p1_wins += 1
            else:
                self.ties += 1
            
            if hand_num % max(1, num_hands // 10) == 0:
                avg_p0 = self.p0_total_payoff / self.total_hands
                logger.info(
                    f"[Progress] {hand_num}/{num_hands} hands - "
                    f"P0: {self.p0_wins}W, P1: {self.p1_wins}W, Ties: {self.ties}, "
                    f"Avg P0 payoff: {avg_p0:.3f}"
                )
        
        return self.get_stats()
    
    # Function metadata:
    #   Inputs: None
    #   Sample:
    #       stats = engine.get_stats()  # dtype=dict
    def get_stats(self) -> dict:
        """Get match statistics."""
        stats = {
            "total_hands": self.total_hands,
            "p0_wins": self.p0_wins,
            "p1_wins": self.p1_wins,
            "ties": self.ties,
            "p0_win_rate": self.p0_wins / self.total_hands if self.total_hands > 0 else 0,
            "p1_win_rate": self.p1_wins / self.total_hands if self.total_hands > 0 else 0,
            "p0_total_payoff": self.p0_total_payoff,
            "p1_total_payoff": self.p1_total_payoff,
            "p0_avg_payoff": self.p0_total_payoff / self.total_hands if self.total_hands > 0 else 0,
            "p1_avg_payoff": self.p1_total_payoff / self.total_hands if self.total_hands > 0 else 0,
        }
        return stats
    
    # Function metadata:
    #   Inputs: None
    #   Sample:
    #       engine.print_stats()  # dtype=NoneType (logs)
    def print_stats(self):
        """Print match statistics to logger."""
        stats = self.get_stats()
        logger.info("="*60)
        logger.info("MATCH STATISTICS")
        logger.info("="*60)
        logger.info(f"Total hands:      {stats['total_hands']}")
        logger.info(f"P0 wins:          {stats['p0_wins']} ({stats['p0_win_rate']:.1%})")
        logger.info(f"P1 wins:          {stats['p1_wins']} ({stats['p1_win_rate']:.1%})")
        logger.info(f"Ties:             {stats['ties']}")
        logger.info(f"P0 total payoff:  {stats['p0_total_payoff']:.2f}")
        logger.info(f"P1 total payoff:  {stats['p1_total_payoff']:.2f}")
        logger.info(f"P0 avg payoff:    {stats['p0_avg_payoff']:.4f} per hand")
        logger.info(f"P1 avg payoff:    {stats['p1_avg_payoff']:.4f} per hand")
        logger.info("="*60)


# Function metadata:
#   Inputs: None
#   Sample:
#       setup_logging()  # dtype=NoneType
def setup_logging():
    """Configure logging."""
    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)


# Function metadata:
#   Inputs: None (CLI entry point)
#   Sample:
#       exit_code = main()  # dtype=int
def main():
    parser = argparse.ArgumentParser(description="Run bot vs bot matches")
    parser.add_argument("--model1", type=str, default="models", 
                        help="Path to first bot checkpoint (default: models)")
    parser.add_argument("--model2", type=str, default="models",
                        help="Path to second bot checkpoint (default: models)")
    parser.add_argument("--hands", type=int, default=100,
                        help="Number of hands to play (default: 100)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print hand history")
    
    args = parser.parse_args()
    
    setup_logging()
    
    try:
        engine = BotMatchEngine(args.model1, args.model2)
        stats = engine.run_match(args.hands, verbose=args.verbose)
        engine.print_stats()
        
    except Exception as e:
        logger.error(f"Error running match: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
