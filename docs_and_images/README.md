# Deep CFR Poker Bot - Production Grade

A production-ready poker bot implementation using Deep Counterfactual Regret Minimization (Deep CFR) with Treys hand evaluation.

## Features

- **Deep CFR Training**: State-of-the-art poker AI training algorithm
- **Treys Integration**: Accurate poker hand evaluation using the Treys library
- **Production Ready**: 
  - Graceful Ctrl+C handling with automatic model saving
  - Comprehensive logging with iteration timing
  - Error handling and validation throughout
  - Type hints and documentation
- **Three Play Modes**:
  - Bot vs Bot: Run matches between checkpoint models
  - Bot vs Human: Interactive CLI for human players
  - Training: Train new models with configurable parameters

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Edit `config.py` to adjust:
- Poker parameters: stack sizes, blinds, action sizes
- Training parameters: iterations, learning rates, buffer sizes
- Device: Automatically selects GPU if available, falls back to CPU

## Usage

### Training a Bot

```bash
python run_deep_cfr.py
```

Features:
- Logs metrics every iteration (losses, payoffs, time per iteration)
- Automatic checkpoint saving every N iterations
- Press Ctrl+C to gracefully save models and exit
- Generates training curves after completion

**Output**: Models saved in `models/` directory
- `adv_p0.pt`: Advantage network for player 0
- `adv_p1.pt`: Advantage network for player 1
- `policy.pt`: Shared policy network
- `training_curves.png`: Loss and payoff visualization

### Bot vs Bot Matches

```bash
# Play 100 hands between two bots from the same checkpoint
python bot_match_engine.py --model1 models --model2 models --hands 100

# Or compare two different checkpoints
python bot_match_engine.py --model1 checkpoint_v1 --model2 checkpoint_v2 --hands 500 --verbose
```

**Options**:
- `--model1`, `--model2`: Path to checkpoint directories
- `--hands`: Number of hands to play
- `--verbose`: Print detailed hand history

**Output**: Win rates, average payoffs, and match statistics

### Interactive Bot vs Human

```bash
# Play against the bot from button position
python interactive_play.py --model models --hands 10 --position button

# Or play from big blind (acts first preflop)
python interactive_play.py --model models --hands 10 --position bb
```

**Options**:
- `--model`: Path to bot checkpoint directory
- `--hands`: Maximum hands to play (can exit early)
- `--position`: Your starting position (button or bb)

**CLI Commands**:
- Enter action number to fold, call, raise, or go all-in
- Type 'q' to quit mid-session

## Project Structure

```
├── run_deep_cfr.py           # Main training script
├── deep_cfr_trainer.py       # Deep CFR algorithm implementation
├── bot_match_engine.py       # Bot vs bot matches
├── interactive_play.py       # Bot vs human interactive play
├── poker_env.py              # Game environment and rules
├── networks.py               # Neural network architectures
├── abstraction.py            # State encoding with Treys evaluation
├── replay_buffer.py          # Experience replay buffers
├── config.py                 # Configuration parameters
└── requirements.txt          # Python dependencies
```

## Architecture

### State Representation

States are encoded as 16-dimensional vectors containing:
- Street (one-hot: preflop, flop, turn, river)
- Current player index
- Normalized pot and stack sizes
- Hand strength (Treys evaluation)
- Hole card features (rank, suit, pairs)

### Networks

- **Advantage Network** (AdvantageNet): 
  - 2 networks (one per player)
  - 512-dim trunk with 6 residual blocks
  - Outputs advantage values for each action
  
- **Policy Network** (PolicyNet):
  - Shared across players
  - Same architecture as AdvantageNet
  - Outputs log-probabilities over actions

### Hand Evaluation

Uses **Treys** library for accurate poker hand rankings:
- Handles all hand categories (high card to straight flush)
- Automatically finds best 5-card hand from 7 cards
- LUT-based strength estimation for state encoding

## Training Details

- **Algorithm**: Deep Counterfactual Regret Minimization (CFR)
- **Sampling Strategy**: External sampling for computational efficiency
- **Loss Functions**:
  - Advantage network: MSE of regret prediction
  - Policy network: KL divergence with masked softmax
- **Buffers**:
  - Replay buffers for experience replay
  - Separate buffers for advantages (by player) and strategy

## Logging

All operations logged to console with timestamps:
```
2025-12-09 22:28:00,143 [INFO] Iter 1: adv_buf0=248, adv_buf1=170, strat_buf=184, 
                         adv_loss=2071.6158, policy_loss=1.4141, eval_payoff_p0=-4.261, time=3.42s
```

Per-iteration metrics:
- Buffer sizes (experience collected)
- Network losses
- Average payoff in self-play
- **Time per iteration** (useful for performance tracking)

## Performance Notes

- GPU acceleration recommended for faster training (automatically detected)
- CPU mode: ~30-60 iterations per hour on modern CPU
- GPU mode: ~200-500 iterations per hour on RTX 3080

## Error Handling

- Automatic fallback from CUDA to CPU
- Graceful shutdown on Ctrl+C with model saving
- Validation of checkpoint directories before loading
- Detailed error messages with context

## Development

For modifications:
1. State representation: Update `encode_state()` in `abstraction.py`
2. Network architecture: Modify `AdvantageNet` or `PolicyNet` in `networks.py`
3. Training algorithm: Adjust `traverse()` and `train()` in `deep_cfr_trainer.py`
4. Game rules: Update `poker_env.py` (currently NLHE, fixed action sizes)

## License

This project is for educational purposes.

## References

- Deep CFR: Steinberg et al. (2020)
- Treys: Python poker hand evaluator
- PyTorch: Deep learning framework
