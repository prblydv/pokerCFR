# QUICKSTART.md

# Quick Start Guide

Get the Deep CFR Poker Bot up and running in 5 minutes.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- ~2 GB disk space
- (Optional) NVIDIA GPU with CUDA support for faster training

## Installation

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

Verify installation:
```bash
python -c "import torch; print('✓ PyTorch installed')"
python -c "import treys; print('✓ Treys installed')"
```

### Step 2: Verify Configuration

Check device detection:
```bash
python -c "import torch; print('Device:', 'CUDA (GPU)' if torch.cuda.is_available() else 'CPU')"
```

## Option 1: Quick Training (Development)

Train a bot in ~30 minutes on CPU or ~5 minutes on GPU:

```bash
python run_deep_cfr.py
```

You'll see output like:
```
2025-12-09 22:28:00,143 [INFO] Iter 1: adv_buf0=248, adv_buf1=170, strat_buf=184, 
                         adv_loss=2071.6158, policy_loss=1.4141, eval_payoff_p0=-4.261, time=3.42s
```

**Stop anytime**: Press `Ctrl+C` to save models and exit gracefully.

**Output files**:
- `models/` - Trained bot checkpoint
- `training_curves.png` - Performance visualization

## Option 2: Interactive Menu

Use the interactive menu for all features:

```bash
python main.py
```

Choose from:
1. Train a new bot
2. Play bot vs bot matches
3. Play against the bot (human vs AI)
4. View training curves
5. List available checkpoints

## Option 3: Direct CLI

### Run a Bot vs Bot Match

```bash
python bot_match_engine.py --hands 50 --verbose
```

Output:
```
============================================================
MATCH STATISTICS
============================================================
Total hands:      50
P0 wins:          23 (46.0%)
P1 wins:          27 (54.0%)
P0 avg payoff:    -0.0864 per hand
```

### Play Against the Bot

```bash
python interactive_play.py --hands 5 --position button
```

Game interface:
```
============================================================
Street: Preflop | Pot: $1.50
Stacks: P0=$199.50 | P1=$199.50
============================================================

Your cards: A♠K♦

To act: YOU (P0)

Available actions:
  1. FOLD
  2. CALL/CHECK
  3. RAISE 2×
  ...
```

## Common Tasks

### Train for Production (Full Run)

Edit `config.py`:
```python
NUM_ITERATIONS = 50000      # Full training
```

Then:
```bash
python run_deep_cfr.py
```

Expected time: 12-48 hours (depends on GPU)

### Compare Two Models

```bash
python bot_match_engine.py \
    --model1 models \
    --model2 older_checkpoint \
    --hands 500
```

### Batch Play Session

```bash
python interactive_play.py --hands 100 --position bb
```

Plays up to 100 hands (can exit early by pressing 'q').

## Troubleshooting

### Issue: "Torch not compiled with CUDA enabled"
**Solution**: Uses CPU automatically (slower but works)

### Issue: Out of memory
**Solution**: Reduce in `config.py`
```python
BATCH_SIZE = 64  # Lower from 128
ADV_BUFFER_CAPACITY = 250_000  # Lower from 500_000
```

### Issue: Training is too slow
**Solution**: Check GPU usage
```bash
nvidia-smi  # Shows GPU memory and utilization
```

If GPU shows 0%, force CPU manually in code or wait for next iteration.

## File Structure

```
.
├── run_deep_cfr.py           ← Main training script
├── bot_match_engine.py       ← Bot vs bot matches
├── interactive_play.py       ← Bot vs human play
├── main.py                   ← Interactive menu
├── config.py                 ← Configuration
├── models/                   ← Trained bots (created after training)
│   ├── adv_p0.pt
│   ├── adv_p1.pt
│   └── policy.pt
└── training_curves.png       ← Performance graphs
```

## Next Steps

1. **Train**: `python run_deep_cfr.py` (or use main.py option 1)
2. **Test**: `python bot_match_engine.py --hands 100`
3. **Play**: `python interactive_play.py --hands 10`
4. **Iterate**: Adjust config.py and retrain

## Understanding Output

### Training Log

```
Iter 1: adv_buf0=248, adv_buf1=170, strat_buf=184, adv_loss=2071.6158, 
        policy_loss=1.4141, eval_payoff_p0=-4.261, time=3.42s
```

- `adv_buf0=248`: 248 examples in player 0 advantage buffer
- `adv_loss=2071.6158`: Loss of advantage network (should ↓ over time)
- `policy_loss=1.4141`: Loss of policy network (should ↓ over time)
- `eval_payoff_p0=-4.261`: Average winnings per hand (should ↑ over time)
- `time=3.42s`: Seconds per iteration

### Match Results

```
P0 wins:   45 (45.0%)        ← Bot 1 win rate
P1 wins:   55 (55.0%)        ← Bot 2 win rate
P0 avg payoff: -0.1234       ← Average $/hand for Bot 1
```

## Pro Tips

1. **Save checkpoints**: After training, copy `models/` directory:
   ```bash
   cp -r models models_v1_finished
   ```

2. **Monitor training**: Open `training_curves.png` to see progress

3. **Fast iteration**: During development, use low iteration count:
   ```python
   NUM_ITERATIONS = 100  # Quick test
   ```

4. **GPU optimization**: Run multiple concurrent matches:
   ```bash
   python bot_match_engine.py --hands 1000 &
   python bot_match_engine.py --hands 1000 &
   ```

## Getting Help

- **README.md** - Full documentation
- **DEPLOYMENT_GUIDE.md** - Production setup
- **PRODUCTION_SUMMARY.md** - What changed and why
- Check `config.py` comments for parameter explanations

## Example Workflow

```bash
# 1. Install (one time)
pip install -r requirements.txt

# 2. Quick training (test)
python run_deep_cfr.py
# (Wait ~30 min on CPU or 5 min on GPU, then Ctrl+C)

# 3. Test the bot
python bot_match_engine.py --hands 50

# 4. Play against it
python interactive_play.py --hands 5

# 5. Full training (production)
# Edit config.py: NUM_ITERATIONS = 50000
python run_deep_cfr.py
# (Wait 1-2 days for full training)

# 6. Validate with tournament
python bot_match_engine.py --hands 500 --verbose
```

That's it! You now have a production-grade poker bot. 🎮♠️
