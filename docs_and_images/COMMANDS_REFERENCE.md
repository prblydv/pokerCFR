# COMMANDS_REFERENCE.md

# Command Reference

Quick reference for all available commands.

## Installation

```bash
# Install dependencies (run once)
pip install -r requirements.txt

# Verify installation
python -c "import torch; print('PyTorch OK')"
python -c "import treys; print('Treys OK')"
```

## Main Entry Points

### Option 1: Interactive Menu (Recommended for First Time)

```bash
python main.py
```

Menu options:
1. Train a new bot
2. Bot vs bot match
3. Play against bot (interactive)
4. View training curves
5. List available checkpoints
6. Exit

---

### Option 2: Direct Training

```bash
# Start training with default config
python run_deep_cfr.py

# Press Ctrl+C anytime to save and exit gracefully
```

Output:
- `models/adv_p0.pt` - Advantage network for player 0
- `models/adv_p1.pt` - Advantage network for player 1  
- `models/policy.pt` - Shared policy network
- `training_curves.png` - Performance visualization

---

### Option 3: Bot vs Bot Matches

#### Basic Match
```bash
python bot_match_engine.py
```

#### Full Options
```bash
python bot_match_engine.py \
    --model1 models \
    --model2 models \
    --hands 100 \
    --verbose
```

**Arguments**:
- `--model1 <path>`: First bot checkpoint (default: `models`)
- `--model2 <path>`: Second bot checkpoint (default: `models`)
- `--hands <n>`: Number of hands (default: `100`)
- `--verbose`: Print hand history

---

### Option 4: Bot vs Human Interactive

#### Quick Game
```bash
python interactive_play.py
```

#### Full Options
```bash
python interactive_play.py \
    --model models \
    --hands 10 \
    --position button
```

**Arguments**:
- `--model <path>`: Bot checkpoint (default: `models`)
- `--hands <n>`: Max hands to play (default: `10`)
- `--position <button|bb>`: Your position (default: `button`)

**During game**:
- Enter action number (1-10) to take action
- Type `q` to quit
- Type `n` to play next hand

---

## Configuration

Edit `config.py` to adjust:

### Training Parameters
```python
NUM_ITERATIONS = 1000          # Quick dev
NUM_ITERATIONS = 50000         # Production

TRAVERSALS_PER_ITER = 5        # CFR traversals
STRAT_SAMPLES_PER_ITER = 50    # Strategy samples

BATCH_SIZE = 128               # Training batch size
ADV_LR = 1e-3                  # Advantage learning rate
POLICY_LR = 1e-3               # Policy learning rate
```

### Game Parameters
```python
STACK_SIZE = 200.0             # Starting stack
SMALL_BLIND = 0.5              # SB amount
BIG_BLIND = 1.0                # BB amount
NUM_ACTIONS = 10               # Actions: fold, call, 7 raises, all-in
```

### Device Configuration
```python
# Automatic detection (GPU if available, else CPU)
# Force CPU:
DEVICE = "cpu"
```

---

## Advanced Usage

### Compare Two Models

```bash
# Train first model
python run_deep_cfr.py > training1.log

# Train second model (or copy and modify)
cp -r models models_v2

# Compare them
python bot_match_engine.py \
    --model1 models \
    --model2 models_v2 \
    --hands 500 \
    --verbose
```

### Save Training Progress

```bash
# Backup models periodically
cp -r models models_backup_$(date +%Y%m%d_%H%M%S)
```

### Monitor Performance

```bash
# Watch training in real-time
tail -f training.log | grep "Iter"

# Extract metrics
grep "adv_loss=" training.log
grep "time=" training.log
```

### Run Multiple Matches

```bash
# Parallel testing
python bot_match_engine.py --hands 500 > match1.log &
python bot_match_engine.py --hands 500 > match2.log &
wait
```

---

## Troubleshooting Commands

### Check GPU Status
```bash
python -c "import torch; print('GPU:', torch.cuda.is_available())"
nvidia-smi  # If installed
```

### Verify Models
```bash
ls -lh models/
# Should show: adv_p0.pt, adv_p1.pt, policy.pt
```

### Test Installation
```bash
python -c "from bot_match_engine import BotMatchEngine; print('✓ OK')"
python -c "from interactive_play import InteractivePokerGame; print('✓ OK')"
```

### View Logs
```bash
# Last 20 lines of training
tail -20 training.log

# Search for errors
grep ERROR training.log

# Count iterations
grep "Iter [0-9]" training.log | wc -l
```

---

## Common Workflows

### Workflow 1: Quick Test (15 minutes)

```bash
# 1. Install (first time only)
pip install -r requirements.txt

# 2. Train briefly
python run_deep_cfr.py  # Let run for ~5 min, then Ctrl+C

# 3. Test bot
python bot_match_engine.py --hands 20

# 4. Play against it
python interactive_play.py --hands 3
```

### Workflow 2: Development (1-2 hours)

```bash
# Edit config.py: NUM_ITERATIONS = 1000
python run_deep_cfr.py  # Will take ~1 hour

# Test thoroughly
python bot_match_engine.py --hands 100 --verbose
python interactive_play.py --hands 10

# Backup if good
cp -r models models_dev_v1
```

### Workflow 3: Production Training (1-2 days)

```bash
# Edit config.py: NUM_ITERATIONS = 50000
python run_deep_cfr.py  # Start and wait

# Monitor progress (in another terminal)
tail -f training.log | grep time=

# After completion, test
python bot_match_engine.py --hands 500
```

### Workflow 4: Validate with Tournament

```bash
# Run 1000-hand match for validation
python bot_match_engine.py --hands 1000 --verbose

# View statistics
tail training_match.log

# If satisfied, deploy
cp -r models models_production_v1
```

---

## Performance Monitoring

### Watch Iteration Time
```bash
grep "time=" training.log | tail -20
```

### Watch Loss Trends
```bash
grep "adv_loss=" training.log | \
  awk '{print $NF}' | \
  tail -20
```

### Watch Win Rate During Training
```bash
grep "eval_payoff_p0=" training.log | \
  awk '{print $NF}' | \
  tail -20
```

---

## File Organization

### After Training
```
.
├── models/
│   ├── adv_p0.pt          ← Player 0 advantage
│   ├── adv_p1.pt          ← Player 1 advantage
│   └── policy.pt          ← Shared policy
├── training_curves.png    ← Performance graph
└── training.log           ← Training output
```

### Save Checkpoint
```bash
mkdir models_v1_$(date +%s)
cp models/* models_v1_$(date +%s)/
```

### Load Old Checkpoint
```bash
python bot_match_engine.py --model1 models_v1_1234567890 --hands 100
```

---

## Quick Command Cheat Sheet

| Task | Command |
|------|---------|
| Interactive menu | `python main.py` |
| Train bot | `python run_deep_cfr.py` |
| Bot vs bot | `python bot_match_engine.py --hands 100` |
| Play game | `python interactive_play.py --hands 10` |
| View curves | Open `training_curves.png` |
| List models | `ls -la models/` |
| Backup | `cp -r models models_backup` |
| Monitor | `tail -f training.log` |
| Check GPU | `nvidia-smi` |
| Stop training | `Ctrl+C` (saves automatically) |

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Error (check logs) |
| 130 | Interrupted by Ctrl+C (normal) |

---

## Getting Help

For each tool:
```bash
python run_deep_cfr.py --help      # (shows config.py settings)
python bot_match_engine.py --help   # Shows all options
python interactive_play.py --help   # Shows all options
python main.py                      # Interactive menu
```

Documentation:
- **README.md** - Full documentation
- **QUICKSTART.md** - 5-minute guide
- **DEPLOYMENT_GUIDE.md** - Production setup
- **COMMANDS_REFERENCE.md** - This file

---

## Examples

### Example 1: Quick Development
```bash
python run_deep_cfr.py  # 5 min, Ctrl+C
python bot_match_engine.py --hands 50
```

### Example 2: Train and Compare
```bash
cp -r models models_old
python run_deep_cfr.py  # Retrain
python bot_match_engine.py --model1 models_old --model2 models --hands 200
```

### Example 3: Marathon Session
```bash
python interactive_play.py --hands 100 --position button
# Play up to 100 hands against bot
```

### Example 4: Benchmark Different Configs
```bash
# Test 1
python bot_match_engine.py --model1 v1 --model2 v2 --hands 300
# Compare results
```

---

That's it! You now have all the commands you need to train, test, and play with your poker bot. 🎮
