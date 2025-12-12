# DEPLOYMENT_GUIDE.md

# Deployment Guide - Production Considerations

This guide covers production deployment of the Deep CFR Poker Bot.

## System Requirements

### Minimum
- CPU: 4 cores
- RAM: 8 GB
- Storage: 2 GB
- Python 3.8+

### Recommended
- CPU: 8+ cores
- RAM: 16 GB
- GPU: NVIDIA GPU with CUDA support (2GB+ VRAM)
- Storage: 10 GB (for training checkpoints)

## Installation & Setup

### 1. Environment Setup

```bash
# Clone repository
git clone <repo-url>
cd pokerbot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import treys; print('Treys installed')"
```

## Training Configuration

### For Development
Edit `config.py`:
```python
NUM_ITERATIONS = 1000        # Quick training
TRAVERSALS_PER_ITER = 2
STRAT_SAMPLES_PER_ITER = 10
```

### For Production
```python
NUM_ITERATIONS = 50000       # Full training
TRAVERSALS_PER_ITER = 5
STRAT_SAMPLES_PER_ITER = 50
```

### GPU Acceleration

The system automatically detects and uses GPU:
```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

To force CPU:
```python
DEVICE = "cpu"
```

## Training Best Practices

### 1. Background Process

On Linux/Mac:
```bash
nohup python run_deep_cfr.py > training.log 2>&1 &
tail -f training.log  # Monitor training
```

On Windows:
```bash
# Use task scheduler or
python run_deep_cfr.py  # Keep terminal open
```

### 2. Checkpoint Management

Models are saved to `models/` directory:
- `models/adv_p0.pt` - Advantage network player 0
- `models/adv_p1.pt` - Advantage network player 1
- `models/policy.pt` - Shared policy network
- `training_curves.png` - Performance visualization

Back up checkpoints periodically:
```bash
cp -r models models_backup_$(date +%s)
```

### 3. Performance Monitoring

Watch logs for:
```
Iter 1: adv_loss=2071.6158, policy_loss=1.4141, eval_payoff_p0=-4.261, time=3.42s
```

- `adv_loss` should decrease over iterations
- `policy_loss` should decrease over iterations
- `eval_payoff_p0` should trend positive (bot improving)
- `time` shows seconds per iteration

### 4. Graceful Shutdown

Press Ctrl+C during training:
- Models are automatically saved
- Training curves are generated
- Process exits cleanly

## Deployment Scenarios

### Scenario 1: Run Training Indefinitely

```bash
#!/bin/bash
# train.sh
while true; do
    python run_deep_cfr.py
    # Backup previous checkpoint
    mv models models_v$(date +%s)
done
```

### Scenario 2: Bot Service

```bash
# bot_service.py
from bot_match_engine import BotMatchEngine

# Load trained model
engine = BotMatchEngine("models", "models")

# Serve matches on demand
for _ in range(100):
    payoff_p0, payoff_p1 = engine.play_hand()
    # Log or return results
```

### Scenario 3: Web API

```bash
# Requires Flask or FastAPI
pip install fastapi uvicorn

# Create api.py with endpoints for:
# - POST /match - run bot vs bot
# - GET /stats - get current stats
# - POST /play - human move + bot response

uvicorn api:app --host 0.0.0.0 --port 8000
```

## Monitoring & Logging

### Log Rotation (Unix/Linux)

Create `/etc/logrotate.d/pokerbot`:
```
/path/to/pokerbot/training.log {
    daily
    rotate 14
    compress
    delaycompress
    missingok
    notifempty
}
```

### Log Analysis

```bash
# Check for errors
grep ERROR training.log

# Monitor loss trends
grep "adv_loss=" training.log | tail -20

# Extract iteration times
grep "time=" training.log | awk '{print $NF}'
```

## Performance Optimization

### 1. Batch Processing

Increase `BATCH_SIZE` in `config.py`:
- Larger batches = faster GPU utilization
- Trade-off: More memory needed

```python
BATCH_SIZE = 256  # Default: 128
```

### 2. Buffer Capacity

```python
ADV_BUFFER_CAPACITY = 1_000_000  # Larger = more memory
STRAT_BUFFER_CAPACITY = 200_000
```

### 3. Network Architecture

Modify in `networks.py`:
```python
class AdvantageNet(nn.Module):
    def __init__(self, state_dim: int, 
                 trunk_dim: int = 256,      # Smaller for speed
                 num_blocks: int = 4,        # Fewer blocks
                 dropout: float = 0.0):      # No dropout in inference
```

## Troubleshooting

### Problem: CUDA Out of Memory
**Solution**: 
- Reduce `BATCH_SIZE`
- Reduce `trunk_dim` in network architecture
- Use CPU instead

### Problem: Training Too Slow
**Solution**:
- Check GPU usage: `nvidia-smi`
- Increase `BATCH_SIZE` and `trunk_dim`
- Reduce evaluation frequency

### Problem: Loss Not Decreasing
**Solution**:
- Check learning rate: `ADV_LR`, `POLICY_LR`
- Ensure sufficient training data (increase traversals)
- Verify state encoding in `abstraction.py`

### Problem: Bot Plays Suboptimally
**Solution**:
- Train longer (more iterations)
- Verify hand evaluation with Treys
- Check legal action masking

## Security Considerations

### For Betting Applications

1. **Deterministic Behavior**: Add seed control
```python
torch.manual_seed(42)
random.seed(42)
```

2. **Action Validation**: Always verify legal actions
```python
if action not in legal_actions:
    action = random.choice(legal_actions)
```

3. **Fairness**: Ensure reproducibility
- Save random state with models
- Use seeded PRNG for all randomness

### Data Security

- Encrypt checkpoints if distributing
- Version control models separately
- Audit all bot decisions in betting context

## Scaling

### Multiple GPUs

```python
# In networks.py
if torch.cuda.device_count() > 1:
    self.adv_net = nn.DataParallel(self.adv_net)
```

### Distributed Training

Consider frameworks:
- `torch.distributed`
- Ray Tune
- Horovod

### Benchmark Improvements

```
Single GPU (RTX 3080):     ~300 iterations/hour
Multi-GPU (2x RTX 3080):   ~550 iterations/hour
CPU (16 cores):            ~50 iterations/hour
```

## Compliance & Testing

### Unit Tests

```bash
python -m pytest tests.py -v
```

### Integration Tests

```bash
# Test all modes
python bot_match_engine.py --hands 10
python interactive_play.py --hands 5 < /dev/null  # Non-interactive
```

### Validation Tests

```python
# Verify hand evaluation
from abstraction import evaluate_7card
score1 = evaluate_7card([0, 1], [2, 3, 4, 5, 6])  # Should be consistent
score2 = evaluate_7card([0, 1], [2, 3, 4, 5, 6])
assert score1 == score2  # Deterministic
```

## Maintenance

### Weekly Tasks
- Backup checkpoints
- Review training logs
- Check system resources

### Monthly Tasks
- Clean up old checkpoints
- Run test matches with new version
- Update documentation

### Quarterly Tasks
- Performance benchmarking
- Model comparison analysis
- Dependency updates

## Support & Debugging

For issues, provide:
1. Log output (last 50 lines)
2. Configuration (`config.py` settings)
3. System info (`nvidia-smi`, Python version)
4. Reproduction steps

## Next Steps

1. Train initial model: `python run_deep_cfr.py`
2. Test with matches: `python bot_match_engine.py`
3. Validate with human play: `python interactive_play.py`
4. Deploy as needed: Customize `main.py`
