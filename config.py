# ---------------------------------------------------------------------------
# File overview:
#   config.py centralizes hyperparameters/constants for poker environment,
#   Deep CFR, and hardware selection. Import it; not executable standalone.
# ---------------------------------------------------------------------------
import logging
import time
import torch

# ---------------------------------------------------------------------------
# Randomness
# ---------------------------------------------------------------------------
# Use a fixed seed for reproducible training runs.
DETERMINISTIC_SEED = 49244
RNG_SEED = DETERMINISTIC_SEED

# If you want non-deterministic runs, uncomment:
# RNG_SEED = int(time.time())

# ---------------------------------------------------------------------------
# Poker game parameters
# ---------------------------------------------------------------------------
STACK_SIZE = 200.0
SMALL_BLIND = 1.0
BIG_BLIND = 2.0

# Number of seated players (use n >= 2). The engine rotates seats hand to hand.
# `numPlayer` is kept as an alias for backwards compatibility with prior configs.
NUM_PLAYERS = 6
numPlayer = NUM_PLAYERS

NUM_ACTIONS = 8

# ---------------------------------------------------------------------------
# Deep CFR training parameters (Phase 1 - bootstrap)
# ---------------------------------------------------------------------------
NUM_ITERATIONS = 20000
TRAVERSALS_PER_ITER = 10
STRAT_SAMPLES_PER_ITER = 50

ADV_BUFFER_CAPACITY = 500_000
STRAT_BUFFER_CAPACITY = 200_000
ADV_BUFFER_BALANCE_GAP = 20_000

BATCH_SIZE = 128
ADV_LR = 1e-3
POLICY_LR = 1e-3
ADV_UPDATES_PER_ITER = 4
POLICY_UPDATES_PER_ITER = 2

# later phases in training (uncomment to continue training with more capacity)
# Phase 2: heavier traversal + more strategy samples
# NUM_ITERATIONS = 30000
# TRAVERSALS_PER_ITER = 20
# STRAT_SAMPLES_PER_ITER = 100
# BATCH_SIZE = 256
# ADV_UPDATES_PER_ITER = 6
# POLICY_UPDATES_PER_ITER = 4

# Phase 3: stabilize with lower LR, more samples
# NUM_ITERATIONS = 40000
# TRAVERSALS_PER_ITER = 30
# STRAT_SAMPLES_PER_ITER = 150
# BATCH_SIZE = 256
# ADV_LR = 5e-4
# POLICY_LR = 5e-4
# ADV_UPDATES_PER_ITER = 6
# POLICY_UPDATES_PER_ITER = 4

# ---------------------------------------------------------------------------
# Evaluation / reporting
# ---------------------------------------------------------------------------
RANDOM_MATCH_INTERVAL = 5
RANDOM_MATCH_HANDS = 500
PRETRAIN_RANDOM_EVAL = True
PRETRAIN_RANDOM_EVAL_HANDS = 1000

# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
RESUME_FROM_LAST = False
CHECKPOINT_PATH = "models"
AUTO_RESUME_ON_START = False

# ---------------------------------------------------------------------------
# Logging & device
# ---------------------------------------------------------------------------
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
