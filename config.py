# ---------------------------------------------------------------------------
# File overview:
#   config.py centralizes hyperparameters/constants for poker environment,
#   Deep CFR, and hardware selection. Import it; not executable standalone.
# ---------------------------------------------------------------------------
import logging

# Randomness
RNG_SEED = 433343
DETERMINISTIC_SEED = 493334444  # set this to control global RNG behavior

# Poker game parameters
STACK_SIZE = 200.0
SMALL_BLIND = 1.0
BIG_BLIND = 2.0
# Number of seated players (use n >= 2). The engine rotates seats hand to hand.
# `numPlayer` is kept as an alias for backwards compatibility with prior configs.
NUM_PLAYERS = 3
numPlayer = NUM_PLAYERS


NUM_ACTIONS = 6

# Deep CFR training parameters
NUM_ITERATIONS = 10000          # increase for stronger bot
TRAVERSALS_PER_ITER = 1
STRAT_SAMPLES_PER_ITER = 50

ADV_BUFFER_CAPACITY = 500_000
STRAT_BUFFER_CAPACITY = 100_000
ADV_BUFFER_BALANCE_GAP = 20_000

BATCH_SIZE = 128
ADV_LR = 1e-3
POLICY_LR = 1e-3
ADV_UPDATES_PER_ITER = 4      # number of advantage batches per iteration
POLICY_UPDATES_PER_ITER = 2   # number of policy batches per iteration

# Evaluation / reporting
RANDOM_MATCH_INTERVAL = 5        # iterations between bot-vs-random evals during training
RANDOM_MATCH_HANDS = 500          # hands per bot-vs-random evaluation
PRETRAIN_RANDOM_EVAL = True       # run eval_vs_random before training if resuming
PRETRAIN_RANDOM_EVAL_HANDS = 1000

# Checkpointing
RESUME_FROM_LAST = False
CHECKPOINT_PATH = "models"
AUTO_RESUME_ON_START = False

# Logging & device
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
