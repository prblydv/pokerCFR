# ---------------------------------------------------------------------------
# File overview:
#   config.py centralizes hyperparameters/constants for poker environment,
#   Deep CFR, and hardware selection. Import it; not executable standalone.
# ---------------------------------------------------------------------------
import logging

# Randomness
RNG_SEED = 4933444
DETERMINISTIC_SEED = 4933444  # set this to control global RNG behavior

# Poker game parameters
STACK_SIZE = 200.0
SMALL_BLIND = 1.0
BIG_BLIND = 2.0


NUM_ACTIONS = 6

# Deep CFR training parameters
NUM_ITERATIONS = 10000          # increase for stronger bot
TRAVERSALS_PER_ITER = 5
STRAT_SAMPLES_PER_ITER = 50

ADV_BUFFER_CAPACITY = 500_000
STRAT_BUFFER_CAPACITY = 100_000
ADV_BUFFER_BALANCE_GAP = 20_000

BATCH_SIZE = 128
ADV_LR = 1e-3
POLICY_LR = 1e-3

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
