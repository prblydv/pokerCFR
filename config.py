# config.py
import logging

# Randomness
RNG_SEED = 4933444

# Poker game parameters
STACK_SIZE = 200.0
SMALL_BLIND = 0.5
BIG_BLIND = 1.0

# Actons: fold, call, 7 raise sizes, all-in
# FOLD
# CALL
# 2× Raise (min-open)
# 2.25× Raise
# 2.5× Raise
# 3× Raise
# 3.5× Raise
# 4.5× Raise
# 6× Raise
# ALL-IN
NUM_ACTIONS = 6

# Deep CFR training parameters
NUM_ITERATIONS = 10000          # increase for stronger bot
TRAVERSALS_PER_ITER = 5
STRAT_SAMPLES_PER_ITER = 50

ADV_BUFFER_CAPACITY = 500_000
STRAT_BUFFER_CAPACITY = 100_000

BATCH_SIZE = 128
ADV_LR = 1e-3
POLICY_LR = 1e-3

# Logging & device
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"

DEVICE = "cuda"  # change to "cuda" if you have a GPU
