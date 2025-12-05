# ============================================================
# config.py — Deep CFR CPU settings
# ============================================================

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RNG_SEED = 42

# Replay buffers
ADV_BUFFER_CAP = 200000
STRAT_BUFFER_CAP = 200000

ADV_BATCH = 2048
STRAT_BATCH = 2048

ADV_LR = 1e-4
POLICY_LR = 1e-4

MAX_DEPTH = 200  # Failsafe recursive cutoff

NUM_ACTIONS = 9

# ------------------------------------------------------------
# Defaults for training / evaluation (consolidated so you don't
# need to search for numbers in the codebase)
# ------------------------------------------------------------
DEFAULT_ITERATIONS = 2000
DEFAULT_TRAVERSALS_PER_ITER = 30
DEFAULT_STRAT_SAMPLES_PER_ITER = 30
DEFAULT_EVAL_FREQ = 200
DEFAULT_SAVE_EVERY = 500
DEFAULT_EVAL_GAMES = 200