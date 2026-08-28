# ============================================================
# train.py — Train Deep CFR (CPU) with evaluation
# ============================================================

from engine import SimpleHoldemEnv9
from cfr_trainer_cpu import DeepCFR_CPU
from networks import AdvantageNet, PolicyNet
from encode_state import encode_state_hero
from config import DEVICE, DEFAULT_ITERATIONS, DEFAULT_TRAVERSALS_PER_ITER, DEFAULT_STRAT_SAMPLES_PER_ITER, DEFAULT_EVAL_FREQ, DEFAULT_SAVE_EVERY, DEFAULT_EVAL_GAMES
from eval_match_cpu import eval_match_cpu, print_eval_stats_colored

# ------------------------------------------------------------
# SETUP
# ------------------------------------------------------------
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

env = SimpleHoldemEnv9()
dummy_state = env.new_hand()
state_dim = encode_state_hero(dummy_state, 0).shape[0]

trainer = DeepCFR_CPU(env, state_dim)
logging.info(f"Trainer created — iterations default={DEFAULT_ITERATIONS}, traversals_per_iter={DEFAULT_TRAVERSALS_PER_ITER}, strat_samples_per_iter={DEFAULT_STRAT_SAMPLES_PER_ITER}, eval_freq={DEFAULT_EVAL_FREQ}, save_every={DEFAULT_SAVE_EVERY}, eval_games={DEFAULT_EVAL_GAMES}")

# ------------------------------------------------------------
# OPTIONAL EVALUATION CONFIG
# ------------------------------------------------------------
def evaluator(policy_net):
    # You can plug this directly:
    stats = eval_match_cpu(env, policy_net, policy_net, num_games=200)
    return stats

# ------------------------------------------------------------
# TRAIN
# ------------------------------------------------------------
stats = trainer.train(
    iterations=2000,
    traversals_per_iter=30,
    strat_samples_per_iter=30,
    evaluator={"fn": evaluator, "freq": 200},
    save_every=500
)
print_eval_stats_colored(stats)
print("Training Complete.")
