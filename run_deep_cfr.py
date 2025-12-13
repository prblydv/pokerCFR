# ---------------------------------------------------------------------------
# File overview:
#   run_deep_cfr.py is the CLI entry for training the Deep CFR agent.
#   Run via `python run_deep_cfr.py` to start training, logging, plotting, etc.
# ---------------------------------------------------------------------------
import logging
import os
import signal
import sys
import random

import numpy as np
import torch
import matplotlib.pyplot as plt

from config import (
    LOG_LEVEL,
    LOG_FORMAT,
    NUM_ITERATIONS,
    TRAVERSALS_PER_ITER,
    STRAT_SAMPLES_PER_ITER,
    AUTO_RESUME_ON_START,
    CHECKPOINT_PATH,
    DETERMINISTIC_SEED,
    PRETRAIN_RANDOM_EVAL,
    PRETRAIN_RANDOM_EVAL_HANDS,
)
from poker_env import SimpleHoldemEnv
from abstraction import encode_state
from deep_cfr_trainer import DeepCFRTrainer

logger = logging.getLogger("DeepCFR")

# Global reference to trainer for signal handling
_trainer = None


# Function metadata:
#   Inputs: no explicit parameters  # dtype=varies
#   Sample:
#       sample_output = setup_logging()  # dtype=Any
def setup_logging():
    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)


def set_global_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Function metadata:
#   Inputs: trainer, out_path  # dtype=varies
#   Sample:
#       sample_output = plot_training_curves(trainer=None, out_path=None)  # dtype=Any
def plot_training_curves(trainer: DeepCFRTrainer, out_path: str = "training_curves.png"):
    """Render separate subplots for advantage loss, policy loss, and eval payoffs."""
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=False)

    # Advantage loss subplot
    ax_adv = axes[0]
    if trainer.adv_losses:
        adv_x = list(range(1, len(trainer.adv_losses) + 1))
        ax_adv.plot(adv_x, trainer.adv_losses, label="Advantage loss", color="tab:blue")
    ax_adv.set_ylabel("Adv loss")
    ax_adv.legend(loc="upper right")

    # Policy loss subplot
    ax_pol = axes[1]
    if trainer.policy_losses:
        pol_x = list(range(1, len(trainer.policy_losses) + 1))
        ax_pol.plot(pol_x, trainer.policy_losses, label="Policy loss", color="tab:orange")
    ax_pol.set_ylabel("Policy loss")
    ax_pol.legend(loc="upper right")

    # Eval payoff subplot
    ax_eval = axes[2]
    if trainer.eval_payoffs:
        eval_x = list(range(1, len(trainer.eval_payoffs) + 1))
        ax_eval.plot(eval_x, trainer.eval_payoffs, label="Eval payoff P0", color="tab:green")
    ax_eval.set_ylabel("P0 payoff")
    ax_eval.set_xlabel("Iteration")
    ax_eval.legend(loc="upper right")

    fig.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
    logger.info(f"Saved training curves to {out_path}")


# Function metadata:
#   Inputs: trainer  # dtype=varies
#   Sample:
#       sample_output = demo_hand(trainer=None)  # dtype=Any
def demo_hand(trainer: DeepCFRTrainer):
    from poker_env import (
        ACTION_FOLD,
        ACTION_CHECK,
        ACTION_CALL,
        ACTION_RAISE_SMALL,
        ACTION_RAISE_MEDIUM,
        ACTION_ALL_IN,
    )

    env = trainer.env
    s = env.new_hand()
    logger.info("Playing one demo hand with policy vs policy...")

    action_names = {
        ACTION_FOLD: "FOLD",
        ACTION_CHECK: "CHECK",
        ACTION_CALL: "CALL",
        ACTION_RAISE_SMALL: "RAISE SMALL",
        ACTION_RAISE_MEDIUM: "RAISE MEDIUM",
        ACTION_ALL_IN: "ALL-IN",
    }

    while not s.terminal:
        player = s.to_act
        a = trainer.choose_action_policy(s, player)
        logger.info(
            f"Player {player} takes action {a} ({action_names.get(a, '?')}), "
            f"street={s.street}, pot={s.pot:.2f}, stacks={s.stacks}"
        )
        s = env.step(s, a)

    payoff_p0 = env.terminal_payoff(s, 0)
    logger.info(
        f"Hand finished. Winner={s.winner}, payoff P0={payoff_p0:.2f}, pot={s.pot:.2f}"
    )


# Function metadata:
#   Inputs: signum, frame  # dtype=varies
#   Sample:
#       sample_output = signal_handler(signum=None, frame=None)  # dtype=Any
def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully by saving models before exit."""
    global _trainer
    logger.warning("\n[SIGNAL] Received interrupt signal. Saving models and exiting gracefully...")
    if _trainer is not None:
        try:
            _trainer.save_models()
            plot_training_curves(_trainer)
            logger.info("Models saved successfully.")
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    sys.exit(0)


# Function metadata:
#   Inputs: no explicit parameters  # dtype=varies
#   Sample:
#       sample_output = main()  # dtype=Any
def main():
    """Main training entry point with production error handling."""
    global _trainer
    setup_logging()
    set_global_seeds(DETERMINISTIC_SEED)
    
    try:
        logger.info("="*60)
        logger.info("DEEP CFR POKER BOT - TRAINING")
        logger.info("="*60)
        
        logger.info("Initializing environment...")
        env = SimpleHoldemEnv()
        example_state = env.new_hand()
        from abstraction import encode_state as enc
        state_dim = enc(example_state, player=0).shape[0]
        logger.info(f"State dimension: {state_dim}")

        logger.info("Initializing trainer...")
        trainer = DeepCFRTrainer(env, state_dim)
        _trainer = trainer
        if "AUTO_RESUME_ON_START" in globals() and AUTO_RESUME_ON_START:
            try:
                loaded = trainer.load_models(CHECKPOINT_PATH)
                if loaded:
                    logger.info("Resumed trainer from existing checkpoints.")
            except Exception:
                logger.warning("Auto-resume failed; starting fresh.", exc_info=True)

        if PRETRAIN_RANDOM_EVAL and trainer.loaded_from_checkpoint:
            try:
                stats = trainer.eval_vs_random(num_hands=PRETRAIN_RANDOM_EVAL_HANDS)
                bot = stats["bot"]
                rnd = stats["random"]
                def fmt_af(value):
                    return "inf" if value == float("inf") else f"{value:.2f}"
                logger.info(
                    "[PreTrainEval] "
                    f"bot_win%={bot['win_pct']:.1f}, bot_VPIP%={bot['vpip_pct']:.1f}, "
                    f"bot_PFR%={bot['pfr_pct']:.1f}, bot_EV={bot['ev']:.2f}, bot_AF={fmt_af(bot['af'])}; "
                    f"random_win%={rnd['win_pct']:.1f}, random_VPIP%={rnd['vpip_pct']:.1f}, "
                    f"random_PFR%={rnd['pfr_pct']:.1f}, random_EV={rnd['ev']:.2f}, random_AF={fmt_af(rnd['af'])}"
                )
            except Exception:
                logger.warning("Pre-training random evaluation failed.", exc_info=True)
        # Register signal handler for graceful shutdown
        signal.signal(signal.SIGINT, signal_handler)
        
        logger.info("Starting Deep CFR training...")
        logger.info(f"Configuration: {NUM_ITERATIONS} iterations, "
                   f"{TRAVERSALS_PER_ITER} traversals, "
                   f"{STRAT_SAMPLES_PER_ITER} strategy samples per iteration")
        
        trainer.train(
            num_iterations=NUM_ITERATIONS,
            traversals_per_iter=TRAVERSALS_PER_ITER,
            strat_samples_per_iter=STRAT_SAMPLES_PER_ITER,
        )
        logger.info("Training complete.")
        
        logger.info("Saving final models...")
        trainer.save_models()
        
        logger.info("Generating training curves...")
        plot_training_curves(trainer)
        
        logger.info("Playing demo hand...")
        demo_hand(trainer)
        
        logger.info("="*60)
        logger.info("TRAINING FINISHED SUCCESSFULLY")
        logger.info("="*60)
        
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user. Saving checkpoints before exit...")
        if _trainer is not None:
            try:
                _trainer.save_models()
                plot_training_curves(_trainer)
                logger.info("Models and plots saved after interrupt.")
            except Exception as e:
                logger.error(f"Error while saving during interrupt: {e}", exc_info=True)
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error during training: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
