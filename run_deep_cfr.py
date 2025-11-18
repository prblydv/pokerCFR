# run_deep_cfr.py
import logging
import os

import matplotlib.pyplot as plt

from config import (
    LOG_LEVEL,
    LOG_FORMAT,
    NUM_ITERATIONS,
    TRAVERSALS_PER_ITER,
    STRAT_SAMPLES_PER_ITER,
)
from poker_env import SimpleHoldemEnv
from abstraction import encode_state
from deep_cfr_trainer import DeepCFRTrainer

logger = logging.getLogger("DeepCFR")


def setup_logging():
    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)


def plot_training_curves(trainer: DeepCFRTrainer, out_path: str = "training_curves.png"):
    fig, ax1 = plt.subplots()

    ax1.set_xlabel("Iteration (x10 for eval points)")
    ax1.set_ylabel("Loss")
    iters_loss = list(range(10, 10 * len(trainer.adv_losses) + 1, 10))
    if trainer.adv_losses:
        ax1.plot(iters_loss, trainer.adv_losses, label="Advantage loss")
    if trainer.policy_losses:
        ax1.plot(iters_loss, trainer.policy_losses, label="Policy loss")
    ax1.legend(loc="upper left")

    if trainer.eval_payoffs:
        ax2 = ax1.twinx()
        ax2.set_ylabel("Avg payoff P0")
        ax2.plot(iters_loss, trainer.eval_payoffs, color="green", label="Eval payoff P0")
        ax2.legend(loc="upper right")

    fig.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
    logger.info(f"Saved training curves to {out_path}")


def demo_hand(trainer: DeepCFRTrainer):
    from poker_env import (
        ACTION_FOLD,
        ACTION_CALL,
        ACTION_RAISE_2X,
        ACTION_RAISE_2_25X,
        ACTION_RAISE_2_5X,
        ACTION_RAISE_3X,
        ACTION_RAISE_3_5X,
        ACTION_RAISE_4_5X,
        ACTION_RAISE_6X,
        ACTION_ALL_IN,
    )

    env = trainer.env
    s = env.new_hand()
    logger.info("Playing one demo hand with policy vs policy...")

    action_names = {
        ACTION_FOLD: "FOLD",
        ACTION_CALL: "CALL/CHECK",
        ACTION_RAISE_2X: "RAISE 2× POT",
        ACTION_RAISE_2_25X: "RAISE 2.25× POT",
        ACTION_RAISE_2_5X: "RAISE 2.5× POT",
        ACTION_RAISE_3X: "RAISE 3× POT",
        ACTION_RAISE_3_5X: "RAISE 3.5× POT",
        ACTION_RAISE_4_5X: "RAISE 4.5× POT",
        ACTION_RAISE_6X: "RAISE 6× POT",
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


def main():
    setup_logging()
    logger.info("Initializing environment...")
    env = SimpleHoldemEnv()
    example_state = env.new_hand()
    from abstraction import encode_state as enc
    state_dim = enc(example_state, player=0).shape[0]
    logger.info(f"State dimension: {state_dim}")

    trainer = DeepCFRTrainer(env, state_dim)
    logger.info("Starting Deep CFR training...")
    trainer.train(
        num_iterations=NUM_ITERATIONS,
        traversals_per_iter=TRAVERSALS_PER_ITER,
        strat_samples_per_iter=STRAT_SAMPLES_PER_ITER,
    )
    logger.info("Training complete.")

    trainer.save_models()
    plot_training_curves(trainer)
    demo_hand(trainer)


if __name__ == "__main__":
    main()
