import itertools
import os
import random

import torch

import opponent_model as om
from team_3v3_eval import run_match


def _set_params(params: dict) -> None:
    om.DECAY = params["DECAY"]
    om.PRIOR_STRENGTH = params["PRIOR_STRENGTH"]
    om.KL_MAX = params["KL_MAX"]
    om.BIAS_MAX = params["BIAS_MAX"]
    om.MAX_ALPHA = params["MAX_ALPHA"]
    om.ALPHA_SAMPLE_K = params["ALPHA_SAMPLE_K"]


def _score(policy_ev_a: float, policy_ev_b: float) -> float:
    # Weight Match B higher since it's the harder opponent.
    return 0.3 * policy_ev_a + 0.7 * policy_ev_b


def main() -> None:
    policy_path = "models/policyev_42.pt"
    hands = 5000
    num_players = 6
    scripted_team_seats = [0, 2, 4]

    if not os.path.isfile(policy_path):
        raise FileNotFoundError(f"Policy file not found: {policy_path}")

    param_space = {
        "DECAY": [0.96, 0.97, 0.985],
        "PRIOR_STRENGTH": [3.0, 4.0, 6.0],
        "KL_MAX": [0.10, 0.20, 0.30],
        "BIAS_MAX": [2.5, 3.5, 4.5],
        "MAX_ALPHA": [0.25, 0.35, 0.45],
        "ALPHA_SAMPLE_K": [6.0, 12.0, 18.0],
    }

    trials = 20
    rng = random.Random(1337)

    keys = list(param_space.keys())
    all_candidates = list(itertools.product(*(param_space[k] for k in keys)))
    rng.shuffle(all_candidates)
    candidates = all_candidates[:trials]

    best = None
    best_score = None

    for idx, values in enumerate(candidates, start=1):
        params = dict(zip(keys, values))
        _set_params(params)

        _, team_a = run_match(
            policy_path=policy_path,
            num_hands=hands,
            num_players=num_players,
            scripted_eval_seats=scripted_team_seats,
            scripted_tag_seats=[],
            use_opponent_modeling=True,
        )
        _, team_b = run_match(
            policy_path=policy_path,
            num_hands=hands,
            num_players=num_players,
            scripted_eval_seats=[],
            scripted_tag_seats=scripted_team_seats,
            use_opponent_modeling=True,
        )

        ev_a = team_a["policy_team"]["avg_ev"]
        ev_b = team_b["policy_team"]["avg_ev"]
        score = _score(ev_a, ev_b)

        print(
            f"[{idx}/{len(candidates)}] score={score:.3f} "
            f"ev_a={ev_a:.3f} ev_b={ev_b:.3f} params={params}"
        )

        if best_score is None or score > best_score:
            best_score = score
            best = params

    print("\nBest params:")
    print(best)
    print(f"Best score: {best_score:.3f}")


if __name__ == "__main__":
    main()
