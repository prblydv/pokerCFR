import math
from typing import Dict, List, Optional

import torch

from poker_env import (
    ACTION_CALL,
    ACTION_CHECK,
    ACTION_FOLD,
    ACTION_BET_POT_25,
    ACTION_BET_POT_50,
    ACTION_BET_POT_100,
    ACTION_BET_POT_200,
    ACTION_ALL_IN,
    STREET_PREFLOP,
    STREET_FLOP,
    STREET_TURN,
    STREET_RIVER,
)

# ---------------------------------------------------------------------------
# Hyperparameters (tuned for adaptation in ~20 hands)
# ---------------------------------------------------------------------------

DECAY = 0.96  # exponential forgetting per observation
PRIOR_STRENGTH = 3.0  # pseudo-count strength
KL_MAX = 0.10
BIAS_MAX = 2.5
MAX_ALPHA = 0.35
ALPHA_SAMPLE_K = 18.0

BLUFF_ACTIONS = {ACTION_BET_POT_25, ACTION_BET_POT_50}
VALUE_ACTIONS = {ACTION_BET_POT_100, ACTION_BET_POT_200, ACTION_ALL_IN}
CALL_ACTIONS = {ACTION_CALL, ACTION_CHECK}
FOLD_ACTIONS = {ACTION_FOLD}

BET_SIZE_BUCKETS = [
    ACTION_BET_POT_25,
    ACTION_BET_POT_50,
    ACTION_BET_POT_100,
    ACTION_BET_POT_200,
    ACTION_ALL_IN,
]

ARCHETYPES = {
    "NIT": {"vpip": 0.15, "pfr": 0.08, "agg": 0.15, "f2b": 0.65, "c2b": 0.25},
    "TAG": {"vpip": 0.25, "pfr": 0.18, "agg": 0.30, "f2b": 0.45, "c2b": 0.35},
    "LAG": {"vpip": 0.40, "pfr": 0.30, "agg": 0.45, "f2b": 0.35, "c2b": 0.45},
    "MANIAC": {"vpip": 0.55, "pfr": 0.45, "agg": 0.65, "f2b": 0.25, "c2b": 0.55},
    "CALLING_STATION": {"vpip": 0.45, "pfr": 0.10, "agg": 0.10, "f2b": 0.20, "c2b": 0.65},
}

ARCHETYPE_PRIOR = {
    "NIT": 0.15,
    "TAG": 0.35,
    "LAG": 0.20,
    "MANIAC": 0.15,
    "CALLING_STATION": 0.15,
}


def _beta_mean(alpha: float, beta: float) -> float:
    denom = alpha + beta
    if denom <= 0:
        return 0.5
    return alpha / denom


def _beta_var(alpha: float, beta: float) -> float:
    denom = alpha + beta
    if denom <= 0:
        return 0.0
    return (alpha * beta) / (denom * denom * (denom + 1.0))


def _dirichlet_mean(counts: torch.Tensor) -> torch.Tensor:
    total = counts.sum()
    if total <= 0:
        return torch.full_like(counts, 1.0 / len(counts))
    return counts / total


def _kl_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    eps = 1e-9
    p = torch.clamp(p, eps, 1.0)
    q = torch.clamp(q, eps, 1.0)
    return float((p * (p.log() - q.log())).sum().item())


def _project_to_kl_ball(
    logits_base: torch.Tensor,
    logits_adj: torch.Tensor,
    kl_max: float,
    mask: torch.Tensor,
) -> torch.Tensor:
    base = torch.softmax(logits_base + mask, dim=-1)
    adj = torch.softmax(logits_adj + mask, dim=-1)
    if _kl_divergence(adj, base) <= kl_max:
        return logits_adj

    delta = logits_adj - logits_base
    lo, hi = 0.0, 1.0
    for _ in range(20):
        mid = 0.5 * (lo + hi)
        cand = logits_base + mid * delta
        cand_p = torch.softmax(cand + mask, dim=-1)
        if _kl_divergence(cand_p, base) <= kl_max:
            lo = mid
        else:
            hi = mid
    return logits_base + lo * delta


class OpponentPosterior:
    """
    Bayesian opponent model with Beta-Bernoulli traits and Dirichlet bet-size prefs.
    Uses exponential decay so it adapts quickly within ~20 hands.
    """

    def __init__(self, prior_strength: float = PRIOR_STRENGTH, decay: float = DECAY):
        self.decay = decay
        self.vpip = [1.0, 1.0]
        self.pfr = [1.0, 1.0]
        self.agg = [1.0, 1.0]
        self.f2b = [1.0, 1.0]
        self.c2b = [1.0, 1.0]
        self.bet_size = {
            STREET_PREFLOP: torch.ones(len(BET_SIZE_BUCKETS)),
            STREET_FLOP: torch.ones(len(BET_SIZE_BUCKETS)),
            STREET_TURN: torch.ones(len(BET_SIZE_BUCKETS)),
            STREET_RIVER: torch.ones(len(BET_SIZE_BUCKETS)),
        }
        self.type_posterior = dict(ARCHETYPE_PRIOR)
        self.n_obs = 0.0
        self._init_priors(prior_strength)

    def _init_priors(self, strength: float) -> None:
        for pair in (self.vpip, self.pfr, self.agg, self.f2b, self.c2b):
            pair[0] *= strength
            pair[1] *= strength
        for k in self.bet_size:
            self.bet_size[k] *= strength

    def _apply_decay(self) -> None:
        for pair in (self.vpip, self.pfr, self.agg, self.f2b, self.c2b):
            pair[0] *= self.decay
            pair[1] *= self.decay
        for k in self.bet_size:
            self.bet_size[k] *= self.decay
        for k in self.type_posterior:
            self.type_posterior[k] *= 1.0

    def update_vpip(self, obs: int) -> None:
        self._apply_decay()
        self.vpip[0] += obs
        self.vpip[1] += 1 - obs
        self._update_type_posterior("vpip", obs)
        self.n_obs += 1.0

    def update_pfr(self, obs: int) -> None:
        self._apply_decay()
        self.pfr[0] += obs
        self.pfr[1] += 1 - obs
        self._update_type_posterior("pfr", obs)
        self.n_obs += 1.0

    def update_agg(self, obs: int) -> None:
        self._apply_decay()
        self.agg[0] += obs
        self.agg[1] += 1 - obs
        self._update_type_posterior("agg", obs)
        self.n_obs += 1.0

    def update_f2b(self, obs: int) -> None:
        self._apply_decay()
        self.f2b[0] += obs
        self.f2b[1] += 1 - obs
        self._update_type_posterior("f2b", obs)
        self.n_obs += 1.0

    def update_c2b(self, obs: int) -> None:
        self._apply_decay()
        self.c2b[0] += obs
        self.c2b[1] += 1 - obs
        self._update_type_posterior("c2b", obs)
        self.n_obs += 1.0

    def update_bet_size(self, street: int, action: int) -> None:
        if action not in BET_SIZE_BUCKETS:
            return
        self._apply_decay()
        idx = BET_SIZE_BUCKETS.index(action)
        self.bet_size[street][idx] += 1.0
        self.n_obs += 1.0

    def _update_type_posterior(self, key: str, obs: int) -> None:
        for t, stats in ARCHETYPES.items():
            p = stats[key]
            like = p if obs else (1.0 - p)
            self.type_posterior[t] *= max(like, 1e-6)
        self._normalize_types()

    def _normalize_types(self) -> None:
        total = sum(self.type_posterior.values())
        if total <= 0:
            n = len(self.type_posterior)
            for k in self.type_posterior:
                self.type_posterior[k] = 1.0 / n
            return
        for k in self.type_posterior:
            self.type_posterior[k] /= total

    def mean(self) -> Dict[str, float]:
        return {
            "vpip": _beta_mean(*self.vpip),
            "pfr": _beta_mean(*self.pfr),
            "agg": _beta_mean(*self.agg),
            "f2b": _beta_mean(*self.f2b),
            "c2b": _beta_mean(*self.c2b),
        }

    def confidence(self) -> float:
        # Combine entropy and posterior variance to control alpha.
        n = len(self.type_posterior)
        ent = 0.0
        for p in self.type_posterior.values():
            if p > 0:
                ent -= p * math.log(p)
        entropy_norm = ent / math.log(n) if n > 1 else 0.0
        var = _beta_var(*self.vpip) + _beta_var(*self.pfr) + _beta_var(*self.agg)
        var = max(0.0, min(1.0, var))
        entropy_conf = max(0.0, min(1.0, 1.0 - entropy_norm))
        sample_conf = self.n_obs / (self.n_obs + ALPHA_SAMPLE_K)
        return entropy_conf * sample_conf * (1.0 - var)


class ExploitController:
    def __init__(self, num_players: int):
        self.models = {pid: OpponentPosterior() for pid in range(num_players)}

    def get_model(self, seat: int) -> OpponentPosterior:
        return self.models[seat]

    def choose_action(
        self,
        state,
        player: int,
        legal_actions: List[int],
        base_logits: torch.Tensor,
        opponent_override: Optional[int] = None,
    ) -> int:
        if not legal_actions:
            return ACTION_CHECK
        opp = opponent_override if opponent_override is not None else self._select_opponent(state, player)
        if opp is None:
            return _sample_from_logits(base_logits, legal_actions)

        model = self.models[opp]
        means = model.mean()
        alpha = min(MAX_ALPHA, model.confidence() * MAX_ALPHA)

        bias = torch.zeros_like(base_logits)
        to_call = max(0.0, state.current_bet - state.contrib[player])
        pot = max(1.0, state.pot)
        facing_bet = to_call > 0
        nit_weight = model.type_posterior.get("NIT", 0.0)
        station_weight = model.type_posterior.get("CALLING_STATION", 0.0)
        maniac_weight = model.type_posterior.get("MANIAC", 0.0)

        # Overfolding opponents: bet more, especially small/medium.
        if means["f2b"] > 0.55:
            for a in BLUFF_ACTIONS:
                bias[a] += 1.0 * means["f2b"]
        # Calling stations: value-heavy, reduce bluffs.
        if means["c2b"] > 0.55 or station_weight > 0.3:
            for a in VALUE_ACTIONS:
                bias[a] += 1.2 * max(means["c2b"], station_weight)
            for a in BLUFF_ACTIONS:
                bias[a] -= 1.0 * max(means["c2b"], station_weight)
        # Maniacs: bluff-catch more, fold less.
        if means["agg"] > 0.5 or maniac_weight > 0.3:
            for a in CALL_ACTIONS:
                bias[a] += 0.8 * max(means["agg"], maniac_weight)
            for a in FOLD_ACTIONS:
                bias[a] -= 0.8 * max(means["agg"], maniac_weight)
        # NITs: steal and overfold vs large bets.
        if nit_weight > 0.25:
            if not facing_bet:
                for a in BLUFF_ACTIONS:
                    bias[a] += 0.8 * nit_weight
            elif to_call > pot:
                for a in FOLD_ACTIONS:
                    bias[a] += 1.0 * nit_weight
                for a in CALL_ACTIONS:
                    bias[a] -= 0.8 * nit_weight

        bias = torch.clamp(bias, -BIAS_MAX, BIAS_MAX)
        logits_adj = base_logits + alpha * bias

        mask = torch.full_like(base_logits, -1e9)
        for a in legal_actions:
            mask[a] = 0.0

        logits_adj = _project_to_kl_ball(base_logits, logits_adj, KL_MAX, mask)
        return _sample_from_logits(logits_adj, legal_actions)

    def _select_opponent(self, state, player: int) -> Optional[int]:
        last_aggr = getattr(state, "last_aggressor", None)
        if last_aggr is not None and last_aggr >= 0 and last_aggr != player:
            if not state.folded[last_aggr] and state.stacks[last_aggr] > 0:
                return last_aggr
        for i in range(state.num_players):
            pid = (player + 1 + i) % state.num_players
            if pid != player and not state.folded[pid] and state.stacks[pid] > 0:
                return pid
        return None


def update_from_action(
    controller: ExploitController,
    hand_state,
    actor: int,
    action: int,
    street: int,
    facing_bet: bool,
    bet_size_bucket: Optional[int],
) -> None:
    model = controller.get_model(actor)

    if street == STREET_PREFLOP:
        vpip_obs = 1 if action in (ACTION_CALL, ACTION_BET_POT_25, ACTION_BET_POT_50,
                                   ACTION_BET_POT_100, ACTION_BET_POT_200, ACTION_ALL_IN) else 0
        pfr_obs = 1 if action in (ACTION_BET_POT_25, ACTION_BET_POT_50,
                                  ACTION_BET_POT_100, ACTION_BET_POT_200, ACTION_ALL_IN) else 0
        model.update_vpip(vpip_obs)
        model.update_pfr(pfr_obs)
    else:
        if facing_bet:
            if action == ACTION_FOLD:
                model.update_f2b(1)
            elif action == ACTION_CALL:
                model.update_c2b(1)
        aggr_obs = 1 if action in (ACTION_BET_POT_25, ACTION_BET_POT_50,
                                   ACTION_BET_POT_100, ACTION_BET_POT_200, ACTION_ALL_IN) else 0
        model.update_agg(aggr_obs)

    if bet_size_bucket is not None:
        model.update_bet_size(street, bet_size_bucket)


def _sample_from_logits(logits: torch.Tensor, legal_actions: List[int]) -> int:
    mask = torch.full_like(logits, -1e9)
    for a in legal_actions:
        mask[a] = 0.0
    probs = torch.softmax(logits + mask, dim=-1)
    action = torch.multinomial(probs, 1).item()
    if action not in legal_actions:
        action = legal_actions[0]
    return action


if __name__ == "__main__":
    # Unit-test-like examples
    ctrl = ExploitController(num_players=2)
    m = ctrl.get_model(1)
    for _ in range(5):
        m.update_vpip(1)
        m.update_pfr(0)
        m.update_agg(0)
    print("Posterior means after loose-passive samples:", m.mean())
    print("Confidence:", m.confidence())
