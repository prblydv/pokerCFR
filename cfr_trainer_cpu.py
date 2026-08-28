# ============================================================
# cfr_trainer_cpu.py — OOM-SAFE, Mathematically Correct Deep CFR
# ============================================================

import random
import time
import copy
import logging
import gc
import torch
import torch.nn as nn
import torch.optim as optim

from networks import AdvantageNet, PolicyNet
from encode_state import encode_state_hero
from replay_buffer import ReservoirBuffer
from config import (
    DEVICE,
    RNG_SEED,
    ADV_BUFFER_CAP,
    STRAT_BUFFER_CAP,
    ADV_BATCH,
    STRAT_BATCH,
    ADV_LR,
    POLICY_LR,
    MAX_DEPTH,
    DEFAULT_EVAL_GAMES,
)

RNG = random.Random(RNG_SEED)


class DeepCFR_CPU:
    def __init__(self, env, state_dim):
        self.env = env
        self.state_dim = state_dim

        self.encoding_cache = {}

        # Advantage nets (reset every iteration)
        self.adv_net = None
        self.adv_opt = None
        self._reset_advantage_networks()

        # Persistent policy net (NEVER reset)
        self.policy_net = PolicyNet(state_dim).to(DEVICE)
        self.policy_opt = optim.Adam(self.policy_net.parameters(), lr=POLICY_LR)

        # Reservoir buffers (support LCFR weights)
        self.adv_buf = [
            ReservoirBuffer(ADV_BUFFER_CAP, RNG),
            ReservoirBuffer(ADV_BUFFER_CAP, RNG)
        ]
        self.strat_buf = ReservoirBuffer(STRAT_BUFFER_CAP, RNG)

        self._prev_policy_state = None

    # ============================================================
    #  SAFE RESET OF ADV NETWORKS  (avoid OOM)
    # ============================================================
    def _reset_advantage_networks(self):
        # ---- Free old nets & optimizers ----
        if self.adv_net is not None:
            del self.adv_net
        if self.adv_opt is not None:
            del self.adv_opt

        gc.collect()                    # free Python objects
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()    # free cached CUDA memory

        # ---- Re-create fresh nets ----
        self.adv_net = [
            AdvantageNet(self.state_dim).to(DEVICE),
            AdvantageNet(self.state_dim).to(DEVICE)
        ]
        self.adv_opt = [
            optim.Adam(self.adv_net[0].parameters(), lr=ADV_LR, weight_decay=1e-4),
            optim.Adam(self.adv_net[1].parameters(), lr=ADV_LR, weight_decay=1e-4)
        ]

    # ============================================================
    #  REGRET MATCHING  (Exact Deep CFR)
    # ============================================================
    @staticmethod
    def regret_matching(adv, legal):
        adv = adv.clone()
        adv[legal == 0] = 0.0

        pos = torch.clamp(adv, min=0.0)
        total = pos.sum()

        if total <= 1e-12:
            masked = adv.clone()
            masked[legal == 0] = -1e12
            best = torch.argmax(masked)
            out = torch.zeros_like(adv)
            out[best] = 1.0
            return out

        return pos / total

    # ============================================================
    #  STATE ENCODING CACHE (bounded)
    # ============================================================
    def _hash_state(self, s, player):
        return (
            tuple(s.board),
            tuple(tuple(h) for h in s.hole),
            tuple(s.stacks),
            s.pot,
            s.to_act,
            s.street,
            s.current_bet,
            s.last_aggressor,
            player,
        )

    def encode_cached(self, s, player):
        # Limit cache size → prevent OOM
        if len(self.encoding_cache) > 50000:
            self.encoding_cache.clear()
            gc.collect()

        k = self._hash_state(s, player)
        if k in self.encoding_cache:
            return self.encoding_cache[k]

        x = encode_state_hero(s, player).to(DEVICE)
        self.encoding_cache[k] = x
        return x

    # ============================================================
    #  EXTERNAL SAMPLING CFR TRAVERSE
    # ============================================================
    def traverse(self, s, player, depth=0):
        if s.terminal:
            return self.env.terminal_payoff(s, player)

        if depth > MAX_DEPTH:
            return 0.0

        legal = self.env.legal_actions(s)
        if not legal:
            return 0.0

        p = s.to_act
        x = self.encode_cached(s, p)

        with torch.no_grad():
            adv_vals = self.adv_net[p](x.unsqueeze(0)).squeeze(0)

        legal_mask = torch.zeros_like(adv_vals)
        for a in legal:
            legal_mask[a] = 1.0

        probs = self.regret_matching(adv_vals, legal_mask)

        # -------------------------------------------------------
        # Traverser node → compute regret samples
        # -------------------------------------------------------
        if p == player:
            action_values = {}
            for a in legal:
                nxt = self.env.step(s, a)
                action_values[a] = float(self.traverse(nxt, player, depth + 1))

            node_v = sum(probs[a].item() * action_values[a] for a in legal)

            advantage = torch.zeros_like(adv_vals)
            for a in legal:
                advantage[a] = action_values[a] - node_v

            weight = float(self.current_iteration)

            self.adv_buf[player].add(
                (x.cpu(), advantage.cpu(), legal_mask.cpu(), weight)
            )

            return node_v

        # -------------------------------------------------------
        # Opponent node → sample one branch
        # -------------------------------------------------------
        prob_list = [probs[a].item() for a in legal]

        if sum(prob_list) <= 0:
            a = random.choice(legal)
        else:
            r = random.random()
            cum = 0.0
            a = legal[-1]
            for act, pval in zip(legal, prob_list):
                cum += pval
                if r <= cum:
                    a = act
                    break

        nxt = self.env.step(s, a)
        return self.traverse(nxt, player, depth + 1)

    # ============================================================
    #  STRATEGY SAMPLING
    # ============================================================
    def sample_strategy_hand(self):
        seat = RNG.randint(0, 1)
        s = self.new_hand_for_player(seat)

        while not s.terminal:
            p = s.to_act
            x = self.encode_cached(s, p)

            with torch.no_grad():
                adv_vals = self.adv_net[p](x.unsqueeze(0)).squeeze(0)

            legal = self.env.legal_actions(s)
            legal_mask = torch.zeros_like(adv_vals)
            for a in legal:
                legal_mask[a] = 1.0

            probs = self.regret_matching(adv_vals, legal_mask)

            weight = float(self.current_iteration)
            self.strat_buf.add((x.cpu(), probs.cpu(), legal_mask.cpu(), weight))

            # sample action
            prob_list = [probs[a].item() for a in legal]
            if sum(prob_list) <= 0:
                a = random.choice(legal)
            else:
                r = random.random()
                cum = 0.0
                a = legal[-1]
                for act, pval in zip(legal, prob_list):
                    cum += pval
                    if r <= cum:
                        a = act
                        break

            s = self.env.step(s, a)

    # ============================================================
    #  ADVANTAGE TRAINING (LCFR)
    # ============================================================
    def train_advantage(self, player):
        if len(self.adv_buf[player]) < ADV_BATCH:
            return None

        batch = self.adv_buf[player].sample(ADV_BATCH)
        xs, ys, mask, weight = zip(*batch)

        xs = torch.stack(xs).to(DEVICE)
        ys = torch.stack(ys).to(DEVICE)
        mask = torch.stack(mask).to(DEVICE)
        weight = torch.tensor(weight, dtype=torch.float32, device=DEVICE)

        preds = self.adv_net[player](xs)

        per_sample = ((preds - ys) * mask).pow(2).sum(1)
        loss = (per_sample * weight).sum() / weight.sum()

        self.adv_opt[player].zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.adv_net[player].parameters(), 1.0)
        self.adv_opt[player].step()

        return loss.item()

    # ============================================================
    #  POLICY TRAINING (MSE on logits)
    # ============================================================
    def train_policy(self):
        if len(self.strat_buf) < STRAT_BATCH:
            return None

        batch = self.strat_buf.sample(STRAT_BATCH)
        xs, target_probs, mask, weight = zip(*batch)

        xs = torch.stack(xs).to(DEVICE)
        target_probs = torch.stack(target_probs).to(DEVICE)
        mask = torch.stack(mask).to(DEVICE)
        weight = torch.tensor(weight, dtype=torch.float32, device=DEVICE)

        # Normalize
        target_probs = target_probs * mask
        target_probs = target_probs / target_probs.sum(1, keepdim=True).clamp(min=1e-12)

        logits = self.policy_net(xs)
        logits = logits * mask

        per_sample = ((logits - target_probs)**2).sum(1)
        loss = (per_sample * weight).sum() / weight.sum()

        self.policy_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.policy_opt.step()

        return loss.item()

    # ============================================================
    #  SEAT SWAP LOGIC
    # ============================================================
    def new_hand_for_player(self, player):
        s = self.env.new_hand()
        if player == 1:
            s.hole = [s.hole[1], s.hole[0]]
            s.stacks = [s.stacks[1], s.stacks[0]]
            s.contrib = [s.contrib[1], s.contrib[0]]
            s.initial_stacks = [s.initial_stacks[1], s.initial_stacks[0]]
            s.sb_player, s.bb_player = s.bb_player, s.sb_player
            s.to_act = 1 - s.to_act
        return s

    # ============================================================
    #  TRAIN LOOP (OOM SAFE)
    # ============================================================
    def train(self, iterations, traversals_per_iter, strat_samples_per_iter,
              evaluator=None, save_every=None):

        stats = []

        for it in range(1, iterations + 1):
            self.current_iteration = it

            # Fresh nets every iter, old memory freed
            self._reset_advantage_networks()

            # clear encoding cache
            self.encoding_cache.clear()
            gc.collect()

            iter_start = time.perf_counter()
            logging.getLogger(__name__).info(f"=== Iteration {it} ===")

            # (1) Advantage building
            for p in [0, 1]:
                for _ in range(traversals_per_iter):
                    s = self.new_hand_for_player(p)
                    self.traverse(s, p)
                self.train_advantage(p)

            # (2) Strategy + policy training
            for _ in range(strat_samples_per_iter):
                self.sample_strategy_hand()
            self.train_policy()

            # (3) Evaluation
            if evaluator and (it % 4 == 0):
                eval_res = evaluator["fn"](self.policy_net)
                logging.getLogger(__name__).info(f"EV = {eval_res.get('ev_per_hand')}")
                stats.append(eval_res)

                # previous policy comparison
                try:
                    import eval_match_cpu as M
                    from eval_match_cpu import print_eval_stats_colored

                    if self._prev_policy_state is not None:
                        prev = PolicyNet(self.state_dim).to(DEVICE)
                        prev.load_state_dict(copy.deepcopy(self._prev_policy_state))
                        prev.eval()

                        self.policy_net.eval()

                        cmp_stats = M.eval_match_cpu(
                            self.env, self.policy_net, prev, num_games=DEFAULT_EVAL_GAMES
                        )
                        print_eval_stats_colored(cmp_stats, it)
                        stats[-1]["compare"] = cmp_stats
                    else:
                        self._prev_policy_state = copy.deepcopy(self.policy_net.state_dict())

                except Exception:
                    pass

            # (4) Save
            if save_every and (it % save_every == 0):
                self.save_models(f"models_iter_{it}")

            logging.getLogger(__name__).info(
                f"Iteration finished in {time.perf_counter() - iter_start:.3f}s"
            )

        return stats

    # ============================================================
    def save_models(self, path):
        import os
        os.makedirs(path, exist_ok=True)
        torch.save(self.adv_net[0].state_dict(), f"{path}/adv0.pt")
        torch.save(self.adv_net[1].state_dict(), f"{path}/adv1.pt")
        torch.save(self.policy_net.state_dict(), f"{path}/policy.pt")
