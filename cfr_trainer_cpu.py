# ============================================================
# cfr_trainer_cpu.py — Mathematically Correct Deep CFR (CPU)
# ============================================================

import random
import time
import copy
import logging
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

        logging.getLogger(__name__).info("Initializing DeepCFR_CPU")
        logging.getLogger(__name__).info(f"env={env.__class__.__name__}, state_dim={state_dim}, device={DEVICE}")

        # Two regret/advantage networks (P0, P1)
        self.adv_net = [
            AdvantageNet(state_dim).to(DEVICE),
            AdvantageNet(state_dim).to(DEVICE)
        ]
        self.adv_opt = [
            optim.Adam(self.adv_net[0].parameters(), lr=ADV_LR),
            optim.Adam(self.adv_net[1].parameters(), lr=ADV_LR)
        ]

        # Policy network
        self.policy_net = PolicyNet(state_dim).to(DEVICE)
        self.policy_opt = optim.Adam(self.policy_net.parameters(), lr=POLICY_LR)

        # Replay buffers
        self.adv_buf = [
            ReservoirBuffer(ADV_BUFFER_CAP, RNG),
            ReservoirBuffer(ADV_BUFFER_CAP, RNG)
        ]
        self.strat_buf = ReservoirBuffer(STRAT_BUFFER_CAP, RNG)

        self.encoding_cache = {}

        # Snapshot for previous policy evaluation (instrumentation only)
        self._prev_policy_state = None
        logging.getLogger(__name__).info(
            f"Adv buffers: {ADV_BUFFER_CAP}, strat buffer: {STRAT_BUFFER_CAP}, adv_batch={ADV_BATCH}, strat_batch={STRAT_BATCH}"
        )

    # --------------------------------------------------------
    # REGRET MATCHING
    # --------------------------------------------------------
    @staticmethod
    def regret_matching(adv, legal):
        adv = adv.clone()
        adv[legal == 0] = 0.0
        pos = torch.clamp(adv, min=0.0)
        total = pos.sum()

        if total <= 1e-9:
            num_legal = int(legal.sum().item())
            out = torch.zeros_like(adv)
            if num_legal > 0:
                out[legal == 1] = 1.0 / num_legal
            return out
        return pos / total

    # --------------------------------------------------------
    # STATE HASHING (for caching encodings)
    # --------------------------------------------------------
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
        k = self._hash_state(s, player)
        if k in self.encoding_cache:
            return self.encoding_cache[k]

        x = encode_state_hero(s, player).to(DEVICE)
        self.encoding_cache[k] = x
        return x

    # --------------------------------------------------------
    # EXTERNAL SAMPLING TRAVERSE
    # --------------------------------------------------------
    def traverse(self, s, player, depth=0):
        if s.terminal:
            return self.env.terminal_payoff(s, player)

        if depth > MAX_DEPTH:
            return 0.0

        legal = self.env.legal_actions(s)
        if not legal:
            return 0.0

        to_act = s.to_act
        x = self.encode_cached(s, to_act)

        with torch.no_grad():
            adv_vals = self.adv_net[to_act](x.unsqueeze(0)).squeeze(0)

        legal_mask = torch.zeros_like(adv_vals, dtype=torch.float32)
        for a in legal:
            legal_mask[a] = 1.0

        probs = self.regret_matching(adv_vals, legal_mask)

        # ----------------------------------------------------
        # Player's own node → compute regret targets
        # ----------------------------------------------------
        if to_act == player:
            action_values = {}

            for a in legal:
                next_state = self.env.step(s, a)
                v = self.traverse(next_state, player, depth + 1)
                action_values[a] = v
            node_v = sum(probs[a].item() * action_values[a] for a in legal)

            # Compute regrets
            advantage = torch.zeros_like(adv_vals)
            for a in legal:
                advantage[a] = action_values[a] - node_v

            # Store advantage sample
            self.adv_buf[player].add(
                (x.cpu(),
                 advantage.cpu(),
                 legal_mask.cpu())
            )

            return node_v

        # ----------------------------------------------------
        # Opponent node → sample 1 action externally
        # ----------------------------------------------------
        else:
            # Turn masked probs into categorical distribution
            prob_list = [probs[a].item() for a in legal]
            if sum(prob_list) <= 0:
                a = random.choice(legal)
            else:
                r = random.random()
                cum = 0.0
                a = legal[-1]
                for act, p in zip(legal, prob_list):
                    cum += p
                    if r <= cum:
                        a = act
                        break

            next_state = self.env.step(s, a)
            return self.traverse(next_state, player, depth + 1)

    # --------------------------------------------------------
    # STRATEGY TRAJECTORIES FOR POLICY NETWORK
    # --------------------------------------------------------
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

            # Save (state, probs)
            self.strat_buf.add((x.cpu(), probs.cpu(), legal_mask.cpu()))

            # Sample action
            prob_list = [probs[a].item() for a in legal]
            if sum(prob_list) <= 0:
                a = random.choice(legal)
            else:
                r = random.random()
                cum = 0
                a = legal[-1]
                for act, p in zip(legal, prob_list):
                    cum += p
                    if r <= cum:
                        a = act
                        break

            s = self.env.step(s, a)

    # --------------------------------------------------------
    # TRAINING SUBROUTINES
    # --------------------------------------------------------
    def train_advantage(self, player):
        if len(self.adv_buf[player]) < ADV_BATCH:
            return None

        batch = self.adv_buf[player].sample(ADV_BATCH)
        xs, ys, mask = zip(*batch)
        xs = torch.stack(xs).to(DEVICE)
        ys = torch.stack(ys).to(DEVICE)
        mask = torch.stack(mask).to(DEVICE)

        preds = self.adv_net[player](xs)
        loss = ((preds - ys) * mask).pow(2).mean()

        self.adv_opt[player].zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.adv_net[player].parameters(), 1.0)
        self.adv_opt[player].step()

        return loss.item()

    def train_policy(self):
        if len(self.strat_buf) < STRAT_BATCH:
            return None

        batch = self.strat_buf.sample(STRAT_BATCH)
        xs, target_probs, mask = zip(*batch)
        xs = torch.stack(xs).to(DEVICE)
        target_probs = torch.stack(target_probs).to(DEVICE)
        mask = torch.stack(mask).to(DEVICE)

        # Normalize strategy inside legal mask
        target_probs = target_probs * mask
        row_sum = target_probs.sum(1, keepdim=True)
        row_sum[row_sum == 0] = 1
        target_probs = target_probs / row_sum

        logp = self.policy_net(xs)
        loss = nn.KLDivLoss(reduction="batchmean")(logp, target_probs)

        self.policy_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.policy_opt.step()

        return loss.item()
    def new_hand_for_player(self, player):
        s = self.env.new_hand()
    
        # If player != 0, swap seats
        if player == 1:
            s.hole = [s.hole[1], s.hole[0]]
            s.stacks = [s.stacks[1], s.stacks[0]]
            s.contrib = [s.contrib[1], s.contrib[0]]
            s.initial_stacks = [s.initial_stacks[1], s.initial_stacks[0]]
    
        return s

    # --------------------------------------------------------
    # TRIAN LOOP
    # --------------------------------------------------------
    def train(self, iterations, traversals_per_iter, strat_samples_per_iter, evaluator=None, save_every=None):
        stats = []

        for it in range(1, iterations + 1):
            iter_start = time.perf_counter()
            logging.getLogger(__name__).info(f"=== Iteration {it} started ===")
            logging.getLogger(__name__).info(
                f"Buffers sizes: adv0={len(self.adv_buf[0])}, adv1={len(self.adv_buf[1])}, strat={len(self.strat_buf)}"
            )

            # --- CFR Regret Traversal ---
            for p in [0, 1]:
                for _ in range(traversals_per_iter):
                    s = self.new_hand_for_player(p)
                    # s = self.env.new_hand()
                    self.traverse(s, p)
                self.train_advantage(p)

            # --- Strategy Samples for Policy Net ---
            for _ in range(strat_samples_per_iter):
                self.sample_strategy_hand()
            self.train_policy()

            # --- Optional Evaluation ---
            if evaluator and (it % 4== 0):
                eval_start = time.perf_counter()
                # 1) evaluator provided by caller — keep compatibility
                res = evaluator["fn"](self.policy_net)
                eval_elapsed = time.perf_counter() - eval_start
                logging.getLogger(__name__).info(f"Evaluator run finished in {eval_elapsed:0.3f}s  EV={res.get('ev_per_hand')}")
                stats.append(res)

                # 2) Instrumentation: compare against previous policy snapshot (if available)
                try:
                    import eval_match_cpu as _eval_mod
                    from eval_match_cpu import print_eval_stats_colored
                    if self._prev_policy_state is not None:
                        # build previous policy net from snapshot
                        prev_net = PolicyNet(self.state_dim).to(DEVICE)
                        prev_net.load_state_dict(copy.deepcopy(self._prev_policy_state))
                        prev_net.eval()
                        self.policy_net.eval()
                        compare_start = time.perf_counter()
                        compare_stats = _eval_mod.eval_match_cpu(self.env, self.policy_net, prev_net, num_games=DEFAULT_EVAL_GAMES)
                        compare_elapsed = time.perf_counter() - compare_start
                        print_eval_stats_colored(compare_stats,it)
                         
                        # keep last compare in stats for inspection
                        stats[-1]["compare_new_vs_prev"] = compare_stats
                    else:
                        # snapshot of current policy for next comparison
                        self._prev_policy_state = copy.deepcopy(self.policy_net.state_dict())
                        logging.getLogger(__name__).info("Saved snapshot of current policy for next evaluation comparison")
                except Exception:
                    # fail-safe: do nothing if eval helper isn't available
                    logging.getLogger(__name__).warning("Could not run compare eval (eval_match_cpu not available)")

            # --- Optional Checkpoint ---
            if save_every and (it % save_every == 0):
                self.save_models(f"models_iter_{it}")

            iter_elapsed = time.perf_counter() - iter_start
            logging.getLogger(__name__).info(f"=== Iteration {it} finished in {iter_elapsed:0.3f}s ===")

        return stats

    # --------------------------------------------------------
    def save_models(self, path):
        import os
        os.makedirs(path, exist_ok=True)
        torch.save(self.adv_net[0].state_dict(), f"{path}/adv0.pt")
        torch.save(self.adv_net[1].state_dict(), f"{path}/adv1.pt")
        torch.save(self.policy_net.state_dict(), f"{path}/policy.pt")

