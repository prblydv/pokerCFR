# deep_cfr_trainer.py
import math
import random
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    RNG_SEED,
    ADV_BUFFER_CAPACITY,
    STRAT_BUFFER_CAPACITY,
    BATCH_SIZE,
    ADV_LR,
    POLICY_LR,
    DEVICE,
)
from poker_env import SimpleHoldemEnv, GameState, NUM_ACTIONS
from abstraction import encode_state
from networks import AdvantageNet, PolicyNet, move_to_device
from replay_buffer import ReservoirBuffer

import logging

logger = logging.getLogger("DeepCFR")

RNG = random.Random(RNG_SEED)


class DeepCFRTrainer:
    def __init__(self, env: SimpleHoldemEnv, state_dim: int):
        self.env = env
        self.state_dim = state_dim

        # One advantage net per player
        self.adv_nets: List[AdvantageNet] = [
            move_to_device(AdvantageNet(state_dim)),
            move_to_device(AdvantageNet(state_dim)),
        ]
        self.adv_opts = [
            optim.Adam(self.adv_nets[0].parameters(), lr=ADV_LR),
            optim.Adam(self.adv_nets[1].parameters(), lr=ADV_LR),
        ]

        # Shared policy net (average strategy)
        self.policy_net: PolicyNet = move_to_device(PolicyNet(state_dim))
        self.policy_opt = optim.Adam(self.policy_net.parameters(), lr=POLICY_LR)

        self.adv_buffers = [
            ReservoirBuffer(ADV_BUFFER_CAPACITY, RNG),
            ReservoirBuffer(ADV_BUFFER_CAPACITY, RNG),
        ]
        self.strat_buffer = ReservoirBuffer(STRAT_BUFFER_CAPACITY, RNG)

        # Track metrics
        self.adv_losses = []
        self.policy_losses = []
        self.eval_payoffs = []

    # --- Helper methods ---

    @staticmethod
    def regret_matching(advantages: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
        adv = advantages.clone()
        adv[legal_mask == 0] = 0.0
        pos = torch.clamp(adv, min=0.0)
        total = pos.sum()
        if total.item() <= 0.0:
            num_legal = legal_mask.sum().item()
            probs = torch.zeros_like(adv)
            if num_legal > 0:
                probs[legal_mask == 1] = 1.0 / num_legal
            return probs
        return pos / total

    @staticmethod
    def sample_action(probs: torch.Tensor) -> int:
        probs_np = probs.detach().cpu().numpy()
        r = RNG.random()
        cum = 0.0
        for i, p in enumerate(probs_np):
            cum += p
            if r <= cum:
                return i
        return len(probs_np) - 1

    # --- Deep CFR traversal (external sampling) ---

    def traverse(self, state: GameState, player: int) -> float:
        """
        External-sampling CFR:
        - From traverser 'player' POV.
        - Sample chance & opponent actions.
        - Enumerate traverser actions and update advantages.
        Returns utility from traverser POV.
        """
        env = self.env
        if state.terminal:
            return env.terminal_payoff(state, player)

        to_act = state.to_act
    
        # Encode state from current actor POV
        x = encode_state(state, to_act).to(DEVICE)
        with torch.no_grad():
            adv_values = self.adv_nets[to_act](x.unsqueeze(0)).squeeze(0)

        legal_actions = env.legal_actions(state)
        # If no actions are legal, treat as terminal (all-in runout or frozen round)
        if len(legal_actions) == 0:
            return self.env.terminal_payoff(state, player)

        legal_mask = torch.zeros(NUM_ACTIONS, dtype=torch.float32, device=DEVICE)
        for a in legal_actions:
            legal_mask[a] = 1.0

        probs = self.regret_matching(adv_values, legal_mask)

        # Traverser node -> enumerate actions
        if to_act == player:
            action_values = []
            for a in range(NUM_ACTIONS):
                if a not in legal_actions:
                    action_values.append(0.0)
                    continue
                next_state = env.step(state, a)
                v = self.traverse(next_state, player)
                action_values.append(v)

            node_val = sum(probs[a].item() * action_values[a] for a in range(NUM_ACTIONS))

            advantages = [
                action_values[a] - node_val if a in legal_actions else 0.0
                for a in range(NUM_ACTIONS)
            ]

            self.adv_buffers[player].add(
                (x.cpu(),
                 torch.tensor(advantages, dtype=torch.float32),
                 legal_mask.cpu())
            )

            return node_val
        else:
            # Sample opponent action
            a = self.sample_action(probs)
            if a not in legal_actions:
                a = RNG.choice(legal_actions)
            next_state = env.step(state, a)
            return self.traverse(next_state, player)

    # --- Strategy sampling ---

    def sample_strategy_trajectory(self):
        env = self.env
        s = env.new_hand()

        while not s.terminal:
            to_act = s.to_act
            x = encode_state(s, to_act).to(DEVICE)

            with torch.no_grad():
                adv_values = self.adv_nets[to_act](x.unsqueeze(0)).squeeze(0)

            legal_actions = env.legal_actions(s)

            # === FIX: EMPTY LEGAL ACTIONS ===
            if len(legal_actions) == 0:
                break

            mask = torch.zeros(NUM_ACTIONS, dtype=torch.float32, device=DEVICE)
            mask[legal_actions] = 1.0

            probs = self.regret_matching(adv_values, mask)
            a = self.sample_action(probs)
            if a not in legal_actions:
                a = RNG.choice(legal_actions)

            self.strat_buffer.add((x.cpu(), probs.cpu(), mask.cpu()))
            s = env.step(s, a)


    # --- Training steps ---

    def train_advantage_net(self, player: int):
        if len(self.adv_buffers[player]) < BATCH_SIZE:
            return None
        net = self.adv_nets[player]
        opt = self.adv_opts[player]
        mse = nn.MSELoss()

        batch = self.adv_buffers[player].sample(BATCH_SIZE)
        xs, ys, masks = zip(*batch)
        xs = torch.stack(xs).to(DEVICE)
        ys = torch.stack(ys).to(DEVICE)
        masks = torch.stack(masks).to(DEVICE)

        preds = net(xs)
        loss = mse(preds * masks, ys * masks)

        opt.zero_grad()
        loss.backward()
        opt.step()

        return loss.item()

    def train_policy_net(self):
        if len(self.strat_buffer) < BATCH_SIZE:
            return None
        ce = nn.KLDivLoss(reduction="batchmean")

        batch = self.strat_buffer.sample(BATCH_SIZE)
        xs, target_probs, masks = zip(*batch)
        xs = torch.stack(xs).to(DEVICE)
        target_probs = torch.stack(target_probs).to(DEVICE)
        masks = torch.stack(masks).to(DEVICE)

        # Mask out illegal actions in target
        target_probs = target_probs * masks
        row_sums = target_probs.sum(dim=1, keepdim=True)
        row_sums[row_sums == 0] = 1.0
        target_probs = target_probs / row_sums

        logp = self.policy_net(xs)
        # apply mask
        logp = logp * masks
        loss = ce(logp, target_probs)

        self.policy_opt.zero_grad()
        loss.backward()
        self.policy_opt.step()

        return loss.item()

    # --- Evaluation (self-play) ---

    def eval_policy(self, num_hands=200):
        env = self.env
        total = 0

        for _ in range(num_hands):
            s = env.new_hand()

            while not s.terminal:
                legal_actions = env.legal_actions(s)

                if len(legal_actions) == 0:
                    break

                x = encode_state(s, s.to_act).to(DEVICE)
                with torch.no_grad():
                    logp = self.policy_net(x.unsqueeze(0)).squeeze(0)

                # mask illegal actions
                mask = torch.full((NUM_ACTIONS,), -1e9, device=DEVICE)
                for a in legal_actions:
                    mask[a] = 0

                probs = torch.softmax(logp + mask, dim=-1)
                a = torch.multinomial(probs, 1).item()

                s = env.step(s, a)

            total += env.terminal_payoff(s, 0)

        return total / num_hands


    # --- Main training loop ---

    def train(self, num_iterations: int,
              traversals_per_iter: int,
              strat_samples_per_iter: int):
        for it in range(1, num_iterations + 1):
            adv_losses_iter = []

            # Advantage learning for each player
            for p in [0, 1]:
                for _ in range(traversals_per_iter):
                    s = self.env.new_hand()
                    self.traverse(s, p)
                loss = self.train_advantage_net(p)
                if loss is not None:
                    adv_losses_iter.append(loss)

            # Strategy sampling
            for _ in range(strat_samples_per_iter):
                self.sample_strategy_trajectory()
            policy_loss = self.train_policy_net()

            if adv_losses_iter:
                avg_adv_loss = sum(adv_losses_iter) / len(adv_losses_iter)
                self.adv_losses.append(avg_adv_loss)
            if policy_loss is not None:
                self.policy_losses.append(policy_loss)

            # Periodic evaluation/logging
            if it % 1 == 0:
                payoff = self.eval_policy(num_hands=100)
                self.eval_payoffs.append(payoff)
                logger.info(
                    f"Iter {it}: "
                    f"adv_buf0={len(self.adv_buffers[0])}, "
                    f"adv_buf1={len(self.adv_buffers[1])}, "
                    f"strat_buf={len(self.strat_buffer)}, "
                    f"adv_loss={avg_adv_loss:.4f}, "
                    f"policy_loss={policy_loss:.4f}, "
                    f"eval_payoff_p0={payoff:.3f}"
                )

    # --- Saving / playing ---

    def save_models(self, path: str = "models"):
        import os
        os.makedirs(path, exist_ok=True)
        torch.save(self.adv_nets[0].state_dict(), f"{path}/adv_p0.pt")
        torch.save(self.adv_nets[1].state_dict(), f"{path}/adv_p1.pt")
        torch.save(self.policy_net.state_dict(), f"{path}/policy.pt")
        logger.info(f"Saved models to {path}/")

    def choose_action_policy(self, state: GameState, player: int) -> int:
        from poker_env import NUM_ACTIONS  # local import

        env = self.env
        x = encode_state(state, player).to(DEVICE)
        with torch.no_grad():
            logp = self.policy_net(x.unsqueeze(0)).squeeze(0)
        probs = torch.exp(logp)
        legal_actions = env.legal_actions(state)
        mask = torch.zeros(NUM_ACTIONS, dtype=torch.float32, device=DEVICE)
        mask[legal_actions] = 1.0
        probs = probs * mask
        total = probs.sum()
        if total.item() <= 0:
            probs = mask / mask.sum()
        else:
            probs = probs / total
        a = self.sample_action(probs)
        if a not in legal_actions:
            a = random.choice(legal_actions)
        return a
