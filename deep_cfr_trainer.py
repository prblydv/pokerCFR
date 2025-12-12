# deep_cfr_trainer.py
import math
import random
import time
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
    RESUME_FROM_LAST,
    CHECKPOINT_PATH,
    RANDOM_MATCH_INTERVAL,
    RANDOM_MATCH_HANDS,
)
from poker_env import (
    SimpleHoldemEnv,
    GameState,
    NUM_ACTIONS,
    ACTION_CALL,
    ACTION_RAISE_SMALL,
    ACTION_RAISE_MEDIUM,
    ACTION_ALL_IN,
    STREET_PREFLOP,
)
from abstraction import encode_state
from networks import AdvantageNet, PolicyNet, move_to_device
from replay_buffer import ReservoirBuffer

import logging

logger = logging.getLogger("DeepCFR")

RNG = random.Random(RNG_SEED)
RAISE_ACTIONS = {ACTION_RAISE_SMALL, ACTION_RAISE_MEDIUM, ACTION_ALL_IN}


class DeepCFRTrainer:
    # Manages the entire Deep CFR workflow: holds env, networks, buffers, metrics,
    # and exposes traversal/eval/train/save/load utilities consumed by run scripts.
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
        self.iter_times = []

        if RESUME_FROM_LAST:
            try:
                loaded = self.load_models(CHECKPOINT_PATH)
                if loaded:
                    logger.info(f"Resumed trainer from checkpoint at '{CHECKPOINT_PATH}'.")
            except Exception:
                logger.warning("Resume flag set but loading checkpoint failed; continuing fresh.", exc_info=True)

    # --- Helper methods ---

    # Converts a vector of regrets/advantages into probabilities via standard
    # regret matching (positive regrets normalized; fallback uniform). Used
    # inside traverse() and sample_strategy_trajectory().
    # Inputs: advantages (Tensor[A]), legal_mask (Tensor[A] 0/1).
    # Output: Tensor[A] of probabilities (sum over legal actions = 1).
    # Example: advantages=[-0.2,0.3,1.5], legal=[1,1,1] → probs=[0,0.167,0.833].
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

    # Samples an action index according to probability tensor probs. Used wherever
    # we need randomized selections (opponent sampling, strategy trajectories).
    # Inputs: probs Tensor[A] summing to 1.
    # Output: int index (0..A-1).
    # Example: probs=[0.2,0.5,0.3] → returns 1 with 50% chance.
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

    # Core CFR recursion. Takes a GameState and traverser id, returns utility from
    # traverser perspective while pushing advantage samples to buffers. Uses
    # external sampling (opponent actions sampled, traverser actions enumerated).
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

            if len(self.adv_buffers[player]) < ADV_BUFFER_CAPACITY:
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

    # Generates one on-policy trajectory using regret-matched policies to collect
    # (state, probs, mask) tuples for policy training.
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

            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1 why do we need this guard???????????????????/!!!!!!!!!!!
            if a not in legal_actions:
                a = RNG.choice(legal_actions)

            self.strat_buffer.add((x.cpu(), probs.cpu(), mask.cpu()))
            s = env.step(s, a)


    # --- Training steps ---

    # Trains the specified player's advantage network using reservoir samples.
    # Returns loss (float) or None if insufficient data.
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

    # Trains shared policy network by minimizing KL divergence against sampled
    # strategy distributions. Returns loss or None.
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

    # Runs num_hands games of self-play to estimate policy payoff (player 0).
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

    def _mirror_state(self, state: GameState) -> GameState:
        mirrored = GameState(
            deck=state.deck[:],
            board=state.board[:],
            hole=[state.hole[1][:], state.hole[0][:]],
            pot=state.pot,
            to_act=1 - state.to_act,
            street=state.street,
            stacks=[state.stacks[1], state.stacks[0]],
            current_bet=state.current_bet,
            last_aggressor=(
                1 - state.last_aggressor if state.last_aggressor in (0, 1) else state.last_aggressor
            ),
            sb_player=1 - state.sb_player,
            bb_player=1 - state.bb_player,
            initial_stacks=[state.initial_stacks[1], state.initial_stacks[0]],
            contrib=[state.contrib[1], state.contrib[0]],
            actions_this_street=state.actions_this_street,
            terminal=state.terminal,
            winner=(
                1 - state.winner if state.terminal and state.winner in (0, 1) else state.winner
            ),
        )
        return mirrored

    def _sample_policy_action(self, state: GameState, player: int, legal_actions: List[int]) -> int:
        x = encode_state(state, player).to(DEVICE)
        with torch.no_grad():
            logp = self.policy_net(x.unsqueeze(0)).squeeze(0)

        mask = torch.full((NUM_ACTIONS,), -1e9, device=logp.device)
        for a in legal_actions:
            mask[a] = 0.0

        probs = torch.softmax(logp + mask, dim=-1)
        action = torch.multinomial(probs, 1).item()
        if action not in legal_actions:
            action = RNG.choice(legal_actions)
        return action

    def eval_vs_random(self, num_hands: int = RANDOM_MATCH_HANDS):
        env = self.env
        stats = {
            0: {"hands": 0, "wins": 0, "vpip": 0, "pfr": 0, "aggr": 0, "calls": 0, "actions": 0, "payoff": 0.0},
            1: {"hands": 0, "wins": 0, "vpip": 0, "pfr": 0, "aggr": 0, "calls": 0, "actions": 0, "payoff": 0.0},
        }

        for _ in range(num_hands):
            s = env.new_hand()
            hand_flags = {
                0: {"vpip": False, "pfr": False, "aggr": 0, "calls": 0, "actions": 0},
                1: {"vpip": False, "pfr": False, "aggr": 0, "calls": 0, "actions": 0},
            }

            while not s.terminal:
                legal = env.legal_actions(s)
                if not legal:
                    break
                player = s.to_act
                if player == 0:
                    action = RNG.choice(legal)
                else:
                    action = self._sample_policy_action(s, player, legal)

                info = hand_flags[player]
                info["actions"] += 1

                if s.street == STREET_PREFLOP:
                    if action == ACTION_CALL or action in RAISE_ACTIONS:
                        info["vpip"] = True
                    if action in RAISE_ACTIONS:
                        info["pfr"] = True

                if action in RAISE_ACTIONS:
                    info["aggr"] += 1
                elif action == ACTION_CALL:
                    info["calls"] += 1

                s = env.step(s, action)

            winner = s.winner
            for player in (0, 1):
                stats[player]["hands"] += 1
                if winner == player:
                    stats[player]["wins"] += 1
                if hand_flags[player]["vpip"]:
                    stats[player]["vpip"] += 1
                if hand_flags[player]["pfr"]:
                    stats[player]["pfr"] += 1
                stats[player]["aggr"] += hand_flags[player]["aggr"]
                stats[player]["calls"] += hand_flags[player]["calls"]
                stats[player]["actions"] += hand_flags[player]["actions"]
            stats[0]["payoff"] += env.terminal_payoff(s, 0)
            stats[1]["payoff"] += env.terminal_payoff(s, 1)

        def summarize(player_stats):
            hands = max(player_stats["hands"], 1)
            win_pct = 100.0 * player_stats["wins"] / hands
            vpip_pct = 100.0 * player_stats["vpip"] / hands
            pfr_pct = 100.0 * player_stats["pfr"] / hands
            calls = player_stats["calls"]
            aggr = player_stats["aggr"]
            if calls == 0:
                af = float("inf") if aggr > 0 else 0.0
            else:
                af = aggr / calls
            avg_actions = player_stats["actions"] / hands
            ev = player_stats["payoff"] / hands
            return {
                "win_pct": win_pct,
                "vpip_pct": vpip_pct,
                "pfr_pct": pfr_pct,
                "af": af,
                "avg_actions": avg_actions,
                "hands": hands,
                "ev": ev,
            }

        return {
            "random": summarize(stats[0]),
            "bot": summarize(stats[1]),
        }

    ...


    # --- Main training loop ---

    # Main training loop: for each iteration perform traversals, update advantage
    # nets, sample strategy trajectories, train policy, evaluate/log.
    def train(self, num_iterations: int,
              traversals_per_iter: int,
              strat_samples_per_iter: int):
        for it in range(1, num_iterations + 1):
            iter_start_time = time.time()
            adv_losses_iter = []

            # Advantage learning for each player
            for p in [0, 1]:
                for _ in range(traversals_per_iter):
                    s = self.env.new_hand()
                    self.traverse(s, p)
                    mirror = self._mirror_state(s)
                    self.traverse(mirror, p)
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
            else:
                avg_adv_loss = None
            if policy_loss is not None:
                self.policy_losses.append(policy_loss)

            # Periodic evaluation/logging
            if it % 1 == 0:
                payoff = self.eval_policy(num_hands=100)
                self.eval_payoffs.append(payoff)
                iter_time = time.time() - iter_start_time
                self.iter_times.append(iter_time)
                adv_loss_str = f"{avg_adv_loss:.4f}" if avg_adv_loss is not None else "n/a"
                policy_loss_str = f"{policy_loss:.4f}" if policy_loss is not None else "n/a"
                logger.info(
                    f"Iter {it}: "
                    f"adv_buf0={len(self.adv_buffers[0])}, "
                    f"adv_buf1={len(self.adv_buffers[1])}, "
                    f"strat_buf={len(self.strat_buffer)}, "
                    f"adv_loss={adv_loss_str}, "
                    f"policy_loss={policy_loss_str}, "
                    f"eval_payoff_p0={payoff:.3f}, "
                    f"time={iter_time:.2f}s"
                )

            if RANDOM_MATCH_INTERVAL > 0 and it % RANDOM_MATCH_INTERVAL == 0:
                match_stats = self.eval_vs_random(num_hands=RANDOM_MATCH_HANDS)
                bot = match_stats["bot"]
                rnd = match_stats["random"]
                def fmt_af(value):
                    return "inf" if value == float("inf") else f"{value:.2f}"
                logger.info(
                    f"[RandomMatch] iter {it}: "
                    f"bot_win%={bot['win_pct']:.1f}, bot_VPIP%={bot['vpip_pct']:.1f}, "
                    f"bot_PFR%={bot['pfr_pct']:.1f}, bot_EV={bot['ev']:.2f}, bot_AF={fmt_af(bot['af'])}; "
                    f"random_win%={rnd['win_pct']:.1f}, random_VPIP%={rnd['vpip_pct']:.1f}, "
                    f"random_PFR%={rnd['pfr_pct']:.1f}, random_EV={rnd['ev']:.2f}, random_AF={fmt_af(rnd['af'])}"
                )

    # --- Saving / playing ---

    # Serializes advantage/policy networks to disk (used after training or on
    # signal handling). Files: adv_p0.pt, adv_p1.pt, policy.pt.
    def save_models(self, path: str = "models"):
        import os
        os.makedirs(path, exist_ok=True)
        torch.save(self.adv_nets[0].state_dict(), f"{path}/adv_p0.pt")
        torch.save(self.adv_nets[1].state_dict(), f"{path}/adv_p1.pt")
        torch.save(self.policy_net.state_dict(), f"{path}/policy.pt")
        logger.info(f"Saved models to {path}/")

    # Attempts to load saved weights for both advantage nets and the policy net.
    # Returns True on success, False otherwise; used for checkpoint resume.
    def load_models(self, path: str = "models") -> bool:
        """Attempt to load model checkpoints from `path`.

        Returns True if all models were found and loaded, False otherwise.
        """
        import os
        if not os.path.isdir(path):
            return False

        try:
            p0 = os.path.join(path, "adv_p0.pt")
            p1 = os.path.join(path, "adv_p1.pt")
            pol = os.path.join(path, "policy.pt")

            if os.path.exists(p0) and os.path.exists(p1) and os.path.exists(pol):
                self.adv_nets[0].load_state_dict(torch.load(p0, map_location=DEVICE))
                self.adv_nets[1].load_state_dict(torch.load(p1, map_location=DEVICE))
                self.policy_net.load_state_dict(torch.load(pol, map_location=DEVICE))

                self.adv_nets[0].eval()
                self.adv_nets[1].eval()
                self.policy_net.eval()

                logger.info(f"Loaded models from {path}/")
                return True
            logger.info(f"No complete checkpoint found in {path}/; starting from scratch")
        except Exception as e:
            logger.error(f"Error loading models from {path}: {e}", exc_info=True)

        return False

    # import torch
    # import random

    # Utility for inference: sample an action from the trained policy network for
    # a given state/player (used by demo hand and deployments).
    def choose_action_policy(self, state: GameState, player: int) -> int:
        from poker_env import NUM_ACTIONS

        env = self.env
        x = encode_state(state, player).to(DEVICE)

        with torch.no_grad():
            logp = self.policy_net(x.unsqueeze(0)).squeeze(0)

        probs = torch.exp(logp)  # If log-softmax output, else use torch.softmax(logp, -1)

        legal_actions = env.legal_actions(state)
        mask = torch.zeros(NUM_ACTIONS, dtype=torch.float32, device=DEVICE)
        mask[legal_actions] = 1.0

        probs = probs * mask
        total = probs.sum()

        if total.item() <= 0:
            # Uniform random over legal actions
            probs = mask / mask.sum()
        else:
            probs = probs / total  # Renormalize

        # MIXED STRATEGY: sample from the probability distribution
        a = torch.multinomial(probs, 1).item()

        if a not in legal_actions:
            a = random.choice(legal_actions)

        return a


