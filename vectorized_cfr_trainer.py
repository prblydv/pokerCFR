import torch
import time
import logging
import random
from typing import List, Tuple
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import torch.nn.functional as F

from config import *
from poker_env import SimpleHoldemEnv, GameState, NUM_ACTIONS
from abstraction import encode_state
from networks import AdvantageNet, PolicyNet, move_to_device
from replay_buffer import ReservoirBuffer

logger = logging.getLogger("VectorizedDeepCFR")
RNG = random.Random(RNG_SEED)

class VectorizedDeepCFRTrainer:
    def __init__(self, env: SimpleHoldemEnv, state_dim: int):
        self.env = env
        self.state_dim = state_dim
        
        self.adv_nets = [
            move_to_device(AdvantageNet(state_dim)),
            move_to_device(AdvantageNet(state_dim)),
        ]
        self.adv_opts = [
            torch.optim.Adam(self.adv_nets[0].parameters(), lr=ADV_LR),
            torch.optim.Adam(self.adv_nets[1].parameters(), lr=ADV_LR),
        ]
        
        self.policy_net = move_to_device(PolicyNet(state_dim))
        self.policy_opt = torch.optim.Adam(self.policy_net.parameters(), lr=POLICY_LR)
        
        self.adv_buffers = [
            ReservoirBuffer(ADV_BUFFER_CAPACITY, RNG),
            ReservoirBuffer(ADV_BUFFER_CAPACITY, RNG),
        ]
        self.strat_buffer = ReservoirBuffer(STRAT_BUFFER_CAPACITY, RNG)
        
        self.adv_losses = []
        self.policy_losses = []
        self.eval_payoffs = []
        
        self._encoding_cache = {}
        self._thread_pool = ThreadPoolExecutor(max_workers=8)

    def _hash_state(self, state: GameState, player: int) -> Tuple:
        return (
            tuple(state.board),
            tuple(tuple(h) for h in state.hole),
            tuple(state.stacks),
            state.pot,
            state.to_act,
            state.street,
            state.current_bet,
            state.last_aggressor,
            player,
        )

    def _get_terminal_value(self, state: GameState, player: int) -> float:
        if state.terminal:
            return self.env.terminal_payoff(state, player)
        else:
            temp_state = self.env.step(state, 0)
            if temp_state.terminal:
                return self.env.terminal_payoff(temp_state, player)
            return 0.0

    def batch_encode_states(self, states: List[GameState]) -> torch.Tensor:
        encodings = []
        for state in states:
            state_key = self._hash_state(state, state.to_act)
            if state_key in self._encoding_cache:
                encodings.append(self._encoding_cache[state_key])
            else:
                encoding = encode_state(state, state.to_act).to(DEVICE)
                self._encoding_cache[state_key] = encoding
                encodings.append(encoding)
        return torch.stack(encodings)

    # def vectorized_traverse_batch(self, root_states: List[GameState], player: int) -> List[float]:
    #     if not root_states:
    #         return []
        
    #     stack = deque()
    #     for state in root_states:
    #         stack.append((state, []))
        
    #     state_values = {}
        
    #     while stack:
    #         batch_states = []
    #         batch_metadatas = []
            
    #         while stack and len(batch_states) < 64:
    #             state, metadata = stack.pop()
    #             batch_states.append(state)
    #             batch_metadatas.append(metadata)
            
    #         if not batch_states:
    #             break
                
    #         batch_encodings = self.batch_encode_states(batch_states)
            
    #         with torch.no_grad():
    #             player_groups = {}
    #             for i, state in enumerate(batch_states):
    #                 p = state.to_act
    #                 if p not in player_groups:
    #                     player_groups[p] = []
    #                 player_groups[p].append(i)
                
    #             batch_advantages = [None] * len(batch_states)
    #             for p, indices in player_groups.items():
    #                 group_encodings = torch.stack([batch_encodings[i] for i in indices])
    #                 advantages = self.adv_nets[p](group_encodings)
    #                 for idx, adv in zip(indices, advantages):
    #                     batch_advantages[idx] = adv
            
    #         for i, (state, metadata, advantages) in enumerate(zip(batch_states, batch_metadatas, batch_advantages)):
    #             state_hash = self._hash_state(state, player)
    #             if state_hash in state_values:
    #                 continue
                    
    #             legal_actions = self.env.legal_actions(state)
                
    #             if not legal_actions:
    #                 state_values[state_hash] = self._get_terminal_value(state, player)
    #                 continue
                    
    #             legal_mask = torch.zeros(NUM_ACTIONS, device=DEVICE)
    #             legal_mask[legal_actions] = 1.0
    #             probs = self.regret_matching(advantages, legal_mask)
                
    #             if state.to_act == player:
    #                 all_children_processed = True
    #                 action_values = []
                    
    #                 for action in legal_actions:
    #                     next_state = self.env.step(state, action)
    #                     next_hash = self._hash_state(next_state, player)
    #                     if next_hash in state_values:
    #                         action_values.append(state_values[next_hash])
    #                     else:
    #                         all_children_processed = False
    #                         stack.append((next_state, metadata + [action]))
    #                         action_values.append(0.0)
                    
    #                 if all_children_processed:
    #                     action_probs = probs[legal_actions]
    #                     node_value = torch.dot(action_probs, torch.tensor(action_values, device=DEVICE)).item()
    #                     state_values[state_hash] = node_value
    #                     advantages_tensor = torch.zeros(NUM_ACTIONS, device=DEVICE)
    #                     for action, value in zip(legal_actions, action_values):
    #                         advantages_tensor[action] = value - node_value
    #                     self.adv_buffers[player].add((
    #                         batch_encodings[i].cpu(),
    #                         advantages_tensor.cpu(),
    #                         legal_mask.cpu()
    #                     ))
    #                 else:
    #                     stack.appendleft((state, metadata))
    #             else:
    #                 action = self.sample_action(probs)
    #                 next_state = self.env.step(state, action)
    #                 next_hash = self._hash_state(next_state, player)
    #                 if next_hash in state_values:
    #                     state_values[state_hash] = state_values[next_hash]
    #                 else:
    #                     stack.append((next_state, metadata + [action]))
        
    #     root_values = [state_values.get(self._hash_state(state, player), 0.0) for state in root_states]
    #     return root_values
    def traverse(self, state: GameState, player: int) -> float:
        """
        Original API method - now uses External Sampling MCCFR
        """
        return self.external_sampling_trial(state, player)
    
    def vectorized_traverse_batch(self, root_states: List[GameState], player: int) -> List[float]:
        """
        Vectorized API method - now uses parallel External Sampling MCCFR
        """
        if not root_states:
            return []
        
        values = []
        for state in root_states:
            # Run multiple trials for each root state and average
            trials = [self.external_sampling_trial(state, player) for _ in range(4)]
            values.append(sum(trials) / len(trials))
        
        return values
    # def external_sampling_trial(self, state: GameState, player: int, prob: float = 1.0, cache=None) -> float:
    #     """Optimized version that samples baseline actions"""
    #     if cache is None:
    #         cache = {}
    #     state_hash = self._hash_state(state, player)
    #     if state_hash in cache:
    #         return cache[state_hash]
    
    #     if state.terminal:
    #         payoff = self.env.terminal_payoff(state, player)
    #         cache[state_hash] = payoff
    #         return payoff
    
    #     current_player = state.to_act
    #     legal_actions = self.env.legal_actions(state)
    #     if not legal_actions:
    #         payoff = self._get_terminal_value(state, player)
    #         cache[state_hash] = payoff
    #         return payoff
    
    #     encoding = encode_state(state, current_player).to(DEVICE)
    #     legal_mask = torch.zeros(NUM_ACTIONS, device=DEVICE)
    #     legal_mask[legal_actions] = 1.0
    
    #     if current_player == player:
    #         with torch.no_grad():
    #             advantages = self.adv_nets[player](encoding.unsqueeze(0)).squeeze(0)
    #         strategy = self.regret_matching(advantages, legal_mask)
    #         action = self.sample_action(strategy)
    #         next_state = self.env.step(state, action)
            
    #         # Payoff for chosen action
    #         payoff = self.external_sampling_trial(next_state, player, prob * strategy[action].item(), cache)
            
    #         # Sample a few other actions to estimate counterfactual value
    #         sampled_actions = [action]  # Always include the chosen action
    #         if len(legal_actions) > 1:
    #             # Sample 2 additional actions for baseline
    #             other_actions = [a for a in legal_actions if a != action]
    #             num_samples = min(2, len(other_actions))
    #             sampled_actions.extend(random.sample(other_actions, num_samples))
            
    #         # Compute values for sampled actions
    #         sampled_values = {}
    #         for a in sampled_actions:
    #             if a == action:
    #                 sampled_values[a] = payoff
    #             else:
    #                 next_state_a = self.env.step(state, a)
    #                 sampled_values[a] = self.external_sampling_trial(next_state_a, player, prob * strategy[a].item(), cache)
            
    #         # Estimate counterfactual value using sampled actions
    #         total_prob = sum(strategy[a].item() for a in sampled_actions)
    #         if total_prob > 0:
    #             cf_value_estimate = sum(strategy[a].item() * sampled_values[a] for a in sampled_actions) / total_prob
    #         else:
    #             cf_value_estimate = sum(sampled_values.values()) / len(sampled_values)
            
    #         # Compute advantage for the chosen action
    #         advantage = torch.zeros(NUM_ACTIONS, device=DEVICE)
    #         advantage[action] = payoff - cf_value_estimate
            
    #         # Store for training
    #         self.adv_buffers[player].add((encoding.cpu(), advantage.cpu(), legal_mask.cpu()))
            
    #         cache[state_hash] = payoff
    #         return payoff
    #     else:
    #         action = random.choice(legal_actions)
    #         next_state = self.env.step(state, action)
    #         payoff = self.external_sampling_trial(next_state, player, prob, cache)
    #         cache[state_hash] = payoff
    #         return payoff
    def external_sampling_trial(self, state: GameState, player: int, reach: float = 1.0, cache=None) -> float:
        """Mathematically correct External Sampling MCCFR"""
        if cache is None:
            cache = {}
        state_hash = self._hash_state(state, player)
        if state_hash in cache:
            return cache[state_hash]
    
        if state.terminal:
            payoff = self.env.terminal_payoff(state, player)
            cache[state_hash] = payoff
            return payoff
    
        current_player = state.to_act
        legal_actions = self.env.legal_actions(state)
        if not legal_actions:
            payoff = self._get_terminal_value(state, player)
            cache[state_hash] = payoff
            return payoff
    
        encoding = encode_state(state, current_player).to(DEVICE)
        legal_mask = torch.zeros(NUM_ACTIONS, device=DEVICE)
        legal_mask[legal_actions] = 1.0
    
        if current_player == player:
            # Current player's decision point
            with torch.no_grad():
                advantages = self.adv_nets[player](encoding.unsqueeze(0)).squeeze(0)
            strategy = self.regret_matching(advantages, legal_mask)
            
            # Sample action according to strategy
            action = self.sample_action(strategy)
            next_state = self.env.step(state, action)
            
            # Recursively get value for chosen action
            value_action = self.external_sampling_trial(next_state, player, reach * strategy[action].item(), cache)
            
            # MATHEMATICALLY CORRECT: Compute counterfactual values for ALL actions
            action_values = {}
            for a in legal_actions:
                next_state_a = self.env.step(state, a)
                action_values[a] = self.external_sampling_trial(next_state_a, player, reach * strategy[a].item(), cache)
            
            # Compute counterfactual value (expected value)
            cf_value = sum(strategy[a].item() * action_values[a] for a in legal_actions)
            
            # Compute advantages for ALL actions (not just the chosen one)
            advantages_tensor = torch.zeros(NUM_ACTIONS, device=DEVICE)
            for a in legal_actions:
                advantages_tensor[a] = action_values[a] - cf_value
            
            # Store advantages for training - CRITICAL for CFR
            self.adv_buffers[player].add((
                encoding.cpu(),
                advantages_tensor.cpu(),
                legal_mask.cpu()
            ))
            
            cache[state_hash] = value_action
            return value_action
            
        else:
            # Opponent's decision point - sample uniformly
            action = random.choice(legal_actions)
            next_state = self.env.step(state, action)
            # Opponent's reach probability doesn't affect our value calculation
            value = self.external_sampling_trial(next_state, player, reach, cache)
            cache[state_hash] = value
            return value           
    def parallel_traverse_initial(self, num_parallel_games: int = 8, player: int = 0) -> float:
        """Run multiple independent external sampling trials in sequence"""
        root_states = [self.env.new_hand() for _ in range(num_parallel_games)]
        values = []
        for state in root_states:
            values.append(self.external_sampling_trial(state, player))
        return sum(values) / len(values) if values else 0.0

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

    def sample_strategy_trajectory(self):
        env = self.env
        s = env.new_hand()
        while not s.terminal:
            to_act = s.to_act
            x = encode_state(s, to_act).to(DEVICE)
            with torch.no_grad():
                adv_values = self.adv_nets[to_act](x.unsqueeze(0)).squeeze(0)
            legal_actions = env.legal_actions(s)
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

    def train_advantage_net(self, player: int):
        if len(self.adv_buffers[player]) < BATCH_SIZE:
            return None
        net = self.adv_nets[player]
        opt = self.adv_opts[player]
        mse = torch.nn.MSELoss()
        
        batch = self.adv_buffers[player].sample(BATCH_SIZE)
        xs, ys, masks = zip(*batch)
        xs = torch.stack(xs).to(DEVICE)
        ys = torch.stack(ys).to(DEVICE)
        masks = torch.stack(masks).to(DEVICE)
        
        preds = net(xs)
        loss = mse(preds * masks, ys * masks)
        
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        opt.step()
        
        return loss.item()

    def train_policy_net(self):
        if len(self.strat_buffer) < BATCH_SIZE:
            return None
        ce = torch.nn.KLDivLoss(reduction="batchmean")
        
        batch = self.strat_buffer.sample(BATCH_SIZE)
        xs, target_probs, masks = zip(*batch)
        xs = torch.stack(xs).to(DEVICE)
        target_probs = torch.stack(target_probs).to(DEVICE)
        masks = torch.stack(masks).to(DEVICE)
        
        target_probs = target_probs * masks
        row_sums = target_probs.sum(dim=1, keepdim=True)
        row_sums[row_sums == 0] = 1.0
        target_probs = target_probs / row_sums
        
        logp = self.policy_net(xs)
        logp = logp * masks
        loss = ce(logp, target_probs)
        
        self.policy_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.policy_opt.step()
        
        return loss.item()

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
                mask = torch.full((NUM_ACTIONS,), -1e9, device=DEVICE)
                for a in legal_actions:
                    mask[a] = 0
                probs = torch.softmax(logp + mask, dim=-1)
                a = torch.multinomial(probs, 1).item()
                s = env.step(s, a)
            total += env.terminal_payoff(s, 0)
        return total / num_hands

    def clear_cache(self):
        self._encoding_cache.clear()

    def train(self, num_iterations: int, traversals_per_iter: int, strat_samples_per_iter: int):
        for it in range(1, num_iterations + 1):
            iteration_start = time.time()
            adv_losses_iter = []
            
            for p in [0, 1]:
                traversal_start = time.time()
                games_per_batch = min(32, traversals_per_iter)
                num_batches = max(1, traversals_per_iter // games_per_batch)
                total_games = 0
                for batch_idx in range(num_batches):
                    batch_value = self.parallel_traverse_initial(games_per_batch, p)
                    total_games += games_per_batch
                traversal_time = time.time() - traversal_start
                logger.info(f"Player {p}: {traversal_time:.2f}s for {total_games} games")
                loss = self.train_advantage_net(p)
                if loss is not None:
                    adv_losses_iter.append(loss)
            
            strategy_start = time.time()
            for _ in range(strat_samples_per_iter):
                self.sample_strategy_trajectory()
            strategy_time = time.time() - strategy_start
            
            policy_loss = self.train_policy_net()
            
            iteration_time = time.time() - iteration_start
            
            if adv_losses_iter:
                avg_adv_loss = sum(adv_losses_iter) / len(adv_losses_iter)
                self.adv_losses.append(avg_adv_loss)
            else:
                avg_adv_loss = 0.0
                
            if policy_loss is not None:
                self.policy_losses.append(policy_loss)
            else:
                policy_loss = 0.0

            logger.info(
                f"Iter {it}: "
                f"adv_buf0={len(self.adv_buffers[0])}, "
                f"adv_buf1={len(self.adv_buffers[1])}, "
                f"strat_buf={len(self.strat_buffer)}, "
                f"adv_loss={avg_adv_loss:.4f}, "
                f"policy_loss={policy_loss:.4f}, "
                f"iter_time={iteration_time:.2f}s"
            )
            
            if it % 100 == 0:
                self.clear_cache()

    def save_models(self, path: str = "models"):
        import os
        os.makedirs(path, exist_ok=True)
        torch.save(self.adv_nets[0].state_dict(), f"{path}/adv_p0.pt")
        torch.save(self.adv_nets[1].state_dict(), f"{path}/adv_p1.pt")
        torch.save(self.policy_net.state_dict(), f"{path}/policy.pt")
        logger.info(f"Saved models to {path}/")

    def choose_action_policy(self, state: GameState, player: int) -> int:
        x = encode_state(state, player).to(DEVICE)
        with torch.no_grad():
            logp = self.policy_net(x.unsqueeze(0)).squeeze(0)
        probs = torch.exp(logp)
        legal_actions = self.env.legal_actions(state)
        mask = torch.zeros(NUM_ACTIONS, dtype=torch.float32, device=DEVICE)
        mask[legal_actions] = 1.0
        probs = probs * mask
        total = probs.sum()
        if total.item() <= 0:
            probs = mask / mask.sum()
        else:
            probs = probs / total
        print(probs)
        a = torch.multinomial(probs, 1).item()
        if a not in legal_actions:
            a = random.choice(legal_actions)
        return a
