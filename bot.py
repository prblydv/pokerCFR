# ---------------------------------------------------------------------------
# File overview:
#   bot.py bundles a lightweight SimpleHoldemEnv, neural nets, replay buffers,
#   and DeepCFRTrainer demo. Run via `python bot.py` to execute `run_demo()`.
# ---------------------------------------------------------------------------

import math
import random
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.optim as optim

# =====================
# 1) Game abstraction
# =====================
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,          # change to DEBUG for more detail
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger("DeepCFR")

RNG = random.Random(42)

ACTION_FOLD = 0
ACTION_CALL = 1
ACTION_HALF_POT = 2
ACTION_POT = 3
ACTION_ALL_IN = 4
NUM_ACTIONS = 5

STREET_PREFLOP = 0
STREET_FLOP = 1
STREET_TURN = 2
STREET_RIVER = 3

@dataclass
class GameState:
    # Simple HU NLHE abstraction
    deck: List[int]              # remaining cards (0..51)
    board: List[int]             # community cards
    hole: List[List[int]]        # hole[0], hole[1]
    pot: float
    to_act: int                  # 0 or 1
    street: int                  # 0..3
    stacks: List[float]          # remaining stacks
    current_bet: float           # amount facing the player
    last_aggressor: int          # index of last bettor or -1
    terminal: bool = False
    winner: int = -1             # 0 or 1 if terminal via showdown/fold, -1 otherwise


class SimpleHoldemEnv:
    """
    Highly simplified HU NLHE environment with:
    - Fixed blinds
    - Discrete bet sizes
    - Crude hand strength evaluator
    """

    # Function metadata:
    #   Inputs: stack_size, sb, bb  # dtype=varies
    #   Sample:
    #       sample_output = __init__(stack_size=100.0, sb=None, bb=None)  # dtype=Any
    def __init__(self, stack_size: float = 100.0, sb: float = 0.5, bb: float = 1.0):
        self.stack_size = stack_size
        self.sb = sb
        self.bb = bb

    # Function metadata:
    #   Inputs: no explicit parameters  # dtype=varies
    #   Sample:
    #       sample_output = new_hand()  # dtype=Any
    def new_hand(self):
        deck = list(range(52))
        RNG.shuffle(deck)

        hole0 = [deck.pop(), deck.pop()]
        hole1 = [deck.pop(), deck.pop()]

        # RANDOMIZE seats
        if RNG.random() < 0.5:
            sb_player, bb_player = 0, 1
        else:
            sb_player, bb_player = 1, 0

        stacks = [self.stack_size, self.stack_size]
        stacks[sb_player] -= self.sb
        stacks[bb_player] -= self.bb

        pot = self.sb + self.bb

        state = GameState(
            deck=deck,
            board=[],
            hole=[hole0, hole1],
            pot=pot,
            to_act=sb_player,         # SB acts first preflop
            street=STREET_PREFLOP,
            stacks=stacks,
            current_bet=self.bb,      # SB is facing the BB
            last_aggressor=bb_player,
        )
        return state


    # --- card and state encoding ---

    # Function metadata:
    #   Inputs: card  # dtype=varies
    #   Sample:
    #       sample_output = card_rank(card=None)  # dtype=Any
    @staticmethod
    def card_rank(card: int) -> int:
        # 0..12 repeated for 4 suits
        return card % 13

    # Function metadata:
    #   Inputs: hole, board  # dtype=varies
    #   Sample:
    #       sample_output = crude_strength(hole=[10, 23], board=[0, 1, 2])  # dtype=Any
    def crude_strength(self, hole: List[int], board: List[int]) -> float:
        # Very rough heuristic: sum of ranks of hole + board, normalized
        ranks = [self.card_rank(c) for c in hole + board]
        return sum(ranks) / (13.0 * 7.0)  # at most 7 cards at river

    # Function metadata:
    #   Inputs: s, player  # dtype=varies
    #   Sample:
    #       sample_output = encode_state(s=None, player=0)  # dtype=Any
    def encode_state(self, s: GameState, player: int) -> torch.Tensor:
        """Encode state into a fixed-size vector for networks."""
        # Street one-hot (4), player index (1), pot, stacks (2), current_bet (1),
        # last_aggressor (1), hole strength, board strength
        street_oh = [0.0] * 4
        street_oh[s.street] = 1.0

        pot = s.pot / (self.stack_size * 2.0)
        stacks_norm = [st / self.stack_size for st in s.stacks]
        curr_bet = s.current_bet / self.stack_size
        last_agg = 0.0 if s.last_aggressor == -1 else float(s.last_aggressor == player)

        hole_str = self.crude_strength(s.hole[player], s.board)
        board_str = self.crude_strength([], s.board) if s.board else 0.0

        vec = street_oh + [
            float(player),
            pot,
            stacks_norm[0],
            stacks_norm[1],
            curr_bet,
            last_agg,
            hole_str,
            board_str,
        ]
        return torch.tensor(vec, dtype=torch.float32)

    # --- game mechanics ---

    # Function metadata:
    #   Inputs: s  # dtype=varies
    #   Sample:
    #       sample_output = legal_actions(s=None)  # dtype=Any
    def legal_actions(self, s: GameState) -> List[int]:
        if s.terminal:
            return []

        actions = [ACTION_FOLD, ACTION_CALL]
        if s.stacks[s.to_act] > 0.0:
            actions += [ACTION_HALF_POT, ACTION_POT, ACTION_ALL_IN]
        return actions

    # Function metadata:
    #   Inputs: s  # dtype=varies
    #   Sample:
    #       sample_output = is_betting_round_over(s=None)  # dtype=Any
    def is_betting_round_over(self, s: GameState) -> bool:
        # Betting round ends when:
        # - last action was call/check and no one is facing a bet.
        # For simplicity: if current_bet == 0 and last_aggressor == -1.
        return s.current_bet == 0 and s.last_aggressor == -1

    # Function metadata:
    #   Inputs: s  # dtype=varies
    #   Sample:
    #       sample_output = deal_next_street(s=None)  # dtype=Any
    def deal_next_street(self, s: GameState) -> None:
        if s.street == STREET_PREFLOP:
            # flop: 3 cards
            s.board.extend([s.deck.pop(), s.deck.pop(), s.deck.pop()])
            s.street = STREET_FLOP
        elif s.street == STREET_FLOP:
            s.board.append(s.deck.pop())
            s.street = STREET_TURN
        elif s.street == STREET_TURN:
            s.board.append(s.deck.pop())
            s.street = STREET_RIVER
        else:
            # go to showdown
            self.resolve_showdown(s)

        # reset betting for new street
        s.current_bet = 0.0
        s.last_aggressor = -1

    # Function metadata:
    #   Inputs: s  # dtype=varies
    #   Sample:
    #       sample_output = resolve_showdown(s=None)  # dtype=Any
    def resolve_showdown(self, s: GameState) -> None:
        v0 = self.crude_strength(s.hole[0], s.board)
        v1 = self.crude_strength(s.hole[1], s.board)
        if abs(v0 - v1) < 1e-6:
            # split pot, zero-sum payoff 0 from P0 POV
            s.winner = -1
        else:
            s.winner = 0 if v0 > v1 else 1
        s.terminal = True

    # Function metadata:
    #   Inputs: s, action  # dtype=varies
    #   Sample:
    #       sample_output = step(s=None, action=None)  # dtype=Any
    def step(self, s: GameState, action: int) -> GameState:
        if s.terminal:
            return s

        player = s.to_act
        opp = 1 - player

        # copy state shallowly (immutable fields we reassign)
        s = GameState(
            deck=list(s.deck),
            board=list(s.board),
            hole=[list(s.hole[0]), list(s.hole[1])],
            pot=s.pot,
            to_act=s.to_act,
            street=s.street,
            stacks=list(s.stacks),
            current_bet=s.current_bet,
            last_aggressor=s.last_aggressor,
            terminal=s.terminal,
            winner=s.winner,
        )

        if action == ACTION_FOLD:
            # Opponent wins pot
            s.terminal = True
            s.winner = opp
            return s

        if action == ACTION_CALL:
            call_amt = min(s.current_bet, s.stacks[player])
            s.stacks[player] -= call_amt
            s.pot += call_amt
            # After calling, betting round closed => no aggressor, no bet
            s.current_bet = 0.0
            s.last_aggressor = -1

        elif action in (ACTION_HALF_POT, ACTION_POT, ACTION_ALL_IN):
            # Determine raise size
            base_pot = max(s.pot, 1.0)
            if action == ACTION_HALF_POT:
                amount = min(s.stacks[player], base_pot * 0.5)
            elif action == ACTION_POT:
                amount = min(s.stacks[player], base_pot * 1.0)
            else:
                amount = s.stacks[player]  # all-in

            # Bet/raise is relative to current amount facing
            total_to_put = s.current_bet + amount
            total_to_put = min(total_to_put, s.stacks[player] + s.current_bet)
            actual_bet = min(total_to_put, s.stacks[player])

            s.stacks[player] -= actual_bet
            s.pot += actual_bet
            s.current_bet = actual_bet
            s.last_aggressor = player

        # Switch turn
        s.to_act = opp

        # Check if betting round ends naturally
        if self.is_betting_round_over(s):
            # If we are at river and no bet: showdown
            if s.street == STREET_RIVER:
                self.resolve_showdown(s)
            else:
                self.deal_next_street(s)

        return s

    # Function metadata:
    #   Inputs: s, hero  # dtype=varies
    #   Sample:
    #       sample_output = terminal_payoff(s=None, hero=0)  # dtype=Any
    def terminal_payoff(self, s: GameState, hero: int) -> float:
        if not s.terminal:
            return 0.0
        if s.winner == -1:
            return 0.0
        return s.pot if s.winner == hero else -s.pot


# =====================
# 2) Networks
# =====================

class AdvantageNet(nn.Module):
    # Function metadata:
    #   Inputs: state_dim, num_actions  # dtype=varies
    #   Sample:
    #       sample_output = __init__(state_dim=None, num_actions=None)  # dtype=Any
    def __init__(self, state_dim: int, num_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )

    # Function metadata:
    #   Inputs: x  # dtype=varies
    #   Sample:
    #       sample_output = forward(x=None)  # dtype=Any
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PolicyNet(nn.Module):
    # Function metadata:
    #   Inputs: state_dim, num_actions  # dtype=varies
    #   Sample:
    #       sample_output = __init__(state_dim=None, num_actions=None)  # dtype=Any
    def __init__(self, state_dim: int, num_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )

    # Function metadata:
    #   Inputs: x  # dtype=varies
    #   Sample:
    #       sample_output = forward(x=None)  # dtype=Any
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return torch.log_softmax(logits, dim=-1)


# =====================
# 3) Replay buffers
# =====================

class ReservoirBuffer:
    # Function metadata:
    #   Inputs: capacity, rng  # dtype=varies
    #   Sample:
    #       sample_output = __init__(capacity=None, rng=None)  # dtype=Any
    def __init__(self, capacity: int, rng: random.Random):
        self.capacity = capacity
        self.data = []
        self.n_seen = 0
        self.rng = rng

    # Function metadata:
    #   Inputs: sample  # dtype=varies
    #   Sample:
    #       sample_output = add(sample=None)  # dtype=Any
    def add(self, sample):
        self.n_seen += 1
        if len(self.data) < self.capacity:
            self.data.append(sample)
        else:
            idx = self.rng.randint(0, self.n_seen - 1)
            if idx < self.capacity:
                self.data[idx] = sample

    # Function metadata:
    #   Inputs: batch_size  # dtype=varies
    #   Sample:
    #       sample_output = sample(batch_size=32)  # dtype=Any
    def sample(self, batch_size: int):
        return [self.rng.choice(self.data) for _ in range(batch_size)]

    # Function metadata:
    #   Inputs: no explicit parameters  # dtype=varies
    #   Sample:
    #       sample_output = __len__()  # dtype=Any
    def __len__(self):
        return len(self.data)


# =====================
# 4) Deep CFR trainer
# =====================

class DeepCFRTrainer:
    # Function metadata:
    #   Inputs: env, state_dim  # dtype=varies
    #   Sample:
    #       sample_output = __init__(env=mock_env, state_dim=None)  # dtype=Any
    def __init__(self, env: SimpleHoldemEnv, state_dim: int):
        self.env = env
        self.state_dim = state_dim

        # One advantage net per player
        self.adv_nets = [
            AdvantageNet(state_dim, NUM_ACTIONS),
            AdvantageNet(state_dim, NUM_ACTIONS),
        ]
        self.adv_opts = [
            optim.Adam(self.adv_nets[0].parameters(), lr=1e-3),
            optim.Adam(self.adv_nets[1].parameters(), lr=1e-3),
        ]

        # One policy (average strategy) net shared
        self.policy_net = PolicyNet(state_dim, NUM_ACTIONS)
        self.policy_opt = optim.Adam(self.policy_net.parameters(), lr=1e-3)

        self.adv_buffers = [
            ReservoirBuffer(50_000, RNG),
            ReservoirBuffer(50_000, RNG),
        ]
        self.strat_buffer = ReservoirBuffer(100_000, RNG)

    # --- Strategy helpers ---

    # Function metadata:
    #   Inputs: advantages, legal_mask  # dtype=varies
    #   Sample:
    #       sample_output = regret_matching(advantages=None, legal_mask=None)  # dtype=Any
    def regret_matching(self, advantages: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
        # advantages shape: [A], legal_mask: [A] with 0/1
        adv = advantages.clone()
        adv[legal_mask == 0] = 0.0
        pos = torch.clamp(adv, min=0.0)
        total = pos.sum()
        if total.item() <= 0.0:
            # uniform over legal
            num_legal = legal_mask.sum().item()
            probs = torch.zeros_like(adv)
            probs[legal_mask == 1] = 1.0 / max(num_legal, 1)
            return probs
        return pos / total

    # Function metadata:
    #   Inputs: probs  # dtype=varies
    #   Sample:
    #       sample_output = sample_action(probs=None)  # dtype=Any
    def sample_action(self, probs: torch.Tensor) -> int:
        # probs is 1D tensor
        probs_np = probs.detach().cpu().numpy()
        r = RNG.random()
        cum = 0.0
        for i, p in enumerate(probs_np):
            cum += p
            if r <= cum:
                return i
        return len(probs_np) - 1

    # --- CFR traversal (external sampling-like, simplified) ---

    # Function metadata:
    #   Inputs: state, player, reach_prob  # dtype=varies
    #   Sample:
    #       sample_output = traverse(state=mock_state, player=0, reach_prob=None)  # dtype=Any
    def traverse(self, state: GameState, player: int, reach_prob: float) -> float:
        """
        Returns value from 'player' perspective.
        """
        env = self.env
        if state.terminal:
            return env.terminal_payoff(state, player)

        # If chance node (pre-deal handled; here we just use fixed env)
        me = player
        to_act = state.to_act

        # Encode state from current actor's POV
        x = env.encode_state(state, to_act)
        with torch.no_grad():
            adv_values = self.adv_nets[to_act](x.unsqueeze(0)).squeeze(0)

        legal_actions = env.legal_actions(state)
        legal_mask = torch.zeros(NUM_ACTIONS, dtype=torch.float32)
        for a in legal_actions:
            legal_mask[a] = 1.0

        # Strategy for current actor via regret-matching
        probs = self.regret_matching(adv_values, legal_mask)

        # If this is the traverser player, we need to compute action-specific values
        if to_act == me:
            # sample one action for forward recursion, but compute advantages
            action_values = []
            for a in range(NUM_ACTIONS):
                if a not in legal_actions:
                    action_values.append(0.0)
                    continue
                next_state = env.step(state, a)
                v = self.traverse(next_state, player, reach_prob * probs[a].item())
                action_values.append(v)

            node_val = sum(probs[a].item() * action_values[a] for a in range(NUM_ACTIONS))

            # Store advantages in buffer (advantage = Q(a) - V)
            advantages = [action_values[a] - node_val if a in legal_actions else 0.0
                          for a in range(NUM_ACTIONS)]
            self.adv_buffers[me].add((x, torch.tensor(advantages, dtype=torch.float32),
                                      legal_mask.clone()))

            return node_val
        else:
            # Opponent / other player: sample action according to probs
            a = self.sample_action(probs)
            if a not in legal_actions:
                # fallback: choose random legal
                a = RNG.choice(legal_actions)
            next_state = env.step(state, a)
            # Value passed through without storing advantages
            return self.traverse(next_state, player, reach_prob)

    # --- Strategy sampling for policy network ---

    # Function metadata:
    #   Inputs: no explicit parameters  # dtype=varies
    #   Sample:
    #       sample_output = sample_strategy_trajectory()  # dtype=Any
    def sample_strategy_trajectory(self):
        """
        Generate samples for strategy buffer using current policy net.
        """
        env = self.env
        s = env.new_hand()
        while not s.terminal:
            to_act = s.to_act
            x = env.encode_state(s, to_act)
            with torch.no_grad():
                logp = self.policy_net(x.unsqueeze(0)).squeeze(0)
            legal_actions = env.legal_actions(s)
            mask = torch.zeros(NUM_ACTIONS, dtype=torch.float32)
            mask[legal_actions] = 1.0
            probs = torch.exp(logp)
            probs = probs * mask
            total = probs.sum()
            if total.item() <= 0:
                probs = mask / mask.sum()
            else:
                probs = probs / total
            a = self.sample_action(probs)
            if a not in legal_actions:
                a = RNG.choice(legal_actions)
            # store state, chosen action, and target distribution (probs)
            self.strat_buffer.add((x, a, probs.clone()))
            s = env.step(s, a)

    # --- Training functions ---

    # Function metadata:
    #   Inputs: player, batch_size, epochs  # dtype=varies
    #   Sample:
    #       sample_output = train_advantage_net(player=0, batch_size=32, epochs=None)  # dtype=Any
    def train_advantage_net(self, player: int, batch_size: int = 64, epochs: int = 1):
        if len(self.adv_buffers[player]) < batch_size:
            return
        net = self.adv_nets[player]
        opt = self.adv_opts[player]
        mse = nn.MSELoss()

        for _ in range(epochs):
            batch = self.adv_buffers[player].sample(batch_size)
            xs, ys, masks = zip(*batch)
            xs = torch.stack(xs)
            ys = torch.stack(ys)
            masks = torch.stack(masks)

            preds = net(xs)
            # Only care about legal actions (masks==1)
            loss = mse(preds * masks, ys * masks)

            opt.zero_grad()
            loss.backward()
            opt.step()

    # Function metadata:
    #   Inputs: batch_size, epochs  # dtype=varies
    #   Sample:
    #       sample_output = train_policy_net(batch_size=32, epochs=None)  # dtype=Any
    def train_policy_net(self, batch_size: int = 64, epochs: int = 1):
        if len(self.strat_buffer) < batch_size:
            return
        ce = nn.KLDivLoss(reduction="batchmean")
        for _ in range(epochs):
            batch = self.strat_buffer.sample(batch_size)
            xs, _, target_probs = zip(*batch)
            xs = torch.stack(xs)
            target_probs = torch.stack(target_probs)

            logp = self.policy_net(xs)
            loss = ce(logp, target_probs)

            self.policy_opt.zero_grad()
            loss.backward()
            self.policy_opt.step()

    # --- Main Deep CFR loop ---

    # Function metadata:
    #   Inputs: num_iterations, traversals_per_iter, strat_samples_per_iter  # dtype=varies
    #   Sample:
    #       sample_output = train(num_iterations=10, traversals_per_iter=5, strat_samples_per_iter=5)  # dtype=Any
    def train(self, num_iterations: int = 2000,
              traversals_per_iter: int = 50,
              strat_samples_per_iter: int = 50):
        for it in range(1, num_iterations + 1):
            # 1) Advantage learning for each player
            for p in [0, 1]:
                for _ in range(traversals_per_iter):
                    s = self.env.new_hand()
                    self.traverse(s, p, reach_prob=1.0)
                self.train_advantage_net(p, batch_size=64, epochs=1)

            # 2) Strategy sampling & training
            for _ in range(strat_samples_per_iter):
                self.sample_strategy_trajectory()
            self.train_policy_net(batch_size=64, epochs=1)

            if it % 5 == 0:
                print(f"Iteration {it}, adv_buf0={len(self.adv_buffers[0])}, "
                      f"adv_buf1={len(self.adv_buffers[1])}, strat_buf={len(self.strat_buffer)}")

    # --- Play function (using policy net) ---

    # Function metadata:
    #   Inputs: state, player  # dtype=varies
    #   Sample:
    #       sample_output = choose_action(state=mock_state, player=0)  # dtype=Any
    def choose_action(self, state: GameState, player: int) -> int:
        env = self.env
        x = env.encode_state(state, player)
        with torch.no_grad():
            logp = self.policy_net(x.unsqueeze(0)).squeeze(0)
        probs = torch.exp(logp)
        legal_actions = env.legal_actions(state)
        mask = torch.zeros(NUM_ACTIONS, dtype=torch.float32)
        mask[legal_actions] = 1.0
        probs = probs * mask
        total = probs.sum()
        if total.item() <= 0:
            probs = mask / mask.sum()
        else:
            probs = probs / total
        return self.sample_action(probs)


# =====================
# 5) Demo: train + play
# =====================

# Function metadata:
#   Inputs: no explicit parameters  # dtype=varies
#   Sample:
#       sample_output = run_demo()  # dtype=Any
def run_demo():
    logger.info("Initializing SimpleHoldemEnv...")
    env = SimpleHoldemEnv()

    example_state = env.new_hand()
    state_dim = env.encode_state(example_state, player=0).shape[0]
    logger.info(f"Detected state dimension: {state_dim}")

    trainer = DeepCFRTrainer(env, state_dim)
    logger.info("Starting Deep CFR training...")
    trainer.train(num_iterations=200, traversals_per_iter=20, strat_samples_per_iter=20)
    logger.info("Training complete.")

    # Play one hand vs policy
    logger.info("\nPlaying a sample hand vs the trained policy (you = P0)")
    s = env.new_hand()

    while not s.terminal:
        if s.to_act == 0:
            a = trainer.choose_action(s, 0)
            logger.info(f"P0 action → {a}")
            s = env.step(s, a)
        else:
            a = trainer.choose_action(s, 1)
            logger.info(f"P1 action → {a}")
            s = env.step(s, a)

    payoff_p0 = env.terminal_payoff(s, 0)
    logger.info(f"Hand finished: Winner={s.winner}, payoff P0={payoff_p0}, pot={s.pot}")

if __name__ == "__main__":
    run_demo()
