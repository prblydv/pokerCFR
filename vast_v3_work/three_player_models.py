"""Neural information-state encoding and networks for three-player Hold'em.

The encoder deliberately exposes only the acting player's private cards.  Public
betting history is retained (up to ``max_history`` actions) so that distinct
betting lines do not collapse into the same information set merely because the
current pot happens to be equal.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from three_player_engine import NUM_ACTIONS


CARD_FEATURES = 18  # 13 ranks + 4 suits + present bit
DEFAULT_MAX_HISTORY = 32

# The legacy encoder contains 183 non-history values and 17 values per public
# action.  Tournament features are deliberately appended *after* that complete
# vector.  Consequently a legacy checkpoint still receives byte-for-byte the
# same input when ``include_tournament_features=False``, and its input layer can
# be copied into the prefix of an expanded network for a warm start.
LEGACY_FIXED_FEATURES = 183
HISTORY_FEATURES = 4 + 3 + NUM_ACTIONS + 1
TOURNAMENT_FEATURE_NAMES = (
    "hero_alive",
    "clockwise_1_alive",
    "clockwise_2_alive",
    "hero_starting_chip_share",
    "clockwise_1_starting_chip_share",
    "clockwise_2_starting_chip_share",
    "hero_stack_behind",
    "effective_stack_vs_clockwise_1",
    "effective_stack_vs_clockwise_2",
    "players_remaining",
    "players_in_hand",
    "heads_up",
    "tournament_chip_scale",
    "shortest_live_starting_stack",
    "largest_live_starting_stack",
)
TOURNAMENT_FEATURES = len(TOURNAMENT_FEATURE_NAMES)
CARD_STATE_PREFIX_FEATURES = 48
CARD_TOKEN_COUNT = 7
CARD_STATE_FEATURES = CARD_TOKEN_COUNT * CARD_FEATURES
HISTORY_OFFSET = CARD_STATE_PREFIX_FEATURES + CARD_STATE_FEATURES
NETWORK_ARCHITECTURES = (
    "residual_mlp",
    "dual_attention_state",
    "deep_cfr_branch",
    "deep_cfr_branch_v2",
    "deep_cfr_branch_v3",
)

# The v2 branch derives these values directly from the seven existing card
# tokens. The encoded observation width therefore remains unchanged and old
# replay reservoirs can be reused during an architecture migration.
POKER_RELATIONAL_FEATURES = 66


def information_state_size(
    max_history: int = DEFAULT_MAX_HISTORY,
    *,
    include_tournament_features: bool = False,
) -> int:
    """Return the encoder width without needing to deal a probe hand."""

    if max_history <= 0:
        raise ValueError("max_history must be positive")
    return (
        LEGACY_FIXED_FEATURES
        + HISTORY_FEATURES * int(max_history)
        + (TOURNAMENT_FEATURES if include_tournament_features else 0)
    )


def _alive_flags(state, explicit: Sequence[bool] | None) -> list[bool]:
    """Resolve tournament entrants while remaining compatible with old states."""

    if explicit is not None:
        values = list(explicit)
    elif hasattr(state, "alive"):
        values = list(state.alive)
    elif hasattr(state, "eliminated"):
        values = [not bool(value) for value in state.eliminated]
    elif hasattr(state, "initial_stacks"):
        # Current stack==0 can merely mean all-in.  Starting stack==0 is the
        # safe legacy-state indication that a seat never entered this hand.
        values = [float(value) > 1e-9 for value in state.initial_stacks]
    else:
        values = [True, True, True]
    if len(values) != 3:
        raise ValueError("alive_flags must contain exactly three values")
    return [bool(value) for value in values]


def _card_features(card: int | None) -> list[float]:
    out = [0.0] * CARD_FEATURES
    if card is None or card < 0:
        return out
    if card >= 52:
        raise ValueError(f"card index must be in [0, 51], got {card}")
    out[card % 13] = 1.0
    out[13 + card // 13] = 1.0
    out[17] = 1.0
    return out


def _event_values(event) -> tuple[int, int, int, float]:
    """Accept either tuple histories or small history dataclasses."""
    if isinstance(event, (tuple, list)):
        if len(event) < 4:
            raise ValueError("history tuples need street, player, action, amount")
        return int(event[0]), int(event[1]), int(event[2]), float(event[3])
    street = int(getattr(event, "street"))
    player = int(getattr(event, "player"))
    action = int(getattr(event, "action"))
    amount = float(
        getattr(event, "amount", getattr(event, "amount_added", 0.0))
    )
    return street, player, action, amount


def encode_information_state(
    state,
    hero: int,
    legal_actions: Iterable[int],
    stack_size: float,
    max_history: int = DEFAULT_MAX_HISTORY,
    *,
    include_tournament_features: bool = False,
    alive_flags: Sequence[bool] | None = None,
    tournament_total_chips: float | None = None,
) -> torch.Tensor:
    """Return a fixed-width, hero-visible information-state tensor.

    Seat-indexed public fields are rotated so index zero is always ``hero``.
    This lets every player-specific network learn all three button positions
    without leaking either opponent's hole cards.

    The default is the legacy encoder so existing 727-wide (at history 32)
    policy snapshots remain directly usable.  New tournament trainers opt into
    the appended suffix with ``include_tournament_features=True``.  Explicit
    ``alive_flags`` take precedence over state fields and are useful for tools
    adapting older state objects.
    """
    if hero not in (0, 1, 2):
        raise ValueError(f"hero must be 0, 1, or 2; got {hero}")
    if stack_size <= 0:
        raise ValueError("stack_size must be positive")
    if max_history <= 0:
        raise ValueError("max_history must be positive")

    # Native states already store every required public field in packed C++
    # structures. Avoid millions of Python property/list allocations during
    # traversal. Explicit alive overrides remain on the general Python path.
    if alive_flags is None and type(state).__module__ == "poker_native_engine":
        try:
            from three_player_native import encode_information_state_native

            encoded = encode_information_state_native(
                state,
                hero,
                legal_actions,
                stack_size,
                max_history,
                include_tournament_features=include_tournament_features,
                tournament_total_chips=tournament_total_chips,
            )
            return torch.from_numpy(encoded)
        except (ImportError, AttributeError):
            # An older optional extension may not contain the fast encoder.
            # Keeping this fallback makes source upgrades safe before rebuild.
            pass

    x: list[float] = []

    # Street and relative button position.
    x.extend(1.0 if state.street == i else 0.0 for i in range(4))
    relative_button = (state.button - hero) % 3
    x.extend(1.0 if relative_button == i else 0.0 for i in range(3))
    x.extend(
        [
            float(hero == state.button),
            float(hero == state.sb_player),
            float(hero == state.bb_player),
        ]
    )

    # Public seat data, in hero-relative clockwise order. Pending/raise-right
    # flags matter after short all-ins and are part of the public game state.
    pending_actors = set(
        getattr(state, "pending_actors", getattr(state, "pending", ()))
    )
    raise_rights = list(getattr(state, "raise_rights", [True, True, True]))
    last_action_bet = list(getattr(state, "last_action_bet", [None, None, None]))
    for offset in range(3):
        seat = (hero + offset) % 3
        x.extend(
            [
                float(state.stacks[seat]) / stack_size,
                float(state.total_contrib[seat]) / stack_size,
                float(state.street_contrib[seat]) / stack_size,
                float(state.folded[seat]),
                float(state.all_in[seat]),
                float(seat in pending_actors),
                float(raise_rights[seat]),
                (
                    float(last_action_bet[seat]) / stack_size
                    if last_action_bet[seat] is not None
                    else 0.0
                ),
                float(last_action_bet[seat] is not None),
            ]
        )

    last_full_raiser = getattr(state, "last_full_raiser", None)
    x.append(float(last_full_raiser is None))
    x.extend(
        float(last_full_raiser is not None and (last_full_raiser - hero) % 3 == i)
        for i in range(3)
    )

    to_call = max(float(state.current_bet) - float(state.street_contrib[hero]), 0.0)
    active_count = sum(not f for f in state.folded)
    pending_count = len(pending_actors)
    x.extend(
        [
            float(state.pot) / (3.0 * stack_size),
            float(state.current_bet) / stack_size,
            float(state.min_raise) / stack_size,
            to_call / stack_size,
            len(state.board) / 5.0,
            active_count / 3.0,
            pending_count / 3.0,
        ]
    )

    # Hero cards and the five public board slots. Opponent cards never appear.
    hero_cards: Sequence[int] = sorted(state.hole[hero])
    if len(hero_cards) != 2:
        raise ValueError("hero must have exactly two hole cards")
    for card in hero_cards:
        x.extend(_card_features(card))
    # The three flop cards and two private cards are unordered poker sets. Turn
    # and river retain their street order.
    canonical_board = sorted(state.board[:3]) + list(state.board[3:])
    for i in range(5):
        x.extend(
            _card_features(canonical_board[i] if i < len(canonical_board) else None)
        )

    # Right-align recent public actions. Each slot is street(4), relative
    # actor(3), action(9), and chips added(1).
    history = list(state.history)[-max_history:]
    blank_slots = max_history - len(history)
    x.extend([0.0] * (blank_slots * (4 + 3 + NUM_ACTIONS + 1)))
    for event in history:
        street, player, action, amount = _event_values(event)
        x.extend(1.0 if street == i else 0.0 for i in range(4))
        relative_actor = (player - hero) % 3
        x.extend(1.0 if relative_actor == i else 0.0 for i in range(3))
        x.extend(1.0 if action == i else 0.0 for i in range(NUM_ACTIONS))
        x.append(amount / stack_size)

    legal_set = set(int(a) for a in legal_actions)
    x.extend(1.0 if a in legal_set else 0.0 for a in range(NUM_ACTIONS))

    if include_tournament_features:
        alive = _alive_flags(state, alive_flags)
        starting_stacks = list(
            getattr(
                state,
                "initial_stacks",
                [
                    float(state.stacks[seat]) + float(state.total_contrib[seat])
                    for seat in range(3)
                ],
            )
        )
        if len(starting_stacks) != 3:
            raise ValueError("initial_stacks must contain exactly three values")
        starting_stacks = [max(0.0, float(value)) for value in starting_stacks]
        inferred_total = sum(starting_stacks)
        total_chips = (
            inferred_total
            if tournament_total_chips is None
            else float(tournament_total_chips)
        )
        if total_chips <= 0:
            raise ValueError("tournament_total_chips must be positive")
        if inferred_total > total_chips + 1e-6:
            raise ValueError(
                "state starting stacks exceed tournament_total_chips"
            )

        relative_seats = [(hero + offset) % 3 for offset in range(3)]
        x.extend(float(alive[seat]) for seat in relative_seats)
        x.extend(starting_stacks[seat] / total_chips for seat in relative_seats)

        hero_behind = max(0.0, float(state.stacks[hero]))
        for seat in relative_seats:
            if seat == hero:
                effective = hero_behind if alive[hero] else 0.0
            elif alive[hero] and alive[seat]:
                effective = min(
                    hero_behind, max(0.0, float(state.stacks[seat]))
                )
            else:
                effective = 0.0
            x.append(effective / stack_size)

        players_remaining = sum(alive)
        players_in_hand = sum(
            alive[seat] and not bool(state.folded[seat]) for seat in range(3)
        )
        live_starting = [
            starting_stacks[seat] for seat in range(3) if alive[seat]
        ]
        shortest = min(live_starting) if live_starting else 0.0
        largest = max(live_starting) if live_starting else 0.0
        x.extend(
            [
                players_remaining / 3.0,
                players_in_hand / 3.0,
                float(players_remaining == 2),
                total_chips / (3.0 * stack_size),
                shortest / stack_size,
                largest / stack_size,
            ]
        )

    expected = information_state_size(
        max_history,
        include_tournament_features=include_tournament_features,
    )
    if len(x) != expected:
        raise RuntimeError(f"encoder produced {len(x)} values; expected {expected}")
    return torch.tensor(x, dtype=torch.float32)


class ResidualBlock(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(F.silu(self.norm(x)))
        h = self.fc2(F.silu(h))
        return x + h


def _structured_layout(input_dim: int) -> tuple[int, int]:
    """Infer history length and optional tournament suffix from encoder width."""
    candidates = []
    for tournament_features in (0, TOURNAMENT_FEATURES):
        history_values = int(input_dim) - LEGACY_FIXED_FEATURES - tournament_features
        if history_values > 0 and history_values % HISTORY_FEATURES == 0:
            candidates.append((history_values // HISTORY_FEATURES, tournament_features))
    if len(candidates) != 1:
        raise ValueError(
            f"input width {input_dim} is not a supported poker information state"
        )
    return candidates[0]


class DualAttentionStateBackbone(nn.Module):
    """Card attention, history attention, recurrent memory, and static fusion."""

    def __init__(self, input_dim: int, hidden: int, blocks: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden = int(hidden)
        self.history_steps, self.tournament_features = _structured_layout(input_dim)
        token_dim = max(32, min(128, self.hidden // 2))
        token_dim = max(4, token_dim - token_dim % 4)
        self.token_dim = token_dim
        dropout = 0.05

        self.card_projection = nn.Linear(CARD_FEATURES, token_dim)
        self.card_positions = nn.Parameter(
            torch.zeros(1, CARD_TOKEN_COUNT + 1, token_dim)
        )
        self.card_summary_token = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.card_attention = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=4,
            dim_feedforward=2 * token_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.history_projection = nn.Linear(HISTORY_FEATURES, token_dim)
        self.history_positions = nn.Parameter(
            torch.zeros(1, self.history_steps + 1, token_dim)
        )
        self.history_summary_token = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.history_attention = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=4,
            dim_feedforward=2 * token_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.history_memory = nn.GRU(token_dim, token_dim, batch_first=True)
        self.history_fusion = nn.Linear(2 * token_dim, token_dim)

        static_dim = CARD_STATE_PREFIX_FEATURES + NUM_ACTIONS + self.tournament_features
        self.static_norm = nn.LayerNorm(static_dim)
        self.static_projection = nn.Linear(static_dim, token_dim)
        self.fusion = nn.Linear(3 * token_dim, hidden)
        self.blocks = nn.ModuleList(ResidualBlock(hidden) for _ in range(blocks))
        self.output_norm = nn.LayerNorm(hidden)

        nn.init.normal_(self.card_summary_token, std=0.02)
        nn.init.normal_(self.history_summary_token, std=0.02)
        nn.init.normal_(self.card_positions, std=0.02)
        nn.init.normal_(self.history_positions, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or int(x.shape[1]) != self.input_dim:
            raise ValueError(
                f"structured network expects [batch, {self.input_dim}] inputs"
            )
        batch = int(x.shape[0])
        history_end = HISTORY_OFFSET + self.history_steps * HISTORY_FEATURES

        cards = x[:, CARD_STATE_PREFIX_FEATURES:HISTORY_OFFSET].reshape(
            batch, CARD_TOKEN_COUNT, CARD_FEATURES
        )
        card_present = cards[:, :, -1] > 0.5
        card_tokens = self.card_projection(cards)
        card_summary = self.card_summary_token.expand(batch, -1, -1)
        card_sequence = torch.cat((card_summary, card_tokens), dim=1)
        card_sequence = card_sequence + self.card_positions
        card_padding = torch.cat(
            (
                torch.zeros(batch, 1, dtype=torch.bool, device=x.device),
                ~card_present,
            ),
            dim=1,
        )
        card_representation = self.card_attention(
            card_sequence, src_key_padding_mask=card_padding
        )[:, 0]

        history = x[:, HISTORY_OFFSET:history_end].reshape(
            batch, self.history_steps, HISTORY_FEATURES
        )
        history_present = history.abs().sum(dim=2) > 0.0
        history_tokens = self.history_projection(history)
        lengths = history_present.sum(dim=1)
        maximum_length = int(lengths.max().item())
        start = self.history_steps - lengths
        order = (
            torch.arange(self.history_steps, device=x.device).unsqueeze(0)
            + start.unsqueeze(1)
        ) % self.history_steps
        compact_history = history_tokens.gather(
            1, order.unsqueeze(2).expand(-1, -1, self.token_dim)
        )[:, :maximum_length]
        compact_positions = self.history_positions[:, 1:].expand(batch, -1, -1).gather(
            1, order.unsqueeze(2).expand(-1, -1, self.token_dim)
        )[:, :maximum_length]
        compact_padding = (
            torch.arange(maximum_length, device=x.device).unsqueeze(0)
            >= lengths.unsqueeze(1)
        )
        history_summary = self.history_summary_token.expand(batch, -1, -1)
        history_sequence = torch.cat((history_summary, compact_history), dim=1)
        history_sequence = history_sequence + torch.cat(
            (self.history_positions[:, :1].expand(batch, -1, -1), compact_positions),
            dim=1,
        )
        history_padding = torch.cat(
            (
                torch.zeros(batch, 1, dtype=torch.bool, device=x.device),
                compact_padding,
            ),
            dim=1,
        )
        attention_history = self.history_attention(
            history_sequence, src_key_padding_mask=history_padding
        )[:, 0]

        if maximum_length:
            memory_sequence, _ = self.history_memory(compact_history)
            final_index = (lengths - 1).clamp(min=0)
            memory_history = memory_sequence[
                torch.arange(batch, device=x.device), final_index
            ]
            memory_history = memory_history * (lengths > 0).unsqueeze(1)
        else:
            memory_history = history_tokens.new_zeros(batch, self.token_dim)
        history_representation = F.silu(
            self.history_fusion(
                torch.cat((attention_history, memory_history), dim=1)
            )
        )

        static = torch.cat(
            (
                x[:, :CARD_STATE_PREFIX_FEATURES],
                x[:, history_end : history_end + NUM_ACTIONS],
                x[:, history_end + NUM_ACTIONS :],
            ),
            dim=1,
        )
        static_representation = F.silu(
            self.static_projection(self.static_norm(static))
        )
        fused = F.silu(
            self.fusion(
                torch.cat(
                    (
                        card_representation,
                        history_representation,
                        static_representation,
                    ),
                    dim=1,
                )
            )
        )
        for block in self.blocks:
            fused = block(fused)
        return self.output_norm(fused)


class DualAttentionStateAdvantageNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 256, blocks: int = 1):
        super().__init__()
        self.backbone = DualAttentionStateBackbone(input_dim, hidden, blocks)
        self.output_layer = nn.Linear(hidden, NUM_ACTIONS)
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.backbone(x))


class DualAttentionStatePolicyNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 256, blocks: int = 1):
        super().__init__()
        self.backbone = DualAttentionStateBackbone(input_dim, hidden, blocks)
        self.output_layer = nn.Linear(hidden, NUM_ACTIONS)
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.backbone(x))


class DeepCFRResidualBlock(nn.Module):
    """The one-linear-layer ReLU residual block used by the Deep-CFR trunk."""

    def __init__(self, width: int):
        super().__init__()
        self.linear = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + F.relu(self.linear(x))


class DeepCFRBranchBackbone(nn.Module):
    """Order-invariant card groups plus the complete ordered public state.

    The original Deep CFR poker network used a compact, game-specific betting
    vector.  This engine has a substantially richer information state, so the
    non-card branch consumes every existing public feature rather than silently
    discarding stacks, legal actions, tournament context, or history events.
    """

    def __init__(self, input_dim: int, hidden: int = 64, blocks: int = 3):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden = int(hidden)
        self.history_steps, self.tournament_features = _structured_layout(input_dim)
        if self.hidden <= 0 or blocks <= 0:
            raise ValueError("hidden and blocks must be positive")

        embedding_dim = 64
        self.rank_embedding = nn.Embedding(13, embedding_dim)
        self.suit_embedding = nn.Embedding(4, embedding_dim)
        self.exact_card_embedding = nn.Embedding(52, embedding_dim)

        # Four poker groups: hole cards, flop, turn, river. Embeddings are
        # summed inside each group, making hole/flop order irrelevant.
        self.card_fc1 = nn.Linear(4 * embedding_dim, 192)
        self.card_fc2 = nn.Linear(192, 192)
        self.card_fc3 = nn.Linear(192, hidden)

        non_card_dim = self.input_dim - CARD_STATE_FEATURES
        self.state_norm = nn.LayerNorm(non_card_dim)
        self.state_fc = nn.Linear(non_card_dim, hidden)
        self.state_residual = DeepCFRResidualBlock(hidden)

        self.combine = nn.Linear(2 * hidden, hidden)
        self.trunk = nn.ModuleList(
            DeepCFRResidualBlock(hidden) for _ in range(int(blocks))
        )

    def _card_representation(self, cards: torch.Tensor) -> torch.Tensor:
        present = cards[:, :, 17] > 0.5
        ranks = cards[:, :, :13].argmax(dim=2)
        suits = cards[:, :, 13:17].argmax(dim=2)
        exact = suits * 13 + ranks
        embedded = (
            self.rank_embedding(ranks)
            + self.suit_embedding(suits)
            + self.exact_card_embedding(exact)
        )
        embedded = embedded * present.unsqueeze(2)
        grouped = torch.stack(
            (
                embedded[:, 0:2].sum(dim=1),
                embedded[:, 2:5].sum(dim=1),
                embedded[:, 5],
                embedded[:, 6],
            ),
            dim=1,
        ).flatten(1)
        h = F.relu(self.card_fc1(grouped))
        h = F.relu(self.card_fc2(h))
        return F.relu(self.card_fc3(h))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or int(x.shape[1]) != self.input_dim:
            raise ValueError(
                f"Deep CFR network expects [batch, {self.input_dim}] inputs"
            )
        batch = int(x.shape[0])
        cards = x[:, CARD_STATE_PREFIX_FEATURES:HISTORY_OFFSET].reshape(
            batch, CARD_TOKEN_COUNT, CARD_FEATURES
        )
        card_representation = self._card_representation(cards)
        public_state = torch.cat(
            (x[:, :CARD_STATE_PREFIX_FEATURES], x[:, HISTORY_OFFSET:]), dim=1
        )
        state_representation = F.relu(self.state_fc(self.state_norm(public_state)))
        state_representation = self.state_residual(state_representation)
        fused = F.relu(
            self.combine(torch.cat((card_representation, state_representation), dim=1))
        )
        for block in self.trunk:
            fused = block(fused)
        # Corresponds to the feature-normalization box in the paper diagram.
        return F.normalize(fused, p=2.0, dim=1, eps=1e-8)


class DeepCFRBranchNetwork(nn.Module):
    """Shared architecture for either advantages or unnormalized policy logits."""

    def __init__(self, input_dim: int, hidden: int = 64, blocks: int = 3):
        super().__init__()
        self.backbone = DeepCFRBranchBackbone(input_dim, hidden, blocks)
        self.output_layer = nn.Linear(hidden, NUM_ACTIONS)
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.backbone(x))


def _straight_window_counts(rank_present: torch.Tensor) -> torch.Tensor:
    """Return present-card counts for all ten five-rank straight windows."""

    windows = (
        (12, 0, 1, 2, 3),  # wheel: A2345
        (0, 1, 2, 3, 4),
        (1, 2, 3, 4, 5),
        (2, 3, 4, 5, 6),
        (3, 4, 5, 6, 7),
        (4, 5, 6, 7, 8),
        (5, 6, 7, 8, 9),
        (6, 7, 8, 9, 10),
        (7, 8, 9, 10, 11),
        (8, 9, 10, 11, 12),
    )
    index = torch.tensor(windows, dtype=torch.long, device=rank_present.device)
    return rank_present[:, index].sum(dim=2)


def poker_relational_features(
    cards: torch.Tensor,
    street_one_hot: torch.Tensor,
    *,
    use_native: bool = True,
) -> torch.Tensor:
    """Derive legal, deterministic made-hand/draw/board features.

    ``cards`` contains only the acting player's two private cards and public
    board cards. No opponent information or Monte-Carlo equity estimate is
    introduced. All work is batched tensor arithmetic so CPU traversal workers
    do not call the Python hand evaluator millions of times.
    """

    if cards.ndim != 3 or cards.shape[1:] != (CARD_TOKEN_COUNT, CARD_FEATURES):
        raise ValueError("cards must have shape [batch, 7, 18]")
    if use_native and cards.device.type == "cpu" and not cards.requires_grad:
        try:
            from three_player_native import poker_relational_features_native

            native = poker_relational_features_native(
                cards.detach().numpy(), street_one_hot.detach().numpy()
            )
            return torch.from_numpy(native)
        except (ImportError, AttributeError):
            # Keep older optional extensions and the pure-Python engine usable.
            pass
    present = cards[:, :, 17]
    ranks = cards[:, :, :13] * present.unsqueeze(2)
    suits = cards[:, :, 13:17] * present.unsqueeze(2)
    hole_ranks = ranks[:, :2]
    hole_suits = suits[:, :2]
    board_ranks = ranks[:, 2:]
    board_suits = suits[:, 2:]

    rank_counts = ranks.sum(dim=1)
    suit_counts = suits.sum(dim=1)
    board_rank_counts = board_ranks.sum(dim=1)
    board_suit_counts = board_suits.sum(dim=1)
    rank_present = (rank_counts > 0).to(cards.dtype)
    straight_counts = _straight_window_counts(rank_present)
    has_straight = (straight_counts >= 5).any(dim=1)
    has_flush = (suit_counts >= 5).any(dim=1)

    straight_flush = torch.zeros_like(has_straight)
    for suit in range(4):
        suited_ranks = (
            ranks * suits[:, :, suit : suit + 1]
        ).sum(dim=1).clamp(max=1.0)
        straight_flush |= (_straight_window_counts(suited_ranks) >= 5).any(dim=1)

    pair_count = (rank_counts >= 2).sum(dim=1)
    trip_count = (rank_counts >= 3).sum(dim=1)
    has_quads = (rank_counts >= 4).any(dim=1)
    has_full_house = (trip_count >= 2) | ((trip_count >= 1) & (pair_count >= 2))
    has_trips = trip_count >= 1
    has_two_pair = pair_count >= 2
    has_pair = pair_count >= 1
    category = torch.stack(
        (
            ~(straight_flush | has_quads | has_full_house | has_flush | has_straight | has_trips | has_two_pair | has_pair),
            has_pair & ~has_two_pair & ~has_trips,
            has_two_pair & ~has_trips,
            has_trips & ~has_full_house & ~has_straight & ~has_flush,
            has_straight & ~has_flush & ~has_full_house & ~has_quads,
            has_flush & ~has_full_house & ~has_quads & ~straight_flush,
            has_full_house & ~has_quads,
            has_quads & ~straight_flush,
            straight_flush,
        ),
        dim=1,
    ).to(cards.dtype)

    street_index = street_one_hot.argmax(dim=1)
    can_draw = street_index < 3
    straight_four = straight_counts == 4
    has_straight_draw = straight_four.any(dim=1) & can_draw & ~has_straight
    # An internal missing rank is a gutshot; a missing endpoint is open-ended.
    window_index = torch.tensor(
        (
            (12, 0, 1, 2, 3),
            (0, 1, 2, 3, 4),
            (1, 2, 3, 4, 5),
            (2, 3, 4, 5, 6),
            (3, 4, 5, 6, 7),
            (4, 5, 6, 7, 8),
            (5, 6, 7, 8, 9),
            (6, 7, 8, 9, 10),
            (7, 8, 9, 10, 11),
            (8, 9, 10, 11, 12),
        ),
        dtype=torch.long,
        device=cards.device,
    )
    window_presence = rank_present[:, window_index]
    missing = 1.0 - window_presence
    open_ended = (
        straight_four
        & ((missing[:, :, 0] > 0.5) | (missing[:, :, 4] > 0.5))
    ).any(dim=1) & can_draw & ~has_straight
    gutshot = (
        straight_four & (missing[:, :, 1:4].sum(dim=2) > 0.5)
    ).any(dim=1) & can_draw & ~has_straight
    flush_draw = (suit_counts == 4).any(dim=1) & can_draw & ~has_flush
    backdoor_flush = (
        (street_index == 1) & (suit_counts == 3).any(dim=1) & ~has_flush
    )

    hole_rank_index = hole_ranks.argmax(dim=2)
    hole_suit_index = hole_suits.argmax(dim=2)
    pocket_pair = hole_rank_index[:, 0] == hole_rank_index[:, 1]
    hole_suited = hole_suit_index[:, 0] == hole_suit_index[:, 1]
    raw_gap = (hole_rank_index[:, 0] - hole_rank_index[:, 1]).abs()
    ace_low_gap = torch.where(
        (hole_rank_index == 12).any(dim=1),
        torch.minimum(raw_gap, 13 - raw_gap),
        raw_gap,
    )
    gap_bucket = F.one_hot(ace_low_gap.clamp(max=4), num_classes=5).to(cards.dtype)
    board_present = board_rank_counts > 0
    hole_board_matches = board_present.gather(1, hole_rank_index).sum(dim=1)
    board_card_count = present[:, 2:].sum(dim=1)
    board_max_rank = torch.where(
        board_present,
        torch.arange(13, device=cards.device).unsqueeze(0),
        torch.full_like(board_rank_counts, -1.0),
    ).amax(dim=1)
    overcards = (hole_rank_index > board_max_rank.unsqueeze(1)).sum(dim=1)
    board_pair_count = (board_rank_counts >= 2).sum(dim=1)
    board_trip_count = (board_rank_counts >= 3).sum(dim=1)
    max_board_suit = board_suit_counts.amax(dim=1)

    scalar_values = (
            pocket_pair,
            hole_suited,
            hole_board_matches / 2.0,
            overcards / 2.0,
            has_straight_draw,
            open_ended,
            gutshot,
            flush_draw,
            backdoor_flush,
            board_pair_count > 0,
            board_trip_count > 0,
            (board_card_count >= 3) & (max_board_suit == board_card_count),
            (board_card_count >= 3) & (max_board_suit == 2),
            board_card_count / 5.0,
            pair_count.to(cards.dtype) / 3.0,
            trip_count.to(cards.dtype) / 2.0,
            has_flush,
            has_straight,
    )
    scalar = torch.stack(
        tuple(value.to(cards.dtype) for value in scalar_values), dim=1
    )
    features = torch.cat(
        (
            rank_counts / 4.0,
            suit_counts / 7.0,
            board_rank_counts / 3.0,
            board_suit_counts / 5.0,
            category,
            gap_bucket,
            scalar,
        ),
        dim=1,
    )
    if int(features.shape[1]) != POKER_RELATIONAL_FEATURES:
        raise RuntimeError("unexpected poker relational feature width")
    return features


class DeepCFRBranchV2Backbone(DeepCFRBranchBackbone):
    """Deep-CFR branch with explicit poker relations and width for this game."""

    def __init__(self, input_dim: int, hidden: int = 128, blocks: int = 3):
        super().__init__(input_dim, hidden, blocks)
        self.poker_feature_norm = nn.LayerNorm(POKER_RELATIONAL_FEATURES)
        self.poker_feature_fc = nn.Linear(POKER_RELATIONAL_FEATURES, hidden)
        self.combine = nn.Linear(3 * hidden, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or int(x.shape[1]) != self.input_dim:
            raise ValueError(
                f"Deep CFR v2 network expects [batch, {self.input_dim}] inputs"
            )
        batch = int(x.shape[0])
        cards = x[:, CARD_STATE_PREFIX_FEATURES:HISTORY_OFFSET].reshape(
            batch, CARD_TOKEN_COUNT, CARD_FEATURES
        )
        card_representation = self._card_representation(cards)
        public_state = torch.cat(
            (x[:, :CARD_STATE_PREFIX_FEATURES], x[:, HISTORY_OFFSET:]), dim=1
        )
        state_representation = self.state_residual(
            F.relu(self.state_fc(self.state_norm(public_state)))
        )
        poker_features = poker_relational_features(cards, x[:, :4])
        poker_representation = F.relu(
            self.poker_feature_fc(self.poker_feature_norm(poker_features))
        )
        fused = F.relu(
            self.combine(
                torch.cat(
                    (card_representation, poker_representation, state_representation),
                    dim=1,
                )
            )
        )
        for block in self.trunk:
            fused = block(fused)
        return F.normalize(fused, p=2.0, dim=1, eps=1e-8)


class DeepCFRBranchV2Network(nn.Module):
    """Shared backbone with independent preflop/flop/turn/river action heads."""

    def __init__(self, input_dim: int, hidden: int = 128, blocks: int = 3):
        super().__init__()
        self.backbone = DeepCFRBranchV2Backbone(input_dim, hidden, blocks)
        self.street_heads = nn.ModuleList(
            nn.Linear(hidden, NUM_ACTIONS) for _ in range(4)
        )
        for head in self.street_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        representation = self.backbone(x)
        outputs = torch.stack(
            [head(representation) for head in self.street_heads], dim=1
        )
        return (outputs * x[:, :4].unsqueeze(2)).sum(dim=1)


class DeepCFRBranchV3Backbone(DeepCFRBranchBackbone):
    """V2 poker relations plus an explicit ordered betting-history encoder."""

    def __init__(self, input_dim: int, hidden: int = 256, blocks: int = 4):
        super().__init__(input_dim, hidden, blocks)
        # V3 replaces the flat public-state branch created by the shared base.
        del self.state_norm
        del self.state_fc
        del self.state_residual
        del self.combine

        self.history_steps, self.tournament_features = _structured_layout(input_dim)
        self.history_token_dim = 64
        self.history_projection = nn.Linear(HISTORY_FEATURES, self.history_token_dim)
        self.history_positions = nn.Parameter(
            torch.zeros(1, self.history_steps + 1, self.history_token_dim)
        )
        self.history_summary_token = nn.Parameter(
            torch.zeros(1, 1, self.history_token_dim)
        )
        self.history_attention = nn.TransformerEncoderLayer(
            d_model=self.history_token_dim,
            nhead=4,
            dim_feedforward=2 * self.history_token_dim,
            dropout=0.05,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.history_memory = nn.GRU(
            self.history_token_dim, self.history_token_dim, batch_first=True
        )
        self.history_fusion = nn.Linear(2 * self.history_token_dim, hidden)

        static_dim = (
            CARD_STATE_PREFIX_FEATURES + NUM_ACTIONS + self.tournament_features
        )
        self.static_norm = nn.LayerNorm(static_dim)
        self.static_fc = nn.Linear(static_dim, hidden)
        self.static_residual = DeepCFRResidualBlock(hidden)

        self.poker_feature_norm = nn.LayerNorm(POKER_RELATIONAL_FEATURES)
        self.poker_feature_fc = nn.Linear(POKER_RELATIONAL_FEATURES, hidden)
        self.combine = nn.Linear(4 * hidden, hidden)

        nn.init.normal_(self.history_summary_token, std=0.02)
        nn.init.normal_(self.history_positions, std=0.02)

    def _history_representation(self, x: torch.Tensor) -> torch.Tensor:
        batch = int(x.shape[0])
        history_end = HISTORY_OFFSET + self.history_steps * HISTORY_FEATURES
        history = x[:, HISTORY_OFFSET:history_end].reshape(
            batch, self.history_steps, HISTORY_FEATURES
        )
        history_present = history.abs().sum(dim=2) > 0.0
        history_tokens = self.history_projection(history)
        lengths = history_present.sum(dim=1)
        maximum_length = int(lengths.max().item())
        if maximum_length:
            start = self.history_steps - lengths
            order = (
                torch.arange(self.history_steps, device=x.device).unsqueeze(0)
                + start.unsqueeze(1)
            ) % self.history_steps
            compact_history = history_tokens.gather(
                1,
                order.unsqueeze(2).expand(
                    -1, -1, self.history_token_dim
                ),
            )[:, :maximum_length]
            compact_positions = self.history_positions[:, 1:].expand(
                batch, -1, -1
            ).gather(
                1,
                order.unsqueeze(2).expand(
                    -1, -1, self.history_token_dim
                ),
            )[:, :maximum_length]
            compact_padding = (
                torch.arange(maximum_length, device=x.device).unsqueeze(0)
                >= lengths.unsqueeze(1)
            )
        else:
            compact_history = history_tokens[:, :0]
            compact_positions = self.history_positions[:, 1:1].expand(
                batch, 0, -1
            )
            compact_padding = torch.zeros(
                batch, 0, dtype=torch.bool, device=x.device
            )

        history_summary = self.history_summary_token.expand(batch, -1, -1)
        history_sequence = torch.cat((history_summary, compact_history), dim=1)
        history_sequence = history_sequence + torch.cat(
            (
                self.history_positions[:, :1].expand(batch, -1, -1),
                compact_positions,
            ),
            dim=1,
        )
        history_padding = torch.cat(
            (
                torch.zeros(batch, 1, dtype=torch.bool, device=x.device),
                compact_padding,
            ),
            dim=1,
        )
        attention_history = self.history_attention(
            history_sequence, src_key_padding_mask=history_padding
        )[:, 0]
        if maximum_length:
            memory_sequence, _ = self.history_memory(compact_history)
            final_index = (lengths - 1).clamp(min=0)
            memory_history = memory_sequence[
                torch.arange(batch, device=x.device), final_index
            ]
            memory_history = memory_history * (lengths > 0).unsqueeze(1)
        else:
            memory_history = history_tokens.new_zeros(
                batch, self.history_token_dim
            )
        return F.silu(
            self.history_fusion(
                torch.cat((attention_history, memory_history), dim=1)
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or int(x.shape[1]) != self.input_dim:
            raise ValueError(
                f"Deep CFR v3 network expects [batch, {self.input_dim}] inputs"
            )
        batch = int(x.shape[0])
        history_end = HISTORY_OFFSET + self.history_steps * HISTORY_FEATURES
        cards = x[:, CARD_STATE_PREFIX_FEATURES:HISTORY_OFFSET].reshape(
            batch, CARD_TOKEN_COUNT, CARD_FEATURES
        )
        card_representation = self._card_representation(cards)
        poker_representation = F.relu(
            self.poker_feature_fc(
                self.poker_feature_norm(
                    poker_relational_features(cards, x[:, :4])
                )
            )
        )
        static = torch.cat(
            (
                x[:, :CARD_STATE_PREFIX_FEATURES],
                x[:, history_end : history_end + NUM_ACTIONS],
                x[:, history_end + NUM_ACTIONS :],
            ),
            dim=1,
        )
        static_representation = self.static_residual(
            F.relu(self.static_fc(self.static_norm(static)))
        )
        history_representation = self._history_representation(x)
        fused = F.relu(
            self.combine(
                torch.cat(
                    (
                        card_representation,
                        poker_representation,
                        static_representation,
                        history_representation,
                    ),
                    dim=1,
                )
            )
        )
        for block in self.trunk:
            fused = block(fused)
        return F.normalize(fused, p=2.0, dim=1, eps=1e-8)


class DeepCFRBranchV3Network(nn.Module):
    """V3 backbone with independent preflop/flop/turn/river action heads."""

    def __init__(self, input_dim: int, hidden: int = 256, blocks: int = 4):
        super().__init__()
        self.backbone = DeepCFRBranchV3Backbone(input_dim, hidden, blocks)
        self.street_heads = nn.ModuleList(
            nn.Linear(hidden, NUM_ACTIONS) for _ in range(4)
        )
        for head in self.street_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        representation = self.backbone(x)
        outputs = torch.stack(
            [head(representation) for head in self.street_heads], dim=1
        )
        return (outputs * x[:, :4].unsqueeze(2)).sum(dim=1)


class AdvantageNetwork(nn.Module):
    """Outputs unconstrained cumulative-regret/advantage estimates."""

    def __init__(self, input_dim: int, hidden: int = 128, blocks: int = 2):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_layer = nn.Linear(input_dim, hidden)
        self.blocks = nn.ModuleList(ResidualBlock(hidden) for _ in range(blocks))
        self.output_norm = nn.LayerNorm(hidden)
        self.output_layer = nn.Linear(hidden, NUM_ACTIONS)
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.input_layer(self.input_norm(x)))
        for block in self.blocks:
            x = block(x)
        return self.output_layer(self.output_norm(x))


class PolicyNetwork(nn.Module):
    """Outputs raw logits; legality is applied outside the network."""

    def __init__(self, input_dim: int, hidden: int = 128, blocks: int = 2):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_layer = nn.Linear(input_dim, hidden)
        self.blocks = nn.ModuleList(ResidualBlock(hidden) for _ in range(blocks))
        self.output_norm = nn.LayerNorm(hidden)
        self.output_layer = nn.Linear(hidden, NUM_ACTIONS)
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.input_layer(self.input_norm(x)))
        for block in self.blocks:
            x = block(x)
        return self.output_layer(self.output_norm(x))


def build_advantage_network(
    architecture: str, input_dim: int, hidden: int, blocks: int
) -> nn.Module:
    if architecture == "residual_mlp":
        return AdvantageNetwork(input_dim, hidden, blocks)
    if architecture == "dual_attention_state":
        return DualAttentionStateAdvantageNetwork(input_dim, hidden, blocks)
    if architecture == "deep_cfr_branch":
        return DeepCFRBranchNetwork(input_dim, hidden, blocks)
    if architecture == "deep_cfr_branch_v2":
        return DeepCFRBranchV2Network(input_dim, hidden, blocks)
    if architecture == "deep_cfr_branch_v3":
        return DeepCFRBranchV3Network(input_dim, hidden, blocks)
    raise ValueError(f"unknown network architecture: {architecture!r}")


def build_policy_network(
    architecture: str, input_dim: int, hidden: int, blocks: int
) -> nn.Module:
    if architecture == "residual_mlp":
        return PolicyNetwork(input_dim, hidden, blocks)
    if architecture == "dual_attention_state":
        return DualAttentionStatePolicyNetwork(input_dim, hidden, blocks)
    if architecture == "deep_cfr_branch":
        return DeepCFRBranchNetwork(input_dim, hidden, blocks)
    if architecture == "deep_cfr_branch_v2":
        return DeepCFRBranchV2Network(input_dim, hidden, blocks)
    if architecture == "deep_cfr_branch_v3":
        return DeepCFRBranchV3Network(input_dim, hidden, blocks)
    raise ValueError(f"unknown network architecture: {architecture!r}")


def masked_softmax(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """Normalize logits over legal actions only."""
    if logits.shape != legal_mask.shape:
        raise ValueError("logits and legal_mask must have the same shape")
    if torch.any(legal_mask.sum(dim=-1) <= 0):
        raise ValueError("each policy row must contain at least one legal action")
    masked = logits.masked_fill(legal_mask <= 0, -1e9)
    return torch.softmax(masked, dim=-1)


__all__ = [
    "AdvantageNetwork",
    "PolicyNetwork",
    "DualAttentionStateAdvantageNetwork",
    "DualAttentionStatePolicyNetwork",
    "DeepCFRBranchBackbone",
    "DeepCFRBranchNetwork",
    "DeepCFRBranchV2Backbone",
    "DeepCFRBranchV2Network",
    "DeepCFRBranchV3Backbone",
    "DeepCFRBranchV3Network",
    "poker_relational_features",
    "NETWORK_ARCHITECTURES",
    "build_advantage_network",
    "build_policy_network",
    "DEFAULT_MAX_HISTORY",
    "HISTORY_FEATURES",
    "LEGACY_FIXED_FEATURES",
    "TOURNAMENT_FEATURE_NAMES",
    "TOURNAMENT_FEATURES",
    "encode_information_state",
    "information_state_size",
    "masked_softmax",
]
