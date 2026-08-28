"""Tkinter table for playing a three-seat Hold'em tournament.

Seat 0 is controlled by the human. Seat 1 runs a bounded real-time re-solver on
top of its average-policy network, while seat 2 uses the fixed scripted
tight-aggressive benchmark.
Stacks carry from one hand to the next, busted seats are skipped, and play
continues heads-up until only one player owns chips.
"""

from __future__ import annotations

import argparse
import inspect
import random
import sys
import threading
import tkinter as tk
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from tkinter import messagebox, ttk

import torch

from real_time_search import RealTimeResolver
from three_player_engine import (
    ACTION_NAMES,
    NUM_ACTIONS,
    STREET_NAMES,
    card_to_string,
)
from three_player_native import ThreePlayerHoldemEnv
from three_player_production import TightAggressiveOpponent
from three_player_models import (
    build_policy_network,
    encode_information_state,
    masked_softmax,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = (
    ROOT
    / "artifacts"
    / "downloaded_blueprints"
    / "policy_00008100.pt"
)
HUMAN_SEAT = 0
TAG_SEAT = 2
SEAT_NAMES = ("You", "Bot 1", "Bot 2 TAG")
STACK_EPSILON = 1e-9


class SnapshotPolicy:
    """Inference wrapper for policy snapshots or full training checkpoints."""

    def __init__(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"Policy checkpoint was not found:\n{path}")

        # Checkpoints are executable pickle data. This GUI only loads the local,
        # trusted artifact selected by the user.
        payload = torch.load(path, map_location="cpu", weights_only=False)
        is_snapshot = payload.get("kind") == "three_player_policy_snapshot"
        is_full_checkpoint = "config" in payload and "policy_nets" in payload
        if not (is_snapshot or is_full_checkpoint):
            raise ValueError(
                "The selected file is not a policy snapshot or full CFR checkpoint"
            )
        if int(payload.get("version", -1)) != 1:
            raise ValueError("Unsupported policy snapshot version")
        if tuple(payload.get("action_names", ())) != tuple(ACTION_NAMES):
            raise ValueError("Checkpoint action names do not match this poker engine")

        states = payload.get("policy_nets", [])
        if len(states) != 3:
            raise ValueError("The checkpoint must contain exactly three policy networks")

        self.path = path
        self.iteration = int(payload["iteration"])
        config = dict(payload.get("config", {}))
        self.input_dim = int(payload["input_dim"])
        self.max_history = int(payload.get("max_history", config.get("max_history", 32)))
        hidden = int(payload.get("hidden", config.get("hidden", 128)))
        blocks = int(payload.get("blocks", config.get("blocks", 2)))
        architecture = str(
            payload.get(
                "network_architecture",
                config.get("network_architecture", "residual_mlp"),
            )
        )
        self.environment = dict(payload.get("environment", {}))
        self.encoder_metadata = dict(payload.get("encoder", {}))
        self.tournament_features = payload.get(
            "include_tournament_features",
            payload.get(
                "tournament_features",
                config.get(
                    "include_tournament_features",
                    self.encoder_metadata.get(
                        "include_tournament_features",
                        self.encoder_metadata.get("tournament_features"),
                    ),
                ),
            ),
        )
        self.stack_size = float(self.environment.get("stack_size", 200.0))
        self.small_blind = float(self.environment.get("sb", 1.0))
        self.big_blind = float(self.environment.get("bb", 2.0))

        self.networks: list[torch.nn.Module] = []
        for state_dict in states:
            network = build_policy_network(
                architecture, self.input_dim, hidden, blocks
            )
            network.load_state_dict(state_dict)
            network.eval()
            self.networks.append(network)

    @torch.inference_mode()
    def probabilities(self, env: ThreePlayerHoldemEnv, state) -> torch.Tensor:
        if state.terminal or state.to_act is None:
            raise ValueError("Policy inference requires a live decision state")
        player = int(state.to_act)
        legal = env.legal_actions(state)
        encoded = self._encode_state(state, player, legal)
        mask = torch.zeros(NUM_ACTIONS, dtype=torch.float32)
        mask[legal] = 1.0
        logits = self.networks[player](encoded.unsqueeze(0))[0]
        return masked_softmax(logits, mask)

    def _encode_state(self, state, player: int, legal: list[int]) -> torch.Tensor:
        """Select the legacy or tournament encoder by checkpoint input width.

        ``policy_00000900.pt`` predates tournament context.  New snapshots can
        use the expanded encoder, so input dimension is the authoritative
        compatibility signal rather than the filename or training iteration.
        """

        signature = inspect.signature(encode_information_state)
        feature_parameter = next(
            (
                name
                for name in (
                    "include_tournament_features",
                    "tournament_features",
                )
                if name in signature.parameters
            ),
            None,
        )
        attempts: list[dict[str, bool]] = []

        if feature_parameter is not None:
            if self.tournament_features is not None:
                attempts.append(
                    {feature_parameter: bool(self.tournament_features)}
                )
            # Try both modes because old snapshots may not contain encoder
            # metadata and expanded snapshots may retain format version 1.
            attempts.extend(
                ({feature_parameter: False}, {feature_parameter: True})
            )
        attempts.append({})

        seen: set[tuple[tuple[str, bool], ...]] = set()
        dimensions: list[int] = []
        for kwargs in attempts:
            key = tuple(sorted(kwargs.items()))
            if key in seen:
                continue
            seen.add(key)
            encoded = encode_information_state(
                state,
                player,
                legal,
                self.stack_size,
                self.max_history,
                **kwargs,
            )
            dimensions.append(int(encoded.numel()))
            if encoded.numel() == self.input_dim:
                return encoded

        found = ", ".join(str(value) for value in sorted(set(dimensions)))
        raise ValueError(
            f"Available state encoder size(s) {found} do not match checkpoint "
            f"input size {self.input_dim}"
        )


def sample_action(probabilities: torch.Tensor, rng: random.Random) -> int:
    """Sample exactly from a nine-action policy, with a rounding fallback."""

    threshold = rng.random()
    cumulative = 0.0
    fallback = int(torch.argmax(probabilities).item())
    for action, probability in enumerate(probabilities.tolist()):
        if probability <= 0.0:
            continue
        fallback = action
        cumulative += float(probability)
        if threshold <= cumulative + 1e-12:
            return action
    return fallback


class PokerGUI:
    TABLE = "#146b3a"
    FELT_EDGE = "#0a4728"
    PANEL = "#17202a"
    TEXT = "#f4f6f7"
    GOLD = "#f4d03f"

    def __init__(
        self,
        root: tk.Tk,
        policy: SnapshotPolicy,
        *,
        seed: int | None,
        bot_delay: int,
        search_ms: int,
        search_rollouts: int,
        use_search: bool,
    ) -> None:
        self.root = root
        self.policy = policy
        self.tag_opponent = TightAggressiveOpponent()
        self.seed = seed
        self.rng = random.Random(seed)
        self.bot_delay = max(0, bot_delay)
        self.search_ms = int(search_ms)
        self.resolver = (
            RealTimeResolver(
                policy,
                self.tag_opponent,
                tag_seat=TAG_SEAT,
                time_budget_ms=search_ms,
                max_rollouts=search_rollouts,
                seed=None if seed is None else seed + 1_000_003,
            )
            if use_search
            else None
        )
        self.env = self._make_environment()
        self.state = None
        self.hand_number = 0
        self.tournament_stacks = [float(policy.stack_size)] * 3
        self.tournament_over = False
        self.cumulative = [0.0, 0.0, 0.0]
        self.bot_job: str | None = None
        self.search_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="poker-search"
        )
        self.search_future: Future | None = None
        self.search_state = None
        self.search_cancel: threading.Event | None = None
        self.action_buttons: list[ttk.Button] = []

        root.title(
            f"3-Player Poker — You vs policy_{policy.iteration:08d} and scripted TAG"
        )
        root.geometry("1120x780")
        root.minsize(940, 680)
        root.configure(bg=self.PANEL)
        root.protocol("WM_DELETE_WINDOW", self.close)

        self._build_widgets()
        self.new_tournament()

    def _make_environment(self) -> ThreePlayerHoldemEnv:
        return ThreePlayerHoldemEnv(
            starting_stack=self.policy.stack_size,
            small_blind=self.policy.small_blind,
            big_blind=self.policy.big_blind,
            seed=self.seed,
        )

    def _build_widgets(self) -> None:
        top = tk.Frame(self.root, bg=self.PANEL, padx=12, pady=8)
        top.pack(fill="x")
        self.status = tk.Label(
            top, text="", bg=self.PANEL, fg=self.TEXT, font=("Segoe UI", 12, "bold")
        )
        self.status.pack(side="left")
        self.model_label = tk.Label(
            top,
            text=(
                f"Model: {self.policy.path.name}  |  iteration {self.policy.iteration}  |  "
                f"blinds {self._chips(self.policy.small_blind)}/"
                f"{self._chips(self.policy.big_blind)}  |  "
                f"{f'{self.search_ms / 1000.0:.1f}s search' if self.resolver else 'blueprint only'}"
            ),
            bg=self.PANEL,
            fg="#bdc3c7",
            font=("Segoe UI", 9),
        )
        self.model_label.pack(side="right")

        middle = tk.Frame(self.root, bg=self.PANEL)
        middle.pack(fill="both", expand=True, padx=12)
        self.canvas = tk.Canvas(
            middle, bg=self.FELT_EDGE, highlightthickness=0, width=790, height=550
        )
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _event: self.draw_table())

        log_frame = tk.Frame(middle, bg=self.PANEL, width=310)
        log_frame.pack(side="right", fill="y", padx=(12, 0))
        log_frame.pack_propagate(False)
        tk.Label(
            log_frame,
            text="Hand log / bot policy",
            bg=self.PANEL,
            fg=self.TEXT,
            font=("Segoe UI", 11, "bold"),
        ).pack(anchor="w", pady=(0, 5))
        self.log = tk.Text(
            log_frame,
            width=39,
            bg="#0e151b",
            fg="#d5dbdb",
            insertbackground="white",
            relief="flat",
            wrap="word",
            font=("Consolas", 9),
            state="disabled",
        )
        scrollbar = ttk.Scrollbar(log_frame, command=self.log.yview)
        scrollbar.pack(side="right", fill="y")
        self.log.configure(yscrollcommand=scrollbar.set)
        self.log.pack(side="left", fill="both", expand=True)

        controls = tk.Frame(self.root, bg=self.PANEL, padx=12, pady=10)
        controls.pack(fill="x")
        self.prompt = tk.Label(
            controls,
            text="",
            bg=self.PANEL,
            fg=self.GOLD,
            font=("Segoe UI", 11, "bold"),
        )
        self.prompt.pack(anchor="w", pady=(0, 6))
        self.actions_frame = tk.Frame(controls, bg=self.PANEL)
        self.actions_frame.pack(side="left", fill="x", expand=True)
        self.next_hand_button = ttk.Button(
            controls, text="Next hand", command=self.new_hand, state="disabled"
        )
        self.next_hand_button.pack(side="right", padx=(8, 0))
        self.new_tournament_button = ttk.Button(
            controls, text="New tournament", command=self.new_tournament
        )
        self.new_tournament_button.pack(side="right", padx=(12, 0))

    def new_tournament(self) -> None:
        """Reset all bankrolls and immediately deal hand one."""

        self._cancel_bot_job()
        self._disable_actions()
        self.env = self._make_environment()
        self.state = None
        self.hand_number = 0
        self.tournament_stacks = [float(self.policy.stack_size)] * 3
        self.tournament_over = False
        self.cumulative = [0.0, 0.0, 0.0]
        self.new_hand()

    def new_hand(self) -> None:
        self._cancel_bot_job()
        survivors = self._survivors()
        if len(survivors) <= 1:
            self.tournament_over = True
            self.next_hand_button.configure(state="disabled")
            if survivors:
                self.prompt.configure(
                    text=f"Tournament over — {SEAT_NAMES[survivors[0]]} won"
                )
            return
        self.hand_number += 1
        self.state = self.env.new_hand(stacks=self.tournament_stacks)
        self.next_hand_button.configure(state="disabled")
        self._clear_log()
        role = self._role(HUMAN_SEAT)
        format_name = "heads-up" if len(survivors) == 2 else "three-handed"
        self._append_log(
            f"HAND {self.hand_number} — {format_name}; You are {role}\n"
            f"Bankrolls {self._stack_summary()}\n"
            f"Blinds "
            f"{self._chips(self.env.sb)}/{self._chips(self.env.bb)}\n\n"
        )
        self.refresh()
        self._continue_game()

    def _continue_game(self) -> None:
        if self.state.terminal:
            self._finish_hand()
        elif self.state.to_act == HUMAN_SEAT:
            self._show_human_actions()
        else:
            self._disable_actions()
            actor = int(self.state.to_act)
            self.prompt.configure(text=f"{SEAT_NAMES[actor]} is thinking…")
            self.bot_job = self.root.after(self.bot_delay, self._bot_turn)

    def _bot_turn(self) -> None:
        self.bot_job = None
        if self.state.terminal or self.state.to_act == HUMAN_SEAT:
            self._continue_game()
            return
        actor = int(self.state.to_act)
        if actor == TAG_SEAT:
            probabilities = self.tag_opponent.probabilities(
                self.env, self.state, actor
            )
            policy_name = "scripted TAG"
            action = sample_action(probabilities, self.rng)
        else:
            if self.resolver is None:
                probabilities = self.policy.probabilities(self.env, self.state)
                policy_name = "trained blueprint (argmax)"
                action = int(torch.argmax(probabilities).item())
            else:
                self._start_policy_search()
                return
        legal = self.env.legal_actions(self.state)
        mix = ", ".join(
            f"{ACTION_NAMES[index]} {100.0 * float(probabilities[index]):.1f}%"
            for index in legal
        )
        self._append_log(f"{SEAT_NAMES[actor]} {policy_name}: {mix}\n")
        self._apply_action(action)

    def _start_policy_search(self) -> None:
        search_state = self.state
        cancel_event = threading.Event()
        self.search_state = search_state
        self.search_cancel = cancel_event
        self.prompt.configure(
            text=(
                f"{SEAT_NAMES[int(search_state.to_act)]} is searching "
                f"(up to {self.search_ms / 1000.0:.1f}s)…"
            )
        )
        self.search_future = self.search_executor.submit(
            self.resolver.resolve,
            self.env,
            search_state,
            cancel_event,
        )
        self.bot_job = self.root.after(50, self._poll_policy_search)

    def _poll_policy_search(self) -> None:
        self.bot_job = None
        future = self.search_future
        search_state = self.search_state
        if future is None:
            return
        if not future.done():
            self.bot_job = self.root.after(50, self._poll_policy_search)
            return

        self.search_future = None
        self.search_state = None
        self.search_cancel = None
        if self.state is not search_state or self.state.terminal:
            return

        actor = int(self.state.to_act)
        try:
            result = future.result()
            probabilities = result.probabilities
            action = result.action
            policy_name = (
                f"real-time resolve: {result.rollouts} rollouts, "
                f"{result.elapsed_ms:.0f} ms"
            )
            values = ", ".join(
                f"{ACTION_NAMES[index]} {value:+.1f}"
                for index, value in result.action_values.items()
            )
            self._append_log(f"Search EV estimates (chips): {values}\n")
        except Exception as exc:
            probabilities = self.policy.probabilities(self.env, self.state)
            policy_name = f"search fallback ({exc})"
            action = sample_action(probabilities, self.rng)

        legal = self.env.legal_actions(self.state)
        mix = ", ".join(
            f"{ACTION_NAMES[index]} {100.0 * float(probabilities[index]):.1f}%"
            for index in legal
        )
        self._append_log(f"{SEAT_NAMES[actor]} {policy_name}: {mix}\n")
        self._apply_action(action)

    def _show_human_actions(self) -> None:
        self._disable_actions()
        legal = self.env.legal_actions(self.state)
        to_call = self.env.amount_to_call(self.state, HUMAN_SEAT)
        if to_call > 0:
            self.prompt.configure(text=f"Your turn — {self._chips(to_call)} to call")
        else:
            self.prompt.configure(text="Your turn — you may check or bet")

        for action in legal:
            label = self._action_label(action)
            button = ttk.Button(
                self.actions_frame,
                text=label,
                command=lambda selected=action: self._human_action(selected),
            )
            button.pack(side="left", padx=(0, 6), pady=2)
            self.action_buttons.append(button)

    def _action_label(self, action: int) -> str:
        readable = ACTION_NAMES[action].replace("_", " ").title()
        if action >= 3:
            target = self.env.action_target(self.state, action)
            payment = target - self.state.street_contrib[HUMAN_SEAT]
            return f"{readable}\n+{self._chips(payment)}"
        if ACTION_NAMES[action] == "call":
            return f"Call\n{self._chips(self.env.amount_to_call(self.state))}"
        return readable

    def _human_action(self, action: int) -> None:
        if self.state.terminal or self.state.to_act != HUMAN_SEAT:
            return
        if action not in self.env.legal_actions(self.state):
            return
        self._disable_actions()
        self._apply_action(action)

    def _apply_action(self, action: int) -> None:
        actor = int(self.state.to_act)
        before = self.state
        target = self.env.action_target(before, action)
        payment = max(0.0, target - before.street_contrib[actor])
        self.state = self.env.step(before, action)
        detail = f" (+{self._chips(payment)})" if payment > 0 else ""
        self._append_log(
            f"{SEAT_NAMES[actor]}: {ACTION_NAMES[action].replace('_', ' ')}{detail}\n\n"
        )
        self.refresh()
        self._continue_game()

    def _finish_hand(self) -> None:
        self._disable_actions()
        for seat, payoff in enumerate(self.state.payoffs):
            self.cumulative[seat] += float(payoff)
        self.tournament_stacks = [
            0.0 if float(stack) <= STACK_EPSILON else float(stack)
            for stack in self.state.stacks
        ]
        winners = ", ".join(SEAT_NAMES[seat] for seat in self.state.winners)
        human_payoff = float(self.state.payoffs[HUMAN_SEAT])
        result = "won" if human_payoff > 0 else "lost" if human_payoff < 0 else "broke even"
        self._append_log("SHOWDOWN / RESULT\n")
        for seat in range(3):
            cards = " ".join(card_to_string(card) for card in self.state.hole[seat])
            cards_text = f"[{cards}]" if cards else "[OUT]"
            self._append_log(
                f"{SEAT_NAMES[seat]} {cards_text}: "
                f"{self._signed(self.state.payoffs[seat])}; bankroll "
                f"{self._chips(self.tournament_stacks[seat])}\n"
            )
        self._append_log(f"Winner(s): {winners}\n")
        self._append_log(f"Next-hand bankrolls: {self._stack_summary()}\n")

        survivors = self._survivors()
        if len(survivors) == 1:
            tournament_winner = survivors[0]
            self.tournament_over = True
            self.prompt.configure(
                text=(
                    f"Tournament over — {SEAT_NAMES[tournament_winner]} won with "
                    f"{self._chips(self.tournament_stacks[tournament_winner])} chips"
                )
            )
            self._append_log(
                f"\nTOURNAMENT WINNER: {SEAT_NAMES[tournament_winner]}\n"
            )
            self.next_hand_button.configure(state="disabled")
        else:
            if self.tournament_stacks[HUMAN_SEAT] <= STACK_EPSILON:
                prompt = f"Hand over — {winners} won. You are OUT; two bots remain"
            else:
                prompt = (
                    f"Hand over — {winners} won. You {result} "
                    f"{self._signed(human_payoff)}"
                )
            self.prompt.configure(text=prompt)
            self.next_hand_button.configure(state="normal")
        self.refresh()

    def refresh(self) -> None:
        if self.state is None:
            return
        street = STREET_NAMES[self.state.street].title()
        pot = float(self.state.pot)
        survivors = len(self._survivors()) if self.state.terminal else sum(
            float(stack) > STACK_EPSILON for stack in self.state.initial_stacks
        )
        table_format = "heads-up" if survivors == 2 else "three-handed"
        if self.tournament_over:
            table_format = "finished"
        self.status.configure(
            text=(
                f"Tournament hand {self.hand_number}  |  {table_format}  |  "
                f"{street}  |  Pot {self._chips(pot)}  |  {survivors} remaining"
            )
        )
        self.draw_table()

    def draw_table(self) -> None:
        if self.state is None:
            return
        canvas = self.canvas
        canvas.delete("all")
        width = max(canvas.winfo_width(), 650)
        height = max(canvas.winfo_height(), 480)
        canvas.create_oval(
            45, 35, width - 45, height - 35, fill=self.TABLE, outline="#c8a951", width=5
        )

        board_y = height * 0.47
        board_width = 5 * 64 + 4 * 8
        board_x = width / 2 - board_width / 2
        for index in range(5):
            card = self.state.board[index] if index < len(self.state.board) else None
            self._draw_card(board_x + index * 72, board_y, card)
        live_pot = float(self.state.pot)
        canvas.create_text(
            width / 2,
            board_y - 28,
            text=f"POT  {self._chips(live_pot)}",
            fill=self.GOLD,
            font=("Segoe UI", 13, "bold"),
        )

        positions = (
            (width / 2, height - 105),
            (155, 120),
            (width - 155, 120),
        )
        for seat, (x, y) in enumerate(positions):
            self._draw_player(seat, x, y)

        score = "TOURNAMENT CHIPS: " + "   |   ".join(
            (
                f"{SEAT_NAMES[seat]} OUT"
                if self._seat_out(seat)
                else f"{SEAT_NAMES[seat]} {self._chips(self._display_chips(seat))}"
            )
            for seat in range(3)
        )
        canvas.create_text(
            width / 2,
            16,
            text=score,
            fill=self.TEXT,
            font=("Segoe UI", 10, "bold"),
        )

    def _draw_player(self, seat: int, x: float, y: float) -> None:
        is_out = self._seat_out(seat)
        active = not is_out and not self.state.terminal and self.state.to_act == seat
        outline = self.GOLD if active else "#8f9aa3"
        fill = (
            "#252a2d"
            if is_out
            else "#253746" if not self.state.folded[seat] else "#3d4449"
        )
        self.canvas.create_rectangle(
            x - 118, y - 48, x + 118, y + 60, fill=fill, outline=outline, width=3
        )
        markers = []
        if is_out:
            markers.append("OUT")
        if not is_out and seat == self.state.button:
            markers.append("BTN")
        if not is_out and seat == self.state.sb_player:
            markers.append("SB")
        if not is_out and seat == self.state.bb_player:
            markers.append("BB")
        if not is_out and self.state.folded[seat]:
            markers.append("FOLDED")
        elif not is_out and self.state.all_in[seat]:
            markers.append("ALL-IN")
        self.canvas.create_text(
            x,
            y - 30,
            text=f"{SEAT_NAMES[seat]}  {' / '.join(markers)}",
            fill=self.TEXT,
            font=("Segoe UI", 10, "bold"),
        )
        self.canvas.create_text(
            x,
            y - 8,
            text=f"Stack {self._chips(self.state.stacks[seat])}  •  In {self._chips(self.state.total_contrib[seat])}",
            fill="#d5dbdb",
            font=("Segoe UI", 9),
        )
        reveal = seat == HUMAN_SEAT or self.state.terminal
        hole_cards = self.state.hole[seat]
        for offset in range(2):
            if is_out or offset >= len(hole_cards):
                card = None
            else:
                card = hole_cards[offset] if reveal else "hidden"
            self._draw_card(x - 45 + offset * 50, y + 13, card, small=True)

    def _draw_card(
        self, x: float, y: float, card: int | str | None, *, small: bool = False
    ) -> None:
        width, height = (42, 38) if small else (64, 82)
        if card == "hidden":
            self.canvas.create_rectangle(
                x, y, x + width, y + height, fill="#315b8a", outline="white", width=2
            )
            self.canvas.create_text(
                x + width / 2, y + height / 2, text="◆", fill="#b9d6f2", font=("Segoe UI", 16)
            )
            return
        if card is None:
            self.canvas.create_rectangle(
                x, y, x + width, y + height, fill="#1a5837", outline="#5d8d72", width=1
            )
            return
        compact = card_to_string(card)
        suit = {"c": "♣", "d": "♦", "h": "♥", "s": "♠"}[compact[1]]
        color = "#c62828" if compact[1] in "dh" else "#111111"
        self.canvas.create_rectangle(
            x, y, x + width, y + height, fill="white", outline="#d0d3d4", width=2
        )
        self.canvas.create_text(
            x + width / 2,
            y + height / 2,
            text=compact[0] + suit,
            fill=color,
            font=("Segoe UI", 12 if small else 20, "bold"),
        )

    def _role(self, seat: int) -> str:
        if self._seat_out(seat):
            return "OUT"
        roles = []
        if seat == self.state.button:
            roles.append("BTN")
        if seat == self.state.sb_player:
            roles.append("SB")
        if seat == self.state.bb_player:
            roles.append("BB")
        return " / ".join(roles) or "live seat"

    def _survivors(self) -> list[int]:
        return [
            seat
            for seat, stack in enumerate(self.tournament_stacks)
            if float(stack) > STACK_EPSILON
        ]

    def _seat_out(self, seat: int) -> bool:
        if self.state is None:
            return self.tournament_stacks[seat] <= STACK_EPSILON
        # A zero stack during a hand is merely all-in. A seat is OUT only if it
        # started the hand busted, or if the completed hand left it busted.
        if self.state.terminal:
            return float(self.state.stacks[seat]) <= STACK_EPSILON
        alive = getattr(self.state, "alive", None)
        if alive is not None:
            return not bool(alive[seat])
        return float(self.state.initial_stacks[seat]) <= STACK_EPSILON

    def _display_chips(self, seat: int) -> float:
        if self.state is None:
            return float(self.tournament_stacks[seat])
        if self.state.terminal:
            return float(self.state.stacks[seat])
        # Chips already committed to this hand remain visible in the "In"
        # field; include them in the tournament total shown above the table.
        return float(self.state.stacks[seat]) + float(self.state.total_contrib[seat])

    def _stack_summary(self) -> str:
        return " | ".join(
            (
                f"{SEAT_NAMES[seat]} OUT"
                if stack <= STACK_EPSILON
                else f"{SEAT_NAMES[seat]} {self._chips(stack)}"
            )
            for seat, stack in enumerate(self.tournament_stacks)
        )

    def _cancel_bot_job(self) -> None:
        if self.bot_job is not None:
            self.root.after_cancel(self.bot_job)
            self.bot_job = None
        if self.search_cancel is not None:
            self.search_cancel.set()
        if self.search_future is not None:
            self.search_future.cancel()
        self.search_future = None
        self.search_state = None
        self.search_cancel = None

    def _disable_actions(self) -> None:
        for button in self.action_buttons:
            button.destroy()
        self.action_buttons.clear()

    def _append_log(self, message: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", message)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    @staticmethod
    def _chips(value: float) -> str:
        return f"{value:.0f}" if abs(value - round(value)) < 1e-9 else f"{value:.2f}"

    def _signed(self, value: float) -> str:
        return f"{value:+.2f}" if abs(value - round(value)) >= 1e-9 else f"{value:+.0f}"

    def close(self) -> None:
        self._cancel_bot_job()
        self.search_executor.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play 3-player poker against one trained policy and one scripted TAG bot"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="policy snapshot or full CFR checkpoint",
    )
    parser.add_argument("--seed", type=int, default=None, help="optional reproducible deal seed")
    parser.add_argument(
        "--bot-delay", type=int, default=150, help="milliseconds between bot actions"
    )
    parser.add_argument(
        "--search-ms",
        type=int,
        default=7000,
        help="maximum real-time search budget per policy decision (default: 7000)",
    )
    parser.add_argument(
        "--search-rollouts",
        type=int,
        default=150000,
        help="hard rollout cap per policy decision (default: 150000)",
    )
    parser.add_argument(
        "--no-search",
        action="store_true",
        help="disable re-solving and play the raw checkpoint policy",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        policy = SnapshotPolicy(args.checkpoint.resolve())
    except Exception as exc:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Could not load poker bot", str(exc))
        root.destroy()
        print(f"Could not load poker bot: {exc}", file=sys.stderr)
        return 1

    root = tk.Tk()
    PokerGUI(
        root,
        policy,
        seed=args.seed,
        bot_delay=args.bot_delay,
        search_ms=args.search_ms,
        search_rollouts=args.search_rollouts,
        use_search=not args.no_search,
    )
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
