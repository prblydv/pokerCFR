"""Manual two-seat GUI for validating the exact heads-up poker engine.

Both seats are controlled by the user and both hole-card pairs are visible.
The GUI intentionally offers both the finite ten-slot blueprint actions and an
exact integer ``raise_to`` input.  This makes it useful for exercising off-tree
actions without changing the policy network's permanent output space.
"""

from __future__ import annotations

import argparse
import math
import sys
import tkinter as tk
from tkinter import messagebox, ttk

from heads_up_engine import (
    ACTION_NAMES,
    NUM_ACTIONS,
    STREET_NAMES,
    HeadsUpHoldemEnv,
    card_to_string,
)


SEAT_NAMES = ("Player 0", "Player 1")
EPSILON = 1e-9


def format_chips(value: float) -> str:
    value = float(value)
    if abs(value - round(value)) <= EPSILON:
        return str(int(round(value)))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _event_attr(event, *names: str, default=None):
    if isinstance(event, dict):
        for name in names:
            if name in event:
                return event[name]
    else:
        for name in names:
            if hasattr(event, name):
                return getattr(event, name)
    return default


def event_summary(event) -> str:
    """Return one exact, semantic history line without bucket translation."""

    street = int(_event_attr(event, "street", default=0))
    player = int(_event_attr(event, "player", "actor", default=0))
    kind = str(
        _event_attr(
            event,
            "kind",
            "semantic_action",
            "action_name",
            "name",
            default="action",
        )
    ).replace("_", " ")
    amount = float(
        _event_attr(event, "amount", "amount_added", "payment", default=0.0)
    )
    target = float(
        _event_attr(
            event,
            "contribution_after",
            "target",
            "raise_to",
            default=amount,
        )
    )
    before = float(_event_attr(event, "current_bet_before", default=0.0))
    after = float(_event_attr(event, "current_bet_after", default=before))
    pot_after = float(_event_attr(event, "pot_after", default=amount))
    full_raise = bool(
        _event_attr(event, "full_raise", "is_full_raise", default=False)
    )
    full = " full raise" if full_raise else ""
    return (
        f"{STREET_NAMES[street]:8s}  P{player} {kind:<18s} "
        f"+{format_chips(amount):>6s}  to {format_chips(target):>6s}  "
        f"bet {format_chips(before)}->{format_chips(after)}  "
        f"pot {format_chips(pot_after)}{full}"
    )


def fixed_action_label(env: HeadsUpHoldemEnv, state, action: int) -> str:
    """Label a finite policy slot with the exact effect the engine will apply."""

    if action < 0 or action >= NUM_ACTIONS:
        raise ValueError(f"action must be in 0..{NUM_ACTIONS - 1}")
    actor = state.to_act
    if actor is None:
        return ACTION_NAMES[action].replace("_", " ").title()
    target = float(env.action_target(state, action))
    contribution = float(state.street_contrib[int(actor)])
    payment = max(0.0, target - contribution)
    readable = ACTION_NAMES[action].replace("_", " ").title()
    if payment <= EPSILON:
        return readable
    return (
        f"{readable}\n"
        f"to {format_chips(target)}  (+{format_chips(payment)})"
    )


def state_facts(env: HeadsUpHoldemEnv, state) -> dict[str, str]:
    """GUI-neutral exact values used by the diagnostics panel and tests."""

    actor = state.to_act
    minimum_increment = float(
        getattr(
            state,
            "min_raise",
            getattr(state, "minimum_raise", getattr(state, "min_raise_increment", 0.0)),
        )
    )
    current_bet = float(state.current_bet)
    if actor is None:
        to_call = 0.0
        max_raise_to = 0.0
        actor_text = "-"
    else:
        actor = int(actor)
        to_call = float(env.amount_to_call(state, actor))
        max_raise_to = float(state.street_contrib[actor]) + float(state.stacks[actor])
        actor_text = f"P{actor}"
    return {
        "street": STREET_NAMES[int(state.street)],
        "actor": actor_text,
        "pot": format_chips(state.pot),
        "current_bet": format_chips(current_bet),
        "minimum_raise_increment": format_chips(minimum_increment),
        "minimum_raise_to": format_chips(current_bet + minimum_increment),
        "to_call": format_chips(to_call),
        "maximum_raise_to": format_chips(max_raise_to),
    }


class HeadsUpManualGUI:
    TABLE = "#146b3a"
    FELT_EDGE = "#0a4728"
    PANEL = "#17202a"
    TEXT = "#f4f6f7"
    MUTED = "#bdc3c7"
    GOLD = "#f4d03f"

    def __init__(
        self,
        root: tk.Tk,
        *,
        starting_stack: int,
        small_blind: int,
        big_blind: int,
        seed: int | None,
        first_button: int,
    ) -> None:
        self.root = root
        self.starting_stack = int(starting_stack)
        self.small_blind = int(small_blind)
        self.big_blind = int(big_blind)
        self.seed = seed
        self.first_button = int(first_button)
        self.env = self._new_environment()
        self.state = None
        self.hand_number = 0
        self.session_stacks = [self.starting_stack, self.starting_stack]
        self.next_button = self.first_button
        self.action_buttons: list[ttk.Button] = []
        self.exact_raise_var = tk.StringVar()

        root.title("Heads-Up Hold'em - exact engine manual test")
        root.geometry("1180x820")
        root.minsize(980, 700)
        root.configure(bg=self.PANEL)

        self._build_widgets()
        self.reset_match()

    def _new_environment(self) -> HeadsUpHoldemEnv:
        return HeadsUpHoldemEnv(
            starting_stack=self.starting_stack,
            small_blind=self.small_blind,
            big_blind=self.big_blind,
            seed=self.seed,
        )

    def _build_widgets(self) -> None:
        top = tk.Frame(self.root, bg=self.PANEL, padx=12, pady=8)
        top.pack(fill="x")
        self.status = tk.Label(
            top,
            text="",
            bg=self.PANEL,
            fg=self.TEXT,
            font=("Segoe UI", 12, "bold"),
        )
        self.status.pack(anchor="w")
        self.diagnostics = tk.Label(
            top,
            text="",
            bg=self.PANEL,
            fg=self.MUTED,
            justify="left",
            font=("Consolas", 10),
        )
        self.diagnostics.pack(anchor="w", pady=(4, 0))

        middle = tk.Frame(self.root, bg=self.PANEL)
        middle.pack(fill="both", expand=True, padx=12)
        self.canvas = tk.Canvas(
            middle,
            bg=self.FELT_EDGE,
            highlightthickness=0,
            width=780,
            height=510,
        )
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _event: self.draw_table())

        history_panel = tk.Frame(middle, bg=self.PANEL, width=405)
        history_panel.pack(side="right", fill="y", padx=(12, 0))
        history_panel.pack_propagate(False)
        tk.Label(
            history_panel,
            text="Exact public action history",
            bg=self.PANEL,
            fg=self.TEXT,
            font=("Segoe UI", 11, "bold"),
        ).pack(anchor="w", pady=(0, 5))
        self.history_text = tk.Text(
            history_panel,
            width=57,
            bg="#0e151b",
            fg="#d5dbdb",
            insertbackground="white",
            relief="flat",
            wrap="none",
            font=("Consolas", 9),
            state="disabled",
        )
        y_scroll = ttk.Scrollbar(history_panel, command=self.history_text.yview)
        y_scroll.pack(side="right", fill="y")
        x_scroll = ttk.Scrollbar(
            history_panel,
            orient="horizontal",
            command=self.history_text.xview,
        )
        x_scroll.pack(side="bottom", fill="x")
        self.history_text.configure(
            yscrollcommand=y_scroll.set,
            xscrollcommand=x_scroll.set,
        )
        self.history_text.pack(side="left", fill="both", expand=True)

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
        self.actions_frame.pack(fill="x")

        exact_row = tk.Frame(controls, bg=self.PANEL)
        exact_row.pack(fill="x", pady=(8, 0))
        tk.Label(
            exact_row,
            text="Arbitrary integer raise-to:",
            bg=self.PANEL,
            fg=self.TEXT,
        ).pack(side="left")
        self.exact_raise_entry = ttk.Entry(
            exact_row,
            textvariable=self.exact_raise_var,
            width=12,
        )
        self.exact_raise_entry.pack(side="left", padx=(6, 6))
        self.exact_raise_button = ttk.Button(
            exact_row,
            text="Apply exact raise",
            command=self.apply_exact_raise,
        )
        self.exact_raise_button.pack(side="left")
        self.exact_range = tk.Label(
            exact_row,
            text="",
            bg=self.PANEL,
            fg=self.MUTED,
        )
        self.exact_range.pack(side="left", padx=(10, 0))
        self.next_hand_button = ttk.Button(
            exact_row,
            text="Next hand (rotate button)",
            command=self.deal_hand,
            state="disabled",
        )
        self.next_hand_button.pack(side="right")
        ttk.Button(
            exact_row,
            text="Reset match",
            command=self.reset_match,
        ).pack(side="right", padx=(0, 8))

    def reset_match(self) -> None:
        self.env = self._new_environment()
        self.state = None
        self.hand_number = 0
        self.session_stacks = [self.starting_stack, self.starting_stack]
        self.next_button = self.first_button
        self.deal_hand()

    def deal_hand(self) -> None:
        if self.state is not None and not self.state.terminal:
            return
        if min(self.session_stacks) <= EPSILON:
            messagebox.showinfo(
                "Match complete",
                "One player has no chips. Use Reset match to start again.",
            )
            return
        button = self.next_button
        self.next_button = 1 - button
        self.hand_number += 1
        self.state = self.env.new_hand(button=button, stacks=self.session_stacks)
        if self.state.terminal:
            self.session_stacks = [
                0 if int(value) <= 0 else int(value)
                for value in self.state.stacks
            ]
        self.next_hand_button.configure(state="disabled")
        self.refresh()

    def apply_fixed_action(self, action: int) -> None:
        if self.state is None or self.state.terminal:
            return
        if action not in self.env.legal_actions(self.state):
            return
        try:
            self.state = self.env.step(self.state, action)
        except Exception as exc:
            messagebox.showerror("Engine rejected action", str(exc))
            return
        self._after_action()

    def apply_exact_raise(self) -> None:
        if self.state is None or self.state.terminal:
            return
        raw = self.exact_raise_var.get().strip()
        try:
            # Explicitly reject decimal/exponential inputs: this control is for
            # real poker-room integer chip targets.
            raise_to = int(raw, 10)
            if str(raise_to) != raw and str(raise_to) != raw.lstrip("+"):
                raise ValueError
            if raise_to < 0:
                raise ValueError
        except ValueError:
            messagebox.showerror(
                "Invalid raise target",
                "Enter one nonnegative integer total street contribution.",
            )
            return
        try:
            self.state = self.env.step_exact(
                self.state,
                "raise_to",
                raise_to=raise_to,
            )
        except Exception as exc:
            messagebox.showerror("Illegal exact raise", str(exc))
            return
        self._after_action()

    def _after_action(self) -> None:
        if self.state.terminal:
            self.session_stacks = [
                0 if int(value) <= 0 else int(value)
                for value in self.state.stacks
            ]
        self.refresh()

    def refresh(self) -> None:
        if self.state is None:
            return
        facts = state_facts(self.env, self.state)
        button = int(self.state.button)
        self.status.configure(
            text=(
                f"Hand {self.hand_number}  |  {facts['street'].title()}  |  "
                f"Button/SB P{button}  |  BB P{1 - button}  |  "
                f"Actor {facts['actor']}"
            )
        )
        self.diagnostics.configure(
            text=(
                f"pot={facts['pot']}   current_bet={facts['current_bet']}   "
                f"min_raise_increment={facts['minimum_raise_increment']}   "
                f"min_raise_to={facts['minimum_raise_to']}\n"
                f"to_call={facts['to_call']}   max_raise_to={facts['maximum_raise_to']}   "
                f"stacks=P0 {format_chips(self.state.stacks[0])}, "
                f"P1 {format_chips(self.state.stacks[1])}   "
                f"street_in=P0 {format_chips(self.state.street_contrib[0])}, "
                f"P1 {format_chips(self.state.street_contrib[1])}"
            )
        )
        self._render_history()
        self._render_controls()
        self.draw_table()

    def _render_history(self) -> None:
        lines = [
            f"Hand {self.hand_number}; button/SB P{self.state.button}",
            (
                f"Starting stacks: P0 {format_chips(self.state.initial_stacks[0])}, "
                f"P1 {format_chips(self.state.initial_stacks[1])}"
            ),
            "",
        ]
        lines.extend(event_summary(event) for event in self.state.history)
        if self.state.terminal:
            lines.extend(("", "TERMINAL"))
            payoffs = getattr(self.state, "payoffs", None)
            payouts = getattr(self.state, "payouts", None)
            winners = tuple(getattr(self.state, "winners", ()))
            uncalled = getattr(self.state, "uncalled_returns", None)
            if uncalled is not None and any(float(value) > EPSILON for value in uncalled):
                lines.append(
                    "Uncalled returns: "
                    + ", ".join(
                        f"P{seat} {format_chips(value)}"
                        for seat, value in enumerate(uncalled)
                    )
                )
            if payouts is not None:
                lines.append(
                    "Payouts: "
                    + ", ".join(
                        f"P{seat} {format_chips(value)}"
                        for seat, value in enumerate(payouts)
                    )
                )
            if payoffs is not None:
                lines.append(
                    "Payoffs: "
                    + ", ".join(
                        f"P{seat} {float(value):+g}"
                        for seat, value in enumerate(payoffs)
                    )
                )
            lines.append(
                "Winner(s): "
                + (", ".join(f"P{seat}" for seat in winners) if winners else "none")
            )
        self.history_text.configure(state="normal")
        self.history_text.delete("1.0", "end")
        self.history_text.insert("end", "\n".join(lines))
        self.history_text.see("end")
        self.history_text.configure(state="disabled")

    def _clear_action_buttons(self) -> None:
        for button in self.action_buttons:
            button.destroy()
        self.action_buttons.clear()

    def _render_controls(self) -> None:
        self._clear_action_buttons()
        if self.state.terminal:
            winners = ", ".join(
                f"P{seat}" for seat in getattr(self.state, "winners", ())
            )
            payoffs = getattr(self.state, "payoffs", (0.0, 0.0))
            self.prompt.configure(
                text=(
                    f"Hand complete - winner(s) {winners or 'none'}; "
                    f"payoffs P0 {float(payoffs[0]):+g}, P1 {float(payoffs[1]):+g}"
                )
            )
            can_continue = min(self.session_stacks) > EPSILON
            self.next_hand_button.configure(
                state="normal" if can_continue else "disabled"
            )
            self.exact_raise_button.configure(state="disabled")
            self.exact_raise_entry.configure(state="disabled")
            self.exact_range.configure(
                text=(
                    "Click Next hand to rotate the button."
                    if can_continue
                    else "Match complete; reset to continue."
                )
            )
            return

        actor = int(self.state.to_act)
        legal = self.env.legal_actions(self.state)
        to_call = float(self.env.amount_to_call(self.state, actor))
        self.prompt.configure(
            text=(
                f"Choose for Player {actor} - "
                + (
                    f"{format_chips(to_call)} to call"
                    if to_call > EPSILON
                    else "may check or bet"
                )
            )
        )
        for index, action in enumerate(legal):
            button = ttk.Button(
                self.actions_frame,
                text=fixed_action_label(self.env, self.state, action),
                command=lambda selected=action: self.apply_fixed_action(selected),
                width=19,
            )
            button.grid(
                row=index // 5,
                column=index % 5,
                padx=(0, 6),
                pady=(0, 5),
                sticky="ew",
            )
            self.action_buttons.append(button)
        for column in range(5):
            self.actions_frame.grid_columnconfigure(column, weight=1)

        maximum = (
            float(self.state.street_contrib[actor]) + float(self.state.stacks[actor])
        )
        minimum_increment = float(
            getattr(
                self.state,
                "min_raise",
                getattr(
                    self.state,
                    "minimum_raise",
                    getattr(self.state, "min_raise_increment", 0.0),
                ),
            )
        )
        minimum = float(self.state.current_bet) + minimum_increment
        raise_rights = list(getattr(self.state, "raise_rights", [True, True]))
        can_raise = (
            bool(raise_rights[actor])
            and maximum > float(self.state.current_bet) + EPSILON
        )
        if can_raise:
            suggested = min(maximum, minimum)
            if abs(suggested - round(suggested)) <= EPSILON:
                self.exact_raise_var.set(str(int(round(suggested))))
            else:
                self.exact_raise_var.set(str(math.ceil(suggested)))
            if maximum + EPSILON < minimum:
                range_text = (
                    f"short all-in only: exact target {format_chips(maximum)}"
                )
            else:
                range_text = (
                    f"full raise range {format_chips(minimum)}.."
                    f"{format_chips(maximum)}; short all-in validated by engine"
                )
            self.exact_range.configure(text=range_text)
            self.exact_raise_entry.configure(state="normal")
            self.exact_raise_button.configure(state="normal")
        else:
            self.exact_raise_var.set("")
            self.exact_range.configure(text="raising is not legal")
            self.exact_raise_entry.configure(state="disabled")
            self.exact_raise_button.configure(state="disabled")
        self.next_hand_button.configure(state="disabled")

    def draw_table(self) -> None:
        if self.state is None:
            return
        canvas = self.canvas
        canvas.delete("all")
        width = max(canvas.winfo_width(), 620)
        height = max(canvas.winfo_height(), 460)
        canvas.create_oval(
            45,
            35,
            width - 45,
            height - 35,
            fill=self.TABLE,
            outline="#c8a951",
            width=5,
        )

        board_y = height * 0.43
        board_width = 5 * 62 + 4 * 8
        board_x = width / 2 - board_width / 2
        for index in range(5):
            card = self.state.board[index] if index < len(self.state.board) else None
            self._draw_card(board_x + index * 70, board_y, card)
        canvas.create_text(
            width / 2,
            board_y - 28,
            text=f"POT  {format_chips(self.state.pot)}",
            fill=self.GOLD,
            font=("Segoe UI", 13, "bold"),
        )

        positions = ((width / 2, height - 115), (width / 2, 92))
        for seat, (x, y) in enumerate(positions):
            self._draw_player(seat, x, y)

    def _draw_player(self, seat: int, x: float, y: float) -> None:
        active = not self.state.terminal and self.state.to_act == seat
        outline = self.GOLD if active else "#8f9aa3"
        fill = "#253746" if not self.state.folded[seat] else "#3d4449"
        self.canvas.create_rectangle(
            x - 175,
            y - 47,
            x + 175,
            y + 68,
            fill=fill,
            outline=outline,
            width=3,
        )
        markers: list[str] = []
        if seat == self.state.button:
            markers.extend(("BTN", "SB"))
        else:
            markers.append("BB")
        if self.state.folded[seat]:
            markers.append("FOLDED")
        elif self.state.all_in[seat]:
            markers.append("ALL-IN")
        self.canvas.create_text(
            x - 72,
            y - 28,
            text=f"{SEAT_NAMES[seat]}  {' / '.join(markers)}",
            fill=self.TEXT,
            font=("Segoe UI", 10, "bold"),
        )
        self.canvas.create_text(
            x - 72,
            y - 7,
            text=(
                f"Stack {format_chips(self.state.stacks[seat])}   "
                f"In {format_chips(self.state.total_contrib[seat])}   "
                f"Street {format_chips(self.state.street_contrib[seat])}"
            ),
            fill="#d5dbdb",
            font=("Segoe UI", 9),
        )
        for offset, card in enumerate(self.state.hole[seat]):
            self._draw_card(x + 55 + offset * 53, y - 26, card, small=True)

    def _draw_card(
        self,
        x: float,
        y: float,
        card: int | None,
        *,
        small: bool = False,
    ) -> None:
        width, height = (45, 56) if small else (62, 80)
        if card is None:
            self.canvas.create_rectangle(
                x,
                y,
                x + width,
                y + height,
                fill="#1a5837",
                outline="#5d8d72",
            )
            return
        compact = card_to_string(int(card))
        suit = {"c": "clubs", "d": "diamonds", "h": "hearts", "s": "spades"}[
            compact[1]
        ]
        symbol = {
            "clubs": "\u2663",
            "diamonds": "\u2666",
            "hearts": "\u2665",
            "spades": "\u2660",
        }[suit]
        color = "#c62828" if compact[1] in "dh" else "#111111"
        self.canvas.create_rectangle(
            x,
            y,
            x + width,
            y + height,
            fill="white",
            outline="#d0d3d4",
            width=2,
        )
        self.canvas.create_text(
            x + width / 2,
            y + height / 2,
            text=compact[0] + symbol,
            fill=color,
            font=("Segoe UI", 13 if small else 20, "bold"),
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manually control both seats in the exact heads-up engine"
    )
    parser.add_argument("--stack", type=int, default=200, help="starting chips per seat")
    parser.add_argument("--sb", type=int, default=1, help="small blind")
    parser.add_argument("--bb", type=int, default=2, help="big blind")
    parser.add_argument("--seed", type=int, default=None, help="optional deal seed")
    parser.add_argument(
        "--button",
        type=int,
        choices=(0, 1),
        default=0,
        help="button/SB for the first hand",
    )
    args = parser.parse_args(argv)
    if args.stack <= 0:
        parser.error("--stack must be positive")
    if not (0 < args.sb < args.bb):
        parser.error("blinds must satisfy 0 < --sb < --bb")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        root = tk.Tk()
        HeadsUpManualGUI(
            root,
            starting_stack=args.stack,
            small_blind=args.sb,
            big_blind=args.bb,
            seed=args.seed,
            first_button=args.button,
        )
        root.mainloop()
        return 0
    except Exception as exc:
        print(f"Could not start heads-up GUI: {exc}", file=sys.stderr)
        try:
            messagebox.showerror("Could not start heads-up GUI", str(exc))
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
