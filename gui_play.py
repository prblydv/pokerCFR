"""
gui_play.py
------------
Simple 6-max GUI to play against five bots using the trained policy.
Layout: minimal “table” using Tkinter; always reveals bot hole cards.
Run: python gui_play.py
"""

import tkinter as tk
from tkinter import messagebox
import threading
import time
import copy
import random
import math

import torch

from poker_env import (
    SimpleHoldemEnv,
    ACTION_FOLD,
    ACTION_CHECK,
    ACTION_CALL,
    ACTION_RAISE_SMALL,
    ACTION_RAISE_MEDIUM,
    ACTION_ALL_IN,
    NUM_ACTIONS,
)
from abstraction import card_rank, card_suit, encode_state
from networks import PolicyNet
from cash_session import CashSession

ACTION_NAMES = {
    ACTION_FOLD: "FOLD",
    ACTION_CHECK: "CHECK",
    ACTION_CALL: "CALL",
    ACTION_RAISE_SMALL: "RAISE SMALL",
    ACTION_RAISE_MEDIUM: "RAISE MEDIUM",
    ACTION_ALL_IN: "ALL-IN",
}

SUIT_CHARS = ["♠", "♥", "♦", "♣"]
RANK_MAP = {11: "J", 12: "Q", 13: "K", 14: "A"}


def card_to_str(c: int) -> str:
    r = RANK_MAP.get(card_rank(c), str(card_rank(c)))
    s = SUIT_CHARS[card_suit(c)]
    return f"{r}{s}"


def card_color(c: int) -> str:
    # Hearts and diamonds red; clubs/spades black
    return "red" if card_suit(c) in (1, 2) else "black"


def load_policy(state_dim: int) -> PolicyNet:
    net = PolicyNet(state_dim)
    state_dict = torch.load("models/policy.pt", map_location="cpu")
    net.load_state_dict(state_dict)
    net.eval()
    return net


def choose_bot_action(policy_net, state, player, legal):
    x = encode_state(state, player).float().unsqueeze(0)
    with torch.no_grad():
        logits = policy_net(x).squeeze(0)
    mask = torch.full((NUM_ACTIONS,), -1e9)
    for a in legal:
        mask[a] = 0.0
    probs = torch.softmax(logits + mask, dim=-1)
    # Sample (stochastic) for more realistic play
    action = torch.multinomial(probs, 1).item()
    if action not in legal:
        action = random.choice(legal)
    return action


class PokerGUI:
    def __init__(self):
        self.env = SimpleHoldemEnv(num_players=6)
        dummy = self.env.new_hand()
        state_dim = encode_state(dummy, 0).shape[0]
        self.policy = load_policy(state_dim)
        self.session = CashSession(self.env)
        self.state = None
        self.animating = False
        self.seat_positions = self._compute_positions(self.env.num_players)
        self.root = tk.Tk()
        self.root.title("GG-like Poker (6-max) - You vs 5 Bots")
        self.canvas = tk.Canvas(self.root, width=1100, height=700, bg="#0b5d33", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.last_action = {}  # pid -> string for last action
        self.current_actor = None

        self.info_var = tk.StringVar()
        self.info_label = tk.Label(self.root, textvariable=self.info_var, fg="white", bg="#0b5d33", font=("Arial", 12))
        self.info_label.pack(fill="x")

        btn_frame = tk.Frame(self.root, bg="#0b5d33")
        btn_frame.pack(fill="x")
        self.btns = {}
        for a in [ACTION_FOLD, ACTION_CHECK, ACTION_CALL, ACTION_RAISE_SMALL, ACTION_RAISE_MEDIUM, ACTION_ALL_IN]:
            b = tk.Button(btn_frame, text=ACTION_NAMES[a], command=lambda act=a: self.on_action(act), width=12, bg="#d9d9d9")
            b.pack(side="left", padx=4, pady=4)
            self.btns[a] = b

        self.root.after(200, self.start_hand)

    def start_hand(self):
        self.state = self.session.start_hand()
        # Fold out players with zero stack at hand start
        for pid in range(self.env.num_players):
            if self.state.stacks[pid] <= 0:
                self.state.folded[pid] = True
        self.refresh_ui("New hand started")
        self.maybe_auto_bot()

    def refresh_ui(self, msg=""):
        self.canvas.delete("all")
        if msg:
            self.info_var.set(msg)
        s = self.state
        if s is None:
            return

        # Table positions for 6 seats
        # Board
        board_str = " ".join(card_to_str(c) for c in s.board) if s.board else "(no board)"
        self.canvas.create_text(550, 320, text=f"Board: {board_str}", fill="white", font=("Arial", 16, "bold"))
        self.canvas.create_text(550, 350, text=f"Pot: {s.pot:.2f}", fill="white", font=("Arial", 13, "bold"))
        self.canvas.create_text(550, 370, text=f"To act: P{s.to_act}", fill="yellow", font=("Arial", 13, "bold"))

        # Seats
        for pid in range(self.env.num_players):
            x, y = self.seat_positions[pid]
            is_hero = pid == 0
            color = "#ffeecc" if is_hero else "#ffffff"
            seat_color = self._seat_color(pid, s)
            self.canvas.create_oval(x - 50, y - 30, x + 50, y + 30, fill=seat_color, outline="white", width=2)
            self.canvas.create_text(x, y - 20, text=f"P{pid}", fill="white", font=("Arial", 11, "bold"))
            stack_txt = f"Stack: {s.stacks[pid]:.1f}"
            self.canvas.create_text(x, y - 5, text=stack_txt, fill=color, font=("Arial", 10, "bold"))
            contrib_txt = f"In: {s.contrib[pid]:.1f}"
            self.canvas.create_text(x, y + 10, text=contrib_txt, fill=color, font=("Arial", 10))
            if s.folded[pid]:
                hole_txt = "Folded"
                self.canvas.create_text(x, y + 30, text=hole_txt, fill="white", font=("Arial", 10, "bold"))
            else:
                # Render hole cards as larger text on mini-cards
                cards = s.hole[pid][:2] if len(s.hole[pid]) >= 2 else []
                if len(cards) == 2:
                    c1, c2 = cards
                else:
                    c1 = c2 = None
                self.canvas.create_rectangle(x - 34, y + 24, x - 4, y + 54, fill="#f7f7f7", outline="#333333")
                self.canvas.create_rectangle(x + 4, y + 24, x + 34, y + 54, fill="#f7f7f7", outline="#333333")
                if c1 is not None:
                    self.canvas.create_text(
                        x - 19, y + 39, text=card_to_str(c1), fill=card_color(c1), font=("Arial", 12, "bold")
                    )
                if c2 is not None:
                    self.canvas.create_text(
                        x + 19, y + 39, text=card_to_str(c2), fill=card_color(c2), font=("Arial", 12, "bold")
                    )
            # Last action text below cards
            if pid in self.last_action:
                self.canvas.create_text(x, y + 60, text=f"{self.last_action[pid]}", fill="gold", font=("Arial", 10, "bold"))
            # Dealer/SB/BB markers
            badge = None
            if pid == s.sb_player and pid == s.button_player:
                badge = "BTN/SB"
            elif pid == s.button_player:
                badge = "BTN"
            elif pid == s.sb_player:
                badge = "SB"
            elif pid == s.bb_player:
                badge = "BB"
            if badge:
                self.canvas.create_rectangle(x - 22, y - 52, x + 22, y - 34, fill="#d4aa00", outline="white")
                self.canvas.create_text(x, y - 43, text=badge, fill="black", font=("Arial", 9, "bold"))

        self.update_buttons()

    def update_buttons(self):
        legal = self.env.legal_actions(self.state)
        for a, btn in self.btns.items():
            btn["state"] = tk.NORMAL if a in legal and self.state.to_act == 0 else tk.DISABLED

    def on_action(self, action):
        if self.state.to_act != 0:
            return
        self.state = self.env.step(self.state, action)
        self.last_action[0] = ACTION_NAMES[action]
        self.refresh_ui(f"You -> {ACTION_NAMES[action]}")
        if self.state.terminal:
            self.on_terminal()
        else:
            self.root.after(200, self.maybe_auto_bot)

    def maybe_auto_bot(self):
        if self.state.terminal:
            self.on_terminal()
            return
        if self.state.to_act == 0:
            self.update_buttons()
            return
        # Bot turn
        actor = self.state.to_act
        self.current_actor = actor
        legal = self.env.legal_actions(self.state)
        if not legal:
            # Skip actors with no legal moves (likely busted/folded)
            self.state.folded[actor] = True
            self.state.players_acted[actor] = True
            next_actor = self._next_live_actor(actor)
            if next_actor is None:
                self.on_terminal()
                return
            self.state.to_act = next_actor
            self.root.after(200, self.maybe_auto_bot)
            return
        delay_ms = random.randint(1000, 3000)
        self.refresh_ui(f"P{actor} thinking...")
        self.root.after(delay_ms, lambda: self._bot_act(actor))

    def _bot_act(self, actor):
        if self.state.terminal or self.state.to_act != actor:
            return
        legal = self.env.legal_actions(self.state)
        if not legal:
            self.state.folded[actor] = True
            self.state.players_acted[actor] = True
            next_actor = self._next_live_actor(actor)
            if next_actor is None:
                self.on_terminal()
                return
            self.state.to_act = next_actor
            self.refresh_ui(f"P{actor} skipped")
            self.root.after(200, self.maybe_auto_bot)
            return
        act = choose_bot_action(self.policy, self.state, actor, legal)
        self.state = self.env.step(self.state, act)
        self.last_action[actor] = ACTION_NAMES.get(act, str(act))
        self.refresh_ui(f"Bot P{actor} -> {ACTION_NAMES.get(act, act)}")
        if self.state.terminal:
            self.on_terminal()
        else:
            self.root.after(300, self.maybe_auto_bot)

    def on_terminal(self):
        if self.animating:
            return
        final_state = self.state
        # Slow roll runout if full board exists (all-in showdown)
        if len(final_state.board) == 5:
            self.animating = True
            self.animate_showdown(final_state)
            return
        winner = "Split" if final_state.winner == -1 else f"P{final_state.winner}"
        self.info_var.set(f"Hand over. Winner: {winner}")
        self.session.apply_results(final_state)
        self.root.after(1500, self.start_hand)

    def animate_showdown(self, final_state):
        """Reveal flop/turn/river with delays for more realism."""
        frames = []
        board = final_state.board[:]
        if len(board) >= 3:
            frames.append(board[:3])  # flop
        if len(board) >= 4:
            frames.append(board[:4])  # turn
        if len(board) >= 5:
            frames.append(board[:5])  # river

        def show_frame(idx):
            if idx >= len(frames):
                # finished
                self.state = final_state
                winner = "Split" if final_state.winner == -1 else f"P{final_state.winner}"
                self.info_var.set(f"Hand over. Winner: {winner}")
                self._show_confetti()
                self._show_earnings(final_state)
                self.session.apply_results(final_state)
                self.animating = False
                self.root.after(5000, self.start_hand)
                return
            frame_state = copy.deepcopy(final_state)
            frame_state.board = frames[idx]
            self.state = frame_state
            self.refresh_ui("Runout...")
            self.root.after(800, lambda: show_frame(idx + 1))

        show_frame(0)

    def _show_confetti(self):
        for _ in range(80):
            x = random.randint(50, 1050)
            y = random.randint(50, 650)
            size = random.randint(3, 8)
            color = random.choice(["#ff4f4f", "#ffd700", "#00e676", "#42a5f5", "#ab47bc"])
            self.canvas.create_oval(x, y, x + size, y + size, fill=color, outline="")
        self.root.update_idletasks()

    def _show_earnings(self, final_state):
        texts = []
        for pid in range(self.env.num_players):
            delta = final_state.stacks[pid] - final_state.initial_stacks[pid]
            texts.append(f"P{pid}: {delta:+.2f}")
        earnings = " | ".join(texts)
        self.info_var.set(self.info_var.get() + f" | Earnings: {earnings}")

    def _seat_color(self, pid, state):
        if state.folded[pid]:
            return "#555555"
        if state.to_act == pid:
            return "#e6d35a"
        if state.stacks[pid] <= 0:
            return "#8b0000"
        if pid == 0:
            return "#2ba56d"
        return "#1f7e50"

    def _next_live_actor(self, current):
        for i in range(1, self.env.num_players + 1):
            nxt = (current + i) % self.env.num_players
            if not self.state.folded[nxt] and self.state.stacks[nxt] > 0:
                return nxt
        return None

    def _compute_positions(self, n):
        # Arrange seats clockwise around a circle, hero (seat 0) at bottom.
        cx, cy = 550, 350
        radius = 260
        positions = []
        for i in range(n):
            angle_deg = 270 + i * (360 / n)  # 270 deg = bottom, clockwise rotation
            angle_rad = math.radians(angle_deg)
            x = cx + radius * math.cos(angle_rad)
            y = cy + radius * math.sin(angle_rad)
            positions.append((x, y))
        return positions

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = PokerGUI()
    app.run()
