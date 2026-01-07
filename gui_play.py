"""
gui_play.py
------------
Simple 6-max GUI to play against five bots using trained policies.
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

try:
    from treys import Evaluator, Card
except ImportError:
    Evaluator = None
    Card = None

from poker_env import (
    SimpleHoldemEnv,
    ACTION_FOLD,
    ACTION_CHECK,
    ACTION_CALL,
    ACTION_BET_POT_25,
    ACTION_BET_POT_50,
    ACTION_BET_POT_100,
    ACTION_BET_POT_200,
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
    ACTION_BET_POT_25: "BET 25% POT",
    ACTION_BET_POT_50: "BET 50% POT",
    ACTION_BET_POT_100: "BET 100% POT",
    ACTION_BET_POT_200: "BET 200% POT",
    ACTION_ALL_IN: "ALL-IN",
}

alwaysshowopponetn_cards = False
useargmax = True
useargmax_epsilon = 0.9
gatingallin = False

SUIT_CHARS = ["♠", "♥", "♦", "♣"]
RANK_MAP = {11: "J", 12: "Q", 13: "K", 14: "A"}
SHOWDOWN_EVAL = Evaluator() if Evaluator is not None else None
RANK_CLASS_HIGH_CARD = 1
RANK_CLASS_PAIR = 2
RANK_CLASS_TWO_PAIR = 3
RANK_CLASS_TRIPS = 4
RANK_CLASS_STRAIGHT = 5
RANK_CLASS_FLUSH = 6
RANK_CLASS_FULL_HOUSE = 7
RANK_CLASS_QUADS = 8
RANK_CLASS_STRAIGHT_FLUSH = 9


def card_to_str(c: int) -> str:
    r = RANK_MAP.get(card_rank(c), str(card_rank(c)))
    s = SUIT_CHARS[card_suit(c)]
    return f"{r}{s}"


def card_color(c: int) -> str:
    # Hearts and diamonds red; clubs/spades black
    return "red" if card_suit(c) in (1, 2) else "black"


def card_to_treys(c: int):
    if Card is None:
        return None
    rank_idx = c % 13
    suit_idx = c // 13
    rank_chars = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
    suit_chars = ["s", "h", "d", "c"]
    return Card.new(f"{rank_chars[rank_idx]}{suit_chars[suit_idx]}")


def hand_class(hole, board) -> str:
    if SHOWDOWN_EVAL is None or len(hole) < 2 or len(board) < 3:
        return "Unknown"
    treys_board = [card_to_treys(c) for c in board]
    treys_hand = [card_to_treys(c) for c in hole[:2]]
    if any(c is None for c in treys_board + treys_hand):
        return "Unknown"
    score = SHOWDOWN_EVAL.evaluate(treys_board, treys_hand)
    class_id = SHOWDOWN_EVAL.get_rank_class(score)
    return SHOWDOWN_EVAL.class_to_string(class_id)


def made_rank_id(hole, board):
    if SHOWDOWN_EVAL is None or len(hole) < 2 or len(board) < 3:
        return None
    treys_board = [card_to_treys(c) for c in board]
    treys_hand = [card_to_treys(c) for c in hole[:2]]
    if any(c is None for c in treys_board + treys_hand):
        return None
    score = SHOWDOWN_EVAL.evaluate(treys_board, treys_hand)
    return SHOWDOWN_EVAL.get_rank_class(score)


def board_is_wet(board):
    if len(board) < 3:
        return False
    suits = [card_suit(c) for c in board]
    suited = max(suits.count(i) for i in range(4)) >= 3
    ranks = sorted(set(card_rank(c) for c in board))
    if 14 in ranks:
        ranks.append(1)
        ranks = sorted(set(ranks))
    max_run = 1
    run = 1
    for i in range(1, len(ranks)):
        if ranks[i] == ranks[i - 1] + 1:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 1
    connected = max_run >= 3
    return suited and connected


def nut_flush_draw_info(hole, board):
    if len(board) < 3:
        return False, False
    suit_counts = [0, 0, 0, 0]
    for c in hole[:2] + board:
        suit_counts[card_suit(c)] += 1
    for suit_idx, count in enumerate(suit_counts):
        if 4 <= count < 5:
            ace_in_hole = any(card_rank(c) == 14 and card_suit(c) == suit_idx for c in hole[:2])
            return ace_in_hole, ace_in_hole
    return False, False


def nut_straight_draw_info(hole, board):
    if len(board) < 3:
        return False, False
    ranks = set(card_rank(c) for c in hole[:2] + board)
    if 14 in ranks:
        ranks.add(1)
    sequences = []
    for start in range(1, 11):
        seq = set(range(start, start + 5))
        count = len(seq & ranks)
        if count == 5:
            return False, False
        if count == 4:
            sequences.append(start + 4)
    if not sequences:
        return False, False
    max_high = max(sequences)
    hole_ranks = set(card_rank(c) for c in hole[:2])
    if 14 in hole_ranks:
        hole_ranks.add(1)
    has_blocker = max_high in hole_ranks
    return has_blocker, has_blocker


def allow_all_in(state, player, facing_reraise):
    if SHOWDOWN_EVAL is None:
        return True

    pot = max(1.0, state.pot)
    spr = state.stacks[player] / pot
    num_live = sum(1 for pid in range(state.num_players) if not state.folded[pid])
    hole = state.hole[player]
    board = state.board
    made_rank = made_rank_id(hole, board)

    if spr <= 1.5:
        return True

    if made_rank is not None and made_rank >= RANK_CLASS_TWO_PAIR:
        return True

    nut_flush, _ = nut_flush_draw_info(hole, board)
    nut_straight, _ = nut_straight_draw_info(hole, board)
    nut_draw = nut_flush or nut_straight

    if nut_draw and num_live == 2 and not facing_reraise:
        return True

    if made_rank == RANK_CLASS_PAIR and spr <= 3.0 and num_live == 2 and not facing_reraise:
        return True

    return False


def load_policy(state_dim: int, path: str) -> PolicyNet:
    net = PolicyNet(state_dim)
    state_dict = torch.load(path, map_location="cpu")
    net.load_state_dict(state_dict)
    net.eval()
    return net


def choose_bot_action(policy_net, state, player, legal, facing_reraise=False):
    filtered = legal[:]
    if gatingallin and ACTION_ALL_IN in filtered and not allow_all_in(state, player, facing_reraise):
        filtered = [a for a in filtered if a != ACTION_ALL_IN]
        if not filtered:
            filtered = legal[:]
    x = encode_state(state, player).float().unsqueeze(0)
    with torch.no_grad():
        logits = policy_net(x).squeeze(0)
    mask = torch.full((NUM_ACTIONS,), -1e9)
    for a in filtered:
        mask[a] = 0.0
    probs = torch.softmax(logits + mask, dim=-1)
    if useargmax:
        eps = min(max(useargmax_epsilon, 0.0), 0.05)
        if eps > 0.0 and random.random() < eps:
            action = torch.multinomial(probs, 1).item()
        else:
            action = torch.argmax(probs, dim=-1).item()
    else:
        # Sample (stochastic) for more realistic play
        action = torch.multinomial(probs, 1).item()
    if action not in filtered:
        action = random.choice(filtered)
    return action


class PokerGUI:
    def __init__(self):
        self.env = SimpleHoldemEnv(num_players=6)
        dummy = self.env.new_hand()
        state_dim = encode_state(dummy, 0).shape[0]
        self.policy_default = load_policy(state_dim, "models/policy phase3_310.pt")
        self.policy_phase2 = load_policy(state_dim, "models/policy phase3_310.pt")
        # Player 0 is you; pick which bot seats use which policy here.
        self.phase2_players = {1, 2, 3}
        self.player_labels = {0: "You"}
        for pid in range(1, self.env.num_players):
            if pid in self.phase2_players:
                self.player_labels[pid] = "ph3"
            else:
                self.player_labels[pid] = "ph1"
        self.session = CashSession(self.env)
        self.state = None
        self.animating = False
        self.showdown_reveal = False
        self.hand_count = 0
        self.hand_history = []
        self.canvas_width = 900
        self.canvas_height = 700
        self.hand_badge_map = {}
        self.raise_history = []
        self.seat_positions = self._compute_positions(self.env.num_players)
        self.root = tk.Tk()
        self.root.title("GG-like Poker (6-max) - You vs 5 Bots")

        self.main_frame = tk.Frame(self.root, bg="#0b5d33")
        self.main_frame.pack(fill="both", expand=True)

        self.history_frame = tk.Frame(self.main_frame, bg="#0b5d33", width=280)
        self.history_frame.pack(side="left", fill="y")

        self.table_frame = tk.Frame(self.main_frame, bg="#0b5d33")
        self.table_frame.pack(side="left", fill="both", expand=True)

        self.canvas = tk.Canvas(
            self.table_frame,
            width=self.canvas_width,
            height=self.canvas_height,
            bg="#0b5d33",
            highlightthickness=0,
        )
        self.canvas.pack(fill="both", expand=True)
        self.last_action = {}  # pid -> string for last action
        self.current_actor = None

        self.hand_count_var = tk.StringVar(value="Hands: 0")
        self.hand_count_label = tk.Label(
            self.history_frame,
            textvariable=self.hand_count_var,
            fg="white",
            bg="#0b5d33",
            font=("Arial", 12, "bold"),
            anchor="w",
        )
        self.hand_count_label.pack(fill="x", padx=8, pady=(8, 4))

        history_scroll = tk.Scrollbar(self.history_frame)
        history_scroll.pack(side="right", fill="y", pady=8)
        self.history_text = tk.Text(
            self.history_frame,
            width=34,
            height=38,
            bg="#0b5d33",
            fg="white",
            insertbackground="white",
            relief="flat",
            wrap="word",
            yscrollcommand=history_scroll.set,
            state="disabled",
        )
        self.history_text.pack(side="left", fill="both", expand=True, padx=8, pady=8)
        history_scroll.config(command=self.history_text.yview)

        self.info_var = tk.StringVar()
        self.info_label = tk.Label(self.table_frame, textvariable=self.info_var, fg="white", bg="#0b5d33", font=("Arial", 12))
        self.info_label.pack(fill="x")

        btn_frame = tk.Frame(self.table_frame, bg="#0b5d33")
        btn_frame.pack(fill="x")
        self.btns = {}
        for a in [
            ACTION_FOLD,
            ACTION_CHECK,
            ACTION_CALL,
            ACTION_BET_POT_25,
            ACTION_BET_POT_50,
            ACTION_BET_POT_100,
            ACTION_BET_POT_200,
            ACTION_ALL_IN,
        ]:
            b = tk.Button(btn_frame, text=ACTION_NAMES[a], command=lambda act=a: self.on_action(act), width=12, bg="#d9d9d9")
            b.pack(side="left", padx=4, pady=4)
            self.btns[a] = b

        self.root.after(200, self.start_hand)

    def start_hand(self):
        self.showdown_reveal = False
        self.state = self.session.start_hand()
        self.last_action = {}
        self.raise_history = []
        # Fold out players with zero stack at hand start
        for pid in range(self.env.num_players):
            if self.state.initial_stacks[pid] <= 0:
                self.state.folded[pid] = True
        self._ensure_valid_to_act()
        self.hand_count += 1
        self.hand_count_var.set(f"Hands: {self.hand_count}")
        self._append_history(f"=== Hand #{self.hand_count} ===")
        positions = self._badge_positions(self.state)
        self.hand_badge_map = self._resolve_badges_from_positions(positions)
        if positions:
            button, sb, bb = positions
            self._append_history(
                f"Button: P{button} | SB: P{sb} | BB: P{bb}"
            )
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
        center_x = self.canvas_width / 2
        board_y = 320
        if s.board:
            card_spacing = 32
            cards_width = (len(s.board) - 1) * card_spacing
            cards_start_x = center_x - cards_width / 2
            self.canvas.create_text(
                cards_start_x - 10,
                board_y,
                text="Board:",
                fill="white",
                font=("Arial", 16, "bold"),
                anchor="e",
            )
            for i, card in enumerate(s.board):
                x = cards_start_x + i * card_spacing
                self.canvas.create_text(
                    x,
                    board_y,
                    text=card_to_str(card),
                    fill=card_color(card),
                    font=("Arial", 16, "bold"),
                )
        else:
            self.canvas.create_text(center_x, board_y, text="Board: (no board)", fill="white", font=("Arial", 16, "bold"))
        self.canvas.create_text(center_x, 350, text=f"Pot: {s.pot:.2f}", fill="white", font=("Arial", 13, "bold"))
        to_act_display = "-" if s.to_act is None or s.to_act < 0 else f"P{s.to_act}"
        self.canvas.create_text(center_x, 370, text=f"To act: {to_act_display}", fill="yellow", font=("Arial", 13, "bold"))

        # Seats
        badge_map = self.hand_badge_map or {}
        for pid in range(self.env.num_players):
            x, y = self.seat_positions[pid]
            is_hero = pid == 0
            color = "#ffeecc" if is_hero else "#ffffff"
            seat_color = self._seat_color(pid, s)
            self.canvas.create_oval(x - 50, y - 30, x + 50, y + 30, fill=seat_color, outline="white", width=2)
            label = self.player_labels.get(pid, "Bot")
            self.canvas.create_text(x, y - 20, text=f"P{pid} ({label})", fill="white", font=("Arial", 11, "bold"))
            stack_txt = f"Stack: {s.stacks[pid]:.1f}"
            self.canvas.create_text(x, y - 5, text=stack_txt, fill=color, font=("Arial", 10, "bold"))
            contrib_txt = f"In: {s.contrib[pid]:.1f}"
            self.canvas.create_text(x, y + 10, text=contrib_txt, fill=color, font=("Arial", 10))
            if s.folded[pid]:
                hole_txt = "Folded"
                self.canvas.create_text(x, y + 30, text=hole_txt, fill="white", font=("Arial", 10, "bold"))
            else:
                # Render hole cards as larger text on mini-cards
                show_cards = self._should_show_hole_cards(pid, s)
                cards = s.hole[pid][:2] if show_cards and len(s.hole[pid]) >= 2 else []
                if len(cards) == 2:
                    c1, c2 = cards
                else:
                    c1 = c2 = None
                self.canvas.create_rectangle(x - 34, y + 24, x - 4, y + 54, fill="#f7f7f7", outline="#333333")
                self.canvas.create_rectangle(x + 4, y + 24, x + 34, y + 54, fill="#f7f7f7", outline="#333333")
                if show_cards:
                    if c1 is not None:
                        self.canvas.create_text(
                            x - 19, y + 39, text=card_to_str(c1), fill=card_color(c1), font=("Arial", 12, "bold")
                        )
                    if c2 is not None:
                        self.canvas.create_text(
                            x + 19, y + 39, text=card_to_str(c2), fill=card_color(c2), font=("Arial", 12, "bold")
                        )
                else:
                    self.canvas.create_text(x - 19, y + 39, text="??", fill="#666666", font=("Arial", 12, "bold"))
                    self.canvas.create_text(x + 19, y + 39, text="??", fill="#666666", font=("Arial", 12, "bold"))

            # Last action text below cards
            if pid in self.last_action:
                self.canvas.create_text(x, y + 60, text=f"{self.last_action[pid]}", fill="gold", font=("Arial", 10, "bold"))
            # Dealer/SB/BB markers
            badge = badge_map.get(pid)
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
        prev_state = self.state
        prev_board = self.state.board[:]
        self.state = self.env.step(self.state, action)
        self.last_action[0] = ACTION_NAMES[action]
        self._append_history(f"You: {ACTION_NAMES[action]}")
        self._log_board_transition(prev_board, self.state.board)
        self._update_raise_history(prev_state, self.state, 0, action)
        self.refresh_ui(f"You -> {ACTION_NAMES[action]}")
        if self.state.terminal:
            self.on_terminal()
        else:
            self.root.after(400, self.maybe_auto_bot)

    def maybe_auto_bot(self):
        if self.state.terminal:
            self.on_terminal()
            return
        self._ensure_valid_to_act()
        if self.state.to_act is None or self.state.to_act < 0:
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
            self._append_history(f"P{actor}: skipped")
            self.root.after(200, self.maybe_auto_bot)
            return
        delay_ms = random.randint(1000, 2000)
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
            self._append_history(f"P{actor}: skipped")
            self.refresh_ui(f"P{actor} skipped")
            self.root.after(200, self.maybe_auto_bot)
            return
        policy_net = self._policy_for_player(actor)
        facing_reraise = self._facing_reraise(actor)
        act = choose_bot_action(policy_net, self.state, actor, legal, facing_reraise=facing_reraise)
        prev_state = self.state
        prev_board = self.state.board[:]
        self.state = self.env.step(self.state, act)
        self.last_action[actor] = ACTION_NAMES.get(act, str(act))
        self._append_history(f"P{actor}: {ACTION_NAMES.get(act, act)}")
        self._log_board_transition(prev_board, self.state.board)
        self._update_raise_history(prev_state, self.state, actor, act)
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
            self.showdown_reveal = True
            self.animate_showdown(final_state)
            return
        winner = "Split" if final_state.winner == -1 else f"P{final_state.winner}"
        self.info_var.set(f"Hand over. Winner: {winner}")
        self._append_history(f"Winner: {winner}")
        self._log_showdown(final_state)
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
                self._append_history(f"Winner: {winner}")
                self._log_showdown(final_state)
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
            x = random.randint(50, max(51, int(self.canvas_width) - 50))
            y = random.randint(50, max(51, int(self.canvas_height) - 50))
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
        self._append_history(f"Earnings: {earnings}")

    def _append_history(self, line):
        self.hand_history.append(line)
        self.history_text.configure(state="normal")
        self.history_text.insert("end", line + "\n")
        self.history_text.see("end")
        self.history_text.configure(state="disabled")

    def _log_board_transition(self, prev_board, new_board):
        if len(new_board) <= len(prev_board):
            return
        if len(new_board) == 3:
            label = "Flop"
        elif len(new_board) == 4:
            label = "Turn"
        elif len(new_board) == 5:
            label = "River"
        else:
            label = "Board"
        board_str = " ".join(card_to_str(c) for c in new_board)
        self._append_history(f"{label}: {board_str}")

    def _log_showdown(self, final_state):
        live_players = [pid for pid in range(self.env.num_players) if not final_state.folded[pid]]
        if len(live_players) < 2 or len(final_state.board) < 5:
            return
        self._append_history("Showdown:")
        for pid in live_players:
            hole = final_state.hole[pid]
            hole_str = " ".join(card_to_str(c) for c in hole[:2]) if len(hole) >= 2 else "??"
            class_str = hand_class(hole, final_state.board)
            self._append_history(f"P{pid}: {class_str} ({hole_str})")

    def _should_show_hole_cards(self, pid, state):
        if pid == 0:
            return True
        if alwaysshowopponetn_cards:
            return True
        if self.showdown_reveal and state.terminal:
            return True
        return False

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

    def _policy_for_player(self, pid):
        if pid in self.phase2_players:
            return self.policy_phase2
        return self.policy_default

    def _update_raise_history(self, prev_state, new_state, actor, action):
        if prev_state.street != new_state.street:
            self.raise_history = []
        if action in (ACTION_BET_POT_25, ACTION_BET_POT_50, ACTION_BET_POT_100, ACTION_BET_POT_200, ACTION_ALL_IN):
            self.raise_history.append(actor)

    def _facing_reraise(self, player):
        if not self.raise_history:
            return False
        if self.raise_history[-1] == player:
            return False
        return player in self.raise_history

    def _next_live_actor(self, current):
        base = current if current is not None else 0
        for i in range(1, self.env.num_players + 1):
            nxt = (base + i) % self.env.num_players
            if not self.state.folded[nxt] and self.state.stacks[nxt] > 0:
                return nxt
        return None

    def _next_live_from(self, state, start, include_start=False, require_stack=True):
        n = self.env.num_players
        idx = start if include_start else (start + 1) % n
        for _ in range(n):
            if not state.folded[idx] and (state.stacks[idx] > 0 or not require_stack):
                return idx
            idx = (idx + 1) % n
        return None

    def _badge_positions(self, state):
        live = [pid for pid in range(self.env.num_players) if not state.folded[pid]]
        if not live:
            return None
        button = self._next_live_from(state, state.button_player, include_start=True, require_stack=False)
        if button is None:
            return None
        if len(live) == 2:
            sb = button
            bb = self._next_live_from(state, button, include_start=False, require_stack=False)
            if bb is None:
                bb = button
        else:
            sb = self._next_live_from(state, button, include_start=False, require_stack=False)
            if sb is None:
                sb = button
            bb = self._next_live_from(state, sb, include_start=False, require_stack=False)
            if bb is None:
                bb = sb
        return button, sb, bb

    def _resolve_badges_from_positions(self, positions):
        if not positions:
            return {}
        button, sb, bb = positions
        badge_map = {}
        if button == sb and sb == bb:
            badge_map[button] = "BTN/SB/BB"
            return badge_map
        if button == sb:
            badge_map[button] = "BTN/SB"
        else:
            badge_map[button] = "BTN"
            badge_map[sb] = "SB"
        badge_map[bb] = "BB"
        return badge_map

    def _ensure_valid_to_act(self):
        if self.state is None:
            return
        if self.state.to_act is None or self.state.to_act < 0:
            next_actor = self._next_live_actor(0)
            if next_actor is not None:
                self.state.to_act = next_actor
            return
        if self.state.folded[self.state.to_act] or self.state.stacks[self.state.to_act] <= 0:
            next_actor = self._next_live_actor(self.state.to_act)
            if next_actor is not None:
                self.state.to_act = next_actor

    def _compute_positions(self, n):
        # Arrange seats clockwise around a circle, hero (seat 0) at bottom.
        cx, cy = self.canvas_width / 2, self.canvas_height / 2
        radius = min(self.canvas_width, self.canvas_height) * 0.37
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
