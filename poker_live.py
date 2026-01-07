

from __future__ import annotations

import json
import logging
import os
import random
import re
import ctypes
import time
from datetime import datetime
from dataclasses import dataclass
from collections import Counter
from typing import Dict, List, Optional, Tuple

import tkinter as tk
from tkinter import ttk

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException

import torch
from pypokerengine.engine.card import Card
from pypokerengine.utils.card_utils import estimate_hole_card_win_rate

from poker_env import (
    GameState,
    SimpleHoldemEnv,
    ACTION_FOLD,
    ACTION_CHECK,
    ACTION_CALL,
    ACTION_BET_POT_25,
    ACTION_BET_POT_50,
    ACTION_BET_POT_100,
    ACTION_BET_POT_200,
    ACTION_ALL_IN,
    ACTION_SEQ_LEN,
    NUM_ACTIONS,
    STREET_PREFLOP,
    STREET_FLOP,
    STREET_TURN,
    STREET_RIVER,
)
from abstraction import encode_state
from networks import PolicyNet, AdvantageNet
from config import DEVICE

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

URL = "https://mgames-poker-fr3.williamhill.com/poker/web/25.1.1.57_1/html/poker/index.html?launcherRedirect=true&hostedMode=3"
# POLICY_PATH = r"C:\\Users\\PRABAL YADAV\\Desktop\\machine learning iim\\pokerbotPlayOnline\\models\\policy phase3_120.pt"
POLICY_PATH = r"C:\\Users\\PRABAL YADAV\\Desktop\\machine learning iim\\pokerbotPlayOnline\\models\\policy phase3_310.pt"
ADV_POLICY_PATHS = [
    r"C:\\Users\\PRABAL YADAV\\Desktop\\machine learning iim\\pokerbotPlayOnline\\models\\adv_p0.pt",
    r"C:\\Users\\PRABAL YADAV\\Desktop\\machine learning iim\\pokerbotPlayOnline\\models\\adv_p1.pt",
    r"C:\\Users\\PRABAL YADAV\\Desktop\\machine learning iim\\pokerbotPlayOnline\\models\\adv_p2.pt",
    r"C:\\Users\\PRABAL YADAV\\Desktop\\machine learning iim\\pokerbotPlayOnline\\models\\adv_p3.pt",
    r"C:\\Users\\PRABAL YADAV\\Desktop\\machine learning iim\\pokerbotPlayOnline\\models\\adv_p4.pt",
    r"C:\\Users\\PRABAL YADAV\\Desktop\\machine learning iim\\pokerbotPlayOnline\\models\\adv_p5.pt",
]

TABLE_NUM_PLAYERS = 6
FORCE_HERO_SEAT_NUMBER = 4  # 1-based seat index from site: player-seat-4

# Set these before running (you can change per table)
TABLE_SMALL_BLIND = 0.05
TABLE_BIG_BLIND = 0.10

# Use a fixed initial stack size in SB units (e.g., 200 SBs)
TABLE_STACK_SBS = 200
TABLE_STACK_SIZE = TABLE_STACK_SBS * TABLE_SMALL_BLIND

# Training blinds (policy was trained with these)
TRAINING_SMALL_BLIND = 1.0
TRAINING_BIG_BLIND = 2.0
TRAINING_STACK_SIZE = TABLE_STACK_SBS * TRAINING_SMALL_BLIND

# Logging
LOG_FULL_STATE = True
LOG_TABLE_TO_FILE = True
LOG_PRINT_CONSOLE = False
LOG_FILE_PATH = "table_state_debug.log"
LOG_EVERY_N = 1

# UI
UI_REFRESH_MS = 100
REQUIRE_ACTION_BUTTONS = False
SHOW_TABLE_VIEW = True
SHOW_INPUT_TABLE = True
MAX_EVENT_LOG = 30
TABLE_HISTORY_DIR = "tablehistory"
TABLE_HISTORY_SNAPSHOT_EVERY_N = 1

# Automation
AUTOMATION = True
FOLD_PROB_THRESHOLD = 0.75
ACTION_SAMPLE_PROB = 0.80  # sample vs argmax selection ratio
AUTO_ACTION_MIN_DELAY_S = 0.75
AUTO_ACTION_MAX_DELAY_S = 3.25
AUTOMATION_STREETS = {STREET_PREFLOP}

# Table view seat positions (mapped to website seat numbers 1..6 -> index 0..5)
# Order: seat-1, seat-2, seat-3, seat-4, seat-5, seat-6
SEAT_POSITIONS_6 = [
    (500, 70),   # seat-1 (top-center)
    (820, 110),  # seat-2 (top-right)
    (900, 230),  # seat-3 (right)
    (500, 360),  # seat-4 (bottom-center)
    (180, 320),  # seat-5 (bottom-left)
    (100, 190),  # seat-6 (left)
]

# Strategy
USE_ADVANTAGE_NET = True

# Selectors (based on your table DOM)
SEAT_SELECTORS = [
    ".table-6-players .player-area",
    ".table .player-area",
    ".player-area",
]
STACK_SELECTORS = [
    ".player-nameplate .text-block.amount",
    ".player-nameplate .amount",
]
BET_SELECTORS = [
    ".player-bet .amount-cont .amount",
    ".player-bet .amount",
]
NAME_SELECTORS = [
    ".player-nameplate .text-block.nickname .target",
    ".player-nameplate .nickname .target",
]
ACTION_TEXT_SELECTORS = [
    ".player-action .action-text",
    ".player-action",
    ".action-text",
    ".player-last-action",
    ".last-action",
    ".action-badge",
    ".action-label",
]
BUTTON_SELECTORS = [
    ".game-position:not(.pt-visibility-hidden) .dealer.table-assets-btn-dealer",
]
ACTIVE_SELECTORS = [
    ".turn-to-act-indicator",
    ".timeout-wrapper",
    ".nameplate-blink",
    ".text-countdown",
]
HOLE_CARD_SELECTORS = [
    ".player-area.my-player .card-wrapper",
    ".player-area.my-player .card-image-backup",
    ".player-area.my-player img.card-image",
    ".cards-holder-hero .card-wrapper",
    ".cards-holder-hero .card-image-backup",
    ".cards-holder-hero img.card-image",
]

ACTION_ID_TO_NAME = {
    ACTION_FOLD: "FOLD",
    ACTION_CHECK: "CHECK",
    ACTION_CALL: "CALL",
    ACTION_BET_POT_25: "BET_POT_25",
    ACTION_BET_POT_50: "BET_POT_50",
    ACTION_BET_POT_100: "BET_POT_100",
    ACTION_BET_POT_200: "BET_POT_200",
    ACTION_ALL_IN: "ALL_IN",
}
ACTION_DISPLAY_NAME = {
    ACTION_FOLD: "FOLD",
    ACTION_CHECK: "CHECK",
    ACTION_CALL: "CALL",
    ACTION_BET_POT_25: "BET 25% POT",
    ACTION_BET_POT_50: "BET 50% POT",
    ACTION_BET_POT_100: "BET 100% POT",
    ACTION_BET_POT_200: "BET 200% POT",
    ACTION_ALL_IN: "ALL-IN",
}

logger = logging.getLogger("table_debug")
LOG_TICK = 0
TABLE_HISTORY: List[dict] = []
TABLE_HISTORY_LOG_PATH = ""
TABLE_HISTORY_SESSION_ID = ""
TABLE_HISTORY_EVENT_ID = 0
AUTO_DECISION_KEY: Optional[Tuple[int, int, int, int]] = None

STREET_LABELS = {
    STREET_PREFLOP: "PREFLOP",
    STREET_FLOP: "FLOP",
    STREET_TURN: "TURN",
    STREET_RIVER: "RIVER",
}


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------

def setup_logger():
    logger.handlers = []
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    if LOG_TABLE_TO_FILE:
        fh = logging.FileHandler(LOG_FILE_PATH, encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    if LOG_PRINT_CONSOLE:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)

def click_action_button(driver, button_id: str) -> bool:
    try:
        button = driver.find_element(By.ID, button_id)
        if button.is_displayed() and button.is_enabled():
            button.click()
            return True
    except Exception:
        return False
    return False

def play_airbus_alert() -> None:
    path = os.path.join(os.path.dirname(__file__), "airbus.mp3")
    if not os.path.exists(path):
        log_event("sound_missing", {"path": path})
        return
    alias = "airbus_alert"
    try:
        ctypes.windll.winmm.mciSendStringW(f"close {alias}", None, 0, None)
        ctypes.windll.winmm.mciSendStringW(f'open "{path}" type mpegvideo alias {alias}', None, 0, None)
        ctypes.windll.winmm.mciSendStringW(f"set {alias} volume to 1000", None, 0, None)
        ctypes.windll.winmm.mciSendStringW(f"play {alias} from 0", None, 0, None)
    except Exception as exc:
        log_event("sound_error", {"error": str(exc), "path": path})

def build_legal_mask(legal_actions: Optional[List[int]], num_actions: int) -> torch.Tensor:
    mask = torch.zeros(num_actions, dtype=torch.float32)
    if legal_actions:
        for a in legal_actions:
            if 0 <= a < num_actions:
                mask[a] = 1.0
    else:
        mask.fill_(1.0)
    return mask

def masked_normalize_tensor(p: torch.Tensor, mask: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = p * mask
    s = p.sum()
    if s <= eps:
        denom = mask.sum().clamp_min(1.0)
        return mask / denom
    return p / s


def init_history_logging() -> None:
    global TABLE_HISTORY_LOG_PATH, TABLE_HISTORY_SESSION_ID
    os.makedirs(TABLE_HISTORY_DIR, exist_ok=True)
    TABLE_HISTORY_SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
    TABLE_HISTORY_LOG_PATH = os.path.join(TABLE_HISTORY_DIR, f"{TABLE_HISTORY_SESSION_ID}_log.jsonl")
    append_history_event({
        "type": "session_start",
        "session_id": TABLE_HISTORY_SESSION_ID,
        "url": URL,
        "sb": TABLE_SMALL_BLIND,
        "bb": TABLE_BIG_BLIND,
        "stack_sbs": TABLE_STACK_SBS,
        "training_bb": TRAINING_BIG_BLIND,
    })


def append_history_event(event: dict) -> None:
    global TABLE_HISTORY_EVENT_ID
    if not event:
        return
    record = {
        "event_id": TABLE_HISTORY_EVENT_ID,
        "ts": time.time(),
        "iso": datetime.now().isoformat(timespec="seconds"),
        "session_id": TABLE_HISTORY_SESSION_ID,
    }
    record.update(event)
    TABLE_HISTORY.append(record)
    if TABLE_HISTORY_LOG_PATH:
        with open(TABLE_HISTORY_LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=True) + "\n")
    TABLE_HISTORY_EVENT_ID += 1


def log_event(message: str, payload: Optional[dict] = None):
    if not LOG_FULL_STATE:
        return
    if payload is None:
        logger.info(message)
    else:
        logger.info(json.dumps({"message": message, "payload": payload}, ensure_ascii=True))


def parse_money(text: str) -> float:
    if not text:
        return 0.0
    cleaned = text.replace(",", "").replace("$", "").replace("€", "").strip()
    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    if not match:
        return 0.0
    try:
        return float(match.group(0))
    except ValueError:
        return 0.0


def format_bb(amount: float) -> str:
    if not TABLE_BIG_BLIND or TABLE_BIG_BLIND <= 0:
        return f"{amount:.2f}"
    return f"{amount / TABLE_BIG_BLIND:.2f}bb"


def extract_text(elem, selectors: List[str]) -> str:
    for sel in selectors:
        try:
            txt = elem.find_element(By.CSS_SELECTOR, sel).text.strip()
            if txt:
                return txt
        except Exception:
            continue
    return ""


def extract_action_text(elem) -> str:
    text = extract_text(elem, ACTION_TEXT_SELECTORS)
    if text:
        return text
    try:
        candidates = elem.find_elements(By.CSS_SELECTOR, "[class*='action']")
    except Exception:
        candidates = []
    for cand in candidates:
        try:
            txt = cand.text.strip()
        except Exception:
            txt = ""
        if not txt:
            continue
        lowered = txt.lower()
        if any(k in lowered for k in ["fold", "check", "call", "raise", "bet", "all-in", "all in", "sit out"]):
            return txt
    return ""


def is_hidden_by_class(elem) -> bool:
    class_name = (elem.get_attribute("class") or "").lower()
    return "pt-hidden" in class_name or "pt-visibility-hidden" in class_name


def has_visible_child(elem, selectors: List[str]) -> bool:
    for sel in selectors:
        try:
            children = elem.find_elements(By.CSS_SELECTOR, sel)
        except Exception:
            children = []
        for child in children:
            try:
                if child.is_displayed():
                    return True
            except Exception:
                pass
            if not is_hidden_by_class(child):
                return True
    return False


def extract_seat_index(elem, fallback_idx: int) -> int:
    class_name = elem.get_attribute("class") or ""
    match = re.search(r"player-seat-(\d+)", class_name)
    if match:
        seat_num = int(match.group(1))
        if 1 <= seat_num <= TABLE_NUM_PLAYERS:
            return seat_num - 1
    for attr in ["data-seat", "data-seat-id", "data-position", "data-index", "data-seatindex"]:
        val = elem.get_attribute(attr)
        if val and val.isdigit():
            seat_num = int(val)
            if 1 <= seat_num <= TABLE_NUM_PLAYERS:
                return seat_num - 1
            return seat_num
        if val:
            match = re.search(r"(\d+)", val)
            if match:
                seat_num = int(match.group(1))
                if 1 <= seat_num <= TABLE_NUM_PLAYERS:
                    return seat_num - 1
                return seat_num
    id_attr = elem.get_attribute("id") or ""
    match = re.search(r"(\d+)", id_attr)
    if match:
        seat_num = int(match.group(1))
        if 1 <= seat_num <= TABLE_NUM_PLAYERS:
            return seat_num - 1
        return seat_num
    return fallback_idx


def read_seat_elements(driver) -> List:
    best = []
    for sel in SEAT_SELECTORS:
        try:
            elems = driver.find_elements(By.CSS_SELECTOR, sel)
        except Exception:
            elems = []
        if len(elems) > len(best):
            best = elems
    return best


def normalize_seats(seats: List[dict], num_players: int) -> List[dict]:
    seat_map = {s.get("seat_index"): s for s in seats if s.get("seat_index") is not None}
    normalized = []
    for idx in range(num_players):
        if idx in seat_map:
            s = seat_map[idx]
        else:
            s = {
                "seat_index": idx,
                "name": "",
                "raw_name_text": "",
                "raw_stack_text": "",
                "raw_bet_text": "",
                "class_name": "",
                "action_text": "",
                "stack": 0.0,
                "bet": 0.0,
                "folded": True,
                "is_hero": False,
                "is_button": False,
                "is_sb": False,
                "is_bb": False,
                "is_active": False,
            }
        normalized.append(s)
    return normalized


def parse_seat_elem(elem, fallback_idx: int) -> dict:
    seat_index = extract_seat_index(elem, fallback_idx)
    name_text = extract_text(elem, NAME_SELECTORS)
    stack_text = extract_text(elem, STACK_SELECTORS)
    bet_text = extract_text(elem, BET_SELECTORS)

    class_name = (elem.get_attribute("class") or "").lower()
    is_hero = "my-player" in class_name

    action_text = extract_action_text(elem)
    action_text_lower = action_text.lower()

    is_folded_action = "fold" in action_text_lower
    is_sit_out_action = "sit out" in action_text_lower or "sit-out" in action_text_lower
    has_fold_class = has_visible_child(elem, [".player-action.action-fold"])
    is_sit_out_class = "sit-out" in class_name or "player-sit-out" in class_name

    is_folded = is_folded_action or is_sit_out_action or has_fold_class or is_sit_out_class

    is_button = has_visible_child(elem, BUTTON_SELECTORS)
    is_sb = has_visible_child(elem, [".small-blind", ".sb"])
    is_bb = has_visible_child(elem, [".big-blind", ".bb"])
    is_active = has_visible_child(elem, ACTIVE_SELECTORS)

    return {
        "seat_index": seat_index,
        "name": name_text,
        "raw_name_text": name_text,
        "raw_stack_text": stack_text,
        "raw_bet_text": bet_text,
        "class_name": class_name,
        "action_text": action_text,
        "stack": parse_money(stack_text),
        "bet": parse_money(bet_text),
        "folded": bool(is_folded),
        "is_hero": bool(is_hero),
        "is_button": bool(is_button),
        "is_sb": bool(is_sb),
        "is_bb": bool(is_bb),
        "is_active": bool(is_active),
    }


def resolve_positions(seats: List[dict], hero_index: int) -> Tuple[int, int, int]:
    n = len(seats)
    button_idx = next((i for i, s in enumerate(seats) if s["is_button"]), None)
    sb_idx = next((i for i, s in enumerate(seats) if s["is_sb"]), None)
    bb_idx = next((i for i, s in enumerate(seats) if s["is_bb"]), None)

    if button_idx is None:
        button_idx = hero_index if hero_index is not None else 0
    if sb_idx is None:
        sb_idx = (button_idx + 1) % n if n else 0
    if bb_idx is None:
        bb_idx = (sb_idx + 1) % n if n else 0

    return button_idx, sb_idx, bb_idx


def resolve_to_act(seats: List[dict], hero_index: int, force_hero_to_act: bool = False) -> Optional[int]:
    active_idx = next((i for i, s in enumerate(seats) if s["is_active"]), None)
    if active_idx is not None:
        return active_idx
    if force_hero_to_act:
        return hero_index
    return None


def read_table_snapshot(driver, force_hero_to_act: bool = False) -> Optional[dict]:
    seat_elems = read_seat_elements(driver)
    if not seat_elems:
        log_event("No seat elements found", {"selectors": SEAT_SELECTORS})
        return None
    seats_raw = [parse_seat_elem(elem, idx) for idx, elem in enumerate(seat_elems)]
    num_players = TABLE_NUM_PLAYERS if TABLE_NUM_PLAYERS else max([s.get("seat_index", 0) for s in seats_raw] + [0]) + 1
    seats = normalize_seats(seats_raw, num_players)

    hero_index = next((i for i, s in enumerate(seats) if s["is_hero"]), None)
    if FORCE_HERO_SEAT_NUMBER:
        forced = max(0, min(num_players - 1, FORCE_HERO_SEAT_NUMBER - 1))
        hero_index = forced
        seats[hero_index]["is_hero"] = True
    if hero_index is None:
        hero_index = 0

    button_idx, sb_idx, bb_idx = resolve_positions(seats, hero_index)
    to_act = resolve_to_act(seats, hero_index, force_hero_to_act=force_hero_to_act)
    return {
        "seats": seats,
        "hero_index": hero_index,
        "button": button_idx,
        "sb": sb_idx,
        "bb": bb_idx,
        "to_act": to_act,
        "num_players": num_players,
        "seat_elem_count": len(seat_elems),
    }


def street_from_board(community_cards: List[str]) -> int:
    count = len(community_cards)
    if count >= 5:
        return STREET_RIVER
    if count == 4:
        return STREET_TURN
    if count == 3:
        return STREET_FLOP
    return STREET_PREFLOP


def translate_suit(suit_symbol: str) -> Optional[str]:
    suit_map = {
        "\u2663": "C",
        "\u2666": "D",
        "\u2665": "H",
        "\u2660": "S",
    }
    return suit_map.get(suit_symbol, None)


def parse_card_from_img_src(src: str) -> Optional[str]:
    if not src:
        return None
    lower = src.lower()
    match = re.search(r"([cdhs])([0-9]{1,2}|[ajkqt])\\.svg", lower)
    if not match:
        match = re.search(r"([0-9]{1,2}|[ajkqt])([cdhs])\\.svg", lower)
        if not match:
            return None
        rank_raw = match.group(1)
        suit_raw = match.group(2)
    else:
        suit_raw = match.group(1)
        rank_raw = match.group(2)

    suit_map = {"c": "C", "d": "D", "h": "H", "s": "S"}
    rank_map = {"a": "A", "k": "K", "q": "Q", "j": "J", "t": "T"}
    suit = suit_map.get(suit_raw, "")
    rank = rank_map.get(rank_raw, rank_raw)
    if not suit or not rank:
        return None
    return f"{rank.upper()}{suit}"


def card_str_to_id(card_str: str) -> Optional[int]:
    if not card_str:
        return None
    text = card_str.strip().upper()
    if len(text) == 3:
        rank = text[:2]
        suit = text[2:]
    else:
        rank = text[0]
        suit = text[1]

    rank_map = {
        "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
        "10": 10, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14,
    }
    suit_map = {"S": 0, "H": 1, "D": 2, "C": 3}
    if rank not in rank_map or suit not in suit_map:
        return None
    return suit_map[suit] * 13 + (rank_map[rank] - 2)


def cards_str_to_ids(cards: List[str]) -> List[int]:
    ids = []
    for c in cards:
        card_id = card_str_to_id(c)
        if card_id is not None:
            ids.append(card_id)
    return ids


def _read_rank_suit(card_elem) -> Tuple[str, str]:
    rank = ""
    suit = ""
    try:
        rank = card_elem.find_element(By.CSS_SELECTOR, ".card-image-backup .card-rank").text.strip()
        suit = card_elem.find_element(By.CSS_SELECTOR, ".card-image-backup .card-suit").text.strip()
    except Exception:
        pass
    if not rank:
        try:
            rank = card_elem.find_element(By.CLASS_NAME, "card-rank").text.strip()
        except Exception:
            rank = ""
    if not suit:
        try:
            suit = card_elem.find_element(By.CLASS_NAME, "card-suit").text.strip()
        except Exception:
            suit = ""
    return rank, suit


def _card_from_elem(card_elem) -> Optional[str]:
    rank, suit = _read_rank_suit(card_elem)
    translated_suit = translate_suit(suit)
    if rank and translated_suit:
        return f"{rank.upper()}{translated_suit}"

    src = ""
    try:
        if card_elem.tag_name == "img":
            src = card_elem.get_attribute("src")
    except Exception:
        src = ""

    if not src:
        try:
            img = card_elem.find_element(By.CSS_SELECTOR, "img.card-image")
            src = img.get_attribute("src")
        except Exception:
            src = ""

    return parse_card_from_img_src(src)


def read_cards_hero_with_debug(driver) -> Tuple[List[str], Dict[str, int]]:
    hole_cards = []
    selector_counts: Dict[str, int] = {}
    for sel in HOLE_CARD_SELECTORS:
        try:
            card_elements = driver.find_elements(By.CSS_SELECTOR, sel)
        except Exception:
            card_elements = []
        selector_counts[sel] = len(card_elements)
        for card_elem in card_elements:
            card = _card_from_elem(card_elem)
            if card:
                hole_cards.append(card)
    # Deduplicate while preserving order
    seen = set()
    unique_cards = []
    for c in hole_cards:
        if c not in seen:
            seen.add(c)
            unique_cards.append(c)
    return unique_cards, selector_counts


def read_community_cards(driver) -> List[str]:
    try:
        community_cards = []
        card_elements = driver.find_elements(By.CSS_SELECTOR, ".community-cards .card-wrapper")
        for card_elem in card_elements:
            card = _card_from_elem(card_elem)
            if card:
                community_cards.append(card)
        return community_cards
    except Exception:
        return []


def _card_rank_value(card: str) -> Optional[int]:
    if not card or len(card) < 2:
        return None
    rank = card[:-1].upper()
    rank_map = {
        "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
        "10": 10, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14,
    }
    return rank_map.get(rank, None)


def _card_suit_value(card: str) -> Optional[str]:
    if not card or len(card) < 2:
        return None
    return card[-1].upper()


def _ranks_and_suits(cards: List[str]) -> Tuple[List[int], List[str]]:
    ranks = []
    suits = []
    for card in cards:
        rank = _card_rank_value(card)
        suit = _card_suit_value(card)
        if rank is not None and suit is not None:
            ranks.append(rank)
            suits.append(suit)
    return ranks, suits


def analyze_board_texture(community_cards: List[str]) -> dict:
    ranks, suits = _ranks_and_suits(community_cards)
    if len(ranks) < 3:
        return {
            "connectivity": "N/A",
            "flush_potential": "N/A",
            "flush_suit": "",
            "flush_count": 0,
            "paired": "N/A",
            "dynamic": "N/A",
        }
    uniq_ranks = sorted(set(ranks))
    gaps = [uniq_ranks[i + 1] - uniq_ranks[i] for i in range(len(uniq_ranks) - 1)]
    max_gap = max(gaps) if gaps else 99
    if max_gap <= 2:
        connectivity = "HIGH"
    elif max_gap <= 4:
        connectivity = "MEDIUM"
    else:
        connectivity = "LOW"

    suit_counts = Counter(suits)
    flush_suit, flush_count = max(suit_counts.items(), key=lambda x: x[1])
    if flush_count >= 3:
        flush_potential = "STRONG"
    elif flush_count == 2:
        flush_potential = "BACKDOOR"
    else:
        flush_potential = "NONE"

    paired = "YES" if any(c >= 2 for c in Counter(ranks).values()) else "NO"
    dynamic = "YES" if connectivity == "HIGH" or flush_potential != "NONE" else "NO"

    return {
        "connectivity": connectivity,
        "flush_potential": flush_potential,
        "flush_suit": flush_suit,
        "flush_count": flush_count,
        "paired": paired,
        "dynamic": dynamic,
    }


def _straight_draw_type(ranks: List[int]) -> Optional[str]:
    if not ranks:
        return None
    rank_set = set(ranks)
    if 14 in rank_set:
        rank_set.add(1)
    open_ended = False
    gutshot = False
    for start in range(1, 11):
        seq = list(range(start, start + 5))
        missing = [r for r in seq if r not in rank_set]
        if len(missing) == 1:
            if missing[0] in (seq[0], seq[-1]):
                open_ended = True
            else:
                gutshot = True
    if open_ended:
        return "OPEN_ENDED"
    if gutshot:
        return "GUTSHOT"
    return None


def _has_straight(ranks: List[int]) -> bool:
    if not ranks:
        return False
    rank_set = set(ranks)
    if 14 in rank_set:
        rank_set.add(1)
    for start in range(1, 11):
        seq = list(range(start, start + 5))
        if all(r in rank_set for r in seq):
            return True
    return False


def detect_draws(hole_cards: List[str], community_cards: List[str]) -> dict:
    all_cards = hole_cards + community_cards
    ranks, suits = _ranks_and_suits(all_cards)
    hero_ranks, hero_suits = _ranks_and_suits(hole_cards)
    suit_counts = Counter(suits)
    hero_suit_counts = Counter(hero_suits)

    flush_draw = False
    nut_flush_draw = False
    backdoor_flush = False
    if len(community_cards) <= 4:
        for suit, total in suit_counts.items():
            if hero_suit_counts.get(suit, 0) > 0 and total == 4:
                flush_draw = True
                if ("A" + suit) in hole_cards:
                    nut_flush_draw = True
            if len(community_cards) == 3 and hero_suit_counts.get(suit, 0) > 0 and total == 3:
                backdoor_flush = True

    straight_draw = _straight_draw_type(ranks)
    return {
        "flush_draw": flush_draw,
        "nut_flush_draw": nut_flush_draw,
        "backdoor_flush": backdoor_flush,
        "straight_draw": straight_draw,
    }


def classify_hand(hole_cards: List[str], community_cards: List[str], draws: dict, snapshot: Optional[dict]) -> dict:
    all_cards = hole_cards + community_cards
    ranks, suits = _ranks_and_suits(all_cards)
    hero_ranks, hero_suits = _ranks_and_suits(hole_cards)
    rank_counts = Counter(ranks)
    board_ranks, _ = _ranks_and_suits(community_cards)
    board_counts = Counter(board_ranks)

    has_quads = any(c == 4 for c in rank_counts.values())
    has_trips = any(c == 3 for c in rank_counts.values())
    pair_ranks = [r for r, c in rank_counts.items() if c == 2]
    has_two_pair = len(pair_ranks) >= 2
    has_pair = len(pair_ranks) == 1
    has_full_house = has_trips and (has_pair or sum(1 for c in rank_counts.values() if c == 3) >= 2)
    has_flush = any(c >= 5 for c in Counter(suits).values())
    has_straight = _has_straight(ranks)

    top_board = max(board_ranks) if board_ranks else None
    hero_pair_rank = next((r for r in hero_ranks if rank_counts.get(r, 0) >= 2), None)
    is_overpair = False
    is_top_pair = False
    if len(hero_ranks) == 2 and hero_ranks[0] == hero_ranks[1] and top_board:
        is_overpair = hero_ranks[0] > top_board
    if hero_pair_rank and top_board and hero_pair_rank == top_board:
        is_top_pair = True

    made_label = "High Card"
    if has_quads:
        made_label = "Quads"
    elif has_full_house:
        made_label = "Full House"
    elif has_flush:
        made_label = "Flush"
    elif has_straight:
        made_label = "Straight"
    elif has_trips:
        made_label = "Set"
    elif has_two_pair:
        made_label = "Two Pair"
    elif is_overpair:
        made_label = "Overpair"
    elif is_top_pair:
        made_label = "Top Pair"
    elif has_pair:
        made_label = "Pair"

    has_draw = draws.get("flush_draw") or draws.get("straight_draw")
    if has_quads or has_full_house:
        category = "NUT MADE HAND"
    elif has_flush or has_straight:
        category = "STRONG MADE HAND"
    elif has_trips or has_two_pair or is_overpair or is_top_pair:
        category = "MEDIUM MADE HAND"
    elif has_pair:
        category = "WEAK MADE HAND"
    else:
        category = "AIR"

    if has_draw and category not in ("AIR", "WEAK MADE HAND"):
        category = "COMBO DRAW"
    elif has_draw and category == "AIR":
        category = "PURE DRAW"

    draw_parts = []
    if draws.get("nut_flush_draw"):
        draw_parts.append("NFD")
    elif draws.get("flush_draw"):
        draw_parts.append("FD")
    if draws.get("straight_draw") == "OPEN_ENDED":
        draw_parts.append("OESD")
    elif draws.get("straight_draw") == "GUTSHOT":
        draw_parts.append("Gutshot")
    if draws.get("backdoor_flush"):
        draw_parts.append("BDFD")
    draw_label = " + ".join(draw_parts) if draw_parts else ""

    hand_label = made_label
    if draw_label:
        hand_label = f"{made_label} + {draw_label}" if made_label != "High Card" else draw_label

    if category in ("NUT MADE HAND", "STRONG MADE HAND"):
        strength = "HIGH"
    elif category in ("MEDIUM MADE HAND", "COMBO DRAW"):
        strength = "MEDIUM-HIGH"
    elif category == "WEAK MADE HAND":
        strength = "LOW-MED"
    elif category == "PURE DRAW":
        strength = "MEDIUM"
    else:
        strength = "LOW"

    ip = False
    if snapshot:
        ip = snapshot.get("hero_index") == snapshot.get("button")
    pos_label = "IP" if ip else "OOP"
    if category in ("NUT MADE HAND", "STRONG MADE HAND", "COMBO DRAW") and ip:
        realization = "HIGH"
    elif ip:
        realization = "MEDIUM"
    else:
        realization = "LOW"

    return {
        "category": category,
        "hand_label": hand_label,
        "strength": strength,
        "equity_realization": f"{realization} ({pos_label})",
    }


def analyze_blockers(hole_cards: List[str], community_cards: List[str]) -> dict:
    hero_ranks, hero_suits = _ranks_and_suits(hole_cards)
    board_ranks, board_suits = _ranks_and_suits(community_cards)
    suit_counts = Counter(board_suits)
    flush_suit = None
    if suit_counts:
        flush_suit = max(suit_counts.items(), key=lambda x: x[1])[0]

    blocks_nut_flush = False
    if flush_suit and ("A" + flush_suit) in hole_cards:
        blocks_nut_flush = True

    blocks_ak = "A" in [c[:-1] for c in hole_cards] or "K" in [c[:-1] for c in hole_cards]
    blocks_sets = any(r in board_ranks for r in hero_ranks)

    return {
        "blocks_nut_flush": blocks_nut_flush,
        "blocks_ak": blocks_ak,
        "blocks_sets": blocks_sets,
    }


def assess_danger_flags(
    snapshot: Optional[dict],
    community_cards: List[str],
    draws: dict,
    board_texture: dict,
    spr: float,
) -> List[str]:
    ranks, _ = _ranks_and_suits(community_cards)
    flags = []
    if len(ranks) >= 3:
        if max(ranks) <= 9 and board_texture.get("connectivity") in ("HIGH", "MEDIUM"):
            flags.append("Board favors BB range")
        if board_texture.get("dynamic") == "YES" and len(community_cards) == 3:
            flags.append("Many turn barrels possible")
        if spr > 0 and spr <= 3 and (draws.get("flush_draw") or draws.get("straight_draw")):
            flags.append("Low SPR + draws")
        if len(community_cards) >= 4 and board_texture.get("paired") == "YES":
            flags.append("Paired turn/river texture")
    return flags


def _suit_name(suit: str) -> str:
    return {
        "H": "HEARTS",
        "D": "DIAMONDS",
        "C": "CLUBS",
        "S": "SPADES",
    }.get(suit, suit)


def build_intel_panel(
    hole_cards: List[str],
    community_cards: List[str],
    snapshot: Optional[dict],
    spr: float,
) -> dict:
    board_texture = analyze_board_texture(community_cards)
    draws = detect_draws(hole_cards, community_cards)
    hand_info = classify_hand(hole_cards, community_cards, draws, snapshot)
    blockers = analyze_blockers(hole_cards, community_cards)
    danger_flags = assess_danger_flags(snapshot, community_cards, draws, board_texture, spr)

    connectivity = board_texture["connectivity"]
    flush_potential = board_texture["flush_potential"]
    paired = board_texture["paired"]
    dynamic = board_texture["dynamic"]
    flush_suit = board_texture.get("flush_suit", "")
    flush_count = board_texture.get("flush_count", 0)
    suit_label = _suit_name(flush_suit) if flush_suit else ""

    def _level_for_texture(value: str) -> str:
        if value in ("N/A", ""):
            return "muted"
        if value in ("HIGH", "STRONG", "YES"):
            return "bad"
        if value in ("MEDIUM", "BACKDOOR"):
            return "warn"
        return "good"

    board_lines = [
        (f"Connectivity: {connectivity}", _level_for_texture(connectivity)),
        (
            f"Flush Potential: {flush_potential}"
            + (f" ({suit_label} {flush_count})" if flush_potential != "N/A" else ""),
            _level_for_texture(flush_potential),
        ),
        (f"Paired: {paired}", _level_for_texture(paired)),
        (f"Dynamic: {dynamic}", _level_for_texture(dynamic)),
    ]

    draws_lines = [
        ("✔ Nut Flush Draw" if draws.get("nut_flush_draw") else "✖ Nut Flush Draw",
         "good" if draws.get("nut_flush_draw") else "muted"),
        ("✔ Open-Ended Straight Draw" if draws.get("straight_draw") == "OPEN_ENDED" else "✖ Open-Ended Straight Draw",
         "good" if draws.get("straight_draw") == "OPEN_ENDED" else "muted"),
        ("✔ Gutshot" if draws.get("straight_draw") == "GUTSHOT" else "✖ Gutshot",
         "warn" if draws.get("straight_draw") == "GUTSHOT" else "muted"),
        ("✔ Backdoor Flush" if draws.get("backdoor_flush") else "✖ Backdoor Flush",
         "warn" if draws.get("backdoor_flush") else "muted"),
    ]

    category = hand_info["category"]
    if category in ("NUT MADE HAND", "STRONG MADE HAND"):
        cat_level = "good"
    elif category in ("MEDIUM MADE HAND", "COMBO DRAW"):
        cat_level = "warn"
    elif category in ("WEAK MADE HAND", "PURE DRAW"):
        cat_level = "warn"
    else:
        cat_level = "bad"

    hand_lines = [
        (f"Category: {hand_info['hand_label']}", cat_level),
        (f"Strength: {hand_info['strength']}", "good" if "HIGH" in hand_info["strength"] else "warn" if "MEDIUM" in hand_info["strength"] else "bad"),
        (f"Equity Realization: {hand_info['equity_realization']}", "good" if "HIGH" in hand_info["equity_realization"] else "warn" if "MEDIUM" in hand_info["equity_realization"] else "bad"),
    ]

    blocker_lines = [
        ("✔ Blocks nut flush" if blockers["blocks_nut_flush"] else "✖ Does not block nut flush",
         "good" if blockers["blocks_nut_flush"] else "muted"),
        ("✔ Blocks AK" if blockers["blocks_ak"] else "✖ Does not block AK",
         "good" if blockers["blocks_ak"] else "muted"),
        ("✔ Blocks sets" if blockers["blocks_sets"] else "✖ Does not block sets",
         "good" if blockers["blocks_sets"] else "muted"),
    ]

    if not danger_flags:
        danger_lines = [("✔ No immediate danger flags", "good")]
    else:
        danger_lines = [(f"⚠ {flag}", "bad") for flag in danger_flags[:2]]
        if len(danger_flags) > 2:
            danger_lines.append((f"⚠ +{len(danger_flags) - 2} more flags", "bad"))

    return {
        "Board Texture": board_lines,
        "Draws Available": draws_lines,
        "Your Hand Classification": hand_lines,
        "Blockers": blocker_lines,
        "Danger Flags": danger_lines,
    }


def gen_cards(cards_str: List[str]) -> List[Card]:
    suit_map = {
        "C": Card.CLUB,
        "H": Card.HEART,
        "S": Card.SPADE,
        "D": Card.DIAMOND,
    }
    rank_map = {
        "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
        "10": 10, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14,
    }
    try:
        cards = []
        for card_str in cards_str:
            if len(card_str) == 3:
                rank = "10"
                suit = card_str[2]
            else:
                rank = card_str[0]
                suit = card_str[1]
            suit = suit_map[suit.upper()]
            rank = rank_map[rank.upper()]
            cards.append(Card(suit, rank))
        return cards
    except Exception as e:
        log_event("gen_cards_error", {"error": str(e), "cards": cards_str})
        return []


def read_pot_size(driver) -> float:
    try:
        total_pot_element = driver.find_element(By.CSS_SELECTOR, ".total-pot-amount")
        total_text = total_pot_element.text.strip()
        if total_text:
            return parse_money(total_text)
    except Exception:
        pass

    try:
        amount_cont_element = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.CLASS_NAME, "amount-cont"))
        )
        try:
            main_pot_element = amount_cont_element.find_element(By.ID, "main-pot")
        except NoSuchElementException:
            main_pot_element = driver.find_element(By.ID, "main-pot")
        main_pot_text = main_pot_element.text.strip()
        return parse_money(main_pot_text)
    except Exception as e:
        log_event("read_pot_error", {"error": str(e)})
        return 0.0


def get_cost_to_call(driver) -> float:
    try:
        call_button = driver.find_element(By.ID, "CALL")
        cost_value_element = call_button.find_element(By.CLASS_NAME, "action-value")
        cost_to_call_text = cost_value_element.text.strip()
        return parse_money(cost_to_call_text)
    except Exception:
        return 0.0


# --------------------------------------------------------------------------------------
# State tracking and policy
# --------------------------------------------------------------------------------------

@dataclass
class TableTracker:
    prev_contrib: List[float]
    prev_folded: List[bool]
    prev_pot: float
    prev_street: Optional[int]
    prev_board_count: int
    prev_action_texts: List[str]
    prev_stacks: List[float]
    last_logged_actions: List[Optional[Tuple[int, float, int, int, str]]]
    action_seq: List[Tuple[int, int, float]]
    initial_stacks: List[float]
    last_aggressor: int
    players_acted: List[bool]
    event_log: List[str]
    hand_id: int

    def __init__(self) -> None:
        self.prev_contrib = []
        self.prev_folded = []
        self.prev_pot = 0.0
        self.prev_street = None
        self.prev_board_count = 0
        self.prev_action_texts = []
        self.prev_stacks = []
        self.last_logged_actions = []
        self.action_seq = []
        self.initial_stacks = []
        self.last_aggressor = -1
        self.players_acted = []
        self.event_log = []
        self.hand_id = 0

    def reset_hand(self, seats: List[dict]) -> None:
        self.action_seq = []
        self.initial_stacks = [TABLE_STACK_SIZE for _ in seats]
        self.last_aggressor = -1
        self.players_acted = [
            True if s.get("folded") or (s.get("stack", 0.0) <= 0.0) else False
            for s in seats
        ]
        self.last_logged_actions = [None for _ in seats]
        self.event_log = ["New hand"]
        self.hand_id += 1
        append_history_event({
            "type": "hand_start",
            "hand_id": self.hand_id,
            "seats": [
                {
                    "seat_index": s.get("seat_index"),
                    "name": s.get("name") or "",
                    "label": self._format_player_label(s, s.get("seat_index", 0)),
                    "stack": s.get("stack", 0.0),
                    "is_hero": bool(s.get("is_hero")),
                }
                for s in seats
            ],
        })

    def _format_player_label(self, seat: dict, index: int) -> str:
        name = (seat.get("name") or "").strip()
        if not name:
            name = f"P{index + 1}"
        if seat.get("is_hero"):
            return f"{name}(H)"
        return name

    def _record_action_event(
        self,
        seat: dict,
        index: int,
        action: int,
        invested: float,
        pot_before: float,
        street: int,
        board_count: int,
    ) -> None:
        action_name = ACTION_ID_TO_NAME.get(action, str(action))
        player_label = self._format_player_label(seat, index)
        display_name = ACTION_DISPLAY_NAME.get(action, action_name)
        amount_text = ""
        if invested > 0.0:
            amount_text = f" {invested:.2f} ({format_bb(invested)})"
        if action == ACTION_ALL_IN:
            amount_text = f" {invested:.2f} ({format_bb(invested)})"
        street_label = STREET_LABELS.get(street, str(street))
        event = f"{player_label} {display_name}{amount_text} [{street_label}]"
        self.event_log.append(event)
        if len(self.event_log) > MAX_EVENT_LOG:
            self.event_log = self.event_log[-MAX_EVENT_LOG:]
        append_history_event({
            "type": "action",
            "hand_id": self.hand_id,
            "seat_index": index,
            "name": seat.get("name") or "",
            "label": self._format_player_label(seat, index),
            "is_hero": bool(seat.get("is_hero")),
            "action_id": action,
            "action_name": action_name,
            "display_name": display_name,
            "invested": round(invested, 6),
            "pot_before": round(pot_before, 6),
            "street": street,
            "board_count": board_count,
        })

    def get_action_chain(self, max_events: int = 10) -> str:
        if not self.event_log:
            return ""
        return " -> ".join(self.event_log[-max_events:])

    def _push_action(self, player: int, action: int, invested: float, pot_before: float) -> None:
        scale = 1.0
        if TABLE_BIG_BLIND and TABLE_BIG_BLIND > 0:
            scale = TRAINING_BIG_BLIND / TABLE_BIG_BLIND
        invested_scaled = invested * scale
        pot_before_scaled = pot_before * scale
        denom = max(1.0, pot_before_scaled)
        size_norm = 0.0
        if invested_scaled > 0.0:
            size_norm = min(invested_scaled / denom, 4.0) / 4.0
        self.action_seq.append((player, action, size_norm))
        if len(self.action_seq) > ACTION_SEQ_LEN:
            self.action_seq = self.action_seq[-ACTION_SEQ_LEN:]
        if action in (ACTION_BET_POT_25, ACTION_BET_POT_50, ACTION_BET_POT_100, ACTION_BET_POT_200, ACTION_ALL_IN):
            self.last_aggressor = player

    def _classify_raise(self, invested: float, to_call: float, pot_before: float, street: int) -> int:
        raise_size = max(0.0, invested - to_call)
        if raise_size <= 0.0:
            return ACTION_BET_POT_25
        denom = max(1.0, pot_before)
        frac = raise_size / denom
        if frac <= 0.375:
            return ACTION_BET_POT_25
        if frac <= 0.75:
            return ACTION_BET_POT_50
        if frac <= 1.5:
            return ACTION_BET_POT_100
        return ACTION_BET_POT_200

    def _action_from_text(self, action_text: str) -> Optional[int]:
        text = (action_text or "").strip().lower()
        if not text:
            return None
        if "fold" in text:
            return ACTION_FOLD
        if "check" in text:
            return ACTION_CHECK
        if "call" in text:
            return ACTION_CALL
        if "all-in" in text or "all in" in text:
            return ACTION_ALL_IN
        if "raise" in text or "bet" in text:
            return ACTION_BET_POT_50
        if "blind" in text or "post" in text:
            return None
        return None

    def update(self, seats: List[dict], pot: float, street: int, board_count: int, to_act: Optional[int]) -> None:
        n = len(seats)
        if n <= 0:
            return

        new_hand = False
        if self.prev_street is None:
            new_hand = True
        elif street < self.prev_street:
            new_hand = True
        elif self.prev_board_count > 0 and board_count == 0:
            new_hand = True
        elif street == STREET_PREFLOP and pot < self.prev_pot and self.prev_pot > 0.0:
            new_hand = True

        if new_hand:
            self.reset_hand(seats)

        street_changed = self.prev_street is not None and street != self.prev_street
        if street_changed:
            append_history_event({
                "type": "street_change",
                "hand_id": self.hand_id,
                "street": street,
                "board_count": board_count,
                "pot": round(pot, 6),
            })

        if new_hand:
            self.prev_contrib = [s.get("bet", 0.0) for s in seats]
            self.prev_folded = [s.get("folded", False) for s in seats]
            self.prev_action_texts = [s.get("action_text", "") for s in seats]
            self.prev_stacks = [s.get("stack", 0.0) for s in seats]
            self.last_logged_actions = [None for _ in seats]
            self.prev_pot = pot
            self.prev_street = street
            self.prev_board_count = board_count
        elif len(self.prev_contrib) != n or len(self.prev_action_texts) != n:
            self.prev_contrib = [s.get("bet", 0.0) for s in seats]
            self.prev_folded = [s.get("folded", False) for s in seats]
            self.prev_action_texts = [s.get("action_text", "") for s in seats]
            self.prev_stacks = [s.get("stack", 0.0) for s in seats]
            self.last_logged_actions = [None for _ in seats]
        else:
            prev_bet = max(self.prev_contrib) if self.prev_contrib else 0.0
            curr_contrib = [s.get("bet", 0.0) for s in seats]
            curr_bet = max(curr_contrib) if curr_contrib else 0.0
            curr_action_texts = [s.get("action_text", "") for s in seats]

            for i in range(n):
                action_text = curr_action_texts[i]
                prev_action_text = self.prev_action_texts[i]
                action_from_text = None
                if action_text and action_text != prev_action_text:
                    action_from_text = self._action_from_text(action_text)

                invested = max(0.0, curr_contrib[i] - self.prev_contrib[i])
                to_call = max(0.0, prev_bet - self.prev_contrib[i])

                def action_key_for(action_id: int, invested_amt: float, text: str) -> Tuple[int, float, int, int, str]:
                    if action_id == ACTION_FOLD:
                        text = "FOLD"
                    return (action_id, round(invested_amt, 6), street, board_count, text or "")

                if action_from_text == ACTION_FOLD and not self.prev_folded[i] and seats[i].get("folded"):
                    action_key = action_key_for(ACTION_FOLD, 0.0, action_text)
                    if self.last_logged_actions[i] != action_key:
                        self._push_action(i, ACTION_FOLD, 0.0, self.prev_pot)
                        self._record_action_event(seats[i], i, ACTION_FOLD, 0.0, self.prev_pot, street, board_count)
                        self.last_logged_actions[i] = action_key
                    self.players_acted[i] = True
                elif action_from_text is not None:
                    action = action_from_text
                    if action in (ACTION_BET_POT_25, ACTION_BET_POT_50, ACTION_BET_POT_100, ACTION_BET_POT_200):
                        action = self._classify_raise(invested, to_call, self.prev_pot, street)
                    if action == ACTION_ALL_IN and seats[i].get("stack", 0.0) > 1e-9 and invested <= 0.0:
                        action = self._classify_raise(invested, to_call, self.prev_pot, street)
                    action_key = action_key_for(action, invested, action_text)
                    if self.last_logged_actions[i] != action_key:
                        self._push_action(i, action, invested, self.prev_pot)
                        self._record_action_event(seats[i], i, action, invested, self.prev_pot, street, board_count)
                        self.last_logged_actions[i] = action_key
                    self.players_acted[i] = True
                elif invested > 0.0:
                    if seats[i].get("stack", 0.0) <= 1e-9:
                        action = ACTION_ALL_IN
                    elif curr_contrib[i] >= curr_bet and curr_bet > prev_bet and curr_contrib[i] == curr_bet:
                        action = self._classify_raise(invested, to_call, self.prev_pot, street)
                    else:
                        action = ACTION_CALL
                    action_key = action_key_for(action, invested, action_text)
                    if self.last_logged_actions[i] != action_key:
                        self._push_action(i, action, invested, self.prev_pot)
                        self._record_action_event(seats[i], i, action, invested, self.prev_pot, street, board_count)
                        self.last_logged_actions[i] = action_key
                    self.players_acted[i] = True
                elif not self.prev_folded[i] and seats[i].get("folded"):
                    action_key = action_key_for(ACTION_FOLD, 0.0, action_text)
                    if self.last_logged_actions[i] != action_key:
                        self._push_action(i, ACTION_FOLD, 0.0, self.prev_pot)
                        self._record_action_event(seats[i], i, ACTION_FOLD, 0.0, self.prev_pot, street, board_count)
                        self.last_logged_actions[i] = action_key
                    self.players_acted[i] = True

            self.prev_contrib = curr_contrib
            self.prev_folded = [s.get("folded", False) for s in seats]
            self.prev_action_texts = curr_action_texts
            self.prev_stacks = [s.get("stack", 0.0) for s in seats]

        self.prev_pot = pot
        self.prev_street = street
        self.prev_board_count = board_count
        if street_changed or new_hand:
            self.players_acted = [
                True if seats[i].get("folded") or seats[i].get("stack", 0.0) <= 0.0 else False
                for i in range(n)
            ]
        if to_act is not None and 0 <= to_act < n and not seats[to_act].get("folded") and seats[to_act].get("stack", 0.0) > 0.0:
            self.players_acted[to_act] = False


def build_game_state(snapshot: dict, hole_cards: List[str], community_cards: List[str], pot: float, tracker: TableTracker, hero_cost_to_call: float = 0.0) -> Tuple[Optional[GameState], Optional[int]]:
    if not snapshot.get("seats"):
        return None, None
    seats = snapshot["seats"]
    hero_index = snapshot.get("hero_index", 0)
    to_act = snapshot.get("to_act")
    if to_act is None:
        to_act = hero_index

    street = street_from_board(community_cards)
    tracker.update(seats, pot, street, len(community_cards), to_act)

    scale = 1.0
    if TABLE_BIG_BLIND and TABLE_BIG_BLIND > 0:
        scale = TRAINING_BIG_BLIND / TABLE_BIG_BLIND

    stacks = [s["stack"] * scale for s in seats]
    contrib = [s["bet"] * scale for s in seats]
    folded = [s["folded"] for s in seats]

    hole = [[] for _ in range(len(seats))]
    hero_cards = cards_str_to_ids(hole_cards)
    if len(hero_cards) == 2:
        hole[hero_index] = hero_cards

    board = cards_str_to_ids(community_cards)
    current_bet = max(contrib) if contrib else 0.0
    if hero_cost_to_call and to_act == hero_index:
        current_bet = max(current_bet, contrib[hero_index] + hero_cost_to_call * scale)
    if tracker.initial_stacks:
        initial_stacks = [v * scale for v in tracker.initial_stacks]
    else:
        initial_stacks = [TABLE_STACK_SIZE * scale for _ in stacks]

    state = GameState(
        deck=[],
        board=board,
        hole=hole,
        pot=pot * scale,
        to_act=to_act,
        street=street,
        stacks=stacks,
        current_bet=current_bet,
        last_aggressor=tracker.last_aggressor,
        sb_player=snapshot.get("sb", 0),
        bb_player=snapshot.get("bb", 1),
        button_player=snapshot.get("button", 0),
        initial_stacks=initial_stacks,
        contrib=contrib,
        folded=folded,
        players_acted=tracker.players_acted if len(tracker.players_acted) == len(stacks) else [
            True if folded[i] or stacks[i] <= 0.0 else False for i in range(len(stacks))
        ],
        num_players=len(stacks),
        actions_this_street=0,
        terminal=False,
        winner=-1,
        action_seq=tracker.action_seq,
    )
    return state, hero_index


def load_policy_net(state_dim: int, path: str) -> PolicyNet:
    net = PolicyNet(state_dim)
    state_dict = torch.load(path, map_location=DEVICE)
    net.load_state_dict(state_dict)
    net.to(DEVICE)
    net.eval()
    return net

def load_adv_net(state_dim: int, path: str) -> AdvantageNet:
    net = AdvantageNet(state_dim)
    state_dict = torch.load(path, map_location=DEVICE)
    net.load_state_dict(state_dict)
    net.to(DEVICE)
    net.eval()
    return net


def ensure_policy_loaded(num_players: int, policy_net: Optional[PolicyNet], policy_env: Optional[SimpleHoldemEnv]) -> Tuple[PolicyNet, SimpleHoldemEnv]:
    if policy_net is not None and policy_env is not None and policy_env.num_players == num_players:
        return policy_net, policy_env
    policy_env = SimpleHoldemEnv(
        stack_size=TRAINING_STACK_SIZE,
        sb=TRAINING_SMALL_BLIND,
        bb=TRAINING_BIG_BLIND,
        num_players=num_players,
    )
    dummy = policy_env.new_hand()
    state_dim = encode_state(dummy, 0).shape[0]
    policy_net = load_policy_net(state_dim, POLICY_PATH)
    return policy_net, policy_env

def ensure_adv_loaded(
    num_players: int,
    adv_nets: Optional[List[AdvantageNet]],
    policy_env: SimpleHoldemEnv,
) -> List[AdvantageNet]:
    if adv_nets is not None and policy_env.num_players == num_players:
        return adv_nets
    dummy = policy_env.new_hand()
    state_dim = encode_state(dummy, 0).shape[0]
    adv_nets = [load_adv_net(state_dim, path) for path in ADV_POLICY_PATHS]
    return adv_nets


def get_policy_action_probs(policy_net: PolicyNet, state: GameState, hero_index: int) -> List[float]:
    x = encode_state(state, hero_index).float().unsqueeze(0)
    with torch.no_grad():
        logits = policy_net(x).squeeze(0)
    probs = torch.softmax(logits, dim=-1).tolist()
    return probs

def get_decision_probs(
    policy_net: PolicyNet,
    adv_nets: Optional[List[AdvantageNet]],
    state: GameState,
    hero_index: int,
    legal_actions: Optional[List[int]],
    use_advantage: bool,
) -> Tuple[List[float], str]:
    x = encode_state(state, hero_index).float().to(DEVICE)
    mask = build_legal_mask(legal_actions, NUM_ACTIONS).to(DEVICE)
    with torch.no_grad():
        if use_advantage and adv_nets is not None:
            adv_vals = []
            for net in adv_nets:
                adv_vals.append(net(x.unsqueeze(0)).squeeze(0))
            adv = torch.stack(adv_vals, dim=0).median(dim=0).values
            pos = torch.clamp(adv, min=0.0)
            probs = masked_normalize_tensor(pos, mask)
            mode = "adv"
        else:
            logits = policy_net(x.unsqueeze(0)).squeeze(0)
            probs = torch.softmax(logits, dim=-1)
            probs = masked_normalize_tensor(probs, mask)
            mode = "policy"
    return probs.detach().cpu().tolist(), mode


def state_vector_summary(state: GameState, hero_index: int) -> Dict[str, float]:
    try:
        vec = encode_state(state, hero_index).float()
        return {
            "len": int(vec.shape[0]),
            "min": float(vec.min().item()),
            "max": float(vec.max().item()),
            "mean": float(vec.mean().item()),
        }
    except Exception as e:
        return {"error": str(e)}


def format_policy_probs(probs: List[float], label: str = "Policy probs") -> str:
    parts = []
    for a in range(NUM_ACTIONS):
        name = ACTION_ID_TO_NAME.get(a, f"ACT_{a}")
        parts.append(f"{name}={probs[a]:.3f}")
    return f"{label}: " + ", ".join(parts)

def topk_actions(probs: List[float], k: int = 3) -> List[dict]:
    ranked = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:k]
    return [{"action": ACTION_ID_TO_NAME.get(i, i), "prob": float(probs[i])} for i in ranked]


def sample_action_from_probs(probs: List[float]) -> int:
    if not probs:
        return ACTION_CHECK
    r = random.random()
    cum = 0.0
    for idx, p in enumerate(probs):
        cum += max(0.0, p)
        if r <= cum:
            return idx
    return int(max(range(len(probs)), key=lambda i: probs[i]))

def select_action_index(probs: List[float], sample_prob: float) -> int:
    if not probs:
        return ACTION_CHECK
    sample_prob = min(max(sample_prob, 0.0), 1.0)
    if random.random() < sample_prob:
        return sample_action_from_probs(probs)
    return int(max(range(len(probs)), key=lambda i: probs[i]))

def is_automation_turn(state: Optional[GameState], hero_index: Optional[int], to_act: Optional[int]) -> bool:
    if state is None or hero_index is None:
        return False
    return to_act == hero_index and state.street in AUTOMATION_STREETS


def log_full_snapshot(snapshot: dict,
                      hole_cards: List[str],
                      community_cards: List[str],
                      pot: float,
                      equity: float,
                      cct: float,
                      state: Optional[GameState],
                      probs: Optional[List[float]],
                      legal_actions: Optional[List[int]]) -> None:
    if not LOG_FULL_STATE:
        return

    hero_index = snapshot.get("hero_index")
    to_act = snapshot.get("to_act")
    seats = snapshot.get("seats", [])
    seat_dump = []
    for s in seats:
        seat_dump.append({
            "seat_index": s.get("seat_index"),
            "name": s.get("name"),
            "stack": s.get("stack"),
            "bet": s.get("bet"),
            "folded": s.get("folded"),
            "is_hero": s.get("is_hero"),
            "is_button": s.get("is_button"),
            "is_sb": s.get("is_sb"),
            "is_bb": s.get("is_bb"),
            "is_active": s.get("is_active"),
            "raw_stack_text": s.get("raw_stack_text"),
            "raw_bet_text": s.get("raw_bet_text"),
            "raw_name_text": s.get("raw_name_text"),
            "class_name": s.get("class_name"),
            "action_text": s.get("action_text"),
        })

    to_call = None
    hero_stack = None
    if state is not None and hero_index is not None and hero_index >= 0:
        to_call = max(0.0, state.current_bet - state.contrib[hero_index])
        hero_stack = state.stacks[hero_index]

    probs_named = {}
    if probs is not None:
        for a in range(NUM_ACTIONS):
            probs_named[ACTION_ID_TO_NAME.get(a, f"ACT_{a}")] = probs[a]

    sum_bets = sum(s.get("bet") or 0.0 for s in seats)
    sum_stacks = sum(s.get("stack") or 0.0 for s in seats)

    scale = 1.0
    if TABLE_BIG_BLIND and TABLE_BIG_BLIND > 0:
        scale = TRAINING_BIG_BLIND / TABLE_BIG_BLIND

    payload = {
        "hero_index": hero_index,
        "to_act": to_act,
        "num_players": snapshot.get("num_players"),
        "button": snapshot.get("button"),
        "sb": snapshot.get("sb"),
        "bb": snapshot.get("bb"),
        "seat_elem_count": snapshot.get("seat_elem_count"),
        "scale_to_training_bb": scale,
        "table_sb": TABLE_SMALL_BLIND,
        "table_bb": TABLE_BIG_BLIND,
        "train_sb": TRAINING_SMALL_BLIND,
        "train_bb": TRAINING_BIG_BLIND,
        "pot_raw": pot,
        "pot_scaled": pot * scale,
        "sum_bets": sum_bets,
        "sum_stacks": sum_stacks,
        "equity": equity,
        "cost_to_call_raw": cct,
        "cost_to_call_scaled": cct * scale,
        "hole_cards": hole_cards,
        "community_cards": community_cards,
        "legal_actions": legal_actions,
        "probs": probs_named,
        "to_call": to_call,
        "hero_stack": hero_stack,
        "state": None,
        "state_vec": None,
        "seats": seat_dump,
    }
    if state is not None:
        payload["state"] = {
            "street": state.street,
            "pot": state.pot,
            "current_bet": state.current_bet,
            "last_aggressor": state.last_aggressor,
            "stacks": state.stacks,
            "contrib": state.contrib,
            "folded": state.folded,
            "players_acted": state.players_acted,
            "action_seq": state.action_seq,
            "board": state.board,
            "hero_hole": state.hole[hero_index] if hero_index is not None else None,
        }
        payload["state_vec"] = state_vector_summary(state, hero_index)

    logger.info(json.dumps(payload, ensure_ascii=True))


# --------------------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------------------

class MonitorUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Poker Live Command Center")
        self.root.geometry("1280x980")
        self.root.configure(bg="#0e1116")

        self.font_title = ("Segoe UI", 14, "bold")
        self.font_header = ("Segoe UI", 11, "bold")
        self.font_body = ("Segoe UI", 10)
        self.font_mono = ("Consolas", 9)

        header_frame = tk.Frame(self.root, bg="#0e1116")
        header_frame.pack(fill="x", padx=10, pady=(10, 6))

        self.info_var = tk.StringVar(value="Waiting...")
        self.info_label = tk.Label(
            header_frame,
            textvariable=self.info_var,
            fg="#f5f7fb",
            bg="#0e1116",
            font=self.font_title,
        )
        self.info_label.pack(anchor="w")

        metrics_frame = tk.Frame(self.root, bg="#0e1116")
        metrics_frame.pack(fill="x", padx=10, pady=(0, 6))

        self.metrics_var = tk.StringVar(value="Equity: - | Pot odds: - | SPR: - | Hero stack: -")
        self.metrics_label = tk.Label(
            metrics_frame,
            textvariable=self.metrics_var,
            fg="#b8c1cc",
            bg="#0e1116",
            font=self.font_body,
        )
        self.metrics_label.pack(anchor="w")

        actions_frame = tk.Frame(self.root, bg="#151a23")
        actions_frame.pack(fill="x", padx=10, pady=(0, 8))
        actions_frame.configure(highlightthickness=1, highlightbackground="#212938")

        self.argmax_var = tk.StringVar(value="Argmax: -")
        self.argmax_label = tk.Label(
            actions_frame,
            textvariable=self.argmax_var,
            fg="#37d8c3",
            bg="#151a23",
            font=self.font_header,
        )
        self.argmax_label.pack(anchor="w", padx=10, pady=(6, 0))

        self.sampled_var = tk.StringVar(value="Sampled: -")
        self.sampled_label = tk.Label(
            actions_frame,
            textvariable=self.sampled_var,
            fg="#ffa657",
            bg="#151a23",
            font=self.font_body,
        )
        self.sampled_label.pack(anchor="w", padx=10, pady=(0, 4))

        self.prob_var = tk.StringVar(value="Action mix: -")
        self.prob_label = tk.Label(
            actions_frame,
            textvariable=self.prob_var,
            fg="#e6edf3",
            bg="#151a23",
            font=self.font_body,
        )
        self.prob_label.pack(anchor="w", padx=10, pady=(0, 6))

        main_frame = tk.Frame(self.root, bg="#0e1116")
        main_frame.pack(fill="both", expand=True, padx=10, pady=(0, 8))

        left_frame = tk.Frame(main_frame, bg="#0e1116")
        left_frame.pack(side="left", fill="both", expand=True)

        right_frame = tk.Frame(main_frame, bg="#0e1116")
        right_frame.pack(side="right", fill="y")

        self.color_good = "#37d8c3"
        self.color_warn = "#f4b267"
        self.color_bad = "#ff6b6b"
        self.color_muted = "#93a4b7"

        self.intel_frame = tk.Frame(left_frame, bg="#151a23", highlightthickness=1, highlightbackground="#212938")
        self.intel_frame.pack(fill="x", pady=(0, 8))
        intel_title = tk.Label(
            self.intel_frame,
            text="Hand & Board Intelligence",
            fg="#8ec9ff",
            bg="#151a23",
            font=self.font_header,
        )
        intel_title.pack(anchor="w", padx=10, pady=(6, 0))

        self.intel_sections = {}
        section_order = [
            ("Board Texture", 4),
            ("Draws Available", 4),
            ("Your Hand Classification", 3),
            ("Blockers", 3),
            ("Danger Flags", 3),
        ]
        for title, line_count in section_order:
            section_frame = tk.Frame(self.intel_frame, bg="#151a23")
            section_frame.pack(fill="x", padx=10, pady=(4, 0))
            label = tk.Label(
                section_frame,
                text=title,
                fg="#c7d1db",
                bg="#151a23",
                font=self.font_body,
            )
            label.pack(anchor="w")
            lines_frame = tk.Frame(self.intel_frame, bg="#151a23")
            lines_frame.pack(fill="x", padx=18, pady=(0, 2))
            labels = []
            for _ in range(line_count):
                line = tk.Label(
                    lines_frame,
                    text="",
                    fg=self.color_muted,
                    bg="#151a23",
                    font=self.font_body,
                    anchor="w",
                    justify="left",
                )
                line.pack(anchor="w")
                labels.append(line)
            self.intel_sections[title] = labels

        self.table_canvas = None
        if SHOW_TABLE_VIEW:
            self.table_canvas = tk.Canvas(left_frame, width=980, height=430, bg="#111820", highlightthickness=0)
            self.table_canvas.pack(fill="x", pady=(0, 8))

        action_chain_frame = tk.Frame(right_frame, bg="#151a23", highlightthickness=1, highlightbackground="#212938")
        action_chain_frame.pack(fill="x", pady=(0, 8))
        chain_title = tk.Label(
            action_chain_frame,
            text="Action Chain",
            fg="#8ec9ff",
            bg="#151a23",
            font=self.font_header,
        )
        chain_title.pack(anchor="w", padx=10, pady=(6, 0))
        self.action_chain_var = tk.StringVar(value="Waiting for actions...")
        self.action_chain_label = tk.Label(
            action_chain_frame,
            textvariable=self.action_chain_var,
            fg="#e6edf3",
            bg="#151a23",
            font=self.font_body,
            wraplength=260,
            justify="left",
        )
        self.action_chain_label.pack(anchor="w", padx=10, pady=(0, 6))

        event_frame = tk.Frame(right_frame, bg="#151a23", highlightthickness=1, highlightbackground="#212938")
        event_frame.pack(fill="both", expand=True)
        event_title = tk.Label(
            event_frame,
            text="Recent Events",
            fg="#8ec9ff",
            bg="#151a23",
            font=self.font_header,
        )
        event_title.pack(anchor="w", padx=10, pady=(6, 0))
        self.event_text = tk.Text(
            event_frame,
            height=18,
            width=34,
            bg="#0f141c",
            fg="#c7d1db",
            font=self.font_mono,
            relief="flat",
        )
        self.event_text.pack(fill="both", expand=True, padx=10, pady=(0, 8))
        self.event_text.configure(state="disabled")

        self.state_text = tk.Text(self.root, height=16, width=120, bg="#0f141c", fg="#c7d1db", font=self.font_mono)
        self.state_text.pack(fill="both", expand=True, padx=10, pady=(0, 8))

        self.seat_text = tk.Text(self.root, height=14, width=120, bg="#0f141c", fg="#c7d1db", font=self.font_mono)
        self.seat_text.pack(fill="both", expand=True, padx=10, pady=(0, 8))

        self.input_table = None
        if SHOW_INPUT_TABLE:
            columns = (
                "seat",
                "name",
                "to_act",
                "folded",
                "stack_raw",
                "bet_raw",
                "stack_sb",
                "bet_sb",
                "stack_scaled",
                "bet_scaled",
            )
            self.input_table = ttk.Treeview(self.root, columns=columns, show="headings", height=7)
            for col in columns:
                self.input_table.heading(col, text=col)
                self.input_table.column(col, width=90, anchor="center")
            self.input_table.pack(fill="x", padx=10, pady=(0, 10))

    def _draw_table_view(self, table_view: dict) -> None:
        if not self.table_canvas:
            return
        canvas = self.table_canvas
        canvas.delete("all")

        width = int(canvas["width"])
        height = int(canvas["height"])
        table_margin = 40
        canvas.create_oval(
            table_margin,
            40,
            width - table_margin,
            height - 40,
            fill="#2c6b2f",
            outline="#17401a",
            width=4,
        )

        pot_text = table_view.get("pot_text", "Pot: 0")
        board_text = table_view.get("board_text", "")
        canvas.create_text(width // 2, height // 2 - 30, text=pot_text, fill="white", font=("Arial", 14, "bold"))
        canvas.create_text(width // 2, height // 2, text=board_text, fill="white", font=("Arial", 12))

        seats = table_view.get("seats", [])
        positions = SEAT_POSITIONS_6
        for seat in seats:
            idx = seat.get("seat_index", 0)
            if idx < 0 or idx >= len(positions):
                continue
            x, y = positions[idx]
            name = seat.get("name") or f"Seat {idx}"
            stack_sb = seat.get("stack_sb", 0.0)
            bet_sb = seat.get("bet_sb", 0.0)
            action_text = seat.get("action_text", "")
            flags = []
            if seat.get("is_hero"):
                flags.append("HERO")
            if seat.get("is_button"):
                flags.append("BTN")
            if seat.get("is_sb"):
                flags.append("SB")
            if seat.get("is_bb"):
                flags.append("BB")
            if seat.get("folded"):
                flags.append("FOLD")
            if seat.get("is_active"):
                flags.append("TURN")
            flag_text = " ".join(flags)
            info = f"{name}\nstack={stack_sb:.2f}sb bet={bet_sb:.2f}sb"
            if flag_text:
                info += f"\n{flag_text}"
            if action_text:
                info += f"\n{action_text}"
            fill = "#0f141c"
            outline = "#2c3648"
            if seat.get("folded"):
                fill = "#1a1f29"
                outline = "#3a404d"
            if seat.get("is_active"):
                fill = "#2a1c10"
                outline = "#ff9f43"
            if seat.get("is_hero"):
                fill = "#142a4a"
                outline = "#5aa6ff"
            canvas.create_rectangle(x - 80, y - 40, x + 80, y + 40, fill=fill, outline=outline, width=2)
            canvas.create_text(x, y, text=info, fill="white", font=("Arial", 9), justify="center")

    def _update_input_table(self, rows: List[dict]) -> None:
        if not self.input_table:
            return
        for item in self.input_table.get_children():
            self.input_table.delete(item)
        for row in rows:
            values = (
                row.get("seat"),
                row.get("name"),
                row.get("to_act"),
                row.get("folded"),
                row.get("stack_raw"),
                row.get("bet_raw"),
                row.get("stack_sb"),
                row.get("bet_sb"),
                row.get("stack_scaled"),
                row.get("bet_scaled"),
            )
            self.input_table.insert("", "end", values=values)

    def _update_intel_panel(self, intel_data: Optional[dict]) -> None:
        if not intel_data:
            for labels in self.intel_sections.values():
                for label in labels:
                    label.configure(text="", fg=self.color_muted)
            return
        for section, labels in self.intel_sections.items():
            items = intel_data.get(section, [])
            for idx, label in enumerate(labels):
                if idx < len(items):
                    text, color = items[idx]
                    label.configure(text=text, fg=color)
                else:
                    label.configure(text="", fg=self.color_muted)

    def update(
        self,
        header: str,
        state_lines: List[str],
        seat_lines: List[str],
        table_view: Optional[dict] = None,
        input_rows: Optional[List[dict]] = None,
        argmax_text: Optional[str] = None,
        sampled_text: Optional[str] = None,
        metrics_text: Optional[str] = None,
        action_chain: Optional[str] = None,
        event_lines: Optional[List[str]] = None,
        probs_text: Optional[str] = None,
        intel_data: Optional[dict] = None,
    ):
        self.info_var.set(header)
        if argmax_text is not None:
            self.argmax_var.set(argmax_text)
        if sampled_text is not None:
            self.sampled_var.set(sampled_text)
        if probs_text is not None:
            self.prob_var.set(probs_text)
        if metrics_text is not None:
            self.metrics_var.set(metrics_text)
        if action_chain is not None:
            self.action_chain_var.set(action_chain or "Waiting for actions...")
        if event_lines is not None:
            self.event_text.configure(state="normal")
            self.event_text.delete("1.0", "end")
            self.event_text.insert("end", "\n".join(event_lines))
            self.event_text.configure(state="disabled")
        if intel_data is not None:
            self._update_intel_panel(intel_data)
        if table_view:
            self._draw_table_view(table_view)
        self.state_text.configure(state="normal")
        self.state_text.delete("1.0", "end")
        self.state_text.insert("end", "\n".join(state_lines))
        self.state_text.configure(state="disabled")

        self.seat_text.configure(state="normal")
        self.seat_text.delete("1.0", "end")
        self.seat_text.insert("end", "\n".join(seat_lines))
        self.seat_text.configure(state="disabled")
        if input_rows is not None:
            self._update_input_table(input_rows)

    def run_loop(self):
        self.root.mainloop()


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def main():
    global LOG_TICK

    setup_logger()
    init_history_logging()
    if LOG_TABLE_TO_FILE:
        print(f"Logging to {LOG_FILE_PATH}")
    if TABLE_HISTORY_LOG_PATH:
        print(f"History log: {TABLE_HISTORY_LOG_PATH}")
    log_event("Logging initialized", {"log_file": LOG_FILE_PATH, "log_console": LOG_PRINT_CONSOLE})

    driver = webdriver.Chrome()
    driver.maximize_window()
    driver.get(URL)

    ui = MonitorUI()

    policy_net = None
    policy_env = None
    adv_nets = None
    table_tracker = TableTracker()

    def tick():
        nonlocal policy_net, policy_env, adv_nets
        global LOG_TICK, AUTO_DECISION_KEY
        LOG_TICK += 1

        try:
            if REQUIRE_ACTION_BUTTONS:
                action_button = None
                for button_id in ["FOLD", "CHECK", "RAISE_TO"]:
                    try:
                        action_button = WebDriverWait(driver, 2).until(
                            EC.presence_of_element_located((By.ID, button_id))
                        )
                        if action_button:
                            break
                    except Exception:
                        pass
                    if not action_button:
                        log_event("No actionable buttons detected")
                        ui.update("Waiting for action buttons...", ["No actionable buttons yet."], [], intel_data={})
                        ui.root.after(UI_REFRESH_MS, tick)
                        return

            hole_cards, hole_debug = read_cards_hero_with_debug(driver)
            community_cards = read_community_cards(driver)
            if len(hole_cards) != 2:
                log_event("Unable to read hole cards", {"hole_cards": hole_cards, "hole_selectors": hole_debug})
                snapshot = read_table_snapshot(driver, force_hero_to_act=False)
                if snapshot:
                    seat_lines = []
                    for s in snapshot.get("seats", []):
                        seat_lines.append(
                            f"Seat {s.get('seat_index')} | name={s.get('name')} | stack={s.get('stack'):.2f} | bet={s.get('bet'):.2f} | folded={s.get('folded')} | active={s.get('is_active')} | btn={s.get('is_button')}"
                        )
                    state_lines = [
                        "Hole cards not visible yet.",
                        f"Hole selector counts: {hole_debug}",
                    ]
                    ui.update("Waiting for hole cards...", state_lines, seat_lines, intel_data={})
                else:
                    ui.update("Waiting for table...", [f"No table snapshot yet. Hole selectors: {hole_debug}"], [], intel_data={})
                ui.root.after(UI_REFRESH_MS, tick)
                return

            hole_card_objects = gen_cards(hole_cards)
            community_card_objects = gen_cards(community_cards) if community_cards else []

            if len(community_cards) == 0:
                equity = estimate_hole_card_win_rate(
                    nb_simulation=1500,
                    nb_player=TABLE_NUM_PLAYERS,
                    hole_card=hole_card_objects,
                    community_card=[]
                )
            else:
                equity = estimate_hole_card_win_rate(
                    nb_simulation=1500,
                    nb_player=TABLE_NUM_PLAYERS,
                    hole_card=hole_card_objects,
                    community_card=community_card_objects
                )

            cct = get_cost_to_call(driver)
            potsize = read_pot_size(driver)
            scale_factor = 1.0
            if TABLE_BIG_BLIND and TABLE_BIG_BLIND > 0:
                scale_factor = TRAINING_BIG_BLIND / TABLE_BIG_BLIND

            snapshot = read_table_snapshot(driver, force_hero_to_act=True)
            if not snapshot:
                log_event("No table snapshot; adjust selectors.")
                ui.update("No table snapshot", ["Check selectors or wait for table load."], [], intel_data={})
                ui.root.after(UI_REFRESH_MS, tick)
                return

            policy_net, policy_env = ensure_policy_loaded(snapshot["num_players"], policy_net, policy_env)
            if USE_ADVANTAGE_NET:
                adv_nets = ensure_adv_loaded(snapshot["num_players"], adv_nets, policy_env)
            state, hero_index = build_game_state(snapshot, hole_cards, community_cards, potsize, table_tracker, hero_cost_to_call=cct)

            probs = None
            legal_actions = None
            decision_mode = None
            if state is not None and snapshot.get("to_act") == hero_index:
                legal_actions = policy_env.legal_actions(state)
                probs, decision_mode = get_decision_probs(
                    policy_net,
                    adv_nets,
                    state,
                    hero_index,
                    legal_actions,
                    USE_ADVANTAGE_NET,
                )
                if LOG_TICK % LOG_EVERY_N == 0:
                    log_full_snapshot(snapshot, hole_cards, community_cards, potsize, equity, cct, state, probs, legal_actions)
            else:
                if state is None:
                    log_event("State build failed", {"snapshot": snapshot})
            if AUTOMATION and probs is not None and is_automation_turn(state, hero_index, snapshot.get("to_act")):
                decision_key = (table_tracker.hand_id, state.street, hero_index, len(state.action_seq))
                if AUTO_DECISION_KEY != decision_key:
                    AUTO_DECISION_KEY = decision_key
                    if AUTO_ACTION_MAX_DELAY_S > 0:
                        delay_s = random.uniform(AUTO_ACTION_MIN_DELAY_S, AUTO_ACTION_MAX_DELAY_S)
                        time.sleep(max(0.0, delay_s))
                    action_probs = probs
                    fold_prob = action_probs[ACTION_FOLD] if ACTION_FOLD < len(action_probs) else 0.0
                    chosen_action = select_action_index(action_probs, ACTION_SAMPLE_PROB)
                    if fold_prob >= FOLD_PROB_THRESHOLD:
                        did_fold = click_action_button(driver, "FOLD")
                        log_event(
                            "auto_fold_prob",
                            {
                                "fold_prob": fold_prob,
                                "fold_clicked": did_fold,
                                "chosen_action": chosen_action,
                                "mode": decision_mode,
                                "topk": topk_actions(action_probs),
                            },
                        )
                        if not did_fold:
                            AUTO_DECISION_KEY = None
                    else:
                        if state is not None and state.street == STREET_PREFLOP:
                            play_airbus_alert()
                        log_event(
                            "auto_hold",
                            {
                                "fold_prob": fold_prob,
                                "chosen_action": chosen_action,
                                "mode": decision_mode,
                                "topk": topk_actions(action_probs),
                            },
                        )

            header = (
                f"Hero seat: {snapshot.get('hero_index')} | To act: {snapshot.get('to_act')} | "
                f"Pot raw: {potsize:.2f} | Pot scaled: {potsize * scale_factor:.2f} | "
                f"Table SB/BB: {TABLE_SMALL_BLIND:.2f}/{TABLE_BIG_BLIND:.2f}"
            )
            hero_stack_raw = 0.0
            if hero_index is not None and 0 <= hero_index < len(snapshot.get("seats", [])):
                hero_stack_raw = snapshot["seats"][hero_index].get("stack", 0.0) or 0.0
            hero_stack_bb = hero_stack_raw / TABLE_BIG_BLIND if TABLE_BIG_BLIND else 0.0
            spr = hero_stack_raw / max(potsize, TABLE_BIG_BLIND or 1.0) if hero_stack_raw else 0.0
            pot_odds = 0.0
            if cct > 0.0:
                pot_odds = cct / max(potsize + cct, 1e-9)
            equity_pct = equity * 100.0
            metrics_text = (
                f"Street: {STREET_LABELS.get(state.street, state.street) if state else '-'} | "
                f"Equity: {equity_pct:.1f}% | Required: {pot_odds * 100.0:.1f}% | "
                f"SPR: {spr:.2f} | Hero stack: {hero_stack_bb:.1f}bb"
            )
            state_lines = []
            if state is None:
                state_lines.append("State: None")
            else:
                to_call = max(0.0, state.current_bet - state.contrib[hero_index])
                state_lines.extend([
                    f"Street: {state.street}",
                    f"Current bet (scaled): {state.current_bet:.2f}",
                    f"To call raw: {cct:.2f}",
                    f"To call scaled: {to_call:.2f}",
                    f"Pot (scaled): {state.pot:.2f}",
                    f"Scale factor: {scale_factor:.4f} (train BB {TRAINING_BIG_BLIND} / table BB {TABLE_BIG_BLIND})",
                    f"Last aggressor: {state.last_aggressor}",
                    f"Action seq (last {ACTION_SEQ_LEN}): {state.action_seq}",
                    f"Hole cards: {hole_cards}",
                    f"Board: {community_cards}",
                ])
                if legal_actions is not None:
                    state_lines.append(f"Legal actions (env): {legal_actions}")
            argmax_text = None
            sampled_text = None
            probs_text = None
            if probs is not None:
                argmax_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
                sampled_idx = sample_action_from_probs(probs)
                argmax_text = f"Argmax action: {ACTION_ID_TO_NAME.get(argmax_idx, argmax_idx)} ({probs[argmax_idx]:.3f})"
                sampled_text = f"Sampled action: {ACTION_ID_TO_NAME.get(sampled_idx, sampled_idx)} ({probs[sampled_idx]:.3f})"
                ranked = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:3]
                probs_text = "Action mix: " + " | ".join(
                    f"{ACTION_ID_TO_NAME.get(i, i)} {probs[i]:.2f}" for i in ranked
                )
                label = f"Decision probs ({decision_mode or 'policy'})"
                state_lines.append(format_policy_probs(probs, label=label))
            elif state is not None:
                state_lines.append("Decision probs: waiting for your turn")

            seat_lines = []
            for s in snapshot.get("seats", []):
                seat_lines.append(
                    f"Seat {s.get('seat_index')} | name={s.get('name')} | stack={s.get('stack'):.2f} | bet={s.get('bet'):.2f} | folded={s.get('folded')} | active={s.get('is_active')} | btn={s.get('is_button')}"
                )

            sb_div = TABLE_SMALL_BLIND if TABLE_SMALL_BLIND > 0 else 1.0
            table_view = None
            if SHOW_TABLE_VIEW:
                table_view = {
                    "pot_text": f"Pot: {format_bb(potsize)} ({potsize:.2f})",
                    "board_text": "Board: " + (" ".join(community_cards) if community_cards else "[]"),
                    "seats": [],
                }
                for s in snapshot.get("seats", []):
                    table_view["seats"].append({
                        "seat_index": s.get("seat_index"),
                        "name": s.get("name") or "",
                        "stack_sb": (s.get("stack") or 0.0) / sb_div,
                        "bet_sb": (s.get("bet") or 0.0) / sb_div,
                        "folded": s.get("folded"),
                        "is_hero": s.get("is_hero"),
                        "is_button": s.get("is_button"),
                        "is_sb": s.get("is_sb"),
                        "is_bb": s.get("is_bb"),
                        "is_active": s.get("is_active"),
                        "action_text": s.get("action_text") or "",
                    })

            input_rows = None
            if SHOW_INPUT_TABLE:
                input_rows = []
                for s in snapshot.get("seats", []):
                    stack_raw = s.get("stack") or 0.0
                    bet_raw = s.get("bet") or 0.0
                    input_rows.append({
                        "seat": s.get("seat_index"),
                        "name": s.get("name") or "",
                        "to_act": "Y" if s.get("seat_index") == snapshot.get("to_act") else "",
                        "folded": "Y" if s.get("folded") else "",
                        "stack_raw": f"{stack_raw:.2f}",
                        "bet_raw": f"{bet_raw:.2f}",
                        "stack_sb": f"{stack_raw / sb_div:.2f}",
                        "bet_sb": f"{bet_raw / sb_div:.2f}",
                        "stack_scaled": f"{stack_raw * scale_factor:.2f}",
                        "bet_scaled": f"{bet_raw * scale_factor:.2f}",
                    })

            intel_panel = build_intel_panel(hole_cards, community_cards, snapshot, spr)
            color_map = {
                "good": ui.color_good,
                "warn": ui.color_warn,
                "bad": ui.color_bad,
                "muted": ui.color_muted,
            }
            intel_data = {
                section: [(text, color_map.get(level, ui.color_muted)) for text, level in items]
                for section, items in intel_panel.items()
            }

            ui.update(
                header,
                state_lines,
                seat_lines,
                table_view=table_view,
                input_rows=input_rows,
                argmax_text=argmax_text,
                sampled_text=sampled_text,
                metrics_text=metrics_text,
                action_chain=table_tracker.get_action_chain(10),
                event_lines=table_tracker.event_log,
                probs_text=probs_text,
                intel_data=intel_data,
            )

            if LOG_TICK % TABLE_HISTORY_SNAPSHOT_EVERY_N == 0:
                append_history_event({
                    "type": "snapshot",
                    "hand_id": table_tracker.hand_id,
                    "tick_id": LOG_TICK,
                    "street": state.street if state else None,
                    "board": community_cards,
                    "hole": hole_cards,
                    "pot": round(potsize, 6),
                    "cost_to_call": round(cct, 6),
                    "to_act": snapshot.get("to_act"),
                    "button": snapshot.get("button"),
                    "sb": snapshot.get("sb"),
                    "bb": snapshot.get("bb"),
                    "equity": round(equity, 6),
                    "seats": [
                        {
                            "seat_index": s.get("seat_index"),
                            "name": s.get("name") or "",
                            "label": (s.get("name") or f"Seat{(s.get('seat_index') or 0) + 1}"),
                            "stack": round(s.get("stack", 0.0), 6),
                            "bet": round(s.get("bet", 0.0), 6),
                            "folded": bool(s.get("folded")),
                            "is_active": bool(s.get("is_active")),
                            "is_button": bool(s.get("is_button")),
                            "is_sb": bool(s.get("is_sb")),
                            "is_bb": bool(s.get("is_bb")),
                            "action_text": s.get("action_text") or "",
                        }
                        for s in snapshot.get("seats", [])
                    ],
                })

        except Exception as e:
            log_event("tick_error", {"error": str(e)})

        ui.root.after(UI_REFRESH_MS, tick)

    ui.root.after(UI_REFRESH_MS, tick)
    ui.run_loop()


if __name__ == "__main__":
    main()















































































































# from __future__ import annotations

# import json
# import logging
# import os
# import random
# import re
# import ctypes
# import time
# from datetime import datetime
# from dataclasses import dataclass
# from typing import Dict, List, Optional, Tuple

# import tkinter as tk
# from tkinter import ttk

# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from selenium.common.exceptions import NoSuchElementException

# import torch
# from pypokerengine.engine.card import Card
# from pypokerengine.utils.card_utils import estimate_hole_card_win_rate

# from poker_env import (
#     GameState,
#     SimpleHoldemEnv,
#     ACTION_FOLD,
#     ACTION_CHECK,
#     ACTION_CALL,
#     ACTION_BET_POT_50,
#     ACTION_BET_POT_100,
#     ACTION_ALL_IN,
#     ACTION_SEQ_LEN,
#     NUM_ACTIONS,
#     STREET_PREFLOP,
#     STREET_FLOP,
#     STREET_TURN,
#     STREET_RIVER,
# )
# from abstraction import encode_state
# from networks import PolicyNet, AdvantageNet
# from config import DEVICE

# # --------------------------------------------------------------------------------------
# # Configuration
# # --------------------------------------------------------------------------------------

# URL = "https://mgames-poker-fr3.williamhill.com/poker/web/25.1.1.57_1/html/poker/index.html?launcherRedirect=true&hostedMode=3"
# # POLICY_PATH = r"C:\\Users\\PRABAL YADAV\\Desktop\\machine learning iim\\pokerbotPlayOnline\\models\\policy phase3_120.pt"
# POLICY_PATH = r"C:\\Users\\PRABAL YADAV\\Desktop\\machine learning iim\\pokerbotPlayOnline\\models\\policy phase3_310.pt"
# ADV_POLICY_PATHS = [
#     r"C:\\Users\\PRABAL YADAV\\Desktop\\machine learning iim\\pokerbotPlayOnline\\models\\adv_p0.pt",
#     r"C:\\Users\\PRABAL YADAV\\Desktop\\machine learning iim\\pokerbotPlayOnline\\models\\adv_p1.pt",
#     r"C:\\Users\\PRABAL YADAV\\Desktop\\machine learning iim\\pokerbotPlayOnline\\models\\adv_p2.pt",
#     r"C:\\Users\\PRABAL YADAV\\Desktop\\machine learning iim\\pokerbotPlayOnline\\models\\adv_p3.pt",
#     r"C:\\Users\\PRABAL YADAV\\Desktop\\machine learning iim\\pokerbotPlayOnline\\models\\adv_p4.pt",
#     r"C:\\Users\\PRABAL YADAV\\Desktop\\machine learning iim\\pokerbotPlayOnline\\models\\adv_p5.pt",
# ]

# TABLE_NUM_PLAYERS = 6
# FORCE_HERO_SEAT_NUMBER = 4  # 1-based seat index from site: player-seat-4

# # Set these before running (you can change per table)
# TABLE_SMALL_BLIND = 0.1
# TABLE_BIG_BLIND = 0.2

# # Use a fixed initial stack size in SB units (e.g., 200 SBs)
# TABLE_STACK_SBS = 200
# TABLE_STACK_SIZE = TABLE_STACK_SBS * TABLE_SMALL_BLIND

# # Training blinds (policy was trained with these)
# TRAINING_SMALL_BLIND = 1.0
# TRAINING_BIG_BLIND = 2.0
# TRAINING_STACK_SIZE = TABLE_STACK_SBS * TRAINING_SMALL_BLIND

# # Logging
# LOG_FULL_STATE = True
# LOG_TABLE_TO_FILE = True
# LOG_PRINT_CONSOLE = False
# LOG_FILE_PATH = "table_state_debug.log"
# LOG_EVERY_N = 1

# # UI
# UI_REFRESH_MS = 100
# REQUIRE_ACTION_BUTTONS = False
# SHOW_TABLE_VIEW = True
# SHOW_INPUT_TABLE = True
# MAX_EVENT_LOG = 30
# TABLE_HISTORY_DIR = "tablehistory"
# TABLE_HISTORY_SNAPSHOT_EVERY_N = 1

# # Automation
# AUTOMATION = True
# FOLD_PROB_THRESHOLD = 0.75
# ACTION_SAMPLE_PROB = 0.80  # sample vs argmax selection ratio
# AUTO_ACTION_MIN_DELAY_S = 0.75
# AUTO_ACTION_MAX_DELAY_S = 3.25
# AUTOMATION_STREETS = {STREET_PREFLOP}

# # Table view seat positions (mapped to website seat numbers 1..6 -> index 0..5)
# # Order: seat-1, seat-2, seat-3, seat-4, seat-5, seat-6
# SEAT_POSITIONS_6 = [
#     (500, 70),   # seat-1 (top-center)
#     (820, 110),  # seat-2 (top-right)
#     (900, 230),  # seat-3 (right)
#     (500, 360),  # seat-4 (bottom-center)
#     (180, 320),  # seat-5 (bottom-left)
#     (100, 190),  # seat-6 (left)
# ]

# # Strategy
# USE_ADVANTAGE_NET = True

# # Selectors (based on your table DOM)
# SEAT_SELECTORS = [
#     ".table-6-players .player-area",
#     ".table .player-area",
#     ".player-area",
# ]
# STACK_SELECTORS = [
#     ".player-nameplate .text-block.amount",
#     ".player-nameplate .amount",
# ]
# BET_SELECTORS = [
#     ".player-bet .amount-cont .amount",
#     ".player-bet .amount",
# ]
# NAME_SELECTORS = [
#     ".player-nameplate .text-block.nickname .target",
#     ".player-nameplate .nickname .target",
# ]
# ACTION_TEXT_SELECTORS = [
#     ".player-action .action-text",
#     ".player-action",
#     ".action-text",
#     ".player-last-action",
#     ".last-action",
#     ".action-badge",
#     ".action-label",
# ]
# BUTTON_SELECTORS = [
#     ".game-position:not(.pt-visibility-hidden) .dealer.table-assets-btn-dealer",
# ]
# ACTIVE_SELECTORS = [
#     ".turn-to-act-indicator",
#     ".timeout-wrapper",
#     ".nameplate-blink",
#     ".text-countdown",
# ]
# HOLE_CARD_SELECTORS = [
#     ".player-area.my-player .card-wrapper",
#     ".player-area.my-player .card-image-backup",
#     ".player-area.my-player img.card-image",
#     ".cards-holder-hero .card-wrapper",
#     ".cards-holder-hero .card-image-backup",
#     ".cards-holder-hero img.card-image",
# ]

# ACTION_ID_TO_NAME = {
#     ACTION_FOLD: "FOLD",
#     ACTION_CHECK: "CHECK",
#     ACTION_CALL: "CALL",
#     ACTION_BET_POT_50: "RAISE_SMALL",
#     ACTION_BET_POT_100: "RAISE_MEDIUM",
#     ACTION_ALL_IN: "ALL_IN",
# }
# ACTION_DISPLAY_NAME = {
#     ACTION_FOLD: "FOLD",
#     ACTION_CHECK: "CHECK",
#     ACTION_CALL: "CALL",
#     ACTION_BET_POT_50: "RAISE",
#     ACTION_BET_POT_100: "RAISE",
#     ACTION_ALL_IN: "ALL-IN",
# }

# logger = logging.getLogger("table_debug")
# LOG_TICK = 0
# TABLE_HISTORY: List[dict] = []
# TABLE_HISTORY_LOG_PATH = ""
# TABLE_HISTORY_SESSION_ID = ""
# TABLE_HISTORY_EVENT_ID = 0
# AUTO_DECISION_KEY: Optional[Tuple[int, int, int, int]] = None

# STREET_LABELS = {
#     STREET_PREFLOP: "PREFLOP",
#     STREET_FLOP: "FLOP",
#     STREET_TURN: "TURN",
#     STREET_RIVER: "RIVER",
# }


# # --------------------------------------------------------------------------------------
# # Utilities
# # --------------------------------------------------------------------------------------

# def setup_logger():
#     logger.handlers = []
#     logger.setLevel(logging.INFO)
#     logger.propagate = False
#     formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
#     if LOG_TABLE_TO_FILE:
#         fh = logging.FileHandler(LOG_FILE_PATH, encoding="utf-8")
#         fh.setFormatter(formatter)
#         logger.addHandler(fh)
#     if LOG_PRINT_CONSOLE:
#         sh = logging.StreamHandler()
#         sh.setFormatter(formatter)
#         logger.addHandler(sh)

# def click_action_button(driver, button_id: str) -> bool:
#     try:
#         button = driver.find_element(By.ID, button_id)
#         if button.is_displayed() and button.is_enabled():
#             button.click()
#             return True
#     except Exception:
#         return False
#     return False

# def play_airbus_alert() -> None:
#     path = os.path.join(os.path.dirname(__file__), "airbus.mp3")
#     if not os.path.exists(path):
#         log_event("sound_missing", {"path": path})
#         return
#     alias = "airbus_alert"
#     try:
#         ctypes.windll.winmm.mciSendStringW(f"close {alias}", None, 0, None)
#         ctypes.windll.winmm.mciSendStringW(f'open "{path}" type mpegvideo alias {alias}', None, 0, None)
#         ctypes.windll.winmm.mciSendStringW(f"set {alias} volume to 1000", None, 0, None)
#         ctypes.windll.winmm.mciSendStringW(f"play {alias} from 0", None, 0, None)
#     except Exception as exc:
#         log_event("sound_error", {"error": str(exc), "path": path})

# def build_legal_mask(legal_actions: Optional[List[int]], num_actions: int) -> torch.Tensor:
#     mask = torch.zeros(num_actions, dtype=torch.float32)
#     if legal_actions:
#         for a in legal_actions:
#             if 0 <= a < num_actions:
#                 mask[a] = 1.0
#     else:
#         mask.fill_(1.0)
#     return mask

# def masked_normalize_tensor(p: torch.Tensor, mask: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
#     p = p * mask
#     s = p.sum()
#     if s <= eps:
#         denom = mask.sum().clamp_min(1.0)
#         return mask / denom
#     return p / s


# def init_history_logging() -> None:
#     global TABLE_HISTORY_LOG_PATH, TABLE_HISTORY_SESSION_ID
#     os.makedirs(TABLE_HISTORY_DIR, exist_ok=True)
#     TABLE_HISTORY_SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
#     TABLE_HISTORY_LOG_PATH = os.path.join(TABLE_HISTORY_DIR, f"{TABLE_HISTORY_SESSION_ID}_log.jsonl")
#     append_history_event({
#         "type": "session_start",
#         "session_id": TABLE_HISTORY_SESSION_ID,
#         "url": URL,
#         "sb": TABLE_SMALL_BLIND,
#         "bb": TABLE_BIG_BLIND,
#         "stack_sbs": TABLE_STACK_SBS,
#         "training_bb": TRAINING_BIG_BLIND,
#     })


# def append_history_event(event: dict) -> None:
#     global TABLE_HISTORY_EVENT_ID
#     if not event:
#         return
#     record = {
#         "event_id": TABLE_HISTORY_EVENT_ID,
#         "ts": time.time(),
#         "iso": datetime.now().isoformat(timespec="seconds"),
#         "session_id": TABLE_HISTORY_SESSION_ID,
#     }
#     record.update(event)
#     TABLE_HISTORY.append(record)
#     if TABLE_HISTORY_LOG_PATH:
#         with open(TABLE_HISTORY_LOG_PATH, "a", encoding="utf-8") as fh:
#             fh.write(json.dumps(record, ensure_ascii=True) + "\n")
#     TABLE_HISTORY_EVENT_ID += 1


# def log_event(message: str, payload: Optional[dict] = None):
#     if not LOG_FULL_STATE:
#         return
#     if payload is None:
#         logger.info(message)
#     else:
#         logger.info(json.dumps({"message": message, "payload": payload}, ensure_ascii=True))


# def parse_money(text: str) -> float:
#     if not text:
#         return 0.0
#     cleaned = text.replace(",", "").replace("$", "").replace("€", "").strip()
#     match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
#     if not match:
#         return 0.0
#     try:
#         return float(match.group(0))
#     except ValueError:
#         return 0.0


# def format_bb(amount: float) -> str:
#     if not TABLE_BIG_BLIND or TABLE_BIG_BLIND <= 0:
#         return f"{amount:.2f}"
#     return f"{amount / TABLE_BIG_BLIND:.2f}bb"


# def extract_text(elem, selectors: List[str]) -> str:
#     for sel in selectors:
#         try:
#             txt = elem.find_element(By.CSS_SELECTOR, sel).text.strip()
#             if txt:
#                 return txt
#         except Exception:
#             continue
#     return ""


# def extract_action_text(elem) -> str:
#     text = extract_text(elem, ACTION_TEXT_SELECTORS)
#     if text:
#         return text
#     try:
#         candidates = elem.find_elements(By.CSS_SELECTOR, "[class*='action']")
#     except Exception:
#         candidates = []
#     for cand in candidates:
#         try:
#             txt = cand.text.strip()
#         except Exception:
#             txt = ""
#         if not txt:
#             continue
#         lowered = txt.lower()
#         if any(k in lowered for k in ["fold", "check", "call", "raise", "bet", "all-in", "all in", "sit out"]):
#             return txt
#     return ""


# def is_hidden_by_class(elem) -> bool:
#     class_name = (elem.get_attribute("class") or "").lower()
#     return "pt-hidden" in class_name or "pt-visibility-hidden" in class_name


# def has_visible_child(elem, selectors: List[str]) -> bool:
#     for sel in selectors:
#         try:
#             children = elem.find_elements(By.CSS_SELECTOR, sel)
#         except Exception:
#             children = []
#         for child in children:
#             try:
#                 if child.is_displayed():
#                     return True
#             except Exception:
#                 pass
#             if not is_hidden_by_class(child):
#                 return True
#     return False


# def extract_seat_index(elem, fallback_idx: int) -> int:
#     class_name = elem.get_attribute("class") or ""
#     match = re.search(r"player-seat-(\d+)", class_name)
#     if match:
#         seat_num = int(match.group(1))
#         if 1 <= seat_num <= TABLE_NUM_PLAYERS:
#             return seat_num - 1
#     for attr in ["data-seat", "data-seat-id", "data-position", "data-index", "data-seatindex"]:
#         val = elem.get_attribute(attr)
#         if val and val.isdigit():
#             seat_num = int(val)
#             if 1 <= seat_num <= TABLE_NUM_PLAYERS:
#                 return seat_num - 1
#             return seat_num
#         if val:
#             match = re.search(r"(\d+)", val)
#             if match:
#                 seat_num = int(match.group(1))
#                 if 1 <= seat_num <= TABLE_NUM_PLAYERS:
#                     return seat_num - 1
#                 return seat_num
#     id_attr = elem.get_attribute("id") or ""
#     match = re.search(r"(\d+)", id_attr)
#     if match:
#         seat_num = int(match.group(1))
#         if 1 <= seat_num <= TABLE_NUM_PLAYERS:
#             return seat_num - 1
#         return seat_num
#     return fallback_idx


# def read_seat_elements(driver) -> List:
#     best = []
#     for sel in SEAT_SELECTORS:
#         try:
#             elems = driver.find_elements(By.CSS_SELECTOR, sel)
#         except Exception:
#             elems = []
#         if len(elems) > len(best):
#             best = elems
#     return best


# def normalize_seats(seats: List[dict], num_players: int) -> List[dict]:
#     seat_map = {s.get("seat_index"): s for s in seats if s.get("seat_index") is not None}
#     normalized = []
#     for idx in range(num_players):
#         if idx in seat_map:
#             s = seat_map[idx]
#         else:
#             s = {
#                 "seat_index": idx,
#                 "name": "",
#                 "raw_name_text": "",
#                 "raw_stack_text": "",
#                 "raw_bet_text": "",
#                 "class_name": "",
#                 "action_text": "",
#                 "stack": 0.0,
#                 "bet": 0.0,
#                 "folded": True,
#                 "is_hero": False,
#                 "is_button": False,
#                 "is_sb": False,
#                 "is_bb": False,
#                 "is_active": False,
#             }
#         normalized.append(s)
#     return normalized


# def parse_seat_elem(elem, fallback_idx: int) -> dict:
#     seat_index = extract_seat_index(elem, fallback_idx)
#     name_text = extract_text(elem, NAME_SELECTORS)
#     stack_text = extract_text(elem, STACK_SELECTORS)
#     bet_text = extract_text(elem, BET_SELECTORS)

#     class_name = (elem.get_attribute("class") or "").lower()
#     is_hero = "my-player" in class_name

#     action_text = extract_action_text(elem)
#     action_text_lower = action_text.lower()

#     is_folded_action = "fold" in action_text_lower
#     is_sit_out_action = "sit out" in action_text_lower or "sit-out" in action_text_lower
#     has_fold_class = has_visible_child(elem, [".player-action.action-fold"])
#     is_sit_out_class = "sit-out" in class_name or "player-sit-out" in class_name

#     is_folded = is_folded_action or is_sit_out_action or has_fold_class or is_sit_out_class

#     is_button = has_visible_child(elem, BUTTON_SELECTORS)
#     is_sb = has_visible_child(elem, [".small-blind", ".sb"])
#     is_bb = has_visible_child(elem, [".big-blind", ".bb"])
#     is_active = has_visible_child(elem, ACTIVE_SELECTORS)

#     return {
#         "seat_index": seat_index,
#         "name": name_text,
#         "raw_name_text": name_text,
#         "raw_stack_text": stack_text,
#         "raw_bet_text": bet_text,
#         "class_name": class_name,
#         "action_text": action_text,
#         "stack": parse_money(stack_text),
#         "bet": parse_money(bet_text),
#         "folded": bool(is_folded),
#         "is_hero": bool(is_hero),
#         "is_button": bool(is_button),
#         "is_sb": bool(is_sb),
#         "is_bb": bool(is_bb),
#         "is_active": bool(is_active),
#     }


# def resolve_positions(seats: List[dict], hero_index: int) -> Tuple[int, int, int]:
#     n = len(seats)
#     button_idx = next((i for i, s in enumerate(seats) if s["is_button"]), None)
#     sb_idx = next((i for i, s in enumerate(seats) if s["is_sb"]), None)
#     bb_idx = next((i for i, s in enumerate(seats) if s["is_bb"]), None)

#     if button_idx is None:
#         button_idx = hero_index if hero_index is not None else 0
#     if sb_idx is None:
#         sb_idx = (button_idx + 1) % n if n else 0
#     if bb_idx is None:
#         bb_idx = (sb_idx + 1) % n if n else 0

#     return button_idx, sb_idx, bb_idx


# def resolve_to_act(seats: List[dict], hero_index: int, force_hero_to_act: bool = False) -> Optional[int]:
#     active_idx = next((i for i, s in enumerate(seats) if s["is_active"]), None)
#     if active_idx is not None:
#         return active_idx
#     if force_hero_to_act:
#         return hero_index
#     return None


# def read_table_snapshot(driver, force_hero_to_act: bool = False) -> Optional[dict]:
#     seat_elems = read_seat_elements(driver)
#     if not seat_elems:
#         log_event("No seat elements found", {"selectors": SEAT_SELECTORS})
#         return None
#     seats_raw = [parse_seat_elem(elem, idx) for idx, elem in enumerate(seat_elems)]
#     num_players = TABLE_NUM_PLAYERS if TABLE_NUM_PLAYERS else max([s.get("seat_index", 0) for s in seats_raw] + [0]) + 1
#     seats = normalize_seats(seats_raw, num_players)

#     hero_index = next((i for i, s in enumerate(seats) if s["is_hero"]), None)
#     if FORCE_HERO_SEAT_NUMBER:
#         forced = max(0, min(num_players - 1, FORCE_HERO_SEAT_NUMBER - 1))
#         hero_index = forced
#         seats[hero_index]["is_hero"] = True
#     if hero_index is None:
#         hero_index = 0

#     button_idx, sb_idx, bb_idx = resolve_positions(seats, hero_index)
#     to_act = resolve_to_act(seats, hero_index, force_hero_to_act=force_hero_to_act)
#     return {
#         "seats": seats,
#         "hero_index": hero_index,
#         "button": button_idx,
#         "sb": sb_idx,
#         "bb": bb_idx,
#         "to_act": to_act,
#         "num_players": num_players,
#         "seat_elem_count": len(seat_elems),
#     }


# def street_from_board(community_cards: List[str]) -> int:
#     count = len(community_cards)
#     if count >= 5:
#         return STREET_RIVER
#     if count == 4:
#         return STREET_TURN
#     if count == 3:
#         return STREET_FLOP
#     return STREET_PREFLOP


# def translate_suit(suit_symbol: str) -> Optional[str]:
#     suit_map = {
#         "\u2663": "C",
#         "\u2666": "D",
#         "\u2665": "H",
#         "\u2660": "S",
#     }
#     return suit_map.get(suit_symbol, None)


# def parse_card_from_img_src(src: str) -> Optional[str]:
#     if not src:
#         return None
#     lower = src.lower()
#     match = re.search(r"([cdhs])([0-9]{1,2}|[ajkqt])\\.svg", lower)
#     if not match:
#         match = re.search(r"([0-9]{1,2}|[ajkqt])([cdhs])\\.svg", lower)
#         if not match:
#             return None
#         rank_raw = match.group(1)
#         suit_raw = match.group(2)
#     else:
#         suit_raw = match.group(1)
#         rank_raw = match.group(2)

#     suit_map = {"c": "C", "d": "D", "h": "H", "s": "S"}
#     rank_map = {"a": "A", "k": "K", "q": "Q", "j": "J", "t": "T"}
#     suit = suit_map.get(suit_raw, "")
#     rank = rank_map.get(rank_raw, rank_raw)
#     if not suit or not rank:
#         return None
#     return f"{rank.upper()}{suit}"


# def card_str_to_id(card_str: str) -> Optional[int]:
#     if not card_str:
#         return None
#     text = card_str.strip().upper()
#     if len(text) == 3:
#         rank = text[:2]
#         suit = text[2:]
#     else:
#         rank = text[0]
#         suit = text[1]

#     rank_map = {
#         "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
#         "10": 10, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14,
#     }
#     suit_map = {"S": 0, "H": 1, "D": 2, "C": 3}
#     if rank not in rank_map or suit not in suit_map:
#         return None
#     return suit_map[suit] * 13 + (rank_map[rank] - 2)


# def cards_str_to_ids(cards: List[str]) -> List[int]:
#     ids = []
#     for c in cards:
#         card_id = card_str_to_id(c)
#         if card_id is not None:
#             ids.append(card_id)
#     return ids


# def _read_rank_suit(card_elem) -> Tuple[str, str]:
#     rank = ""
#     suit = ""
#     try:
#         rank = card_elem.find_element(By.CSS_SELECTOR, ".card-image-backup .card-rank").text.strip()
#         suit = card_elem.find_element(By.CSS_SELECTOR, ".card-image-backup .card-suit").text.strip()
#     except Exception:
#         pass
#     if not rank:
#         try:
#             rank = card_elem.find_element(By.CLASS_NAME, "card-rank").text.strip()
#         except Exception:
#             rank = ""
#     if not suit:
#         try:
#             suit = card_elem.find_element(By.CLASS_NAME, "card-suit").text.strip()
#         except Exception:
#             suit = ""
#     return rank, suit


# def _card_from_elem(card_elem) -> Optional[str]:
#     rank, suit = _read_rank_suit(card_elem)
#     translated_suit = translate_suit(suit)
#     if rank and translated_suit:
#         return f"{rank.upper()}{translated_suit}"

#     src = ""
#     try:
#         if card_elem.tag_name == "img":
#             src = card_elem.get_attribute("src")
#     except Exception:
#         src = ""

#     if not src:
#         try:
#             img = card_elem.find_element(By.CSS_SELECTOR, "img.card-image")
#             src = img.get_attribute("src")
#         except Exception:
#             src = ""

#     return parse_card_from_img_src(src)


# def read_cards_hero_with_debug(driver) -> Tuple[List[str], Dict[str, int]]:
#     hole_cards = []
#     selector_counts: Dict[str, int] = {}
#     for sel in HOLE_CARD_SELECTORS:
#         try:
#             card_elements = driver.find_elements(By.CSS_SELECTOR, sel)
#         except Exception:
#             card_elements = []
#         selector_counts[sel] = len(card_elements)
#         for card_elem in card_elements:
#             card = _card_from_elem(card_elem)
#             if card:
#                 hole_cards.append(card)
#     # Deduplicate while preserving order
#     seen = set()
#     unique_cards = []
#     for c in hole_cards:
#         if c not in seen:
#             seen.add(c)
#             unique_cards.append(c)
#     return unique_cards, selector_counts


# def read_community_cards(driver) -> List[str]:
#     try:
#         community_cards = []
#         card_elements = driver.find_elements(By.CSS_SELECTOR, ".community-cards .card-wrapper")
#         for card_elem in card_elements:
#             card = _card_from_elem(card_elem)
#             if card:
#                 community_cards.append(card)
#         return community_cards
#     except Exception:
#         return []


# def gen_cards(cards_str: List[str]) -> List[Card]:
#     suit_map = {
#         "C": Card.CLUB,
#         "H": Card.HEART,
#         "S": Card.SPADE,
#         "D": Card.DIAMOND,
#     }
#     rank_map = {
#         "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
#         "10": 10, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14,
#     }
#     try:
#         cards = []
#         for card_str in cards_str:
#             if len(card_str) == 3:
#                 rank = "10"
#                 suit = card_str[2]
#             else:
#                 rank = card_str[0]
#                 suit = card_str[1]
#             suit = suit_map[suit.upper()]
#             rank = rank_map[rank.upper()]
#             cards.append(Card(suit, rank))
#         return cards
#     except Exception as e:
#         log_event("gen_cards_error", {"error": str(e), "cards": cards_str})
#         return []


# def read_pot_size(driver) -> float:
#     try:
#         total_pot_element = driver.find_element(By.CSS_SELECTOR, ".total-pot-amount")
#         total_text = total_pot_element.text.strip()
#         if total_text:
#             return parse_money(total_text)
#     except Exception:
#         pass

#     try:
#         amount_cont_element = WebDriverWait(driver, 5).until(
#             EC.presence_of_element_located((By.CLASS_NAME, "amount-cont"))
#         )
#         try:
#             main_pot_element = amount_cont_element.find_element(By.ID, "main-pot")
#         except NoSuchElementException:
#             main_pot_element = driver.find_element(By.ID, "main-pot")
#         main_pot_text = main_pot_element.text.strip()
#         return parse_money(main_pot_text)
#     except Exception as e:
#         log_event("read_pot_error", {"error": str(e)})
#         return 0.0


# def get_cost_to_call(driver) -> float:
#     try:
#         call_button = driver.find_element(By.ID, "CALL")
#         cost_value_element = call_button.find_element(By.CLASS_NAME, "action-value")
#         cost_to_call_text = cost_value_element.text.strip()
#         return parse_money(cost_to_call_text)
#     except Exception:
#         return 0.0


# # --------------------------------------------------------------------------------------
# # State tracking and policy
# # --------------------------------------------------------------------------------------

# @dataclass
# class TableTracker:
#     prev_contrib: List[float]
#     prev_folded: List[bool]
#     prev_pot: float
#     prev_street: Optional[int]
#     prev_board_count: int
#     prev_action_texts: List[str]
#     prev_stacks: List[float]
#     last_logged_actions: List[Optional[Tuple[int, float, int, int, str]]]
#     action_seq: List[Tuple[int, int, float]]
#     initial_stacks: List[float]
#     last_aggressor: int
#     players_acted: List[bool]
#     event_log: List[str]
#     hand_id: int

#     def __init__(self) -> None:
#         self.prev_contrib = []
#         self.prev_folded = []
#         self.prev_pot = 0.0
#         self.prev_street = None
#         self.prev_board_count = 0
#         self.prev_action_texts = []
#         self.prev_stacks = []
#         self.last_logged_actions = []
#         self.action_seq = []
#         self.initial_stacks = []
#         self.last_aggressor = -1
#         self.players_acted = []
#         self.event_log = []
#         self.hand_id = 0

#     def reset_hand(self, seats: List[dict]) -> None:
#         self.action_seq = []
#         self.initial_stacks = [TABLE_STACK_SIZE for _ in seats]
#         self.last_aggressor = -1
#         self.players_acted = [
#             True if s.get("folded") or (s.get("stack", 0.0) <= 0.0) else False
#             for s in seats
#         ]
#         self.last_logged_actions = [None for _ in seats]
#         self.event_log = ["New hand"]
#         self.hand_id += 1
#         append_history_event({
#             "type": "hand_start",
#             "hand_id": self.hand_id,
#             "seats": [
#                 {
#                     "seat_index": s.get("seat_index"),
#                     "name": s.get("name") or "",
#                     "label": self._format_player_label(s, s.get("seat_index", 0)),
#                     "stack": s.get("stack", 0.0),
#                     "is_hero": bool(s.get("is_hero")),
#                 }
#                 for s in seats
#             ],
#         })

#     def _format_player_label(self, seat: dict, index: int) -> str:
#         name = (seat.get("name") or "").strip()
#         if not name:
#             name = f"P{index + 1}"
#         if seat.get("is_hero"):
#             return f"{name}(H)"
#         return name

#     def _record_action_event(
#         self,
#         seat: dict,
#         index: int,
#         action: int,
#         invested: float,
#         pot_before: float,
#         street: int,
#         board_count: int,
#     ) -> None:
#         action_name = ACTION_ID_TO_NAME.get(action, str(action))
#         player_label = self._format_player_label(seat, index)
#         display_name = ACTION_DISPLAY_NAME.get(action, action_name)
#         amount_text = ""
#         if invested > 0.0:
#             amount_text = f" {invested:.2f} ({format_bb(invested)})"
#         if action == ACTION_ALL_IN:
#             amount_text = f" {invested:.2f} ({format_bb(invested)})"
#         street_label = STREET_LABELS.get(street, str(street))
#         event = f"{player_label} {display_name}{amount_text} [{street_label}]"
#         self.event_log.append(event)
#         if len(self.event_log) > MAX_EVENT_LOG:
#             self.event_log = self.event_log[-MAX_EVENT_LOG:]
#         append_history_event({
#             "type": "action",
#             "hand_id": self.hand_id,
#             "seat_index": index,
#             "name": seat.get("name") or "",
#             "label": self._format_player_label(seat, index),
#             "is_hero": bool(seat.get("is_hero")),
#             "action_id": action,
#             "action_name": action_name,
#             "display_name": display_name,
#             "invested": round(invested, 6),
#             "pot_before": round(pot_before, 6),
#             "street": street,
#             "board_count": board_count,
#         })

#     def get_action_chain(self, max_events: int = 10) -> str:
#         if not self.event_log:
#             return ""
#         return " -> ".join(self.event_log[-max_events:])

#     def _push_action(self, player: int, action: int, invested: float, pot_before: float) -> None:
#         scale = 1.0
#         if TABLE_BIG_BLIND and TABLE_BIG_BLIND > 0:
#             scale = TRAINING_BIG_BLIND / TABLE_BIG_BLIND
#         invested_scaled = invested * scale
#         pot_before_scaled = pot_before * scale
#         denom = max(1.0, pot_before_scaled)
#         size_norm = 0.0
#         if invested_scaled > 0.0:
#             size_norm = min(invested_scaled / denom, 4.0) / 4.0
#         self.action_seq.append((player, action, size_norm))
#         if len(self.action_seq) > ACTION_SEQ_LEN:
#             self.action_seq = self.action_seq[-ACTION_SEQ_LEN:]
#         if action in (ACTION_BET_POT_25, ACTION_BET_POT_50, ACTION_BET_POT_100, ACTION_BET_POT_200, ACTION_ALL_IN):
#             self.last_aggressor = player

#     def _classify_raise(self, invested: float, to_call: float, pot_before: float, street: int) -> int:
#         raise_size = max(0.0, invested - to_call)
#         if street == STREET_PREFLOP:
#             if raise_size <= 2.5 * TABLE_BIG_BLIND:
#                 return ACTION_BET_POT_50
#             return ACTION_BET_POT_100
#         denom = max(1.0, pot_before)
#         frac = raise_size / denom
#         if frac <= 0.75:
#             return ACTION_BET_POT_50
#         return ACTION_BET_POT_100

#     def _action_from_text(self, action_text: str) -> Optional[int]:
#         text = (action_text or "").strip().lower()
#         if not text:
#             return None
#         if "fold" in text:
#             return ACTION_FOLD
#         if "check" in text:
#             return ACTION_CHECK
#         if "call" in text:
#             return ACTION_CALL
#         if "all-in" in text or "all in" in text:
#             return ACTION_ALL_IN
#         if "raise" in text or "bet" in text:
#             return ACTION_BET_POT_50
#         if "blind" in text or "post" in text:
#             return None
#         return None

#     def update(self, seats: List[dict], pot: float, street: int, board_count: int, to_act: Optional[int]) -> None:
#         n = len(seats)
#         if n <= 0:
#             return

#         new_hand = False
#         if self.prev_street is None:
#             new_hand = True
#         elif street < self.prev_street:
#             new_hand = True
#         elif self.prev_board_count > 0 and board_count == 0:
#             new_hand = True
#         elif street == STREET_PREFLOP and pot < self.prev_pot and self.prev_pot > 0.0:
#             new_hand = True

#         if new_hand:
#             self.reset_hand(seats)

#         street_changed = self.prev_street is not None and street != self.prev_street
#         if street_changed:
#             append_history_event({
#                 "type": "street_change",
#                 "hand_id": self.hand_id,
#                 "street": street,
#                 "board_count": board_count,
#                 "pot": round(pot, 6),
#             })

#         if new_hand:
#             self.prev_contrib = [s.get("bet", 0.0) for s in seats]
#             self.prev_folded = [s.get("folded", False) for s in seats]
#             self.prev_action_texts = [s.get("action_text", "") for s in seats]
#             self.prev_stacks = [s.get("stack", 0.0) for s in seats]
#             self.last_logged_actions = [None for _ in seats]
#             self.prev_pot = pot
#             self.prev_street = street
#             self.prev_board_count = board_count
#         elif len(self.prev_contrib) != n or len(self.prev_action_texts) != n:
#             self.prev_contrib = [s.get("bet", 0.0) for s in seats]
#             self.prev_folded = [s.get("folded", False) for s in seats]
#             self.prev_action_texts = [s.get("action_text", "") for s in seats]
#             self.prev_stacks = [s.get("stack", 0.0) for s in seats]
#             self.last_logged_actions = [None for _ in seats]
#         else:
#             prev_bet = max(self.prev_contrib) if self.prev_contrib else 0.0
#             curr_contrib = [s.get("bet", 0.0) for s in seats]
#             curr_bet = max(curr_contrib) if curr_contrib else 0.0
#             curr_action_texts = [s.get("action_text", "") for s in seats]

#             for i in range(n):
#                 action_text = curr_action_texts[i]
#                 prev_action_text = self.prev_action_texts[i]
#                 action_from_text = None
#                 if action_text and action_text != prev_action_text:
#                     action_from_text = self._action_from_text(action_text)

#                 invested = max(0.0, curr_contrib[i] - self.prev_contrib[i])
#                 to_call = max(0.0, prev_bet - self.prev_contrib[i])

#                 def action_key_for(action_id: int, invested_amt: float, text: str) -> Tuple[int, float, int, int, str]:
#                     if action_id == ACTION_FOLD:
#                         text = "FOLD"
#                     return (action_id, round(invested_amt, 6), street, board_count, text or "")

#                 if action_from_text == ACTION_FOLD and not self.prev_folded[i] and seats[i].get("folded"):
#                     action_key = action_key_for(ACTION_FOLD, 0.0, action_text)
#                     if self.last_logged_actions[i] != action_key:
#                         self._push_action(i, ACTION_FOLD, 0.0, self.prev_pot)
#                         self._record_action_event(seats[i], i, ACTION_FOLD, 0.0, self.prev_pot, street, board_count)
#                         self.last_logged_actions[i] = action_key
#                     self.players_acted[i] = True
#                 elif action_from_text is not None:
#                     action = action_from_text
#                     if action in (ACTION_BET_POT_25, ACTION_BET_POT_50, ACTION_BET_POT_100, ACTION_BET_POT_200):
#                         action = self._classify_raise(invested, to_call, self.prev_pot, street)
#                     if action == ACTION_ALL_IN and seats[i].get("stack", 0.0) > 1e-9 and invested <= 0.0:
#                         action = self._classify_raise(invested, to_call, self.prev_pot, street)
#                     action_key = action_key_for(action, invested, action_text)
#                     if self.last_logged_actions[i] != action_key:
#                         self._push_action(i, action, invested, self.prev_pot)
#                         self._record_action_event(seats[i], i, action, invested, self.prev_pot, street, board_count)
#                         self.last_logged_actions[i] = action_key
#                     self.players_acted[i] = True
#                 elif invested > 0.0:
#                     if seats[i].get("stack", 0.0) <= 1e-9:
#                         action = ACTION_ALL_IN
#                     elif curr_contrib[i] >= curr_bet and curr_bet > prev_bet and curr_contrib[i] == curr_bet:
#                         action = self._classify_raise(invested, to_call, self.prev_pot, street)
#                     else:
#                         action = ACTION_CALL
#                     action_key = action_key_for(action, invested, action_text)
#                     if self.last_logged_actions[i] != action_key:
#                         self._push_action(i, action, invested, self.prev_pot)
#                         self._record_action_event(seats[i], i, action, invested, self.prev_pot, street, board_count)
#                         self.last_logged_actions[i] = action_key
#                     self.players_acted[i] = True
#                 elif not self.prev_folded[i] and seats[i].get("folded"):
#                     action_key = action_key_for(ACTION_FOLD, 0.0, action_text)
#                     if self.last_logged_actions[i] != action_key:
#                         self._push_action(i, ACTION_FOLD, 0.0, self.prev_pot)
#                         self._record_action_event(seats[i], i, ACTION_FOLD, 0.0, self.prev_pot, street, board_count)
#                         self.last_logged_actions[i] = action_key
#                     self.players_acted[i] = True

#             self.prev_contrib = curr_contrib
#             self.prev_folded = [s.get("folded", False) for s in seats]
#             self.prev_action_texts = curr_action_texts
#             self.prev_stacks = [s.get("stack", 0.0) for s in seats]

#         self.prev_pot = pot
#         self.prev_street = street
#         self.prev_board_count = board_count
#         if street_changed or new_hand:
#             self.players_acted = [
#                 True if seats[i].get("folded") or seats[i].get("stack", 0.0) <= 0.0 else False
#                 for i in range(n)
#             ]
#         if to_act is not None and 0 <= to_act < n and not seats[to_act].get("folded") and seats[to_act].get("stack", 0.0) > 0.0:
#             self.players_acted[to_act] = False


# def build_game_state(snapshot: dict, hole_cards: List[str], community_cards: List[str], pot: float, tracker: TableTracker, hero_cost_to_call: float = 0.0) -> Tuple[Optional[GameState], Optional[int]]:
#     if not snapshot.get("seats"):
#         return None, None
#     seats = snapshot["seats"]
#     hero_index = snapshot.get("hero_index", 0)
#     to_act = snapshot.get("to_act")
#     if to_act is None:
#         to_act = hero_index

#     street = street_from_board(community_cards)
#     tracker.update(seats, pot, street, len(community_cards), to_act)

#     scale = 1.0
#     if TABLE_BIG_BLIND and TABLE_BIG_BLIND > 0:
#         scale = TRAINING_BIG_BLIND / TABLE_BIG_BLIND

#     stacks = [s["stack"] * scale for s in seats]
#     contrib = [s["bet"] * scale for s in seats]
#     folded = [s["folded"] for s in seats]

#     hole = [[] for _ in range(len(seats))]
#     hero_cards = cards_str_to_ids(hole_cards)
#     if len(hero_cards) == 2:
#         hole[hero_index] = hero_cards

#     board = cards_str_to_ids(community_cards)
#     current_bet = max(contrib) if contrib else 0.0
#     if hero_cost_to_call and to_act == hero_index:
#         current_bet = max(current_bet, contrib[hero_index] + hero_cost_to_call * scale)
#     if tracker.initial_stacks:
#         initial_stacks = [v * scale for v in tracker.initial_stacks]
#     else:
#         initial_stacks = [TABLE_STACK_SIZE * scale for _ in stacks]

#     state = GameState(
#         deck=[],
#         board=board,
#         hole=hole,
#         pot=pot * scale,
#         to_act=to_act,
#         street=street,
#         stacks=stacks,
#         current_bet=current_bet,
#         last_aggressor=tracker.last_aggressor,
#         sb_player=snapshot.get("sb", 0),
#         bb_player=snapshot.get("bb", 1),
#         button_player=snapshot.get("button", 0),
#         initial_stacks=initial_stacks,
#         contrib=contrib,
#         folded=folded,
#         players_acted=tracker.players_acted if len(tracker.players_acted) == len(stacks) else [
#             True if folded[i] or stacks[i] <= 0.0 else False for i in range(len(stacks))
#         ],
#         num_players=len(stacks),
#         actions_this_street=0,
#         terminal=False,
#         winner=-1,
#         action_seq=tracker.action_seq,
#     )
#     return state, hero_index


# def load_policy_net(state_dim: int, path: str) -> PolicyNet:
#     net = PolicyNet(state_dim)
#     state_dict = torch.load(path, map_location=DEVICE)
#     net.load_state_dict(state_dict)
#     net.to(DEVICE)
#     net.eval()
#     return net

# def load_adv_net(state_dim: int, path: str) -> AdvantageNet:
#     net = AdvantageNet(state_dim)
#     state_dict = torch.load(path, map_location=DEVICE)
#     net.load_state_dict(state_dict)
#     net.to(DEVICE)
#     net.eval()
#     return net


# def ensure_policy_loaded(num_players: int, policy_net: Optional[PolicyNet], policy_env: Optional[SimpleHoldemEnv]) -> Tuple[PolicyNet, SimpleHoldemEnv]:
#     if policy_net is not None and policy_env is not None and policy_env.num_players == num_players:
#         return policy_net, policy_env
#     policy_env = SimpleHoldemEnv(
#         stack_size=TRAINING_STACK_SIZE,
#         sb=TRAINING_SMALL_BLIND,
#         bb=TRAINING_BIG_BLIND,
#         num_players=num_players,
#     )
#     dummy = policy_env.new_hand()
#     state_dim = encode_state(dummy, 0).shape[0]
#     policy_net = load_policy_net(state_dim, POLICY_PATH)
#     return policy_net, policy_env

# def ensure_adv_loaded(
#     num_players: int,
#     adv_nets: Optional[List[AdvantageNet]],
#     policy_env: SimpleHoldemEnv,
# ) -> List[AdvantageNet]:
#     if adv_nets is not None and policy_env.num_players == num_players:
#         return adv_nets
#     dummy = policy_env.new_hand()
#     state_dim = encode_state(dummy, 0).shape[0]
#     adv_nets = [load_adv_net(state_dim, path) for path in ADV_POLICY_PATHS]
#     return adv_nets


# def get_policy_action_probs(policy_net: PolicyNet, state: GameState, hero_index: int) -> List[float]:
#     x = encode_state(state, hero_index).float().unsqueeze(0)
#     with torch.no_grad():
#         logp = policy_net(x).squeeze(0)
#     probs = torch.softmax(logp, dim=-1).tolist()
#     return probs

# def get_decision_probs(
#     policy_net: PolicyNet,
#     adv_nets: Optional[List[AdvantageNet]],
#     state: GameState,
#     hero_index: int,
#     legal_actions: Optional[List[int]],
#     use_advantage: bool,
# ) -> Tuple[List[float], str]:
#     x = encode_state(state, hero_index).float().to(DEVICE)
#     mask = build_legal_mask(legal_actions, NUM_ACTIONS).to(DEVICE)
#     with torch.no_grad():
#         if use_advantage and adv_nets is not None:
#             adv_vals = []
#             for net in adv_nets:
#                 adv_vals.append(net(x.unsqueeze(0)).squeeze(0))
#             adv = torch.stack(adv_vals, dim=0).median(dim=0).values
#             pos = torch.clamp(adv, min=0.0)
#             probs = masked_normalize_tensor(pos, mask)
#             mode = "adv"
#         else:
#             logp = policy_net(x.unsqueeze(0)).squeeze(0)
#             probs = torch.softmax(logp, dim=-1)
#             probs = masked_normalize_tensor(probs, mask)
#             mode = "policy"
#     return probs.detach().cpu().tolist(), mode


# def state_vector_summary(state: GameState, hero_index: int) -> Dict[str, float]:
#     try:
#         vec = encode_state(state, hero_index).float()
#         return {
#             "len": int(vec.shape[0]),
#             "min": float(vec.min().item()),
#             "max": float(vec.max().item()),
#             "mean": float(vec.mean().item()),
#         }
#     except Exception as e:
#         return {"error": str(e)}


# def format_policy_probs(probs: List[float], label: str = "Policy probs") -> str:
#     parts = []
#     for a in range(NUM_ACTIONS):
#         name = ACTION_ID_TO_NAME.get(a, f"ACT_{a}")
#         parts.append(f"{name}={probs[a]:.3f}")
#     return f"{label}: " + ", ".join(parts)

# def topk_actions(probs: List[float], k: int = 3) -> List[dict]:
#     ranked = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:k]
#     return [{"action": ACTION_ID_TO_NAME.get(i, i), "prob": float(probs[i])} for i in ranked]


# def sample_action_from_probs(probs: List[float]) -> int:
#     if not probs:
#         return ACTION_CHECK
#     r = random.random()
#     cum = 0.0
#     for idx, p in enumerate(probs):
#         cum += max(0.0, p)
#         if r <= cum:
#             return idx
#     return int(max(range(len(probs)), key=lambda i: probs[i]))

# def select_action_index(probs: List[float], sample_prob: float) -> int:
#     if not probs:
#         return ACTION_CHECK
#     sample_prob = min(max(sample_prob, 0.0), 1.0)
#     if random.random() < sample_prob:
#         return sample_action_from_probs(probs)
#     return int(max(range(len(probs)), key=lambda i: probs[i]))

# def is_automation_turn(state: Optional[GameState], hero_index: Optional[int], to_act: Optional[int]) -> bool:
#     if state is None or hero_index is None:
#         return False
#     return to_act == hero_index and state.street in AUTOMATION_STREETS


# def log_full_snapshot(snapshot: dict,
#                       hole_cards: List[str],
#                       community_cards: List[str],
#                       pot: float,
#                       equity: float,
#                       cct: float,
#                       state: Optional[GameState],
#                       probs: Optional[List[float]],
#                       legal_actions: Optional[List[int]]) -> None:
#     if not LOG_FULL_STATE:
#         return

#     hero_index = snapshot.get("hero_index")
#     to_act = snapshot.get("to_act")
#     seats = snapshot.get("seats", [])
#     seat_dump = []
#     for s in seats:
#         seat_dump.append({
#             "seat_index": s.get("seat_index"),
#             "name": s.get("name"),
#             "stack": s.get("stack"),
#             "bet": s.get("bet"),
#             "folded": s.get("folded"),
#             "is_hero": s.get("is_hero"),
#             "is_button": s.get("is_button"),
#             "is_sb": s.get("is_sb"),
#             "is_bb": s.get("is_bb"),
#             "is_active": s.get("is_active"),
#             "raw_stack_text": s.get("raw_stack_text"),
#             "raw_bet_text": s.get("raw_bet_text"),
#             "raw_name_text": s.get("raw_name_text"),
#             "class_name": s.get("class_name"),
#             "action_text": s.get("action_text"),
#         })

#     to_call = None
#     hero_stack = None
#     if state is not None and hero_index is not None and hero_index >= 0:
#         to_call = max(0.0, state.current_bet - state.contrib[hero_index])
#         hero_stack = state.stacks[hero_index]

#     probs_named = {}
#     if probs is not None:
#         for a in range(NUM_ACTIONS):
#             probs_named[ACTION_ID_TO_NAME.get(a, f"ACT_{a}")] = probs[a]

#     sum_bets = sum(s.get("bet") or 0.0 for s in seats)
#     sum_stacks = sum(s.get("stack") or 0.0 for s in seats)

#     scale = 1.0
#     if TABLE_BIG_BLIND and TABLE_BIG_BLIND > 0:
#         scale = TRAINING_BIG_BLIND / TABLE_BIG_BLIND

#     payload = {
#         "hero_index": hero_index,
#         "to_act": to_act,
#         "num_players": snapshot.get("num_players"),
#         "button": snapshot.get("button"),
#         "sb": snapshot.get("sb"),
#         "bb": snapshot.get("bb"),
#         "seat_elem_count": snapshot.get("seat_elem_count"),
#         "scale_to_training_bb": scale,
#         "table_sb": TABLE_SMALL_BLIND,
#         "table_bb": TABLE_BIG_BLIND,
#         "train_sb": TRAINING_SMALL_BLIND,
#         "train_bb": TRAINING_BIG_BLIND,
#         "pot_raw": pot,
#         "pot_scaled": pot * scale,
#         "sum_bets": sum_bets,
#         "sum_stacks": sum_stacks,
#         "equity": equity,
#         "cost_to_call_raw": cct,
#         "cost_to_call_scaled": cct * scale,
#         "hole_cards": hole_cards,
#         "community_cards": community_cards,
#         "legal_actions": legal_actions,
#         "probs": probs_named,
#         "to_call": to_call,
#         "hero_stack": hero_stack,
#         "state": None,
#         "state_vec": None,
#         "seats": seat_dump,
#     }
#     if state is not None:
#         payload["state"] = {
#             "street": state.street,
#             "pot": state.pot,
#             "current_bet": state.current_bet,
#             "last_aggressor": state.last_aggressor,
#             "stacks": state.stacks,
#             "contrib": state.contrib,
#             "folded": state.folded,
#             "players_acted": state.players_acted,
#             "action_seq": state.action_seq,
#             "board": state.board,
#             "hero_hole": state.hole[hero_index] if hero_index is not None else None,
#         }
#         payload["state_vec"] = state_vector_summary(state, hero_index)

#     logger.info(json.dumps(payload, ensure_ascii=True))


# # --------------------------------------------------------------------------------------
# # UI
# # --------------------------------------------------------------------------------------

# class MonitorUI:
#     def __init__(self):
#         self.root = tk.Tk()
#         self.root.title("Poker Live Command Center")
#         self.root.geometry("1280x980")
#         self.root.configure(bg="#0e1116")

#         self.font_title = ("Segoe UI", 14, "bold")
#         self.font_header = ("Segoe UI", 11, "bold")
#         self.font_body = ("Segoe UI", 10)
#         self.font_mono = ("Consolas", 9)

#         header_frame = tk.Frame(self.root, bg="#0e1116")
#         header_frame.pack(fill="x", padx=10, pady=(10, 6))

#         self.info_var = tk.StringVar(value="Waiting...")
#         self.info_label = tk.Label(
#             header_frame,
#             textvariable=self.info_var,
#             fg="#f5f7fb",
#             bg="#0e1116",
#             font=self.font_title,
#         )
#         self.info_label.pack(anchor="w")

#         metrics_frame = tk.Frame(self.root, bg="#0e1116")
#         metrics_frame.pack(fill="x", padx=10, pady=(0, 6))

#         self.metrics_var = tk.StringVar(value="Equity: - | Pot odds: - | SPR: - | Hero stack: -")
#         self.metrics_label = tk.Label(
#             metrics_frame,
#             textvariable=self.metrics_var,
#             fg="#b8c1cc",
#             bg="#0e1116",
#             font=self.font_body,
#         )
#         self.metrics_label.pack(anchor="w")

#         actions_frame = tk.Frame(self.root, bg="#151a23")
#         actions_frame.pack(fill="x", padx=10, pady=(0, 8))
#         actions_frame.configure(highlightthickness=1, highlightbackground="#212938")

#         self.argmax_var = tk.StringVar(value="Argmax: -")
#         self.argmax_label = tk.Label(
#             actions_frame,
#             textvariable=self.argmax_var,
#             fg="#37d8c3",
#             bg="#151a23",
#             font=self.font_header,
#         )
#         self.argmax_label.pack(anchor="w", padx=10, pady=(6, 0))

#         self.sampled_var = tk.StringVar(value="Sampled: -")
#         self.sampled_label = tk.Label(
#             actions_frame,
#             textvariable=self.sampled_var,
#             fg="#ffa657",
#             bg="#151a23",
#             font=self.font_body,
#         )
#         self.sampled_label.pack(anchor="w", padx=10, pady=(0, 4))

#         self.prob_var = tk.StringVar(value="Action mix: -")
#         self.prob_label = tk.Label(
#             actions_frame,
#             textvariable=self.prob_var,
#             fg="#e6edf3",
#             bg="#151a23",
#             font=self.font_body,
#         )
#         self.prob_label.pack(anchor="w", padx=10, pady=(0, 6))

#         main_frame = tk.Frame(self.root, bg="#0e1116")
#         main_frame.pack(fill="both", expand=True, padx=10, pady=(0, 8))

#         left_frame = tk.Frame(main_frame, bg="#0e1116")
#         left_frame.pack(side="left", fill="both", expand=True)

#         right_frame = tk.Frame(main_frame, bg="#0e1116")
#         right_frame.pack(side="right", fill="y")

#         self.table_canvas = None
#         if SHOW_TABLE_VIEW:
#             self.table_canvas = tk.Canvas(left_frame, width=980, height=430, bg="#111820", highlightthickness=0)
#             self.table_canvas.pack(fill="x", pady=(0, 8))

#         action_chain_frame = tk.Frame(right_frame, bg="#151a23", highlightthickness=1, highlightbackground="#212938")
#         action_chain_frame.pack(fill="x", pady=(0, 8))
#         chain_title = tk.Label(
#             action_chain_frame,
#             text="Action Chain",
#             fg="#8ec9ff",
#             bg="#151a23",
#             font=self.font_header,
#         )
#         chain_title.pack(anchor="w", padx=10, pady=(6, 0))
#         self.action_chain_var = tk.StringVar(value="Waiting for actions...")
#         self.action_chain_label = tk.Label(
#             action_chain_frame,
#             textvariable=self.action_chain_var,
#             fg="#e6edf3",
#             bg="#151a23",
#             font=self.font_body,
#             wraplength=260,
#             justify="left",
#         )
#         self.action_chain_label.pack(anchor="w", padx=10, pady=(0, 6))

#         event_frame = tk.Frame(right_frame, bg="#151a23", highlightthickness=1, highlightbackground="#212938")
#         event_frame.pack(fill="both", expand=True)
#         event_title = tk.Label(
#             event_frame,
#             text="Recent Events",
#             fg="#8ec9ff",
#             bg="#151a23",
#             font=self.font_header,
#         )
#         event_title.pack(anchor="w", padx=10, pady=(6, 0))
#         self.event_text = tk.Text(
#             event_frame,
#             height=18,
#             width=34,
#             bg="#0f141c",
#             fg="#c7d1db",
#             font=self.font_mono,
#             relief="flat",
#         )
#         self.event_text.pack(fill="both", expand=True, padx=10, pady=(0, 8))
#         self.event_text.configure(state="disabled")

#         self.state_text = tk.Text(self.root, height=16, width=120, bg="#0f141c", fg="#c7d1db", font=self.font_mono)
#         self.state_text.pack(fill="both", expand=True, padx=10, pady=(0, 8))

#         self.seat_text = tk.Text(self.root, height=14, width=120, bg="#0f141c", fg="#c7d1db", font=self.font_mono)
#         self.seat_text.pack(fill="both", expand=True, padx=10, pady=(0, 8))

#         self.input_table = None
#         if SHOW_INPUT_TABLE:
#             columns = (
#                 "seat",
#                 "name",
#                 "to_act",
#                 "folded",
#                 "stack_raw",
#                 "bet_raw",
#                 "stack_sb",
#                 "bet_sb",
#                 "stack_scaled",
#                 "bet_scaled",
#             )
#             self.input_table = ttk.Treeview(self.root, columns=columns, show="headings", height=7)
#             for col in columns:
#                 self.input_table.heading(col, text=col)
#                 self.input_table.column(col, width=90, anchor="center")
#             self.input_table.pack(fill="x", padx=10, pady=(0, 10))

#     def _draw_table_view(self, table_view: dict) -> None:
#         if not self.table_canvas:
#             return
#         canvas = self.table_canvas
#         canvas.delete("all")

#         width = int(canvas["width"])
#         height = int(canvas["height"])
#         table_margin = 40
#         canvas.create_oval(
#             table_margin,
#             40,
#             width - table_margin,
#             height - 40,
#             fill="#2c6b2f",
#             outline="#17401a",
#             width=4,
#         )

#         pot_text = table_view.get("pot_text", "Pot: 0")
#         board_text = table_view.get("board_text", "")
#         canvas.create_text(width // 2, height // 2 - 30, text=pot_text, fill="white", font=("Arial", 14, "bold"))
#         canvas.create_text(width // 2, height // 2, text=board_text, fill="white", font=("Arial", 12))

#         seats = table_view.get("seats", [])
#         positions = SEAT_POSITIONS_6
#         for seat in seats:
#             idx = seat.get("seat_index", 0)
#             if idx < 0 or idx >= len(positions):
#                 continue
#             x, y = positions[idx]
#             name = seat.get("name") or f"Seat {idx}"
#             stack_sb = seat.get("stack_sb", 0.0)
#             bet_sb = seat.get("bet_sb", 0.0)
#             action_text = seat.get("action_text", "")
#             flags = []
#             if seat.get("is_hero"):
#                 flags.append("HERO")
#             if seat.get("is_button"):
#                 flags.append("BTN")
#             if seat.get("is_sb"):
#                 flags.append("SB")
#             if seat.get("is_bb"):
#                 flags.append("BB")
#             if seat.get("folded"):
#                 flags.append("FOLD")
#             if seat.get("is_active"):
#                 flags.append("TURN")
#             flag_text = " ".join(flags)
#             info = f"{name}\nstack={stack_sb:.2f}sb bet={bet_sb:.2f}sb"
#             if flag_text:
#                 info += f"\n{flag_text}"
#             if action_text:
#                 info += f"\n{action_text}"
#             fill = "#0f141c"
#             outline = "#2c3648"
#             if seat.get("folded"):
#                 fill = "#1a1f29"
#                 outline = "#3a404d"
#             if seat.get("is_active"):
#                 fill = "#2a1c10"
#                 outline = "#ff9f43"
#             if seat.get("is_hero"):
#                 fill = "#142a4a"
#                 outline = "#5aa6ff"
#             canvas.create_rectangle(x - 80, y - 40, x + 80, y + 40, fill=fill, outline=outline, width=2)
#             canvas.create_text(x, y, text=info, fill="white", font=("Arial", 9), justify="center")

#     def _update_input_table(self, rows: List[dict]) -> None:
#         if not self.input_table:
#             return
#         for item in self.input_table.get_children():
#             self.input_table.delete(item)
#         for row in rows:
#             values = (
#                 row.get("seat"),
#                 row.get("name"),
#                 row.get("to_act"),
#                 row.get("folded"),
#                 row.get("stack_raw"),
#                 row.get("bet_raw"),
#                 row.get("stack_sb"),
#                 row.get("bet_sb"),
#                 row.get("stack_scaled"),
#                 row.get("bet_scaled"),
#             )
#             self.input_table.insert("", "end", values=values)

#     def update(
#         self,
#         header: str,
#         state_lines: List[str],
#         seat_lines: List[str],
#         table_view: Optional[dict] = None,
#         input_rows: Optional[List[dict]] = None,
#         argmax_text: Optional[str] = None,
#         sampled_text: Optional[str] = None,
#         metrics_text: Optional[str] = None,
#         action_chain: Optional[str] = None,
#         event_lines: Optional[List[str]] = None,
#         probs_text: Optional[str] = None,
#     ):
#         self.info_var.set(header)
#         if argmax_text is not None:
#             self.argmax_var.set(argmax_text)
#         if sampled_text is not None:
#             self.sampled_var.set(sampled_text)
#         if probs_text is not None:
#             self.prob_var.set(probs_text)
#         if metrics_text is not None:
#             self.metrics_var.set(metrics_text)
#         if action_chain is not None:
#             self.action_chain_var.set(action_chain or "Waiting for actions...")
#         if event_lines is not None:
#             self.event_text.configure(state="normal")
#             self.event_text.delete("1.0", "end")
#             self.event_text.insert("end", "\n".join(event_lines))
#             self.event_text.configure(state="disabled")
#         if table_view:
#             self._draw_table_view(table_view)
#         self.state_text.configure(state="normal")
#         self.state_text.delete("1.0", "end")
#         self.state_text.insert("end", "\n".join(state_lines))
#         self.state_text.configure(state="disabled")

#         self.seat_text.configure(state="normal")
#         self.seat_text.delete("1.0", "end")
#         self.seat_text.insert("end", "\n".join(seat_lines))
#         self.seat_text.configure(state="disabled")
#         if input_rows is not None:
#             self._update_input_table(input_rows)

#     def run_loop(self):
#         self.root.mainloop()


# # --------------------------------------------------------------------------------------
# # Main
# # --------------------------------------------------------------------------------------


# def main():
#     global LOG_TICK

#     setup_logger()
#     init_history_logging()
#     if LOG_TABLE_TO_FILE:
#         print(f"Logging to {LOG_FILE_PATH}")
#     if TABLE_HISTORY_LOG_PATH:
#         print(f"History log: {TABLE_HISTORY_LOG_PATH}")
#     log_event("Logging initialized", {"log_file": LOG_FILE_PATH, "log_console": LOG_PRINT_CONSOLE})

#     driver = webdriver.Chrome()
#     driver.maximize_window()
#     driver.get(URL)

#     ui = MonitorUI()

#     policy_net = None
#     policy_env = None
#     adv_nets = None
#     table_tracker = TableTracker()

#     def tick():
#         nonlocal policy_net, policy_env, adv_nets
#         global LOG_TICK, AUTO_DECISION_KEY
#         LOG_TICK += 1

#         try:
#             if REQUIRE_ACTION_BUTTONS:
#                 action_button = None
#                 for button_id in ["FOLD", "CHECK", "RAISE_TO"]:
#                     try:
#                         action_button = WebDriverWait(driver, 2).until(
#                             EC.presence_of_element_located((By.ID, button_id))
#                         )
#                         if action_button:
#                             break
#                     except Exception:
#                         pass
#                 if not action_button:
#                     log_event("No actionable buttons detected")
#                     ui.update("Waiting for action buttons...", ["No actionable buttons yet."], [])
#                     ui.root.after(UI_REFRESH_MS, tick)
#                     return

#             hole_cards, hole_debug = read_cards_hero_with_debug(driver)
#             community_cards = read_community_cards(driver)
#             if len(hole_cards) != 2:
#                 log_event("Unable to read hole cards", {"hole_cards": hole_cards, "hole_selectors": hole_debug})
#                 snapshot = read_table_snapshot(driver, force_hero_to_act=False)
#                 if snapshot:
#                     seat_lines = []
#                     for s in snapshot.get("seats", []):
#                         seat_lines.append(
#                             f"Seat {s.get('seat_index')} | name={s.get('name')} | stack={s.get('stack'):.2f} | bet={s.get('bet'):.2f} | folded={s.get('folded')} | active={s.get('is_active')} | btn={s.get('is_button')}"
#                         )
#                     state_lines = [
#                         "Hole cards not visible yet.",
#                         f"Hole selector counts: {hole_debug}",
#                     ]
#                     ui.update("Waiting for hole cards...", state_lines, seat_lines)
#                 else:
#                     ui.update("Waiting for table...", [f"No table snapshot yet. Hole selectors: {hole_debug}"], [])
#                 ui.root.after(UI_REFRESH_MS, tick)
#                 return

#             hole_card_objects = gen_cards(hole_cards)
#             community_card_objects = gen_cards(community_cards) if community_cards else []

#             if len(community_cards) == 0:
#                 equity = estimate_hole_card_win_rate(
#                     nb_simulation=1500,
#                     nb_player=TABLE_NUM_PLAYERS,
#                     hole_card=hole_card_objects,
#                     community_card=[]
#                 )
#             else:
#                 equity = estimate_hole_card_win_rate(
#                     nb_simulation=1500,
#                     nb_player=TABLE_NUM_PLAYERS,
#                     hole_card=hole_card_objects,
#                     community_card=community_card_objects
#                 )

#             cct = get_cost_to_call(driver)
#             potsize = read_pot_size(driver)
#             scale_factor = 1.0
#             if TABLE_BIG_BLIND and TABLE_BIG_BLIND > 0:
#                 scale_factor = TRAINING_BIG_BLIND / TABLE_BIG_BLIND

#             snapshot = read_table_snapshot(driver, force_hero_to_act=True)
#             if not snapshot:
#                 log_event("No table snapshot; adjust selectors.")
#                 ui.update("No table snapshot", ["Check selectors or wait for table load."], [])
#                 ui.root.after(UI_REFRESH_MS, tick)
#                 return

#             policy_net, policy_env = ensure_policy_loaded(snapshot["num_players"], policy_net, policy_env)
#             if USE_ADVANTAGE_NET:
#                 adv_nets = ensure_adv_loaded(snapshot["num_players"], adv_nets, policy_env)
#             state, hero_index = build_game_state(snapshot, hole_cards, community_cards, potsize, table_tracker, hero_cost_to_call=cct)

#             probs = None
#             legal_actions = None
#             decision_mode = None
#             if state is not None and snapshot.get("to_act") == hero_index:
#                 legal_actions = policy_env.legal_actions(state)
#                 probs, decision_mode = get_decision_probs(
#                     policy_net,
#                     adv_nets,
#                     state,
#                     hero_index,
#                     legal_actions,
#                     USE_ADVANTAGE_NET,
#                 )
#                 if LOG_TICK % LOG_EVERY_N == 0:
#                     log_full_snapshot(snapshot, hole_cards, community_cards, potsize, equity, cct, state, probs, legal_actions)
#             else:
#                 if state is None:
#                     log_event("State build failed", {"snapshot": snapshot})
#             if AUTOMATION and probs is not None and is_automation_turn(state, hero_index, snapshot.get("to_act")):
#                 decision_key = (table_tracker.hand_id, state.street, hero_index, len(state.action_seq))
#                 if AUTO_DECISION_KEY != decision_key:
#                     AUTO_DECISION_KEY = decision_key
#                     if AUTO_ACTION_MAX_DELAY_S > 0:
#                         delay_s = random.uniform(AUTO_ACTION_MIN_DELAY_S, AUTO_ACTION_MAX_DELAY_S)
#                         time.sleep(max(0.0, delay_s))
#                     action_probs = probs
#                     fold_prob = action_probs[ACTION_FOLD] if ACTION_FOLD < len(action_probs) else 0.0
#                     chosen_action = select_action_index(action_probs, ACTION_SAMPLE_PROB)
#                     if fold_prob >= FOLD_PROB_THRESHOLD:
#                         did_fold = click_action_button(driver, "FOLD")
#                         log_event(
#                             "auto_fold_prob",
#                             {
#                                 "fold_prob": fold_prob,
#                                 "fold_clicked": did_fold,
#                                 "chosen_action": chosen_action,
#                                 "mode": decision_mode,
#                                 "topk": topk_actions(action_probs),
#                             },
#                         )
#                         if not did_fold:
#                             AUTO_DECISION_KEY = None
#                     else:
#                         if state is not None and state.street == STREET_PREFLOP:
#                             play_airbus_alert()
#                         log_event(
#                             "auto_hold",
#                             {
#                                 "fold_prob": fold_prob,
#                                 "chosen_action": chosen_action,
#                                 "mode": decision_mode,
#                                 "topk": topk_actions(action_probs),
#                             },
#                         )

#             header = (
#                 f"Hero seat: {snapshot.get('hero_index')} | To act: {snapshot.get('to_act')} | "
#                 f"Pot raw: {potsize:.2f} | Pot scaled: {potsize * scale_factor:.2f} | "
#                 f"Table SB/BB: {TABLE_SMALL_BLIND:.2f}/{TABLE_BIG_BLIND:.2f}"
#             )
#             hero_stack_raw = 0.0
#             if hero_index is not None and 0 <= hero_index < len(snapshot.get("seats", [])):
#                 hero_stack_raw = snapshot["seats"][hero_index].get("stack", 0.0) or 0.0
#             hero_stack_bb = hero_stack_raw / TABLE_BIG_BLIND if TABLE_BIG_BLIND else 0.0
#             spr = hero_stack_raw / max(potsize, TABLE_BIG_BLIND or 1.0) if hero_stack_raw else 0.0
#             pot_odds = 0.0
#             if cct > 0.0:
#                 pot_odds = cct / max(potsize + cct, 1e-9)
#             equity_pct = equity * 100.0
#             metrics_text = (
#                 f"Street: {STREET_LABELS.get(state.street, state.street) if state else '-'} | "
#                 f"Equity: {equity_pct:.1f}% | Required: {pot_odds * 100.0:.1f}% | "
#                 f"SPR: {spr:.2f} | Hero stack: {hero_stack_bb:.1f}bb"
#             )
#             state_lines = []
#             if state is None:
#                 state_lines.append("State: None")
#             else:
#                 to_call = max(0.0, state.current_bet - state.contrib[hero_index])
#                 state_lines.extend([
#                     f"Street: {state.street}",
#                     f"Current bet (scaled): {state.current_bet:.2f}",
#                     f"To call raw: {cct:.2f}",
#                     f"To call scaled: {to_call:.2f}",
#                     f"Pot (scaled): {state.pot:.2f}",
#                     f"Scale factor: {scale_factor:.4f} (train BB {TRAINING_BIG_BLIND} / table BB {TABLE_BIG_BLIND})",
#                     f"Last aggressor: {state.last_aggressor}",
#                     f"Action seq (last {ACTION_SEQ_LEN}): {state.action_seq}",
#                     f"Hole cards: {hole_cards}",
#                     f"Board: {community_cards}",
#                 ])
#                 if legal_actions is not None:
#                     state_lines.append(f"Legal actions (env): {legal_actions}")
#             argmax_text = None
#             sampled_text = None
#             probs_text = None
#             if probs is not None:
#                 argmax_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
#                 sampled_idx = sample_action_from_probs(probs)
#                 argmax_text = f"Argmax action: {ACTION_ID_TO_NAME.get(argmax_idx, argmax_idx)} ({probs[argmax_idx]:.3f})"
#                 sampled_text = f"Sampled action: {ACTION_ID_TO_NAME.get(sampled_idx, sampled_idx)} ({probs[sampled_idx]:.3f})"
#                 ranked = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:3]
#                 probs_text = "Action mix: " + " | ".join(
#                     f"{ACTION_ID_TO_NAME.get(i, i)} {probs[i]:.2f}" for i in ranked
#                 )
#                 label = f"Decision probs ({decision_mode or 'policy'})"
#                 state_lines.append(format_policy_probs(probs, label=label))
#             elif state is not None:
#                 state_lines.append("Decision probs: waiting for your turn")

#             seat_lines = []
#             for s in snapshot.get("seats", []):
#                 seat_lines.append(
#                     f"Seat {s.get('seat_index')} | name={s.get('name')} | stack={s.get('stack'):.2f} | bet={s.get('bet'):.2f} | folded={s.get('folded')} | active={s.get('is_active')} | btn={s.get('is_button')}"
#                 )

#             sb_div = TABLE_SMALL_BLIND if TABLE_SMALL_BLIND > 0 else 1.0
#             table_view = None
#             if SHOW_TABLE_VIEW:
#                 table_view = {
#                     "pot_text": f"Pot: {format_bb(potsize)} ({potsize:.2f})",
#                     "board_text": "Board: " + (" ".join(community_cards) if community_cards else "[]"),
#                     "seats": [],
#                 }
#                 for s in snapshot.get("seats", []):
#                     table_view["seats"].append({
#                         "seat_index": s.get("seat_index"),
#                         "name": s.get("name") or "",
#                         "stack_sb": (s.get("stack") or 0.0) / sb_div,
#                         "bet_sb": (s.get("bet") or 0.0) / sb_div,
#                         "folded": s.get("folded"),
#                         "is_hero": s.get("is_hero"),
#                         "is_button": s.get("is_button"),
#                         "is_sb": s.get("is_sb"),
#                         "is_bb": s.get("is_bb"),
#                         "is_active": s.get("is_active"),
#                         "action_text": s.get("action_text") or "",
#                     })

#             input_rows = None
#             if SHOW_INPUT_TABLE:
#                 input_rows = []
#                 for s in snapshot.get("seats", []):
#                     stack_raw = s.get("stack") or 0.0
#                     bet_raw = s.get("bet") or 0.0
#                     input_rows.append({
#                         "seat": s.get("seat_index"),
#                         "name": s.get("name") or "",
#                         "to_act": "Y" if s.get("seat_index") == snapshot.get("to_act") else "",
#                         "folded": "Y" if s.get("folded") else "",
#                         "stack_raw": f"{stack_raw:.2f}",
#                         "bet_raw": f"{bet_raw:.2f}",
#                         "stack_sb": f"{stack_raw / sb_div:.2f}",
#                         "bet_sb": f"{bet_raw / sb_div:.2f}",
#                         "stack_scaled": f"{stack_raw * scale_factor:.2f}",
#                         "bet_scaled": f"{bet_raw * scale_factor:.2f}",
#                     })

#             ui.update(
#                 header,
#                 state_lines,
#                 seat_lines,
#                 table_view=table_view,
#                 input_rows=input_rows,
#                 argmax_text=argmax_text,
#                 sampled_text=sampled_text,
#                 metrics_text=metrics_text,
#                 action_chain=table_tracker.get_action_chain(10),
#                 event_lines=table_tracker.event_log,
#                 probs_text=probs_text,
#             )

#             if LOG_TICK % TABLE_HISTORY_SNAPSHOT_EVERY_N == 0:
#                 append_history_event({
#                     "type": "snapshot",
#                     "hand_id": table_tracker.hand_id,
#                     "tick_id": LOG_TICK,
#                     "street": state.street if state else None,
#                     "board": community_cards,
#                     "hole": hole_cards,
#                     "pot": round(potsize, 6),
#                     "cost_to_call": round(cct, 6),
#                     "to_act": snapshot.get("to_act"),
#                     "button": snapshot.get("button"),
#                     "sb": snapshot.get("sb"),
#                     "bb": snapshot.get("bb"),
#                     "equity": round(equity, 6),
#                     "seats": [
#                         {
#                             "seat_index": s.get("seat_index"),
#                             "name": s.get("name") or "",
#                             "label": (s.get("name") or f"Seat{(s.get('seat_index') or 0) + 1}"),
#                             "stack": round(s.get("stack", 0.0), 6),
#                             "bet": round(s.get("bet", 0.0), 6),
#                             "folded": bool(s.get("folded")),
#                             "is_active": bool(s.get("is_active")),
#                             "is_button": bool(s.get("is_button")),
#                             "is_sb": bool(s.get("is_sb")),
#                             "is_bb": bool(s.get("is_bb")),
#                             "action_text": s.get("action_text") or "",
#                         }
#                         for s in snapshot.get("seats", [])
#                     ],
#                 })

#         except Exception as e:
#             log_event("tick_error", {"error": str(e)})

#         ui.root.after(UI_REFRESH_MS, tick)

#     ui.root.after(UI_REFRESH_MS, tick)
#     ui.run_loop()


# if __name__ == "__main__":
#     main()
