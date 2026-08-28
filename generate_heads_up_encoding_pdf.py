"""Generate a wide visual reference for the HU Deep-CFR input encoder."""

from __future__ import annotations

from pathlib import Path
from textwrap import fill

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from heads_up_engine import ACTION_NAMES
from heads_up_models import (
    ACTION_DESCRIPTOR_FEATURE_NAMES,
    ACTION_DESCRIPTOR_FEATURES,
    CARD_FEATURES,
    CARD_STATE_FEATURES,
    CARD_TOKEN_COUNT,
    DEFAULT_MAX_HISTORY,
    GLOBAL_FEATURE_NAMES,
    HISTORY_FEATURE_NAMES,
    HISTORY_FEATURES,
    HISTORY_OFFSET,
    PUBLIC_PREFIX_FEATURES,
    SEAT_FEATURE_NAMES,
    action_descriptor_offset,
    information_state_size,
    legal_mask_offset,
)


OUTPUT = Path(
    "artifacts/heads_up_documentation/hu_training_input_encoding_1038.pdf"
)
OVERVIEW_PNG = Path(
    "artifacts/heads_up_documentation/hu_training_input_encoding_1038_overview.png"
)
PAGE_SIZE = (24, 13.5)
BG = "#f7f5ef"
INK = "#16212b"
MUTED = "#566573"
COLORS = {
    "position": "#4C78A8",
    "seat": "#72A0C1",
    "global": "#59A14F",
    "cards": "#F28E2B",
    "history": "#B07AA1",
    "legal": "#E15759",
    "descriptor": "#EDC948",
}


POSITION_NAMES = (
    "street_preflop",
    "street_flop",
    "street_turn",
    "street_river",
    "relative_button_hero",
    "relative_button_opponent",
    "hero_is_button",
    "hero_is_small_blind",
    "hero_is_big_blind",
    "actor_none_terminal",
    "actor_hero",
    "actor_opponent",
    "last_full_raiser_none",
    "last_full_raiser_hero",
    "last_full_raiser_opponent",
)
RANK_NAMES = tuple(f"rank_{rank}" for rank in "23456789TJQKA")
SUIT_NAMES = ("suit_clubs", "suit_diamonds", "suit_hearts", "suit_spades")
CARD_FEATURE_NAMES = RANK_NAMES + SUIT_NAMES + ("present",)
CARD_TOKEN_NAMES = (
    "hero_hole_card_1",
    "hero_hole_card_2",
    "flop_card_1",
    "flop_card_2",
    "flop_card_3",
    "turn_card",
    "river_card",
)


def _new_page(title: str, subtitle: str = ""):
    fig, ax = plt.subplots(figsize=PAGE_SIZE)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 13.5)
    ax.axis("off")
    ax.text(0.7, 12.85, title, fontsize=24, weight="bold", color=INK, va="top")
    if subtitle:
        ax.text(0.72, 12.35, subtitle, fontsize=10.5, color=MUTED, va="top")
    ax.text(
        23.3,
        0.28,
        "Source: heads_up_models.py / heads_up_cfr.py • generated 2026-08-04",
        fontsize=7.5,
        color=MUTED,
        ha="right",
    )
    return fig, ax


def _box(ax, x, y, w, h, color, title, detail="", *, fontsize=10, alpha=1.0):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.025,rounding_size=0.08",
        linewidth=1.0,
        edgecolor="white",
        facecolor=color,
        alpha=alpha,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.61, title, ha="center", va="center", fontsize=fontsize,
            weight="bold", color="white" if color != COLORS["descriptor"] else INK)
    if detail:
        ax.text(x + w / 2, y + h * 0.28, detail, ha="center", va="center",
                fontsize=max(6, fontsize - 2), color="white" if color != COLORS["descriptor"] else INK)


def _feature_table(ax, x, y_top, width, row_h, start, names, color, *, columns=1, fontsize=8):
    rows = (len(names) + columns - 1) // columns
    col_w = width / columns
    for index, name in enumerate(names):
        col = index // rows
        row = index % rows
        y = y_top - (row + 1) * row_h
        ax.add_patch(Rectangle((x + col * col_w, y), col_w, row_h,
                               facecolor="white" if row % 2 == 0 else "#eef1f2",
                               edgecolor="#c9d1d6", linewidth=0.5))
        ax.add_patch(Rectangle((x + col * col_w, y), 0.7, row_h,
                               facecolor=color, edgecolor=color))
        ax.text(x + col * col_w + 0.35, y + row_h / 2, str(start + index),
                ha="center", va="center", fontsize=fontsize - 0.5, color="white", weight="bold")
        ax.text(x + col * col_w + 0.82, y + row_h / 2, name,
                ha="left", va="center", fontsize=fontsize, color=INK)


def page_overview(pdf: PdfPages) -> None:
    width = information_state_size(DEFAULT_MAX_HISTORY)
    fig, ax = _new_page(
        "HU training input: the complete 1,038-value array",
        "One hero-visible information state • float32 • default max_history = 32 • indices are inclusive",
    )
    sections = [
        ("Public state", 0, 55, COLORS["global"]),
        ("7 card tokens", 56, 181, COLORS["cards"]),
        ("32 × 23 history", 182, 917, COLORS["history"]),
        ("Legal mask", 918, 927, COLORS["legal"]),
        ("10 × 11 action effects", 928, 1037, COLORS["descriptor"]),
    ]
    x0, total_w, y, h = 0.8, 22.4, 9.5, 1.35
    cursor = x0
    for label, start, end, color in sections:
        w = total_w * (end - start + 1) / width
        ax.add_patch(Rectangle((cursor, y), w, h, facecolor=color, edgecolor=BG, linewidth=1.5))
        if w > 1.5:
            ax.text(cursor + w / 2, y + 0.78, label, ha="center", va="center",
                    fontsize=9 if w < 3 else 12, weight="bold", color=INK if color == COLORS["descriptor"] else "white")
            ax.text(cursor + w / 2, y + 0.33, f"x[{start}:{end}] • {end-start+1}",
                    ha="center", va="center", fontsize=7.5,
                    color=INK if color == COLORS["descriptor"] else "white")
        cursor += w
    ax.text(x0, y + h + 0.2, "x[0]", fontsize=9, color=INK)
    ax.text(x0 + total_w, y + h + 0.2, "x[1037]", fontsize=9, color=INK, ha="right")

    sub = [
        ("Position / street", "15\nx[0:14]", COLORS["position"]),
        ("Hero seat", "10\nx[15:24]", COLORS["seat"]),
        ("Opponent seat", "10\nx[25:34]", COLORS["seat"]),
        ("Global / ratios", "21\nx[35:55]", COLORS["global"]),
        ("Cards", "126\nx[56:181]", COLORS["cards"]),
        ("History", "736\nx[182:917]", COLORS["history"]),
        ("Legal", "10\nx[918:927]", COLORS["legal"]),
        ("Descriptors", "110\nx[928:1037]", COLORS["descriptor"]),
    ]
    box_w = 2.55
    for i, (title, detail, color) in enumerate(sub):
        _box(ax, 0.8 + i * 2.78, 6.55, box_w, 1.55, color, title, detail, fontsize=10)

    ax.text(0.8, 5.75, "How the state becomes the tensor", fontsize=15, weight="bold", color=INK)
    flow = [
        ("Exact engine state", "chips, cards, semantic history", COLORS["position"]),
        ("Hero-relative encoder", "BB normalization + ratios", COLORS["global"]),
        ("x ∈ ℝ¹⁰³⁸", "single float32 row", COLORS["history"]),
        ("Advantage / policy net", "10 fixed action outputs", COLORS["legal"]),
    ]
    for i, (title, detail, color) in enumerate(flow):
        x = 1.0 + i * 5.7
        _box(ax, x, 3.9, 4.35, 1.15, color, title, detail, fontsize=11)
        if i < len(flow) - 1:
            ax.add_patch(FancyArrowPatch((x + 4.4, 4.47), (x + 5.55, 4.47),
                                         arrowstyle="-|>", mutation_scale=16,
                                         color=MUTED, linewidth=1.7))
    notes = (
        "Visibility boundary: hero's 2 private cards are included; opponent hole cards are never read.\n"
        "History boundary: the most recent 32 public semantic events are retained and left-padded with zero rows.\n"
        "Action boundary: the ten policy slots are finite, but every legal slot carries its exact current chip effect."
    )
    ax.text(0.9, 2.85, notes, fontsize=11, color=INK, va="top", linespacing=1.65,
            bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor="#d2d7da"))
    pdf.savefig(fig, bbox_inches="tight")
    fig.savefig(OVERVIEW_PNG, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def page_public(pdf: PdfPages) -> None:
    fig, ax = _new_page(
        "Array map 1/4 — public state x[0:55]",
        "All seat identities are expressed relative to the acting hero. Monetary values marked _bb are divided by the big blind.",
    )
    _feature_table(ax, 0.7, 11.75, 7.1, 0.45, 0, POSITION_NAMES, COLORS["position"], columns=1, fontsize=7.8)
    ax.text(0.7, 12.02, "POSITION / STREET (15)", fontsize=11, weight="bold", color=COLORS["position"])

    hero_names = tuple(f"hero.{name}" for name in SEAT_FEATURE_NAMES)
    opp_names = tuple(f"opponent.{name}" for name in SEAT_FEATURE_NAMES)
    _feature_table(ax, 8.15, 11.75, 7.25, 0.45, 15, hero_names, COLORS["seat"], fontsize=7.8)
    _feature_table(ax, 8.15, 6.85, 7.25, 0.45, 25, opp_names, "#497AA0", fontsize=7.8)
    ax.text(8.15, 12.02, "SEAT FEATURES (10 EACH)", fontsize=11, weight="bold", color=COLORS["seat"])

    _feature_table(ax, 15.75, 11.75, 7.55, 0.45, 35, GLOBAL_FEATURE_NAMES, COLORS["global"], columns=2, fontsize=6.7)
    ax.text(15.75, 12.02, "GLOBAL / HERO DECISION CONTEXT (21)", fontsize=11, weight="bold", color=COLORS["global"])

    ax.text(15.75, 6.35, "Important formulas", fontsize=13, weight="bold", color=INK)
    formulas = [
        "to_call = max(0, current_bet − hero_street_contribution)",
        "call_payment = min(hero_stack, to_call)",
        "pot_after_call = pot + call_payment",
        "effective_after_call = min(hero_stack − call_payment, opponent_stack)",
        "SPR_after_call = effective_after_call / pot_after_call",
        "hero_pot_odds = call_payment / pot_after_call",
        "board_progress = number_of_board_cards / 5",
    ]
    for i, formula in enumerate(formulas):
        ax.text(15.9, 5.88 - i * 0.58, "• " + formula, fontsize=8.2, color=INK)
    ax.text(
        0.9,
        4.35,
        "0–14 categorical / binary\n15–34 seat state\n35–55 continuous + binary\n\nTotal: 56 values",
        fontsize=12,
        color=INK,
        va="top",
        bbox=dict(boxstyle="round,pad=0.55", facecolor="white", edgecolor=COLORS["global"], linewidth=1.5),
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _draw_card(ax, x, y, title, start, color):
    ax.add_patch(FancyBboxPatch((x, y), 2.65, 3.05, boxstyle="round,pad=0.04,rounding_size=0.16",
                                facecolor="white", edgecolor=color, linewidth=2.0))
    ax.text(x + 1.325, y + 2.72, title, ha="center", fontsize=10, weight="bold", color=INK)
    ax.text(x + 1.325, y + 2.42, f"x[{start}:{start+17}]", ha="center", fontsize=8, color=MUTED)
    ax.text(x + 0.18, y + 2.00, "13 rank one-hot", fontsize=8.5, color=INK, weight="bold")
    ax.text(x + 0.18, y + 1.72, "2 3 4 5 6 7 8 9 T J Q K A", fontsize=7.2, family="monospace", color=MUTED)
    ax.text(x + 0.18, y + 1.28, "4 suit one-hot", fontsize=8.5, color=INK, weight="bold")
    ax.text(x + 0.18, y + 1.00, "♣   ♦   ♥   ♠", fontsize=14, color=MUTED)
    ax.text(x + 0.18, y + 0.48, "+ present bit", fontsize=8.5, color=INK, weight="bold")
    ax.text(x + 2.36, y + 0.48, "18", fontsize=13, color=color, weight="bold", ha="right")


def page_cards(pdf: PdfPages) -> None:
    fig, ax = _new_page(
        "Array map 2/4 — cards x[56:181]",
        "Seven 18-value card tokens = 126 values. Card ID is suit × 13 + rank (2..A; clubs, diamonds, hearts, spades).",
    )
    for i, title in enumerate(CARD_TOKEN_NAMES):
        x = 0.75 + (i % 4) * 5.75
        y = 7.85 if i < 4 else 3.95
        _draw_card(ax, x, y, title, PUBLIC_PREFIX_FEATURES + i * CARD_FEATURES, COLORS["cards"])

    ax.text(18.1, 6.55, "Canonical ordering", fontsize=13, weight="bold", color=INK)
    ax.text(
        18.1,
        6.1,
        "• Hero hole cards: sorted by card ID\n"
        "• Flop's three cards: sorted by card ID\n"
        "• Turn and river: retain street order\n"
        "• Missing board slots: 18 zeros\n"
        "• Opponent hole cards: excluded",
        fontsize=10.5,
        color=INK,
        va="top",
        linespacing=1.55,
        bbox=dict(boxstyle="round,pad=0.55", facecolor="white", edgecolor=COLORS["cards"]),
    )
    ax.text(0.9, 3.15, "Inside every card token", fontsize=12, weight="bold", color=INK)
    _feature_table(ax, 0.9, 2.9, 21.8, 0.38, 0, CARD_FEATURE_NAMES, COLORS["cards"], columns=3, fontsize=7.5)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_history(pdf: PdfPages) -> None:
    fig, ax = _new_page(
        "Array map 3/4 — ordered semantic history x[182:917]",
        "32 rows × 23 features = 736 values. Rows are left-zero-padded; slot 31 is always the newest retained public event.",
    )
    left, bottom, total_w, total_h = 1.9, 1.05, 21.2, 10.55
    row_h = total_h / DEFAULT_MAX_HISTORY
    label_w = 1.15
    cell_w = (total_w - label_w) / HISTORY_FEATURES
    abbreviations = (
        "P", "PF", "F", "T", "R", "H", "O", "fold", "chk", "call", "bet", "raise",
        "AI", "full", "add", "to", "betB", "betA", "inc", "potB", "potA", "add/pot", "to/pot",
    )
    for col, label in enumerate(abbreviations):
        x = left + label_w + col * cell_w
        ax.add_patch(Rectangle((x, bottom + total_h), cell_w, 0.58,
                               facecolor=COLORS["history"], edgecolor="white", linewidth=0.35))
        ax.text(x + cell_w / 2, bottom + total_h + 0.29, label, ha="center", va="center",
                rotation=90 if len(label) > 3 else 0, fontsize=5.5, color="white", weight="bold")
    for slot in range(DEFAULT_MAX_HISTORY):
        y = bottom + total_h - (slot + 1) * row_h
        start = HISTORY_OFFSET + slot * HISTORY_FEATURES
        ax.add_patch(Rectangle((left, y), label_w, row_h,
                               facecolor="#73577a" if slot % 2 == 0 else COLORS["history"],
                               edgecolor="white", linewidth=0.35))
        ax.text(left + label_w / 2, y + row_h / 2,
                f"{slot:02d}  x[{start}:{start+22}]", ha="center", va="center",
                fontsize=5.6, color="white", weight="bold")
        for col in range(HISTORY_FEATURES):
            x = left + label_w + col * cell_w
            ax.add_patch(Rectangle((x, y), cell_w, row_h,
                                   facecolor="white" if slot % 2 == 0 else "#eee7f0",
                                   edgecolor="#c9bdcc", linewidth=0.22))
            ax.text(x + cell_w / 2, y + row_h / 2, str(start + col),
                    ha="center", va="center", fontsize=4.2, color=INK)
    ax.text(0.65, 11.78, "slot / index range", fontsize=7.5, color=MUTED)
    ax.text(0.65, 10.95, "oldest / padding", fontsize=7.5, color=MUTED, rotation=90, va="top")
    ax.text(0.65, 2.0, "newest", fontsize=7.5, color=MUTED, rotation=90, va="top")
    ax.add_patch(FancyArrowPatch((1.35, 10.9), (1.35, 1.25), arrowstyle="-|>", mutation_scale=14,
                                 color=COLORS["history"], linewidth=1.4))
    ax.text(23.35, 10.95,
            "P = present\nH/O = actor hero/opponent\nAI = all-in\nfull = full raise\n\n_bb columns:\nadd, to, betB, betA,\ninc, potB, potA\n\nratios:\nadd/pot, to/pot",
            fontsize=7.2, color=INK, va="top")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_actions(pdf: PdfPages) -> None:
    fig, ax = _new_page(
        "Array map 4/4 — legal mask and exact action effects x[918:1037]",
        "Ten stable policy actions; exact targets are state-dependent. Illegal actions have mask 0 and eleven zero descriptor values.",
    )
    legal_start = legal_mask_offset(DEFAULT_MAX_HISTORY)
    desc_start = action_descriptor_offset(DEFAULT_MAX_HISTORY)
    ax.text(0.7, 11.75, "LEGAL MASK (10 one-bit values)", fontsize=12, weight="bold", color=COLORS["legal"])
    for action, name in enumerate(ACTION_NAMES):
        x = 0.75 + action * 2.25
        _box(ax, x, 10.3, 2.0, 1.05, COLORS["legal"], name, f"x[{legal_start+action}]", fontsize=8.5)

    left, bottom, total_w, total_h = 0.75, 2.15, 22.25, 7.25
    row_h = total_h / len(ACTION_NAMES)
    name_w = 2.6
    cell_w = (total_w - name_w) / ACTION_DESCRIPTOR_FEATURES
    short = ("pay", "target", "pot'", "pay/potC", "target/potC", "stack'", "SPR'", "AI", "agg", "full", "reopen")
    for col, label in enumerate(short):
        x = left + name_w + col * cell_w
        ax.add_patch(Rectangle((x, bottom + total_h), cell_w, 0.62,
                               facecolor=COLORS["descriptor"], edgecolor="white", linewidth=0.5))
        ax.text(x + cell_w / 2, bottom + total_h + 0.31, label, ha="center", va="center",
                fontsize=7.3, weight="bold", color=INK)
    for action, name in enumerate(ACTION_NAMES):
        y = bottom + total_h - (action + 1) * row_h
        start = desc_start + action * ACTION_DESCRIPTOR_FEATURES
        ax.add_patch(Rectangle((left, y), name_w, row_h,
                               facecolor="#c9a62e" if action % 2 == 0 else COLORS["descriptor"],
                               edgecolor="white", linewidth=0.5))
        ax.text(left + 0.18, y + row_h * 0.62, f"{action}: {name}", fontsize=8.2,
                weight="bold", color=INK, va="center")
        ax.text(left + 0.18, y + row_h * 0.27, f"x[{start}:{start+10}]", fontsize=6.8,
                color=INK, va="center")
        for col in range(ACTION_DESCRIPTOR_FEATURES):
            x = left + name_w + col * cell_w
            ax.add_patch(Rectangle((x, y), cell_w, row_h,
                                   facecolor="white" if action % 2 == 0 else "#faf3d6",
                                   edgecolor="#d6c98f", linewidth=0.35))
            ax.text(x + cell_w / 2, y + row_h / 2, str(start + col),
                    ha="center", va="center", fontsize=6.2, color=INK)

    ax.text(0.8, 1.55,
            "Descriptor columns: " + "  •  ".join(ACTION_DESCRIPTOR_FEATURE_NAMES),
            fontsize=8.2, color=INK)
    ax.text(0.8, 1.05,
            "potC = pot_after_hero_call  |  payment, target, resulting_pot, remaining_stack are /BB  |  SPR' = resulting_effective_stack / resulting_pot",
            fontsize=8.5, color=MUTED)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_reference(pdf: PdfPages) -> None:
    fig, ax = _new_page(
        "Encoding rules, boundaries, and dimensional proof",
        "This page explains what the numbers mean and what the training observation deliberately cannot see.",
    )
    items = [
        ("Hero-visible only", "Includes the acting hero's two hole cards and all public state. The opponent's hole cards are never read, even at terminal states.", COLORS["cards"]),
        ("Scale invariant", "Chip quantities are divided by the big blind. Pot/stack features are dimensionless ratios, so scaling every chip and blind by the same factor leaves x unchanged.", COLORS["global"]),
        ("Hero-relative seats", "Hero is the player whose decision is encoded. Seat features are emitted hero first, opponent second; actor, button, blinds, and last raiser are relative labels.", COLORS["seat"]),
        ("Semantic recent history", "History stores fold/check/call/bet/raise and exact public amounts—not the policy bucket that produced the action. Only the most recent 32 events remain.", COLORS["history"]),
        ("Finite action schema", "Outputs are fold, check, call, min-raise, 1/3-pot, 1/2-pot, 3/4-pot, pot, overbet, all-in. The legal mask and descriptors bind them to exact engine transitions.", COLORS["legal"]),
        ("Decision-state contract", "Live encoding requires hero == state.to_act and exact descriptors for every legal action. Missing legal descriptors or descriptors for illegal actions are rejected.", COLORS["descriptor"]),
    ]
    for i, (title, body, color) in enumerate(items):
        col, row = i % 2, i // 2
        x, y = 0.8 + col * 11.55, 10.2 - row * 2.45
        ax.add_patch(FancyBboxPatch((x, y), 10.75, 1.85, boxstyle="round,pad=0.05,rounding_size=0.12",
                                    facecolor="white", edgecolor=color, linewidth=2))
        ax.add_patch(Rectangle((x, y), 0.2, 1.85, facecolor=color, edgecolor=color))
        ax.text(x + 0.45, y + 1.43, title, fontsize=12, weight="bold", color=INK)
        ax.text(x + 0.45, y + 1.08, fill(body, width=82), fontsize=8.6,
                color=MUTED, va="top")

    ax.text(0.8, 2.82, "Dimensional proof", fontsize=14, weight="bold", color=INK)
    proof = [
        ("Public", "15 + 2×10 + 21", 56, COLORS["global"]),
        ("Cards", "7×18", 126, COLORS["cards"]),
        ("History", "32×23", 736, COLORS["history"]),
        ("Legal", "10", 10, COLORS["legal"]),
        ("Descriptors", "10×11", 110, COLORS["descriptor"]),
    ]
    for i, (name, math_text, count, color) in enumerate(proof):
        _box(ax, 0.8 + i * 4.55, 1.25, 3.85, 1.15, color, name, f"{math_text} = {count}", fontsize=11)
        if i < len(proof) - 1:
            ax.text(4.92 + i * 4.55, 1.82, "+", fontsize=20, weight="bold", color=MUTED, ha="center")
    ax.text(23.0, 1.82, "= 1,038", fontsize=17, weight="bold", color=INK, ha="right")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def validate_contract() -> None:
    assert DEFAULT_MAX_HISTORY == 32
    assert len(POSITION_NAMES) == 15
    assert len(SEAT_FEATURE_NAMES) == 10
    assert len(GLOBAL_FEATURE_NAMES) == 21
    assert len(CARD_FEATURE_NAMES) == CARD_FEATURES == 18
    assert len(CARD_TOKEN_NAMES) == CARD_TOKEN_COUNT == 7
    assert CARD_STATE_FEATURES == 126
    assert HISTORY_OFFSET == 182
    assert len(HISTORY_FEATURE_NAMES) == HISTORY_FEATURES == 23
    assert legal_mask_offset(DEFAULT_MAX_HISTORY) == 918
    assert action_descriptor_offset(DEFAULT_MAX_HISTORY) == 928
    assert len(ACTION_DESCRIPTOR_FEATURE_NAMES) == ACTION_DESCRIPTOR_FEATURES == 11
    assert information_state_size(DEFAULT_MAX_HISTORY) == 1038


def main() -> None:
    validate_contract()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUTPUT, metadata={
        "Title": "Heads-Up Deep CFR Training Input Encoding (1,038 values)",
        "Author": "pokerCFR encoder documentation",
        "Subject": "Exact visual map of hu_information_state_v2_recent_history",
        "Keywords": "poker Deep CFR information state encoder input vector",
    }) as pdf:
        page_overview(pdf)
        page_public(pdf)
        page_cards(pdf)
        page_history(pdf)
        page_actions(pdf)
        page_reference(pdf)
    print(f"wrote {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
