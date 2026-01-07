import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def load_jsonl(path: Path) -> List[dict]:
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def find_latest_history_file(history_dir: Path) -> Optional[Path]:
    if not history_dir.exists():
        return None
    files = sorted(history_dir.glob("*_log.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def get_hero_name(records: List[dict]) -> Optional[str]:
    for r in records:
        if r.get("type") == "hand_start":
            for s in r.get("seats", []):
                if s.get("is_hero"):
                    return s.get("name") or s.get("label")
    return None


def get_session_bb(records: List[dict]) -> float:
    for r in records:
        if r.get("type") == "session_start":
            bb = r.get("bb")
            if isinstance(bb, (int, float)):
                return float(bb)
            break
    return 0.0


def compute_hand_stacks(records: List[dict]) -> List[Tuple[int, float]]:
    hand_stacks: List[Tuple[int, float]] = []
    for r in records:
        if r.get("type") != "hand_start":
            continue
        hero_stack = None
        for s in r.get("seats", []):
            if s.get("is_hero"):
                hero_stack = s.get("stack")
                break
        if hero_stack is not None:
            hand_stacks.append((r.get("hand_id", -1), float(hero_stack)))
    return hand_stacks


def compute_action_stats(records: List[dict]) -> Dict[str, object]:
    hero_actions = [r for r in records if r.get("type") == "action" and r.get("is_hero")]
    all_hands = {r.get("hand_id") for r in records if r.get("type") == "hand_start"}

    act_counts = Counter([a.get("action_name") for a in hero_actions])
    preflop_actions = [a for a in hero_actions if a.get("street") == 0]

    vpip_hands = set()
    pfr_hands = set()
    preflop_folds = set()

    vpip_actions = {"CALL", "BET_POT_25", "BET_POT_50", "BET_POT_100", "BET_POT_200", "ALL_IN"}
    pfr_actions = {"BET_POT_25", "BET_POT_50", "BET_POT_100", "BET_POT_200", "ALL_IN"}

    for a in preflop_actions:
        hid = a.get("hand_id")
        name = a.get("action_name")
        if name in vpip_actions:
            vpip_hands.add(hid)
        if name in pfr_actions:
            pfr_hands.add(hid)
        if name == "FOLD":
            preflop_folds.add(hid)

    def pct(count: int, total: int) -> float:
        return (count / total * 100.0) if total else 0.0

    vpip = pct(len(vpip_hands), len(all_hands))
    pfr = pct(len(pfr_hands), len(all_hands))
    pf_fold = pct(len(preflop_folds), len(all_hands))

    # Aggression ratio: raises / calls (exclude preflop if needed later)
    calls = act_counts.get("CALL", 0)
    raises = (
        act_counts.get("BET_POT_25", 0)
        + act_counts.get("BET_POT_50", 0)
        + act_counts.get("BET_POT_100", 0)
        + act_counts.get("BET_POT_200", 0)
        + act_counts.get("ALL_IN", 0)
    )
    aggression = (raises / calls) if calls else float("inf") if raises else 0.0

    return {
        "hero_actions": hero_actions,
        "action_counts": act_counts,
        "vpip": vpip,
        "pfr": pfr,
        "pf_fold": pf_fold,
        "aggression_ratio": aggression,
        "hands": len(all_hands),
    }


def compute_opponent_action_stats(records: List[dict]) -> Dict[str, object]:
    opp_actions = [r for r in records if r.get("type") == "action" and not r.get("is_hero")]
    opp_counts = Counter([a.get("action_name") for a in opp_actions])
    total = sum(opp_counts.values())
    opp_pct = {k: (v / total * 100.0) if total else 0.0 for k, v in opp_counts.items()}

    per_player = defaultdict(Counter)
    for a in opp_actions:
        label = a.get("label") or a.get("name") or f"Seat{(a.get('seat_index') or 0) + 1}"
        per_player[label][a.get("action_name")] += 1

    per_player_summary = {}
    for player, counts in per_player.items():
        total_p = sum(counts.values())
        top = counts.most_common(3)
        per_player_summary[player] = {
            "total_actions": total_p,
            "top_actions": top,
        }

    return {
        "opponent_counts": opp_counts,
        "opponent_pct": opp_pct,
        "per_player": per_player_summary,
    }


def compute_stack_delta(hand_stacks: List[Tuple[int, float]]) -> Tuple[float, List[float]]:
    deltas: List[float] = []
    for i in range(1, len(hand_stacks)):
        prev = hand_stacks[i - 1][1]
        curr = hand_stacks[i][1]
        deltas.append(curr - prev)
    return sum(deltas), deltas


def summarize(path: Path) -> str:
    records = load_jsonl(path)
    hero_name = get_hero_name(records) or "Hero"
    bb = get_session_bb(records)

    hand_stacks = compute_hand_stacks(records)
    total_delta, deltas = compute_stack_delta(hand_stacks)
    stats = compute_action_stats(records)
    opp_stats = compute_opponent_action_stats(records)

    total_delta_bb = total_delta / bb if bb else None
    avg_delta_bb = (total_delta_bb / stats["hands"]) if (bb and stats["hands"]) else None

    lines = []
    lines.append(f"File: {path}")
    lines.append(f"Hero: {hero_name}")
    lines.append(f"Hands: {stats['hands']} | Hero actions: {len(stats['hero_actions'])}")
    lines.append(f"Net: {total_delta:.2f}{' (' + str(round(total_delta_bb, 2)) + 'bb)' if total_delta_bb is not None else ''}")
    if avg_delta_bb is not None:
        lines.append(f"BB/hand: {avg_delta_bb:.3f}")
    lines.append(
        f"VPIP: {stats['vpip']:.1f}% | PFR: {stats['pfr']:.1f}% | PF Fold: {stats['pf_fold']:.1f}%"
    )
    lines.append(
        f"Action counts: {dict(stats['action_counts'])}"
    )
    lines.append(f"Aggression ratio (raises/calls): {stats['aggression_ratio']:.2f}")
    lines.append(
        "Opponents (overall action mix %): "
        + ", ".join(f"{k} {opp_stats['opponent_pct'][k]:.1f}%" for k in sorted(opp_stats["opponent_pct"]))
    )
    top_players = sorted(
        opp_stats["per_player"].items(), key=lambda item: item[1]["total_actions"], reverse=True
    )[:5]
    if top_players:
        lines.append("Opponents (top action patterns):")
        for player, pdata in top_players:
            top_actions = ", ".join(f"{name} {count}" for name, count in pdata["top_actions"])
            lines.append(f"  - {player}: {top_actions}")
    return "\n".join(lines)


def main():
    history_dir = Path("tablehistory")
    latest = find_latest_history_file(history_dir)
    if not latest:
        print("No history logs found in tablehistory/.")
        return
    print(summarize(latest))


if __name__ == "__main__":
    main()
