import logging
import os
import random
from typing import Dict, List, Optional, Set, Tuple

import torch

from config import DEVICE, LOG_LEVEL, LOG_FORMAT
from poker_env import (
    SimpleHoldemEnv,
    NUM_ACTIONS,
    ACTION_CALL,
    STREET_PREFLOP,
)
from abstraction import encode_state
from networks import PolicyNet, move_to_device
from deep_cfr_trainer import (
    scripted_eval_action,
    scripted_tag_action,
    RAISE_ACTIONS,
)
from opponent_model import ExploitController, update_from_action


logger = logging.getLogger("Team3v3Eval")
RNG = random.Random(1337)


def _parse_seat_list(value: str) -> List[int]:
    """Parse a comma-separated list of player seats."""
    if not value:
        return []

    seats = []
    for item in value.split(","):
        item = item.strip()
        if item:
            seats.append(int(item))

    return seats


def _validate_seats(
    seats: List[int],
    num_players: int,
    name: str,
) -> None:
    """Validate that all configured seats are valid."""
    invalid = [seat for seat in seats if seat < 0 or seat >= num_players]

    if invalid:
        raise ValueError(
            f"{name} contains invalid seats {invalid}. "
            f"Valid seats are 0 through {num_players - 1}."
        )

    if len(set(seats)) != len(seats):
        raise ValueError(f"{name} contains duplicate seats: {seats}")


def _sample_policy_action(
    policy_net: PolicyNet,
    state,
    player: int,
    legal_actions: List[int],
) -> int:
    """Sample an action from the policy while respecting legal actions."""
    if not legal_actions:
        return 0

    x = encode_state(state, player).to(DEVICE)

    with torch.no_grad():
        logits = policy_net(x.unsqueeze(0)).squeeze(0)

    # Mask illegal actions.
    mask = torch.full(
        (NUM_ACTIONS,),
        -1e9,
        device=logits.device,
    )

    for action in legal_actions:
        mask[action] = 0.0

    probs = torch.softmax(logits + mask, dim=-1)

    action = torch.multinomial(probs, 1).item()

    # Defensive fallback.
    if action not in legal_actions:
        action = RNG.choice(legal_actions)

    return action


def _init_stats(num_players: int) -> Dict[int, Dict]:
    """Create empty statistics for every player."""
    return {
        pid: {
            "hands": 0,
            "wins": 0,
            "showdown_hands": 0,
            "showdown_wins": 0,
            "vpip": 0,
            "pfr": 0,
            "aggr": 0,
            "calls": 0,
            "actions": 0,
            "payoff": 0.0,
        }
        for pid in range(num_players)
    }


def _select_scripted_opponent(
    state,
    player: int,
    scripted_seats: Set[int],
) -> Optional[int]:
    """Select a scripted opponent for opponent modelling."""

    last_aggressor = getattr(
        state,
        "last_aggressor",
        None,
    )

    if (
        last_aggressor is not None
        and last_aggressor in scripted_seats
        and last_aggressor != player
    ):
        if (
            not state.folded[last_aggressor]
            and state.stacks[last_aggressor] > 0
        ):
            return last_aggressor

    for i in range(state.num_players):
        pid = (player + 1 + i) % state.num_players

        if pid in scripted_seats and pid != player:
            if not state.folded[pid] and state.stacks[pid] > 0:
                return pid

    return None


def _summarize_player(stats: Dict) -> Dict:
    """Convert raw player statistics into percentages and averages."""

    hands = max(stats["hands"], 1)

    win_pct = 100.0 * stats["wins"] / hands
    vpip_pct = 100.0 * stats["vpip"] / hands
    pfr_pct = 100.0 * stats["pfr"] / hands

    calls = stats["calls"]
    aggression = stats["aggr"]

    if calls == 0:
        af = float("inf") if aggression > 0 else 0.0
    else:
        af = aggression / calls

    avg_actions = stats["actions"] / hands
    ev = stats["payoff"] / hands

    showdown_hands = max(
        stats["showdown_hands"],
        1,
    )

    showdown_win_pct = (
        100.0
        * stats["showdown_wins"]
        / showdown_hands
    )

    showdown_seen_pct = (
        100.0
        * stats["showdown_hands"]
        / hands
    )

    return {
        "hands": hands,
        "win_pct": win_pct,
        "showdown_win_pct": showdown_win_pct,
        "showdown_seen_pct": showdown_seen_pct,
        "vpip_pct": vpip_pct,
        "pfr_pct": pfr_pct,
        "af": af,
        "avg_actions": avg_actions,
        "ev": ev,
    }


def _team_summary(
    stats: Dict[int, Dict],
    seats: List[int],
    total_hands: int,
    bb: float,
) -> Dict:
    """Calculate aggregate statistics for a team."""

    if not seats:
        return {}

    total_ev = sum(
        stats[pid]["payoff"]
        for pid in seats
    )

    team_wins = sum(
        stats[pid]["wins"]
        for pid in seats
    )

    team_showdown_wins = sum(
        stats[pid]["showdown_wins"]
        for pid in seats
    )

    team_showdown_hands = sum(
        stats[pid]["showdown_hands"]
        for pid in seats
    )

    avg_ev = total_ev / max(1, total_hands)

    showdown_win_pct = (
        100.0
        * team_showdown_wins
        / max(1, team_showdown_hands)
    )

    return {
        "team_seats": seats,
        "hands": total_hands,
        "wins": team_wins,
        "avg_ev": avg_ev,
        "bb_per_100": (
            avg_ev / max(bb, 1e-9)
        ) * 100.0,
        "showdown_win_pct": showdown_win_pct,
    }


def run_match(
    policy_path: str,
    num_hands: int,
    num_players: int,
    scripted_eval_seats: List[int],
    scripted_tag_seats: List[int],
    use_opponent_modeling: bool,
) -> Tuple[Dict[int, Dict], Dict]:

    env = SimpleHoldemEnv(
        num_players=num_players
    )

    if num_hands <= 0:
        raise ValueError(
            "num_hands must be greater than zero."
        )

    if not os.path.isfile(policy_path):
        raise FileNotFoundError(
            f"Policy file not found: {policy_path}"
        )

    # Validate configured seats.
    _validate_seats(
        scripted_eval_seats,
        num_players,
        "scripted_eval_seats",
    )

    _validate_seats(
        scripted_tag_seats,
        num_players,
        "scripted_tag_seats",
    )

    if set(scripted_eval_seats) & set(scripted_tag_seats):
        raise ValueError(
            "A seat cannot be both scripted_eval "
            "and scripted_tag."
        )

    # Determine policy input dimension.
    initial_state = env.new_hand()

    state_dim = encode_state(
        initial_state,
        0,
    ).shape[0]

    # Load policy.
    policy = move_to_device(
        PolicyNet(state_dim)
    )

    policy.load_state_dict(
        torch.load(
            policy_path,
            map_location=DEVICE,
        )
    )

    policy.eval()

    stats = _init_stats(num_players)

    exploit_controller = (
        ExploitController(num_players)
        if use_opponent_modeling
        else None
    )

    scripted_eval_seats = set(
        scripted_eval_seats
    )

    scripted_tag_seats = set(
        scripted_tag_seats
    )

    scripted_seats = (
        scripted_eval_seats
        | scripted_tag_seats
    )

    policy_seats = [
        pid
        for pid in range(num_players)
        if pid not in scripted_seats
    ]

    # Run requested number of hands.
    for hand_number in range(num_hands):

        state = env.new_hand()

        hand_flags = {
            pid: {
                "vpip": False,
                "pfr": False,
                "aggr": 0,
                "calls": 0,
                "actions": 0,
            }
            for pid in range(num_players)
        }

        while not state.terminal:

            legal_actions = env.legal_actions(
                state
            )

            if not legal_actions:
                break

            player = state.to_act

            # Scripted evaluation player.
            if player in scripted_eval_seats:

                action = scripted_eval_action(
                    state,
                    player,
                    legal_actions,
                )

            # Scripted TAG player.
            elif player in scripted_tag_seats:

                action = scripted_tag_action(
                    state,
                    player,
                    legal_actions,
                )

            # Learned policy player.
            else:

                if exploit_controller is not None:

                    x = encode_state(
                        state,
                        player,
                    ).to(DEVICE)

                    with torch.no_grad():
                        logits = policy(
                            x.unsqueeze(0)
                        ).squeeze(0)

                    opponent = _select_scripted_opponent(
                        state,
                        player,
                        scripted_seats,
                    )

                    if opponent is not None:

                        action = (
                            exploit_controller.choose_action(
                                state,
                                player,
                                legal_actions,
                                logits,
                                opponent_override=opponent,
                            )
                        )

                    else:

                        action = _sample_policy_action(
                            policy,
                            state,
                            player,
                            legal_actions,
                        )

                else:

                    action = _sample_policy_action(
                        policy,
                        state,
                        player,
                        legal_actions,
                    )

            # Record action statistics.
            info = hand_flags[player]

            info["actions"] += 1

            if state.street == STREET_PREFLOP:

                if (
                    action == ACTION_CALL
                    or action in RAISE_ACTIONS
                ):
                    info["vpip"] = True

                if action in RAISE_ACTIONS:
                    info["pfr"] = True

            if action in RAISE_ACTIONS:
                info["aggr"] += 1

            elif action == ACTION_CALL:
                info["calls"] += 1

            # Update opponent model.
            if (
                exploit_controller is not None
                and player in scripted_seats
            ):

                to_call = max(
                    0.0,
                    state.current_bet
                    - state.contrib[player],
                )

                facing_bet = to_call > 0

                bet_bucket = (
                    action
                    if action in RAISE_ACTIONS
                    else None
                )

                update_from_action(
                    exploit_controller,
                    state,
                    player,
                    action,
                    state.street,
                    facing_bet,
                    bet_bucket,
                )

            # Advance environment.
            state = env.step(
                state,
                action,
            )

        # Determine whether this hand reached showdown.
        showdown = (
            state.terminal
            and len(state.board) >= 5
        )

        # Store hand results.
        for pid in range(num_players):

            stats[pid]["hands"] += 1

            if state.winner == pid:

                stats[pid]["wins"] += 1

                if showdown:
                    stats[pid]["showdown_wins"] += 1

            if (
                showdown
                and not state.folded[pid]
            ):
                stats[pid]["showdown_hands"] += 1

            if hand_flags[pid]["vpip"]:
                stats[pid]["vpip"] += 1

            if hand_flags[pid]["pfr"]:
                stats[pid]["pfr"] += 1

            stats[pid]["aggr"] += (
                hand_flags[pid]["aggr"]
            )

            stats[pid]["calls"] += (
                hand_flags[pid]["calls"]
            )

            stats[pid]["actions"] += (
                hand_flags[pid]["actions"]
            )

            stats[pid]["payoff"] += (
                env.terminal_payoff(
                    state,
                    pid,
                )
            )

    # Build final summaries.
    summaries = {
        pid: _summarize_player(
            stats[pid]
        )
        for pid in range(num_players)
    }

    team_policy = _team_summary(
        stats,
        policy_seats,
        num_hands,
        env.bb,
    )

    team_scripted = _team_summary(
        stats,
        sorted(scripted_seats),
        num_hands,
        env.bb,
    )

    return summaries, {
        "policy_team": team_policy,
        "scripted_team": team_scripted,
    }


def _print_results(
    summaries: Dict[int, Dict],
    team_summaries: Dict,
) -> None:
    """Print player and team evaluation results."""

    def fmt_af(value):
        if value == float("inf"):
            return "inf"

        return f"{value:.2f}"

    print("\nPer-player results")

    for pid, summary in summaries.items():

        print(
            f"P{pid}: "
            f"hands={summary['hands']}, "
            f"win%={summary['win_pct']:.1f}, "
            f"ev/hand={summary['ev']:.3f}, "
            f"showdown_win%="
            f"{summary['showdown_win_pct']:.1f}, "
            f"showdown_seen%="
            f"{summary['showdown_seen_pct']:.1f}, "
            f"VPIP%={summary['vpip_pct']:.1f}, "
            f"PFR%={summary['pfr_pct']:.1f}, "
            f"AF={fmt_af(summary['af'])}, "
            f"avg_actions="
            f"{summary['avg_actions']:.2f}"
        )

    print("\nTeam results")

    for label, summary in team_summaries.items():

        if not summary:
            continue

        print(
            f"{label}: "
            f"seats={summary['team_seats']}, "
            f"hands={summary['hands']}, "
            f"wins={summary['wins']}, "
            f"avg_ev/hand="
            f"{summary['avg_ev']:.3f}, "
            f"bb/100="
            f"{summary['bb_per_100']:.2f}, "
            f"showdown_win%="
            f"{summary['showdown_win_pct']:.1f}"
        )


def main() -> None:

    policy_path = "models/policyev_42.pt"

    # Change this number to control the exact
    # number of hands in each match.
    hands = 5000

    num_players = 6

    scripted_team_seats = "0,2,4"

    use_opponent_modeling = True

    logging.basicConfig(
        level=LOG_LEVEL,
        format=LOG_FORMAT,
    )

    scripted_team_seats = _parse_seat_list(
        scripted_team_seats
    )

    if len(scripted_team_seats) == 0:
        raise ValueError(
            "At least one scripted seat is required "
            "for a 3v3 matchup."
        )

    _validate_seats(
        scripted_team_seats,
        num_players,
        "scripted_team_seats",
    )

    if len(scripted_team_seats) * 2 != num_players:

        logger.warning(
            "Non-3v3 setup: scripted seats=%s, "
            "num_players=%s",
            sorted(scripted_team_seats),
            num_players,
        )

    print(
        f"Running evaluation with "
        f"{hands} hands per match."
    )

    # -------------------------------------------------
    # MATCH A
    # scripted_eval_action vs policy
    # -------------------------------------------------

    print(
        "\nMatch A: "
        "scripted_eval_action team vs policy team"
    )

    summaries_a, team_summaries_a = run_match(
        policy_path=policy_path,
        num_hands=hands,
        num_players=num_players,
        scripted_eval_seats=scripted_team_seats,
        scripted_tag_seats=[],
        use_opponent_modeling=use_opponent_modeling,
    )

    _print_results(
        summaries_a,
        team_summaries_a,
    )

    # -------------------------------------------------
    # MATCH B
    # scripted_tag_action vs policy
    # -------------------------------------------------

    print(
        "\nMatch B: "
        "scripted_tag_action team vs policy team"
    )

    summaries_b, team_summaries_b = run_match(
        policy_path=policy_path,
        num_hands=hands,
        num_players=num_players,
        scripted_eval_seats=[],
        scripted_tag_seats=scripted_team_seats,
        use_opponent_modeling=use_opponent_modeling,
    )

    _print_results(
        summaries_b,
        team_summaries_b,
    )


if __name__ == "__main__":
    main()
