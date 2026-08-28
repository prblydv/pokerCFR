"""Large differential validation for native state and relational encoders."""

from __future__ import annotations

import copy
import itertools
import random
import time

import torch

import three_player_models as models
from three_player_engine import ThreePlayerHoldemEnv as PythonEnv
from three_player_native import ThreePlayerHoldemEnv as NativeEnv


def card_tokens(hole: list[int], board: list[int]) -> torch.Tensor:
    result = torch.zeros((7, models.CARD_FEATURES), dtype=torch.float32)
    for slot, card in enumerate(hole + board):
        result[slot, card % 13] = 1.0
        result[slot, 13 + card // 13] = 1.0
        result[slot, 17] = 1.0
    return result


def street_vector(board_size: int) -> torch.Tensor:
    street = torch.zeros(4, dtype=torch.float32)
    street[{0: 0, 3: 1, 4: 2, 5: 3}[board_size]] = 1.0
    return street


def compare_relational(cards: torch.Tensor, streets: torch.Tensor) -> None:
    reference = models.poker_relational_features(
        cards, streets, use_native=False
    )
    candidate = models.poker_relational_features(cards, streets)
    if not torch.equal(reference, candidate):
        different = torch.nonzero(reference != candidate)
        row, feature = map(int, different[0])
        raise AssertionError(
            f"relational mismatch row={row} feature={feature}: "
            f"torch={reference[row, feature]} native={candidate[row, feature]}"
        )


def exhaustive_hole_sweep() -> int:
    # Each canonical starting combination is tested on every street. Boards are
    # rotated when they overlap a hole card, preserving distinct physical cards.
    checked = 0
    rows, streets = [], []
    board_templates = (
        [],
        [0, 14, 28],
        [1, 16, 31, 46],
        [3, 17, 32, 47, 9],
    )
    for hole in itertools.combinations(range(52), 2):
        for template in board_templates:
            used = set(hole)
            board = []
            for preferred in template:
                card = preferred
                while card in used:
                    card = (card + 1) % 52
                used.add(card)
                board.append(card)
            rows.append(card_tokens(list(hole), board))
            streets.append(street_vector(len(board)))
            checked += 1
    compare_relational(torch.stack(rows), torch.stack(streets))
    return checked


def randomized_relational(count: int = 250_000, batch_size: int = 4096) -> int:
    rng = random.Random(771_209)
    checked = 0
    board_sizes = (0, 3, 4, 5)
    while checked < count:
        size = min(batch_size, count - checked)
        rows, streets = [], []
        for offset in range(size):
            board_size = board_sizes[(checked + offset) % 4]
            dealt = rng.sample(range(52), 2 + board_size)
            rows.append(card_tokens(dealt[:2], dealt[2:]))
            streets.append(street_vector(board_size))
        compare_relational(torch.stack(rows), torch.stack(streets))
        checked += size
    return checked


def targeted_relational_cases() -> int:
    c = lambda rank, suit: rank + 13 * suit
    cases = [
        # Straight flush, wheel straight, full house and quads.
        ([c(6, 0), c(7, 0)], [c(8, 0), c(9, 0), c(10, 0), c(1, 2), c(3, 3)]),
        ([c(12, 0), c(0, 1)], [c(1, 2), c(2, 3), c(3, 0), c(8, 1), c(10, 2)]),
        ([c(12, 0), c(12, 1)], [c(12, 2), c(11, 0), c(11, 1), c(4, 2), c(2, 3)]),
        ([c(12, 0), c(12, 1)], [c(12, 2), c(12, 3), c(11, 1), c(4, 2), c(2, 3)]),
        # Open-ended, gutshot, flush draw, backdoor flush, paired/trip boards.
        ([c(6, 0), c(7, 1)], [c(8, 2), c(9, 3), c(1, 0)]),
        ([c(6, 0), c(8, 1)], [c(9, 2), c(10, 3), c(1, 0)]),
        ([c(12, 0), c(8, 0)], [c(3, 0), c(5, 0), c(1, 2)]),
        ([c(12, 0), c(8, 0)], [c(3, 0), c(5, 1), c(1, 2)]),
        ([c(12, 0), c(8, 1)], [c(3, 0), c(3, 1), c(1, 2)]),
        ([c(12, 0), c(8, 1)], [c(3, 0), c(3, 1), c(3, 2)]),
    ]
    rows, streets = [], []
    for hole, board in cases:
        rows.append(card_tokens(hole, board))
        streets.append(street_vector(len(board)))
    compare_relational(torch.stack(rows), torch.stack(streets))

    features = models.poker_relational_features(torch.stack(rows), torch.stack(streets))
    # Category offsets: 34..42. Scalar offsets: 48..65.
    assert features[0, 42] == 1 and features[1, 38] == 1
    assert features[2, 40] == 1 and features[3, 41] == 1
    assert features[4, 53] == 1 and features[5, 54] == 1
    assert features[6, 55] == 1 and features[7, 56] == 1
    assert features[8, 57] == 1 and features[9, 58] == 1
    return len(cases)


def randomized_state_encoders(hands: int = 3_000) -> tuple[int, list[torch.Tensor]]:
    rng = random.Random(918_277)
    modes = (
        [200.0, 200.0, 200.0],
        [0.0, 241.0, 359.0],
        [397.0, 2.0, 201.0],
        [75.5, 410.25, 114.25],
        [598.0, 1.0, 1.0],
    )
    checked = 0
    network_states: list[torch.Tensor] = []
    for hand in range(hands):
        deck = list(range(52))
        rng.shuffle(deck)
        stacks = modes[hand % len(modes)]
        live = [seat for seat, value in enumerate(stacks) if value > 0]
        button = live[hand % len(live)]
        py_env, native_env = PythonEnv(seed=1), NativeEnv(seed=1)
        py_state = py_env.new_hand(button=button, stacks=stacks, deck=deck)
        native_state = native_env.new_hand(button=button, stacks=stacks, deck=deck)
        while not py_state.terminal:
            legal = py_env.legal_actions(py_state)
            if native_env.legal_actions(native_state) != legal:
                raise AssertionError("legal actions diverged")
            for max_history in (1, 4, 32):
                for tournament in (False, True):
                    kwargs = dict(
                        include_tournament_features=tournament,
                        tournament_total_chips=600.0 if tournament else None,
                    )
                    reference = models.encode_information_state(
                        py_state, py_state.to_act, legal, 200.0, max_history, **kwargs
                    )
                    candidate = models.encode_information_state(
                        native_state, native_state.to_act, legal, 200.0, max_history, **kwargs
                    )
                    if not torch.equal(reference, candidate):
                        different = torch.nonzero(reference != candidate).flatten()
                        raise AssertionError(
                            f"state encoder mismatch hand={hand}, history={max_history}, "
                            f"tournament={tournament}, features={different[:10].tolist()}"
                        )
                    if max_history == 32 and tournament and len(network_states) < 8192:
                        network_states.append(candidate)
                    checked += 1
            action = rng.choice(legal)
            py_state = py_env.step(py_state, action)
            native_state = native_env.step(native_state, action)
    return checked, network_states


def network_and_gradient_parity(encoded: list[torch.Tensor]) -> int:
    x = torch.stack(encoded)
    torch.manual_seed(3301)
    network = models.DeepCFRBranchV2Network(x.shape[1], hidden=128, blocks=3)
    with torch.no_grad():
        for head in network.street_heads:
            head.weight.normal_(mean=0.0, std=0.02)
            head.bias.normal_(mean=0.0, std=0.02)
    reference_network = copy.deepcopy(network)
    native_output = network(x)
    original = models.poker_relational_features
    models.poker_relational_features = (
        lambda cards, streets: original(cards, streets, use_native=False)
    )
    try:
        reference_output = reference_network(x)
        if not torch.equal(native_output, reference_output):
            raise AssertionError("complete network outputs are not bit exact")
        target = torch.randn(native_output.shape, generator=torch.Generator().manual_seed(44))
        (native_output - target).square().mean().backward()
        (reference_output - target).square().mean().backward()
        for (name_a, parameter_a), (name_b, parameter_b) in zip(
            network.named_parameters(), reference_network.named_parameters()
        ):
            assert name_a == name_b
            if not torch.equal(parameter_a.grad, parameter_b.grad):
                raise AssertionError(f"gradient mismatch in {name_a}")
    finally:
        models.poker_relational_features = original
    return int(x.shape[0])


def main() -> None:
    started = time.perf_counter()
    targeted = targeted_relational_cases()
    holes = exhaustive_hole_sweep()
    random_features = randomized_relational()
    encoded_states, network_states = randomized_state_encoders()
    network_rows = network_and_gradient_parity(network_states)
    elapsed = time.perf_counter() - started
    print(f"targeted relational cases: {targeted}")
    print(f"exhaustive physical-hole/street cases: {holes}")
    print(f"random relational cases: {random_features}")
    print(f"state encoder comparisons: {encoded_states}")
    print(f"network output + gradient comparisons: {network_rows}")
    print(f"ALL EXACT; elapsed={elapsed:.2f}s")


if __name__ == "__main__":
    main()
