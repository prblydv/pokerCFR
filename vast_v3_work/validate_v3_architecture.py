"""Focused correctness and size validation for deep_cfr_branch_v3."""

from __future__ import annotations

import time

import torch

from three_player_cfr import ThreePlayerNeuralCFR
from three_player_models import (
    HISTORY_FEATURES,
    HISTORY_OFFSET,
    build_advantage_network,
    build_policy_network,
    information_state_size,
)
from three_player_native import ThreePlayerHoldemEnv


def main() -> None:
    input_dim = information_state_size(
        32, include_tournament_features=True
    )
    policy = build_policy_network(
        "deep_cfr_branch_v3", input_dim, 256, 4
    )
    advantage = build_advantage_network(
        "deep_cfr_branch_v3", input_dim, 256, 4
    )
    parameters = sum(parameter.numel() for parameter in policy.parameters())
    policy_megabytes = 3 * parameters * 4 / 1_000_000
    assert policy_megabytes >= 10.0

    env = ThreePlayerHoldemEnv(stack_size=200, sb=1, bb=2, seed=901)
    trainer = ThreePlayerNeuralCFR(
        env,
        device="cpu",
        hidden=256,
        blocks=4,
        network_architecture="deep_cfr_branch_v3",
        max_history=32,
        include_tournament_features=True,
        tournament_total_chips=600,
        advantage_capacity=256,
        policy_capacity=256,
        recent_capacity=0,
        max_nodes_per_traversal=64,
        max_depth=16,
        reinitialize_advantage_each_iteration=False,
        seed=902,
    )
    state = env.new_hand()
    legal = env.legal_actions(state)
    encoded = trainer.encode(state, int(state.to_act), legal)
    batch = encoded.unsqueeze(0).repeat(8, 1)
    batch[:, 0] = 1.0
    outputs = policy(batch)
    assert outputs.shape == (8, 9)
    assert torch.isfinite(outputs).all()

    altered = batch.clone()
    history_slot = HISTORY_OFFSET + 31 * HISTORY_FEATURES
    altered[:, history_slot : history_slot + 4] = torch.tensor(
        [1.0, 0.0, 0.0, 0.0]
    )
    altered[:, history_slot + 4 : history_slot + 7] = torch.tensor(
        [0.0, 1.0, 0.0]
    )
    altered[:, history_slot + 7 + 1] = 1.0
    policy.eval()
    original_representation = policy.backbone(batch)
    altered_representation = policy.backbone(altered)
    assert not torch.allclose(
        original_representation, altered_representation
    )

    optimizer = torch.optim.AdamW(advantage.parameters(), lr=1e-3)
    targets = torch.randn(8, 9)
    prediction = advantage(altered)
    loss = (prediction - targets).square().mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)

    started = time.perf_counter()
    result = trainer.train_iteration(
        traversals_per_player=1,
        advantage_steps=1,
        policy_steps=1,
        batch_size=32,
        traversal_workers=1,
    )
    elapsed = time.perf_counter() - started
    assert int(result["iteration"]) == 1
    assert result["nodes"] > 0
    print(f"input_dim={input_dim}")
    print(f"parameters_per_network={parameters}")
    print(f"three_policy_weight_mb={policy_megabytes:.3f}")
    print(f"diagnostic_iteration_seconds={elapsed:.3f}")
    print(f"diagnostic_nodes={int(result['nodes'])}")


if __name__ == "__main__":
    main()
