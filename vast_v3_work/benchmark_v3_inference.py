"""Compare v2 and v3 forward speed on traversal and fit batch shapes."""

from __future__ import annotations

import time

import torch

from three_player_models import (
    build_policy_network,
    encode_information_state,
    information_state_size,
)
from three_player_native import ThreePlayerHoldemEnv


def elapsed(model, batch: torch.Tensor, repeats: int) -> float:
    with torch.inference_mode():
        for _ in range(10):
            model(batch)
        if batch.is_cuda:
            torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(repeats):
            model(batch)
        if batch.is_cuda:
            torch.cuda.synchronize()
    return time.perf_counter() - started


def main() -> None:
    torch.set_num_threads(1)
    input_dim = information_state_size(
        32, include_tournament_features=True
    )
    env = ThreePlayerHoldemEnv(stack_size=200, sb=1, bb=2, seed=903)
    state = env.new_hand()
    encoded = encode_information_state(
        state,
        int(state.to_act),
        env.legal_actions(state),
        200,
        32,
        include_tournament_features=True,
        tournament_total_chips=600,
    ).unsqueeze(0)

    cpu_times = {}
    for architecture, hidden, blocks in (
        ("deep_cfr_branch_v2", 128, 3),
        ("deep_cfr_branch_v3", 256, 4),
    ):
        model = build_policy_network(
            architecture, input_dim, hidden, blocks
        ).eval()
        cpu_times[architecture] = elapsed(model, encoded, 500)
        print(
            f"cpu_batch1_{architecture}_us="
            f"{1e6 * cpu_times[architecture] / 500:.3f}"
        )
    print(
        "cpu_batch1_v3_over_v2="
        f"{cpu_times['deep_cfr_branch_v3'] / cpu_times['deep_cfr_branch_v2']:.3f}"
    )

    if torch.cuda.is_available():
        gpu_times = {}
        batch = encoded.repeat(8192, 1).cuda()
        for architecture, hidden, blocks in (
            ("deep_cfr_branch_v2", 128, 3),
            ("deep_cfr_branch_v3", 256, 4),
        ):
            model = build_policy_network(
                architecture, input_dim, hidden, blocks
            ).cuda().eval()
            gpu_times[architecture] = elapsed(model, batch, 20)
            print(
                f"gpu_batch8192_{architecture}_ms="
                f"{1e3 * gpu_times[architecture] / 20:.3f}"
            )
        print(
            "gpu_batch8192_v3_over_v2="
            f"{gpu_times['deep_cfr_branch_v3'] / gpu_times['deep_cfr_branch_v2']:.3f}"
        )

        torch.cuda.empty_cache()
        model = build_policy_network(
            "deep_cfr_branch_v3", input_dim, 256, 4
        ).cuda().train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        targets = torch.randn(8192, 9, device="cuda")
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        prediction = model(batch)
        loss = (prediction - targets).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        print(
            "gpu_batch8192_v3_train_step_ms="
            f"{1e3 * (time.perf_counter() - started):.3f}"
        )
        print(
            "gpu_batch8192_v3_train_peak_mb="
            f"{torch.cuda.max_memory_allocated() / 1_000_000:.1f}"
        )


if __name__ == "__main__":
    main()
