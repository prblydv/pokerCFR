# ---------------------------------------------------------------------------
# File overview:
#   networks.py defines small MLP Advantage/Policy networks for Deep CFR.
# ---------------------------------------------------------------------------

import torch
import torch.nn as nn

from config import NUM_ACTIONS, DEVICE


# ---------------------------------------------------------
# Advantage Network (Regret / Advantage function)
# ---------------------------------------------------------

class AdvantageNet(nn.Module):
    """
    Small MLP approximator for regret/advantages.

    Input:  state vector (B, state_dim)
    Output: advantages (B, NUM_ACTIONS), unconstrained real values.
    """
    def __init__(self, state_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, NUM_ACTIONS),
        )

        nn.init.uniform_(self.net[-1].weight, -1e-3, 1e-3)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------
# Policy Network (Average Strategy)
# ---------------------------------------------------------

class PolicyNet(nn.Module):
    """
    Small MLP approximator for the average strategy.

    Input:  state vector (B, state_dim)
    Output: logits (B, NUM_ACTIONS). Softmax is applied outside.
    """
    def __init__(self, state_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, NUM_ACTIONS),
        )

        nn.init.uniform_(self.net[-1].weight, -1e-3, 1e-3)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------
# Device helper
# ---------------------------------------------------------

def move_to_device(model: nn.Module) -> nn.Module:
    return model.to(DEVICE)
