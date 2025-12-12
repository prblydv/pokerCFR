# ---------------------------------------------------------------------------
# File overview:
#   networks.py defines residual Advantage/Policy networks and helpers for
#   Deep CFR experiments. Import this module; it has no direct CLI entry.
# ---------------------------------------------------------------------------

import torch
import torch.nn as nn

from config import NUM_ACTIONS, DEVICE


# ---------------------------------------------------------
# Utility blocks
# ---------------------------------------------------------

class ResidualBlock(nn.Module):
    """
    Pre-norm residual MLP block:
    y = x + Dropout(FF(LayerNorm(x)))
    FF: Linear → GELU → Linear
    """
    # Function metadata:
    #   Inputs: dim (int), hidden_mult (int), dropout (float)
    #   Sample:
    #       block = ResidualBlock(dim=128)
    #       # dtype=ResidualBlock
    def __init__(self, dim: int, hidden_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * hidden_mult)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * hidden_mult, dim)
        self.dropout = nn.Dropout(dropout)

    # Function metadata:
    #   Inputs: x (torch.Tensor) shape (B, dim)
    #   Sample:
    #       y = block.forward(torch.zeros(2, 128))  # dtype=torch.FloatTensor
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.fc2(h)
        h = self.dropout(h)
        return x + h


class GatingHead(nn.Module):
    """
    Optional gating on top of the shared trunk.
    Helps shape outputs for each head (advantage / policy).
    """
    # Function metadata:
    #   Inputs: dim (int)
    #   Sample:
    #       head = GatingHead(dim=512)
    #       # dtype=GatingHead
    def __init__(self, dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, dim)
        self.act = nn.SiLU()

    # Function metadata:
    #   Inputs: x (torch.Tensor)
    #   Sample:
    #       gated = head.forward(torch.zeros(1, 512))  # dtype=torch.FloatTensor
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.fc(self.ln(x)))


# ---------------------------------------------------------
# Advantage Network (Regret / Advantage function)
# ---------------------------------------------------------

class AdvantageNet(nn.Module):
    """
    High-capacity residual MLP approximator for regret/advantages.

    Input:  state vector (B, state_dim)
    Output: advantages (B, NUM_ACTIONS), unconstrained real values.
    """
    # Function metadata:
    #   Inputs: state_dim (int), trunk_dim (int), num_blocks (int), dropout (float)
    #   Sample:
    #       adv = AdvantageNet(state_dim=128)
    #       # dtype=AdvantageNet
    def __init__(
        self,
        state_dim: int,
        trunk_dim: int = 512,
        num_blocks: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_ln = nn.LayerNorm(state_dim)
        self.input_proj = nn.Linear(state_dim, trunk_dim)

        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResidualBlock(trunk_dim, hidden_mult=4, dropout=dropout))
        self.blocks = nn.Sequential(*blocks)

        self.head_gate = GatingHead(trunk_dim)
        self.output = nn.Linear(trunk_dim, NUM_ACTIONS)

        # Small init on last layer to keep early-game stable
        nn.init.uniform_(self.output.weight, -1e-3, 1e-3)
        nn.init.zeros_(self.output.bias)

    # Function metadata:
    #   Inputs: x (torch.Tensor) shaped [B, state_dim]
    #   Sample:
    #       logits = adv.forward(torch.zeros(4, 128))  # dtype=torch.FloatTensor
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, state_dim)
        returns: (B, NUM_ACTIONS)
        """
        x = self.input_ln(x)
        h = self.input_proj(x)
        h = self.blocks(h)
        h = self.head_gate(h)
        return self.output(h)


# ---------------------------------------------------------
# Policy Network (Average Strategy)
# ---------------------------------------------------------

class PolicyNet(nn.Module):
    """
    High-capacity residual MLP approximator for the average strategy.

    Input:  state vector (B, state_dim)
    Output: log-probs over actions (B, NUM_ACTIONS)
    """
    # Function metadata:
    #   Inputs: state_dim (int), trunk_dim (int), num_blocks (int), dropout (float)
    #   Sample:
    #       pol = PolicyNet(state_dim=128)
    #       # dtype=PolicyNet
    def __init__(
        self,
        state_dim: int,
        trunk_dim: int = 512,
        num_blocks: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_ln = nn.LayerNorm(state_dim)
        self.input_proj = nn.Linear(state_dim, trunk_dim)

        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResidualBlock(trunk_dim, hidden_mult=4, dropout=dropout))
        self.blocks = nn.Sequential(*blocks)

        self.head_gate = GatingHead(trunk_dim)
        self.output = nn.Linear(trunk_dim, NUM_ACTIONS)

        nn.init.uniform_(self.output.weight, -1e-3, 1e-3)
        nn.init.zeros_(self.output.bias)

    # Function metadata:
    #   Inputs: x (torch.Tensor)
    #   Sample:
    #       logp = pol.forward(torch.zeros(4, 128))  # dtype=torch.FloatTensor
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, state_dim)
        returns: log-probs (B, NUM_ACTIONS)
        """
        # print("ENCODED STATE:", x.cpu().numpy())
        x = self.input_ln(x)
        h = self.input_proj(x)
        h = self.blocks(h)
        h = self.head_gate(h)
        logits = self.output(h)
        return torch.log_softmax(logits, dim=-1)


# ---------------------------------------------------------
# Device helper
# ---------------------------------------------------------

# Function metadata:
#   Inputs: model (nn.Module)
#   Sample:
#       device_model = move_to_device(PolicyNet(128))  # dtype=nn.Module on DEVICE
def move_to_device(model: nn.Module) -> nn.Module:
    return model.to(DEVICE)
