import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Helper — move module to device
# ============================================================
def move_to_device(module, device="cpu"):
    return module.to(device)


# ============================================================
# Residual Block (core building block)
# LayerNorm -> SiLU -> Linear -> LayerNorm -> SiLU -> Linear -> Residual
# ============================================================
class ResBlock(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden)
        self.fc1 = nn.Linear(hidden, hidden)
        self.ln2 = nn.LayerNorm(hidden)
        self.fc2 = nn.Linear(hidden, hidden)

    def forward(self, x):
        h = self.fc1(F.silu(self.ln1(x)))
        h = self.fc2(F.silu(self.ln2(h)))
        return x + h


# ============================================================
# Advantage Network — Deep CFR Regret Network
# Strong architecture:
#   LN → FC → SiLU → 4×ResBlocks → LN → FC
# Output: raw regrets (no softmax)
# ============================================================
class AdvantageNet(nn.Module):
    def __init__(self, state_dim: int = 109, hidden: int = 512, num_actions: int = 9):
        super().__init__()

        # Input tower
        self.ln0 = nn.LayerNorm(state_dim)
        self.fc_in = nn.Linear(state_dim, hidden)

        # Residual core
        self.blocks = nn.ModuleList([ResBlock(hidden) for _ in range(4)])

        # Output tower
        self.ln_out = nn.LayerNorm(hidden)
        self.fc_out = nn.Linear(hidden, num_actions)

        self.reset_parameters()

    # Xavier proper initialization
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, state_dim)
        returns: (B, num_actions) raw regret estimates
        """
        x = self.fc_in(self.ln0(x))
        x = F.silu(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_out(x)
        return self.fc_out(x)  # No activation — regrets can be negative


# ============================================================
# Policy Network — Supervised Strategy Net
# Strong architecture:
#   LN → FC → SiLU → 4×ResBlocks → LN → FC → log_softmax
# Output: log-probabilities for numerical stability
# ============================================================
class PolicyNet(nn.Module):
    def __init__(self, state_dim: int = 109, hidden: int = 512, num_actions: int = 9):
        super().__init__()

        # Input tower
        self.ln0 = nn.LayerNorm(state_dim)
        self.fc_in = nn.Linear(state_dim, hidden)

        # Residual core
        self.blocks = nn.ModuleList([ResBlock(hidden) for _ in range(4)])

        # Output tower
        self.ln_out = nn.LayerNorm(hidden)
        self.fc_out = nn.Linear(hidden, num_actions)

        self.reset_parameters()

    # Xavier init
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, state_dim)
        returns: (B, num_actions) log probabilities
        """
        x = self.fc_in(self.ln0(x))
        x = F.silu(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_out(x)
        logits = self.fc_out(x)
        return F.log_softmax(logits, dim=-1)











# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # ============================================================
# # Helper — move module to device
# # ============================================================
# def move_to_device(module, device="cpu"):
#     return module.to(device)


# # ============================================================
# # Advantage Network: outputs regret/advantage estimates
# # Shape: input = (state_dim)
# #        output = (NUM_ACTIONS)
# # ============================================================
# class AdvantageNet(nn.Module):
#     def __init__(self, state_dim: int, hidden: int = 256, num_actions: int = 9):
#         super().__init__()

#         self.fc1 = nn.Linear(state_dim, hidden)
#         self.fc2 = nn.Linear(hidden, hidden)
#         self.fc3 = nn.Linear(hidden, num_actions)

#         self.reset_parameters()
#         # Informational log for debugging / tracing
#         try:
#             import logging
#             logging.getLogger(__name__).info(
#                 f"Initialized AdvantageNet(state_dim={state_dim}, hidden={hidden}, num_actions={num_actions})"
#             )
#         except Exception:
#             pass

#     def reset_parameters(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 nn.init.zeros_(m.bias)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         x: (B, state_dim)
#         returns: (B, num_actions)
#         """
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         return self.fc3(x)


# # ============================================================
# # Policy Network (supervised learning stage)
# # Outputs *log probabilities* for stability
# # ============================================================
# class PolicyNet(nn.Module):
#     def __init__(self, state_dim: int, hidden: int = 256, num_actions: int = 9):
#         super().__init__()

#         self.fc1 = nn.Linear(state_dim, hidden)
#         self.fc2 = nn.Linear(hidden, hidden)
#         self.fc3 = nn.Linear(hidden, num_actions)

#         self.reset_parameters()
#         try:
#             import logging
#             logging.getLogger(__name__).info(
#                 f"Initialized PolicyNet(state_dim={state_dim}, hidden={hidden}, num_actions={num_actions})"
#             )
#         except Exception:
#             pass

#     def reset_parameters(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 nn.init.zeros_(m.bias)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         x: (B, state_dim)
#         returns: log-probs, shape (B, num_actions)
#         """
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         logits = self.fc3(x)
#         return F.log_softmax(logits, dim=-1)
