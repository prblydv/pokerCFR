import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# Helper — move module to device
# ============================================================
def move_to_device(module, device="cpu"):
    return module.to(device)


# ============================================================
# Advantage Network: outputs regret/advantage estimates
# Shape: input = (state_dim)
#        output = (NUM_ACTIONS)
# ============================================================
class AdvantageNet(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 256, num_actions: int = 9):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, num_actions)

        self.reset_parameters()
        # Informational log for debugging / tracing
        try:
            import logging
            logging.getLogger(__name__).info(
                f"Initialized AdvantageNet(state_dim={state_dim}, hidden={hidden}, num_actions={num_actions})"
            )
        except Exception:
            pass

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, state_dim)
        returns: (B, num_actions)
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ============================================================
# Policy Network (supervised learning stage)
# Outputs *log probabilities* for stability
# ============================================================
class PolicyNet(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 256, num_actions: int = 9):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, num_actions)

        self.reset_parameters()
        try:
            import logging
            logging.getLogger(__name__).info(
                f"Initialized PolicyNet(state_dim={state_dim}, hidden={hidden}, num_actions={num_actions})"
            )
        except Exception:
            pass

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, state_dim)
        returns: log-probs, shape (B, num_actions)
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return F.log_softmax(logits, dim=-1)
