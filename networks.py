# # # networks.py
# # import torch
# # import torch.nn as nn

# # from config import NUM_ACTIONS, DEVICE


# # class AdvantageNet(nn.Module):
# #     def __init__(self, state_dim: int):
# #         super().__init__()
# #         self.net = nn.Sequential(
# #             nn.Linear(state_dim, 128),
# #             nn.ReLU(),
# #             nn.Linear(128, 128),
# #             nn.ReLU(),
# #             nn.Linear(128, NUM_ACTIONS),
# #         )

# #     def forward(self, x: torch.Tensor) -> torch.Tensor:
# #         return self.net(x)


# # class PolicyNet(nn.Module):
# #     def __init__(self, state_dim: int):
# #         super().__init__()
# #         self.net = nn.Sequential(
# #             nn.Linear(state_dim, 128),
# #             nn.ReLU(),
# #             nn.Linear(128, 128),
# #             nn.ReLU(),
# #             nn.Linear(128, NUM_ACTIONS),
# #         )

# #     def forward(self, x: torch.Tensor) -> torch.Tensor:
# #         logits = self.net(x)
# #         return torch.log_softmax(logits, dim=-1)


# # def move_to_device(model: nn.Module) -> nn.Module:
# #     return model.to(DEVICE)










# # networks.py  — STRONG UPGRADED CFR NETWORKS
# import torch
# import torch.nn as nn

# from config import NUM_ACTIONS, DEVICE


# # ---------------------------------------------------------
# # Utility building blocks
# # ---------------------------------------------------------

# class ResidualBlock(nn.Module):
#     """A small residual block improves stability and accuracy greatly."""
#     def __init__(self, dim):
#         super().__init__()
#         self.fc1 = nn.Linear(dim, dim)
#         self.norm1 = nn.LayerNorm(dim)
#         self.act = nn.GELU()
#         self.fc2 = nn.Linear(dim, dim)
#         self.norm2 = nn.LayerNorm(dim)

#     def forward(self, x):
#         identity = x
#         out = self.fc1(x)
#         out = self.norm1(out)
#         out = self.act(out)
#         out = self.fc2(out)
#         out = self.norm2(out)
#         return self.act(out + identity)


# # ---------------------------------------------------------
# # Advantage Network
# # ---------------------------------------------------------

# class AdvantageNet(nn.Module):
#     """
#     Outputs regret/advantage estimates.
#     Deep + stable + residual.
#     """
#     def __init__(self, state_dim: int):
#         super().__init__()

#         hidden = 256

#         self.input = nn.Sequential(
#             nn.Linear(state_dim, hidden),
#             nn.LayerNorm(hidden),
#             nn.GELU(),
#         )

#         self.blocks = nn.Sequential(
#             ResidualBlock(hidden),
#             ResidualBlock(hidden),
#             ResidualBlock(hidden),
#         )

#         self.output = nn.Linear(hidden, NUM_ACTIONS)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         h = self.input(x)
#         h = self.blocks(h)
#         return self.output(h)


# # ---------------------------------------------------------
# # Policy Network
# # ---------------------------------------------------------

# class PolicyNet(nn.Module):
#     """
#     Outputs log-probabilities for actions.
#     Uses identical architecture to AdvantageNet but separate weights.
#     """
#     def __init__(self, state_dim: int):
#         super().__init__()

#         hidden = 256

#         self.input = nn.Sequential(
#             nn.Linear(state_dim, hidden),
#             nn.LayerNorm(hidden),
#             nn.GELU(),
#         )

#         self.blocks = nn.Sequential(
#             ResidualBlock(hidden),
#             ResidualBlock(hidden),
#             ResidualBlock(hidden),
#         )

#         self.output = nn.Linear(hidden, NUM_ACTIONS)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         h = self.input(x)
#         h = self.blocks(h)
#         logits = self.output(h)
#         return torch.log_softmax(logits, dim=-1)


# # ---------------------------------------------------------
# # Device helper
# # ---------------------------------------------------------

# def move_to_device(model: nn.Module) -> nn.Module:
#     return model.to(DEVICE)

























# networks.py — Transformer + Residual Hybrid Network for Deep CFR
import torch
import torch.nn as nn

from config import NUM_ACTIONS, DEVICE


# ---------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(1, 1, dim))

    def forward(self, x):
        return x + self.pos


# ---------------------------------------------------------
# Transformer Block (Attention + FFN)
# ---------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()

        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)

        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.ln2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Multi-head attention
        attn_out, _ = self.attn(x, x, x)
        x = self.ln1(x + self.dropout(attn_out))

        # Feed-forward
        ff_out = self.ff(x)
        x = self.ln2(x + self.dropout(ff_out))

        return x


# ---------------------------------------------------------
# Residual MLP Block
# ---------------------------------------------------------
class ResidualMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)
        self.act = nn.GELU()

    def forward(self, x):
        identity = x
        out = self.act(self.fc1(x))
        out = self.fc2(out)
        return self.ln(identity + out)


# ---------------------------------------------------------
# Advantage Network (Regret/Advantage function)
# ---------------------------------------------------------
class AdvantageNet(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()

        embed_dim = 128     # Transformer hidden size
        self.embed_dim = embed_dim

        # Project state vector → embedding
        self.input_proj = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        # Attention-based processing
        self.pos_enc = PositionalEncoding(embed_dim)
        self.transformer = nn.Sequential(
            TransformerBlock(embed_dim, heads=4),
            TransformerBlock(embed_dim, heads=4),
        )

        # Additional MLP residual blocks
        self.mlp = nn.Sequential(
            ResidualMLP(embed_dim),
            ResidualMLP(embed_dim),
        )

        # Final output → NUM_ACTIONS
        self.output = nn.Linear(embed_dim, NUM_ACTIONS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert (B, state_dim) → (B, 1, embed_dim)
        h = self.input_proj(x).unsqueeze(1)
        h = self.pos_enc(h)
        h = self.transformer(h)
        h = self.mlp(h)
        h = h.squeeze(1)
        return self.output(h)


# ---------------------------------------------------------
# Policy Network (Average Strategy)
# ---------------------------------------------------------
class PolicyNet(nn.Module):
    """
    Identical architecture to AdvantageNet, but separate parameters.
    Produces log-softmax outputs for policy learning.
    """
    def __init__(self, state_dim: int):
        super().__init__()

        embed_dim = 128

        self.input_proj = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        self.pos_enc = PositionalEncoding(embed_dim)
        self.transformer = nn.Sequential(
            TransformerBlock(embed_dim, heads=4),
            TransformerBlock(embed_dim, heads=4),
        )

        self.mlp = nn.Sequential(
            ResidualMLP(embed_dim),
            ResidualMLP(embed_dim),
        )

        self.output = nn.Linear(embed_dim, NUM_ACTIONS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x).unsqueeze(1)
        h = self.pos_enc(h)
        h = self.transformer(h)
        h = self.mlp(h)
        h = h.squeeze(1)

        logits = self.output(h)
        return torch.log_softmax(logits, dim=-1)


# ---------------------------------------------------------
# Device helper
# ---------------------------------------------------------
def move_to_device(model: nn.Module) -> nn.Module:
    return model.to(DEVICE)
