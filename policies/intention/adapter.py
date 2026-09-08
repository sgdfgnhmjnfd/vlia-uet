import torch
from torch import nn


class IntentionAdapter(nn.Module):
    """Project an intention representation into the SmolVLA VLM hidden space."""

    def __init__(self, intention_dim: int, vlm_dim: int):
        super().__init__()
        self.proj = nn.Linear(intention_dim, vlm_dim)

    def forward(self, z_int: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_int: [B, d_int]

        Returns:
            intention_token: [B, 1, D_vlm]
        """
        if z_int.ndim != 2:
            raise ValueError(f"Expected z_int [B, d_int], got {tuple(z_int.shape)}")

        return self.proj(z_int).unsqueeze(1)