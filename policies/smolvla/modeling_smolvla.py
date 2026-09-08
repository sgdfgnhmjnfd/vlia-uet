"""VLIA extension for SmolVLA.

V0 design:
    SmolVLA prefix:
        [image][language][intention][state]

The intention representation z_int is projected into the current
SmolVLA VLM hidden dimension using IntentionAdapter.

Do not assume a fixed prefix length.
"""

from __future__ import annotations

import torch
from torch import nn

from policies.intention.adapter import IntentionAdapter


class VLIAIntentionModule(nn.Module):
    """Convert z_int into one SmolVLA-compatible prefix token."""

    def __init__(self, intention_dim: int, vlm_dim: int):
        super().__init__()
        self.adapter = IntentionAdapter(
            intention_dim=intention_dim,
            vlm_dim=vlm_dim,
        )

    def forward(self, z_int: torch.Tensor) -> torch.Tensor:
        return self.adapter(z_int)