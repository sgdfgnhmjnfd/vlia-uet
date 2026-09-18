

import torch
from torch import nn

from models.temporal_attention_intention_encoder import (
    TemporalAttentionIntentionEncoder,
)


class StructuredTemporalIntentionEncoder(nn.Module):
    """
    Temporal-attention intention encoder with structured WHAT-WHY-NEXT
    supervision.

    WHY remains the primary intention prediction and is exposed as z_int.
    WHAT and NEXT are auxiliary prediction heads.

    NEXT is a supervision target only. It is never used as model input.
    """

    def __init__(
        self,
        visual_dim: int = 960,
        model_dim: int = 256,
        intention_dim: int = 256,
        max_frames: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.visual_dim = visual_dim
        self.model_dim = model_dim
        self.intention_dim = intention_dim
        self.max_frames = max_frames
        self.dropout = dropout

        self.backbone = TemporalAttentionIntentionEncoder(
            visual_dim=visual_dim,
            model_dim=model_dim,
            intention_dim=intention_dim,
            max_frames=max_frames,
            dropout=dropout,
        )

        self.what_head = nn.Sequential(
            nn.Linear(
                intention_dim,
                intention_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(
                intention_dim,
            ),
        )

        self.next_head = nn.Sequential(
            nn.Linear(
                intention_dim,
                intention_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(
                intention_dim,
            ),
        )

    def forward(
        self,
        frame_features: torch.Tensor,
        frame_mask: torch.Tensor | None = None,
    ):
        output = self.backbone(
            frame_features,
            frame_mask=frame_mask,
        )

        if isinstance(
            output,
            dict,
        ):
            z_int = output[
                "z_int"
            ]
        else:
            z_int = output

        what_pred = self.what_head(
            z_int
        )

        next_pred = self.next_head(
            z_int
        )

        return {
            "z_int": z_int,
            "why_pred": z_int,
            "what_pred": what_pred,
            "next_pred": next_pred,
        }
