from __future__ import annotations

import torch
from torch import nn


class TemporalAttentionIntentionEncoder(
    nn.Module
):
    """
    Lightweight temporal attention pooling for intention prediction.

    Input:
        frame_features: [B, T, visual_dim]

    Pipeline:
        frame projection
        + learnable temporal positional embedding
        -> scalar temporal attention
        -> weighted temporal pooling
        -> output projection

    Output:
        z_int: [B, intention_dim]
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

        self.visual_dim = (
            visual_dim
        )

        self.model_dim = (
            model_dim
        )

        self.intention_dim = (
            intention_dim
        )

        self.max_frames = (
            max_frames
        )

        self.input_projection = nn.Sequential(
            nn.Linear(
                visual_dim,
                model_dim,
            ),
            nn.LayerNorm(
                model_dim,
            ),
            nn.GELU(),
        )

        self.position_embedding = (
            nn.Parameter(
                torch.zeros(
                    1,
                    max_frames,
                    model_dim,
                )
            )
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.attention = nn.Sequential(
            nn.Linear(
                model_dim,
                model_dim // 2,
            ),
            nn.GELU(),
            nn.Linear(
                model_dim // 2,
                1,
            ),
        )

        self.output_projection = nn.Sequential(
            nn.Linear(
                model_dim,
                intention_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(
                intention_dim,
            ),
        )

        self._reset_parameters()

    def _reset_parameters(
        self,
    ):
        nn.init.normal_(
            self.position_embedding,
            mean=0.0,
            std=0.02,
        )

    def forward(
        self,
        frame_features: torch.Tensor,
        frame_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ):
        if frame_features.ndim != 3:
            raise ValueError(
                "Expected frame_features "
                "[B,T,D], got "
                f"{tuple(frame_features.shape)}"
            )

        batch_size, num_frames, visual_dim = (
            frame_features.shape
        )

        if visual_dim != self.visual_dim:
            raise ValueError(
                "Unexpected visual dimension: "
                f"{visual_dim}, "
                f"expected {self.visual_dim}."
            )

        if num_frames > self.max_frames:
            raise ValueError(
                "Number of frames exceeds "
                "max_frames: "
                f"{num_frames} > "
                f"{self.max_frames}."
            )

        x = self.input_projection(
            frame_features
        )

        position = (
            self.position_embedding[
                :,
                :num_frames,
            ]
        )

        x = self.dropout(
            x + position
        )

        attention_logits = (
            self.attention(
                x
            )
            .squeeze(-1)
        )

        if frame_mask is not None:
            if frame_mask.shape != (
                batch_size,
                num_frames,
            ):
                raise ValueError(
                    "Unexpected frame_mask shape: "
                    f"{tuple(frame_mask.shape)}"
                )

            attention_logits = (
                attention_logits
                .masked_fill(
                    ~frame_mask.bool(),
                    float("-inf"),
                )
            )

        attention_weights = (
            torch.softmax(
                attention_logits,
                dim=1,
            )
        )

        pooled = torch.sum(
            x
            * attention_weights.unsqueeze(
                -1
            ),
            dim=1,
        )

        z_int = (
            self.output_projection(
                pooled
            )
        )

        if not return_attention:
            return z_int

        return {
            "z_int":
                z_int,

            "attention_weights":
                attention_weights,
        }