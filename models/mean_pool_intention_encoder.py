from __future__ import annotations

import torch
from torch import nn


class MeanPoolIntentionEncoder(
    nn.Module
):
    """
    Diagnostic static baseline.

    Input:
        frame_features: [B, T, visual_dim]

    Pipeline:
        temporal mean
        -> Linear visual_dim -> model_dim
        -> LayerNorm
        -> GELU
        -> Linear model_dim -> intention_dim
        -> LayerNorm

    Output:
        z_int: [B, intention_dim]
    """

    def __init__(
        self,
        visual_dim: int = 960,
        model_dim: int = 256,
        intention_dim: int = 256,
        **kwargs,
    ):
        super().__init__()

        self.visual_dim = visual_dim
        self.model_dim = model_dim
        self.intention_dim = intention_dim

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

        self.output_projection = nn.Sequential(
            nn.Linear(
                model_dim,
                intention_dim,
            ),
            nn.LayerNorm(
                intention_dim,
            ),
        )

    def forward(
        self,
        frame_features: torch.Tensor,
    ):
        if frame_features.ndim != 3:
            raise ValueError(
                "Expected frame_features "
                "[B,T,D], got "
                f"{tuple(frame_features.shape)}"
            )

        if (
            frame_features.shape[-1]
            != self.visual_dim
        ):
            raise ValueError(
                "Unexpected visual dimension: "
                f"{frame_features.shape[-1]}, "
                f"expected {self.visual_dim}."
            )

        pooled = frame_features.mean(
            dim=1
        )

        hidden = (
            self.input_projection(
                pooled
            )
        )

        z_int = (
            self.output_projection(
                hidden
            )
        )

        return z_int