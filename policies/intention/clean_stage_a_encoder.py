from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class CleanStageAIntentionEncoder(nn.Module):
    """Inference-only Stage-A Clean V1 intention encoder.

    Input:
        visual_feature: [B, 960]

    Output:
        z_int: [B, 256]

    The Stage-A Clean V1 predictor is visual-only.
    Task/history branches are represented by zero vectors to match
    the architecture used during training.

    PCA is NOT part of inference. It was used only to construct the
    256-D supervision target during Stage-A training.
    """

    def __init__(
        self,
        visual_feature_dim: int = 960,
        alignment_dim: int = 960,
        intention_dim: int = 256,
    ):
        super().__init__()

        self.visual_feature_dim = visual_feature_dim
        self.alignment_dim = alignment_dim
        self.intention_dim = intention_dim

        fusion_input_dim = 3 * visual_feature_dim

        self.fusion = nn.Sequential(
            nn.Linear(
                fusion_input_dim,
                1024,
            ),
            nn.GELU(),
            nn.Linear(
                1024,
                alignment_dim,
            ),
            nn.LayerNorm(
                alignment_dim,
            ),
        )

        self.intention_projection = nn.Sequential(
            nn.Linear(
                alignment_dim,
                intention_dim,
            ),
            nn.LayerNorm(
                intention_dim,
            ),
        )

    def forward(
        self,
        visual_feature: torch.Tensor,
    ) -> torch.Tensor:
        if visual_feature.ndim != 2:
            raise ValueError(
                "Expected visual_feature [B, D], "
                f"got {tuple(visual_feature.shape)}"
            )

        if (
            visual_feature.shape[-1]
            != self.visual_feature_dim
        ):
            raise ValueError(
                "Expected visual feature dim "
                f"{self.visual_feature_dim}, "
                f"got {visual_feature.shape[-1]}"
            )

        zeros = torch.zeros_like(
            visual_feature
        )

        fused_input = torch.cat(
            [
                visual_feature,
                zeros,
                zeros,
            ],
            dim=-1,
        )

        z_align = self.fusion(
            fused_input
        )

        z_int = self.intention_projection(
            z_align
        )

        return z_int

    @classmethod
    def from_artifact(
        cls,
        artifact_path: str | Path,
        map_location: str | torch.device = "cpu",
    ) -> "CleanStageAIntentionEncoder":
        artifact_path = Path(
            artifact_path
        )

        obj = torch.load(
            artifact_path,
            map_location=map_location,
            weights_only=False,
        )

        if not isinstance(obj, dict):
            raise TypeError(
                "Expected Stage-A artifact dict, "
                f"got {type(obj)}"
            )

        if "state_dict" not in obj:
            raise KeyError(
                "Artifact is missing state_dict"
            )

        if "architecture" not in obj:
            raise KeyError(
                "Artifact is missing architecture"
            )

        arch = obj["architecture"]

        predictor_input = arch.get(
            "predictor_input"
        )

        if predictor_input != "visual_only":
            raise ValueError(
                "Expected visual_only predictor, "
                f"got {predictor_input}"
            )

        model = cls(
            visual_feature_dim=int(
                arch["visual_feature_dim"]
            ),
            alignment_dim=int(
                arch["alignment_dim"]
            ),
            intention_dim=int(
                arch["intention_dim"]
            ),
        )

        model.load_state_dict(
            obj["state_dict"],
            strict=True,
        )

        return model