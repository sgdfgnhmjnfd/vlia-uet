from __future__ import annotations

import torch
from torch import nn


class TemporalIntentionEncoder(nn.Module):
    """
    Temporal human-intention prediction module for VLIA.

    The frozen visual backbone is external to this module.

    Pipeline:
        per-frame SmolVLM features
            [B, T, visual_dim]

        -> input projection
            [B, T, model_dim]

        -> temporal positional embedding

        -> Transformer temporal encoder

        -> learnable INTENT query

        -> cross-attention over temporal features

        -> intention projection
            [B, intention_dim]

    Default dimensions:
        visual_dim     = 960
        model_dim      = 256
        intention_dim  = 256

    This module is intended to replace simple temporal
    mean pooling in the Stage-A intention predictor.
    """

    def __init__(
        self,
        visual_dim: int = 960,
        model_dim: int = 256,
        intention_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        max_frames: int = 16,
        dropout: float = 0.1,
        feedforward_dim: int | None = None,
    ):
        super().__init__()

        if visual_dim <= 0:
            raise ValueError(
                "visual_dim must be positive."
            )

        if model_dim <= 0:
            raise ValueError(
                "model_dim must be positive."
            )

        if intention_dim <= 0:
            raise ValueError(
                "intention_dim must be positive."
            )

        if num_layers <= 0:
            raise ValueError(
                "num_layers must be positive."
            )

        if num_heads <= 0:
            raise ValueError(
                "num_heads must be positive."
            )

        if max_frames <= 0:
            raise ValueError(
                "max_frames must be positive."
            )

        if model_dim % num_heads != 0:
            raise ValueError(
                "model_dim must be divisible "
                f"by num_heads, got "
                f"{model_dim} and {num_heads}."
            )

        if feedforward_dim is None:
            feedforward_dim = (
                4 * model_dim
            )

        self.visual_dim = visual_dim
        self.model_dim = model_dim
        self.intention_dim = intention_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_frames = max_frames
        self.dropout = dropout
        self.feedforward_dim = (
            feedforward_dim
        )

        # -----------------------------------------------------
        # 1. Frozen SmolVLM feature -> temporal model space.
        #
        # [B, T, 960]
        #       ->
        # [B, T, 256]
        # -----------------------------------------------------

        self.input_projection = nn.Sequential(
            nn.Linear(
                visual_dim,
                model_dim,
            ),
            nn.LayerNorm(
                model_dim,
            ),
        )

        # -----------------------------------------------------
        # 2. Learnable temporal positional representation.
        #
        # Unlike temporal mean pooling, this explicitly
        # preserves frame order.
        # -----------------------------------------------------

        self.position_embedding = nn.Parameter(
            torch.zeros(
                1,
                max_frames,
                model_dim,
            )
        )

        self.input_dropout = nn.Dropout(
            dropout
        )

        # -----------------------------------------------------
        # 3. Temporal reasoning.
        # -----------------------------------------------------

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=num_heads,
                dim_feedforward=(
                    feedforward_dim
                ),
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
        )

        self.temporal_encoder = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
                norm=nn.LayerNorm(
                    model_dim
                ),
            )
        )

        # -----------------------------------------------------
        # 4. Learnable INTENT query.
        #
        # The query attends over all temporally contextualized
        # frame features and extracts information relevant to
        # the current behavioral intention.
        # -----------------------------------------------------

        self.intent_query = nn.Parameter(
            torch.zeros(
                1,
                1,
                model_dim,
            )
        )

        self.query_attention = (
            nn.MultiheadAttention(
                embed_dim=model_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
        )

        # Residual refinement after query attention.
        self.query_norm = nn.LayerNorm(
            model_dim
        )

        # -----------------------------------------------------
        # 5. Final intention representation.
        #
        # Output:
        #     z_int [B, 256]
        #
        # This representation will later be injected into
        # VLIA through the IntentionAdapter.
        # -----------------------------------------------------

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

    def _reset_parameters(self):
        """
        Initialize learnable temporal/query embeddings.

        Standard Linear / Transformer parameters keep their
        PyTorch initialization.
        """

        nn.init.normal_(
            self.position_embedding,
            mean=0.0,
            std=0.02,
        )

        nn.init.normal_(
            self.intent_query,
            mean=0.0,
            std=0.02,
        )

    def _validate_inputs(
        self,
        frame_features: torch.Tensor,
        frame_mask: torch.Tensor | None,
    ):
        if not isinstance(
            frame_features,
            torch.Tensor,
        ):
            raise TypeError(
                "frame_features must be "
                "a torch.Tensor."
            )

        if frame_features.ndim != 3:
            raise ValueError(
                "Expected frame_features "
                "[B, T, D], got "
                f"{tuple(frame_features.shape)}."
            )

        batch_size, num_frames, dim = (
            frame_features.shape
        )

        if batch_size <= 0:
            raise ValueError(
                "Batch size must be positive."
            )

        if num_frames <= 0:
            raise ValueError(
                "At least one frame is required."
            )

        if dim != self.visual_dim:
            raise ValueError(
                "Unexpected visual feature "
                f"dimension: expected "
                f"{self.visual_dim}, got {dim}."
            )

        if num_frames > self.max_frames:
            raise ValueError(
                f"Received {num_frames} frames, "
                "but max_frames is "
                f"{self.max_frames}."
            )

        if frame_mask is None:
            return

        if not isinstance(
            frame_mask,
            torch.Tensor,
        ):
            raise TypeError(
                "frame_mask must be "
                "a torch.Tensor."
            )

        expected_shape = (
            batch_size,
            num_frames,
        )

        if frame_mask.shape != (
            expected_shape
        ):
            raise ValueError(
                "Expected frame_mask shape "
                f"{expected_shape}, got "
                f"{tuple(frame_mask.shape)}."
            )

        valid_mask = frame_mask.bool()

        # Every sample must contain at least one observed
        # frame. Otherwise attention would operate entirely
        # on padding.
        if not valid_mask.any(
            dim=1
        ).all():
            raise ValueError(
                "Every sample must contain "
                "at least one valid frame."
            )

    def encode_temporal_context(
        self,
        frame_features: torch.Tensor,
        frame_mask: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
    ]:
        """
        Encode ordered frame features into contextualized
        temporal features.

        Args:
            frame_features:
                [B, T, visual_dim]

            frame_mask:
                Optional bool-like mask [B, T].

                True:
                    valid observed frame

                False:
                    padded frame

        Returns:
            temporal_features:
                [B, T, model_dim]

            padding_mask:
                [B, T] or None

                PyTorch attention convention:
                    True = padded / ignored.
        """

        self._validate_inputs(
            frame_features,
            frame_mask,
        )

        _, num_frames, _ = (
            frame_features.shape
        )

        x = self.input_projection(
            frame_features.float()
        )
        # [B, T, model_dim]

        position = (
            self.position_embedding[
                :, :num_frames, :
            ]
        )

        x = x + position

        x = self.input_dropout(
            x
        )

        padding_mask = None

        if frame_mask is not None:
            padding_mask = (
                ~frame_mask.bool()
            )

        x = self.temporal_encoder(
            x,
            src_key_padding_mask=(
                padding_mask
            ),
        )
        # [B, T, model_dim]

        return x, padding_mask

    def forward(
        self,
        frame_features: torch.Tensor,
        frame_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ):
        """
        Predict a semantic intention representation.

        Args:
            frame_features:
                [B, T, visual_dim]

            frame_mask:
                Optional [B, T].

                True  = valid frame
                False = padded frame

            return_attention:
                If True, also return temporal attention
                weights from the INTENT query.

        Returns:
            if return_attention=False:
                z_int:
                    [B, intention_dim]

            if return_attention=True:
                {
                    "z_int":
                        [B, intention_dim],

                    "attention_weights":
                        [B, 1, T],

                    "temporal_features":
                        [B, T, model_dim],
                }
        """

        temporal_features, padding_mask = (
            self.encode_temporal_context(
                frame_features=(
                    frame_features
                ),
                frame_mask=frame_mask,
            )
        )

        batch_size = (
            temporal_features.shape[0]
        )

        # One learnable semantic query per sample.
        query = self.intent_query.expand(
            batch_size,
            -1,
            -1,
        )
        # [B, 1, model_dim]

        pooled, attention_weights = (
            self.query_attention(
                query=query,
                key=temporal_features,
                value=temporal_features,
                key_padding_mask=(
                    padding_mask
                ),
                need_weights=(
                    return_attention
                ),
                average_attn_weights=True,
            )
        )
        # pooled:
        #     [B, 1, model_dim]
        #
        # attention_weights:
        #     [B, 1, T]
        #     when requested.

        # Residual connection preserves information in the
        # learned semantic query while refining it with the
        # observed temporal context.
        pooled = self.query_norm(
            pooled + query
        )

        pooled = pooled[:, 0, :]
        # [B, model_dim]

        z_int = self.output_projection(
            pooled
        )
        # [B, intention_dim]

        if return_attention:
            return {
                "z_int": z_int,
                "attention_weights": (
                    attention_weights
                ),
                "temporal_features": (
                    temporal_features
                ),
            }

        return z_int