import torch
from torch import nn
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
)


class IntentionVisualEncoder(nn.Module):
    """
    Frozen SmolVLM visual backbone for VLIA Stage A.

    Input:
        List of RGB uint8 frames [H, W, 3].

    Outputs:
        encode_video_features():
            Frozen pooled visual feature [1, 960].

        forward():
            Projected visual representation
            [1, intention_dim].
    """

    def __init__(
        self,
        model_id="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        intention_dim=256,
    ):
        super().__init__()

        self.processor = AutoProcessor.from_pretrained(
            model_id,
        )

        self.vlm = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )

        model = self.vlm.model

        self.vision_model = model.vision_model
        self.connector = model.connector

        # Freeze pretrained visual backbone.
        self.vision_model.eval()
        self.connector.eval()

        for param in self.vision_model.parameters():
            param.requires_grad = False

        for param in self.connector.parameters():
            param.requires_grad = False

        hidden_size = self.vlm.config.text_config.hidden_size

        self.visual_projection = nn.Sequential(
            nn.Linear(hidden_size, intention_dim),
            nn.LayerNorm(intention_dim),
        )

    def train(self, mode=True):
        """
        Keep the frozen visual backbone in eval mode even
        when the surrounding module is switched to train mode.
        """
        super().train(mode)

        self.vision_model.eval()
        self.connector.eval()

        return self

    def preprocess_frames(self, frames):
        """
        Process each original frame independently using the
        official SmolVLM image processor.

        Returns:
            pixel_values:
                [N_tiles_total, C, H, W]

            tiles_per_frame:
                Number of SmolVLM tiles generated for each
                original video frame.
        """

        processed_frames = []
        tiles_per_frame = []

        for frame in frames:
            inputs = self.processor.image_processor(
                images=frame,
                return_tensors="pt",
            )

            pixel_values = inputs["pixel_values"]

            # SmolVLM may produce multiple image tiles for
            # one high-resolution input frame.
            #
            # Collapse processor-specific leading dimensions
            # while preserving [C, H, W].
            pixel_values = pixel_values.reshape(
                -1,
                *pixel_values.shape[-3:],
            )

            processed_frames.append(pixel_values)

            tiles_per_frame.append(
                pixel_values.shape[0]
            )

        pixel_values = torch.cat(
            processed_frames,
            dim=0,
        )

        return pixel_values, tiles_per_frame

    def encode_video_features(self, frames):
        """
        Encode video frames using the frozen SmolVLM visual
        backbone.

        Aggregation:
            visual tokens -> tile
            tiles -> original frame
            frames -> video

        Returns:
            video_feature: [1, 960]
        """

        pixel_values, tiles_per_frame = (
            self.preprocess_frames(frames)
        )

        device = next(
            self.visual_projection.parameters()
        ).device

        pixel_values = pixel_values.to(device)

        with torch.no_grad():
            hidden = self.vision_model(
                pixel_values=pixel_values.to(
                    dtype=self.vision_model.dtype
                ),
                patch_attention_mask=None,
            ).last_hidden_state

            hidden = self.connector(hidden)

        # hidden:
        # [N_tiles_total, N_visual_tokens, 960]
        #
        # Pool visual tokens within each tile.
        tile_features = hidden.mean(dim=1)
        # [N_tiles_total, 960]

        # Recover the original frame structure.
        frame_features = []
        offset = 0

        for num_tiles in tiles_per_frame:
            frame_tiles = tile_features[
                offset : offset + num_tiles
            ]

            # Aggregate all tiles generated from one
            # original video frame.
            frame_feature = frame_tiles.mean(
                dim=0
            )

            frame_features.append(
                frame_feature
            )

            offset += num_tiles

        frame_features = torch.stack(
            frame_features,
            dim=0,
        )
        # [T, 960]

        # V0 temporal aggregation.
        video_feature = frame_features.mean(
            dim=0,
            keepdim=True,
        )
        # [1, 960]

        return video_feature.float()

    def forward(self, frames):
        """
        Backward-compatible projected visual representation.

        Stage-A multimodal fusion should use
        encode_video_features() to obtain the frozen
        960-D feature before projection.
        """

        video_feature = self.encode_video_features(
            frames
        )

        z_visual = self.visual_projection(
            video_feature
        )
        # [1, intention_dim]

        return z_visual


class IntentionTextEncoder(nn.Module):
    """
    Frozen SmolVLM text encoder for VLIA Stage A.

    Text
        -> frozen SmolVLM text model
        -> contextual token hidden states
        -> masked mean pooling
        -> 960-D representation
    """

    def __init__(
        self,
        vlm,
        processor,
    ):
        super().__init__()

        self.text_model = vlm.model.text_model
        self.tokenizer = processor.tokenizer

        self.text_model.eval()

        for param in self.text_model.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        """
        Keep the frozen text backbone in eval mode.
        """
        super().train(mode)

        self.text_model.eval()

        return self

    def forward(self, texts):
        """
        Args:
            texts:
                List of strings.

        Returns:
            pooled:
                [B, 960]
        """

        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        device = next(
            self.text_model.parameters()
        ).device

        input_ids = tokens[
            "input_ids"
        ].to(device)

        attention_mask = tokens[
            "attention_mask"
        ].to(device)

        with torch.no_grad():
            outputs = self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )

        hidden = outputs.last_hidden_state.float()
        # [B, L, 960]

        mask = attention_mask.unsqueeze(
            -1
        ).to(
            hidden.dtype
        )
        # [B, L, 1]

        pooled = (
            (hidden * mask).sum(dim=1)
            / mask.sum(dim=1).clamp(min=1.0)
        )
        # [B, 960]

        return pooled
class IntentionEncoderV0(nn.Module):
    """
    VLIA Stage-A multimodal intention predictor.

    Predictor inputs:
        - egocentric visual observation
        - task-level language
        - semantic history

    Stage-A alignment representation:
        z_align: [1, 960]

    VLIA intention representation:
        z_int: [1, intention_dim]

    WHY/future information is never used as predictor input.
    """

    def __init__(
        self,
        visual_encoder,
        text_encoder,
        intention_dim=256,
    ):
        super().__init__()

        self.visual_encoder = visual_encoder
        self.text_encoder = text_encoder

        self.hidden_size = (
            visual_encoder.vlm.config.text_config.hidden_size
        )

        fusion_input_dim = self.hidden_size * 3

        self.fusion = nn.Sequential(
            nn.Linear(
                fusion_input_dim,
                1024,
            ),
            nn.GELU(),
            nn.Linear(
                1024,
                self.hidden_size,
            ),
            nn.LayerNorm(
                self.hidden_size,
            ),
        )

        self.intention_projection = nn.Sequential(
            nn.Linear(
                self.hidden_size,
                intention_dim,
            ),
            nn.LayerNorm(
                intention_dim,
            ),
        )

    def encode_history(self, history):
        if len(history) == 0:
            device = next(
                self.fusion.parameters()
            ).device

            return torch.zeros(
                1,
                self.hidden_size,
                dtype=torch.float32,
                device=device,
            )

        history_text = " ; ".join(history)

        return self.text_encoder(
            [history_text]
        )

    def encode_alignment_feature(
        self,
        frames,
        task,
        history,
    ):
        visual_feature = (
            self.visual_encoder.encode_video_features(
                frames
            )
        )
        # [1, 960]

        task_feature = self.text_encoder(
            [task]
        )
        # [1, 960]

        history_feature = self.encode_history(
            history
        )
        # [1, 960]

        fused_feature = torch.cat(
            [
                visual_feature,
                task_feature,
                history_feature,
            ],
            dim=-1,
        )
        # [1, 2880]

        z_align = self.fusion(
            fused_feature
        )
        # [1, 960]

        return z_align

    def forward(
        self,
        frames,
        task,
        history,
    ):
        z_align = self.encode_alignment_feature(
            frames=frames,
            task=task,
            history=history,
        )

        z_int = self.intention_projection(
            z_align
        )
        # [1, intention_dim]

        return z_int
class StageAAlignmentModel(nn.Module):
    """
    Stage-A intention alignment.

    Predictor:
        observation + task + history -> z_align

    Frozen target:
        WHY -> frozen SmolVLM text representation

    Loss:
        1 - cosine(z_align, z_why)
    """

    def __init__(
        self,
        intention_encoder,
        text_encoder,
    ):
        super().__init__()

        self.intention_encoder = intention_encoder
        self.text_encoder = text_encoder

    def encode_why(self, why):
        """
        WHY is supervision only.

        No gradients are propagated through the frozen
        SmolVLM target encoder.
        """

        with torch.no_grad():
            z_why = self.text_encoder(
                [why]
            )

        return z_why.detach()

    def forward(
        self,
        frames,
        task,
        history,
        why,
    ):
        z_align = (
            self.intention_encoder
            .encode_alignment_feature(
                frames=frames,
                task=task,
                history=history,
            )
        )

        z_why = self.encode_why(
            why
        )

        cosine_similarity = (
            torch.nn.functional.cosine_similarity(
                z_align,
                z_why,
                dim=-1,
            )
        )

        loss = (
            1.0
            - cosine_similarity.mean()
        )

        return {
            "loss": loss,
            "z_align": z_align,
            "z_why": z_why,
            "cosine_similarity": (
                cosine_similarity
            ),
        }