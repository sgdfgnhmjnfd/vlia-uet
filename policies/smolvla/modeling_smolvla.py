"""VLIA extension for LeRobot SmolVLA.

V0 prefix:
    [image][language][intention][state]

The semantic intention z_int stays in its own latent space and is projected
to the VLM hidden dimension only at the SmolVLA prefix interface.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from lerobot.policies.smolvla.modeling_smolvla import (
    VLAFlowMatching,
    make_att_2d_masks,
)

from policies.intention.adapter import IntentionAdapter


class VLIAIntentionModule(nn.Module):
    """Standalone adapter module used by the basic smoke test."""

    def __init__(self, intention_dim: int, vlm_dim: int):
        super().__init__()
        self.adapter = IntentionAdapter(
            intention_dim=intention_dim,
            vlm_dim=vlm_dim,
        )

    def forward(self, z_int: torch.Tensor) -> torch.Tensor:
        return self.adapter(z_int)


class VLIAFlowMatching(VLAFlowMatching):
    """SmolVLA flow matching with one optional semantic intention token."""

    def __init__(self, config, rtc_processor=None):
        super().__init__(config, rtc_processor=rtc_processor)

        self.use_intention_token = getattr(
            config,
            "use_intention_token",
            True,
        )
        self.intention_dim = getattr(
            config,
            "intention_dim",
            960,
        )

        # Never hardcode 960.
        self.vlm_dim = (
            self.vlm_with_expert.config.text_config.hidden_size
        )

        self.intention_adapter = IntentionAdapter(
            intention_dim=self.intention_dim,
            vlm_dim=self.vlm_dim,
        )

    def embed_prefix(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state: torch.Tensor = None,
        z_int: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build [IMAGE][LANGUAGE][INTENTION][STATE].

        If z_int is absent or intention is disabled, this returns exactly
        the upstream SmolVLA prefix.
        """

        prefix_embs, prefix_pad_masks, prefix_att_masks = (
            super().embed_prefix(
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                state=state,
            )
        )

        if not self.use_intention_token or z_int is None:
            return (
                prefix_embs,
                prefix_pad_masks,
                prefix_att_masks,
            )

        if state is None:
            raise ValueError(
                "state must be provided when injecting intention"
            )

        if z_int.ndim != 2:
            raise ValueError(
                "Expected z_int [B, d_int], "
                f"got {tuple(z_int.shape)}"
            )

        batch_size = prefix_embs.shape[0]

        if z_int.shape[0] != batch_size:
            raise ValueError(
                f"Batch mismatch: prefix B={batch_size}, "
                f"z_int B={z_int.shape[0]}"
            )

        if z_int.shape[1] != self.intention_dim:
            raise ValueError(
                f"Expected intention dim {self.intention_dim}, "
                f"got {z_int.shape[1]}"
            )

        # VLIA V0 targets the normal variable-length SmolVLA prefix.
        if self.prefix_length is not None and self.prefix_length > 0:
            raise NotImplementedError(
                "VLIA V0 currently expects prefix_length <= 0."
            )

        # Keep adapter computation on its own parameter device/dtype.
        proj_weight = self.intention_adapter.proj.weight
        z_int = z_int.to(
            device=proj_weight.device,
            dtype=proj_weight.dtype,
        )

        intention_emb = self.intention_adapter(z_int)

        # Match the actual prefix tensor dtype/device.
        intention_emb = intention_emb.to(
            device=prefix_embs.device,
            dtype=prefix_embs.dtype,
        )

        # Upstream SmolVLA ends its prefix with the state token.
        state_seq_len = 1 if state.ndim == 2 else state.shape[1]
        insert_at = prefix_embs.shape[1] - state_seq_len

        if insert_at < 0:
            raise RuntimeError(
                f"Invalid intention insertion index: {insert_at}"
            )

        intention_pad_mask = torch.ones(
            batch_size,
            1,
            dtype=prefix_pad_masks.dtype,
            device=prefix_pad_masks.device,
        )

        # Behavioral condition segment, same as state.
        intention_att_mask = torch.ones(
            batch_size,
            1,
            dtype=prefix_att_masks.dtype,
            device=prefix_att_masks.device,
        )

        prefix_embs = torch.cat(
            [
                prefix_embs[:, :insert_at],
                intention_emb,
                prefix_embs[:, insert_at:],
            ],
            dim=1,
        )

        prefix_pad_masks = torch.cat(
            [
                prefix_pad_masks[:, :insert_at],
                intention_pad_mask,
                prefix_pad_masks[:, insert_at:],
            ],
            dim=1,
        )

        prefix_att_masks = torch.cat(
            [
                prefix_att_masks[:, :insert_at],
                intention_att_mask,
                prefix_att_masks[:, insert_at:],
            ],
            dim=1,
        )

        return (
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
        )

    def forward(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        actions,
        noise=None,
        time=None,
        z_int: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Training forward with optional intention conditioning."""

        if noise is None:
            noise = self.sample_noise(
                actions.shape,
                actions.device,
            )

        if time is None:
            time = self.sample_time(
                actions.shape[0],
                actions.device,
            )

        time_expanded = time[:, None, None]

        x_t = (
            time_expanded * noise
            + (1 - time_expanded) * actions
        )

        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = (
            self.embed_prefix(
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                state=state,
                z_int=z_int,
            )
        )

        suffix_embs, suffix_pad_masks, suffix_att_masks = (
            self.embed_suffix(
                x_t,
                time,
            )
        )

        pad_masks = torch.cat(
            [prefix_pad_masks, suffix_pad_masks],
            dim=1,
        )

        att_masks = torch.cat(
            [prefix_att_masks, suffix_att_masks],
            dim=1,
        )

        att_2d_masks = make_att_2d_masks(
            pad_masks,
            att_masks,
        )

        position_ids = torch.cumsum(
            pad_masks,
            dim=1,
        ) - 1

        (_, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[
                prefix_embs,
                suffix_embs,
            ],
            use_cache=False,
            fill_kv_cache=False,
        )

        suffix_out = suffix_out[
            :,
            -self.config.chunk_size :,
        ]

        suffix_out = suffix_out.to(
            dtype=torch.float32
        )

        v_t = self.action_out_proj(
            suffix_out
        )

        losses = F.mse_loss(
            u_t,
            v_t,
            reduction="none",
        )

        return losses

    def sample_actions(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise=None,
        z_int: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Inference with intention included in the cached prefix."""

        bsize = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (
                bsize,
                self.config.chunk_size,
                self.config.max_action_dim,
            )

            noise = self.sample_noise(
                actions_shape,
                device,
            )

        prefix_embs, prefix_pad_masks, prefix_att_masks = (
            self.embed_prefix(
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                state=state,
                z_int=z_int,
            )
        )

        prefix_att_2d_masks = make_att_2d_masks(
            prefix_pad_masks,
            prefix_att_masks,
        )

        prefix_position_ids = torch.cumsum(
            prefix_pad_masks,
            dim=1,
        ) - 1

        # Intention is part of prefix here, therefore it enters the KV cache.
        _, past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[
                prefix_embs,
                None,
            ],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )

        num_steps = self.config.num_steps
        dt = -1.0 / num_steps

        x_t = noise

        for step in range(num_steps):
            time = 1.0 + step * dt

            time_tensor = torch.tensor(
                time,
                dtype=torch.float32,
                device=device,
            ).expand(bsize)

            def denoise_step_partial_call(
                input_x_t,
                current_timestep=time_tensor,
            ):
                return self.denoise_step(
                    x_t=input_x_t,
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    timestep=current_timestep,
                )

            if self._rtc_enabled():
                inference_delay = kwargs.get(
                    "inference_delay"
                )
                prev_chunk_left_over = kwargs.get(
                    "prev_chunk_left_over"
                )
                execution_horizon = kwargs.get(
                    "execution_horizon"
                )

                v_t = self.rtc_processor.denoise_step(
                    x_t=x_t,
                    prev_chunk_left_over=prev_chunk_left_over,
                    inference_delay=inference_delay,
                    time=time,
                    original_denoise_step_partial=(
                        denoise_step_partial_call
                    ),
                    execution_horizon=execution_horizon,
                )
            else:
                v_t = denoise_step_partial_call(
                    x_t
                )

            x_t = x_t + dt * v_t

            if (
                self.rtc_processor is not None
                and self.rtc_processor.is_debug_enabled()
            ):
                self.rtc_processor.track(
                    time=time,
                    x_t=x_t,
                    v_t=v_t,
                )

        return x_t


from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.smolvla.modeling_smolvla import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
    SmolVLAPolicy,
    require_package,
)


INTENTION_KEY = "intention"


class VLIASmolVLAPolicy(SmolVLAPolicy):
    """SmolVLA policy that threads z_int through training and inference."""

    def __init__(self, config, **kwargs):
        """Initialize VLIA without constructing baseline VLAFlowMatching."""

        require_package("transformers", extra="smolvla")

        # Calling SmolVLAPolicy.__init__ would instantiate the baseline
        # VLAFlowMatching first. Initialize its parent directly instead.
        PreTrainedPolicy.__init__(self, config)

        config.validate_features()
        self.config = config

        self.init_rtc_processor()

        self.model = VLIAFlowMatching(
            config,
            rtc_processor=self.rtc_processor,
        )

        self.reset()

    def _get_z_int(self, batch: dict[str, torch.Tensor]) -> torch.Tensor | None:
        """Read optional intention representation from the batch."""
        if not getattr(self.config, "use_intention_token", True):
            return None

        z_int = batch.get(INTENTION_KEY)

        if z_int is None:
            return None

        return z_int

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        noise=None,
        time=None,
        reduction: str = "mean",
    ):
        """Training forward with optional intention conditioning."""

        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(
                batch[OBS_STATE]
            )
            batch[ACTION] = self._pi_aloha_encode_actions_inv(
                batch[ACTION]
            )

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)

        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("action_is_pad")

        z_int = self._get_z_int(batch)

        loss_dict = {}

        losses = self.model.forward(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            actions,
            noise,
            time,
            z_int=z_int,
        )

        original_action_dim = self.config.action_feature.shape[0]
        losses = losses[:, :, :original_action_dim]

        loss_dict["losses_after_forward"] = (
            losses.clone().mean().item()
        )

        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)

            loss_dict["losses_after_in_ep_bound"] = (
                losses.clone().mean().item()
            )

        losses = losses[:, :, : self.config.max_action_dim]

        loss_dict["losses_after_rm_padding"] = (
            losses.clone().mean().item()
        )

        if reduction == "none":
            if actions_is_pad is None:
                per_sample_loss = losses.mean(dim=(1, 2))
            else:
                num_valid = (
                    (~actions_is_pad).sum(dim=1)
                    * losses.shape[-1]
                ).clamp_min(1)

                per_sample_loss = (
                    losses.sum(dim=(1, 2))
                    / num_valid
                )

            loss_dict["loss"] = (
                per_sample_loss.mean().item()
            )

            return per_sample_loss, loss_dict

        if actions_is_pad is None:
            loss = losses.mean()
        else:
            num_valid = (
                (~actions_is_pad).sum()
                * losses.shape[-1]
            ).clamp_min(1)

            loss = losses.sum() / num_valid

        loss_dict["loss"] = loss.item()

        return loss, loss_dict

    def _get_action_chunk(
        self,
        batch: dict[str, torch.Tensor],
        noise: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Inference action chunk with optional intention conditioning."""

        for k in batch:
            if k in self._queues and k != ACTION:
                batch[k] = torch.stack(
                    list(self._queues[k]),
                    dim=1,
                )

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)

        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        z_int = self._get_z_int(batch)

        actions = self.model.sample_actions(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            noise=noise,
            z_int=z_int,
            **kwargs,
        )

        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]

        if self.config.adapt_to_pi_aloha:
            actions = self._pi_aloha_encode_actions(actions)

        return actions
