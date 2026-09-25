import torch

from lerobot.configs import PreTrainedConfig
from lerobot.policies.smolvla.configuration_smolvla import (
    SmolVLAConfig,
)

from policies.intention.clean_stage_a_encoder import (
    CleanStageAIntentionEncoder,
)
from policies.smolvla.configuration_smolvla import (
    VLIASmolVLAConfig,
)
from policies.smolvla.modeling_smolvla import (
    VLIASmolVLAPolicy,
)


PRETRAINED = "lerobot/smolvla_base"

ARTIFACT = (
    "/media/dhqg/d1/vlia_outputs/"
    "stage_a_clean_v1/"
    "intention_encoder_v1.pt"
)


def clone_vlia_config(base_cfg):
    kwargs = {}

    for field in (
        base_cfg.__dataclass_fields__.values()
    ):
        if field.init:
            kwargs[field.name] = getattr(
                base_cfg,
                field.name,
            )

    return VLIASmolVLAConfig(
        **kwargs,
        use_intention_token=True,
        intention_dim=256,
    )


def main():
    print("=" * 72)
    print("PREDICTED VLIA PREFIX TEST")
    print("=" * 72)

    base_cfg = PreTrainedConfig.from_pretrained(
        pretrained_name_or_path=PRETRAINED,
    )

    if not isinstance(
        base_cfg,
        SmolVLAConfig,
    ):
        raise TypeError(
            f"Expected SmolVLAConfig, got {type(base_cfg)}"
        )

    base_cfg.num_vlm_layers = 16
    base_cfg.resize_imgs_with_padding = (
        512,
        512,
    )
    base_cfg.chunk_size = 50
    base_cfg.n_action_steps = 50

    vlia_cfg = clone_vlia_config(
        base_cfg
    )

    print(
        "VLIA intention_dim:",
        vlia_cfg.intention_dim,
    )

    policy = VLIASmolVLAPolicy.from_pretrained(
        PRETRAINED,
        config=vlia_cfg,
        strict=False,
    )

    policy.eval()

    encoder = (
        CleanStageAIntentionEncoder
        .from_artifact(
            ARTIFACT
        )
    )

    encoder.eval()

    visual_feature = torch.randn(
        1,
        960,
    )

    with torch.no_grad():
        z_int = encoder(
            visual_feature
        )

    print(
        "z_int:",
        tuple(z_int.shape),
    )

    model = policy.model

    print(
        "model intention_dim:",
        model.intention_dim,
    )

    assert model.intention_dim == 256
    assert z_int.shape == (1, 256)

    # Synthetic prefix components.
    #
    # We test the actual VLIA embed_prefix interface,
    # not only the standalone adapter.

    device = next(
        model.parameters()
    ).device

    z_int = z_int.to(
        device=device,
    )

    images = [
        torch.randn(
            1,
            3,
            512,
            512,
            device=device,
        )
    ]

    img_masks = [
        torch.ones(
            1,
            dtype=torch.bool,
            device=device,
        )
    ]

    lang_tokens = torch.zeros(
        1,
        48,
        dtype=torch.long,
        device=device,
    )

    lang_masks = torch.ones(
        1,
        48,
        dtype=torch.bool,
        device=device,
    )

    state = torch.zeros(
        1,
        1,
        vlia_cfg.max_state_dim,
        device=device,
    )

    with torch.no_grad():
        base_prefix = (
            model.embed_prefix(
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                state=state,
                z_int=None,
            )
        )

        predicted_prefix = (
            model.embed_prefix(
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                state=state,
                z_int=z_int,
            )
        )

    base_embs = base_prefix[0]
    pred_embs = predicted_prefix[0]

    print(
        "baseline prefix:",
        tuple(base_embs.shape),
    )
    print(
        "predicted prefix:",
        tuple(pred_embs.shape),
    )

    assert (
        pred_embs.shape[1]
        == base_embs.shape[1] + 1
    )

    assert (
        pred_embs.shape[-1]
        == base_embs.shape[-1]
    )

    print()
    print(
        "PREDICTED VLIA PREFIX: PASS"
    )


if __name__ == "__main__":
    main()