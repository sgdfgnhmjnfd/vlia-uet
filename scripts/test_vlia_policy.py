import gc

import torch

from lerobot.configs.types import FeatureType, PolicyFeature

from policies.smolvla.configuration_smolvla import VLIASmolVLAConfig
from policies.smolvla.modeling_smolvla import (
    VLIASmolVLAPolicy,
    VLIAFlowMatching,
    INTENTION_KEY,
)


IMAGE_KEY = "observation.images.camera1"
STATE_KEY = "observation.state"
LANG_TOKEN_KEY = "observation.language.tokens"
LANG_MASK_KEY = "observation.language.attention_mask"
ACTION_KEY = "action"


def build_config():
    return VLIASmolVLAConfig(
        input_features={
            IMAGE_KEY: PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 512, 512),
            ),
            STATE_KEY: PolicyFeature(
                type=FeatureType.STATE,
                shape=(8,),
            ),
        },
        output_features={
            ACTION_KEY: PolicyFeature(
                type=FeatureType.ACTION,
                shape=(6,),
            ),
        },
        use_intention_token=True,
        intention_dim=256,
        prefix_length=0,

        # Smoke test only: make inference cheap.
        num_steps=2,

        # Smaller input is enough to verify the policy path.
        resize_imgs_with_padding=(512, 512),
    )


def build_batch(cfg, device):
    B = 1

    # Policy expects raw image range [0, 1].
    image = torch.rand(
        B, 3, 512, 512,
        dtype=torch.float32,
        device=device,
    )

    state = torch.randn(
        B, 8,
        dtype=torch.float32,
        device=device,
    )

    # Policy already expects tokenized language here.
    lang_tokens = torch.ones(
        B, 8,
        dtype=torch.long,
        device=device,
    )

    lang_mask = torch.ones(
        B, 8,
        dtype=torch.bool,
        device=device,
    )

    # Dataset-level action dim is 6.
    # prepare_action() will pad it to max_action_dim=32.
    actions = torch.randn(
        B,
        cfg.chunk_size,
        6,
        dtype=torch.float32,
        device=device,
    )

    z_int = torch.randn(
        B,
        cfg.intention_dim,
        dtype=torch.float32,
        device=device,
    )

    return {
        IMAGE_KEY: image,
        STATE_KEY: state,
        LANG_TOKEN_KEY: lang_tokens,
        LANG_MASK_KEY: lang_mask,
        ACTION_KEY: actions,
        INTENTION_KEY: z_int,
    }


def test_training(policy, cfg, batch, device):
    print("\n========== POLICY TRAIN ==========")

    assert isinstance(policy.model, VLIAFlowMatching)

    # Freeze the backbone. We specifically want to prove:
    #
    # action loss -> intention prefix -> intention adapter
    policy.requires_grad_(False)
    policy.model.intention_adapter.requires_grad_(True)

    policy.eval()
    policy.zero_grad(set_to_none=True)

    B = 1

    # Forward uses padded action dimension internally.
    noise = torch.randn(
        B,
        cfg.chunk_size,
        cfg.max_action_dim,
        dtype=torch.float32,
        device=device,
    )

    time = torch.full(
        (B,),
        0.5,
        dtype=torch.float32,
        device=device,
    )

    loss, loss_dict = policy.forward(
        batch,
        noise=noise,
        time=time,
    )

    print("loss:", loss.detach().item())
    print("loss_dict:", loss_dict)

    assert loss.ndim == 0
    assert torch.isfinite(loss)

    loss.backward()

    grad = policy.model.intention_adapter.proj.weight.grad

    assert grad is not None, (
        "Policy forward did not backpropagate to IntentionAdapter"
    )

    assert torch.isfinite(grad).all(), (
        "IntentionAdapter gradient contains NaN/Inf"
    )

    grad_norm = grad.norm().detach().item()

    print("adapter grad norm:", grad_norm)

    assert grad_norm > 0.0, (
        "IntentionAdapter gradient is zero"
    )

    print("POLICY TRAIN FORWARD: PASS")
    print("POLICY -> INTENTION GRADIENT: PASS")


def test_inference(policy, cfg, batch, device):
    print("\n========== POLICY INFERENCE ==========")

    policy.eval()

    B = 1

    noise = torch.randn(
        B,
        cfg.chunk_size,
        cfg.max_action_dim,
        dtype=torch.float32,
        device=device,
    )

    # Action is not available at real online inference time.
    inference_batch = {
        k: v
        for k, v in batch.items()
        if k != ACTION_KEY
    }

    with torch.no_grad():
        actions = policy._get_action_chunk(
            inference_batch,
            noise=noise,
        )

    print("actions shape:", tuple(actions.shape))
    print("action mean:", actions.mean().item())
    print("action std:", actions.std().item())

    # Policy must unpad 32 -> original dataset action dim 6.
    assert actions.shape == (
        B,
        cfg.chunk_size,
        6,
    )

    assert torch.isfinite(actions).all()

    print("POLICY INFERENCE: PASS")
    print("POLICY INTENTION KV-CACHE PATH: PASS")


def test_optional_fallback(policy, cfg, batch, device):
    print("\n========== NO-INTENTION FALLBACK ==========")

    policy.eval()

    fallback_batch = {
        k: v
        for k, v in batch.items()
        if k not in (ACTION_KEY, INTENTION_KEY)
    }

    noise = torch.randn(
        1,
        cfg.chunk_size,
        cfg.max_action_dim,
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():
        actions = policy._get_action_chunk(
            fallback_batch,
            noise=noise,
        )

    assert actions.shape == (
        1,
        cfg.chunk_size,
        6,
    )
    assert torch.isfinite(actions).all()

    print("NO-INTENTION BASELINE FALLBACK: PASS")


def main():
    torch.manual_seed(0)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("device:", device)

    cfg = build_config()

    print("image_features:", cfg.image_features)
    print("action_feature:", cfg.action_feature)
    print("intention_dim:", cfg.intention_dim)

    print("\nLoading VLIA policy...")
    policy = VLIASmolVLAPolicy(cfg).to(device)

    print("policy class:", type(policy).__name__)
    print("model class:", type(policy.model).__name__)

    assert isinstance(
        policy.model,
        VLIAFlowMatching,
    )

    batch = build_batch(cfg, device)

    print("batch keys:")
    for key, value in batch.items():
        print(" ", key, tuple(value.shape))

    test_training(
        policy,
        cfg,
        batch,
        device,
    )

    # Remove training graph before inference.
    policy.zero_grad(set_to_none=True)
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    test_inference(
        policy,
        cfg,
        batch,
        device,
    )

    test_optional_fallback(
        policy,
        cfg,
        batch,
        device,
    )

    print("\n======================================")
    print("VLIA POLICY INTEGRATION: ALL PASS")
    print("======================================")


if __name__ == "__main__":
    main()
