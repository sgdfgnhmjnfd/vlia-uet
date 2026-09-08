import torch

from policies.smolvla.configuration_smolvla import (
    VLIASmolVLAConfig,
)
from policies.smolvla.modeling_smolvla import (
    VLIAFlowMatching,
)


def main():
    torch.manual_seed(0)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Only two denoising steps for smoke testing.
    cfg = VLIASmolVLAConfig(
        use_intention_token=True,
        intention_dim=256,
        prefix_length=0,
        num_steps=2,
    )

    print("Loading VLIA SmolVLA...")
    model = VLIAFlowMatching(cfg).to(device)
    model.eval()

    B = 1

    images = [
        torch.zeros(
            B,
            3,
            512,
            512,
            dtype=torch.float32,
            device=device,
        )
    ]

    img_masks = [
        torch.ones(
            B,
            dtype=torch.bool,
            device=device,
        )
    ]

    lang_tokens = torch.ones(
        B,
        8,
        dtype=torch.long,
        device=device,
    )

    lang_masks = torch.ones(
        B,
        8,
        dtype=torch.bool,
        device=device,
    )

    state = torch.randn(
        B,
        cfg.max_state_dim,
        device=device,
    )

    z_int = torch.randn(
        B,
        cfg.intention_dim,
        device=device,
    )

    noise = torch.randn(
        B,
        cfg.chunk_size,
        cfg.max_action_dim,
        device=device,
    )

    with torch.no_grad():
        actions = model.sample_actions(
            images=images,
            img_masks=img_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            state=state,
            noise=noise,
            z_int=z_int,
        )

    print(
        "actions shape:",
        tuple(actions.shape),
    )

    assert actions.shape == (
        B,
        cfg.chunk_size,
        cfg.max_action_dim,
    )

    assert torch.isfinite(actions).all(), (
        "Inference produced non-finite actions"
    )

    print(
        "action mean:",
        actions.mean().item(),
    )

    print(
        "action std:",
        actions.std().item(),
    )

    print("REAL SAMPLE_ACTIONS: PASS")
    print("INTENTION KV-CACHE PATH: PASS")


if __name__ == "__main__":
    main()
