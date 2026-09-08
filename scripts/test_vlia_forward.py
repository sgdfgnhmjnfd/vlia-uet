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

    cfg = VLIASmolVLAConfig(
        use_intention_token=True,
        intention_dim=256,
        prefix_length=0,
        num_steps=2,
    )

    print("Loading VLIA SmolVLA...")
    model = VLIAFlowMatching(cfg).to(device)

    # We only need to verify that action loss differentiates through
    # the intention interface. Freeze everything else to reduce memory
    # and make the test semantically precise.
    model.requires_grad_(False)

    model.intention_adapter.requires_grad_(True)

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

    actions = torch.randn(
        B,
        cfg.chunk_size,
        cfg.max_action_dim,
        device=device,
    )

    z_int = torch.randn(
        B,
        cfg.intention_dim,
        device=device,
    )

    noise = torch.randn_like(actions)

    # Avoid random edge cases in sampled timestep.
    time = torch.full(
        (B,),
        0.5,
        dtype=torch.float32,
        device=device,
    )

    model.zero_grad(set_to_none=True)

    losses = model(
        images=images,
        img_masks=img_masks,
        lang_tokens=lang_tokens,
        lang_masks=lang_masks,
        state=state,
        actions=actions,
        noise=noise,
        time=time,
        z_int=z_int,
    )

    print("loss tensor shape:", tuple(losses.shape))

    assert torch.isfinite(losses).all(), (
        "Non-finite training loss"
    )

    loss = losses.mean()

    print(
        "mean loss:",
        loss.detach().item(),
    )

    loss.backward()

    grad = (
        model.intention_adapter
        .proj.weight.grad
    )

    assert grad is not None, (
        "Intention adapter did not receive gradient"
    )

    assert torch.isfinite(grad).all(), (
        "Intention adapter gradient is non-finite"
    )

    grad_norm = grad.norm().detach().item()

    print(
        "adapter grad norm:",
        grad_norm,
    )

    assert grad_norm > 0.0, (
        "Intention adapter gradient is zero"
    )

    print("REAL FORWARD: PASS")
    print("INTENTION BACKWARD: PASS")


if __name__ == "__main__":
    main()
