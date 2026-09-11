import torch

from policies.smolvla.configuration_smolvla import VLIASmolVLAConfig
from policies.smolvla.modeling_smolvla import VLIAFlowMatching


def main():
    torch.manual_seed(0)

    cfg = VLIASmolVLAConfig(
        use_intention_token=True,
        intention_dim=960,
        prefix_length=0,
    )

    print("Loading SmolVLA...")
    model = VLIAFlowMatching(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    B = 1

    # embed_prefix expects already-prepared image tensors.
    images = [
        torch.zeros(
            B, 3, 512, 512,
            dtype=torch.float32,
            device=device,
        )
    ]

    img_masks = [
        torch.ones(B, dtype=torch.bool, device=device)
    ]

    # Valid embedding IDs are sufficient for this architectural smoke test.
    lang_tokens = torch.ones(
        B, 8,
        dtype=torch.long,
        device=device,
    )

    lang_masks = torch.ones(
        B, 8,
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

    # Real upstream SmolVLA prefix:
    # [IMAGE][LANGUAGE][STATE]
    with torch.no_grad():
        base_embs, base_pad, base_att = super(
            VLIAFlowMatching, model
        ).embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state=state,
        )

        # Real VLIA prefix:
        # [IMAGE][LANGUAGE][INTENTION][STATE]
        vlia_embs, vlia_pad, vlia_att = model.embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state=state,
            z_int=z_int,
        )

        expected_int = model.intention_adapter(z_int).to(
            device=vlia_embs.device,
            dtype=vlia_embs.dtype,
        )

    print("baseline prefix:", tuple(base_embs.shape))
    print("VLIA prefix:    ", tuple(vlia_embs.shape))

    # B. Exactly one new token.
    assert vlia_embs.shape[1] == base_embs.shape[1] + 1

    # C. State must remain the final prefix token.
    assert torch.allclose(
        vlia_embs[:, -1],
        base_embs[:, -1],
        atol=1e-5,
        rtol=1e-4,
    )

    # Therefore intention must be immediately before state.
    assert torch.allclose(
        vlia_embs[:, -2],
        expected_int[:, 0],
        atol=1e-5,
        rtol=1e-4,
    )

    # D. Intention is a real, non-padding prefix token.
    assert bool(vlia_pad[0, -2].item()) is True

    # Attention segment ID for intention = state segment = 1.
    assert int(vlia_att[0, -2].item()) == 1

    # State should also be segment 1.
    assert int(vlia_att[0, -1].item()) == 1

    # Baseline behavior when z_int is absent.
    with torch.no_grad():
        no_int_embs, no_int_pad, no_int_att = model.embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state=state,
            z_int=None,
        )

    assert no_int_embs.shape == base_embs.shape
    assert torch.equal(no_int_pad, base_pad)
    assert torch.equal(no_int_att, base_att)

    print("intention position: before STATE")
    print("intention pad mask:", bool(vlia_pad[0, -2].item()))
    print("intention att segment:", int(vlia_att[0, -2].item()))
    print("state att segment:", int(vlia_att[0, -1].item()))
    print("BASELINE FALLBACK: PASS")
    print("REAL PREFIX TEST: PASS")


if __name__ == "__main__":
    main()
