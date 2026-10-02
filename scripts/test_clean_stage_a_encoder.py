import torch

from policies.intention.clean_stage_a_encoder import (
    CleanStageAIntentionEncoder,
)


ARTIFACT = (
    "/media/dhqg/d1/vlia_outputs/"
    "stage_a_clean_v1/"
    "intention_encoder_v1.pt"
)


def main():
    encoder = (
        CleanStageAIntentionEncoder
        .from_artifact(
            ARTIFACT
        )
    )

    encoder.eval()

    print(
        "visual dim:",
        encoder.visual_feature_dim,
    )
    print(
        "alignment dim:",
        encoder.alignment_dim,
    )
    print(
        "intention dim:",
        encoder.intention_dim,
    )

    x = torch.randn(
        4,
        960,
    )

    with torch.no_grad():
        z_int = encoder(x)

    print(
        "input:",
        tuple(x.shape),
    )
    print(
        "z_int:",
        tuple(z_int.shape),
    )

    assert z_int.shape == (
        4,
        256,
    )

    assert torch.isfinite(
        z_int
    ).all()

    print()
    print(
        "CLEAN STAGE-A ENCODER: PASS"
    )


if __name__ == "__main__":
    main()