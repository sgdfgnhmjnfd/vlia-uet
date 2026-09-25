import torch

from policies.intention.clean_stage_a_encoder import (
    CleanStageAIntentionEncoder,
)
from policies.intention.adapter import (
    IntentionAdapter,
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

    adapter = IntentionAdapter(
        intention_dim=256,
        vlm_dim=960,
    )

    adapter.eval()

    visual = torch.randn(
        4,
        960,
    )

    with torch.no_grad():
        z_int = encoder(
            visual
        )

        intention_token = adapter(
            z_int
        )

    print(
        "visual:",
        tuple(visual.shape),
    )
    print(
        "z_int:",
        tuple(z_int.shape),
    )
    print(
        "intention_token:",
        tuple(
            intention_token.shape
        ),
    )

    assert z_int.shape == (
        4,
        256,
    )

    assert intention_token.shape == (
        4,
        1,
        960,
    )

    assert torch.isfinite(
        intention_token
    ).all()

    print()
    print(
        "PREDICTED INTENTION ADAPTER: PASS"
    )


if __name__ == "__main__":
    main()