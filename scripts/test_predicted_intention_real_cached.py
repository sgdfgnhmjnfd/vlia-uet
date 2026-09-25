from pathlib import Path

import torch

from policies.intention.clean_stage_a_encoder import (
    CleanStageAIntentionEncoder,
)
from policies.intention.adapter import (
    IntentionAdapter,
)


STAGE_A_ARTIFACT = Path(
    "/media/dhqg/d1/vlia_outputs/"
    "stage_a_clean_v1/"
    "intention_encoder_v1.pt"
)

CACHE_ROOT = Path(
    "/media/dhqg/d1/datasets/"
    "egointent/cache/stage_a_v0"
)


def find_cache_file() -> Path:
    # Prefer the newer cache layout first:
    #   stage_a_v0/train/*.pt
    train_root = (
        CACHE_ROOT
        / "train"
    )

    cache_files = sorted(
        train_root.glob("*.pt")
    )

    if cache_files:
        return cache_files[0]

    # Fallback to the older cache layout:
    #   stage_a_v0/samples/train/*.pt
    legacy_train_root = (
        CACHE_ROOT
        / "samples"
        / "train"
    )

    cache_files = sorted(
        legacy_train_root.glob("*.pt")
    )

    if cache_files:
        return cache_files[0]

    raise FileNotFoundError(
        "No train cache files found under "
        f"{CACHE_ROOT}"
    )


def get_visual_feature(
    record: dict,
) -> torch.Tensor:
    # Newer cache schema.
    if "visual_feature" in record:
        visual_feature = record[
            "visual_feature"
        ]

    # Older cache schema.
    elif "visual" in record:
        visual_feature = record[
            "visual"
        ]

    else:
        raise KeyError(
            "Cache contains neither "
            "'visual_feature' nor 'visual'. "
            f"Available keys: "
            f"{list(record.keys())}"
        )

    if not isinstance(
        visual_feature,
        torch.Tensor,
    ):
        raise TypeError(
            "Expected cached visual feature "
            "to be a torch.Tensor, "
            f"got {type(visual_feature)}"
        )

    # Support both:
    #   [960]
    #   [1, 960]
    if visual_feature.ndim == 1:
        visual_feature = (
            visual_feature.unsqueeze(0)
        )

    if visual_feature.shape != (
        1,
        960,
    ):
        raise ValueError(
            "Expected cached visual feature "
            f"[1, 960], got "
            f"{tuple(visual_feature.shape)}"
        )

    return visual_feature.float()


def main():
    print("=" * 72)
    print(
        "REAL CACHED PREDICTED INTENTION TEST"
    )
    print("=" * 72)

    cache_path = find_cache_file()

    print(
        "cache:",
        cache_path,
    )

    record = torch.load(
        cache_path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(
        record,
        dict,
    ):
        raise TypeError(
            "Expected cache record dict, "
            f"got {type(record)}"
        )

    print(
        "sample_id:",
        record.get("sample_id"),
    )

    print(
        "video_uid:",
        record.get("video_uid"),
    )

    print(
        "task:",
        record.get("task"),
    )

    print(
        "why:",
        record.get("why"),
    )

    print(
        "video_path:",
        record.get("video_path"),
    )

    print(
        "cache keys:",
        list(record.keys()),
    )

    visual_feature = (
        get_visual_feature(
            record
        )
    )

    print(
        "cached visual:",
        tuple(
            visual_feature.shape
        ),
    )

    # ---------------------------------------------------------
    # Load Stage-A Clean V1 predictor.
    # ---------------------------------------------------------

    encoder = (
        CleanStageAIntentionEncoder
        .from_artifact(
            STAGE_A_ARTIFACT
        )
    )

    encoder.eval()

    print(
        "encoder visual dim:",
        encoder.visual_feature_dim,
    )

    print(
        "encoder intention dim:",
        encoder.intention_dim,
    )

    # ---------------------------------------------------------
    # Predicted VLIA adapter.
    #
    # Stage-A Clean V1:
    #   960 -> 256
    #
    # VLIA prefix:
    #   256 -> 960
    # ---------------------------------------------------------

    adapter = IntentionAdapter(
        intention_dim=256,
        vlm_dim=960,
    )

    adapter.eval()

    with torch.no_grad():
        z_int = encoder(
            visual_feature
        )

        intention_token = adapter(
            z_int
        )

    print()
    print(
        "visual feature:",
        tuple(
            visual_feature.shape
        ),
    )

    print(
        "z_int:",
        tuple(
            z_int.shape
        ),
    )

    print(
        "intention token:",
        tuple(
            intention_token.shape
        ),
    )

    print(
        "z_int finite:",
        bool(
            torch.isfinite(
                z_int
            ).all()
        ),
    )

    print(
        "token finite:",
        bool(
            torch.isfinite(
                intention_token
            ).all()
        ),
    )

    # ---------------------------------------------------------
    # Assertions.
    # ---------------------------------------------------------

    assert visual_feature.shape == (
        1,
        960,
    )

    assert z_int.shape == (
        1,
        256,
    )

    assert intention_token.shape == (
        1,
        1,
        960,
    )

    assert torch.isfinite(
        z_int
    ).all()

    assert torch.isfinite(
        intention_token
    ).all()

    print()
    print(
        "REAL CACHED PREDICTED INTENTION: PASS"
    )


if __name__ == "__main__":
    main()