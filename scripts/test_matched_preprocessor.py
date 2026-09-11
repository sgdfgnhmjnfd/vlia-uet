import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path("/home/dhqg/vlia-uet")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from lerobot.configs import PreTrainedConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.utils.collate import lerobot_collate_fn

from vlia_data.libero_oracle_dataset import LiberoOracleSubset


PRETRAINED = "lerobot/smolvla_base"

RENAME_MAP = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.image2": "observation.images.camera2",
}


def main():
    print("=" * 72)
    print("MATCHED PREPROCESSOR TEST")
    print("=" * 72)

    print()
    print("=== LOAD CONFIG ===")

    cfg = PreTrainedConfig.from_pretrained(
        pretrained_name_or_path=PRETRAINED,
    )

    if not isinstance(cfg, SmolVLAConfig):
        raise TypeError(
            f"Expected SmolVLAConfig, got {type(cfg).__name__}"
        )

    # Explicit final matched-experiment settings.
    cfg.num_vlm_layers = 16
    cfg.resize_imgs_with_padding = (512, 512)
    cfg.chunk_size = 50
    cfg.n_action_steps = 50

    print("device:", cfg.device)
    print("num_vlm_layers:", cfg.num_vlm_layers)
    print("resize:", cfg.resize_imgs_with_padding)
    print("chunk_size:", cfg.chunk_size)
    print("n_action_steps:", cfg.n_action_steps)

    print()
    print("=== BUILD NATIVE DATASET ===")

    meta = LeRobotDatasetMetadata(
        "lerobot/libero"
    )

    delta_timestamps = resolve_delta_timestamps(
        cfg,
        meta,
    )

    base = LeRobotDataset(
        "lerobot/libero",
        delta_timestamps=delta_timestamps,
        video_backend="torchcodec",
        return_uint8=True,
    )

    oracle = LiberoOracleSubset(
        base_dataset=base,
        add_intention=True,
        semantic_action_padding=True,
    )

    baseline = LiberoOracleSubset(
        base_dataset=base,
        add_intention=False,
        semantic_action_padding=True,
    )

    print("base:", len(base))
    print("oracle:", len(oracle))
    print("baseline:", len(baseline))
    print("oracle num_frames:", oracle.num_frames)
    print("oracle num_episodes:", oracle.num_episodes)

    assert len(oracle) == 88302
    assert len(baseline) == 88302

    print()
    print("=== DATALOADER ===")

    collate_fn = (
        lerobot_collate_fn
        if base.meta.has_language_columns
        else None
    )

    oracle_loader = DataLoader(
        oracle,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    baseline_loader = DataLoader(
        baseline,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    oracle_batch = next(
        iter(oracle_loader)
    )

    baseline_batch = next(
        iter(baseline_loader)
    )

    print("raw oracle keys:")
    for key in sorted(oracle_batch):
        value = oracle_batch[key]

        if isinstance(value, torch.Tensor):
            print(
                f"  {key:45s} "
                f"{tuple(value.shape)} "
                f"{value.dtype}"
            )
        else:
            print(
                f"  {key:45s} "
                f"{type(value).__name__}"
            )

    assert oracle_batch["action"].shape == (
        2,
        50,
        7,
    )

    assert oracle_batch["intention"].shape == (
        2,
        960,
    )

    assert oracle_batch[
        "action_is_pad"
    ].shape == (
        2,
        50,
    )

    assert oracle_batch[
        "intention_action_is_pad"
    ].shape == (
        2,
        50,
    )

    print()
    print("=== BUILD PREPROCESSOR ===")

    # Match native lerobot_train behavior:
    # load pretrained processor and override
    # normalization stats/features for this dataset.
    preprocessor_overrides = {
        "device_processor": {
            "device": cfg.device,
        },
        "normalizer_processor": {
            "stats": base.meta.stats,
            "features": {
                **cfg.input_features,
                **cfg.output_features,
            },
            "norm_map": cfg.normalization_mapping,
        },
        "rename_observations_processor": {
            "rename_map": RENAME_MAP,
        },
    }

    preprocessor, _ = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=PRETRAINED,
        dataset_stats=base.meta.stats,
        preprocessor_overrides=preprocessor_overrides,
    )

    print()
    print("=== PREPROCESS ORACLE ===")

    # Native trainer converts uint8 images before preprocessor.
    for cam_key in base.meta.camera_keys:
        if (
            cam_key in oracle_batch
            and oracle_batch[cam_key].dtype
            == torch.uint8
        ):
            oracle_batch[cam_key] = (
                oracle_batch[cam_key]
                .to(torch.float32)
                / 255.0
            )
    # Keep custom VLIA latent outside the native SmolVLA
    # processor. It must not be normalized or tokenized.
    oracle_intention = (
        oracle_batch["intention"]
        .clone()
    )
    oracle_processed = preprocessor(
        oracle_batch
    )
    # Reattach the raw 960-D semantic latent after native
    # preprocessing and move it to the policy device.
    oracle_processed["intention"] = (
        oracle_intention.to(
            device=oracle_processed["action"].device,
            dtype=torch.float32,
        )
    )
    print("processed oracle keys:")
    for key in sorted(
        oracle_processed
    ):
        value = oracle_processed[key]

        if isinstance(value, torch.Tensor):
            print(
                f"  {key:45s} "
                f"{tuple(value.shape)} "
                f"{value.dtype} "
                f"{value.device}"
            )
        else:
            print(
                f"  {key:45s} "
                f"{type(value).__name__}"
            )

    print()
    print("=== REQUIRED ORACLE KEYS ===")

    required = [
        "action",
        "action_is_pad",
        "intention",
        "observation.state",
        "observation.language.tokens",
        "observation.language.attention_mask",
    ]

    for key in required:
        present = (
            key in oracle_processed
        )

        print(
            f"{key:45s}",
            present,
        )

        if not present:
            raise KeyError(
                f"Preprocessor removed required key: {key}"
            )

    assert oracle_processed[
        "intention"
    ].shape == (
        2,
        960,
    )

    assert oracle_processed[
        "action_is_pad"
    ].shape == (
        2,
        50,
    )

    assert oracle_processed[
        "action"
    ].shape[:2] == (
        2,
        50,
    )

    assert torch.isfinite(
        oracle_processed[
            "intention"
        ]
    ).all()

    print()
    print("=== PREPROCESS BASELINE ===")

    for cam_key in base.meta.camera_keys:
        if (
            cam_key in baseline_batch
            and baseline_batch[cam_key].dtype
            == torch.uint8
        ):
            baseline_batch[cam_key] = (
                baseline_batch[cam_key]
                .to(torch.float32)
                / 255.0
            )

    baseline_processed = preprocessor(
        baseline_batch
    )

    print(
        "baseline has intention:",
        "intention"
        in baseline_processed,
    )

    assert (
        "intention"
        not in baseline_processed
    )

    assert torch.equal(
        oracle_processed[
            "action_is_pad"
        ],
        baseline_processed[
            "action_is_pad"
        ],
    )

    assert torch.allclose(
        oracle_processed["action"],
        baseline_processed["action"],
    )

    print()
    print("=" * 72)
    print("MATCHED PREPROCESSOR: PASS")
    print("=" * 72)


if __name__ == "__main__":
    main()