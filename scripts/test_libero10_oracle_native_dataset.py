import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path("/home/dhqg/vlia-uet")

# Append project root after site-packages so Hugging Face
# `datasets` is not shadowed by the legacy local datasets/ package.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

from vlia_data.libero_oracle_dataset import LiberoOracleSubset


def shape_of(value):
    if isinstance(value, torch.Tensor):
        return tuple(value.shape)
    return type(value).__name__


def main():
    print("=== SMOLVLA CONFIG ===")

    policy_cfg = SmolVLAConfig()

    print("chunk_size:", policy_cfg.chunk_size)
    print("n_action_steps:", policy_cfg.n_action_steps)
    print(
        "action_delta_indices:",
        policy_cfg.action_delta_indices[:5],
        "...",
        policy_cfg.action_delta_indices[-5:],
    )
    print(
        "observation_delta_indices:",
        policy_cfg.observation_delta_indices,
    )

    has_drop = hasattr(
        policy_cfg,
        "drop_n_last_frames",
    )

    print(
        "has drop_n_last_frames:",
        has_drop,
    )

    if has_drop:
        print(
            "drop_n_last_frames:",
            policy_cfg.drop_n_last_frames,
        )

    assert policy_cfg.chunk_size == 50
    assert policy_cfg.action_delta_indices == list(range(50))

    print()
    print("=== BUILD NATIVE TEMPORAL DATASET ===")

    meta = LeRobotDatasetMetadata(
        "lerobot/libero"
    )

    delta_timestamps = resolve_delta_timestamps(
        policy_cfg,
        meta,
    )

    print("fps:", meta.fps)

    print(
        "action delta count:",
        len(delta_timestamps["action"]),
    )

    print(
        "action delta first/last:",
        delta_timestamps["action"][0],
        delta_timestamps["action"][-1],
    )

    base = LeRobotDataset(
        "lerobot/libero",
        delta_timestamps=delta_timestamps,
        video_backend="torchcodec",
        return_uint8=True,
    )

    print(
        "base len:",
        len(base),
    )

    print(
        "base num_frames:",
        base.num_frames,
    )

    print(
        "base num_episodes:",
        base.num_episodes,
    )

    print()
    print("=== WRAP CANONICAL ORACLE SUBSET ===")

    oracle = LiberoOracleSubset(
        base_dataset=base,
        add_intention=True,
    )

    baseline = LiberoOracleSubset(
        base_dataset=base,
        add_intention=False,
    )

    print(
        "oracle len:",
        len(oracle),
    )

    print(
        "baseline len:",
        len(baseline),
    )

    assert len(oracle) == 88302
    assert len(baseline) == 88302

    print()
    print("=== SINGLE SAMPLE ===")

    sample = oracle[0]

    keys_to_check = [
        "observation.state",
        "observation.images.image",
        "observation.images.image2",
        "action",
        "index",
        "frame_index",
        "episode_index",
        "task_index",
        "intention",
    ]

    for key in keys_to_check:
        if key in sample:
            print(
                f"{key:32s}",
                shape_of(sample[key]),
            )
        else:
            print(
                f"{key:32s}",
                "MISSING",
            )

    assert "action" in sample
    assert isinstance(
        sample["action"],
        torch.Tensor,
    )

    print(
        "action shape:",
        tuple(sample["action"].shape),
    )

    assert sample["action"].shape[0] == 50
    assert sample["action"].shape[-1] == 7

    assert sample["intention"].shape == (
        960,
    )

    print()
    print("=== END-OF-EPISODE SAMPLE ===")

    # Episode 0 has global rows 0..213.
    # Test a canonical sample near its end so we see exactly
    # how LeRobot pads/clamps the 50-step action query.
    subset_positions = (
        oracle.canonical_indices
        == 213
    ).nonzero(
        as_tuple=False
    ).squeeze(-1)

    if len(subset_positions) == 1:
        subset_pos = int(
            subset_positions[0].item()
        )

        end_sample = oracle[
            subset_pos
        ]

        print(
            "subset index:",
            subset_pos,
        )

        print(
            "global index:",
            int(end_sample["index"]),
        )

        print(
            "action shape:",
            tuple(
                end_sample["action"].shape
            ),
        )

        if "action_is_pad" in end_sample:
            print(
                "action_is_pad shape:",
                tuple(
                    end_sample[
                        "action_is_pad"
                    ].shape
                ),
            )

            print(
                "action_is_pad:",
                end_sample[
                    "action_is_pad"
                ].tolist(),
            )
        else:
            print(
                "action_is_pad: not present"
            )

    else:
        print(
            "global frame 213 not present "
            "in canonical subset"
        )

    print()
    print("=== DATALOADER COLLATE ===")

    loader = DataLoader(
        oracle,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    batch = next(iter(loader))

    print(
        "batch action:",
        tuple(
            batch["action"].shape
        ),
    )

    print(
        "batch intention:",
        tuple(
            batch["intention"].shape
        ),
    )

    print(
        "batch index:",
        tuple(
            batch["index"].shape
        ),
    )

    assert batch["action"].shape == (
        4,
        50,
        7,
    )

    assert batch["intention"].shape == (
        4,
        960,
    )

    assert torch.isfinite(
        batch["intention"]
    ).all()

    baseline_loader = DataLoader(
        baseline,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    baseline_batch = next(
        iter(
            baseline_loader
        )
    )

    assert (
        "intention"
        not in baseline_batch
    )

    assert torch.equal(
        batch["index"],
        baseline_batch["index"],
    )

    assert torch.equal(
        batch["action"],
        baseline_batch["action"],
    )

    print()
    print("=== DATASET METADATA CAVEAT ===")

    print(
        "len(wrapper):",
        len(oracle),
    )

    print(
        "wrapper.num_frames:",
        oracle.num_frames,
    )

    print(
        "wrapper.num_episodes:",
        oracle.num_episodes,
    )

    print()
    print(
        "NOTE: if wrapper.num_frames still reports "
        "273465, we will override that in the wrapper "
        "before integrating the native trainer."
    )

    print()
    print("NATIVE TEMPORAL ORACLE DATASET: PASS")


if __name__ == "__main__":
    main()