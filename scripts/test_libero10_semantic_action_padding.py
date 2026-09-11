import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(
    "/home/dhqg/vlia-uet"
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(
        str(PROJECT_ROOT)
    )


from lerobot.datasets.dataset_metadata import (
    LeRobotDatasetMetadata,
)

from lerobot.datasets.factory import (
    resolve_delta_timestamps,
)

from lerobot.datasets.lerobot_dataset import (
    LeRobotDataset,
)

from lerobot.policies.smolvla.configuration_smolvla import (
    SmolVLAConfig,
)

from vlia_data.libero_oracle_dataset import (
    LiberoOracleSubset,
)


def make_base_dataset():
    cfg = SmolVLAConfig()

    meta = LeRobotDatasetMetadata(
        "lerobot/libero"
    )

    delta_timestamps = (
        resolve_delta_timestamps(
            cfg,
            meta,
        )
    )

    return LeRobotDataset(
        "lerobot/libero",
        delta_timestamps=delta_timestamps,
        video_backend="torchcodec",
        return_uint8=True,
    )


def find_subset_index(
    dataset,
    global_index,
):
    matches = (
        dataset.canonical_indices
        == global_index
    ).nonzero(
        as_tuple=False
    ).squeeze(-1)

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one match for "
            f"global {global_index}, "
            f"got {len(matches)}"
        )

    return int(
        matches[0].item()
    )


def main():
    base = make_base_dataset()

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

    print(
        "len:",
        len(oracle),
    )

    print(
        "num_frames:",
        oracle.num_frames,
    )

    print(
        "num_episodes:",
        oracle.num_episodes,
    )

    assert len(oracle) == 88302
    assert oracle.num_frames == 88302
    assert oracle.num_episodes == 335

    print()
    print(
        "=== FRAME 20: CROSS-PHASE CASE ==="
    )

    subset_index = find_subset_index(
        oracle,
        20,
    )

    sample = oracle[
        subset_index
    ]

    baseline_sample = baseline[
        subset_index
    ]

    action = sample[
        "action"
    ]

    effective_pad = sample[
        "action_is_pad"
    ]

    semantic_pad = sample[
        "intention_action_is_pad"
    ]

    print(
        "action shape:",
        tuple(action.shape),
    )

    print(
        "semantic valid:",
        int(
            (~semantic_pad).sum()
        ),
    )

    print(
        "effective valid:",
        int(
            (~effective_pad).sum()
        ),
    )

    print(
        "semantic mask:",
        semantic_pad.tolist(),
    )

    print(
        "effective mask:",
        effective_pad.tolist(),
    )

    # Episode 0 phase 0 is frames 0..39.
    # Starting from frame 20:
    #
    # offsets 0..19 stay in phase 0
    # offsets 20..49 cross phase boundary
    assert int(
        (~semantic_pad).sum()
    ) == 20

    assert (
        semantic_pad[:20]
        == False
    ).all()

    assert (
        semantic_pad[20:]
        == True
    ).all()

    # Frame 20 is far from episode end, so effective
    # padding here must equal semantic padding.
    assert torch.equal(
        effective_pad,
        semantic_pad,
    )

    # Every semantically padded target must repeat
    # the last valid target, matching native LeRobot
    # episode padding behavior.
    last_valid_action = (
        action[19]
    )

    repeated = torch.allclose(
        action[20:],
        last_valid_action
        .unsqueeze(0)
        .expand_as(
            action[20:]
        ),
    )

    print(
        "semantic padded actions "
        "repeat last valid:",
        repeated,
    )

    assert repeated

    # Baseline and Oracle must see exactly the same
    # action targets and masks.
    assert torch.equal(
        sample["action"],
        baseline_sample["action"],
    )

    assert torch.equal(
        sample["action_is_pad"],
        baseline_sample[
            "action_is_pad"
        ],
    )

    assert (
        "intention"
        not in baseline_sample
    )

    assert sample[
        "intention"
    ].shape == (
        960,
    )

    print()
    print(
        "=== FRAME 40: NEW PHASE START ==="
    )

    subset_index_40 = (
        find_subset_index(
            oracle,
            40,
        )
    )

    sample_40 = oracle[
        subset_index_40
    ]

    semantic_pad_40 = sample_40[
        "intention_action_is_pad"
    ]

    # phase 1 = frames 40..93 inclusive
    # so there are at least 50 valid same-phase
    # action steps starting at frame 40.
    print(
        "semantic valid:",
        int(
            (~semantic_pad_40).sum()
        ),
    )

    assert int(
        (~semantic_pad_40).sum()
    ) == 50

    assert not semantic_pad_40.any()

    print()
    print(
        "=== FRAME 93: ONE STEP TO BOUNDARY ==="
    )

    subset_index_93 = (
        find_subset_index(
            oracle,
            93,
        )
    )

    sample_93 = oracle[
        subset_index_93
    ]

    semantic_pad_93 = sample_93[
        "intention_action_is_pad"
    ]

    print(
        "semantic valid:",
        int(
            (~semantic_pad_93).sum()
        ),
    )

    assert int(
        (~semantic_pad_93).sum()
    ) == 1

    assert not bool(
        semantic_pad_93[0]
    )

    assert bool(
        semantic_pad_93[1]
    )

    print()
    print(
        "=== FRAME 213: EPISODE END ==="
    )

    subset_index_213 = (
        find_subset_index(
            oracle,
            213,
        )
    )

    sample_213 = oracle[
        subset_index_213
    ]

    print(
        "effective valid:",
        int(
            (
                ~sample_213[
                    "action_is_pad"
                ]
            ).sum()
        ),
    )

    print(
        "semantic valid:",
        int(
            (
                ~sample_213[
                    "intention_action_is_pad"
                ]
            ).sum()
        ),
    )

    # Native episode end leaves only current action valid.
    assert int(
        (
            ~sample_213[
                "action_is_pad"
            ]
        ).sum()
    ) == 1

    print()
    print(
        "SEMANTIC ACTION PADDING: PASS"
    )


if __name__ == "__main__":
    main()