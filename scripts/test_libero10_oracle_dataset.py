import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(
    "/home/dhqg/vlia-uet"
)

# Append rather than prepend.
#
# This is intentional:
# Hugging Face `datasets` from site-packages must resolve
# before the project's legacy `datasets/` package.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(
        str(PROJECT_ROOT)
    )


from lerobot.datasets.lerobot_dataset import (
    LeRobotDataset,
)

from vlia_data.libero_oracle_dataset import (
    LiberoOracleSubset,
)


def scalar_int(x):
    if isinstance(
        x,
        torch.Tensor,
    ):
        return int(
            x.item()
        )

    return int(x)


def main():
    print(
        "Loading base LIBERO dataset..."
    )

    base = LeRobotDataset(
        "lerobot/libero"
    )

    print(
        "base len:",
        len(base),
    )

    oracle = LiberoOracleSubset(
        base_dataset=base,
        add_intention=True,
        semantic_action_padding=False,
    )

    matched_baseline = (
        LiberoOracleSubset(
            base_dataset=base,
            add_intention=False,
            semantic_action_padding=False,
        )
    )

    print(
        "oracle len:",
        len(oracle),
    )

    print(
        "matched baseline len:",
        len(matched_baseline),
    )

    assert len(base) == 273465
    assert len(oracle) == 88302

    assert (
        len(matched_baseline)
        == len(oracle)
    )

    # Test samples spread across the subset.
    test_indices = [
        0,
        1,
        100,
        len(oracle) // 2,
        len(oracle) - 1,
    ]

    print()
    print("=== SAMPLE CHECKS ===")

    for subset_index in (
        test_indices
    ):
        oracle_sample = (
            oracle[
                subset_index
            ]
        )

        baseline_sample = (
            matched_baseline[
                subset_index
            ]
        )

        oracle_global_index = (
            scalar_int(
                oracle_sample[
                    "index"
                ]
            )
        )

        baseline_global_index = (
            scalar_int(
                baseline_sample[
                    "index"
                ]
            )
        )

        oracle_episode = (
            scalar_int(
                oracle_sample[
                    "episode_index"
                ]
            )
        )

        oracle_task = (
            scalar_int(
                oracle_sample[
                    "task_index"
                ]
            )
        )

        z_int = oracle_sample[
            "intention"
        ]

        print(
            f"subset={subset_index:6d} | "
            f"global={oracle_global_index:6d} | "
            f"episode={oracle_episode:4d} | "
            f"task={oracle_task} | "
            f"z={tuple(z_int.shape)}"
        )

        # Exact matched-data condition.
        assert (
            oracle_global_index
            == baseline_global_index
        )

        assert (
            "intention"
            not in baseline_sample
        )

        assert z_int.shape == (
            960,
        )

        assert z_int.dtype == (
            torch.float32
        )

        assert torch.isfinite(
            z_int
        ).all()

        assert 0 <= oracle_task <= 9

    # Verify exact canonical index equality.
    assert torch.equal(
        oracle.canonical_indices,
        matched_baseline.canonical_indices,
    )

    # Verify all selected lookup values valid.
    oracle_ids = (
        oracle.frame_to_oracle_id[
            oracle.canonical_indices
        ]
    )

    print()
    print(
        "canonical indices:",
        tuple(
            oracle.canonical_indices.shape
        ),
    )

    print(
        "oracle id min/max:",
        int(oracle_ids.min()),
        int(oracle_ids.max()),
    )

    print(
        "unique oracle ids:",
        len(
            torch.unique(
                oracle_ids
            )
        ),
    )

    assert len(
        torch.unique(
            oracle_ids
        )
    ) == 38

    print()
    print(
        "MATCHED BASELINE / ORACLE "
        "DATASET: PASS"
    )


if __name__ == "__main__":
    main()