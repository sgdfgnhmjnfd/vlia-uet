import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


DEFAULT_OUTPUT_ROOT = Path(
    "/media/dhqg/d1/datasets/libero_oracle/v0"
)

SOURCE_NAME = "constructed_privileged_oracle_v0"

MIN_RUN = 5


# Canonical number of debounced gripper transitions observed
# in the majority of demonstrations for each LIBERO-10 task.
#
# Important:
# These are NOT ground-truth LIBERO phase labels.
# They define a high-confidence subset for a constructed
# privileged intention oracle.
EXPECTED_TRANSITIONS = {
    0: 4,
    1: 3,
    2: 2,
    3: 4,
    4: 4,
    5: 4,
    6: 4,
    7: 4,
    8: 2,
    9: 1,
}


# Each task with K canonical transitions has K + 1
# semantic segments.
#
# WHY describes the local semantic objective of the
# current phase, rather than the low-level motor action.
PHASE_TEMPLATES = {
    0: [
        {
            "phase_name": "acquire_white_mug",
            "why": (
                "Acquire the white mug so it can be "
                "placed on the left plate."
            ),
        },
        {
            "phase_name": "place_white_mug_left_plate",
            "why": (
                "Move the white mug onto the left plate "
                "to complete the first placement."
            ),
        },
        {
            "phase_name": "acquire_yellow_white_mug",
            "why": (
                "Acquire the yellow and white mug so it "
                "can be placed on the right plate."
            ),
        },
        {
            "phase_name": "place_yellow_white_mug_right_plate",
            "why": (
                "Move the yellow and white mug onto the "
                "right plate to complete the second placement."
            ),
        },
        {
            "phase_name": "finish_both_mug_placements",
            "why": (
                "Leave both mugs in their target positions "
                "to complete the task."
            ),
        },
    ],

    1: [
        {
            "phase_name": "acquire_white_mug",
            "why": (
                "Acquire the white mug so it can be "
                "placed on the plate."
            ),
        },
        {
            "phase_name": "place_white_mug_on_plate",
            "why": (
                "Move the white mug onto the plate to "
                "complete the first placement."
            ),
        },
        {
            "phase_name": "acquire_chocolate_pudding",
            "why": (
                "Acquire the chocolate pudding so it can "
                "be positioned to the right of the plate."
            ),
        },
        {
            "phase_name": "place_pudding_right_of_plate",
            "why": (
                "Move the chocolate pudding to the right "
                "of the plate to complete the task."
            ),
        },
    ],

    2: [
        {
            "phase_name": "acquire_yellow_white_mug",
            "why": (
                "Acquire the yellow and white mug so it "
                "can be put into the microwave."
            ),
        },
        {
            "phase_name": "insert_mug_into_microwave",
            "why": (
                "Move the yellow and white mug into the "
                "microwave before closing it."
            ),
        },
        {
            "phase_name": "close_microwave",
            "why": (
                "Close the microwave after placing the mug "
                "inside to complete the task."
            ),
        },
    ],

    3: [
        {
            "phase_name": "reach_stove_control",
            "why": (
                "Reach the stove control so the stove can "
                "be turned on."
            ),
        },
        {
            "phase_name": "activate_stove",
            "why": (
                "Manipulate the stove control to turn the "
                "stove on for the moka pot."
            ),
        },
        {
            "phase_name": "acquire_moka_pot",
            "why": (
                "Acquire the moka pot so it can be placed "
                "on the activated stove."
            ),
        },
        {
            "phase_name": "place_moka_pot_on_stove",
            "why": (
                "Move the moka pot onto the stove to "
                "complete the required placement."
            ),
        },
        {
            "phase_name": "finish_stove_moka_task",
            "why": (
                "Leave the moka pot on the activated stove "
                "to complete the task."
            ),
        },
    ],

    4: [
        {
            "phase_name": "acquire_alphabet_soup",
            "why": (
                "Acquire the alphabet soup so it can be "
                "placed in the basket."
            ),
        },
        {
            "phase_name": "place_alphabet_soup_in_basket",
            "why": (
                "Move the alphabet soup into the basket "
                "to complete the first placement."
            ),
        },
        {
            "phase_name": "acquire_cream_cheese_box",
            "why": (
                "Acquire the cream cheese box so it can "
                "also be placed in the basket."
            ),
        },
        {
            "phase_name": "place_cream_cheese_box_in_basket",
            "why": (
                "Move the cream cheese box into the basket "
                "to complete the second placement."
            ),
        },
        {
            "phase_name": "finish_basket_task",
            "why": (
                "Leave both requested objects in the basket "
                "to complete the task."
            ),
        },
    ],

    5: [
        {
            "phase_name": "acquire_alphabet_soup",
            "why": (
                "Acquire the alphabet soup so it can be "
                "placed in the basket."
            ),
        },
        {
            "phase_name": "place_alphabet_soup_in_basket",
            "why": (
                "Move the alphabet soup into the basket "
                "to complete the first placement."
            ),
        },
        {
            "phase_name": "acquire_tomato_sauce",
            "why": (
                "Acquire the tomato sauce so it can also "
                "be placed in the basket."
            ),
        },
        {
            "phase_name": "place_tomato_sauce_in_basket",
            "why": (
                "Move the tomato sauce into the basket "
                "to complete the second placement."
            ),
        },
        {
            "phase_name": "finish_basket_task",
            "why": (
                "Leave both requested objects in the basket "
                "to complete the task."
            ),
        },
    ],

    6: [
        {
            "phase_name": "acquire_first_moka_pot",
            "why": (
                "Acquire the first moka pot so it can be "
                "placed on the stove."
            ),
        },
        {
            "phase_name": "place_first_moka_pot_on_stove",
            "why": (
                "Move the first moka pot onto the stove "
                "to complete the first placement."
            ),
        },
        {
            "phase_name": "acquire_second_moka_pot",
            "why": (
                "Acquire the second moka pot so it can "
                "also be placed on the stove."
            ),
        },
        {
            "phase_name": "place_second_moka_pot_on_stove",
            "why": (
                "Move the second moka pot onto the stove "
                "to complete the second placement."
            ),
        },
        {
            "phase_name": "finish_both_moka_placements",
            "why": (
                "Leave both moka pots on the stove to "
                "complete the task."
            ),
        },
    ],

    7: [
        {
            "phase_name": "acquire_cream_cheese_box",
            "why": (
                "Acquire the cream cheese box so it can be "
                "placed in the basket."
            ),
        },
        {
            "phase_name": "place_cream_cheese_box_in_basket",
            "why": (
                "Move the cream cheese box into the basket "
                "to complete the first placement."
            ),
        },
        {
            "phase_name": "acquire_butter",
            "why": (
                "Acquire the butter so it can also be "
                "placed in the basket."
            ),
        },
        {
            "phase_name": "place_butter_in_basket",
            "why": (
                "Move the butter into the basket to "
                "complete the second placement."
            ),
        },
        {
            "phase_name": "finish_basket_task",
            "why": (
                "Leave both requested objects in the basket "
                "to complete the task."
            ),
        },
    ],

    8: [
        {
            "phase_name": "acquire_black_bowl",
            "why": (
                "Acquire the black bowl so it can be put "
                "into the bottom drawer."
            ),
        },
        {
            "phase_name": "place_bowl_in_bottom_drawer",
            "why": (
                "Move the black bowl into the bottom drawer "
                "before closing the drawer."
            ),
        },
        {
            "phase_name": "close_bottom_drawer",
            "why": (
                "Close the bottom drawer after placing the "
                "bowl inside to complete the task."
            ),
        },
    ],

    9: [
        {
            "phase_name": "acquire_book",
            "why": (
                "Acquire the book so it can be placed in "
                "the back compartment of the caddy."
            ),
        },
        {
            "phase_name": "place_book_in_back_compartment",
            "why": (
                "Move the book into the back compartment "
                "of the caddy to complete the task."
            ),
        },
    ],
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo-id",
        type=str,
        default="lerobot/libero",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )

    parser.add_argument(
        "--min-run",
        type=int,
        default=MIN_RUN,
    )

    parser.add_argument(
        "--include-ambiguous",
        action="store_true",
        help=(
            "Also write non-canonical episodes. "
            "Not recommended for Oracle V0 training."
        ),
    )

    return parser.parse_args()


def run_length_encode(values):
    if len(values) == 0:
        return []

    runs = []

    start = 0
    current = values[0]

    for i in range(1, len(values)):
        if values[i] != current:
            runs.append(
                [start, i, int(current)]
            )

            start = i
            current = values[i]

    runs.append(
        [start, len(values), int(current)]
    )

    return runs


def debounce_binary(
    values,
    min_run=5,
):
    """
    Remove short command pulses when they are surrounded
    by the same stable gripper state.

    The operation is deliberately conservative:
    short runs are removed only when both neighboring
    states exist and agree.
    """

    values = values.copy()

    changed = True

    while changed:
        changed = False

        runs = run_length_encode(
            values
        )

        if len(runs) <= 1:
            break

        for idx, (
            start,
            end,
            value,
        ) in enumerate(runs):
            length = end - start

            if length >= min_run:
                continue

            prev_value = (
                runs[idx - 1][2]
                if idx > 0
                else None
            )

            next_value = (
                runs[idx + 1][2]
                if idx + 1 < len(runs)
                else None
            )

            if (
                prev_value is not None
                and next_value is not None
                and prev_value == next_value
            ):
                values[start:end] = (
                    prev_value
                )

                changed = True
                break

    return values


def transition_indices(values):
    if len(values) <= 1:
        return np.asarray(
            [],
            dtype=np.int64,
        )

    return (
        np.where(
            values[1:]
            != values[:-1]
        )[0]
        + 1
    ).astype(
        np.int64
    )


def get_task_texts(ds):
    """
    Build task_index -> natural-language task text from
    LeRobot dataset metadata.
    """

    task_texts = {}

    tasks_table = ds.meta.tasks

    for task_text, row in (
        tasks_table.iterrows()
    ):
        task_index = int(
            row["task_index"]
        )

        task_texts[
            task_index
        ] = str(task_text)

    return task_texts


def get_phase_index(
    local_frame_index,
    transitions,
):
    """
    Segment 0 is before the first transition.

    Every transition advances to the next semantic
    segment.

    Example:
        transitions = [40, 94, 161, 207]

        frames 0..39    -> phase 0
        frames 40..93   -> phase 1
        frames 94..160  -> phase 2
        frames 161..206 -> phase 3
        frames 207..end -> phase 4
    """

    return int(
        np.searchsorted(
            transitions,
            local_frame_index,
            side="right",
        )
    )


def validate_templates():
    for task_index in range(10):
        expected = (
            EXPECTED_TRANSITIONS[
                task_index
            ]
        )

        phases = PHASE_TEMPLATES[
            task_index
        ]

        required = expected + 1

        if len(phases) != required:
            raise ValueError(
                f"Task {task_index}: "
                f"expected {required} phase templates, "
                f"got {len(phases)}"
            )

        for phase_id, phase in enumerate(
            phases
        ):
            if not phase[
                "phase_name"
            ].strip():
                raise ValueError(
                    f"Task {task_index}, "
                    f"phase {phase_id}: "
                    "empty phase_name"
                )

            if not phase[
                "why"
            ].strip():
                raise ValueError(
                    f"Task {task_index}, "
                    f"phase {phase_id}: "
                    "empty WHY"
                )


def main():
    args = parse_args()

    validate_templates()

    # Import here so this script can be executed outside
    # the repository root, avoiding collision between the
    # project's datasets/ package and Hugging Face datasets.
    from lerobot.datasets.lerobot_dataset import (
        LeRobotDataset,
    )

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    frames_path = (
        args.output_root
        / "libero10_oracle_frames.jsonl"
    )

    episodes_path = (
        args.output_root
        / "libero10_oracle_episodes.jsonl"
    )

    summary_path = (
        args.output_root
        / "summary.json"
    )

    print(
        "Loading dataset:",
        args.repo_id,
    )

    ds = LeRobotDataset(
        args.repo_id
    )

    hf = ds.hf_dataset

    task_texts = get_task_texts(
        ds
    )

    # Arrow metadata only.
    episode_indices = hf[
        "episode_index"
    ]

    task_indices = hf[
        "task_index"
    ]

    frame_indices = hf[
        "frame_index"
    ]

    global_indices = hf[
        "index"
    ]

    actions = hf[
        "action"
    ]

    episode_rows = defaultdict(
        list
    )

    episode_task = {}

    print(
        "Indexing LIBERO-10 metadata..."
    )

    for row_idx, (
        episode_index,
        task_index,
    ) in enumerate(
        zip(
            episode_indices,
            task_indices,
        )
    ):
        episode_index = int(
            episode_index
        )

        task_index = int(
            task_index
        )

        if task_index >= 10:
            continue

        if (
            episode_index
            in episode_task
        ):
            if (
                episode_task[
                    episode_index
                ]
                != task_index
            ):
                raise RuntimeError(
                    "Episode contains "
                    "multiple task indices: "
                    f"{episode_index}"
                )
        else:
            episode_task[
                episode_index
            ] = task_index

        episode_rows[
            episode_index
        ].append(
            row_idx
        )

    print(
        "LIBERO-10 episodes:",
        len(episode_rows),
    )

    task_total = Counter()
    task_canonical = Counter()
    task_ambiguous = Counter()

    transition_distribution = (
        defaultdict(Counter)
    )

    total_frames_written = 0
    canonical_frames_written = 0
    ambiguous_frames_written = 0

    episode_records = []

    # Build episode-level metadata first.
    for episode_index in sorted(
        episode_rows
    ):
        rows = episode_rows[
            episode_index
        ]

        task_index = episode_task[
            episode_index
        ]

        task_total[
            task_index
        ] += 1

        gripper_raw = np.asarray(
            [
                float(
                    actions[row][-1]
                )
                for row in rows
            ],
            dtype=np.float32,
        )

        gripper_state = np.where(
            gripper_raw > 0,
            1,
            -1,
        ).astype(
            np.int8
        )

        clean_state = debounce_binary(
            gripper_state,
            min_run=args.min_run,
        )

        transitions = (
            transition_indices(
                clean_state
            )
        )

        num_transitions = int(
            len(transitions)
        )

        transition_distribution[
            task_index
        ][
            num_transitions
        ] += 1

        expected = (
            EXPECTED_TRANSITIONS[
                task_index
            ]
        )

        canonical = (
            num_transitions
            == expected
        )

        if canonical:
            task_canonical[
                task_index
            ] += 1
        else:
            task_ambiguous[
                task_index
            ] += 1

        record = {
            "episode_index": int(
                episode_index
            ),
            "task_index": int(
                task_index
            ),
            "task": task_texts[
                task_index
            ],
            "num_frames": len(
                rows
            ),
            "expected_transitions": int(
                expected
            ),
            "num_transitions": int(
                num_transitions
            ),
            "transitions": (
                transitions.tolist()
            ),
            "canonical": bool(
                canonical
            ),
            "quality": (
                "canonical_gripper_pattern"
                if canonical
                else "ambiguous_gripper_pattern"
            ),
            "source": SOURCE_NAME,
            "boundary_source": (
                "debounced_demo_gripper_command"
            ),
            "min_run": int(
                args.min_run
            ),
        }

        episode_records.append(
            record
        )

    # Write episode audit manifest.
    with open(
        episodes_path,
        "w",
        encoding="utf-8",
    ) as f:
        for record in (
            episode_records
        ):
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    # Map episode metadata for frame generation.
    episode_info = {
        record[
            "episode_index"
        ]: record
        for record in episode_records
    }

    print(
        "Writing frame-level Oracle annotations..."
    )

    with open(
        frames_path,
        "w",
        encoding="utf-8",
    ) as f:
        for episode_index in sorted(
            episode_rows
        ):
            info = episode_info[
                episode_index
            ]

            canonical = info[
                "canonical"
            ]

            if (
                not canonical
                and not args.include_ambiguous
            ):
                continue

            # Ambiguous episodes do not have a trustworthy
            # canonical semantic phase mapping in V0.
            # They are included only when explicitly requested.
            if not canonical:
                continue

            task_index = info[
                "task_index"
            ]

            transitions = np.asarray(
                info[
                    "transitions"
                ],
                dtype=np.int64,
            )

            phases = PHASE_TEMPLATES[
                task_index
            ]

            rows = episode_rows[
                episode_index
            ]

            for local_position, row in enumerate(
                rows
            ):
                phase_id = (
                    get_phase_index(
                        local_position,
                        transitions,
                    )
                )

                if not (
                    0
                    <= phase_id
                    < len(phases)
                ):
                    raise RuntimeError(
                        f"Invalid phase_id "
                        f"{phase_id} for "
                        f"episode {episode_index}"
                    )

                phase = phases[
                    phase_id
                ]

                frame_index = int(
                    frame_indices[row]
                )

                global_index = int(
                    global_indices[row]
                )

                gripper_command = float(
                    actions[row][-1]
                )

                frame_record = {
                    "index": (
                        global_index
                    ),
                    "episode_index": int(
                        episode_index
                    ),
                    "frame_index": (
                        frame_index
                    ),
                    "task_index": int(
                        task_index
                    ),
                    "task": task_texts[
                        task_index
                    ],
                    "phase_id": int(
                        phase_id
                    ),
                    "phase_name": phase[
                        "phase_name"
                    ],
                    "why": phase[
                        "why"
                    ],
                    "gripper_command": (
                        gripper_command
                    ),
                    "quality": (
                        "canonical_gripper_pattern"
                    ),
                    "source": (
                        SOURCE_NAME
                    ),
                    "boundary_source": (
                        "debounced_demo_gripper_command"
                    ),
                    "privileged": True,
                    "uses_future_demo_structure": True,
                }

                f.write(
                    json.dumps(
                        frame_record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                total_frames_written += 1
                canonical_frames_written += 1

    summary = {
        "source": SOURCE_NAME,
        "repo_id": args.repo_id,
        "scope": "LIBERO task_index 0-9",
        "method": (
            "Task-specific semantic phase templates "
            "aligned using debounced demonstration "
            "gripper-command transitions."
        ),
        "scientific_status": (
            "Privileged constructed Oracle supervision; "
            "not LIBERO ground-truth intention annotation."
        ),
        "boundary_source": (
            "debounced_demo_gripper_command"
        ),
        "min_run": int(
            args.min_run
        ),
        "total_libero10_episodes": len(
            episode_rows
        ),
        "canonical_episodes": int(
            sum(
                task_canonical.values()
            )
        ),
        "ambiguous_episodes": int(
            sum(
                task_ambiguous.values()
            )
        ),
        "canonical_episode_fraction": (
            sum(
                task_canonical.values()
            )
            / len(
                episode_rows
            )
        ),
        "frames_written": int(
            total_frames_written
        ),
        "canonical_frames_written": int(
            canonical_frames_written
        ),
        "ambiguous_frames_written": int(
            ambiguous_frames_written
        ),
        "expected_transitions": {
            str(k): int(v)
            for k, v in (
                EXPECTED_TRANSITIONS.items()
            )
        },
        "per_task": {},
    }

    for task_index in range(10):
        total = int(
            task_total[
                task_index
            ]
        )

        canonical = int(
            task_canonical[
                task_index
            ]
        )

        ambiguous = int(
            task_ambiguous[
                task_index
            ]
        )

        summary[
            "per_task"
        ][
            str(task_index)
        ] = {
            "task": task_texts[
                task_index
            ],
            "episodes": total,
            "canonical": canonical,
            "ambiguous": ambiguous,
            "canonical_fraction": (
                canonical / total
                if total > 0
                else 0.0
            ),
            "transition_distribution": {
                str(k): int(v)
                for k, v in sorted(
                    transition_distribution[
                        task_index
                    ].items()
                )
            },
            "phase_templates": (
                PHASE_TEMPLATES[
                    task_index
                ]
            ),
        }

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 72)
    print("LIBERO-10 ORACLE V0 BUILD")
    print("=" * 72)

    print(
        "total episodes:",
        summary[
            "total_libero10_episodes"
        ],
    )

    print(
        "canonical episodes:",
        summary[
            "canonical_episodes"
        ],
    )

    print(
        "ambiguous episodes:",
        summary[
            "ambiguous_episodes"
        ],
    )

    print(
        "canonical fraction:",
        f"{summary['canonical_episode_fraction']:.4f}",
    )

    print(
        "frames written:",
        summary[
            "frames_written"
        ],
    )

    print()
    print("PER TASK")

    for task_index in range(10):
        stats = summary[
            "per_task"
        ][
            str(task_index)
        ]

        print(
            f"task {task_index}: "
            f"{stats['canonical']}/"
            f"{stats['episodes']} canonical "
            f"({stats['canonical_fraction']:.3f})"
        )

    print()
    print(
        "frames:",
        frames_path,
    )

    print(
        "episodes:",
        episodes_path,
    )

    print(
        "summary:",
        summary_path,
    )

    print()
    print(
        "IMPORTANT: these labels are privileged "
        "constructed Oracle supervision, not "
        "LIBERO ground-truth intentions."
    )

    print("PASS")


if __name__ == "__main__":
    main()