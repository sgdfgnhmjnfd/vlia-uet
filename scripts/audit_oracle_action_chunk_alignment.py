import json
from collections import Counter, defaultdict
from pathlib import Path


ORACLE_FRAMES = Path(
    "/media/dhqg/d1/datasets/libero_oracle/v0/"
    "libero10_oracle_frames.jsonl"
)

CHUNK_SIZE = 50


def main():
    records_by_episode = defaultdict(list)

    with open(
        ORACLE_FRAMES,
        encoding="utf-8",
    ) as f:
        for line in f:
            x = json.loads(line)

            records_by_episode[
                int(x["episode_index"])
            ].append(x)

    for episode_index in records_by_episode:
        records_by_episode[
            episode_index
        ].sort(
            key=lambda x: int(
                x["frame_index"]
            )
        )

    total = 0
    phase_pure = 0
    crosses_phase = 0

    per_task = defaultdict(
        lambda: Counter(
            total=0,
            pure=0,
            crossing=0,
        )
    )

    distance_to_boundary_hist = Counter()

    crossing_examples = []

    for episode_index, records in (
        records_by_episode.items()
    ):
        n = len(records)

        phase_ids = [
            int(x["phase_id"])
            for x in records
        ]

        for i, record in enumerate(
            records
        ):
            total += 1

            task_index = int(
                record["task_index"]
            )

            per_task[
                task_index
            ]["total"] += 1

            current_phase = (
                phase_ids[i]
            )

            # Native LeRobot action horizon is:
            #
            # t, t+1, ..., t+49
            #
            # Near episode end, LeRobot pads instead of
            # crossing into another episode. Padding does
            # not count as a semantic phase transition.
            valid_end = min(
                i + CHUNK_SIZE,
                n,
            )

            future_phases = (
                phase_ids[
                    i:valid_end
                ]
            )

            pure = all(
                phase == current_phase
                for phase in future_phases
            )

            if pure:
                phase_pure += 1

                per_task[
                    task_index
                ]["pure"] += 1
            else:
                crosses_phase += 1

                per_task[
                    task_index
                ]["crossing"] += 1

                first_change = None

                for offset, phase in enumerate(
                    future_phases
                ):
                    if (
                        phase
                        != current_phase
                    ):
                        first_change = offset
                        break

                if first_change is None:
                    raise RuntimeError(
                        "Crossing sample has no "
                        "detected phase change"
                    )

                distance_to_boundary_hist[
                    first_change
                ] += 1

                if len(
                    crossing_examples
                ) < 30:
                    crossing_examples.append(
                        {
                            "episode_index": (
                                episode_index
                            ),
                            "frame_index": int(
                                record[
                                    "frame_index"
                                ]
                            ),
                            "task_index": (
                                task_index
                            ),
                            "phase_id": (
                                current_phase
                            ),
                            "phase_name": (
                                record[
                                    "phase_name"
                                ]
                            ),
                            "first_phase_change_offset": (
                                first_change
                            ),
                        }
                    )

    assert (
        total
        == phase_pure
        + crosses_phase
    )

    print(
        "chunk size:",
        CHUNK_SIZE,
    )

    print(
        "total canonical frames:",
        total,
    )

    print(
        "phase-pure chunks:",
        phase_pure,
    )

    print(
        "cross-phase chunks:",
        crosses_phase,
    )

    print(
        "phase-pure fraction:",
        phase_pure / total,
    )

    print(
        "cross-phase fraction:",
        crosses_phase / total,
    )

    print()
    print(
        "=== PER TASK ==="
    )

    for task in range(10):
        stats = per_task[
            task
        ]

        task_total = (
            stats["total"]
        )

        pure_fraction = (
            stats["pure"]
            / task_total
        )

        crossing_fraction = (
            stats["crossing"]
            / task_total
        )

        print(
            f"task {task}: "
            f"total={task_total:6d} | "
            f"pure={stats['pure']:6d} "
            f"({pure_fraction:.3f}) | "
            f"cross={stats['crossing']:6d} "
            f"({crossing_fraction:.3f})"
        )

    print()
    print(
        "=== FIRST PHASE-CHANGE OFFSET ==="
    )

    for offset in sorted(
        distance_to_boundary_hist
    ):
        print(
            f"offset {offset:2d}: "
            f"{distance_to_boundary_hist[offset]}"
        )

    print()
    print(
        "=== FIRST 30 CROSSING EXAMPLES ==="
    )

    for x in crossing_examples:
        print(
            f"task={x['task_index']} "
            f"ep={x['episode_index']} "
            f"frame={x['frame_index']} "
            f"phase={x['phase_id']} "
            f"{x['phase_name']} "
            f"-> change after "
            f"{x['first_phase_change_offset']} "
            f"steps"
        )

    print()
    print("PASS")


if __name__ == "__main__":
    main()