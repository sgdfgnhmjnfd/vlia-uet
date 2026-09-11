import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


DEFAULT_ORACLE_ROOT = Path(
    "/media/dhqg/d1/datasets/libero_oracle/v0"
)


class LiberoOracleSubset(Dataset):
    """
    Canonical LIBERO-10 subset for controlled
    Baseline-vs-Oracle VLIA experiments.

    Both conditions use exactly the same canonical
    current-frame indices and exactly the same
    semantic action-boundary padding.

    matched baseline:
        add_intention=False

    Oracle VLIA:
        add_intention=True
        sample["intention"] = z_oracle [960]

    Important:
        Oracle annotations are privileged constructed
        phase-level supervision, not LIBERO
        ground-truth intentions.
    """

    def __init__(
        self,
        base_dataset,
        oracle_root=DEFAULT_ORACLE_ROOT,
        add_intention=True,
        verify_index=True,
        semantic_action_padding=True,
    ):
        super().__init__()

        self.base_dataset = base_dataset

        self.oracle_root = Path(
            oracle_root
        )

        self.add_intention = bool(
            add_intention
        )

        self.verify_index = bool(
            verify_index
        )

        self.semantic_action_padding = bool(
            semantic_action_padding
        )

        embeddings_path = (
            self.oracle_root
            / "oracle_embeddings_960.pt"
        )

        lookup_path = (
            self.oracle_root
            / "frame_to_oracle_id.pt"
        )

        frames_path = (
            self.oracle_root
            / "libero10_oracle_frames.jsonl"
        )

        if not embeddings_path.exists():
            raise FileNotFoundError(
                embeddings_path
            )

        if not lookup_path.exists():
            raise FileNotFoundError(
                lookup_path
            )

        if not frames_path.exists():
            raise FileNotFoundError(
                frames_path
            )

        embedding_record = torch.load(
            embeddings_path,
            map_location="cpu",
            weights_only=False,
        )

        lookup_record = torch.load(
            lookup_path,
            map_location="cpu",
            weights_only=False,
        )

        self.oracle_embeddings = (
            embedding_record[
                "embeddings"
            ]
            .float()
            .contiguous()
        )

        self.frame_to_oracle_id = (
            lookup_record[
                "frame_to_oracle_id"
            ]
            .long()
            .contiguous()
        )

        if (
            self.oracle_embeddings.ndim
            != 2
        ):
            raise ValueError(
                "Oracle embeddings must be "
                "rank-2"
            )

        if (
            self.oracle_embeddings.shape[1]
            != 960
        ):
            raise ValueError(
                "Expected intention dimension "
                f"960, got "
                f"{self.oracle_embeddings.shape[1]}"
            )

        if not torch.isfinite(
            self.oracle_embeddings
        ).all():
            raise ValueError(
                "Oracle embeddings contain "
                "NaN or Inf"
            )

        if (
            len(self.frame_to_oracle_id)
            != len(self.base_dataset)
        ):
            raise ValueError(
                "Lookup/base dataset mismatch: "
                f"lookup={len(self.frame_to_oracle_id)}, "
                f"dataset={len(self.base_dataset)}"
            )

        self.canonical_indices = (
            torch.nonzero(
                self.frame_to_oracle_id
                >= 0,
                as_tuple=False,
            )
            .squeeze(-1)
            .long()
            .contiguous()
        )

        if len(
            self.canonical_indices
        ) == 0:
            raise ValueError(
                "No canonical Oracle frames found"
            )

        oracle_ids = (
            self.frame_to_oracle_id[
                self.canonical_indices
            ]
        )

        if int(
            oracle_ids.min()
        ) < 0:
            raise ValueError(
                "Canonical frame maps to "
                "negative Oracle ID"
            )

        if int(
            oracle_ids.max()
        ) >= len(
            self.oracle_embeddings
        ):
            raise ValueError(
                "Oracle ID exceeds embedding table"
            )

        # --------------------------------------------------
        # Build semantic phase metadata.
        #
        # semantic_valid_steps[g] =
        # number of action steps starting at global frame g
        # that remain inside the same semantic phase.
        #
        # Capped later by actual action chunk length.
        # --------------------------------------------------

        self.semantic_valid_steps = torch.zeros(
            len(self.base_dataset),
            dtype=torch.int16,
        )

        self.phase_id_by_global_index = torch.full(
            (
                len(self.base_dataset),
            ),
            fill_value=-1,
            dtype=torch.int16,
        )

        records_by_episode = {}

        canonical_episode_ids = set()

        with open(
            frames_path,
            encoding="utf-8",
        ) as f:
            for line in f:
                record = json.loads(
                    line
                )

                global_index = int(
                    record["index"]
                )

                episode_index = int(
                    record["episode_index"]
                )

                frame_index = int(
                    record["frame_index"]
                )

                phase_id = int(
                    record["phase_id"]
                )

                canonical_episode_ids.add(
                    episode_index
                )

                self.phase_id_by_global_index[
                    global_index
                ] = phase_id

                if (
                    episode_index
                    not in records_by_episode
                ):
                    records_by_episode[
                        episode_index
                    ] = []

                records_by_episode[
                    episode_index
                ].append(
                    (
                        global_index,
                        frame_index,
                        phase_id,
                    )
                )

        self._num_canonical_episodes = len(
            canonical_episode_ids
        )

        for (
            episode_index,
            records,
        ) in records_by_episode.items():
            records.sort(
                key=lambda x: x[1]
            )

            n = len(records)

            # Sanity:
            # every canonical episode must contain all
            # of its annotated frames in order.
            for i in range(
                1,
                n,
            ):
                prev_frame = records[
                    i - 1
                ][1]

                cur_frame = records[
                    i
                ][1]

                if (
                    cur_frame
                    != prev_frame + 1
                ):
                    raise RuntimeError(
                        "Non-contiguous annotation in "
                        f"episode {episode_index}: "
                        f"{prev_frame} -> {cur_frame}"
                    )

            run_start = 0

            while run_start < n:
                phase_id = records[
                    run_start
                ][2]

                run_end = (
                    run_start + 1
                )

                while (
                    run_end < n
                    and records[
                        run_end
                    ][2]
                    == phase_id
                ):
                    run_end += 1

                # For every current frame inside this
                # semantic phase, count how many future
                # action steps remain in this same phase.
                for pos in range(
                    run_start,
                    run_end,
                ):
                    global_index = records[
                        pos
                    ][0]

                    valid_steps = (
                        run_end - pos
                    )

                    self.semantic_valid_steps[
                        global_index
                    ] = int(
                        valid_steps
                    )

                run_start = run_end

        selected_valid_steps = (
            self.semantic_valid_steps[
                self.canonical_indices
            ]
        )

        if int(
            selected_valid_steps.min()
        ) <= 0:
            raise RuntimeError(
                "Canonical frame has invalid "
                "semantic_valid_steps"
            )

    @property
    def num_frames(self):
        return len(
            self.canonical_indices
        )

    @property
    def num_episodes(self):
        return (
            self._num_canonical_episodes
        )

    def __len__(self):
        return len(
            self.canonical_indices
        )

    @staticmethod
    def _to_int(value):
        if isinstance(
            value,
            torch.Tensor,
        ):
            if value.numel() != 1:
                raise ValueError(
                    "Expected scalar tensor, "
                    f"got {tuple(value.shape)}"
                )

            return int(
                value.item()
            )

        return int(value)

    def get_base_index(
        self,
        subset_index,
    ):
        return int(
            self.canonical_indices[
                subset_index
            ].item()
        )

    def get_oracle_id_from_global_index(
        self,
        global_index,
    ):
        oracle_id = int(
            self.frame_to_oracle_id[
                global_index
            ].item()
        )

        if oracle_id < 0:
            raise ValueError(
                f"Frame {global_index} is not "
                "canonical Oracle V0"
            )

        return oracle_id

    def _apply_semantic_action_padding(
        self,
        sample,
        global_index,
    ):
        """
        Make the action target consistent with the
        current semantic intention.

        Native LeRobot episode padding repeats the last
        valid action and marks later positions in
        action_is_pad=True.

        We use exactly the same behavior at semantic
        phase boundaries.
        """

        if (
            "action"
            not in sample
        ):
            return sample

        actions = sample[
            "action"
        ]

        if not isinstance(
            actions,
            torch.Tensor,
        ):
            raise TypeError(
                "Expected action tensor"
            )

        if actions.ndim != 2:
            raise ValueError(
                "Expected action chunk [T, A], "
                f"got {tuple(actions.shape)}"
            )

        chunk_size = actions.shape[0]

        semantic_valid = int(
            self.semantic_valid_steps[
                global_index
            ].item()
        )

        semantic_valid = min(
            semantic_valid,
            chunk_size,
        )

        if semantic_valid <= 0:
            raise RuntimeError(
                f"Invalid semantic valid length "
                f"{semantic_valid} for frame "
                f"{global_index}"
            )

        semantic_pad = torch.zeros(
            chunk_size,
            dtype=torch.bool,
        )

        if (
            semantic_valid
            < chunk_size
        ):
            semantic_pad[
                semantic_valid:
            ] = True

        native_pad = sample.get(
            "action_is_pad"
        )

        if native_pad is None:
            native_pad = torch.zeros(
                chunk_size,
                dtype=torch.bool,
            )
        else:
            native_pad = (
                native_pad
                .bool()
                .clone()
            )

        if (
            native_pad.shape
            != semantic_pad.shape
        ):
            raise ValueError(
                "action_is_pad/action mismatch: "
                f"{tuple(native_pad.shape)} vs "
                f"{tuple(semantic_pad.shape)}"
            )

        effective_pad = (
            native_pad
            | semantic_pad
        )

        # Number of valid steps before either semantic
        # boundary or episode boundary.
        valid_positions = (
            ~effective_pad
        ).nonzero(
            as_tuple=False
        ).squeeze(-1)

        if (
            valid_positions.numel()
            == 0
        ):
            raise RuntimeError(
                "No valid action remains for "
                f"frame {global_index}"
            )

        last_valid_index = int(
            valid_positions[-1].item()
        )

        # A valid prefix is expected. Native LeRobot
        # padding is suffix-only, and semantic padding
        # is also suffix-only.
        expected_valid = torch.arange(
            last_valid_index + 1
        )

        if not torch.equal(
            valid_positions.cpu(),
            expected_valid,
        ):
            raise RuntimeError(
                "Effective action mask is not "
                "prefix-valid/suffix-padded"
            )

        padded_actions = (
            actions.clone()
        )

        if (
            last_valid_index
            + 1
            < chunk_size
        ):
            last_valid_action = (
                padded_actions[
                    last_valid_index
                ]
                .clone()
            )

            padded_actions[
                last_valid_index + 1:
            ] = last_valid_action

        sample[
            "action"
        ] = padded_actions

        sample[
            "action_is_pad"
        ] = effective_pad

        # Debug/audit-only key.
        # The policy does not need this because
        # action_is_pad already contains the union.
        sample[
            "intention_action_is_pad"
        ] = semantic_pad

        return sample

    def __getitem__(
        self,
        subset_index,
    ):
        base_index = (
            self.get_base_index(
                subset_index
            )
        )

        sample = self.base_dataset[
            base_index
        ]

        if not isinstance(
            sample,
            dict,
        ):
            raise TypeError(
                "Expected base dataset sample "
                "to be a dictionary"
            )

        sample = dict(sample)

        if "index" not in sample:
            raise KeyError(
                "Base sample does not contain "
                "'index'"
            )

        global_index = self._to_int(
            sample["index"]
        )

        if (
            self.verify_index
            and global_index
            != base_index
        ):
            raise RuntimeError(
                "Global index differs from "
                "dataset row index: "
                f"base={base_index}, "
                f"sample={global_index}"
            )

        oracle_id = (
            self.get_oracle_id_from_global_index(
                global_index
            )
        )

        if (
            self.semantic_action_padding
        ):
            sample = (
                self._apply_semantic_action_padding(
                    sample,
                    global_index,
                )
            )

        if self.add_intention:
            sample[
                "intention"
            ] = (
                self.oracle_embeddings[
                    oracle_id
                ]
                .clone()
            )

        return sample

    def oracle_embedding_for_subset_index(
        self,
        subset_index,
    ):
        base_index = (
            self.get_base_index(
                subset_index
            )
        )

        oracle_id = (
            self.get_oracle_id_from_global_index(
                base_index
            )
        )

        return (
            self.oracle_embeddings[
                oracle_id
            ]
            .clone()
        )

    def __getattr__(
        self,
        name: str,
    ) -> Any:
        if name == "base_dataset":
            raise AttributeError(
                name
            )

        base_dataset = (
            object.__getattribute__(
                self,
                "base_dataset",
            )
        )

        return getattr(
            base_dataset,
            name,
        )