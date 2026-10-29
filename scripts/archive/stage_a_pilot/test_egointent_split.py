import json
from pathlib import Path

from datasets.egointent_adapter import load_egointent_split


ROOT = Path(
    "/media/dhqg/d1/datasets/egointent"
)

SPLIT_CONFIG = Path(
    "config/data/egointent_pilot_split_v0.json"
)


def extract_video_uid(sample):
    prefix = "egointent_"

    if not sample.sample_id.startswith(prefix):
        raise ValueError(
            f"Unexpected sample_id: {sample.sample_id}"
        )

    body = sample.sample_id[len(prefix):]

    return body.rsplit("_", 1)[0]


def main():
    train_samples = load_egointent_split(
        root=ROOT,
        split_config=SPLIT_CONFIG,
        split="train",
        require_videos=True,
    )

    val_samples = load_egointent_split(
        root=ROOT,
        split_config=SPLIT_CONFIG,
        split="val",
        require_videos=True,
    )

    print("train:", len(train_samples))
    print("val:", len(val_samples))
    print("total:", len(train_samples) + len(val_samples))

    assert len(train_samples) == 556
    assert len(val_samples) == 215
    assert len(train_samples) + len(val_samples) == 771

    train_uids = {
        extract_video_uid(sample)
        for sample in train_samples
    }

    val_uids = {
        extract_video_uid(sample)
        for sample in val_samples
    }

    overlap = train_uids & val_uids

    print("train video_uids:", len(train_uids))
    print("val video_uids:", len(val_uids))
    print("uid overlap:", overlap)

    assert len(train_uids) == 6
    assert len(val_uids) == 2
    assert not overlap

    all_samples = train_samples + val_samples

    assert len({
        sample.sample_id
        for sample in all_samples
    }) == 771

    for sample in all_samples:
        assert sample.why.strip()
        assert sample.phase == "unassigned"

        video_path = sample.observation["video_path"]

        assert video_path is not None
        assert Path(video_path).exists()

        # Stage-A predictor sample must not expose future labels.
        assert not hasattr(sample, "observed_next_step")
        assert not hasattr(sample, "plausible_next_steps")

        observation_keys = set(sample.observation.keys())

        assert "observed_next_step" not in observation_keys
        assert "plausible_next_steps" not in observation_keys

    with open(SPLIT_CONFIG, encoding="utf-8") as f:
        config = json.load(f)

    config_train_uids = {
        item["video_uid"]
        for item in config["train"]
    }

    config_val_uids = {
        item["video_uid"]
        for item in config["val"]
    }

    assert train_uids == config_train_uids
    assert val_uids == config_val_uids

    print()
    print("First train sample:")
    print("  id:", train_samples[0].sample_id)
    print("  task:", train_samples[0].task)
    print("  history:", train_samples[0].history)
    print("  WHY:", train_samples[0].why)
    print(
        "  video:",
        train_samples[0].observation["video_path"],
    )

    print()
    print("PASS")


if __name__ == "__main__":
    main()