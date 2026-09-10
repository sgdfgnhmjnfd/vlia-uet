import json
from pathlib import Path

from datasets.intention_alignment_dataset import IntentionAlignmentSample


def load_egointent_event(label_path):
    label_path = Path(label_path)

    with open(label_path, encoding="utf-8") as f:
        data = json.load(f)

    return data


def find_step_video(event_dir, step_id):
    event_dir = Path(event_dir)

    candidates = [
        event_dir / f"step{step_id}.mp4",
        event_dir / f"step_{step_id}.mp4",
        event_dir / f"{step_id}.mp4",
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


def build_egointent_alignment_samples(label_path):
    label_path = Path(label_path)
    event_dir = label_path.parent

    data = load_egointent_event(label_path)

    video_uid = data["video_uid"]
    scene = data["scene"]
    event = data["event"]

    samples = []

    for step in data["steps"]:
        step_id = step["step_id"]

        video_path = find_step_video(
            event_dir,
            step_id,
        )

        sample = IntentionAlignmentSample(
            sample_id=(
                f"egointent_{video_uid}_{step_id}"
            ),

            # Event is used as the available task-level context.
            task=event,

            # Previous local intents form observable semantic history.
            history=[
                previous["local_intent"]
                for previous in data["steps"]
                if previous["step_id"] < step_id
            ],

            observation={
                "images": [],
                "state": None,
                "video_path": (
                    str(video_path)
                    if video_path is not None
                    else None
                ),
                "start_time": step["start_time"],
                "obs_end_time": step["obs_end_time"],
            },

            # Direct EgoIntent procedural-intent supervision.
            why=step["procedural_intent"],

            # EgoIntent does not provide our V1.1 phase taxonomy.
            phase="unassigned",
        )

        samples.append(sample)

    return samples