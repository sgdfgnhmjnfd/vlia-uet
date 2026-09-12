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
    """
    Build Stage-A intention-alignment samples for one EgoIntent event.

    Predictor inputs:
        - egocentric pre-outcome video
        - event string as weak task-level context
        - previous annotated local intents as semantic-history proxy

    Supervision:
        - procedural_intent -> WHY

    Explicitly excluded from predictor inputs:
        - current local_intent
        - observed_next_step
        - plausible_next_steps

    Note:
        Previous local_intent labels are annotation-derived history.
        They should not be treated as deployment-observable signals
        without a separate prediction mechanism.
    """

    label_path = Path(label_path)
    event_dir = label_path.parent

    data = load_egointent_event(label_path)

    video_uid = data["video_uid"]
    event = data["event"]

    steps = sorted(
        data["steps"],
        key=lambda x: x["step_id"],
    )

    samples = []

    for step in steps:
        step_id = step["step_id"]

        video_path = find_step_video(
            event_dir,
            step_id,
        )

        history = [
            previous["local_intent"]
            for previous in steps
            if previous["step_id"] < step_id
        ]

        sample = IntentionAlignmentSample(
            sample_id=f"egointent_{video_uid}_{step_id}",

            # EgoIntent does not provide a natural-language robot task
            # instruction. The event name is used as weak task context.
            task=event,

            # Annotation-derived semantic-history proxy.
            history=history,

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

            # EgoIntent procedural intent is used as WHY supervision.
            why=step["procedural_intent"],

            # EgoIntent does not provide the VLIA V1.1 phase taxonomy.
            phase="unassigned",
        )

        samples.append(sample)

    return samples


def index_egointent_events(root):
    """
    Index all EgoIntent event label files by video_uid.
    """

    root = Path(root)

    event_index = {}

    for label_path in sorted(root.rglob("step_label.json")):
        data = load_egointent_event(label_path)
        video_uid = data["video_uid"]

        if video_uid in event_index:
            raise ValueError(
                f"Duplicate EgoIntent video_uid: {video_uid}"
            )

        event_index[video_uid] = {
            "label_path": label_path,
            "scene": data["scene"],
            "event": data["event"],
            "num_steps": len(data["steps"]),
        }

    return event_index


def load_egointent_split(
    root,
    split_config,
    split,
    require_videos=True,
):
    """
    Load a fixed EgoIntent split from a JSON split specification.

    Splitting is performed strictly by video_uid. No step-level random
    split is performed here.
    """

    root = Path(root)
    split_config = Path(split_config)

    if split not in {"train", "val", "test"}:
        raise ValueError(
            f"Unsupported split '{split}'. "
            "Expected train, val, or test."
        )

    with open(split_config, encoding="utf-8") as f:
        config = json.load(f)

    if split not in config:
        raise KeyError(
            f"Split '{split}' not found in {split_config}"
        )

    event_index = index_egointent_events(root)

    requested_uids = [
        item["video_uid"]
        for item in config[split]
    ]

    if len(requested_uids) != len(set(requested_uids)):
        raise ValueError(
            f"Duplicate video_uid found in split '{split}'."
        )

    samples = []

    for item in config[split]:
        video_uid = item["video_uid"]

        if video_uid not in event_index:
            raise FileNotFoundError(
                f"video_uid {video_uid} from split '{split}' "
                f"was not found under {root}"
            )

        event_info = event_index[video_uid]

        if event_info["scene"] != item["scene"]:
            raise ValueError(
                f"Scene mismatch for {video_uid}: "
                f"config={item['scene']} "
                f"dataset={event_info['scene']}"
            )

        if event_info["event"] != item["event"]:
            raise ValueError(
                f"Event mismatch for {video_uid}: "
                f"config={item['event']} "
                f"dataset={event_info['event']}"
            )

        if event_info["num_steps"] != item["num_steps"]:
            raise ValueError(
                f"Step-count mismatch for {video_uid}: "
                f"config={item['num_steps']} "
                f"dataset={event_info['num_steps']}"
            )

        event_samples = build_egointent_alignment_samples(
            event_info["label_path"]
        )

        if require_videos:
            missing_videos = [
                sample.sample_id
                for sample in event_samples
                if (
                    sample.observation["video_path"] is None
                    or not Path(
                        sample.observation["video_path"]
                    ).exists()
                )
            ]

            if missing_videos:
                raise FileNotFoundError(
                    f"{len(missing_videos)} missing videos "
                    f"for video_uid {video_uid}. "
                    f"First missing sample: "
                    f"{missing_videos[0]}"
                )

        samples.extend(event_samples)

    return samples