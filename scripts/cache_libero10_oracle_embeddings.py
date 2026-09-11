import argparse
import json
from pathlib import Path

import torch
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
)


MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

DEFAULT_ORACLE_ROOT = Path(
    "/media/dhqg/d1/datasets/libero_oracle/v0"
)

DEFAULT_NUM_DATASET_FRAMES = 273465


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--oracle-root",
        type=Path,
        default=DEFAULT_ORACLE_ROOT,
    )

    parser.add_argument(
        "--model-id",
        type=str,
        default=MODEL_ID,
    )

    parser.add_argument(
        "--num-dataset-frames",
        type=int,
        default=DEFAULT_NUM_DATASET_FRAMES,
    )

    return parser.parse_args()


@torch.no_grad()
def encode_texts(
    text_model,
    tokenizer,
    texts,
    device,
):
    tokens = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    input_ids = tokens[
        "input_ids"
    ].to(device)

    attention_mask = tokens[
        "attention_mask"
    ].to(device)

    outputs = text_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True,
    )

    hidden = (
        outputs
        .last_hidden_state
        .float()
    )

    mask = (
        attention_mask
        .unsqueeze(-1)
        .to(hidden.dtype)
    )

    pooled = (
        (hidden * mask).sum(dim=1)
        /
        mask.sum(dim=1).clamp(
            min=1.0
        )
    )

    return pooled


def main():
    args = parse_args()

    frames_path = (
        args.oracle_root
        / "libero10_oracle_frames.jsonl"
    )

    embeddings_path = (
        args.oracle_root
        / "oracle_embeddings_960.pt"
    )

    lookup_path = (
        args.oracle_root
        / "frame_to_oracle_id.pt"
    )

    catalog_path = (
        args.oracle_root
        / "oracle_catalog.json"
    )

    if not frames_path.exists():
        raise FileNotFoundError(
            frames_path
        )

    # --------------------------------------------------
    # Read Oracle annotations and build deterministic
    # WHY catalog.
    # --------------------------------------------------

    frame_records = []
    unique_why = set()

    with open(
        frames_path,
        encoding="utf-8",
    ) as f:
        for line in f:
            record = json.loads(
                line
            )

            frame_records.append(
                record
            )

            unique_why.add(
                record["why"]
            )

    why_texts = sorted(
        unique_why
    )

    why_to_id = {
        why: idx
        for idx, why
        in enumerate(why_texts)
    }

    print(
        "annotated frames:",
        len(frame_records),
    )

    print(
        "unique WHY:",
        len(why_texts),
    )

    assert len(
        frame_records
    ) == 88302

    assert len(
        why_texts
    ) == 38

    # --------------------------------------------------
    # Load frozen SmolVLM text encoder.
    # --------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "device:",
        device,
    )

    print(
        "model:",
        args.model_id,
    )

    processor = (
        AutoProcessor
        .from_pretrained(
            args.model_id
        )
    )

    vlm = (
        AutoModelForImageTextToText
        .from_pretrained(
            args.model_id,
            torch_dtype=(
                torch.bfloat16
                if device.type == "cuda"
                else torch.float32
            ),
        )
        .to(device)
    )

    text_model = (
        vlm.model.text_model
    )

    text_model.eval()

    for param in (
        text_model.parameters()
    ):
        param.requires_grad = False

    print(
        "Encoding 38 unique WHY texts..."
    )

    embeddings = encode_texts(
        text_model=text_model,
        tokenizer=processor.tokenizer,
        texts=why_texts,
        device=device,
    )

    embeddings = (
        embeddings
        .detach()
        .cpu()
        .float()
    )

    print(
        "embedding shape:",
        tuple(
            embeddings.shape
        ),
    )

    assert embeddings.shape == (
        38,
        960,
    )

    assert torch.isfinite(
        embeddings
    ).all()

    # --------------------------------------------------
    # Build global-frame-index lookup.
    #
    # -1 = frame is not part of canonical Oracle V0.
    # >=0 = row in oracle_embeddings_960.pt
    # --------------------------------------------------

    frame_to_oracle_id = (
        torch.full(
            (
                args.num_dataset_frames,
            ),
            fill_value=-1,
            dtype=torch.int16,
        )
    )

    seen_indices = set()

    for record in frame_records:
        global_index = int(
            record["index"]
        )

        if not (
            0
            <= global_index
            < args.num_dataset_frames
        ):
            raise ValueError(
                f"Global index out of range: "
                f"{global_index}"
            )

        if global_index in seen_indices:
            raise ValueError(
                "Duplicate global frame index: "
                f"{global_index}"
            )

        seen_indices.add(
            global_index
        )

        oracle_id = why_to_id[
            record["why"]
        ]

        frame_to_oracle_id[
            global_index
        ] = oracle_id

    num_labeled = int(
        (
            frame_to_oracle_id
            >= 0
        ).sum()
    )

    assert num_labeled == 88302

    # --------------------------------------------------
    # Save.
    # --------------------------------------------------

    torch.save(
        {
            "embeddings": embeddings,
            "model_id": args.model_id,
            "hidden_size": 960,
            "num_oracle_labels": (
                len(why_texts)
            ),
            "why_texts": why_texts,
        },
        embeddings_path,
    )

    torch.save(
        {
            "frame_to_oracle_id": (
                frame_to_oracle_id
            ),
            "unlabeled_value": -1,
            "num_dataset_frames": (
                args.num_dataset_frames
            ),
            "num_labeled_frames": (
                num_labeled
            ),
        },
        lookup_path,
    )

    catalog = {
        "source": (
            "constructed_privileged_oracle_v0"
        ),
        "scientific_status": (
            "Privileged constructed Oracle "
            "supervision; not LIBERO "
            "ground-truth intention."
        ),
        "model_id": (
            args.model_id
        ),
        "embedding_dim": 960,
        "num_unique_why": (
            len(why_texts)
        ),
        "num_labeled_frames": (
            num_labeled
        ),
        "entries": [
            {
                "oracle_id": idx,
                "why": why,
            }
            for idx, why
            in enumerate(why_texts)
        ],
    }

    with open(
        catalog_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            catalog,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(
        "embeddings:",
        embeddings_path,
    )

    print(
        "lookup:",
        lookup_path,
    )

    print(
        "catalog:",
        catalog_path,
    )

    print()
    print(
        "labeled frames:",
        num_labeled,
    )

    print(
        "unlabeled frames:",
        args.num_dataset_frames
        - num_labeled,
    )

    print("PASS")


if __name__ == "__main__":
    main()