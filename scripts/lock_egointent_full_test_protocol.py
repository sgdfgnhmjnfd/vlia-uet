#!/usr/bin/env python3
"""Freeze the candidate full-dataset video-level test protocol.

This script reads split manifests and experiment configuration only. It never
opens temporal feature tensors or computes evaluation metrics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "results" / "test_protocol_full_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-root",
        type=Path,
        default=Path("/media/dhqg/d3/datasets/egointent_full_split_v1"),
    )
    parser.add_argument(
        "--pilot-split",
        type=Path,
        default=REPO_ROOT / "config/data/egointent_pilot_split_v0.json",
    )
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=REPO_ROOT / "results/prefix_length_full_v1/config.json",
    )
    parser.add_argument(
        "--pca",
        type=Path,
        default=Path("/media/dhqg/d3/vlia_outputs/stage_a_clean_v1/why_pca_960_to_256.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite protocol output: {args.output_dir}")
    for path in (args.split_root, args.pilot_split, args.experiment_config, args.pca):
        if not path.exists():
            raise FileNotFoundError(path)

    pilot = json.loads(args.pilot_split.read_text(encoding="utf-8"))
    pilot_train_uids = {str(row["video_uid"]) for row in pilot["train"]}
    pilot_val_uids = {str(row["video_uid"]) for row in pilot["val"]}
    pilot_uids = pilot_train_uids | pilot_val_uids

    split_rows = {
        name: load_jsonl(args.split_root / f"{name}.jsonl")
        for name in ("train", "val", "test")
    }
    split_uids = {
        name: {str(row["video_uid"]) for row in rows}
        for name, rows in split_rows.items()
    }
    split_ids = {
        name: {str(row["sample_id"]) for row in rows}
        for name, rows in split_rows.items()
    }
    for name in split_rows:
        if len(split_ids[name]) != len(split_rows[name]):
            raise ValueError(f"Duplicate sample IDs in {name} split")
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if split_uids[left] & split_uids[right]:
            raise ValueError(f"UID overlap between {left} and {right}")
        if split_ids[left] & split_ids[right]:
            raise ValueError(f"Sample ID overlap between {left} and {right}")

    pilot_train_overlap = split_uids["train"] & pilot_train_uids
    pilot_val_overlap = split_uids["val"] & pilot_val_uids
    if not pilot_train_uids <= split_uids["train"]:
        raise ValueError("Pilot train UIDs are not all in the full training split")
    if not pilot_val_uids <= (split_uids["val"] | split_uids["test"]):
        raise ValueError("A pilot validation UID is missing from full validation/test")

    excluded_test_uids = split_uids["test"] & pilot_val_uids
    candidate_test_uids = split_uids["test"] - pilot_uids
    if not candidate_test_uids:
        raise ValueError("No candidate test UIDs remain after excluding pilot sources")
    candidate_test_rows = [
        row for row in split_rows["test"]
        if str(row["video_uid"]) in candidate_test_uids
    ]
    candidate_test_manifest = [
        {
            "sample_id": str(row["sample_id"]),
            "video_uid": str(row["video_uid"]),
            "scene": str(row["scene"]),
            "event": str(row["event"]),
            "setting": str(row["setting"]),
            "why": str(row["why"]),
        }
        for row in candidate_test_rows
    ]

    experiment = json.loads(args.experiment_config.read_text(encoding="utf-8"))
    run_args = experiment["args"]
    if set(experiment["train_videos"]) != split_uids["train"]:
        raise ValueError("Existing full run training UIDs do not match the locked full train split")
    if set(experiment["val_videos"]) != split_uids["val"]:
        raise ValueError("Existing full run validation UIDs do not match the locked full validation split")
    if [int(x.strip()) for x in run_args["prefixes"].strip("[]").split(",")] != [2, 4, 6, 8]:
        raise ValueError("Existing full validation run is not configured for prefixes 2/4/6/8")
    if [int(x.strip()) for x in run_args["seeds"].strip("[]").split(",")] != [0, 1, 2]:
        raise ValueError("Existing full validation run is not configured for seeds 0/1/2")
    if int(run_args["epochs"]) != 100:
        raise ValueError("Existing full validation run is not configured for 100 epochs")

    input_hashes = {
        "train_jsonl": sha256(args.split_root / "train.jsonl"),
        "val_jsonl": sha256(args.split_root / "val.jsonl"),
        "test_jsonl": sha256(args.split_root / "test.jsonl"),
        "pilot_split_json": sha256(args.pilot_split),
        "full_validation_experiment_config": sha256(args.experiment_config),
        "why_pca": sha256(args.pca),
        "prefix_training_script": sha256(REPO_ROOT / "scripts/run_prefix_length_ablation.py"),
        "locked_test_evaluation_script": sha256(REPO_ROOT / "scripts/evaluate_locked_prefix_test.py"),
        "protocol_lock_script": sha256(Path(__file__).resolve()),
    }
    manifest = {
        "protocol_id": "egointent_full_candidate_test_v1",
        "status": "locked_before_candidate_test_feature_access",
        "created_utc": "2026-10-15",
        "dataset_root": "/media/dhqg/d3/datasets/egointent_full",
        "split_root": str(args.split_root),
        "split_counts": {
            name: {
                "samples": len(split_rows[name]),
                "video_uids": len(split_uids[name]),
            }
            for name in ("train", "val", "test")
        },
        "train_video_uids": sorted(split_uids["train"]),
        "validation_video_uids": sorted(split_uids["val"]),
        "excluded_test_video_uids": sorted(excluded_test_uids),
        "candidate_test_video_uids": sorted(candidate_test_uids),
        "candidate_test_sample_count": len(candidate_test_rows),
        "candidate_test_sample_manifest": candidate_test_manifest,
        "pilot_uid_overlap": {
            "train": sorted(pilot_train_overlap),
            "validation": sorted(split_uids["val"] & pilot_uids),
            "excluded_from_candidate_test": sorted(excluded_test_uids),
        },
        "protocol": {
            "unit_of_split": "video_uid",
            "training_split": "existing full train split",
            "checkpoint_selection": "highest validation MRR at the model's own prefix length; strict greater-than tie handling",
            "validation_split": "existing full validation split; used for checkpoint selection only",
            "test_queries": "all samples from the four candidate UIDs only",
            "test_gallery": "all candidate test samples from those same four UIDs, using exact WHY text as multi-positive relevance",
            "prefixes": [2, 4, 6, 8],
            "seeds": [0, 1, 2],
            "epochs": 100,
            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "temperature": 0.07,
            "cosine_weight": 0.1,
            "hidden_dim": 512,
            "dropout": 0.0,
            "grad_clip": 1.0,
            "model": "MeanPoolBaseline, trained and evaluated at the same prefix length",
            "text_target_transform": "existing frozen 960-to-256 WHY PCA fit on pilot train UIDs (a subset of full train); fixed before candidate test evaluation",
            "metrics": ["MRR", "Top1", "Top3", "Top5", "n_examples", "same_task_errors", "cross_task_errors"],
            "test_usage": "one evaluation run after locking; no threshold or model selection from candidate test results",
        },
        "limitations": [
            "Candidate set is video-UID-disjoint from pilot training and validation, but not scene-disjoint: garden appeared in pilot validation.",
            "Candidate UIDs had no exact reference in inspected local result files, but unrecorded external use cannot be excluded.",
            "This is a candidate video-level test, not a pristine or scene-independent test benchmark.",
            "The text target PCA was fit on pilot train, not refit on all full-train UIDs.",
        ],
        "input_sha256": input_hashes,
    }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "locked_protocol.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "candidate_test_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "video_uid", "scene", "event", "setting", "why"],
        )
        writer.writeheader()
        writer.writerows(candidate_test_manifest)
    lines = [
        "# Locked candidate test protocol: EgoIntent full v1",
        "",
        "**Protocol locked before reading candidate test feature tensors.** This is a video-UID-held-out candidate test, not a pristine or scene-disjoint benchmark.",
        "",
        f"- Train: {len(split_rows['train'])} samples / {len(split_uids['train'])} UIDs.",
        f"- Validation: {len(split_rows['val'])} samples / {len(split_uids['val'])} UIDs.",
        f"- Candidate test: {len(candidate_test_rows)} samples / {len(candidate_test_uids)} UIDs.",
        f"- Excluded pilot-validation test UIDs: {', '.join(sorted(excluded_test_uids))}.",
        "- Candidate UIDs: " + ", ".join(sorted(candidate_test_uids)) + ".",
        "- Model: mean-pool baseline trained and evaluated at each of 2/4/6/8 frames; seeds 0/1/2; 100 epochs.",
        "- Checkpoint selection: validation MRR at matching training/evaluation prefix; candidate test is not used for selection.",
        "- Gallery: candidate-test samples only; exact WHY text match defines relevance.",
        "- Test use: one run, no tuning or repeated selection on test results.",
        "",
        "## Limitations",
        "",
        *[f"- {item}" for item in manifest["limitations"]],
        "",
        "Input and code hashes are recorded in `locked_protocol.json`. Do not change this manifest after candidate-test evaluation.",
        "",
    ]
    (args.output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(
        f"LOCKED: candidate test {len(candidate_test_rows)} samples/"
        f"{len(candidate_test_uids)} UIDs; test feature tensors were not opened."
    )


if __name__ == "__main__":
    main()
