#!/usr/bin/env python3
"""Train fixed prefix models on train/validation, then evaluate locked test once."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
import run_prefix_length_ablation as prefix  # noqa: E402


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_candidate_rows(lock: dict) -> list[dict]:
    return lock["candidate_test_sample_manifest"]


def load_test_cache(cache_root: Path, rows: list[dict]):
    import torch

    by_id = {}
    for row in rows:
        sample_id = row["sample_id"]
        cache_id = sample_id.replace("/", "__")
        path = cache_root / "test" / f"{cache_id}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing locked candidate feature: {path}")
        record = torch.load(path, map_location="cpu", weights_only=False)
        if str(record.get("sample_id")) != sample_id:
            raise ValueError(f"Sample ID mismatch in {path}")
        if str(record.get("video_uid")) != row["video_uid"]:
            raise ValueError(f"Video UID mismatch in {path}")
        if str(record.get("why", "")).strip() != row["why"].strip():
            raise ValueError(f"WHY label mismatch in {path}")
        if str(record.get("task", "")).strip() != row["event"].strip():
            raise ValueError(f"Task/event mismatch in {path}")
        frames = record.get("frame_features")
        if not torch.is_tensor(frames) or tuple(frames.shape) != (8, 960):
            raise ValueError(f"Unexpected frame features in {path}")
        by_id[sample_id] = {
            **row,
            "frame_features": frames.float(),
        }
    return [by_id[row["sample_id"]] for row in rows]


def encode_why_texts(texts: list[str], device):
    import torch
    from models.intention_encoder import IntentionTextEncoder, IntentionVisualEncoder

    unique = list(dict.fromkeys(texts))
    visual_encoder = IntentionVisualEncoder().to(device).eval()
    text_encoder = IntentionTextEncoder(
        vlm=visual_encoder.vlm,
        processor=visual_encoder.processor,
    ).to(device).eval()
    encoded = {}
    batch_size = 64
    with torch.inference_mode():
        for start in range(0, len(unique), batch_size):
            batch = unique[start : start + batch_size]
            values = text_encoder(batch).detach().float().cpu()
            if tuple(values.shape) != (len(batch), 960):
                raise ValueError(f"Unexpected text target shape: {tuple(values.shape)}")
            for text, value in zip(batch, values):
                encoded[text] = value
    del text_encoder, visual_encoder
    return encoded


def evaluate_scores(scores: np.ndarray, rows: list[dict]) -> tuple[dict, list[int]]:
    why = [str(row["why"]).strip() for row in rows]
    tasks = [str(row["event"]).strip() for row in rows]
    pos = np.asarray([[a == b for b in why] for a in why], dtype=bool)
    ranks = prefix.first_positive_rank(scores, pos)
    top1_idx = scores.argmax(axis=1)
    correct = pos[np.arange(len(rows)), top1_idx]
    same_task = (~correct) & (np.asarray(tasks)[top1_idx] == np.asarray(tasks))
    cross_task = (~correct) & ~same_task
    return {
        "n_examples": len(rows),
        "MRR": float((1.0 / ranks).mean()),
        "Top1": float((ranks <= 1).mean()),
        "Top3": float((ranks <= 3).mean()),
        "Top5": float((ranks <= 5).mean()),
        "top1_correct": int(correct.sum()),
        "top1_wrong": int((~correct).sum()),
        "same_task_wrong": int(same_task.sum()),
        "cross_task_wrong": int(cross_task.sum()),
    }, top1_idx.tolist()


def selftest() -> None:
    rows = [
        {"why": "goal-a", "event": "task-a"},
        {"why": "goal-a", "event": "task-a"},
        {"why": "goal-b", "event": "task-b"},
    ]
    scores = np.asarray([[1.0, 0.1, 0.0], [0.0, 0.2, 1.0], [0.0, 0.1, 1.0]])
    metrics, top1 = evaluate_scores(scores, rows)
    assert top1 == [0, 2, 2]
    assert metrics["n_examples"] == 3
    assert metrics["top1_correct"] == 2
    assert metrics["same_task_wrong"] == 0
    assert metrics["cross_task_wrong"] == 1
    assert abs(metrics["MRR"] - (1 + 0.5 + 1) / 3) < 1e-12
    print("locked-test evaluation selftest: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--temporal-cache", type=Path, required=True)
    parser.add_argument("--stage-a-cache", type=Path, required=True)
    parser.add_argument("--why-pca", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return

    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite test results: {args.output_dir}")
    lock_path = args.protocol / "locked_protocol.json"
    if not lock_path.is_file():
        raise FileNotFoundError(f"Locked protocol not found: {lock_path}")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "locked_before_candidate_test_feature_access":
        raise ValueError("Protocol is not marked locked before test feature access")
    for name, path in (
        ("why_pca", args.why_pca),
        ("prefix_training_script", REPO_ROOT / "scripts/run_prefix_length_ablation.py"),
        ("protocol_lock_script", REPO_ROOT / "scripts/lock_egointent_full_test_protocol.py"),
    ):
        if file_sha256(path) != lock["input_sha256"][name]:
            raise ValueError(f"Locked input hash changed: {name} ({path})")
    evaluator_hash = file_sha256(Path(__file__).resolve())
    locked_evaluator_hash = lock["input_sha256"]["locked_test_evaluation_script"]
    if evaluator_hash != locked_evaluator_hash:
        addendum_path = args.protocol / "protocol_addendum.json"
        if not addendum_path.is_file():
            raise ValueError(
                "Test evaluator differs from the locked version and no audited "
                "implementation addendum is present."
            )
        addendum = json.loads(addendum_path.read_text(encoding="utf-8"))
        if (
            addendum.get("locked_protocol_sha256") != file_sha256(lock_path)
            or addendum.get("original_evaluator_sha256") != locked_evaluator_hash
            or addendum.get("corrected_evaluator_sha256") != evaluator_hash
            or addendum.get("no_test_metrics_produced") is not True
        ):
            raise ValueError("Implementation addendum does not match the locked protocol/evaluator")
    if lock["protocol"]["prefixes"] != [2, 4, 6, 8]:
        raise ValueError("Unexpected locked prefix configuration")
    if lock["protocol"]["seeds"] != [0, 1, 2]:
        raise ValueError("Unexpected locked seed configuration")

    import torch
    import train_temporal_progression_intention as temporal

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_rows = temporal.load_temporal_split(args.temporal_cache, args.stage_a_cache, "train")
    val_rows = temporal.load_temporal_split(args.temporal_cache, args.stage_a_cache, "val")
    train_uids = {row["video_uid"] for row in train_rows}
    val_uids = {row["video_uid"] for row in val_rows}
    if train_uids != set(lock["train_video_uids"]):
        raise ValueError("Training UIDs differ from locked protocol")
    if val_uids != set(lock["validation_video_uids"]):
        raise ValueError("Validation UIDs differ from locked protocol")

    transform = temporal.load_pca_transform(args.why_pca)
    train_frames, train_targets, train_why = temporal.stack_split(train_rows, transform, device)
    val_frames, val_targets, val_why = temporal.stack_split(val_rows, transform, device)
    val_positive = temporal.build_positive_mask(val_why, device).cpu().numpy()
    train_positive = temporal.build_positive_mask(train_why, device)
    candidate_manifest = load_candidate_rows(lock)
    test_rows = load_test_cache(args.temporal_cache, candidate_manifest)
    test_uids = {row["video_uid"] for row in test_rows}
    if test_uids != set(lock["candidate_test_video_uids"]):
        raise ValueError("Loaded test UID set differs from locked protocol")
    if len(test_rows) != lock["candidate_test_sample_count"]:
        raise ValueError("Loaded test sample count differs from locked protocol")
    test_frames = torch.stack([row["frame_features"] for row in test_rows]).to(device)
    test_why = [row["why"] for row in test_rows]
    test_task_targets = encode_why_texts(test_why, device)
    test_targets = torch.stack([test_task_targets[text] for text in test_why]).to(device)
    test_targets = torch.nn.functional.normalize(transform(test_targets), dim=-1)

    args_model = SimpleNamespace(
        hidden_dim=512,
        dropout=0.0,
        lr=0.001,
        weight_decay=0.0001,
        temperature=0.07,
        cosine_weight=0.1,
        grad_clip=1.0,
        epochs=100,
        prefixes=[2, 4, 6, 8],
    )
    train_data = {
        "device": device,
        "full_len": int(train_frames.shape[1]),
        "train_frames": train_frames,
        "train_why_target": train_targets,
        "train_why_texts": train_why,
        "train_pos": train_positive,
        "val_frames": val_frames,
        "val_why_target": val_targets,
        "val_why_texts": val_why,
        "val_pos_np": val_positive,
    }

    result_rows = []
    per_example_rows = []
    for k in lock["protocol"]["prefixes"]:
        for seed in lock["protocol"]["seeds"]:
            run_result, model, selected_state = prefix.run_one(
                temporal, torch, seed, k, train_data, args_model
            )
            model.load_state_dict(selected_state)
            model.eval()
            with torch.inference_mode():
                predicted = prefix.predict_prefix(model, test_frames, k)
                scores = (predicted @ test_targets.T).detach().float().cpu().numpy()
            metrics, top1_idx = evaluate_scores(scores, test_rows)
            result_rows.append({
                "prefix_frames": k,
                "seed": seed,
                "selected_epoch": int(run_result["best_epoch"]),
                **metrics,
            })
            ranks = prefix.first_positive_rank(
                scores,
                np.asarray([[a == b for b in test_why] for a in test_why], dtype=bool),
            )
            for index, row in enumerate(test_rows):
                predicted_row = test_rows[top1_idx[index]]
                per_example_rows.append({
                    "sample_id": row["sample_id"],
                    "video_uid": row["video_uid"],
                    "scene": row["scene"],
                    "task_id": row["event"],
                    "ground_truth_why": row["why"],
                    "predicted_why": predicted_row["why"],
                    "predicted_sample_id": predicted_row["sample_id"],
                    "predicted_video_uid": predicted_row["video_uid"],
                    "rank": int(ranks[index]),
                    "is_top1_error": int(row["why"] != predicted_row["why"]),
                    "prefix_frames": k,
                    "seed": seed,
                })
            del model, selected_state, run_result, scores

    summary = []
    for k in lock["protocol"]["prefixes"]:
        group = [row for row in result_rows if row["prefix_frames"] == k]
        entry = {"prefix_frames": k, "n_seeds": len(group)}
        for key in ("MRR", "Top1", "Top3", "Top5"):
            values = np.asarray([row[key] for row in group], dtype=float)
            entry[f"{key}_mean"] = float(values.mean())
            entry[f"{key}_std"] = float(values.std(ddof=1))
        for key in ("n_examples", "top1_correct", "top1_wrong", "same_task_wrong", "cross_task_wrong"):
            entry[f"{key}_mean"] = float(np.mean([row[key] for row in group]))
        summary.append(entry)

    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite test results: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "protocol": str(lock_path),
        "protocol_sha256": file_sha256(lock_path),
        "test_evaluation_count": 1,
        "test_tensor_load_scope": {
            "video_uids": sorted(test_uids),
            "samples": len(test_rows),
        },
        "test_gallery": "same 407 candidate samples for every prefix and seed; exact WHY text is positive",
        "pca_sha256": lock["input_sha256"]["why_pca"],
        "device": str(device),
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "results.json").write_text(
        json.dumps(
            {"per_seed": result_rows, "summary": summary},
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "per_seed_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result_rows[0]))
        writer.writeheader()
        writer.writerows(result_rows)
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    with (args.output_dir / "per_example_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_example_rows[0]))
        writer.writeheader()
        writer.writerows(per_example_rows)

    report = [
        "# Locked candidate test results",
        "",
        "Evaluated once using the protocol in `test_protocol_full_v1/locked_protocol.json`. The reported test is candidate-only, video-UID-held-out, not scene-disjoint, and not guaranteed untouched outside the inspected local artifacts.",
        "",
        "| Frames | MRR mean ± SD | Top-1 mean ± SD | Top-3 mean ± SD | Top-5 mean ± SD | Test examples/seed |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        report.append(
            f"| {row['prefix_frames']} | {row['MRR_mean']:.4f} ± {row['MRR_std']:.4f} | "
            f"{row['Top1_mean']:.4f} ± {row['Top1_std']:.4f} | "
            f"{row['Top3_mean']:.4f} ± {row['Top3_std']:.4f} | "
            f"{row['Top5_mean']:.4f} ± {row['Top5_std']:.4f} | {int(row['n_examples_mean'])} |"
        )
    report.extend([
        "",
        "Checkpoint selection used validation MRR only. No test-based threshold tuning or model selection was performed. Seed variation and repeated samples within each source video are not independent video-level replicates.",
        "",
        "This test candidate contains four source UIDs from two scenes; one full-test UID overlapping pilot validation was excluded before protocol lock. It shares the garden scene with pilot validation, so the result is not scene-independent.",
        "",
    ])
    (args.output_dir / "summary.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Locked candidate test evaluation completed once: {args.output_dir}")
    for row in summary:
        print(
            f"k={row['prefix_frames']}: MRR={row['MRR_mean']:.4f}±{row['MRR_std']:.4f} "
            f"Top1={row['Top1_mean']:.4f}±{row['Top1_std']:.4f}"
        )


if __name__ == "__main__":
    main()
