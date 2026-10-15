#!/usr/bin/env python3
"""Summarize the already-completed locked prefix test run without rerunning it."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def chance_metrics(why: list[str]) -> dict[str, float]:
    n = len(why)
    counts: dict[str, int] = defaultdict(int)
    for label in why:
        counts[label] += 1
    chance_rank = 0.0
    chance_top1 = 0.0
    chance_top3 = 0.0
    for label in why:
        positives = counts[label]
        q = 1.0
        expected_reciprocal_rank = 0.0
        expected_top3 = 0.0
        for rank in range(1, n - positives + 2):
            probability = q * positives / (n - rank + 1)
            expected_reciprocal_rank += probability / rank
            if rank <= 3:
                expected_top3 += probability
            q *= (n - positives - rank + 1) / (n - rank + 1)
        chance_rank += expected_reciprocal_rank
        chance_top1 += positives / n
        chance_top3 += expected_top3
    return {
        "MRR": chance_rank / n,
        "Top1": chance_top1 / n,
        "Top3": chance_top3 / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    lock = json.loads((args.protocol / "locked_protocol.json").read_text(encoding="utf-8"))
    raw = read_csv(args.results / "per_example_predictions.csv")
    seed_metrics = read_csv(args.results / "per_seed_metrics.csv")
    if len(raw) != 407 * 4 * 3:
        raise ValueError(f"Unexpected per-example row count: {len(raw)}")
    if {row["sample_id"] for row in raw} != {
        row["sample_id"] for row in lock["candidate_test_sample_manifest"]
    }:
        raise ValueError("Prediction identities do not match the locked sample manifest")
    if {row["video_uid"] for row in raw} != set(lock["candidate_test_video_uids"]):
        raise ValueError("Prediction UIDs do not match the locked candidate UID set")

    metadata_by_id = {
        row["sample_id"]: row
        for row in lock["candidate_test_sample_manifest"]
    }
    why = [metadata_by_id[sample_id]["why"] for sample_id in sorted(metadata_by_id)]
    chance = chance_metrics(why)

    error_summary = []
    for k in lock["protocol"]["prefixes"]:
        rows = [row for row in raw if int(row["prefix_frames"]) == k]
        errors = [row for row in rows if row["is_top1_error"] == "1"]
        same_task = sum(
            metadata_by_id[row["sample_id"]]["event"]
            == metadata_by_id[row["predicted_sample_id"]]["event"]
            for row in errors
        )
        error_summary.append({
            "prefix_frames": k,
            "seed_example_errors": len(errors),
            "same_task_errors": same_task,
            "cross_task_errors": len(errors) - same_task,
            "same_task_fraction_of_errors": same_task / len(errors) if errors else None,
        })

    by_seed_k_id = {
        (int(row["seed"]), int(row["prefix_frames"]), row["sample_id"]): row
        for row in raw
    }
    video_by_sample = {
        sample_id: metadata["video_uid"]
        for sample_id, metadata in metadata_by_id.items()
    }
    comparisons = []
    rng = np.random.default_rng(20261015)
    uid_order = sorted(set(video_by_sample.values()))
    sample_ids = sorted(metadata_by_id)
    video_masks = {
        uid: np.asarray([video_by_sample[sample_id] == uid for sample_id in sample_ids])
        for uid in uid_order
    }
    for from_k, to_k in ((2, 4), (4, 6), (6, 8), (2, 8)):
        per_sample_delta = []
        top1_gains = 0
        top1_losses = 0
        for sample_id in sample_ids:
            seed_deltas = []
            for seed in lock["protocol"]["seeds"]:
                old = by_seed_k_id[(seed, from_k, sample_id)]
                new = by_seed_k_id[(seed, to_k, sample_id)]
                seed_deltas.append(1 / int(new["rank"]) - 1 / int(old["rank"]))
                old_ok = old["is_top1_error"] == "0"
                new_ok = new["is_top1_error"] == "0"
                top1_gains += int(new_ok and not old_ok)
                top1_losses += int(old_ok and not new_ok)
            per_sample_delta.append(float(np.mean(seed_deltas)))
        per_sample_delta = np.asarray(per_sample_delta)
        video_sums = np.asarray([
            per_sample_delta[video_masks[uid]].sum()
            for uid in uid_order
        ])
        video_counts = np.asarray([
            int(video_masks[uid].sum())
            for uid in uid_order
        ])
        draws = rng.integers(0, len(uid_order), size=(50000, len(uid_order)))
        boot = video_sums[draws].sum(axis=1) / video_counts[draws].sum(axis=1)
        lo, hi = np.quantile(boot, [0.025, 0.975])
        delta_by_video = {
            uid: float(per_sample_delta[video_masks[uid]].mean())
            for uid in uid_order
        }
        comparisons.append({
            "from_k": from_k,
            "to_k": to_k,
            "n_examples": len(sample_ids),
            "n_videos": len(uid_order),
            "n_seeds": len(lock["protocol"]["seeds"]),
            "delta_mrr_paired": float(per_sample_delta.mean()),
            "video_cluster_bootstrap_ci95_low": float(lo),
            "video_cluster_bootstrap_ci95_high": float(hi),
            "top1_gains_seed_examples": top1_gains,
            "top1_losses_seed_examples": top1_losses,
            "delta_mrr_by_video_json": json.dumps(delta_by_video, sort_keys=True),
            "bootstrap_seed": 20261015,
            "bootstrap_resamples": 50000,
        })

    write_csv(
        args.results / "paired_test_comparisons.csv",
        comparisons,
        list(comparisons[0]),
    )
    error_summary_obj = {
        "n_test_examples": len(sample_ids),
        "n_test_video_uids": len(uid_order),
        "chance": chance,
        "error_breakdown": error_summary,
        "test_metric_scope": "post-hoc descriptive analysis of the single locked evaluation; no tuning",
    }
    (args.results / "test_analysis.json").write_text(
        json.dumps(error_summary_obj, indent=2) + "\n",
        encoding="utf-8",
    )
    report = [
        "# Locked candidate test: paired analysis",
        "",
        "This file is a post-hoc analysis of the one locked evaluation, not a new evaluation or a model-selection step. Candidate test remains video-UID-held-out only; it is not scene-disjoint or guaranteed untouched outside the inspected local artifacts.",
        "",
        f"Exact multi-positive random-ranking chance: MRR {chance['MRR']:.4f}, Top-1 {chance['Top1']:.4f}, Top-3 {chance['Top3']:.4f}.",
        "",
        "| Frames | Top-1 error seed-examples | Same-task errors | Cross-task errors | Same-task share |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in error_summary:
        report.append(
            f"| {row['prefix_frames']} | {row['seed_example_errors']} | {row['same_task_errors']} | "
            f"{row['cross_task_errors']} | {row['same_task_fraction_of_errors']:.3f} |"
        )
    report.extend([
        "",
        "## Paired prefix differences",
        "",
        "The paired ΔMRR uses the same candidate test examples and averages across seeds per example. The intervals resample the four video UIDs (50,000 replicates) and are exploratory given only four independent clusters.",
        "",
        "| Pair | Paired ΔMRR | 4-video cluster bootstrap 95% interval | Top-1 gains/losses (seed-examples) |",
        "|---|---:|---:|---:|",
    ])
    for row in comparisons:
        report.append(
            f"| {row['from_k']}→{row['to_k']} | {row['delta_mrr_paired']:+.4f} | "
            f"[{row['video_cluster_bootstrap_ci95_low']:+.4f}, {row['video_cluster_bootstrap_ci95_high']:+.4f}] | "
            f"{row['top1_gains_seed_examples']} / {row['top1_losses_seed_examples']} |"
        )
    report.extend([
        "",
        "These results are descriptive only. There are few source-video clusters and low Top-1 counts; no claim of statistical superiority, calibrated uncertainty, semantic sufficiency, or broad generalization is supported.",
        "",
    ])
    (args.results / "test_analysis.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Post-hoc test analysis written: {args.results}")


if __name__ == "__main__":
    main()
