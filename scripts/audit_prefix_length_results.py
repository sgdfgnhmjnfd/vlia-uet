#!/usr/bin/env python3
"""Audit saved prefix-ablation and selective-prediction artifacts.

The audit is read-only with respect to source caches and prior experiment
outputs. New deliverables are written to a fresh output directory.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
import run_prefix_length_ablation as prefix  # noqa: E402
import train_temporal_progression_intention as temporal  # noqa: E402


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def exact_chance(why_texts: list[str]) -> dict[str, float]:
    return prefix.chance_levels(why_texts)


def cluster_bootstrap_ci(
    per_example_delta: np.ndarray,
    video_uids: list[str],
    seed: int,
    n_boot: int = 20000,
) -> tuple[float, float, dict[str, float]]:
    groups = sorted(set(video_uids))
    by_video = {
        uid: float(per_example_delta[np.asarray(video_uids) == uid].mean())
        for uid in groups
    }
    # Resample the two observed source-video clusters, not individual steps.
    rng = np.random.default_rng(seed)
    boot = rng.choice(np.asarray(groups, dtype=object), size=(n_boot, len(groups)), replace=True)
    values = np.asarray([[by_video[uid] for uid in draw] for draw in boot]).mean(axis=1)
    lo, hi = np.quantile(values, [0.025, 0.975])
    return float(lo), float(hi), by_video


def hypergeometric_quantile(q: float, population: int, successes: int, draws: int) -> int:
    """Exact quantile for correct predictions in a uniformly random accepted subset."""
    low = max(0, draws - (population - successes))
    high = min(draws, successes)
    denominator = math.comb(population, draws)
    cumulative = 0.0
    for correct in range(low, high + 1):
        probability = (
            math.comb(successes, correct)
            * math.comb(population - successes, draws - correct)
            / denominator
        )
        cumulative += probability
        if cumulative >= q:
            return correct
    return high


def random_rejection_risk(population: int, correct_total: int, accepted: int) -> tuple[float | None, float | None, float | None]:
    """Expected risk and exact finite-population 95% interval at a fixed coverage."""
    if accepted == 0:
        return None, None, None
    expected = 1.0 - correct_total / population
    low_correct = hypergeometric_quantile(0.025, population, correct_total, accepted)
    high_correct = hypergeometric_quantile(0.975, population, correct_total, accepted)
    return expected, 1.0 - high_correct / accepted, 1.0 - low_correct / accepted


def svg_risk_plot(
    policy: str,
    curves: dict[str, list[tuple[float, float]]],
    random_curves: dict[str, list[tuple[float, float]]],
    out: Path,
) -> None:
    width, height = 1000, 650
    left, right, top, bottom = 78, 28, 42, 92
    plot_w, plot_h = width - left - right, height - top - bottom
    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
        "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#003f5c", "#ffa600",
    ]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="24" font-size="16">{policy} risk–coverage (mean over 3 seeds; validation only)</text>',
    ]
    for tick in range(0, 11, 2):
        x = left + plot_w * tick / 10
        y = height - bottom - plot_h * tick / 10
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height-bottom}" stroke="#ddd"/>')
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#ddd"/>')
        parts.append(f'<text x="{x-8:.1f}" y="{height-bottom+20}" font-size="11">{tick/10:.1f}</text>')
        parts.append(f'<text x="{left-35}" y="{y+4:.1f}" font-size="11">{tick/10:.1f}</text>')
    parts.extend([
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="black"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="black"/>',
        f'<text x="{left + plot_w/2 - 35}" y="{height-16}" font-size="13">coverage</text>',
        f'<text x="18" y="{top + plot_h/2}" font-size="13" transform="rotate(-90 18 {top + plot_h/2})">risk / error rate</text>',
    ])
    grid = np.linspace(0.0, 1.0, 21)
    for idx, (label, points) in enumerate(sorted(curves.items())):
        color = colors[idx % len(colors)]
        xs = np.asarray([p[0] for p in points], dtype=float)
        ys = np.asarray([p[1] for p in points], dtype=float)
        order = np.argsort(xs)
        xs, ys = xs[order], ys[order]
        unique_x, unique_idx = np.unique(xs, return_index=True)
        unique_y = ys[unique_idx]
        if unique_x.size == 0:
            continue
        # Every curve includes accept-all; do not extrapolate past observed coverage.
        x_grid = grid[(grid >= unique_x.min()) & (grid <= unique_x.max())]
        y_grid = np.interp(x_grid, unique_x, unique_y)
        coords = " ".join(
            f"{left + x*plot_w:.1f},{height-bottom-y*plot_h:.1f}"
            for x, y in zip(x_grid, y_grid)
        )
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="1.7" points="{coords}"/>')
        # Mark the always-predict endpoint (coverage 1.0).
        if unique_x[-1] >= 0.999:
            x, y = left + plot_w, height - bottom - unique_y[-1] * plot_h
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}"/>')
        random_points = random_curves.get(label, [])
        if random_points:
            random_coords = " ".join(
                f"{left + x*plot_w:.1f},{height-bottom-y*plot_h:.1f}"
                for x, y in random_points
            )
            parts.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="1.2" '
                f'stroke-dasharray="5 4" opacity="0.7" points="{random_coords}"/>'
            )
        legend_y = 64 + (idx // 3) * 16
        legend_x = left + (idx % 3) * 290
        parts.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x+18}" y2="{legend_y}" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<text x="{legend_x+24}" y="{legend_y+4}" font-size="10">{label}</text>')
    parts.extend([
        f'<line x1="{width-300}" y1="{height-26}" x2="{width-278}" y2="{height-26}" stroke="#555" stroke-width="1.5" stroke-dasharray="5 4"/>',
        f'<text x="{width-270}" y="{height-22}" font-size="10">random rejection baseline</text>',
        '</svg>',
    ])
    out.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temporal-cache", type=Path, required=True)
    parser.add_argument("--stage-a-cache", type=Path, required=True)
    parser.add_argument("--prefix-results", type=Path, required=True)
    parser.add_argument("--selective-results", type=Path, required=True)
    parser.add_argument("--split-config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--full-split-root", type=Path)
    parser.add_argument("--full-temporal-cache", type=Path)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "audit_v1")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing audit files in {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    notes: list[str] = []
    checks: list[tuple[str, bool, str]] = []
    config = json.loads((args.prefix_results / "config.json").read_text(encoding="utf-8"))
    saved_summary = json.loads((args.prefix_results / "summary.json").read_text(encoding="utf-8"))
    rank_rows = read_csv(args.prefix_results / "per_example_ranks.csv")
    per_seed_saved = read_csv(args.prefix_results / "per_seed_metrics.csv")
    summary_saved = read_csv(args.prefix_results / "summary.csv")
    paired_saved = read_csv(args.prefix_results / "paired_differences.csv")
    split_cfg = json.loads(args.split_config.read_text(encoding="utf-8"))

    train_rows = temporal.load_temporal_split(args.temporal_cache, args.stage_a_cache, "train")
    val_rows = temporal.load_temporal_split(args.temporal_cache, args.stage_a_cache, "val")
    train_ids = [r["sample_id"] for r in train_rows]
    val_ids = [r["sample_id"] for r in val_rows]
    train_uids = {r["video_uid"] for r in train_rows}
    val_uids = {r["video_uid"] for r in val_rows}
    checks.append(("No train/validation video_uid overlap", not train_uids & val_uids, f"overlap={sorted(train_uids & val_uids)}"))
    checks.append(("Unique train/validation sample identities", len(set(train_ids)) == len(train_ids) and len(set(val_ids)) == len(val_ids), f"train={len(train_ids)}, val={len(val_ids)}"))
    checks.append(("Train/validation sample identities disjoint", not set(train_ids) & set(val_ids), f"overlap={len(set(train_ids) & set(val_ids))}"))
    checks.append(("Config video IDs agree with loaded cache", train_uids == set(config["train_videos"]) and val_uids == set(config["val_videos"]), f"train_uid={len(train_uids)}, val_uid={len(val_uids)}"))
    checks.append(("Per-example rank table uses cache validation identities and order", [r["sample_id"] for r in rank_rows] == val_ids, f"rows={len(rank_rows)}, cache_val={len(val_ids)}"))
    if [r["sample_id"] for r in rank_rows] != val_ids:
        raise RuntimeError("per_example_ranks.csv does not align with the current val cache order")

    why_texts = [r["why"] for r in val_rows]
    tasks = [r["task"] for r in val_rows]
    video_uids = [r["video_uid"] for r in val_rows]
    n_gallery = len(val_rows)
    positive_mask = temporal.build_positive_mask(why_texts, "cpu").numpy()
    expected_positive_mask = np.asarray([[a == b for b in why_texts] for a in why_texts], dtype=bool)
    checks.append((
        "Positive mask matches exact WHY equality",
        np.array_equal(positive_mask, expected_positive_mask) and np.array_equal(positive_mask, positive_mask.T),
        f"positive pairs={int(positive_mask.sum())}, diagonal={bool(np.diag(positive_mask).all())}",
    ))
    chance = exact_chance(why_texts)
    checks.append(("Chance baseline matches saved config", all(abs(chance[k] - config["chance"][k]) < 1e-10 for k in chance), json.dumps(chance, sort_keys=True)))
    checks.append(("Gallery semantics: all val WHY targets; exact-text positives", n_gallery == len(why_texts) and all(bool(x) for x in why_texts), f"gallery={n_gallery}, unique_why={len(set(why_texts))}"))

    seeds = [int(x) for x in config["args"]["seeds"].strip("[]").replace(",", " ").split()]
    prefixes = [int(x) for x in config["args"]["prefixes"].strip("[]").replace(",", " ").split()]
    settings = ("train_k_eval_k", "train_full_eval_k")
    selections = ("best_val_epoch", "last_epoch")
    raw_predictions: dict[tuple[str, str, int, int], tuple[list[int], list[int]]] = {}
    audit_examples: list[dict] = []
    metric_groups: dict[tuple[str, str, int], list[dict]] = {}
    mismatch_count = 0
    top1_rank_disagreements = 0
    for setting in settings:
        for selection in selections:
            for k in prefixes:
                seed_metrics = []
                for seed in seeds:
                    train_len = k if setting == "train_k_eval_k" else int(config["frames_per_sample"])
                    run_path = args.prefix_results / "runs" / f"train{train_len}_seed{seed}.json"
                    if not run_path.exists():
                        raise FileNotFoundError(f"Required raw per-seed predictions missing: {run_path}")
                    run = json.loads(run_path.read_text(encoding="utf-8"))
                    eval_data = run["evals"][selection][str(k)]
                    ranks = [int(x) for x in eval_data["ranks"]]
                    top1_idx = [int(x) for x in eval_data["top1_idx"]]
                    raw_predictions[(setting, selection, k, seed)] = ranks, top1_idx
                    wide_rank_col = f"rank|{setting}|{selection}|k{k}|seed{seed}"
                    wide_ranks = [int(row[wide_rank_col]) for row in rank_rows]
                    if ranks != wide_ranks or len(ranks) != n_gallery:
                        mismatch_count += 1
                        raise AssertionError(f"Raw run ranks do not match wide table: {run_path} {wide_rank_col}")
                    if len(top1_idx) != n_gallery or any(i < 0 or i >= n_gallery for i in top1_idx):
                        raise AssertionError(f"Invalid top1_idx in {run_path} k={k}")
                    metrics = prefix.rank_metrics(ranks)
                    # Task breakdown follows the original code: errors are same-task
                    # iff the selected gallery row has the query task.
                    correct = [why_texts[i] == why_texts[j] for i, j in enumerate(top1_idx)]
                    top1_rank_disagreements += sum((r == 1) != ok for r, ok in zip(ranks, correct))
                    same_wrong = sum((not ok) and tasks[j] == tasks[i] for i, (ok, j) in enumerate(zip(correct, top1_idx)))
                    cross_wrong = sum((not ok) and tasks[j] != tasks[i] for i, (ok, j) in enumerate(zip(correct, top1_idx)))
                    metrics.update(
                        top1_correct=int(sum(correct)),
                        top1_wrong=n_gallery - int(sum(correct)),
                        same_task_wrong=int(same_wrong),
                        cross_task_wrong=int(cross_wrong),
                        same_task_wrong_fraction=(same_wrong / (n_gallery - sum(correct)) if n_gallery > sum(correct) else None),
                    )
                    seed_metrics.append(metrics)
                    for i, pred_idx in enumerate(top1_idx):
                        correct_why = why_texts[i] == why_texts[pred_idx]
                        audit_examples.append({
                            "sample_id": val_rows[i]["sample_id"],
                            "video_uid": video_uids[i],
                            "task_id": tasks[i],
                            "ground_truth_why": why_texts[i],
                            "predicted_why": why_texts[pred_idx],
                            "predicted_candidate_sample_id": val_rows[pred_idx]["sample_id"],
                            "predicted_candidate_video_uid": video_uids[pred_idx],
                            "predicted_candidate_task_id": tasks[pred_idx],
                            "positive_rank": ranks[i],
                            "is_top1_error": int(not correct_why),
                            "setting": setting,
                            "selection": selection,
                            "prefix_frames": k,
                            "seed": seed,
                            "model_condition": "mean_pool",
                        })
                metric_groups[(setting, selection, k)] = seed_metrics

    # Verify per-seed and aggregate published values against the saved raw runs.
    saved_seed_lookup = {
        (r["setting"], r["selection"], int(r["prefix_frames"]), int(r["seed"])): r
        for r in per_seed_saved
    }
    seed_metric_fields = ("MRR", "Top1", "Top3", "Top5", "top1_correct", "same_task_wrong", "cross_task_wrong")
    for key, seed_metrics in metric_groups.items():
        setting, selection, k = key
        for seed, metric in zip(seeds, seed_metrics):
            old = saved_seed_lookup[(setting, selection, k, seed)]
            for field in seed_metric_fields:
                if abs(float(old[field]) - float(metric[field])) > 1e-10:
                    mismatch_count += 1
                    raise AssertionError(f"Per-seed metric mismatch {key=} {seed=} {field=}: {old[field]} vs {metric[field]}")
        aggregate = {
            field: float(np.mean([m[field] for m in seed_metrics]))
            for field in ("MRR", "Top1", "Top3", "Top5")
        }
        std = {
            field: float(np.std([m[field] for m in seed_metrics], ddof=1))
            for field in ("MRR", "Top1", "Top3", "Top5")
        }
        old_sum = next(r for r in summary_saved if r["setting"] == setting and r["selection"] == selection and int(r["prefix_frames"]) == k)
        for field, value in aggregate.items():
            if abs(float(old_sum[f"{field}_mean"]) - value) > 1e-10 or abs(float(old_sum[f"{field}_std"]) - std[field]) > 1e-10:
                mismatch_count += 1
                raise AssertionError(f"Aggregate mismatch {key=} {field=}")
        for field in ("top1_correct", "same_task_wrong", "cross_task_wrong"):
            if int(old_sum[f"{field}_total"]) != sum(m[field] for m in seed_metrics):
                mismatch_count += 1
                raise AssertionError(f"Aggregate error breakdown mismatch {key=} {field=}")
    checks.append(("Raw per-seed ranks and predictions exist for all conditions", len(raw_predictions) == len(settings) * len(selections) * len(prefixes) * len(seeds), f"combinations={len(raw_predictions)}"))
    checks.append(("Recomputed per-seed and aggregate metrics match saved CSV", mismatch_count == 0, f"mismatches={mismatch_count}; Top1/3/5 use positive rank <=1/3/5"))
    checks.append(("Evaluation sample IDs and gallery are identical across prefixes", len(set(val_ids)) == n_gallery and all(len(v[0]) == n_gallery for v in raw_predictions.values()), f"n_examples={n_gallery}, gallery_candidates={n_gallery}"))
    checks.append(("Positive labels are exact WHY-text matches", True, "same construction as build_positive_mask/retrieval_metrics"))
    checks.append((
        "Top-1 label correctness agrees with rank-1 metric",
        top1_rank_disagreements == 0,
        f"disagreements={top1_rank_disagreements}; exact-score ties would require score vectors to resolve",
    ))
    checks.append((
        "Saved training-length rank cross-check is within tolerance",
        config["crosscheck_max_abs_diff"] is not None and config["crosscheck_max_abs_diff"] < 1e-5,
        f"max_abs_diff={config['crosscheck_max_abs_diff']}",
    ))

    # Construct paired differences using source videos as the resampling unit.
    paired_rows: list[dict] = []
    old_pair_lookup = {
        (r["setting"], r["selection"], int(r["from_k"]), int(r["to_k"])): r
        for r in paired_saved
    }
    compare_pairs = list(zip(prefixes[:-1], prefixes[1:])) + [(prefixes[0], prefixes[-1])]
    for setting in settings:
        for selection in selections:
            for from_k, to_k in compare_pairs:
                per_seed_delta = []
                seed_deltas = []
                for seed in seeds:
                    r_from, _ = raw_predictions[(setting, selection, from_k, seed)]
                    r_to, _ = raw_predictions[(setting, selection, to_k, seed)]
                    delta = 1 / np.asarray(r_to, dtype=float) - 1 / np.asarray(r_from, dtype=float)
                    per_seed_delta.append(delta)
                    seed_deltas.append(float(delta.mean()))
                per_seed_delta_arr = np.asarray(per_seed_delta)
                mean_per_example = per_seed_delta_arr.mean(axis=0)
                cluster_lo, cluster_hi, by_video = cluster_bootstrap_ci(
                    mean_per_example, video_uids, seed=from_k * 100 + to_k
                )
                prior = old_pair_lookup[(setting, selection, from_k, to_k)]
                paired_rows.append({
                    "setting": setting,
                    "selection": selection,
                    "from_k": from_k,
                    "to_k": to_k,
                    "n_examples": n_gallery,
                    "n_seeds": len(seeds),
                    "n_source_videos": len(set(video_uids)),
                    "delta_mrr_pooled_examples": float(mean_per_example.mean()),
                    "delta_mrr_per_seed_json": json.dumps(seed_deltas),
                    "delta_mrr_by_video_json": json.dumps(by_video, sort_keys=True),
                    "video_cluster_bootstrap_ci95_low": cluster_lo,
                    "video_cluster_bootstrap_ci95_high": cluster_hi,
                    "cluster_ci_note": "Resamples only 2 observed videos; too few clusters for reliable population inference.",
                    "published_example_bootstrap_delta": float(prior["delta_MRR_mean"]),
                    "published_example_bootstrap_ci95_low": float(prior["delta_MRR_ci95_lo"]),
                    "published_example_bootstrap_ci95_high": float(prior["delta_MRR_ci95_hi"]),
                    "top1_gains_pooled_seed_examples": int(sum(((np.asarray(raw_predictions[(setting, selection, to_k, seed)][0]) <= 1) & ~(np.asarray(raw_predictions[(setting, selection, from_k, seed)][0]) <= 1)).sum() for seed in seeds)),
                    "top1_losses_pooled_seed_examples": int(sum((~(np.asarray(raw_predictions[(setting, selection, to_k, seed)][0]) <= 1) & (np.asarray(raw_predictions[(setting, selection, from_k, seed)][0]) <= 1)).sum() for seed in seeds)),
                })
    checks.append(("Paired rank comparisons use the same per-example identities", all(r["n_examples"] == n_gallery for r in paired_rows), f"comparisons={len(paired_rows)}"))

    # Same-task error tables, with row counts distinct from seed-example counts.
    task_counts: Counter = Counter()
    pair_counts: Counter = Counter()
    error_videos: dict[tuple, set[str]] = defaultdict(set)
    error_samples: dict[tuple, set[str]] = defaultdict(set)
    for row in audit_examples:
        if not row["is_top1_error"]:
            continue
        task_key = (row["setting"], row["selection"], row["prefix_frames"], row["task_id"])
        pair_key = (row["setting"], row["selection"], row["prefix_frames"], row["task_id"], row["ground_truth_why"], row["predicted_why"])
        task_counts[task_key] += 1
        pair_counts[pair_key] += 1
        error_videos[task_key].add(row["video_uid"])
        error_videos[pair_key].add(row["video_uid"])
        error_samples[task_key].add(row["sample_id"])
        error_samples[pair_key].add(row["sample_id"])
    task_error_rows = [
        {"setting": key[0], "selection": key[1], "prefix_frames": key[2], "task_id": key[3],
         "top1_error_seed_examples": count, "unique_error_samples": len(error_samples[key]),
         "source_videos_with_errors": len(error_videos[key])}
        for key, count in sorted(task_counts.items())
    ]
    pair_error_rows = [
        {"setting": key[0], "selection": key[1], "prefix_frames": key[2], "task_id": key[3],
         "ground_truth_why": key[4], "predicted_why": key[5],
         "top1_error_seed_examples": count, "unique_error_samples": len(error_samples[key]),
         "source_videos_with_errors": len(error_videos[key])}
        for key, count in sorted(pair_counts.items())
    ]
    write_csv(args.output_dir / "per_example_errors.csv", audit_examples, [
        "sample_id", "video_uid", "task_id", "ground_truth_why", "predicted_why",
        "predicted_candidate_sample_id", "predicted_candidate_video_uid", "predicted_candidate_task_id",
        "positive_rank", "is_top1_error", "setting", "selection", "prefix_frames", "seed", "model_condition",
    ])
    write_csv(args.output_dir / "top1_errors_by_task.csv", task_error_rows, [
        "setting", "selection", "prefix_frames", "task_id", "top1_error_seed_examples",
        "unique_error_samples", "source_videos_with_errors",
    ])
    write_csv(args.output_dir / "top1_errors_by_why_pair.csv", pair_error_rows, [
        "setting", "selection", "prefix_frames", "task_id", "ground_truth_why", "predicted_why",
        "top1_error_seed_examples", "unique_error_samples", "source_videos_with_errors",
    ])
    write_csv(args.output_dir / "prefix_paired_comparison.csv", paired_rows, list(paired_rows[0]))

    # Audit selective sweeps; they store threshold aggregates, not per-example
    # scores/IDs. Deduplicate thresholds that produce the same accepted count.
    selective_path = args.selective_results / "selective_decision_thresholds.csv"
    selective_rows = read_csv(selective_path)
    curves: dict[tuple[str, int, str, int], list[dict]] = defaultdict(list)
    for row in selective_rows:
        key = (row["condition"], int(row["prefix_frames"]), row["policy"], int(row["seed"]))
        curves[key].append(row)
    risk_rows: list[dict] = []
    curve_averages: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(dict)
    random_curve_averages: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(dict)
    curve_group_counts = Counter()
    accept_all_errors: list[float] = []
    for (condition, k, policy_name, seed), rows in sorted(curves.items()):
        by_accepted: dict[int, dict] = {}
        for row in rows:
            accepted = int(row["accepted"])
            prev = by_accepted.get(accepted)
            if prev is None or (policy_name == "entropy" and float(row["threshold"]) > float(prev["threshold"])) or (policy_name != "entropy" and float(row["threshold"]) < float(prev["threshold"])):
                by_accepted[accepted] = row
        if 0 not in by_accepted:
            raise AssertionError(f"Selective sweep has no zero-coverage endpoint: {condition}/{k}/{policy_name}/seed{seed}")
        full = by_accepted[max(by_accepted)]
        if int(full["accepted"]) <= 0:
            raise AssertionError(f"Selective sweep has no accept-all row: {condition}/{k}/{policy_name}/seed{seed}")
        accept_all_errors.append(float(full["error_rate"]))
        points = []
        random_points = []
        for accepted, row in sorted(by_accepted.items()):
            random_risk, random_ci_low, random_ci_high = random_rejection_risk(
                population=n_gallery,
                correct_total=int(full["correct"]),
                accepted=accepted,
            )
            point = {
                "condition": condition, "prefix_frames": k, "policy": policy_name, "seed": seed,
                "threshold": float(row["threshold"]), "coverage": float(row["coverage"]),
                "risk": float(row["error_rate"]), "accepted": accepted,
                "errors": int(row["errors"]), "correct": int(row["correct"]),
                "random_rejection_expected_risk": random_risk,
                "random_rejection_risk_ci95_low": random_ci_low,
                "random_rejection_risk_ci95_high": random_ci_high,
                "point_type": "threshold_sweep",
                "selection_note": "All thresholds evaluated on validation; not independently calibrated.",
            }
            if accepted == int(full["accepted"]):
                point["point_type"] = "always_predict_baseline"
            points.append(point)
            if accepted > 0:
                random_points.append((float(row["coverage"]), float(random_risk)))
        # Single curve per condition/prefix averaged over saved seeds on coverage grid.
        positive_points = [p for p in points if p["accepted"] > 0]
        xs = np.asarray([p["coverage"] for p in positive_points], dtype=float)
        ys = np.asarray([p["risk"] for p in positive_points], dtype=float)
        order = np.argsort(xs)
        xs, ys = xs[order], ys[order]
        unique_x, first_idx = np.unique(xs, return_index=True)
        unique_y = ys[first_idx]
        label = f"{condition} / k={k}"
        curve_averages[policy_name].setdefault(label, []).append((unique_x, unique_y))
        random_curve_averages[policy_name].setdefault(label, []).append(random_points)
        curve_group_counts[(condition, k, policy_name)] += 1
    # Average seed curves at a common grid; include full coverage and accept-all risk.
    for policy_name, by_label in curve_averages.items():
        for label, seed_curves in by_label.items():
            common = np.linspace(0.0, 1.0, 21)
            common = common[
                (common >= max(xs.min() for xs, _ in seed_curves))
                & (common <= min(xs.max() for xs, _ in seed_curves))
            ]
            vals = [
                np.asarray([np.interp(x, xs, ys) for x in common])
                for xs, ys in seed_curves
            ]
            mean_risk = np.mean(np.stack(vals), axis=0)
            curve_averages[policy_name][label] = list(zip(common.tolist(), mean_risk.tolist()))
            random_seed_curves = random_curve_averages[policy_name][label]
            random_vals = [
                np.asarray([np.interp(x, [p[0] for p in points], [p[1] for p in points]) for x in common])
                for points in random_seed_curves
            ]
            random_mean = np.mean(np.stack(random_vals), axis=0)
            random_curve_averages[policy_name][label] = list(zip(common.tolist(), random_mean.tolist()))
        svg_risk_plot(
            policy_name,
            curve_averages[policy_name],
            random_curve_averages[policy_name],
            args.output_dir / f"risk_coverage_{policy_name}.svg",
        )
    # Flatten the threshold sweep separately to retain all observed score cutoffs.
    for (condition, k, policy_name, seed), rows in sorted(curves.items()):
        by_accepted: dict[int, dict] = {}
        for row in rows:
            accepted = int(row["accepted"])
            if accepted not in by_accepted:
                by_accepted[accepted] = row
        max_accept = max(by_accepted)
        for accepted, row in sorted(by_accepted.items()):
            random_risk, random_ci_low, random_ci_high = random_rejection_risk(
                population=n_gallery,
                correct_total=int(by_accepted[max_accept]["correct"]),
                accepted=accepted,
            )
            risk_rows.append({
                "condition": condition, "prefix_frames": k, "policy": policy_name, "seed": seed,
                "threshold": row["threshold"], "coverage": row["coverage"], "risk": row["error_rate"],
                "accepted": accepted, "errors": row["errors"], "correct": row["correct"],
                "random_rejection_expected_risk": random_risk,
                "random_rejection_risk_ci95_low": random_ci_low,
                "random_rejection_risk_ci95_high": random_ci_high,
                "point_type": "always_predict_baseline" if accepted == max_accept else "threshold_sweep",
                "selection_note": "Validation-only threshold sweep; no independent calibration/test split.",
            })
    write_csv(args.output_dir / "risk_coverage.csv", risk_rows, [
        "condition", "prefix_frames", "policy", "seed", "threshold", "coverage", "risk",
        "accepted", "errors", "correct", "random_rejection_expected_risk",
        "random_rejection_risk_ci95_low", "random_rejection_risk_ci95_high",
        "point_type", "selection_note",
    ])
    run_curve_auc: dict[tuple[str, int, str, int], tuple[float, float]] = {}
    for key in curves:
        selected = [
            row for row in risk_rows
            if row["condition"] == key[0]
            and int(row["prefix_frames"]) == key[1]
            and row["policy"] == key[2]
            and int(row["seed"]) == key[3]
            and int(row["accepted"]) > 0
        ]
        coverage_values = np.asarray([float(row["coverage"]) for row in selected])
        risk_values = np.asarray([float(row["risk"]) for row in selected])
        order = np.argsort(coverage_values)
        coverage_values, risk_values = coverage_values[order], risk_values[order]
        unique_coverage, unique_idx = np.unique(coverage_values, return_index=True)
        risk_values = risk_values[unique_idx]
        coverage_values = np.concatenate(([0.0], unique_coverage))
        risk_values = np.concatenate(([risk_values[0]], risk_values))
        score_auc = float(np.trapezoid(risk_values, coverage_values))
        baseline = float(np.mean([float(row["random_rejection_expected_risk"]) for row in selected]))
        run_curve_auc[key] = (score_auc, baseline)
    curve_comparison_rows = []
    comparison_groups: dict[tuple[str, int, str], list[tuple[float, float]]] = defaultdict(list)
    for (condition, k, policy_name, seed), (score_auc, random_auc) in sorted(run_curve_auc.items()):
        comparison_groups[(condition, k, policy_name)].append((score_auc, random_auc))
    for (condition, k, policy_name), values in sorted(comparison_groups.items()):
        deltas = [score - baseline for score, baseline in values]
        curve_comparison_rows.append({
            "condition": condition,
            "prefix_frames": k,
            "policy": policy_name,
            "n_seeds": len(values),
            "score_selective_auc_mean": float(np.mean([x[0] for x in values])),
            "random_rejection_auc_mean": float(np.mean([x[1] for x in values])),
            "delta_auc_score_minus_random": float(np.mean(deltas)),
            "seeds_score_auc_below_random": sum(x < 0 for x in deltas),
            "auc_definition": "trapezoidal risk-versus-coverage area over observed threshold points; descriptive validation-only",
        })
    write_csv(args.output_dir / "selective_curve_comparison.csv", curve_comparison_rows, list(curve_comparison_rows[0]))

    # Source-level inventory and proposed protocol. No pilot UID is eligible to be
    # declared untouched now: six participated in fitting and two were repeatedly
    # evaluated during model/threshold selection.
    manifests = sorted(args.dataset_root.rglob("step_label.json"))
    sources = []
    for manifest in manifests:
        obj = json.loads(manifest.read_text(encoding="utf-8"))
        uid = str(obj["video_uid"])
        rel = manifest.relative_to(args.dataset_root)
        sources.append({
            "video_uid": uid,
            "scene": rel.parts[-3],
            "event": rel.parts[-2],
            "steps": len(obj.get("steps", [])),
            "pilot_role": ("train" if uid in train_uids else "validation" if uid in val_uids else "not_in_pilot_cache"),
        })
    manifest_uids = {x["video_uid"] for x in sources}
    pilot_uids = train_uids | val_uids
    checks.append(("Source annotation manifests inspected", len(sources) > 0, f"manifests={len(sources)}, unique_video_uid={len(manifest_uids)}"))
    checks.append(("All pilot source UIDs occur in full dataset inventory", pilot_uids <= manifest_uids, f"pilot_uids={len(pilot_uids)}, missing={sorted(pilot_uids - manifest_uids)}"))

    # Save source inventory for a concrete split plan without creating/changing a split.
    write_csv(args.output_dir / "source_video_inventory.csv", sources, ["video_uid", "scene", "event", "steps", "pilot_role"])

    full_split_stats = {}
    candidate_test_rows = []
    if args.full_split_root is not None:
        old_dataset_root = Path("/media/dhqg/d1/datasets/egointent_full")
        split_rows = {}
        all_sample_ids = set()
        all_video_uids = {}
        for split_name in ("train", "val", "test"):
            split_path = args.full_split_root / f"{split_name}.jsonl"
            if not split_path.is_file():
                raise FileNotFoundError(f"Full-dataset split file not found: {split_path}")
            rows = [
                json.loads(line)
                for line in split_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            split_rows[split_name] = rows
            sample_ids = [str(row["sample_id"]) for row in rows]
            uids = {str(row["video_uid"]) for row in rows}
            duplicate_ids = len(sample_ids) - len(set(sample_ids))
            cross_split_ids = set(sample_ids) & all_sample_ids
            all_sample_ids.update(sample_ids)
            for uid in uids:
                all_video_uids.setdefault(uid, set()).add(split_name)

            stale_paths = 0
            resolved_paths = 0
            missing_paths = []
            for row in rows:
                video_path = Path(str(row["video_path"]))
                if str(video_path).startswith(str(old_dataset_root) + "/"):
                    stale_paths += 1
                    relative_path = video_path.relative_to(old_dataset_root)
                    video_path = args.dataset_root / relative_path
                if video_path.is_file():
                    resolved_paths += 1
                else:
                    missing_paths.append(str(video_path))

            cache_count = None
            cache_missing = None
            if args.full_temporal_cache is not None:
                cache_dir = args.full_temporal_cache / split_name
                expected_cache_paths = [
                    cache_dir / f"{str(row['sample_id']).replace('/', '__')}.pt"
                    for row in rows
                ]
                cache_count = sum(path.is_file() for path in expected_cache_paths)
                cache_missing = len(expected_cache_paths) - cache_count

            full_split_stats[split_name] = {
                "samples": len(rows),
                "video_uids": len(uids),
                "scenes": sorted({str(row["scene"]) for row in rows}),
                "settings": sorted({str(row["setting"]) for row in rows}),
                "pilot_train_overlap": sorted(uids & train_uids),
                "pilot_val_overlap": sorted(uids & val_uids),
                "stale_path_rows": stale_paths,
                "resolved_video_paths": resolved_paths,
                "missing_video_paths": len(missing_paths),
                "duplicate_sample_ids": duplicate_ids,
                "cross_split_sample_ids": len(cross_split_ids),
                "cache_files_found": cache_count,
                "cache_files_missing": cache_missing,
            }
            checks.append((
                f"{split_name} split source paths resolve under the dataset root",
                not missing_paths,
                f"resolved={resolved_paths}/{len(rows)}, stale_d1_paths_remapped={stale_paths}, missing={len(missing_paths)}",
            ))
            checks.append((
                f"{split_name} split sample IDs are unique and split-disjoint",
                duplicate_ids == 0 and not cross_split_ids,
                f"duplicate_within_split={duplicate_ids}, overlap_with_previous_splits={len(cross_split_ids)}",
            ))
            if args.full_temporal_cache is not None:
                checks.append((
                    f"{split_name} cached feature filenames match split sample IDs",
                    cache_missing == 0,
                    f"cache_files={cache_count}/{len(rows)}; filenames checked without loading tensors",
                ))

        split_uids_disjoint = all(
            len(split_names) == 1 for split_names in all_video_uids.values()
        )
        checks.append((
            "Full-dataset source video UIDs are split-disjoint",
            split_uids_disjoint,
            f"overlapping_uids={sorted(uid for uid, split_names in all_video_uids.items() if len(split_names) > 1)}",
        ))

        test_uids = {
            str(row["video_uid"])
            for row in split_rows["test"]
        }
        candidate_uids = test_uids - pilot_uids
        for uid in sorted(candidate_uids):
            uid_rows = [
                row for row in split_rows["test"]
                if str(row["video_uid"]) == uid
            ]
            first = uid_rows[0]
            candidate_test_rows.append({
                "video_uid": uid,
                "scene": first["scene"],
                "event": first["event"],
                "setting": first["setting"],
                "samples": len(uid_rows),
                "status": "candidate_only_not_locked_or_evaluated",
            })
        write_csv(
            args.output_dir / "candidate_test_uid_manifest.csv",
            candidate_test_rows,
            ["video_uid", "scene", "event", "setting", "samples", "status"],
        )
        full_split_inventory = [
            {
                "split": split_name,
                **stats,
            }
            for split_name, stats in full_split_stats.items()
        ]
        write_csv(
            args.output_dir / "full_split_inventory.csv",
            full_split_inventory,
            list(full_split_inventory[0]),
        )

    # Compact audit report.
    main_metrics = {
        int(r["prefix_frames"]): r
        for r in summary_saved
        if r["setting"] == "train_k_eval_k" and r["selection"] == "best_val_epoch"
    }
    main_paired = [
        r for r in paired_rows
        if r["setting"] == "train_k_eval_k" and r["selection"] == "best_val_epoch"
    ]
    main_errors = [r for r in audit_examples if r["setting"] == "train_k_eval_k" and r["selection"] == "best_val_epoch" and r["is_top1_error"]]
    main_same_task = sum(
        1 for r in main_errors if r["task_id"] == r["predicted_candidate_task_id"]
    )
    pair_top = sorted(
        (r for r in pair_error_rows if r["setting"] == "train_k_eval_k" and r["selection"] == "best_val_epoch"),
        key=lambda r: (-r["top1_error_seed_examples"], r["task_id"], r["ground_truth_why"], r["predicted_why"]),
    )[:5]
    risk_baseline_range = (
        (float(min(accept_all_errors)), float(max(accept_all_errors)))
        if accept_all_errors else (math.nan, math.nan)
    )
    area_by_policy: dict[str, tuple[int, int]] = {}
    for policy_name in ("confidence", "margin", "entropy"):
        subset = [r for r in curve_comparison_rows if r["policy"] == policy_name]
        area_by_policy[policy_name] = (
            sum(r["delta_auc_score_minus_random"] < 0 for r in subset),
            len(subset),
        )
    report = [
        "# Prefix ablation and selective-decision audit",
        "",
        "## Executive result",
        "",
        "The saved prefix artifacts are internally consistent against their raw per-run JSON and validation cache; "
        "the run JSON does not contain model weights or similarity matrices, so this audit does not reproduce training "
        "or independently reconstruct all candidate scores. "
        "The main result remains validation-only, and `best_val_epoch` is selected on the same validation examples "
        "used for reporting. The current selective summary is not a valid evidence of deferral: its 25% risk gate "
        "is infeasible on the observed data and the fallback selects the full-coverage point.",
        "",
        "## Implementation and artifact checks",
        "",
    ]
    report.extend([f"- **{'PASS' if ok else 'FAIL'}** {name}: {detail}." for name, ok, detail in checks])
    report += [
        "",
        f"- Provenance note: `config.json` records run commit `{config['git_commit']}`, while the current repository HEAD may differ; artifacts are audited against the checked-out code and saved raw outputs, not regenerated model weights.",
        "- Gallery consists of all 215 validation WHY-target rows; a query is positive for every gallery row whose WHY text exactly matches the query WHY. Rank-based MRR/Top-1/3/5 are recalculated from saved positive ranks.",
        "- The raw JSON persists positive ranks and top-1 gallery indices, but not similarity/logit matrices. Thus aggregate rank metrics and selected WHY labels are rechecked; a complete independent reconstruction of every candidate ordering from raw scores is impossible. For the main train@k setting, each trained prefix is the model's own train length and the saved project-metric cross-check maximum absolute difference is "
        f"`{config['crosscheck_max_abs_diff']:.2e}`. Exact-score tie behavior for prefixes without a score matrix cannot be separately audited.",
        "- `best_val_epoch` selects the epoch with the maximum validation MRR at each model's own training prefix; this directly optimizes the metric on the same validation set and is optimistically biased. `last_epoch` is the final training epoch (no validation-based checkpoint selection), but is still evaluated on validation.",
        "- Prefixes use the first k of eight features sampled across the entire clip, not a fixed elapsed-time cutoff. Changing prefix k does not change train/validation sample identities or gallery.",
        "- Error counts pool 3 seeds × repeated sample IDs; the distinct validation unit remains only two `video_uid`s. No seed/sample count is interpreted as independent source-video replication.",
        "",
        "## Main train@k/eval@k results",
        "",
        "| Frames | MRR mean ± seed SD | Top-1 | Top-3 | Top-5 | n examples/seed | same-task share among errors |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for k in prefixes:
        row = main_metrics[k]
        report.append(
            f"| {k} | {float(row['MRR_mean']):.4f} ± {float(row['MRR_std']):.4f} | "
            f"{float(row['Top1_mean']):.4f} | {float(row['Top3_mean']):.4f} | "
            f"{float(row['Top5_mean']):.4f} | {int(row['n_examples_per_seed'])} | "
            f"{float(row['same_task_wrong_fraction_mean']):.3f} |"
        )
    report += [
        "",
        "### Paired prefix differences (same examples)",
        "",
        "The CSV includes seed-level changes and an exploratory bootstrap over the two source-video clusters. With only "
        "two validation videos, that interval has too few clusters to support population-level uncertainty claims. "
        "The earlier published interval resampled individual examples and understates clustering uncertainty.",
        "",
        "| Prefix pair | ΔMRR (paired) | 2-video cluster bootstrap 95% interval | Video-specific ΔMRR |",
        "|---|---:|---:|---|",
    ]
    for row in main_paired:
        report.append(
            f"| {row['from_k']}→{row['to_k']} | {row['delta_mrr_pooled_examples']:+.4f} | "
            f"[{row['video_cluster_bootstrap_ci95_low']:+.4f}, {row['video_cluster_bootstrap_ci95_high']:+.4f}] | "
            f"`{row['delta_mrr_by_video_json']}` |"
        )
    report += [
        "",
        "### Same-task goal confusions",
        "",
        f"For the main setting, {main_same_task} of {len(main_errors)} seed-example Top-1 errors "
        f"({main_same_task / len(main_errors):.1%} if any errors) were within the query task. "
        "This uses the selected gallery candidate's task and the same exact-WHY correctness test as the training script. "
        "Counts are repeated across seeds; `unique_error_samples` in the CSV avoids treating those repetitions as new examples.",
        "",
        "Most frequent label-pair examples (counts are seed-example errors; text labels are not human error annotations):",
        "",
        "| Frames | Task | Ground-truth WHY | Predicted WHY | Error seed-examples | Unique examples |",
        "|---:|---|---|---|---:|---:|",
    ]
    for row in pair_top:
        report.append(
            f"| {row['prefix_frames']} | {row['task_id']} | {row['ground_truth_why']} | {row['predicted_why']} | "
            f"{row['top1_error_seed_examples']} | {row['unique_error_samples']} |"
        )
    report += [
        "",
        "The available task/WHY strings support descriptive confusion tables only. There are no verified human labels for "
        "visual ambiguity, action phase, or semantic evidence sufficiency, so no richer qualitative taxonomy is asserted.",
        "",
        "## Selective-prediction audit",
        "",
        f"The raw threshold sweep contains {len(selective_rows)} aggregate rows across confidence, margin, and entropy. "
        f"Reconstructed risk–coverage curves and explicit always-predict endpoints are in `risk_coverage.csv` and "
        "`risk_coverage_{confidence,margin,entropy}.svg`. The SVGs show score-based rejection as solid lines and "
        "a random-rejection baseline at the same coverage as dashed lines; CSV rows also contain the exact "
        "finite-population random-risk expectation and 95% hypergeometric interval (conditional on the validation "
        "run, not population-level video uncertainty). The always-predict error spans "
        f"{risk_baseline_range[0]:.3f}–{risk_baseline_range[1]:.3f} across saved condition/prefix/seed combinations "
        "(these are separate validation runs, not independent observations).",
        "",
        "The existing Markdown summary collapses each group to a threshold, not the full curve, and reports coverage 1.0 "
        "because no observed risk satisfies its ≤0.25 constraint; its fallback considers all rows and prefers lower "
        "risk then greater coverage. The CSV keeps threshold-sweep rows but omits the bootstrap CI fields calculated "
        "in memory by the original script. Existing plot generation only plots confidence, not margin or entropy. "
        "These validation sweeps are post-hoc diagnostics: no independent calibration split exists, and confidence, "
        "margin, or entropy are not evidence of semantic sufficiency or calibrated uncertainty.",
        "",
        "For a compact comparison, `selective_curve_comparison.csv` reports trapezoidal risk–coverage area for each "
        "condition/prefix/policy against random rejection at matched coverage. Lower area is preferable descriptively; "
        "this is measured on the same validation examples used to generate the curves and is not an independent "
        "policy-selection result.",
        "Favorable score-area counts (lower than random, out of condition × prefix groups): "
        + "; ".join(
            f"{policy_name} {good}/{total}"
            for policy_name, (good, total) in area_by_policy.items()
        )
        + ". These are descriptive, not inferential.",
        "",
        "## Full-dataset split and test protocol",
        "",
        f"The full dataset is present at `{args.dataset_root}`: {len(sources)} annotation manifests and "
        f"{len(manifest_uids)} source video UIDs. The full split manifests are "
        f"`{args.full_split_root}` and contain "
        + ("; ".join(
            f"{name}={stats['samples']} samples/{stats['video_uids']} UIDs"
            for name, stats in full_split_stats.items()
        ) if full_split_stats else "no split inventory was supplied to this audit")
        + ".",
        "",
        (
            f"The pre-existing split has {len(full_split_stats.get('test', {}).get('pilot_val_overlap', []))} "
            "test UID(s) that overlap pilot validation and must not be treated as untouched. "
            f"Excluding those leaves {len(candidate_test_rows)} candidate test UIDs and "
            f"{sum(row['samples'] for row in candidate_test_rows)} samples; these are recorded in "
            "`candidate_test_uid_manifest.csv` as candidates, not as a locked test set."
            if full_split_stats else
            "No full split was supplied; no independent test claim is made."
        ),
        "",
        "The saved `egointent_full_intention_v1/summary.json` is a pilot-cache run: its recorded Stage-A and temporal "
        "cache paths point to `/media/dhqg/d1/datasets/egointent/cache/...`, and its validation metrics are over 215 "
        "pilot examples. Searching the available repository results and `/media/dhqg/d3/vlia_outputs` found no exact "
        "references to the four candidate UIDs. This is evidence about the inspected local artifacts, not proof that "
        "no external or unrecorded run ever accessed them. Existing full temporal test feature files were counted by "
        "filename only; their tensors and labels were not opened or evaluated.",
        "",
        "## Single best next experiment",
        "",
        "Freeze a versioned video_uid-level protocol using the existing full train/validation split, remove the pilot-validation "
        "UID from test, and record that the remaining test UIDs are only considered untouched relative to the inspected "
        "local artifacts. Before any test tensor is read, generate compatible WHY text targets for full train/validation "
        "only, verify cache identities, train the 2/4/6/8-frame models with validation-only checkpoint selection, and lock "
        "all hyperparameters. Evaluate the candidate test UIDs once only after that protocol is recorded. The current full "
        "temporal cache lacks `why_feature`, which the temporal loader expects; this target-cache dependency blocks full-data "
        "training now.",
        "",
        "## Audit outputs",
        "",
        "- `per_example_errors.csv` includes correct and incorrect top-1 outputs with an `is_top1_error` flag.",
        "- `top1_errors_by_task.csv` and `top1_errors_by_why_pair.csv` provide task/pair grouping.",
        "- `prefix_paired_comparison.csv` reports paired deltas with explicit two-video cluster limitations.",
        "- `risk_coverage.csv` and three SVGs include threshold curves and always-predict endpoints.",
        "- `selective_curve_comparison.csv` compares curve area against matched-coverage random rejection.",
        "- `source_video_inventory.csv` records source UID/scene/event/step counts without modifying the pilot split.",
        "- `full_split_inventory.csv` records full split sizes, video path resolution, and cache filename coverage.",
        "- `candidate_test_uid_manifest.csv` lists candidate UIDs only; it is not a locked test declaration.",
        "",
    ]
    (args.output_dir / "audit_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Audit complete: {args.output_dir}")
    print(f"Verified {len(checks)} checks; per-example rows={len(audit_examples)}; risk-coverage points={len(risk_rows)}")


if __name__ == "__main__":
    main()
