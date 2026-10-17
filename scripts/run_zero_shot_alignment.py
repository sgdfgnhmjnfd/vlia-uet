#!/usr/bin/env python3
"""E1 zero-shot SigLIP image-text alignment on train/validation only."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_scene_cv as CV  # noqa: E402
import run_prefix_length_ablation as P  # noqa: E402
from hoi_utils import crop_from_box_or_full, sample_exact_video_frames  # noqa: E402


MODEL_NAME = "google/siglip-base-patch16-224"
TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PREFIXES = (2, 4, 6, 8)
BRANCHES = ("full", "hand", "center")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "results/hoi_detection_v1/manifest.jsonl",
    )
    parser.add_argument(
        "--detections",
        type=Path,
        default=ROOT / "results/hoi_detection_v1/detections.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/e1_zero_shot_v1",
    )
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=Path("/media/dhqg/d3/vlia_outputs/e1"),
    )
    parser.add_argument("--hand-model", type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def clean_kway(
    video_uids: list[str],
    texts: list[str],
    sample_ids: list[str],
    mini_embeddings: np.ndarray,
    K: int = 5,
    threshold: float = 0.8,
) -> np.ndarray:
    """Build deterministic same-video distractors, semantically separated."""
    n = len(texts)
    out = np.full((n, K), -1, dtype=np.int64)
    by_video: dict[str, list[int]] = defaultdict(list)
    for i, video in enumerate(video_uids):
        by_video[video].append(i)
    for i, video in enumerate(video_uids):
        pool: list[int] = []
        seen_texts = {texts[i]}
        for j in by_video[video]:
            if texts[j] in seen_texts:
                continue
            seen_texts.add(texts[j])
            if float(mini_embeddings[i] @ mini_embeddings[j]) < threshold:
                pool.append(j)
        rng = np.random.RandomState(
            zlib.crc32(f"{sample_ids[i]}|clean{K}".encode()) & 0xFFFFFFFF
        )
        order = rng.permutation(pool).tolist()
        selected: list[int] = []
        for candidate in order:
            if all(
                float(mini_embeddings[candidate] @ mini_embeddings[prior]) < threshold
                for prior in selected
            ):
                selected.append(candidate)
            if len(selected) == K - 1:
                break
        if len(selected) == K - 1:
            out[i] = [i, *selected]
    return out


def paraphrase_positives(
    fold_indices: np.ndarray,
    texts: list[str],
    mini_embeddings: np.ndarray,
    threshold: float = 0.9,
) -> np.ndarray:
    fold_texts = np.asarray(texts)[fold_indices]
    sim = mini_embeddings[fold_indices] @ mini_embeddings[fold_indices].T
    return (sim >= threshold) | (fold_texts[:, None] == fold_texts[None, :])


def softmax_stats(scores: np.ndarray, temperature: float = 0.07) -> tuple[float, float, float]:
    scaled = scores.astype(np.float64) / temperature
    scaled -= scaled.max()
    probs = np.exp(scaled)
    probs /= probs.sum()
    entropy = -float(np.sum(probs * np.log(np.maximum(probs, 1e-15)))) / np.log(len(probs))
    ordered = np.sort(scores)
    return float(probs.max()), float(ordered[-1] - ordered[-2]), entropy


def clean_candidates_selftest() -> None:
    assert callable(CV.embed_texts)
    texts = ["a", "b", "c", "d", "e", "f"]
    vecs = np.eye(6, dtype=np.float32)
    cands = clean_kway(["v"] * 6, texts, [f"s{i}" for i in range(6)], vecs)
    assert cands.shape == (6, 5)
    assert np.all(cands >= 0)
    assert all(len(set(texts[j] for j in row)) == 5 for row in cands)
    positives = paraphrase_positives(np.arange(6), texts, vecs)
    assert np.array_equal(np.diag(positives), np.ones(6, dtype=bool))
    chance = [P._chance_one(6, int(mask.sum()))[0] for mask in positives]
    assert len(chance) == 6 and all(0 < x <= 1 for x in chance)
    point, lo, hi = CV.video_bootstrap(
        np.repeat(np.arange(3), 2), np.ones(6), np.ones(6), B=250
    )
    assert point == lo == hi == 1.0
    assert softmax_stats(np.array([0.3, 0.2, 0.1]))[0] > 0
    print("selftest: PASS (clean K-way, paraphrase positives/chance, video bootstrap)")


def load_data(manifest_path: Path, detections_path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[int, dict[str, Any]]]]:
    rows = read_jsonl(manifest_path)
    if not rows:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    if any(row.get("split") not in ("train", "val") for row in rows):
        raise ValueError("Manifest must contain train/val records only.")
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("Manifest sample_id values must be unique.")
    detection_rows = read_jsonl(detections_path)
    by_sample: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in detection_rows:
        if row.get("split") not in ("train", "val"):
            raise ValueError(f"Disallowed detection split: {row.get('split')}")
        by_sample[row["sample_id"]][int(row["frame_index"])] = row
    expected_ids = {row["sample_id"] for row in rows}
    if set(by_sample) != expected_ids:
        raise ValueError(
            "Detection sample identities differ from manifest: "
            f"missing={len(expected_ids - set(by_sample))}, "
            f"extra={len(set(by_sample) - expected_ids)}"
        )
    if any(set(by_sample[sid]) != set(range(8)) for sid in expected_ids):
        raise ValueError("Each manifest sample must have exactly 8 frame detections.")
    return rows, by_sample


def gate_verdict(by_sample: dict[str, dict[int, dict[str, Any]]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_scene: dict[str, list[bool]] = defaultdict(list)
    all_flags = []
    sample_scene = {row["sample_id"]: row["scene"] for row in rows}
    fallback_frames = defaultdict(int)
    for sample_id, detections in by_sample.items():
        for frame_index in range(8):
            detected = bool(detections[frame_index]["detected"])
            all_flags.append(detected)
            by_scene[sample_scene[sample_id]].append(detected)
            if not detected:
                fallback_frames["hand"] += 1
                fallback_frames["center"] += 1
    rates = {scene: float(np.mean(flags)) for scene, flags in by_scene.items()}
    overall = float(np.mean(all_flags))
    passed = overall >= 0.60 and all(rate >= 0.30 for rate in rates.values())
    total = len(all_flags)
    return {
        "pass": passed,
        "overall_rate": overall,
        "scene_rates": rates,
        "fallback_frames": {
            branch: count / total for branch, count in fallback_frames.items()
        },
        "total_frames": total,
    }


def _model_features(model: Any, processor: Any, frames: list[np.ndarray], device: Any, batch_size: int) -> np.ndarray:
    import torch
    import torch.nn.functional as F
    from PIL import Image

    outputs = []
    for start in range(0, len(frames), batch_size):
        batch = [Image.fromarray(frame) for frame in frames[start : start + batch_size]]
        inputs = processor(images=batch, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        with torch.inference_mode():
            features = model.get_image_features(pixel_values=pixel_values)
            if hasattr(features, "pooler_output"):
                features = features.pooler_output
            features = F.normalize(features.float(), dim=-1)
        outputs.append(features.cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float32)


def embed_unique_texts(model: Any, processor: Any, texts: list[str], device: Any, batch_size: int) -> tuple[np.ndarray, int]:
    import torch
    import torch.nn.functional as F

    unique = sorted(set(texts))
    tokenizer = processor.tokenizer
    untruncated = tokenizer(
        unique,
        padding=False,
        truncation=False,
        add_special_tokens=True,
    )["input_ids"]
    truncation_count = sum(len(ids) > 64 for ids in untruncated)
    results = []
    for start in range(0, len(unique), batch_size):
        batch = unique[start : start + batch_size]
        tokens = processor(
            text=batch,
            padding="max_length",
            max_length=64,
            truncation=True,
            return_tensors="pt",
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        with torch.inference_mode():
            features = model.get_text_features(**tokens)
            if hasattr(features, "pooler_output"):
                features = features.pooler_output
            features = F.normalize(features.float(), dim=-1)
        results.append(features.cpu().numpy())
    all_embeddings = np.concatenate(results, axis=0)
    table = {text: all_embeddings[i] for i, text in enumerate(unique)}
    return np.stack([table[text] for text in texts]), truncation_count


def load_siglip(device: Any) -> tuple[Any, Any, dict[str, str]]:
    import torch
    import transformers
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
    versions = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "model": MODEL_NAME,
    }
    return model, processor, versions


def compute_frame_embeddings(
    rows: list[dict[str, Any]],
    by_sample: dict[str, dict[int, dict[str, Any]]],
    branches: list[str],
    model: Any,
    processor: Any,
    device: Any,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    pending: dict[str, list[np.ndarray]] = {branch: [] for branch in branches}
    chunks: dict[str, list[np.ndarray]] = {branch: [] for branch in branches}
    fallback = {branch: 0 for branch in branches if branch != "full"}
    total_frames = len(rows) * 8

    def flush(branch: str) -> None:
        if pending[branch]:
            chunks[branch].append(
                _model_features(
                    model, processor, pending[branch], device, batch_size
                )
            )
            pending[branch].clear()

    for record_index, record in enumerate(rows, start=1):
        frames = sample_exact_video_frames(record["video_path"])
        for frame_index, rgb in enumerate(frames):
            box = by_sample[record["sample_id"]][frame_index]["box"]
            if "full" in branches:
                pending["full"].append(rgb)
                if len(pending["full"]) >= batch_size:
                    flush("full")
            if box is not None:
                x1, y1, x2, y2 = (int(v) for v in box)
                box_tuple = (x1, y1, x2, y2)
            else:
                box_tuple = None
                for branch in fallback:
                    fallback[branch] += 1
            if "hand" in branches or "center" in branches:
                hand_crop, center_crop = crop_from_box_or_full(rgb, box_tuple)
                if "hand" in branches:
                    pending["hand"].append(hand_crop)
                    if len(pending["hand"]) >= batch_size:
                        flush("hand")
                if "center" in branches:
                    pending["center"].append(center_crop)
                    if len(pending["center"]) >= batch_size:
                        flush("center")
        if record_index % 100 == 0:
            print(
                f"encoded source samples {record_index}/{len(rows)} "
                f"({record_index * 8}/{total_frames} frames)",
                flush=True,
            )
    for branch in branches:
        flush(branch)
    vectors = {
        branch: np.concatenate(chunks[branch], axis=0).reshape(len(rows), 8, -1)
        for branch in branches
    }
    for branch, values in vectors.items():
        if values.shape[:2] != (len(rows), 8):
            raise AssertionError(
                f"{branch} frame embedding shape {values.shape} does not match samples x 8"
            )
        if not np.isfinite(values).all():
            raise AssertionError(f"{branch} contains non-finite frame embeddings.")
    return vectors, fallback


def _kway_accuracy(
    embeddings: np.ndarray,
    text_targets: np.ndarray,
    candidates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    valid = (candidates >= 0).all(axis=1)
    safe = np.where(candidates >= 0, candidates, 0)
    scores = np.einsum("nd,nkd->nk", embeddings, text_targets[safe])
    hits = (scores[:, :1] > scores[:, 1:]).all(axis=1)
    return hits, valid


def evaluate(
    rows: list[dict[str, Any]],
    frame_vectors: dict[str, np.ndarray],
    why_targets: np.ndarray,
    mini_targets: np.ndarray,
    device: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[int, dict[str, np.ndarray]]]]:
    n = len(rows)
    vids = np.asarray([row["video_uid"] for row in rows])
    scenes = np.asarray([row["scene"] for row in rows])
    texts = [row["why"] for row in rows]
    sample_ids = [row["sample_id"] for row in rows]
    text_vocab = {text: i for i, text in enumerate(sorted(set(texts)))}
    text_id = np.asarray([text_vocab[text] for text in texts])
    fold_map = CV.assign_folds(scenes.tolist(), 8)
    fold_of = np.asarray([fold_map[scene] for scene in scenes])
    if not all(len(set(fold_of[vids == video])) == 1 for video in np.unique(vids)):
        raise AssertionError("A video crosses scene-grouped folds.")
    cands5 = CV.build_kway(vids.tolist(), texts, sample_ids, 5)
    cands10 = CV.build_kway(vids.tolist(), texts, sample_ids, 10)
    clean5 = clean_kway(vids.tolist(), texts, sample_ids, mini_targets)
    exact_chance = CV.gallery_chance(text_id, fold_of)

    summary_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    per_example: dict[str, dict[int, dict[str, np.ndarray]]] = defaultdict(dict)
    for branch, frames in frame_vectors.items():
        for k in PREFIXES:
            query_embeddings = frames[:, :k].mean(axis=1)
            query_embeddings /= np.maximum(
                np.linalg.norm(query_embeddings, axis=1, keepdims=True), 1e-12
            )
            ranks = np.zeros(n, dtype=np.int64)
            k5_hits = np.zeros(n, dtype=bool)
            k10_hits = np.zeros(n, dtype=bool)
            clean_hits = np.zeros(n, dtype=bool)
            clean_valid = np.zeros(n, dtype=bool)
            para_rr = np.zeros(n, dtype=np.float64)
            para_chance = np.zeros(n, dtype=np.float64)
            for fold in sorted(set(fold_of.tolist())):
                idx = np.where(fold_of == fold)[0]
                similarities = query_embeddings[idx] @ why_targets[idx].T
                same_text = text_id[idx, None] == text_id[None, idx]
                ranks[idx] = P.first_positive_rank(similarities, same_text)
                local5, _ = _kway_accuracy(
                    query_embeddings[idx], why_targets, cands5[idx]
                )
                local10, _ = _kway_accuracy(
                    query_embeddings[idx], why_targets, cands10[idx]
                )
                k5_hits[idx] = local5
                k10_hits[idx] = local10
                local_clean, valid_clean = _kway_accuracy(
                    query_embeddings[idx], why_targets, clean5[idx]
                )
                clean_hits[idx] = local_clean
                clean_valid[idx] = valid_clean
                para_pos = paraphrase_positives(idx, texts, mini_targets)
                para_ranks = P.first_positive_rank(similarities, para_pos)
                para_rr[idx] = 1.0 / para_ranks
                n_gallery = len(idx)
                for local_i in range(len(idx)):
                    para_chance[idx[local_i]] = P._chance_one(
                        n_gallery, int(para_pos[local_i].sum())
                    )[0]

            exact_rr = 1.0 / ranks
            metrics = {
                "mrr": exact_rr,
                "top1": (ranks <= 1).astype(float),
                "top3": (ranks <= 3).astype(float),
                "kway5": k5_hits.astype(float),
                "kway10": k10_hits.astype(float),
                "clean5": clean_hits.astype(float),
                "para_rr": para_rr,
            }
            per_example[branch][k] = {
                **metrics,
                "clean_valid": clean_valid,
                "para_chance": para_chance,
            }
            row: dict[str, Any] = {
                "branch": branch,
                "prefix_frames": k,
                "n_examples": n,
                "n_videos": int(len(np.unique(vids))),
                "MRR": float(exact_rr.mean()),
                "MRR_chance": float(exact_chance["MRR"].mean()),
                "MRR_lift": float(exact_rr.mean() / exact_chance["MRR"].mean()),
                "Top1": float((ranks <= 1).mean()),
                "Top3": float((ranks <= 3).mean()),
                "5way_acc": float(k5_hits.mean()),
                "5way_lift": float(k5_hits.mean() / 0.2),
                "10way_acc": float(k10_hits.mean()),
                "10way_lift": float(k10_hits.mean() / 0.1),
                "clean5_n_valid": int(clean_valid.sum()),
                "clean5_excluded": int(n - clean_valid.sum()),
                "clean5_acc": (
                    float(clean_hits[clean_valid].mean())
                    if clean_valid.any()
                    else None
                ),
                "para_MRR": float(para_rr.mean()),
                "para_MRR_chance": float(para_chance.mean()),
                "para_MRR_lift": float(para_rr.mean() / para_chance.mean()),
            }
            for metric, values, den in (
                ("MRR", exact_rr, exact_chance["MRR"]),
                ("5way", k5_hits.astype(float), np.full(n, 0.2)),
                ("10way", k10_hits.astype(float), np.full(n, 0.1)),
                ("clean5", clean_hits[clean_valid].astype(float), np.full(int(clean_valid.sum()), 0.2)),
                ("para_MRR", para_rr, para_chance),
            ):
                cluster = vids if metric != "clean5" else vids[clean_valid]
                if not len(values):
                    point = lo = hi = float("nan")
                else:
                    point, lo, hi = CV.video_bootstrap(cluster, values, den)
                row[f"{metric}_lift_boot"] = point
                row[f"{metric}_lift_ci_lo"] = lo
                row[f"{metric}_lift_ci_hi"] = hi
            summary_rows.append(row)

    for branch in frame_vectors:
        for ka, kb in ((2, 4), (4, 6), (6, 8), (2, 8)):
            for metric in ("mrr", "kway5"):
                delta = (
                    per_example[branch][kb][metric]
                    - per_example[branch][ka][metric]
                )
                point, lo, hi = CV.video_bootstrap(vids, delta, np.ones(n))
                paired_rows.append(
                    {
                        "comparison": "prefix",
                        "metric": metric,
                        "branch_a": branch,
                        "branch_b": branch,
                        "from_k": ka,
                        "to_k": kb,
                        "delta": point,
                        "ci95_lo": lo,
                        "ci95_hi": hi,
                        "n_videos": int(len(np.unique(vids))),
                    }
                )
    for branch_a, branch_b in (("hand", "full"), ("hand", "center"), ("center", "full")):
        if branch_a not in per_example or branch_b not in per_example:
            continue
        for k in PREFIXES:
            for metric in ("mrr", "kway5"):
                delta = per_example[branch_a][k][metric] - per_example[branch_b][k][metric]
                point, lo, hi = CV.video_bootstrap(vids, delta, np.ones(n))
                paired_rows.append(
                    {
                        "comparison": "branch",
                        "metric": metric,
                        "branch_a": branch_a,
                        "branch_b": branch_b,
                        "from_k": k,
                        "to_k": k,
                        "delta": point,
                        "ci95_lo": lo,
                        "ci95_hi": hi,
                        "n_videos": int(len(np.unique(vids))),
                    }
                )
    return summary_rows, paired_rows, per_example


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_plot(out: Path, summary: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for branch in sorted({row["branch"] for row in summary}):
        rows = sorted(
            (row for row in summary if row["branch"] == branch),
            key=lambda row: row["prefix_frames"],
        )
        y = np.asarray([row["MRR_lift_boot"] for row in rows])
        lo = np.asarray([row["MRR_lift_ci_lo"] for row in rows])
        hi = np.asarray([row["MRR_lift_ci_hi"] for row in rows])
        ax.errorbar(
            [row["prefix_frames"] for row in rows],
            y,
            yerr=[y - lo, hi - y],
            marker="o",
            capsize=3,
            label=branch,
        )
    ax.axhline(1.0, color="gray", linestyle=":", label="chance")
    ax.set_xticks(PREFIXES)
    ax.set_xlabel("Observed frames")
    ax.set_ylabel("MRR lift over exact gallery chance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "lift_vs_frames.png", dpi=160)
    plt.close(fig)


def write_summary_md(path: Path, rows: list[dict[str, Any]], paired: list[dict[str, Any]]) -> None:
    lines = [
        "# E1 zero-shot SigLIP alignment",
        "",
        "Train/validation-only, scene-grouped 8-fold evaluation. MRR lift "
        "intervals bootstrap source videos; these are descriptive.",
        "",
        "| Branch | k | MRR | MRR chance | MRR lift [95% CI] | Top-1 | Top-3 | 5-way acc. | 5-way lift [95% CI] | 10-way acc. |",
        "|---|---:|---:|---:|---|---:|---:|---:|---|---:|",
    ]
    for row in sorted(rows, key=lambda item: (item["branch"], item["prefix_frames"])):
        lines.append(
            f"| {row['branch']} | {row['prefix_frames']} | {row['MRR']:.4f} | "
            f"{row['MRR_chance']:.5f} | {row['MRR_lift_boot']:.2f} "
            f"[{row['MRR_lift_ci_lo']:.2f}, {row['MRR_lift_ci_hi']:.2f}] | "
            f"{row['Top1']:.4f} | {row['Top3']:.4f} | {row['5way_acc']:.4f} | "
            f"{row['5way_lift_boot']:.2f} [{row['5way_lift_ci_lo']:.2f}, "
            f"{row['5way_lift_ci_hi']:.2f}] | {row['10way_acc']:.4f} |"
        )
    lines += [
        "",
        "## Paired differences",
        "",
        "| Comparison | Metric | From -> to | Delta [95% CI] |",
        "|---|---|---|---|",
    ]
    for row in paired:
        lines.append(
            f"| {row['comparison']} {row['branch_a']} - {row['branch_b']} | "
            f"{row['metric']} | {row['from_k']} -> {row['to_k']} | "
            f"{row['delta']:+.5f} [{row['ci95_lo']:+.5f}, {row['ci95_hi']:+.5f}] |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_interpretation(
    path: Path,
    rows: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    gate: dict[str, Any],
    fallback_rates: dict[str, float],
    truncation_count: int,
    n_videos: int,
    n_scenes: int,
) -> None:
    full = sorted(
        (row for row in rows if row["branch"] == "full"),
        key=lambda row: row["prefix_frames"],
    )
    mrr_pass = [
        row["prefix_frames"]
        for row in full
        if row["MRR_lift_ci_lo"] > 1.0
    ]
    five_lift_pass = [
        row["prefix_frames"]
        for row in full
        if row["5way_lift_ci_lo"] > 1.0
    ]
    five_acc_pass = [
        row["prefix_frames"]
        for row in full
        if row["5way_acc"] >= 0.30
    ]
    g1_pass = (
        len(mrr_pass) >= 2
        and len(five_lift_pass) >= 2
        and len(five_acc_pass) >= 2
    )
    lines = [
        "# E1 zero-shot interpretation",
        "",
        "All numbers below are computed from the saved E1 CSVs. The evaluation "
        "uses train/validation only and is not an untouched-test result.",
        "",
        "## Main table",
        "",
        "| Branch | k | MRR | Chance | MRR lift [95% CI] | Top-1 | Top-3 | 5-way acc. | 5-way lift [95% CI] | 10-way acc. |",
        "|---|---:|---:|---:|---|---:|---:|---:|---|---:|",
    ]
    for row in sorted(rows, key=lambda item: (item["branch"], item["prefix_frames"])):
        lines.append(
            f"| {row['branch']} | {row['prefix_frames']} | {row['MRR']:.4f} | "
            f"{row['MRR_chance']:.5f} | {row['MRR_lift_boot']:.2f} "
            f"[{row['MRR_lift_ci_lo']:.2f}, {row['MRR_lift_ci_hi']:.2f}] | "
            f"{row['Top1']:.4f} | {row['Top3']:.4f} | {row['5way_acc']:.4f} | "
            f"{row['5way_lift_boot']:.2f} [{row['5way_lift_ci_lo']:.2f}, "
            f"{row['5way_lift_ci_hi']:.2f}] | {row['10way_acc']:.4f} |"
        )
    lines += [
        "",
        "## Preregistered decisions",
        "",
        f"- **G1 (full-frame branch): {'PASS' if g1_pass else 'FAIL'}.** "
        f"MRR-lift lower CI > 1 at {len(mrr_pass)}/4 prefixes "
        f"({mrr_pass}); 5-way-lift lower CI > 1 at {len(five_lift_pass)}/4 "
        f"({five_lift_pass}); 5-way accuracy >= 0.30 at {len(five_acc_pass)}/4 "
        f"({five_acc_pass}).",
    ]
    if gate["pass"]:
        g2_details = []
        g2_pass = True
        for branch_a, branch_b in (("hand", "full"), ("hand", "center")):
            checks = {}
            for metric in ("mrr", "kway5"):
                selected = [
                    row
                    for row in paired
                    if row["comparison"] == "branch"
                    and row["branch_a"] == branch_a
                    and row["branch_b"] == branch_b
                    and row["metric"] == metric
                ]
                passing = [
                    row["from_k"] for row in selected if row["ci95_lo"] > 0
                ]
                checks[metric] = passing
                if len(passing) < 2:
                    g2_pass = False
            g2_details.append(
                f"{branch_a}-{branch_b}: "
                + ", ".join(
                    f"{metric} lower CI > 0 at {len(prefixes)}/4 {prefixes}"
                    for metric, prefixes in checks.items()
                )
            )
        lines.append(
            f"- **G2 (HOI helps): {'PASS' if g2_pass else 'FAIL'}.** "
            + "; ".join(g2_details)
            + "."
        )
    else:
        lines.append(
            "- **G2 (HOI helps): SKIPPED.** The preregistered detection gate "
            "failed; hand/center branches were not evaluated."
        )
    lines += [
        "",
        "## Detection gate and branch fallbacks",
        "",
        f"- Overall hand detection rate: {gate['overall_rate']:.4f}.",
        f"- Gate: **{'PASS' if gate['pass'] else 'FAIL'}** "
        "(overall >= 0.60 and every scene >= 0.30).",
    ]
    for scene, rate in sorted(gate["scene_rates"].items()):
        lines.append(f"- Scene `{scene}` detection rate: {rate:.4f}.")
    for branch, rate in sorted(fallback_rates.items()):
        lines.append(f"- `{branch}` frames falling back to full frame: {rate:.4f}.")
    lines += [
        "",
        "## Secondary, preregistered diagnostics",
        "",
        "These variants do not affect G1/G2 decisions.",
        "",
        "| Branch | k | Clean 5-way acc. | Valid | Excluded | Paraphrase-aware MRR | Chance | Lift |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: (item["branch"], item["prefix_frames"])):
        clean = (
            "NA" if row["clean5_acc"] is None else f"{row['clean5_acc']:.4f}"
        )
        lines.append(
            f"| {row['branch']} | {row['prefix_frames']} | {clean} | "
            f"{row['clean5_n_valid']} | {row['clean5_excluded']} | "
            f"{row['para_MRR']:.4f} | {row['para_MRR_chance']:.5f} | "
            f"{row['para_MRR_lift_boot']:.3f} "
            f"[{row['para_MRR_lift_ci_lo']:.3f}, {row['para_MRR_lift_ci_hi']:.3f}] |"
        )
    lines += [
        "",
        "## Limitations",
        "",
        f"- The evaluated set contains {n_videos} source videos and {n_scenes} scenes.",
        "- Exact-text positives are used for the primary MRR; paraphrases may be counted as errors.",
        f"- The SigLIP tokenizer truncated {truncation_count} unique WHY strings at max_length=64.",
        "- Same-video K-way distractors may still be semantic paraphrases; the clean 5-way variant is secondary.",
        "- No untouched test set was evaluated. The results do not establish generalization, significance, or superiority.",
        "- Any HOI crops are fixed geometric proxies based on hand landmarks and are not an object detector.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    import torch

    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.output_dir}")
    image_cache = args.embedding_cache / "siglip_frame_embeddings_e1_v1.npz"
    if not args.smoke and image_cache.exists():
        raise FileExistsError(
            f"Refusing to overwrite embedding cache: {image_cache}"
        )
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    rows, by_sample = load_data(args.manifest, args.detections)
    if args.smoke:
        rows = rows[:60]
        by_sample = {row["sample_id"]: by_sample[row["sample_id"]] for row in rows}
    gate = gate_verdict(by_sample, rows)
    branches = ["full"]
    if gate["pass"]:
        branches = ["full", "hand", "center"]
    if args.smoke:
        branches = ["full"]
    device = torch.device(
        args.device
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model, processor, versions = load_siglip(device)
    frame_vectors, fallback = compute_frame_embeddings(
        rows, by_sample, branches, model, processor, device, args.batch_size
    )
    texts = [row["why"] for row in rows]
    why_targets, truncation_count = embed_unique_texts(
        model, processor, texts, device, args.batch_size
    )
    args.embedding_cache.mkdir(parents=True, exist_ok=True)
    mini = CV.embed_texts(
        texts,
        TEXT_MODEL,
        device,
        args.embedding_cache / "text_embeddings.pt",
    ).astype(np.float32)
    mini /= np.maximum(np.linalg.norm(mini, axis=1, keepdims=True), 1e-12)
    summary, paired, per_example = evaluate(
        rows, frame_vectors, why_targets.astype(np.float32), mini, device
    )
    if args.smoke:
        if any(row["split"] not in ("train", "val") for row in rows):
            raise AssertionError("Smoke evaluation touched a disallowed split.")
        probe = frame_vectors["full"][:, :2].mean(axis=1)
        probe /= np.maximum(np.linalg.norm(probe, axis=1, keepdims=True), 1e-12)
        cosine = probe @ why_targets.astype(np.float32).T
        if not np.isfinite(cosine).all():
            raise AssertionError("Non-finite full-branch cosine scores.")
        if args.output_dir != Path("/tmp/e1_zero_shot_smoke"):
            raise ValueError("Smoke must use /tmp/e1_zero_shot_smoke.")
        args.output_dir.mkdir(parents=True)
        np.savez_compressed(
            args.output_dir / "smoke_embeddings.npz",
            full=frame_vectors["full"].astype(np.float16),
        )
        (args.output_dir / "smoke_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(
            f"smoke PASS: {len(rows)} train/val samples, embeddings "
            f"{frame_vectors['full'].shape}, ordered frames 1..8, cosine "
            f"{cosine.shape} finite; no test paths loaded"
        )
        return 0

    out = args.output_dir
    out.mkdir(parents=True)
    np.savez_compressed(
        image_cache,
        sample_ids=np.asarray([row["sample_id"] for row in rows]),
        branches=np.asarray(branches),
        **{
            branch: values.astype(np.float16)
            for branch, values in frame_vectors.items()
        },
    )
    fallback_rates = {
        branch: float(count / (len(rows) * 8))
        for branch, count in fallback.items()
    }
    write_csv(out / "summary.csv", summary)
    write_csv(out / "paired_differences.csv", paired)
    write_plot(out, summary)
    (out / "summary.json").write_text(
        json.dumps(
            {
                "summary": summary,
                "paired_differences": paired,
                "gate": gate,
                "fallback_rates": fallback_rates,
                "truncated_unique_why_count": truncation_count,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_summary_md(out / "summary.md", summary, paired)

    ids = [row["sample_id"] for row in rows]
    selected_videos = []
    seen_scenes = set()
    for video in sorted({row["video_uid"] for row in rows}):
        scene = next(row["scene"] for row in rows if row["video_uid"] == video)
        if scene not in seen_scenes:
            selected_videos.append(video)
            seen_scenes.add(scene)
        if len(selected_videos) == 6:
            break
    viz_samples = []
    per_video: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        if row["video_uid"] in selected_videos:
            per_video[row["video_uid"]].append(i)
    canonical_kway5 = CV.build_kway(
        [item["video_uid"] for item in rows], texts, ids, 5
    )
    for video in selected_videos:
        for i in sorted(per_video[video], key=lambda i: rows[i]["sample_id"])[:4]:
            candidate_indices = canonical_kway5[i]
            if np.any(candidate_indices < 0):
                continue
            sample_scores = {}
            for k in PREFIXES:
                query = frame_vectors["full"][i, :k].mean(axis=0)
                query /= max(float(np.linalg.norm(query)), 1e-12)
                cosine = (why_targets[candidate_indices] @ query).astype(float)
                order = np.argsort(-cosine)
                ranked = [
                    {
                        "text": texts[int(candidate_indices[j])],
                        "cosine": float(cosine[j]),
                    }
                    for j in order
                ]
                conf, margin, entropy = softmax_stats(cosine)
                sample_scores[str(k)] = {
                    "candidates": ranked,
                    "confidence": conf,
                    "margin": margin,
                    "normalized_entropy": entropy,
                }
            viz_samples.append(
                {
                    "sample_id": rows[i]["sample_id"],
                    "video_uid": video,
                    "scene": rows[i]["scene"],
                    "scores_by_k": sample_scores,
                }
            )
    expected_viz_samples = sum(
        min(4, len(per_video[video])) for video in selected_videos
    )
    if len(viz_samples) != expected_viz_samples:
        raise AssertionError(
            f"Expected {expected_viz_samples} score records, got {len(viz_samples)}"
        )
    (out / "viz_scores.json").write_text(
        json.dumps({"samples": viz_samples}, indent=2), encoding="utf-8"
    )
    config = {
        "args": {key: str(value) for key, value in vars(args).items()},
        "git_commit": P.git_commit(),
        "model": MODEL_NAME,
        "text_model": TEXT_MODEL,
        "versions": versions
        | {
            "mediapipe": json.loads(
                (args.detections.parent / "detector_metadata.json").read_text(
                    encoding="utf-8"
                )
            )["mediapipe_version"]
        },
        "n_samples": len(rows),
        "n_videos": len({row["video_uid"] for row in rows}),
        "n_scenes": len({row["scene"] for row in rows}),
        "splits_used": sorted({row["split"] for row in rows}),
        "branches": branches,
        "gate": gate,
        "fallback_rates": fallback_rates,
        "siglip_unique_why_truncated": truncation_count,
        "siglip_frame_embedding_cache": str(image_cache),
        "selected_viz_videos": selected_videos,
        "viz_samples": len(viz_samples),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    write_interpretation(
        out / "INTERPRETATION.md",
        summary,
        paired,
        gate,
        fallback_rates,
        truncation_count,
        len({row["video_uid"] for row in rows}),
        len({row["scene"] for row in rows}),
    )
    print(f"Saved E1 results to {out}")
    return 0


def main() -> int:
    args = parse_args()
    if args.selftest:
        clean_candidates_selftest()
        return 0
    if args.smoke:
        if args.output_dir == ROOT / "results/e1_zero_shot_v1":
            args.output_dir = Path("/tmp/e1_zero_shot_smoke")
        if args.output_dir.exists():
            raise FileExistsError(f"Refusing to overwrite smoke output: {args.output_dir}")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
