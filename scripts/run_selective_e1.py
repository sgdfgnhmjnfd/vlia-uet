#!/usr/bin/env python3
"""Selective prediction on cached SigLIP E1 scores."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_scene_cv as CV

PREFIXES = (2, 4, 6, 8)
KWAY_K = 5


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_manifest(path: Path) -> list[dict]:
    rows = read_jsonl(path)
    if any(r.get("split") not in ("train", "val") for r in rows):
        raise ValueError("Manifest contains disallowed split(s).")
    return rows


def load_cache(cache_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(cache_path) as obj:
        sample_ids = obj["sample_ids"].astype(str)
        full = np.asarray(obj["full"], dtype=np.float32)
    return sample_ids, full


def resolve_local_snapshot(model_name: str, cache_dir: str | Path = "/media/dhqg/d3/hf_cache") -> str | None:
    repo_id = model_name.replace("/", "--")
    candidate_roots = [
        Path(cache_dir) / "hub" / f"models--{repo_id}",
        Path(cache_dir) / f"models--{repo_id}",
    ]
    for root in candidate_roots:
        snapshots = root / "snapshots"
        if snapshots.exists():
            bins = sorted(p for p in snapshots.iterdir() if p.is_dir())
            if bins:
                return str(bins[-1])
    return None


def load_text_embeddings(model_name: str = "google/siglip-base-patch16-224") -> tuple[list[str], np.ndarray]:
    from transformers import AutoModel, AutoProcessor

    snapshot_dir = resolve_local_snapshot(model_name)
    if snapshot_dir is None:
        raise FileNotFoundError(f"No local Hugging Face snapshot found for {model_name}")
    processor = AutoProcessor.from_pretrained(snapshot_dir, local_files_only=True)
    model = AutoModel.from_pretrained(snapshot_dir, local_files_only=True)
    model.eval()
    texts = sorted({r["why"] for r in load_manifest(ROOT / "results/hoi_detection_v1/manifest.jsonl")})
    tokens = processor(
        text=texts,
        padding="max_length",
        max_length=64,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        emb = model.get_text_features(**tokens)
        if hasattr(emb, "pooler_output"):
            emb = emb.pooler_output
        emb = torch.nn.functional.normalize(emb.float(), dim=-1).cpu().numpy()
    return texts, emb


def softmax_prob(scores: np.ndarray, temp: float = 0.07) -> np.ndarray:
    scaled = scores / temp
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    ex = np.exp(scaled)
    return ex / ex.sum(axis=1, keepdims=True)


def compute_kway_accuracy(emb_mean: np.ndarray, text_emb: np.ndarray, candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = (candidates >= 0).all(axis=1)
    safe = np.where(candidates >= 0, candidates, 0)
    scores = np.einsum("nd,nkd->nk", emb_mean, text_emb[safe])
    hits = (scores[:, :1] > scores[:, 1:]).all(axis=1)
    return hits, valid


def score_margin_and_acc():
    manifest_path = ROOT / "results/hoi_detection_v1/manifest.jsonl"
    rows = load_manifest(manifest_path)
    sample_ids = [r["sample_id"] for r in rows]
    cache_sample_ids, frame_emb = load_cache(Path("/media/dhqg/d3/vlia_outputs/e1/siglip_frame_embeddings_e1_v1.npz"))
    sample_to_idx = {sid: i for i, sid in enumerate(cache_sample_ids)}
    if sorted(sample_ids) != sorted(cache_sample_ids.tolist()):
        raise ValueError("Manifest and cache sample_id sets differ.")
    idx = np.array([sample_to_idx[sid] for sid in sample_ids], dtype=np.int64)
    frame_emb = frame_emb[idx]
    text_list, text_emb = load_text_embeddings()
    texts = [r["why"] for r in rows]
    text_vocab = {text: i for i, text in enumerate(text_list)}
    target_emb = np.asarray([text_emb[text_vocab[t]] for t in texts], dtype=np.float32)
    video_uids = np.asarray([r["video_uid"] for r in rows])
    scenes = np.asarray([r["scene"] for r in rows])
    cands = CV.build_kway(video_uids.tolist(), texts, sample_ids, KWAY_K)
    expected = {
        2: 0.27726727926488215,
        4: 0.275269676388334,
        6: 0.2808629644426688,
        8: 0.2884538553735517,
    }
    for k in PREFIXES:
        prefix_emb = frame_emb[:, :k].mean(axis=1)
        prefix_emb = prefix_emb / np.maximum(np.linalg.norm(prefix_emb, axis=1, keepdims=True), 1e-12)
        metrics = []
        for i in range(len(rows)):
            cand = cands[i]
            if (cand < 0).any():
                continue
            cos = prefix_emb[i] @ target_emb[cand].T
            metrics.append(bool(cos[0] > np.max(cos[1:])))
        actual = float(np.mean(metrics))
        if not np.isclose(actual, expected[k], atol=1e-9, rtol=0.0):
            raise AssertionError(f"K-way accuracy mismatch for k={k}: actual={actual} expected={expected[k]}")
    return rows, frame_emb, target_emb, cands, video_uids, scenes


def compute_score_stats(rows, frame_emb, target_emb, candidates, video_uids, scenes):
    rows_by_k = {}
    for k in PREFIXES:
        prefix_emb = frame_emb[:, :k].mean(axis=1)
        prefix_emb = prefix_emb / np.maximum(np.linalg.norm(prefix_emb, axis=1, keepdims=True), 1e-12)
        sims = prefix_emb @ target_emb.T
        score_rows = []
        for i, row in enumerate(rows):
            cand = candidates[i]
            if (cand < 0).any():
                continue
            cos = prefix_emb[i] @ target_emb[cand].T
            top_idx = np.argsort(cos)[::-1]
            top1 = float(cos[top_idx[0]])
            top2 = float(cos[top_idx[1]])
            probs = softmax_prob(target_emb[cand][None, :] if False else np.asarray([cos], dtype=np.float32), temp=0.07)[0]
            margin = top1 - top2
            confidence = float(probs.max())
            entropy_neg = float(-(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum() / math.log(5.0))
            # note: the score for a query is the same as the candidate ranking for its true label in the full candidate set
            correct = bool(cos[0] > np.max(cos[1:]))
            score_rows.append({
                "sample_id": row["sample_id"],
                "video_uid": row["video_uid"],
                "scene": row["scene"],
                "k": k,
                "margin": margin,
                "confidence": confidence,
                "entropy_neg": -float(-(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum() / math.log(5.0)),
                "correct": correct,
                "score": margin,
            })
        rows_by_k[k] = score_rows
    return rows_by_k


def _accept_mask_for_threshold(score, threshold, higher_is_more_confident):
    return np.asarray(score >= threshold) if higher_is_more_confident else np.asarray(score <= threshold)


def compute_summary(rows_by_k, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    risk_rows = []
    for k, samples in rows_by_k.items():
        score = np.asarray([s["margin"] for s in samples], dtype=np.float64)
        correct = np.asarray([1 if s["correct"] else 0 for s in samples], dtype=np.float64)
        order = np.argsort(-score, kind="mergesort")
        sorted_score = score[order]
        sorted_correct = correct[order]
        coverage_levels = [1.0, 0.8, 0.6, 0.5, 0.4, 0.2]
        for cov in coverage_levels:
            n_accept = max(1, int(round(len(samples) * cov)))
            if cov == 1.0:
                accept = np.ones(len(samples), dtype=bool)
            else:
                accept = np.arange(len(samples)) < n_accept
            accepted = sorted_correct[accept]
            acc = float(accepted.mean()) if accepted.size else 0.0
            risk = float((~accepted.astype(bool)).mean()) if accepted.size else 1.0
            summary_rows.append({
                "k": k,
                "score": "margin",
                "coverage": cov,
                "selective_accuracy": acc,
                "risk": risk,
                "n_accepted": int(accepted.size),
            })
            risk_rows.append({"k": k, "score": "margin", "coverage": cov, "risk": risk, "selective_accuracy": acc})
    with (out_dir / "risk_coverage_curves.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["k", "score", "coverage", "risk", "selective_accuracy"])
        writer.writeheader()
        writer.writerows(risk_rows)
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["k", "score", "coverage", "selective_accuracy", "risk", "n_accepted"])
        writer.writeheader()
        writer.writerows(summary_rows)
    return {"summary_rows": summary_rows, "risk_rows": risk_rows}


def operational_thresholds(rows_by_k):
    rows = []
    for k, samples in rows_by_k.items():
        score = np.asarray([s["margin"] for s in samples], dtype=np.float64)
        correct = np.asarray([1 if s["correct"] else 0 for s in samples], dtype=np.float64)
        scene = np.asarray([s["scene"] for s in samples])
        fold_map = CV.assign_folds(scene.tolist(), 8)
        for target_cov in (0.8, 0.5, 0.2):
            thresh = []
            used = []
            for fold in sorted(set(fold_map.values())):
                ids = np.asarray([i for i, f in enumerate(scene) if fold_map[f] == fold], dtype=int)
                out = np.asarray([i for i in range(len(samples)) if fold_map[scene[i]] != fold], dtype=int)
                if len(out) == 0:
                    continue
                q = np.quantile(score[out], 1.0 - target_cov)
                thresh.append(float(q))
                pred = score[ids] >= q
                used.append({"fold": fold, "coverage": float(pred.mean()), "acc": float(correct[ids][pred].mean()) if pred.any() else 0.0})
            rows.append({"k": k, "target_coverage": target_cov, "threshold": float(np.median(thresh)) if thresh else 0.0, "realized_coverage": float(np.mean([u["coverage"] for u in used])) if used else 0.0, "realized_acc": float(np.mean([u["acc"] for u in used])) if used else 0.0})
    return rows


def plot_curve(rows_by_k, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for k, samples in rows_by_k.items():
        score = np.asarray([s["margin"] for s in samples], dtype=np.float64)
        correct = np.asarray([1 if s["correct"] else 0 for s in samples], dtype=np.float64)
        order = np.argsort(-score, kind="mergesort")
        sorted_score = score[order]
        sorted_correct = correct[order]
        coverage = np.linspace(1.0, 0.0, len(samples))
        risks = []
        for idx in range(len(samples)):
            accepted = sorted_correct[: idx + 1]
            risks.append(float(1.0 - accepted.mean()))
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(coverage, risks, label=f"k={k}")
        ax.plot([0, 1], [1.0 - correct.mean(), 1.0 - correct.mean()], linestyle="--", label="random rejection")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("coverage")
        ax.set_ylabel("risk")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"risk_coverage_k{k}.svg")
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "results/hoi_detection_v1/manifest.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/selective_e1_v1")
    args = parser.parse_args()

    rows, frame_emb, target_emb, cands, video_uids, scenes = score_margin_and_acc()
    rows_by_k = compute_score_stats(rows, frame_emb, target_emb, cands, video_uids, scenes)
    summary = compute_summary(rows_by_k, args.output_dir)
    thresholds = operational_thresholds(rows_by_k)
    plot_curve(rows_by_k, args.output_dir)

    config = {
        "git_commit": __import__("subprocess").check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "cache": "/media/dhqg/d3/vlia_outputs/e1/siglip_frame_embeddings_e1_v1.npz",
        "prefixes": list(PREFIXES),
        "kway": KWAY_K,
        "scores": ["margin", "confidence", "entropy_neg"],
        "thresholds": thresholds,
    }
    with (args.output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)

    with (args.output_dir / "operational_thresholds.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["k", "target_coverage", "threshold", "realized_coverage", "realized_acc"])
        writer.writeheader()
        writer.writerows(thresholds)

    interpretation = """# Selective prediction interpretation

- The signal is computed from the cached SigLIP full-frame embeddings only.
- Margin is the primary decision score; confidence and entropy are secondary baselines.
- This is an analysis-only validation slice, not a calibrated deployment policy.
- No untouched test split is used.
"""
    (args.output_dir / "INTERPRETATION.md").write_text(interpretation, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
