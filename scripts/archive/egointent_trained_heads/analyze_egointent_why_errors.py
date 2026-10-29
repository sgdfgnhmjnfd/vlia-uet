from pathlib import Path
import argparse
import json
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# PCA loader
# ============================================================

def load_pca_transform(path: Path):
    obj = torch.load(path, map_location="cpu")

    mean = None
    comp = None

    if torch.is_tensor(obj):
        comp = obj.float()

    elif isinstance(obj, dict):
        mean_keys = ["mean", "pca_mean", "mu", "center"]
        comp_keys = [
            "components",
            "components_",
            "pca_components",
            "projection",
            "proj",
            "V",
            "basis",
        ]

        for k in mean_keys:
            if k in obj and torch.is_tensor(obj[k]):
                mean = obj[k].float()
                break

        for k in comp_keys:
            if k in obj and torch.is_tensor(obj[k]):
                comp = obj[k].float()
                break

        if comp is None:
            for container_key in ["state_dict", "pca", "transform"]:
                nested = obj.get(container_key)
                if not isinstance(nested, dict):
                    continue

                for k in mean_keys:
                    if k in nested and torch.is_tensor(nested[k]):
                        mean = nested[k].float()
                        break

                for k in comp_keys:
                    if k in nested and torch.is_tensor(nested[k]):
                        comp = nested[k].float()
                        break

                if comp is not None:
                    break

    if comp is None:
        raise RuntimeError(
            f"Could not infer PCA components from {path}"
        )

    comp = comp.squeeze()

    if comp.shape == (960, 256):
        basis = comp
    elif comp.shape == (256, 960):
        basis = comp.T
    else:
        raise RuntimeError(
            f"Unexpected PCA component shape: {tuple(comp.shape)}"
        )

    if mean is None:
        mean = torch.zeros(960, dtype=torch.float32)

    mean = mean.squeeze().float()

    if mean.numel() != 960:
        raise RuntimeError(
            f"Unexpected PCA mean size: {mean.numel()}"
        )

    mean = mean.reshape(1, 960)
    basis = basis.reshape(960, 256).float()

    def transform(x):
        return (x.float().cpu() - mean) @ basis

    return transform


# ============================================================
# Data loading
# ============================================================

def load_stage_a_metadata(stage_a_root: Path, split: str):
    root = stage_a_root / split
    rows = {}

    for p in sorted(root.glob("*.pt")):
        x = torch.load(p, map_location="cpu")

        sample_id = str(x["sample_id"])

        rows[sample_id] = {
            "sample_id": sample_id,
            "video_uid": str(x["video_uid"]),
            "task": str(x.get("task", "")),
            "why": str(x["why"]),
            "why_feature": x["why_feature"].float(),
        }

    if not rows:
        raise RuntimeError(
            f"No Stage-A metadata found in {root}"
        )

    return rows


def find_temporal_feature(x, path):
    for key in [
        "frame_features",
        "temporal_features",
        "visual_features",
        "frames_feature",
    ]:
        if key in x and torch.is_tensor(x[key]):
            feat = x[key].float()

            if feat.ndim == 2 and feat.shape[-1] == 960:
                return feat

    tensor_keys = {
        k: tuple(v.shape)
        for k, v in x.items()
        if torch.is_tensor(v)
    }

    raise RuntimeError(
        f"No [T,960] temporal tensor in {path}. "
        f"Tensor keys: {tensor_keys}"
    )


def load_temporal_split(
    temporal_root: Path,
    stage_a_root: Path,
    split: str,
):
    temporal_dir = temporal_root / split
    metadata = load_stage_a_metadata(
        stage_a_root,
        split,
    )

    rows = []

    for p in sorted(temporal_dir.rglob("*.pt")):
        x = torch.load(p, map_location="cpu")

        sample_id = str(
            x.get("sample_id", p.stem)
        )

        if sample_id not in metadata:
            continue

        feat = find_temporal_feature(
            x,
            p,
        )

        meta = metadata[sample_id]

        rows.append(
            {
                "sample_id": sample_id,
                "video_uid": meta["video_uid"],
                "task": meta["task"],
                "why": meta["why"],
                "why_feature": meta["why_feature"],
                "frames": feat,
            }
        )

    if not rows:
        raise RuntimeError(
            f"No matched temporal samples in {temporal_dir}"
        )

    return rows


def stack_val(rows, why_transform, device):
    frames = torch.stack(
        [r["frames"] for r in rows]
    ).to(device)

    why_960 = torch.stack(
        [r["why_feature"] for r in rows]
    )

    why_256 = why_transform(
        why_960
    ).to(device)

    why_256 = F.normalize(
        why_256,
        dim=-1,
    )

    return frames, why_256


# ============================================================
# Model
# ============================================================

class RetrievalMLP(nn.Module):
    def __init__(
        self,
        input_dim=960,
        hidden_dim=512,
        output_dim=256,
        dropout=0.0,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_dim,
                output_dim,
            ),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x):
        z = self.net(x)

        return F.normalize(
            z,
            dim=-1,
        )


# ============================================================
# Per-seed prediction
# ============================================================

@torch.no_grad()
def predict_seed(
    checkpoint_path: Path,
    frames,
    why_target,
    rows,
    hidden_dim,
    device,
):
    ckpt = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model = RetrievalMLP(
        input_dim=960,
        hidden_dim=hidden_dim,
        output_dim=256,
        dropout=0.0,
    ).to(device)

    model.load_state_dict(
        ckpt["model_state_dict"]
    )

    model.eval()

    visual = frames.mean(dim=1)

    pred = model(
        visual
    )

    sim = pred @ why_target.T

    ranking = torch.argsort(
        sim,
        dim=1,
        descending=True,
    )

    outputs = []

    for i, row in enumerate(rows):
        positive_idx = [
            j
            for j, gallery_row in enumerate(rows)
            if gallery_row["why"] == row["why"]
        ]

        positive_set = set(
            positive_idx
        )

        rank = None

        for r, j in enumerate(
            ranking[i].tolist(),
            start=1,
        ):
            if j in positive_set:
                rank = r
                break

        top1_idx = int(
            ranking[i, 0].item()
        )

        top5_idx = [
            int(x)
            for x in ranking[i, :5].tolist()
        ]

        outputs.append(
            {
                "sample_id": row["sample_id"],
                "video_uid": row["video_uid"],
                "task": row["task"],
                "true_why": row["why"],
                "rank": rank,
                "rr": 1.0 / rank,
                "top1_task": rows[top1_idx]["task"],
                "top1_why": rows[top1_idx]["why"],
                "top1_correct": (
                    rows[top1_idx]["why"]
                    == row["why"]
                ),
                "top1_same_task": (
                    rows[top1_idx]["task"]
                    == row["task"]
                ),
                "top5_why": [
                    rows[j]["why"]
                    for j in top5_idx
                ],
            }
        )

    return outputs, sim


# ============================================================
# Aggregation helpers
# ============================================================

def mean_std(values):
    x = np.asarray(
        values,
        dtype=np.float64,
    )

    mean = float(
        x.mean()
    )

    std = (
        float(
            x.std(ddof=1)
        )
        if len(x) > 1
        else 0.0
    )

    return mean, std


def aggregate_per_task(seed_outputs):
    """
    seed_outputs:
        list of list-of-sample-results, one list per seed.
    """

    task_names = sorted(
        {
            sample["task"]
            for outputs in seed_outputs
            for sample in outputs
        }
    )

    result = {}

    for task in task_names:
        seed_mrr = []
        seed_top1 = []
        seed_top3 = []
        seed_top5 = []

        for outputs in seed_outputs:
            task_rows = [
                x
                for x in outputs
                if x["task"] == task
            ]

            if not task_rows:
                continue

            ranks = np.asarray(
                [
                    x["rank"]
                    for x in task_rows
                ],
                dtype=np.float64,
            )

            seed_mrr.append(
                float(
                    np.mean(
                        1.0 / ranks
                    )
                )
            )

            seed_top1.append(
                float(
                    np.mean(
                        ranks <= 1
                    )
                )
            )

            seed_top3.append(
                float(
                    np.mean(
                        ranks <= 3
                    )
                )
            )

            seed_top5.append(
                float(
                    np.mean(
                        ranks <= 5
                    )
                )
            )

        mrr_m, mrr_s = mean_std(
            seed_mrr
        )

        t1_m, t1_s = mean_std(
            seed_top1
        )

        t3_m, t3_s = mean_std(
            seed_top3
        )

        t5_m, t5_s = mean_std(
            seed_top5
        )

        n = sum(
            1
            for x in seed_outputs[0]
            if x["task"] == task
        )

        result[task] = {
            "n": n,
            "mrr_mean": mrr_m,
            "mrr_std": mrr_s,
            "top1_mean": t1_m,
            "top1_std": t1_s,
            "top3_mean": t3_m,
            "top3_std": t3_s,
            "top5_mean": t5_m,
            "top5_std": t5_s,
        }

    return result


def aggregate_confusion(seed_outputs):
    total = 0
    same_task_wrong = 0
    cross_task_wrong = 0
    correct = 0

    task_confusions = Counter()
    why_confusions = Counter()

    for outputs in seed_outputs:
        for x in outputs:
            total += 1

            if x["top1_correct"]:
                correct += 1
                continue

            if x["top1_same_task"]:
                same_task_wrong += 1
            else:
                cross_task_wrong += 1

            task_confusions[
                (
                    x["task"],
                    x["top1_task"],
                )
            ] += 1

            why_confusions[
                (
                    x["true_why"],
                    x["top1_why"],
                )
            ] += 1

    wrong = (
        same_task_wrong
        + cross_task_wrong
    )

    return {
        "total_predictions": total,
        "correct_top1": correct,
        "wrong_top1": wrong,
        "same_task_wrong": same_task_wrong,
        "cross_task_wrong": cross_task_wrong,
        "same_task_wrong_fraction": (
            same_task_wrong / wrong
            if wrong > 0
            else 0.0
        ),
        "cross_task_wrong_fraction": (
            cross_task_wrong / wrong
            if wrong > 0
            else 0.0
        ),
        "top_task_confusions": [
            {
                "true_task": a,
                "pred_task": b,
                "count": c,
            }
            for (a, b), c
            in task_confusions.most_common(20)
        ],
        "top_why_confusions": [
            {
                "true_why": a,
                "pred_why": b,
                "count": c,
            }
            for (a, b), c
            in why_confusions.most_common(30)
        ],
    }


def semantic_error_analysis(
    seed_outputs,
    why_target,
    rows,
):
    """
    Compare text-target cosine for:
      correct WHY vs retrieved top-1 WHY

    This is only a diagnostic of semantic closeness in the
    frozen WHY embedding space.
    """

    why_lookup = {}

    for i, row in enumerate(rows):
        why_lookup.setdefault(
            row["why"],
            why_target[i],
        )

    near_error_sims = []
    far_error_sims = []

    all_error_sims = []

    for outputs in seed_outputs:
        for x in outputs:
            if x["top1_correct"]:
                continue

            a = why_lookup[
                x["true_why"]
            ]

            b = why_lookup[
                x["top1_why"]
            ]

            sim = float(
                torch.dot(
                    a,
                    b,
                ).item()
            )

            all_error_sims.append(
                sim
            )

    if not all_error_sims:
        return {
            "num_errors": 0,
        }

    threshold = float(
        np.median(
            all_error_sims
        )
    )

    near_error_sims = [
        x
        for x in all_error_sims
        if x >= threshold
    ]

    far_error_sims = [
        x
        for x in all_error_sims
        if x < threshold
    ]

    return {
        "num_errors": len(
            all_error_sims
        ),
        "semantic_similarity_mean": float(
            np.mean(
                all_error_sims
            )
        ),
        "semantic_similarity_median": float(
            np.median(
                all_error_sims
            )
        ),
        "semantic_similarity_std": float(
            np.std(
                all_error_sims,
                ddof=1,
            )
            if len(all_error_sims) > 1
            else 0.0
        ),
        "near_error_count": len(
            near_error_sims
        ),
        "far_error_count": len(
            far_error_sims
        ),
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--temporal-cache",
        type=Path,
        default=Path(
            "/media/dhqg/d1/datasets/"
            "egointent/cache/"
            "temporal_intention_v1"
        ),
    )

    parser.add_argument(
        "--stage-a-cache",
        type=Path,
        default=Path(
            "/media/dhqg/d1/datasets/"
            "egointent/cache/"
            "stage_a_v0"
        ),
    )

    parser.add_argument(
        "--why-pca",
        type=Path,
        default=Path(
            "/media/dhqg/d1/vlia_outputs/"
            "stage_a_clean_v1/"
            "why_pca_960_to_256.pt"
        ),
    )

    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path(
            "/media/dhqg/d1/vlia_outputs/"
            "prefix_mean_delta_v1/"
            "mean_8f"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/media/dhqg/d1/vlia_outputs/"
            "mean8f_error_analysis_v1"
        ),
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[
            0,
            1,
            2,
        ],
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--device",
        type=str,
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )

    args = parser.parse_args()

    device = torch.device(
        args.device
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    why_transform = load_pca_transform(
        args.why_pca
    )

    val_rows = load_temporal_split(
        args.temporal_cache,
        args.stage_a_cache,
        "val",
    )

    (
        val_frames,
        val_why_target,
    ) = stack_val(
        val_rows,
        why_transform,
        device,
    )

    print("=" * 80)
    print("MEAN-8F ERROR ANALYSIS")
    print("=" * 80)
    print("val samples:", len(val_rows))
    print("seeds:", args.seeds)

    seed_outputs = []

    for seed in args.seeds:
        checkpoint = (
            args.checkpoint_root
            / f"seed_{seed}"
            / "best.pt"
        )

        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Missing checkpoint: {checkpoint}"
            )

        outputs, _ = predict_seed(
            checkpoint_path=checkpoint,
            frames=val_frames,
            why_target=val_why_target,
            rows=val_rows,
            hidden_dim=args.hidden_dim,
            device=device,
        )

        seed_outputs.append(
            outputs
        )

        mrr = np.mean(
            [
                x["rr"]
                for x in outputs
            ]
        )

        top1 = np.mean(
            [
                x["rank"] <= 1
                for x in outputs
            ]
        )

        print(
            f"seed={seed} "
            f"MRR={mrr:.6f} "
            f"Top1={top1:.6f}"
        )

        with open(
            args.output_dir
            / f"seed_{seed}_predictions.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                outputs,
                f,
                indent=2,
            )

    per_task = aggregate_per_task(
        seed_outputs
    )

    confusion = aggregate_confusion(
        seed_outputs
    )

    semantic = semantic_error_analysis(
        seed_outputs,
        val_why_target,
        val_rows,
    )

    report = {
        "config": {
            "checkpoint_root": str(
                args.checkpoint_root
            ),
            "seeds": args.seeds,
            "val_samples": len(
                val_rows
            ),
        },
        "per_task": per_task,
        "confusion": confusion,
        "semantic_error": semantic,
    }

    with open(
        args.output_dir
        / "error_analysis.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
        )

    print()
    print("=" * 80)
    print("PER-TASK MRR")
    print("=" * 80)

    for task, row in sorted(
        per_task.items(),
        key=lambda kv: kv[1][
            "mrr_mean"
        ],
        reverse=True,
    ):
        print(
            f"{task:24s} "
            f"n={row['n']:3d} "
            f"MRR="
            f"{row['mrr_mean']:.6f} ± "
            f"{row['mrr_std']:.6f} "
            f"Top1="
            f"{row['top1_mean']:.6f}"
        )

    print()
    print("=" * 80)
    print("TOP-1 ERROR TYPE")
    print("=" * 80)

    print(
        "wrong_top1:",
        confusion[
            "wrong_top1"
        ],
    )

    print(
        "same_task_wrong:",
        confusion[
            "same_task_wrong"
        ],
        f"({confusion['same_task_wrong_fraction']:.3f})",
    )

    print(
        "cross_task_wrong:",
        confusion[
            "cross_task_wrong"
        ],
        f"({confusion['cross_task_wrong_fraction']:.3f})",
    )

    print()
    print("=" * 80)
    print("TOP TASK CONFUSIONS")
    print("=" * 80)

    for row in confusion[
        "top_task_confusions"
    ][:10]:
        print(
            f"{row['true_task']} "
            f"-> {row['pred_task']} "
            f"| count={row['count']}"
        )

    print()
    print("=" * 80)
    print("TOP WHY CONFUSIONS")
    print("=" * 80)

    for row in confusion[
        "top_why_confusions"
    ][:15]:
        print(
            f"TRUE: {row['true_why']}"
        )
        print(
            f"PRED: {row['pred_why']}"
        )
        print(
            f"count={row['count']}"
        )
        print("-" * 60)

    print()
    print("=" * 80)
    print("SEMANTIC ERROR DIAGNOSTIC")
    print("=" * 80)

    for k, v in semantic.items():
        print(
            f"{k}: {v}"
        )

    print()
    print(
        "Saved:",
        args.output_dir
        / "error_analysis.json",
    )


if __name__ == "__main__":
    main()
