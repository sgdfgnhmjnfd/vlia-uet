from pathlib import Path
import argparse
import json
import math
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# PCA loader
# ============================================================

def load_pca_transform(path: Path):
    """
    Robust loader for the previously saved WHY PCA transform.

    Returns:
        transform(x): [N, 960] -> [N, 256]

    Supports several common checkpoint layouts:
      - dict with mean + components
      - dict with mean + V
      - dict with pca_mean + pca_components
      - dict with mean + projection
      - a raw [960, 256] or [256, 960] tensor
    """
    obj = torch.load(path, map_location="cpu")

    mean = None
    comp = None

    if torch.is_tensor(obj):
        comp = obj.float()

    elif isinstance(obj, dict):
        mean_keys = [
            "mean",
            "pca_mean",
            "mu",
            "center",
        ]

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

        # Some checkpoints may nest a state dict.
        if comp is None:
            for container_key in [
                "state_dict",
                "pca",
                "transform",
            ]:
                if container_key in obj and isinstance(obj[container_key], dict):
                    nested = obj[container_key]

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
            f"Could not infer PCA components from checkpoint: {path}\n"
            f"Loaded object type: {type(obj)}\n"
            f"Keys: {list(obj.keys()) if isinstance(obj, dict) else 'N/A'}"
        )

    comp = comp.squeeze()

    if comp.ndim != 2:
        raise RuntimeError(
            f"PCA components must be 2D, got shape {tuple(comp.shape)}"
        )

    if mean is not None:
        mean = mean.squeeze().float()

    # Normalize orientation to [960, 256].
    if comp.shape == (960, 256):
        basis = comp
    elif comp.shape == (256, 960):
        basis = comp.T
    else:
        raise RuntimeError(
            "Expected PCA components shaped [960,256] or [256,960], "
            f"got {tuple(comp.shape)}"
        )

    if mean is None:
        print(
            "WARNING: PCA checkpoint contains no explicit mean. "
            "Using zero mean."
        )
        mean = torch.zeros(960, dtype=torch.float32)

    if mean.numel() != 960:
        raise RuntimeError(
            f"Expected PCA mean length 960, got {mean.numel()}"
        )

    mean = mean.reshape(1, 960)
    basis = basis.reshape(960, 256)

    def transform(x):
        x = x.float().cpu()
        return (x - mean) @ basis

    print("Loaded WHY PCA:")
    print("  path:", path)
    print("  mean:", tuple(mean.shape))
    print("  basis:", tuple(basis.shape))

    return transform


# ============================================================
# EgoIntent
# ============================================================

def load_egointent_split(cache_root: Path, split: str):
    root = cache_root / split

    if not root.exists():
        raise FileNotFoundError(f"Missing EgoIntent split: {root}")

    rows = []

    for p in sorted(root.glob("*.pt")):
        x = torch.load(p, map_location="cpu")

        visual = x["visual_feature"].float()
        why_feature = x["why_feature"].float()

        if tuple(visual.shape) != (960,):
            raise RuntimeError(
                f"Unexpected visual_feature shape {tuple(visual.shape)} in {p}"
            )

        if tuple(why_feature.shape) != (960,):
            raise RuntimeError(
                f"Unexpected why_feature shape {tuple(why_feature.shape)} in {p}"
            )

        rows.append(
            {
                "sample_id": str(x["sample_id"]),
                "video_uid": str(x["video_uid"]),
                "task": str(x["task"]),
                "why": str(x["why"]),
                "visual": visual,
                "why_feature": why_feature,
            }
        )

    if not rows:
        raise RuntimeError(f"No EgoIntent samples found in {root}")

    return rows


def stack_egointent(rows, why_transform, device):
    visual = torch.stack(
        [x["visual"] for x in rows]
    ).to(device)

    why_960 = torch.stack(
        [x["why_feature"] for x in rows]
    )

    why_target = why_transform(
        why_960
    ).to(device)

    why_target = F.normalize(
        why_target,
        dim=-1,
    )

    why_texts = [x["why"] for x in rows]

    return visual, why_target, why_texts


# ============================================================
# ENIGMA cross-view
# ============================================================

def load_enigma_pairs(cache_root: Path, video_ids):
    video_ids = {str(v) for v in video_ids}

    grouped = defaultdict(dict)

    for p in sorted(cache_root.rglob("*.pt")):
        x = torch.load(p, map_location="cpu")

        video_uid = str(x["video_uid"])

        if video_uid not in video_ids:
            continue

        feat = x["frame_features"].float()

        if feat.ndim != 2 or feat.shape[-1] != 960:
            raise RuntimeError(
                f"Unexpected ENIGMA feature shape {tuple(feat.shape)} in {p}"
            )

        # Keep cross-view input directly comparable to the projection
        # diagnostic that succeeded.
        feat = feat.mean(dim=0)

        grouped[str(x["pair_group_id"])][str(x["view_type"])] = {
            "feat": feat,
            "video_uid": video_uid,
            "phase": str(x["phase"]),
            "what": str(x.get("what", "")),
        }

    pairs = []

    for pair_id, views in sorted(grouped.items()):
        if "ego" not in views or "exo" not in views:
            continue

        ego = views["ego"]
        exo = views["exo"]

        pairs.append(
            {
                "pair_group_id": pair_id,
                "video_uid": ego["video_uid"],
                "phase": ego["phase"],
                "ego": ego["feat"],
                "exo": exo["feat"],
            }
        )

    if not pairs:
        raise RuntimeError(
            f"No complete ENIGMA pairs found for video IDs {sorted(video_ids)}"
        )

    return pairs


def stack_enigma(pairs, device):
    ego = torch.stack(
        [x["ego"] for x in pairs]
    ).to(device)

    exo = torch.stack(
        [x["exo"] for x in pairs]
    ).to(device)

    return ego, exo


# ============================================================
# Shared intention encoder
# ============================================================

class SharedIntentionEncoder(nn.Module):
    """
    Shared 960 -> 256 encoder used by:
      - EgoIntent visual observation
      - ENIGMA ego view
      - ENIGMA exo view

    WHY text targets remain frozen PCA targets.
    """

    def __init__(
        self,
        input_dim=960,
        hidden_dim=512,
        output_dim=256,
        dropout=0.0,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x):
        z = self.net(x)
        return F.normalize(z, dim=-1)


# ============================================================
# Losses
# ============================================================

def build_positive_mask(labels, device):
    n = len(labels)

    mask = torch.zeros(
        (n, n),
        dtype=torch.bool,
        device=device,
    )

    groups = defaultdict(list)

    for i, label in enumerate(labels):
        groups[label].append(i)

    for indices in groups.values():
        idx = torch.tensor(
            indices,
            dtype=torch.long,
            device=device,
        )

        mask[idx[:, None], idx[None, :]] = True

    return mask


def multi_positive_infonce(
    pred,
    target,
    positive_mask,
    temperature,
):
    """
    Multi-positive InfoNCE.

    For each query, every target with the same exact WHY string
    is treated as positive.
    """
    logits = (pred @ target.T) / temperature

    log_prob = logits - torch.logsumexp(
        logits,
        dim=1,
        keepdim=True,
    )

    pos_count = positive_mask.sum(
        dim=1
    ).clamp_min(1)

    loss = -(
        log_prob * positive_mask.float()
    ).sum(dim=1) / pos_count

    return loss.mean()


def symmetric_pair_infonce(
    z_ego,
    z_exo,
    temperature,
):
    logits = (z_ego @ z_exo.T) / temperature

    target = torch.arange(
        logits.shape[0],
        device=logits.device,
    )

    return 0.5 * (
        F.cross_entropy(logits, target)
        + F.cross_entropy(logits.T, target)
    )


# ============================================================
# WHY retrieval evaluation
# ============================================================

@torch.no_grad()
def evaluate_why(
    model,
    visual,
    why_target,
    why_texts,
):
    model.eval()

    pred = model(visual)

    sim = pred @ why_target.T

    ranking = torch.argsort(
        sim,
        dim=1,
        descending=True,
    )

    ranks = []

    for i, query_why in enumerate(why_texts):
        positives = {
            j
            for j, gallery_why in enumerate(why_texts)
            if gallery_why == query_why
        }

        first_rank = None

        for rank, j in enumerate(
            ranking[i].tolist(),
            start=1,
        ):
            if j in positives:
                first_rank = rank
                break

        ranks.append(first_rank)

    ranks = torch.tensor(
        ranks,
        dtype=torch.float32,
        device=visual.device,
    )

    positive_scores = []

    for i, query_why in enumerate(why_texts):
        idx = [
            j
            for j, gallery_why in enumerate(why_texts)
            if gallery_why == query_why
        ]

        positive_scores.append(
            sim[i, idx].max()
        )

    positive_scores = torch.stack(
        positive_scores
    )

    # Hardest non-positive per query.
    hardest_negatives = []

    for i, query_why in enumerate(why_texts):
        negative_idx = [
            j
            for j, gallery_why in enumerate(why_texts)
            if gallery_why != query_why
        ]

        hardest_negatives.append(
            sim[i, negative_idx].max()
        )

    hardest_negatives = torch.stack(
        hardest_negatives
    )

    return {
        "Top1": (ranks <= 1).float().mean().item(),
        "Top3": (ranks <= 3).float().mean().item(),
        "Top5": (ranks <= 5).float().mean().item(),
        "MRR": (1.0 / ranks).mean().item(),
        "median_rank": ranks.median().item(),
        "mean_rank": ranks.mean().item(),
        "positive_cosine_mean": positive_scores.mean().item(),
        "hardest_negative_cosine_mean": hardest_negatives.mean().item(),
        "positive_minus_hardest_margin": (
            positive_scores - hardest_negatives
        ).mean().item(),
    }


# ============================================================
# ENIGMA held-out retrieval
# ============================================================

@torch.no_grad()
def evaluate_crossview(
    model,
    ego,
    exo,
):
    model.eval()

    z_ego = model(ego)
    z_exo = model(exo)

    sim = z_ego @ z_exo.T

    def one_direction(mat):
        n = mat.shape[0]

        ranking = torch.argsort(
            mat,
            dim=1,
            descending=True,
        )

        ranks = []

        for i in range(n):
            rank = (
                (ranking[i] == i)
                .nonzero(as_tuple=False)
                .item()
                + 1
            )

            ranks.append(rank)

        ranks = torch.tensor(
            ranks,
            dtype=torch.float32,
            device=mat.device,
        )

        positive = mat.diag()

        eye = torch.eye(
            n,
            dtype=torch.bool,
            device=mat.device,
        )

        negative = mat[~eye].view(
            n,
            n - 1,
        )

        hardest_negative = negative.max(
            dim=1
        ).values

        return {
            "R@1": (ranks <= 1).float().mean().item(),
            "R@3": (ranks <= 3).float().mean().item(),
            "R@5": (ranks <= 5).float().mean().item(),
            "MRR": (1.0 / ranks).mean().item(),
            "positive_cosine_mean": positive.mean().item(),
            "positive_minus_hardest_margin": (
                positive - hardest_negative
            ).mean().item(),
        }

    return {
        "e2x": one_direction(sim),
        "x2e": one_direction(sim.T),
    }


# ============================================================
# Utilities
# ============================================================

def print_metrics(title, m):
    print()
    print(title)
    print("-" * len(title))

    for k, v in m.items():
        if "rank" in k:
            print(f"{k}: {v:.2f}")
        else:
            print(f"{k}: {v:.6f}")


def metric_mean_std(values):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    mean = float(values.mean())

    std = (
        float(values.std(ddof=1))
        if len(values) > 1
        else 0.0
    )

    return mean, std


# ============================================================
# Train one condition / one seed
# ============================================================

def train_one(
    seed,
    lambda_cv,
    train_visual,
    train_why_target,
    train_why_texts,
    val_visual,
    val_why_target,
    val_why_texts,
    enigma_train_ego,
    enigma_train_exo,
    enigma_val_ego,
    enigma_val_exo,
    args,
):
    set_seed(seed)

    device = train_visual.device

    model = SharedIntentionEncoder(
        hidden_dim=args.hidden_dim,
        output_dim=256,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    why_positive_mask = build_positive_mask(
        train_why_texts,
        device=device,
    )

    best_state = None
    best_epoch = -1
    best_val_mrr = -1.0

    history = []

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        pred_why = model(
            train_visual
        )

        loss_why = multi_positive_infonce(
            pred_why,
            train_why_target,
            why_positive_mask,
            temperature=args.why_temperature,
        )

        cosine_diag = (
            1.0
            - (
                pred_why
                * train_why_target
            ).sum(dim=-1).mean()
        )

        loss_why_total = (
            loss_why
            + args.cosine_weight * cosine_diag
        )

        if lambda_cv > 0:
            z_ego = model(
                enigma_train_ego
            )

            z_exo = model(
                enigma_train_exo
            )

            loss_cv = symmetric_pair_infonce(
                z_ego,
                z_exo,
                temperature=args.cv_temperature,
            )
        else:
            loss_cv = torch.zeros(
                (),
                device=device,
            )

        loss = (
            loss_why_total
            + lambda_cv * loss_cv
        )

        loss.backward()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.grad_clip,
            )

        optimizer.step()

        should_eval = (
            epoch == 1
            or epoch % args.eval_every == 0
            or epoch == args.epochs
        )

        if should_eval:
            val_metrics = evaluate_why(
                model,
                val_visual,
                val_why_target,
                val_why_texts,
            )

            cv_metrics = evaluate_crossview(
                model,
                enigma_val_ego,
                enigma_val_exo,
            )

            print(
                f"[lambda_cv={lambda_cv:.3f}] "
                f"[seed={seed}] "
                f"epoch={epoch:03d} "
                f"loss={loss.item():.5f} "
                f"why={loss_why_total.item():.5f} "
                f"cv={loss_cv.item():.5f} "
                f"VAL_WHY_MRR={val_metrics['MRR']:.6f} "
                f"CV_MRR="
                f"{cv_metrics['e2x']['MRR']:.4f}/"
                f"{cv_metrics['x2e']['MRR']:.4f}"
            )

            history.append(
                {
                    "epoch": epoch,
                    "loss": loss.item(),
                    "loss_why": loss_why_total.item(),
                    "loss_cv": loss_cv.item(),
                    "val_why": val_metrics,
                    "val_crossview": cv_metrics,
                }
            )

            # Main checkpoint criterion remains WHY prediction.
            if val_metrics["MRR"] > best_val_mrr:
                best_val_mrr = val_metrics["MRR"]
                best_epoch = epoch

                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }

    if best_state is None:
        raise RuntimeError("No best checkpoint captured.")

    model.load_state_dict(
        best_state
    )

    final_val_why = evaluate_why(
        model,
        val_visual,
        val_why_target,
        val_why_texts,
    )

    final_cv = evaluate_crossview(
        model,
        enigma_val_ego,
        enigma_val_exo,
    )

    return {
        "seed": seed,
        "lambda_cv": lambda_cv,
        "best_epoch": best_epoch,
        "best_val_why": final_val_why,
        "best_crossview": final_cv,
        "history": history,
        "state_dict": best_state,
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--egointent-cache",
        type=Path,
        default=Path(
            "/media/dhqg/d1/datasets/"
            "egointent/cache/stage_a_v0"
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
        "--enigma-cache",
        type=Path,
        default=Path(
            "/media/dhqg/d1/datasets/"
            "enigma360/cache/crossview_v1/train"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/media/dhqg/d1/vlia_outputs/"
            "egointent_why_plus_crossview_v1"
        ),
    )

    parser.add_argument(
        "--enigma-train-video-ids",
        nargs="+",
        default=["54", "55", "56"],
    )

    parser.add_argument(
        "--enigma-val-video-ids",
        nargs="+",
        default=["57", "58"],
    )

    parser.add_argument(
        "--lambdas",
        nargs="+",
        type=float,
        default=[0.0, 0.1, 0.3],
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2],
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--eval-every",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--why-temperature",
        type=float,
        default=0.07,
    )

    parser.add_argument(
        "--cv-temperature",
        type=float,
        default=0.07,
    )

    parser.add_argument(
        "--cosine-weight",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
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

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        args.device
    )

    # --------------------------------------------------------
    # Load EgoIntent
    # --------------------------------------------------------

    why_transform = load_pca_transform(
        args.why_pca
    )

    train_rows = load_egointent_split(
        args.egointent_cache,
        "train",
    )

    val_rows = load_egointent_split(
        args.egointent_cache,
        "val",
    )

    (
        train_visual,
        train_why_target,
        train_why_texts,
    ) = stack_egointent(
        train_rows,
        why_transform,
        device,
    )

    (
        val_visual,
        val_why_target,
        val_why_texts,
    ) = stack_egointent(
        val_rows,
        why_transform,
        device,
    )

    # --------------------------------------------------------
    # Load ENIGMA
    # --------------------------------------------------------

    enigma_train_pairs = load_enigma_pairs(
        args.enigma_cache,
        args.enigma_train_video_ids,
    )

    enigma_val_pairs = load_enigma_pairs(
        args.enigma_cache,
        args.enigma_val_video_ids,
    )

    (
        enigma_train_ego,
        enigma_train_exo,
    ) = stack_enigma(
        enigma_train_pairs,
        device,
    )

    (
        enigma_val_ego,
        enigma_val_exo,
    ) = stack_enigma(
        enigma_val_pairs,
        device,
    )

    print("=" * 80)
    print("WHY + CROSS-VIEW CONTROLLED EXPERIMENT")
    print("=" * 80)
    print("device:", device)
    print("EgoIntent train:", len(train_rows))
    print("EgoIntent val:", len(val_rows))
    print("ENIGMA train pairs:", len(enigma_train_pairs))
    print("ENIGMA val pairs:", len(enigma_val_pairs))
    print("lambdas:", args.lambdas)
    print("seeds:", args.seeds)

    all_results = []

    for lambda_cv in args.lambdas:
        print()
        print("#" * 80)
        print(f"LAMBDA_CV = {lambda_cv}")
        print("#" * 80)

        lambda_dir = (
            args.output_dir
            / f"lambda_{lambda_cv:g}"
        )

        lambda_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        condition_results = []

        for seed in args.seeds:
            result = train_one(
                seed=seed,
                lambda_cv=lambda_cv,
                train_visual=train_visual,
                train_why_target=train_why_target,
                train_why_texts=train_why_texts,
                val_visual=val_visual,
                val_why_target=val_why_target,
                val_why_texts=val_why_texts,
                enigma_train_ego=enigma_train_ego,
                enigma_train_exo=enigma_train_exo,
                enigma_val_ego=enigma_val_ego,
                enigma_val_exo=enigma_val_exo,
                args=args,
            )

            seed_dir = (
                lambda_dir
                / f"seed_{seed}"
            )

            seed_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            torch.save(
                {
                    "seed": seed,
                    "lambda_cv": lambda_cv,
                    "best_epoch": result["best_epoch"],
                    "model_state_dict": result["state_dict"],
                    "best_val_why": result["best_val_why"],
                    "best_crossview": result["best_crossview"],
                    "config": vars(args),
                },
                seed_dir / "best.pt",
            )

            with open(
                seed_dir / "history.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    result["history"],
                    f,
                    indent=2,
                )

            print_metrics(
                (
                    f"FINAL | lambda={lambda_cv} | seed={seed} "
                    "| EgoIntent WHY val"
                ),
                result["best_val_why"],
            )

            print_metrics(
                (
                    f"FINAL | lambda={lambda_cv} | seed={seed} "
                    "| ENIGMA val EGO->EXO"
                ),
                result["best_crossview"]["e2x"],
            )

            print_metrics(
                (
                    f"FINAL | lambda={lambda_cv} | seed={seed} "
                    "| ENIGMA val EXO->EGO"
                ),
                result["best_crossview"]["x2e"],
            )

            compact = {
                "seed": seed,
                "lambda_cv": lambda_cv,
                "best_epoch": result["best_epoch"],
                "best_val_why": result["best_val_why"],
                "best_crossview": result["best_crossview"],
            }

            condition_results.append(
                compact
            )

            all_results.append(
                compact
            )

        # ----------------------------------------------------
        # Multi-seed condition summary
        # ----------------------------------------------------

        why_mrr = [
            x["best_val_why"]["MRR"]
            for x in condition_results
        ]

        why_top1 = [
            x["best_val_why"]["Top1"]
            for x in condition_results
        ]

        cv_e2x_mrr = [
            x["best_crossview"]["e2x"]["MRR"]
            for x in condition_results
        ]

        cv_x2e_mrr = [
            x["best_crossview"]["x2e"]["MRR"]
            for x in condition_results
        ]

        why_mrr_mean, why_mrr_std = metric_mean_std(
            why_mrr
        )

        why_top1_mean, why_top1_std = metric_mean_std(
            why_top1
        )

        cv_e2x_mean, cv_e2x_std = metric_mean_std(
            cv_e2x_mrr
        )

        cv_x2e_mean, cv_x2e_std = metric_mean_std(
            cv_x2e_mrr
        )

        print()
        print("=" * 80)
        print(
            f"SUMMARY | lambda_cv={lambda_cv}"
        )
        print("=" * 80)

        print(
            "WHY MRR:",
            f"{why_mrr_mean:.6f} ± {why_mrr_std:.6f}",
            "|",
            [round(x, 6) for x in why_mrr],
        )

        print(
            "WHY Top1:",
            f"{why_top1_mean:.6f} ± {why_top1_std:.6f}",
            "|",
            [round(x, 6) for x in why_top1],
        )

        print(
            "CV EGO->EXO MRR:",
            f"{cv_e2x_mean:.6f} ± {cv_e2x_std:.6f}",
            "|",
            [round(x, 6) for x in cv_e2x_mrr],
        )

        print(
            "CV EXO->EGO MRR:",
            f"{cv_x2e_mean:.6f} ± {cv_x2e_std:.6f}",
            "|",
            [round(x, 6) for x in cv_x2e_mrr],
        )

    # --------------------------------------------------------
    # Final summary across lambdas
    # --------------------------------------------------------

    summary = {}

    for lambda_cv in args.lambdas:
        rows = [
            x
            for x in all_results
            if float(x["lambda_cv"]) == float(lambda_cv)
        ]

        why_mrr = [
            x["best_val_why"]["MRR"]
            for x in rows
        ]

        why_top1 = [
            x["best_val_why"]["Top1"]
            for x in rows
        ]

        cv_e2x = [
            x["best_crossview"]["e2x"]["MRR"]
            for x in rows
        ]

        cv_x2e = [
            x["best_crossview"]["x2e"]["MRR"]
            for x in rows
        ]

        m, s = metric_mean_std(
            why_mrr
        )

        top1_m, top1_s = metric_mean_std(
            why_top1
        )

        e2x_m, e2x_s = metric_mean_std(
            cv_e2x
        )

        x2e_m, x2e_s = metric_mean_std(
            cv_x2e
        )

        summary[str(lambda_cv)] = {
            "why_mrr_values": why_mrr,
            "why_mrr_mean": m,
            "why_mrr_sample_std": s,
            "why_top1_values": why_top1,
            "why_top1_mean": top1_m,
            "why_top1_sample_std": top1_s,
            "cv_e2x_mrr_values": cv_e2x,
            "cv_e2x_mrr_mean": e2x_m,
            "cv_e2x_mrr_sample_std": e2x_s,
            "cv_x2e_mrr_values": cv_x2e,
            "cv_x2e_mrr_mean": x2e_m,
            "cv_x2e_mrr_sample_std": x2e_s,
        }

    with open(
        args.output_dir / "summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "config": vars(args),
                "summary": summary,
                "results": all_results,
            },
            f,
            indent=2,
            default=str,
        )

    print()
    print("=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)

    for lambda_cv in args.lambdas:
        row = summary[str(lambda_cv)]

        print(
            f"lambda={lambda_cv:g} | "
            f"WHY MRR "
            f"{row['why_mrr_mean']:.6f} ± "
            f"{row['why_mrr_sample_std']:.6f} | "
            f"WHY Top1 "
            f"{row['why_top1_mean']:.6f} ± "
            f"{row['why_top1_sample_std']:.6f} | "
            f"CV MRR "
            f"{row['cv_e2x_mrr_mean']:.4f}/"
            f"{row['cv_x2e_mrr_mean']:.4f}"
        )

    print()
    print(
        "Saved summary:",
        args.output_dir / "summary.json",
    )


if __name__ == "__main__":
    main()
