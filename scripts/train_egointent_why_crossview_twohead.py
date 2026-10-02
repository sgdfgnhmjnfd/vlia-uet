from pathlib import Path
import argparse
import json
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
                if container_key not in obj:
                    continue

                nested = obj[container_key]

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
        mean = torch.zeros(960)

    mean = mean.squeeze()

    if mean.numel() != 960:
        raise RuntimeError(
            f"Unexpected PCA mean size: {mean.numel()}"
        )

    mean = mean.reshape(1, 960).float()
    basis = basis.reshape(960, 256).float()

    def transform(x):
        return (x.float().cpu() - mean) @ basis

    print("Loaded WHY PCA:", path)
    print("mean:", tuple(mean.shape))
    print("basis:", tuple(basis.shape))

    return transform


# ============================================================
# EgoIntent loading
# ============================================================

def load_egointent_split(cache_root: Path, split: str):
    root = cache_root / split

    rows = []

    for p in sorted(root.glob("*.pt")):
        x = torch.load(p, map_location="cpu")

        visual = x["visual_feature"].float()
        why_feature = x["why_feature"].float()

        if tuple(visual.shape) != (960,):
            raise RuntimeError(
                f"Bad visual shape {tuple(visual.shape)} in {p}"
            )

        if tuple(why_feature.shape) != (960,):
            raise RuntimeError(
                f"Bad WHY shape {tuple(why_feature.shape)} in {p}"
            )

        rows.append(
            {
                "sample_id": str(x["sample_id"]),
                "video_uid": str(x["video_uid"]),
                "why": str(x["why"]),
                "visual": visual,
                "why_feature": why_feature,
            }
        )

    if not rows:
        raise RuntimeError(f"No samples in {root}")

    return rows


def stack_egointent(rows, why_transform, device):
    visual = torch.stack(
        [x["visual"] for x in rows]
    ).to(device)

    why_960 = torch.stack(
        [x["why_feature"] for x in rows]
    )

    why_256 = why_transform(
        why_960
    ).to(device)

    why_256 = F.normalize(
        why_256,
        dim=-1,
    )

    why_texts = [
        x["why"]
        for x in rows
    ]

    return visual, why_256, why_texts


# ============================================================
# ENIGMA loading
# ============================================================

def load_enigma_pairs(cache_root: Path, video_ids):
    video_ids = {
        str(x)
        for x in video_ids
    }

    grouped = defaultdict(dict)

    for p in sorted(cache_root.rglob("*.pt")):
        x = torch.load(
            p,
            map_location="cpu",
        )

        video_uid = str(x["video_uid"])

        if video_uid not in video_ids:
            continue

        feat = x["frame_features"].float()

        if (
            feat.ndim != 2
            or feat.shape[-1] != 960
        ):
            raise RuntimeError(
                f"Bad ENIGMA feature shape "
                f"{tuple(feat.shape)} in {p}"
            )

        feat = feat.mean(dim=0)

        grouped[
            str(x["pair_group_id"])
        ][
            str(x["view_type"])
        ] = {
            "feat": feat,
            "video_uid": video_uid,
            "phase": str(x["phase"]),
        }

    pairs = []

    for pair_id, views in sorted(
        grouped.items()
    ):
        if (
            "ego" not in views
            or "exo" not in views
        ):
            continue

        if (
            views["ego"]["video_uid"]
            != views["exo"]["video_uid"]
        ):
            raise RuntimeError(
                f"Video mismatch in pair {pair_id}"
            )

        if (
            views["ego"]["phase"]
            != views["exo"]["phase"]
        ):
            raise RuntimeError(
                f"Phase mismatch in pair {pair_id}"
            )

        pairs.append(
            {
                "pair_group_id": pair_id,
                "video_uid": views["ego"]["video_uid"],
                "phase": views["ego"]["phase"],
                "ego": views["ego"]["feat"],
                "exo": views["exo"]["feat"],
            }
        )

    if not pairs:
        raise RuntimeError(
            f"No complete pairs for {sorted(video_ids)}"
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
# Model
# ============================================================

class MultiTaskIntentionEncoder(nn.Module):
    """
    Shared trunk with task-specific heads.

    WHY path:
        960 -> shared 512 -> WHY 256

    Cross-view path:
        960 -> shared 512 -> CV 256

    Compared with the previous experiment, the 960->512 trunk is
    still shared, so ENIGMA gradients can shape the visual
    representation. However, the WHY semantic space is no longer
    forced to be identical to the ego/exo instance-matching space.
    """

    def __init__(
        self,
        input_dim=960,
        hidden_dim=512,
        output_dim=256,
        dropout=0.0,
    ):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.why_head = nn.Sequential(
            nn.Linear(
                hidden_dim,
                output_dim,
            ),
            nn.LayerNorm(output_dim),
        )

        self.cv_head = nn.Sequential(
            nn.Linear(
                hidden_dim,
                output_dim,
            ),
            nn.LayerNorm(output_dim),
        )

    def encode_shared(self, x):
        return self.trunk(x)

    def encode_why(self, x):
        h = self.encode_shared(x)
        z = self.why_head(h)
        return F.normalize(
            z,
            dim=-1,
        )

    def encode_crossview(self, x):
        h = self.encode_shared(x)
        z = self.cv_head(h)
        return F.normalize(
            z,
            dim=-1,
        )


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

        mask[
            idx[:, None],
            idx[None, :],
        ] = True

    return mask


def multi_positive_infonce(
    pred,
    target,
    positive_mask,
    temperature,
):
    logits = (
        pred @ target.T
    ) / temperature

    log_prob = (
        logits
        - torch.logsumexp(
            logits,
            dim=1,
            keepdim=True,
        )
    )

    pos_count = (
        positive_mask
        .sum(dim=1)
        .clamp_min(1)
    )

    loss = -(
        log_prob
        * positive_mask.float()
    ).sum(dim=1) / pos_count

    return loss.mean()


def symmetric_pair_infonce(
    z_ego,
    z_exo,
    temperature,
):
    logits = (
        z_ego @ z_exo.T
    ) / temperature

    target = torch.arange(
        logits.shape[0],
        device=logits.device,
    )

    return 0.5 * (
        F.cross_entropy(
            logits,
            target,
        )
        + F.cross_entropy(
            logits.T,
            target,
        )
    )


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate_why(
    model,
    visual,
    why_target,
    why_texts,
):
    model.eval()

    pred = model.encode_why(
        visual
    )

    sim = pred @ why_target.T

    ranking = torch.argsort(
        sim,
        dim=1,
        descending=True,
    )

    ranks = []
    positive_scores = []
    hardest_negative = []

    for i, query_why in enumerate(
        why_texts
    ):
        pos_idx = [
            j
            for j, gallery_why
            in enumerate(why_texts)
            if gallery_why == query_why
        ]

        neg_idx = [
            j
            for j, gallery_why
            in enumerate(why_texts)
            if gallery_why != query_why
        ]

        pos_set = set(pos_idx)

        first_rank = None

        for rank, j in enumerate(
            ranking[i].tolist(),
            start=1,
        ):
            if j in pos_set:
                first_rank = rank
                break

        ranks.append(
            first_rank
        )

        positive_scores.append(
            sim[i, pos_idx].max()
        )

        hardest_negative.append(
            sim[i, neg_idx].max()
        )

    ranks = torch.tensor(
        ranks,
        dtype=torch.float32,
        device=visual.device,
    )

    positive_scores = torch.stack(
        positive_scores
    )

    hardest_negative = torch.stack(
        hardest_negative
    )

    return {
        "Top1": (
            ranks <= 1
        ).float().mean().item(),
        "Top3": (
            ranks <= 3
        ).float().mean().item(),
        "Top5": (
            ranks <= 5
        ).float().mean().item(),
        "MRR": (
            1.0 / ranks
        ).mean().item(),
        "median_rank": ranks.median().item(),
        "mean_rank": ranks.mean().item(),
        "positive_cosine_mean": (
            positive_scores.mean().item()
        ),
        "hardest_negative_cosine_mean": (
            hardest_negative.mean().item()
        ),
        "positive_minus_hardest_margin": (
            positive_scores
            - hardest_negative
        ).mean().item(),
    }


@torch.no_grad()
def evaluate_crossview(
    model,
    ego,
    exo,
):
    model.eval()

    z_ego = model.encode_crossview(
        ego
    )

    z_exo = model.encode_crossview(
        exo
    )

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
                .nonzero(
                    as_tuple=False
                )
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

        negatives = (
            mat[~eye]
            .view(n, n - 1)
        )

        hardest_negative = (
            negatives
            .max(dim=1)
            .values
        )

        return {
            "R@1": (
                ranks <= 1
            ).float().mean().item(),
            "R@3": (
                ranks <= 3
            ).float().mean().item(),
            "R@5": (
                ranks <= 5
            ).float().mean().item(),
            "MRR": (
                1.0 / ranks
            ).mean().item(),
            "positive_cosine_mean": (
                positive.mean().item()
            ),
            "positive_minus_hardest_margin": (
                positive
                - hardest_negative
            ).mean().item(),
        }

    return {
        "e2x": one_direction(sim),
        "x2e": one_direction(sim.T),
    }


# ============================================================
# Gradient conflict diagnostic
# ============================================================

def flattened_grad(
    loss,
    parameters,
    retain_graph,
):
    grads = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )

    chunks = []

    for p, g in zip(
        parameters,
        grads,
    ):
        if g is None:
            chunks.append(
                torch.zeros_like(
                    p
                ).reshape(-1)
            )
        else:
            chunks.append(
                g.reshape(-1)
            )

    return torch.cat(chunks)


def gradient_cosine(
    why_loss,
    cv_loss,
    shared_parameters,
):
    g_why = flattened_grad(
        why_loss,
        shared_parameters,
        retain_graph=True,
    )

    g_cv = flattened_grad(
        cv_loss,
        shared_parameters,
        retain_graph=True,
    )

    denom = (
        g_why.norm()
        * g_cv.norm()
    ).clamp_min(1e-12)

    return (
        torch.dot(
            g_why,
            g_cv,
        )
        / denom
    ).item()


# ============================================================
# Helpers
# ============================================================

def metric_mean_std(values):
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


def print_metrics(title, metrics):
    print()
    print(title)
    print("-" * len(title))

    for k, v in metrics.items():
        if "rank" in k:
            print(
                f"{k}: {v:.2f}"
            )
        else:
            print(
                f"{k}: {v:.6f}"
            )


# ============================================================
# One seed
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

    model = MultiTaskIntentionEncoder(
        hidden_dim=args.hidden_dim,
        output_dim=256,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    why_positive_mask = (
        build_positive_mask(
            train_why_texts,
            device,
        )
    )

    shared_parameters = list(
        model.trunk.parameters()
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

        pred_why = (
            model.encode_why(
                train_visual
            )
        )

        loss_why_nce = (
            multi_positive_infonce(
                pred_why,
                train_why_target,
                why_positive_mask,
                args.why_temperature,
            )
        )

        loss_why_cosine = (
            1.0
            - (
                pred_why
                * train_why_target
            )
            .sum(dim=-1)
            .mean()
        )

        loss_why = (
            loss_why_nce
            + args.cosine_weight
            * loss_why_cosine
        )

        z_ego = (
            model.encode_crossview(
                enigma_train_ego
            )
        )

        z_exo = (
            model.encode_crossview(
                enigma_train_exo
            )
        )

        loss_cv = (
            symmetric_pair_infonce(
                z_ego,
                z_exo,
                args.cv_temperature,
            )
        )

        if (
            lambda_cv > 0
            and (
                epoch == 1
                or epoch
                % args.grad_cos_every
                == 0
            )
        ):
            grad_cos = gradient_cosine(
                loss_why,
                loss_cv,
                shared_parameters,
            )
        else:
            grad_cos = None

        total_loss = (
            loss_why
            + lambda_cv
            * loss_cv
        )

        total_loss.backward()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.grad_clip,
            )

        optimizer.step()

        should_eval = (
            epoch == 1
            or epoch
            % args.eval_every
            == 0
            or epoch
            == args.epochs
        )

        if should_eval:
            val_why = evaluate_why(
                model,
                val_visual,
                val_why_target,
                val_why_texts,
            )

            val_cv = evaluate_crossview(
                model,
                enigma_val_ego,
                enigma_val_exo,
            )

            print(
                f"[lambda={lambda_cv:.3f}] "
                f"[seed={seed}] "
                f"epoch={epoch:03d} "
                f"total={total_loss.item():.5f} "
                f"why={loss_why.item():.5f} "
                f"cv={loss_cv.item():.5f} "
                f"WHY_MRR={val_why['MRR']:.6f} "
                f"CV_MRR="
                f"{val_cv['e2x']['MRR']:.4f}/"
                f"{val_cv['x2e']['MRR']:.4f}"
                + (
                    f" grad_cos={grad_cos:+.4f}"
                    if grad_cos is not None
                    else ""
                )
            )

            history.append(
                {
                    "epoch": epoch,
                    "total_loss": (
                        total_loss.item()
                    ),
                    "why_loss": (
                        loss_why.item()
                    ),
                    "cv_loss": (
                        loss_cv.item()
                    ),
                    "grad_cos": grad_cos,
                    "val_why": val_why,
                    "val_cv": val_cv,
                }
            )

            if (
                val_why["MRR"]
                > best_val_mrr
            ):
                best_val_mrr = (
                    val_why["MRR"]
                )

                best_epoch = epoch

                best_state = {
                    k: (
                        v.detach()
                        .cpu()
                        .clone()
                    )
                    for k, v
                    in model.state_dict().items()
                }

    if best_state is None:
        raise RuntimeError(
            "No checkpoint selected."
        )

    model.load_state_dict(
        best_state
    )

    final_why = evaluate_why(
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

    grad_cos_values = [
        x["grad_cos"]
        for x in history
        if x["grad_cos"]
        is not None
    ]

    return {
        "seed": seed,
        "lambda_cv": lambda_cv,
        "best_epoch": best_epoch,
        "best_val_why": final_why,
        "best_crossview": final_cv,
        "grad_cos_values": (
            grad_cos_values
        ),
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
            "enigma360/cache/"
            "crossview_v1/train"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/media/dhqg/d1/vlia_outputs/"
            "egointent_why_crossview_twohead_v1"
        ),
    )

    parser.add_argument(
        "--enigma-train-video-ids",
        nargs="+",
        default=[
            "54",
            "55",
            "56",
        ],
    )

    parser.add_argument(
        "--enigma-val-video-ids",
        nargs="+",
        default=[
            "57",
            "58",
        ],
    )

    parser.add_argument(
        "--lambdas",
        nargs="+",
        type=float,
        default=[
            0.0,
            0.1,
            0.3,
        ],
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
        "--grad-cos-every",
        type=int,
        default=10,
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

    device = torch.device(
        args.device
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    why_transform = (
        load_pca_transform(
            args.why_pca
        )
    )

    train_rows = (
        load_egointent_split(
            args.egointent_cache,
            "train",
        )
    )

    val_rows = (
        load_egointent_split(
            args.egointent_cache,
            "val",
        )
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

    enigma_train_pairs = (
        load_enigma_pairs(
            args.enigma_cache,
            args.enigma_train_video_ids,
        )
    )

    enigma_val_pairs = (
        load_enigma_pairs(
            args.enigma_cache,
            args.enigma_val_video_ids,
        )
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
    print(
        "WHY + CROSS-VIEW TWO-HEAD EXPERIMENT"
    )
    print("=" * 80)
    print("device:", device)
    print(
        "EgoIntent train:",
        len(train_rows),
    )
    print(
        "EgoIntent val:",
        len(val_rows),
    )
    print(
        "ENIGMA train pairs:",
        len(enigma_train_pairs),
    )
    print(
        "ENIGMA val pairs:",
        len(enigma_val_pairs),
    )
    print(
        "lambdas:",
        args.lambdas,
    )
    print(
        "seeds:",
        args.seeds,
    )

    all_results = []

    for lambda_cv in args.lambdas:
        print()
        print("#" * 80)
        print(
            f"LAMBDA_CV = {lambda_cv}"
        )
        print("#" * 80)

        condition = []

        lambda_dir = (
            args.output_dir
            / f"lambda_{lambda_cv:g}"
        )

        lambda_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

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
                    "best_epoch": result[
                        "best_epoch"
                    ],
                    "model_state_dict": (
                        result[
                            "state_dict"
                        ]
                    ),
                    "best_val_why": (
                        result[
                            "best_val_why"
                        ]
                    ),
                    "best_crossview": (
                        result[
                            "best_crossview"
                        ]
                    ),
                    "grad_cos_values": (
                        result[
                            "grad_cos_values"
                        ]
                    ),
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
                    f"FINAL | lambda={lambda_cv} "
                    f"| seed={seed} "
                    "| WHY VAL"
                ),
                result["best_val_why"],
            )

            print_metrics(
                (
                    f"FINAL | lambda={lambda_cv} "
                    f"| seed={seed} "
                    "| CV EGO->EXO"
                ),
                result[
                    "best_crossview"
                ]["e2x"],
            )

            print_metrics(
                (
                    f"FINAL | lambda={lambda_cv} "
                    f"| seed={seed} "
                    "| CV EXO->EGO"
                ),
                result[
                    "best_crossview"
                ]["x2e"],
            )

            compact = {
                "seed": seed,
                "lambda_cv": lambda_cv,
                "best_epoch": (
                    result["best_epoch"]
                ),
                "best_val_why": (
                    result["best_val_why"]
                ),
                "best_crossview": (
                    result[
                        "best_crossview"
                    ]
                ),
                "grad_cos_values": (
                    result[
                        "grad_cos_values"
                    ]
                ),
            }

            condition.append(
                compact
            )

            all_results.append(
                compact
            )

        why_mrr = [
            x["best_val_why"]["MRR"]
            for x in condition
        ]

        why_top1 = [
            x["best_val_why"]["Top1"]
            for x in condition
        ]

        cv_e2x = [
            x["best_crossview"][
                "e2x"
            ]["MRR"]
            for x in condition
        ]

        cv_x2e = [
            x["best_crossview"][
                "x2e"
            ]["MRR"]
            for x in condition
        ]

        grad_cos = [
            g
            for x in condition
            for g in x[
                "grad_cos_values"
            ]
        ]

        mrr_m, mrr_s = (
            metric_mean_std(
                why_mrr
            )
        )

        top1_m, top1_s = (
            metric_mean_std(
                why_top1
            )
        )

        e2x_m, e2x_s = (
            metric_mean_std(
                cv_e2x
            )
        )

        x2e_m, x2e_s = (
            metric_mean_std(
                cv_x2e
            )
        )

        if grad_cos:
            gc_m, gc_s = (
                metric_mean_std(
                    grad_cos
                )
            )
        else:
            gc_m = None
            gc_s = None

        print()
        print("=" * 80)
        print(
            f"SUMMARY | lambda={lambda_cv}"
        )
        print("=" * 80)

        print(
            f"WHY MRR: "
            f"{mrr_m:.6f} ± "
            f"{mrr_s:.6f}"
        )

        print(
            f"WHY Top1: "
            f"{top1_m:.6f} ± "
            f"{top1_s:.6f}"
        )

        print(
            f"CV E2X MRR: "
            f"{e2x_m:.6f} ± "
            f"{e2x_s:.6f}"
        )

        print(
            f"CV X2E MRR: "
            f"{x2e_m:.6f} ± "
            f"{x2e_s:.6f}"
        )

        if gc_m is not None:
            print(
                f"Shared-gradient cosine: "
                f"{gc_m:+.6f} ± "
                f"{gc_s:.6f}"
            )

    # ========================================================
    # Final compact summary
    # ========================================================

    summary = {}

    for lambda_cv in args.lambdas:
        rows = [
            x
            for x in all_results
            if float(
                x["lambda_cv"]
            )
            == float(lambda_cv)
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
            x["best_crossview"][
                "e2x"
            ]["MRR"]
            for x in rows
        ]

        cv_x2e = [
            x["best_crossview"][
                "x2e"
            ]["MRR"]
            for x in rows
        ]

        grad_cos = [
            g
            for x in rows
            for g in x[
                "grad_cos_values"
            ]
        ]

        mrr_m, mrr_s = (
            metric_mean_std(
                why_mrr
            )
        )

        top1_m, top1_s = (
            metric_mean_std(
                why_top1
            )
        )

        e2x_m, e2x_s = (
            metric_mean_std(
                cv_e2x
            )
        )

        x2e_m, x2e_s = (
            metric_mean_std(
                cv_x2e
            )
        )

        if grad_cos:
            gc_m, gc_s = (
                metric_mean_std(
                    grad_cos
                )
            )
        else:
            gc_m = None
            gc_s = None

        summary[str(lambda_cv)] = {
            "why_mrr_values": why_mrr,
            "why_mrr_mean": mrr_m,
            "why_mrr_sample_std": mrr_s,
            "why_top1_values": why_top1,
            "why_top1_mean": top1_m,
            "why_top1_sample_std": (
                top1_s
            ),
            "cv_e2x_mrr_values": (
                cv_e2x
            ),
            "cv_e2x_mrr_mean": (
                e2x_m
            ),
            "cv_e2x_mrr_sample_std": (
                e2x_s
            ),
            "cv_x2e_mrr_values": (
                cv_x2e
            ),
            "cv_x2e_mrr_mean": (
                x2e_m
            ),
            "cv_x2e_mrr_sample_std": (
                x2e_s
            ),
            "grad_cos_mean": gc_m,
            "grad_cos_sample_std": gc_s,
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
        row = summary[
            str(lambda_cv)
        ]

        grad_text = ""

        if (
            row["grad_cos_mean"]
            is not None
        ):
            grad_text = (
                f" | grad_cos "
                f"{row['grad_cos_mean']:+.4f}"
            )

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
            f"{grad_text}"
        )

    print()
    print(
        "Saved summary:",
        args.output_dir / "summary.json",
    )


if __name__ == "__main__":
    main()
