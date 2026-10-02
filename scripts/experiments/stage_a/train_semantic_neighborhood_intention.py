from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# PCA
# ============================================================

def load_pca_transform(path: Path):
    obj = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

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
            for ck in ["state_dict", "pca", "transform"]:
                nested = obj.get(ck)

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

    def transform(x: torch.Tensor) -> torch.Tensor:
        return (x.float().cpu() - mean) @ basis

    return transform


# ============================================================
# Data
# ============================================================

def load_stage_a_split(
    cache_root: Path,
    split: str,
    why_transform,
):
    root = cache_root / split

    if not root.exists():
        raise FileNotFoundError(
            f"Split not found: {root}"
        )

    rows = []

    for p in sorted(root.rglob("*.pt")):
        x = torch.load(
            p,
            map_location="cpu",
            weights_only=False,
        )

        visual = (
            x["visual_feature"]
            .detach()
            .float()
            .cpu()
        )

        why_960 = (
            x["why_feature"]
            .detach()
            .float()
            .cpu()
            .reshape(1, -1)
        )

        if visual.shape != (960,):
            raise ValueError(
                f"Bad visual shape in {p}: {tuple(visual.shape)}"
            )

        why_target = F.normalize(
            why_transform(why_960).squeeze(0),
            dim=-1,
        )

        rows.append(
            {
                "sample_id": str(x["sample_id"]),
                "video_uid": str(
                    x.get("video_uid", "")
                ),
                "task": str(
                    x.get(
                        "task",
                        x.get(
                            "event",
                            x.get("activity", ""),
                        ),
                    )
                ),
                "why": str(x["why"]),
                "visual": visual,
                "why_target": why_target,
            }
        )

    if not rows:
        raise RuntimeError(
            f"No samples loaded from {root}"
        )

    return rows


def stack_rows(rows, device):
    visual = torch.stack(
        [x["visual"] for x in rows]
    ).to(device)

    why_target = torch.stack(
        [x["why_target"] for x in rows]
    ).to(device)

    return visual, why_target


# ============================================================
# Model
# ============================================================

class IntentionEncoder(nn.Module):
    """
    Strong direct baseline:
        960 -> 512 -> GELU -> 256 -> LayerNorm -> L2
    """

    def __init__(
        self,
        hidden_dim=512,
        output_dim=256,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                960,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_dim,
                output_dim,
            ),
            nn.LayerNorm(
                output_dim,
            ),
        )

    def forward(self, x):
        z = self.net(x)

        return F.normalize(
            z,
            dim=-1,
        )


# ============================================================
# Baseline exact-positive loss
# ============================================================

def build_exact_positive_mask(
    labels,
    device,
):
    n = len(labels)

    mask = torch.zeros(
        (n, n),
        dtype=torch.bool,
        device=device,
    )

    groups = defaultdict(list)

    for i, label in enumerate(labels):
        groups[label].append(i)

    for idxs in groups.values():
        idx = torch.tensor(
            idxs,
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


def cosine_alignment_loss(
    pred,
    target,
):
    return (
        1.0
        - (
            pred * target
        ).sum(dim=-1)
    ).mean()


# ============================================================
# Semantic neighborhood supervision
# ============================================================

@torch.no_grad()
def build_semantic_teacher(
    target,
    exact_positive_mask,
    topk,
    teacher_temperature,
):
    """
    Build a soft semantic neighborhood distribution using only
    TRAIN WHY embeddings.

    Exact positives are always retained.
    Additionally, the top-k semantically closest WHY targets are
    retained for each query. This reduces false negatives caused by
    free-form / semantically overlapping WHY labels.
    """
    similarity = (
        target @ target.T
    )

    n = similarity.shape[0]

    k = min(
        int(topk),
        n,
    )

    topk_idx = torch.topk(
        similarity,
        k=k,
        dim=1,
        largest=True,
    ).indices

    support = torch.zeros(
        (n, n),
        dtype=torch.bool,
        device=target.device,
    )

    support.scatter_(
        1,
        topk_idx,
        True,
    )

    support = (
        support
        | exact_positive_mask
    )

    teacher_logits = (
        similarity
        / teacher_temperature
    )

    teacher_logits = (
        teacher_logits
        .masked_fill(
            ~support,
            float("-inf"),
        )
    )

    teacher_probs = F.softmax(
        teacher_logits,
        dim=1,
    )

    return (
        teacher_probs,
        support,
        similarity,
    )


def semantic_neighborhood_kl(
    pred,
    target,
    teacher_probs,
    temperature,
):
    student_logits = (
        pred @ target.T
    ) / temperature

    student_log_prob = (
        F.log_softmax(
            student_logits,
            dim=1,
        )
    )

    loss = -(
        teacher_probs
        * student_log_prob
    ).sum(dim=1).mean()

    return loss


def semantic_margin_loss(
    pred,
    target,
    teacher_similarity,
    exact_positive_mask,
    topk,
    margin,
):
    """
    Optional ranking regularizer:
    the strongest non-exact semantic neighbor should be ranked
    above a semantically distant target.

    This does NOT require string equality.
    """
    with torch.no_grad():
        n = teacher_similarity.shape[0]

        sim = teacher_similarity.clone()

        sim = sim.masked_fill(
            exact_positive_mask,
            float("-inf"),
        )

        k = min(
            int(topk),
            max(1, n - 1),
        )

        near_idx = torch.topk(
            sim,
            k=k,
            dim=1,
            largest=True,
        ).indices[:, 0]

        far_idx = torch.argmin(
            teacher_similarity,
            dim=1,
        )

    student_sim = (
        pred @ target.T
    )

    row = torch.arange(
        student_sim.shape[0],
        device=pred.device,
    )

    near_score = student_sim[
        row,
        near_idx,
    ]

    far_score = student_sim[
        row,
        far_idx,
    ]

    return F.relu(
        margin
        - near_score
        + far_score
    ).mean()


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def retrieval_metrics(
    pred,
    target,
    rows,
):
    labels = [
        x["why"]
        for x in rows
    ]

    sim = (
        pred @ target.T
    )

    ranking = torch.argsort(
        sim,
        dim=1,
        descending=True,
    )

    ranks = []

    correct = 0
    same_task_wrong = 0
    cross_task_wrong = 0

    for i, row in enumerate(rows):
        positives = {
            j
            for j, label in enumerate(labels)
            if label == row["why"]
        }

        rank = None

        for r, j in enumerate(
            ranking[i].tolist(),
            start=1,
        ):
            if j in positives:
                rank = r
                break

        ranks.append(rank)

        top1_idx = int(
            ranking[i, 0].item()
        )

        top1 = rows[
            top1_idx
        ]

        if top1["why"] == row["why"]:
            correct += 1

        elif top1["task"] == row["task"]:
            same_task_wrong += 1

        else:
            cross_task_wrong += 1

    ranks_t = torch.tensor(
        ranks,
        dtype=torch.float32,
        device=pred.device,
    )

    wrong = (
        same_task_wrong
        + cross_task_wrong
    )

    return {
        "Top1": (
            ranks_t <= 1
        ).float().mean().item(),
        "Top3": (
            ranks_t <= 3
        ).float().mean().item(),
        "Top5": (
            ranks_t <= 5
        ).float().mean().item(),
        "MRR": (
            1.0 / ranks_t
        ).mean().item(),
        "median_rank": (
            ranks_t.median().item()
        ),
        "mean_rank": (
            ranks_t.mean().item()
        ),
        "top1_correct_count": correct,
        "same_task_wrong_count": (
            same_task_wrong
        ),
        "cross_task_wrong_count": (
            cross_task_wrong
        ),
        "same_task_wrong_fraction": (
            same_task_wrong / wrong
            if wrong > 0
            else 0.0
        ),
    }


@torch.no_grad()
def evaluate(
    model,
    visual,
    why_target,
    rows,
):
    model.eval()

    pred = model(
        visual
    )

    return retrieval_metrics(
        pred,
        why_target,
        rows,
    )


# ============================================================
# Training
# ============================================================

def clone_state_dict(model):
    return {
        k: (
            v.detach()
            .cpu()
            .clone()
        )
        for k, v
        in model.state_dict().items()
    }


def train_one(
    condition,
    seed,
    train_visual,
    train_why_target,
    train_rows,
    val_visual,
    val_why_target,
    val_rows,
    args,
):
    set_seed(seed)

    device = train_visual.device

    model = IntentionEncoder(
        hidden_dim=args.hidden_dim,
        output_dim=256,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    exact_mask = (
        build_exact_positive_mask(
            [
                x["why"]
                for x in train_rows
            ],
            device,
        )
    )

    (
        teacher_probs,
        teacher_support,
        teacher_similarity,
    ) = build_semantic_teacher(
        target=train_why_target,
        exact_positive_mask=(
            exact_mask
        ),
        topk=args.semantic_topk,
        teacher_temperature=(
            args.teacher_temperature
        ),
    )

    avg_support = float(
        teacher_support
        .float()
        .sum(dim=1)
        .mean()
        .item()
    )

    best_mrr = -1.0
    best_epoch = -1
    best_eval = None
    best_state = None
    history = []

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        pred = model(
            train_visual
        )

        exact_nce = (
            multi_positive_infonce(
                pred,
                train_why_target,
                exact_mask,
                args.temperature,
            )
        )

        cosine = (
            cosine_alignment_loss(
                pred,
                train_why_target,
            )
        )

        exact_loss = (
            exact_nce
            + args.cosine_weight
            * cosine
        )

        if condition in {
            "semantic_soft",
            "semantic_soft_rank",
        }:
            semantic_loss = (
                semantic_neighborhood_kl(
                    pred=pred,
                    target=train_why_target,
                    teacher_probs=(
                        teacher_probs
                    ),
                    temperature=(
                        args.temperature
                    ),
                )
            )
        else:
            semantic_loss = torch.zeros(
                (),
                device=device,
            )

        if condition == "semantic_soft_rank":
            rank_loss = (
                semantic_margin_loss(
                    pred=pred,
                    target=train_why_target,
                    teacher_similarity=(
                        teacher_similarity
                    ),
                    exact_positive_mask=(
                        exact_mask
                    ),
                    topk=(
                        args.semantic_topk
                    ),
                    margin=(
                        args.rank_margin
                    ),
                )
            )
        else:
            rank_loss = torch.zeros(
                (),
                device=device,
            )

        if condition == "soft_only":
            total_loss = (
                semantic_neighborhood_kl(
                    pred=pred,
                    target=train_why_target,
                    teacher_probs=(
                        teacher_probs
                    ),
                    temperature=(
                        args.temperature
                    ),
                )
                + args.cosine_weight
                * cosine
            )

        else:
            total_loss = (
                exact_loss
                + args.semantic_weight
                * semantic_loss
                + args.rank_weight
                * rank_loss
            )

        total_loss.backward()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.grad_clip,
            )

        optimizer.step()

        if (
            epoch == 1
            or epoch % args.eval_every == 0
            or epoch == args.epochs
        ):
            val_eval = evaluate(
                model=model,
                visual=val_visual,
                why_target=val_why_target,
                rows=val_rows,
            )

            history.append(
                {
                    "epoch": epoch,
                    "loss_total": float(
                        total_loss.item()
                    ),
                    "loss_exact": float(
                        exact_loss.item()
                    ),
                    "loss_semantic": float(
                        semantic_loss.item()
                    ),
                    "loss_rank": float(
                        rank_loss.item()
                    ),
                    "val": val_eval,
                }
            )

            mrr = val_eval[
                "MRR"
            ]

            if mrr > best_mrr:
                best_mrr = mrr
                best_epoch = epoch
                best_eval = val_eval
                best_state = (
                    clone_state_dict(
                        model
                    )
                )

    return {
        "condition": condition,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val": best_eval,
        "model_state_dict": best_state,
        "history": history,
        "avg_semantic_support": (
            avg_support
        ),
    }


# ============================================================
# Summary
# ============================================================

def mean_std(values):
    x = np.asarray(
        values,
        dtype=np.float64,
    )

    return (
        float(x.mean()),
        float(
            x.std(
                ddof=1
            )
            if len(x) > 1
            else 0.0
        ),
    )


def summarize(rows):
    fields = {
        "mrr": [
            x["best_val"]["MRR"]
            for x in rows
        ],
        "top1": [
            x["best_val"]["Top1"]
            for x in rows
        ],
        "top3": [
            x["best_val"]["Top3"]
            for x in rows
        ],
        "same_task_wrong_fraction": [
            x["best_val"][
                "same_task_wrong_fraction"
            ]
            for x in rows
        ],
    }

    out = {}

    for name, values in fields.items():
        mean, std = mean_std(
            values
        )

        out[
            f"{name}_values"
        ] = values

        out[
            f"{name}_mean"
        ] = mean

        out[
            f"{name}_sample_std"
        ] = std

    return out


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cache-root",
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
        "--output-dir",
        type=Path,
        default=Path(
            "/media/dhqg/d1/vlia_outputs/"
            "semantic_neighborhood_v1"
        ),
    )

    parser.add_argument(
        "--conditions",
        nargs="+",
        default=[
            "strong_direct",
            "semantic_soft",
            "semantic_soft_rank",
            "soft_only",
        ],
        choices=[
            "strong_direct",
            "semantic_soft",
            "semantic_soft_rank",
            "soft_only",
        ],
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
        "--temperature",
        type=float,
        default=0.07,
    )

    parser.add_argument(
        "--cosine-weight",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--semantic-topk",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--teacher-temperature",
        type=float,
        default=0.07,
    )

    parser.add_argument(
        "--semantic-weight",
        type=float,
        default=0.3,
    )

    parser.add_argument(
        "--rank-weight",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--rank-margin",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=512,
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

    for p in [
        args.cache_root,
        args.why_pca,
    ]:
        if not p.exists():
            raise FileNotFoundError(
                f"Required path not found: {p}"
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
        load_stage_a_split(
            cache_root=args.cache_root,
            split="train",
            why_transform=why_transform,
        )
    )

    val_rows = (
        load_stage_a_split(
            cache_root=args.cache_root,
            split="val",
            why_transform=why_transform,
        )
    )

    (
        train_visual,
        train_why_target,
    ) = stack_rows(
        train_rows,
        device,
    )

    (
        val_visual,
        val_why_target,
    ) = stack_rows(
        val_rows,
        device,
    )

    print("=" * 96)
    print(
        "SANR: SEMANTIC-AWARE NEIGHBORHOOD REGULARIZATION"
    )
    print("=" * 96)

    print(
        "train samples:",
        len(train_rows),
    )

    print(
        "val samples:",
        len(val_rows),
    )

    print(
        "train tasks:",
        sorted(
            {
                x["task"]
                for x in train_rows
            }
        ),
    )

    print(
        "val tasks:",
        sorted(
            {
                x["task"]
                for x in val_rows
            }
        ),
    )

    print(
        "conditions:",
        args.conditions,
    )

    print(
        "seeds:",
        args.seeds,
    )

    print(
        "semantic_topk:",
        args.semantic_topk,
    )

    print(
        "semantic_weight:",
        args.semantic_weight,
    )

    print(
        "teacher_temperature:",
        args.teacher_temperature,
    )

    all_results = []

    for condition in args.conditions:
        print()
        print("#" * 96)
        print(
            "CONDITION:",
            condition,
        )
        print("#" * 96)

        condition_dir = (
            args.output_dir
            / condition
        )

        condition_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for seed in args.seeds:
            result = train_one(
                condition=condition,
                seed=seed,
                train_visual=(
                    train_visual
                ),
                train_why_target=(
                    train_why_target
                ),
                train_rows=train_rows,
                val_visual=(
                    val_visual
                ),
                val_why_target=(
                    val_why_target
                ),
                val_rows=val_rows,
                args=args,
            )

            seed_dir = (
                condition_dir
                / f"seed_{seed}"
            )

            seed_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            torch.save(
                {
                    "condition": (
                        condition
                    ),
                    "seed": seed,
                    "best_epoch": (
                        result[
                            "best_epoch"
                        ]
                    ),
                    "best_val": (
                        result[
                            "best_val"
                        ]
                    ),
                    "model_state_dict": (
                        result[
                            "model_state_dict"
                        ]
                    ),
                    "avg_semantic_support": (
                        result[
                            "avg_semantic_support"
                        ]
                    ),
                    "config": vars(
                        args
                    ),
                },
                seed_dir
                / "best.pt",
            )

            with open(
                seed_dir
                / "history.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    result[
                        "history"
                    ],
                    f,
                    indent=2,
                )

            compact = {
                "condition": (
                    condition
                ),
                "seed": seed,
                "best_epoch": (
                    result[
                        "best_epoch"
                    ]
                ),
                "best_val": (
                    result[
                        "best_val"
                    ]
                ),
                "avg_semantic_support": (
                    result[
                        "avg_semantic_support"
                    ]
                ),
            }

            all_results.append(
                compact
            )

            v = result[
                "best_val"
            ]

            print(
                f"seed={seed} | "
                f"epoch="
                f"{result['best_epoch']} | "
                f"WHY MRR="
                f"{v['MRR']:.6f} | "
                f"Top1="
                f"{v['Top1']:.6f} | "
                f"Top3="
                f"{v['Top3']:.6f} | "
                f"same-task-error="
                f"{v['same_task_wrong_fraction']:.4f} | "
                f"avg_support="
                f"{result['avg_semantic_support']:.2f}"
            )

    summary = {}

    for condition in args.conditions:
        rows = [
            x
            for x in all_results
            if x["condition"] == condition
        ]

        summary[
            condition
        ] = summarize(
            rows
        )

    payload = {
        "config": {
            **vars(args),
            "cache_root": str(
                args.cache_root
            ),
            "why_pca": str(
                args.why_pca
            ),
            "output_dir": str(
                args.output_dir
            ),
        },
        "summary": summary,
        "results": all_results,
    }

    with open(
        args.output_dir
        / "summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            payload,
            f,
            indent=2,
            default=str,
        )

    print()
    print("=" * 96)
    print(
        "FINAL COMPARISON"
    )
    print("=" * 96)

    for condition in args.conditions:
        row = summary[
            condition
        ]

        print(
            f"{condition:20s} | "
            f"MRR "
            f"{row['mrr_mean']:.6f} ± "
            f"{row['mrr_sample_std']:.6f} | "
            f"Top1 "
            f"{row['top1_mean']:.6f} ± "
            f"{row['top1_sample_std']:.6f} | "
            f"Top3 "
            f"{row['top3_mean']:.6f} ± "
            f"{row['top3_sample_std']:.6f} | "
            f"same-task-error "
            f"{row['same_task_wrong_fraction_mean']:.4f} ± "
            f"{row['same_task_wrong_fraction_sample_std']:.4f}"
        )

    print()
    print(
        "Saved:",
        args.output_dir
        / "summary.json",
    )


if __name__ == "__main__":
    main()
