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


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# PCA loading
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

    def transform(x: torch.Tensor) -> torch.Tensor:
        return (x.float().cpu() - mean) @ basis

    return transform


# ============================================================
# Data
# ============================================================

def load_structured_feature_map(
    temporal_root: Path,
    split: str,
):
    root = temporal_root / split

    if not root.exists():
        raise FileNotFoundError(
            f"Temporal split not found: {root}"
        )

    out = {}

    for p in sorted(root.rglob("*.pt")):
        x = torch.load(
            p,
            map_location="cpu",
            weights_only=False,
        )

        sample_id = str(
            x.get("sample_id", p.stem)
        )

        if "what_feature" not in x or "what" not in x:
            raise KeyError(
                f"Missing WHAT data in {p}"
            )

        out[sample_id] = {
            "what": str(x["what"]),
            "what_feature": (
                x["what_feature"]
                .detach()
                .float()
                .cpu()
            ),
        }

    if not out:
        raise RuntimeError(
            f"No structured samples found in {root}"
        )

    return out


def load_stage_a_split(
    stage_a_root: Path,
    temporal_root: Path,
    split: str,
    what_transform,
    why_transform,
):
    root = stage_a_root / split

    if not root.exists():
        raise FileNotFoundError(
            f"Stage-A split not found: {root}"
        )

    structured = load_structured_feature_map(
        temporal_root,
        split,
    )

    rows = []

    for p in sorted(root.rglob("*.pt")):
        x = torch.load(
            p,
            map_location="cpu",
            weights_only=False,
        )

        sample_id = str(x["sample_id"])

        if sample_id not in structured:
            raise KeyError(
                f"Missing WHAT data for sample {sample_id}"
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

        what_960 = (
            structured[sample_id]["what_feature"]
            .reshape(1, -1)
        )

        task = str(
            x.get(
                "task",
                x.get(
                    "event",
                    x.get(
                        "activity",
                        "",
                    ),
                ),
            )
        )

        if not task:
            raise RuntimeError(
                f"Missing task label for {sample_id}"
            )

        rows.append(
            {
                "sample_id": sample_id,
                "video_uid": str(
                    x.get(
                        "video_uid",
                        "",
                    )
                ),
                "task": task,
                "what": structured[
                    sample_id
                ]["what"],
                "why": str(x["why"]),
                "visual": visual,
                "what_target": F.normalize(
                    what_transform(
                        what_960
                    ).squeeze(0),
                    dim=-1,
                ),
                "why_target": F.normalize(
                    why_transform(
                        why_960
                    ).squeeze(0),
                    dim=-1,
                ),
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

    what_target = torch.stack(
        [x["what_target"] for x in rows]
    ).to(device)

    why_target = torch.stack(
        [x["why_target"] for x in rows]
    ).to(device)

    return visual, what_target, why_target


# ============================================================
# Masks / retrieval losses
# ============================================================

def build_label_mask(
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

    return -(
        log_prob
        * positive_mask.float()
    ).sum(dim=1).div(
        pos_count
    ).mean()


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
# Models
# ============================================================

class StrongBranch(nn.Module):
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


class RSRModel(nn.Module):
    """
    Relational Semantic Regularization.

    No hard WHAT -> WHY bottleneck.

    visual -> WHY embedding
    visual -> WHAT embedding

    The relation loss matches the pairwise geometry of
    predicted semantic embeddings to ground-truth semantic geometry.
    """

    def __init__(
        self,
        hidden_dim=512,
    ):
        super().__init__()

        self.why_head = StrongBranch(
            hidden_dim=hidden_dim,
            output_dim=256,
        )

        self.what_head = StrongBranch(
            hidden_dim=hidden_dim,
            output_dim=256,
        )

    def forward(self, visual):
        return {
            "why": self.why_head(
                visual
            ),
            "what": self.what_head(
                visual
            ),
        }


# ============================================================
# Relational semantic losses
# ============================================================

def centered_similarity(
    x,
):
    sim = x @ x.T

    mean_row = sim.mean(
        dim=1,
        keepdim=True,
    )

    mean_col = sim.mean(
        dim=0,
        keepdim=True,
    )

    mean_all = sim.mean()

    return (
        sim
        - mean_row
        - mean_col
        + mean_all
    )


def relation_geometry_loss(
    pred_why,
    pred_what,
    true_why,
    true_what,
):
    """
    Match relation structure, not exact intermediate semantics.

    1) WHY geometry:
       sim(pred WHY) ~ sim(true WHY)

    2) WHAT geometry:
       sim(pred WHAT) ~ sim(true WHAT)

    3) Cross semantic relation:
       pred WHAT <-> pred WHY
       should match
       true WHAT <-> true WHY
    """
    p_why = centered_similarity(
        pred_why
    )

    t_why = centered_similarity(
        true_why
    ).detach()

    p_what = centered_similarity(
        pred_what
    )

    t_what = centered_similarity(
        true_what
    ).detach()

    pred_cross = (
        pred_what
        @ pred_why.T
    )

    true_cross = (
        true_what
        @ true_why.T
    ).detach()

    loss_why_geom = F.mse_loss(
        p_why,
        t_why,
    )

    loss_what_geom = F.mse_loss(
        p_what,
        t_what,
    )

    loss_cross = F.mse_loss(
        pred_cross,
        true_cross,
    )

    total = (
        loss_why_geom
        + loss_what_geom
        + loss_cross
    ) / 3.0

    return (
        total,
        {
            "why_geom": (
                loss_why_geom
            ),
            "what_geom": (
                loss_what_geom
            ),
            "cross": (
                loss_cross
            ),
        },
    )


def rank_relation_loss(
    pred_why,
    pred_what,
    true_why,
    true_what,
    margin,
):
    """
    Pairwise ordering consistency:
    if true WHAT-WHY compatibility of j is higher than k,
    predicted compatibility should preserve that ordering.

    Uses one hard pair per query to keep the experiment lightweight.
    """
    with torch.no_grad():
        true_rel = (
            true_what
            @ true_why.T
        )

        best_idx = torch.argmax(
            true_rel,
            dim=1,
        )

        worst_idx = torch.argmin(
            true_rel,
            dim=1,
        )

    pred_rel = (
        pred_what
        @ pred_why.T
    )

    row = torch.arange(
        pred_rel.shape[0],
        device=pred_rel.device,
    )

    pos = pred_rel[
        row,
        best_idx,
    ]

    neg = pred_rel[
        row,
        worst_idx,
    ]

    return F.relu(
        margin
        - pos
        + neg
    ).mean()


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def retrieval_metrics(
    pred,
    target,
    labels,
):
    sim = (
        pred @ target.T
    )

    ranking = torch.argsort(
        sim,
        dim=1,
        descending=True,
    )

    ranks = []

    for i, label in enumerate(labels):
        positives = {
            j
            for j, x
            in enumerate(labels)
            if x == label
        }

        rank = None

        for r, j in enumerate(
            ranking[i].tolist(),
            start=1,
        ):
            if j in positives:
                rank = r
                break

        ranks.append(
            rank
        )

    ranks_t = torch.tensor(
        ranks,
        dtype=torch.float32,
        device=pred.device,
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
    }


@torch.no_grad()
def error_breakdown(
    why_pred,
    why_target,
    rows,
):
    sim = (
        why_pred
        @ why_target.T
    )

    ranking = torch.argsort(
        sim,
        dim=1,
        descending=True,
    )

    correct = 0
    same_task_wrong = 0
    cross_task_wrong = 0

    for i, row in enumerate(rows):
        j = int(
            ranking[i, 0]
            .item()
        )

        pred_row = rows[
            j
        ]

        if (
            pred_row["why"]
            == row["why"]
        ):
            correct += 1

        elif (
            pred_row["task"]
            == row["task"]
        ):
            same_task_wrong += 1

        else:
            cross_task_wrong += 1

    wrong = (
        same_task_wrong
        + cross_task_wrong
    )

    return {
        "top1_correct_count": (
            correct
        ),
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
    what_target,
    why_target,
    rows,
):
    model.eval()

    out = model(
        visual
    )

    why_metrics = (
        retrieval_metrics(
            out["why"],
            why_target,
            [x["why"] for x in rows],
        )
    )

    why_metrics.update(
        error_breakdown(
            out["why"],
            why_target,
            rows,
        )
    )

    what_metrics = (
        retrieval_metrics(
            out["what"],
            what_target,
            [x["what"] for x in rows],
        )
    )

    return {
        "why": why_metrics,
        "what": what_metrics,
    }


# ============================================================
# Training
# ============================================================

def clone_state_dict(
    model,
):
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
    train_what_target,
    train_why_target,
    train_rows,
    val_visual,
    val_what_target,
    val_why_target,
    val_rows,
    args,
):
    set_seed(
        seed
    )

    device = (
        train_visual.device
    )

    model = RSRModel(
        hidden_dim=(
            args.hidden_dim
        )
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    why_mask = build_label_mask(
        [x["why"] for x in train_rows],
        device,
    )

    what_mask = build_label_mask(
        [x["what"] for x in train_rows],
        device,
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

        out = model(
            train_visual
        )

        why_nce = (
            multi_positive_infonce(
                out["why"],
                train_why_target,
                why_mask,
                args.temperature,
            )
        )

        why_cos = (
            cosine_alignment_loss(
                out["why"],
                train_why_target,
            )
        )

        why_loss = (
            why_nce
            + args.cosine_weight
            * why_cos
        )

        if condition == "strong_direct":
            what_loss = torch.zeros(
                (),
                device=device,
            )

            relation_loss = torch.zeros(
                (),
                device=device,
            )

            rank_loss = torch.zeros(
                (),
                device=device,
            )

            relation_parts = {
                "why_geom": torch.zeros(
                    (),
                    device=device,
                ),
                "what_geom": torch.zeros(
                    (),
                    device=device,
                ),
                "cross": torch.zeros(
                    (),
                    device=device,
                ),
            }

        else:
            what_nce = (
                multi_positive_infonce(
                    out["what"],
                    train_what_target,
                    what_mask,
                    args.temperature,
                )
            )

            what_cos = (
                cosine_alignment_loss(
                    out["what"],
                    train_what_target,
                )
            )

            what_loss = (
                what_nce
                + args.cosine_weight
                * what_cos
            )

            if condition in {
                "relational",
                "relational_rank",
            }:
                (
                    relation_loss,
                    relation_parts,
                ) = relation_geometry_loss(
                    pred_why=out["why"],
                    pred_what=out["what"],
                    true_why=train_why_target,
                    true_what=train_what_target,
                )
            else:
                relation_loss = torch.zeros(
                    (),
                    device=device,
                )

                relation_parts = {
                    "why_geom": torch.zeros(
                        (),
                        device=device,
                    ),
                    "what_geom": torch.zeros(
                        (),
                        device=device,
                    ),
                    "cross": torch.zeros(
                        (),
                        device=device,
                    ),
                }

            if condition == "relational_rank":
                rank_loss = (
                    rank_relation_loss(
                        pred_why=out["why"],
                        pred_what=out["what"],
                        true_why=train_why_target,
                        true_what=train_what_target,
                        margin=args.rank_margin,
                    )
                )
            else:
                rank_loss = torch.zeros(
                    (),
                    device=device,
                )

        total_loss = (
            why_loss
            + args.what_weight
            * what_loss
            + args.relation_weight
            * relation_loss
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
            or epoch
            % args.eval_every
            == 0
            or epoch
            == args.epochs
        ):
            val_eval = (
                evaluate(
                    model=model,
                    visual=val_visual,
                    what_target=(
                        val_what_target
                    ),
                    why_target=(
                        val_why_target
                    ),
                    rows=val_rows,
                )
            )

            history.append(
                {
                    "epoch": epoch,
                    "loss_total": float(
                        total_loss.item()
                    ),
                    "loss_why": float(
                        why_loss.item()
                    ),
                    "loss_what": float(
                        what_loss.item()
                    ),
                    "loss_relation": float(
                        relation_loss.item()
                    ),
                    "loss_rank": float(
                        rank_loss.item()
                    ),
                    "relation_why_geom": float(
                        relation_parts[
                            "why_geom"
                        ].item()
                    ),
                    "relation_what_geom": float(
                        relation_parts[
                            "what_geom"
                        ].item()
                    ),
                    "relation_cross": float(
                        relation_parts[
                            "cross"
                        ].item()
                    ),
                    "val": val_eval,
                }
            )

            mrr = (
                val_eval[
                    "why"
                ][
                    "MRR"
                ]
            )

            if mrr > best_mrr:
                best_mrr = mrr
                best_epoch = (
                    epoch
                )
                best_eval = (
                    val_eval
                )
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
    metrics = {
        "why_mrr": [
            x["best_val"]["why"]["MRR"]
            for x in rows
        ],
        "why_top1": [
            x["best_val"]["why"]["Top1"]
            for x in rows
        ],
        "what_mrr": [
            x["best_val"]["what"]["MRR"]
            for x in rows
        ],
        "same_task_wrong_fraction": [
            x["best_val"]["why"][
                "same_task_wrong_fraction"
            ]
            for x in rows
        ],
    }

    out = {}

    for name, values in metrics.items():
        mean, std = (
            mean_std(
                values
            )
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
        "--stage-a-cache",
        type=Path,
        default=Path(
            "/media/dhqg/d1/datasets/"
            "egointent/cache/stage_a_v0"
        ),
    )

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
        "--what-pca",
        type=Path,
        default=Path(
            "/media/dhqg/d1/vlia_outputs/"
            "structured_intention_v1/"
            "what_pca_960_to_256.pt"
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
            "relational_semantic_v1"
        ),
    )

    parser.add_argument(
        "--conditions",
        nargs="+",
        default=[
            "strong_direct",
            "multitask_strong",
            "relational",
            "relational_rank",
        ],
        choices=[
            "strong_direct",
            "multitask_strong",
            "relational",
            "relational_rank",
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
        "--what-weight",
        type=float,
        default=0.3,
    )

    parser.add_argument(
        "--relation-weight",
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
        args.stage_a_cache,
        args.temporal_cache,
        args.what_pca,
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

    what_transform = (
        load_pca_transform(
            args.what_pca
        )
    )

    why_transform = (
        load_pca_transform(
            args.why_pca
        )
    )

    train_rows = (
        load_stage_a_split(
            stage_a_root=(
                args.stage_a_cache
            ),
            temporal_root=(
                args.temporal_cache
            ),
            split="train",
            what_transform=(
                what_transform
            ),
            why_transform=(
                why_transform
            ),
        )
    )

    val_rows = (
        load_stage_a_split(
            stage_a_root=(
                args.stage_a_cache
            ),
            temporal_root=(
                args.temporal_cache
            ),
            split="val",
            what_transform=(
                what_transform
            ),
            why_transform=(
                why_transform
            ),
        )
    )

    (
        train_visual,
        train_what_target,
        train_why_target,
    ) = stack_rows(
        train_rows,
        device,
    )

    (
        val_visual,
        val_what_target,
        val_why_target,
    ) = stack_rows(
        val_rows,
        device,
    )

    print("=" * 96)
    print(
        "RSR: RELATIONAL SEMANTIC REGULARIZATION"
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
        "what_weight:",
        args.what_weight,
    )

    print(
        "relation_weight:",
        args.relation_weight,
    )

    print(
        "rank_weight:",
        args.rank_weight,
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
                train_what_target=(
                    train_what_target
                ),
                train_why_target=(
                    train_why_target
                ),
                train_rows=(
                    train_rows
                ),
                val_visual=(
                    val_visual
                ),
                val_what_target=(
                    val_what_target
                ),
                val_why_target=(
                    val_why_target
                ),
                val_rows=(
                    val_rows
                ),
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
                f"{v['why']['MRR']:.6f} | "
                f"WHY Top1="
                f"{v['why']['Top1']:.6f} | "
                f"WHAT MRR="
                f"{v['what']['MRR']:.6f} | "
                f"same-task-error="
                f"{v['why']['same_task_wrong_fraction']:.4f}"
            )

    summary = {}

    for condition in args.conditions:
        rows = [
            x
            for x in all_results
            if x["condition"]
            == condition
        ]

        summary[
            condition
        ] = summarize(
            rows
        )

    payload = {
        "config": {
            **vars(args),
            "stage_a_cache": str(
                args.stage_a_cache
            ),
            "temporal_cache": str(
                args.temporal_cache
            ),
            "what_pca": str(
                args.what_pca
            ),
            "why_pca": str(
                args.why_pca
            ),
            "output_dir": str(
                args.output_dir
            ),
        },
        "summary": (
            summary
        ),
        "results": (
            all_results
        ),
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
            f"{condition:18s} | "
            f"WHY MRR "
            f"{row['why_mrr_mean']:.6f} ± "
            f"{row['why_mrr_sample_std']:.6f} | "
            f"WHY Top1 "
            f"{row['why_top1_mean']:.6f} ± "
            f"{row['why_top1_sample_std']:.6f} | "
            f"WHAT MRR "
            f"{row['what_mrr_mean']:.6f} ± "
            f"{row['what_mrr_sample_std']:.6f} | "
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
