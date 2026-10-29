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

        if comp is None:
            for container_key in [
                "state_dict",
                "pca",
                "transform",
            ]:
                nested = obj.get(
                    container_key
                )

                if not isinstance(
                    nested,
                    dict,
                ):
                    continue

                for k in mean_keys:
                    if (
                        k in nested
                        and torch.is_tensor(
                            nested[k]
                        )
                    ):
                        mean = (
                            nested[k]
                            .float()
                        )
                        break

                for k in comp_keys:
                    if (
                        k in nested
                        and torch.is_tensor(
                            nested[k]
                        )
                    ):
                        comp = (
                            nested[k]
                            .float()
                        )
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
            f"Unexpected PCA shape: {tuple(comp.shape)}"
        )

    if mean is None:
        mean = torch.zeros(
            960,
            dtype=torch.float32,
        )

    mean = mean.squeeze().float()

    if mean.numel() != 960:
        raise RuntimeError(
            f"Unexpected PCA mean size: {mean.numel()}"
        )

    mean = mean.reshape(
        1,
        960,
    )

    basis = basis.reshape(
        960,
        256,
    ).float()

    def transform(x):
        return (
            x.float().cpu()
            - mean
        ) @ basis

    return transform


# ============================================================
# Data
# ============================================================

def load_stage_a_split(
    cache_root: Path,
    split: str,
):
    root = cache_root / split
    rows = []

    for p in sorted(
        root.glob("*.pt")
    ):
        x = torch.load(
            p,
            map_location="cpu",
            weights_only=False,
        )

        visual = (
            x["visual_feature"]
            .float()
        )

        why_feature = (
            x["why_feature"]
            .float()
        )

        if visual.shape != (960,):
            raise RuntimeError(
                f"Bad visual shape "
                f"{tuple(visual.shape)} "
                f"in {p}"
            )

        if why_feature.shape != (960,):
            raise RuntimeError(
                f"Bad WHY shape "
                f"{tuple(why_feature.shape)} "
                f"in {p}"
            )

        rows.append(
            {
                "sample_id": str(
                    x["sample_id"]
                ),
                "video_uid": str(
                    x["video_uid"]
                ),
                "task": str(
                    x.get(
                        "task",
                        "",
                    )
                ),
                "why": str(
                    x["why"]
                ),
                "visual": visual,
                "why_feature": (
                    why_feature
                ),
            }
        )

    if not rows:
        raise RuntimeError(
            f"No samples found in {root}"
        )

    return rows


def stack_split(
    rows,
    why_transform,
    device,
):
    visual = torch.stack(
        [
            x["visual"]
            for x in rows
        ]
    ).to(device)

    why_960 = torch.stack(
        [
            x["why_feature"]
            for x in rows
        ]
    )

    why_target = (
        why_transform(
            why_960
        )
        .to(device)
    )

    why_target = F.normalize(
        why_target,
        dim=-1,
    )

    why_labels = [
        x["why"]
        for x in rows
    ]

    task_labels = [
        x["task"]
        for x in rows
    ]

    return (
        visual,
        why_target,
        why_labels,
        task_labels,
    )


# ============================================================
# Model
# ============================================================

class IntentionMLP(nn.Module):
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
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                hidden_dim,
                output_dim,
            ),
            nn.LayerNorm(
                output_dim
            ),
        )

    def forward(self, x):
        z = self.net(
            x
        )

        return F.normalize(
            z,
            dim=-1,
        )


# ============================================================
# Masks
# ============================================================

def build_global_positive_mask(
    why_labels,
    device,
):
    n = len(
        why_labels
    )

    mask = torch.zeros(
        (n, n),
        dtype=torch.bool,
        device=device,
    )

    groups = defaultdict(
        list
    )

    for i, why in enumerate(
        why_labels
    ):
        groups[why].append(
            i
        )

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


def build_same_task_masks(
    why_labels,
    task_labels,
    device,
):
    n = len(
        why_labels
    )

    same_task = torch.zeros(
        (n, n),
        dtype=torch.bool,
        device=device,
    )

    positive = torch.zeros(
        (n, n),
        dtype=torch.bool,
        device=device,
    )

    for i in range(n):
        for j in range(n):
            if (
                task_labels[i]
                == task_labels[j]
            ):
                same_task[i, j] = True

                if (
                    why_labels[i]
                    == why_labels[j]
                ):
                    positive[i, j] = True

    valid_query = (
        (
            same_task
            & ~positive
        ).sum(dim=1)
        > 0
    )

    return (
        same_task,
        positive,
        valid_query,
    )


# ============================================================
# Losses
# ============================================================

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


def intra_task_hard_negative_loss(
    pred,
    target,
    same_task_mask,
    positive_mask,
    valid_query,
    temperature,
    topk,
):
    """
    Fine-grained intra-task loss.

    For each query:
      - positives: same WHY within the same task
      - negatives: different WHY within the same task
      - choose current top-k hardest same-task negatives

    Loss is multi-positive log-softmax over
    positives + selected same-task hard negatives.
    """
    similarity = (
        pred @ target.T
    )

    losses = []

    valid_indices = torch.nonzero(
        valid_query,
        as_tuple=False,
    ).flatten()

    for i in valid_indices.tolist():
        pos_mask = (
            positive_mask[i]
        )

        neg_mask = (
            same_task_mask[i]
            & ~positive_mask[i]
        )

        pos_idx = torch.nonzero(
            pos_mask,
            as_tuple=False,
        ).flatten()

        neg_idx = torch.nonzero(
            neg_mask,
            as_tuple=False,
        ).flatten()

        if (
            pos_idx.numel() == 0
            or neg_idx.numel() == 0
        ):
            continue

        neg_scores = (
            similarity[
                i,
                neg_idx,
            ]
        )

        k = min(
            int(topk),
            int(
                neg_idx.numel()
            ),
        )

        hard_local = torch.topk(
            neg_scores,
            k=k,
            largest=True,
        ).indices

        hard_neg_idx = (
            neg_idx[
                hard_local
            ]
        )

        candidate_idx = torch.cat(
            [
                pos_idx,
                hard_neg_idx,
            ],
            dim=0,
        )

        logits = (
            similarity[
                i,
                candidate_idx,
            ]
            / temperature
        )

        pos_count = int(
            pos_idx.numel()
        )

        log_denom = torch.logsumexp(
            logits,
            dim=0,
        )

        log_pos = torch.logsumexp(
            logits[
                :pos_count
            ],
            dim=0,
        )

        losses.append(
            -(
                log_pos
                - log_denom
            )
        )

    if not losses:
        return torch.zeros(
            (),
            device=pred.device,
        )

    return torch.stack(
        losses
    ).mean()


# ============================================================
# Evaluation
# ============================================================

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

    sim = (
        pred
        @ why_target.T
    )

    ranking = torch.argsort(
        sim,
        dim=1,
        descending=True,
    )

    ranks = []
    same_task_wrong = 0
    cross_task_wrong = 0
    top1_correct = 0

    for i, row in enumerate(
        rows
    ):
        positives = {
            j
            for j, gallery
            in enumerate(rows)
            if (
                gallery["why"]
                == row["why"]
            )
        }

        first_rank = None

        for rank, j in enumerate(
            ranking[i].tolist(),
            start=1,
        ):
            if j in positives:
                first_rank = rank
                break

        ranks.append(
            first_rank
        )

        top1_idx = int(
            ranking[i, 0]
            .item()
        )

        top1_row = rows[
            top1_idx
        ]

        if (
            top1_row["why"]
            == row["why"]
        ):
            top1_correct += 1

        elif (
            top1_row["task"]
            == row["task"]
        ):
            same_task_wrong += 1

        else:
            cross_task_wrong += 1

    ranks = torch.tensor(
        ranks,
        dtype=torch.float32,
        device=visual.device,
    )

    wrong = (
        same_task_wrong
        + cross_task_wrong
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
        "median_rank": (
            ranks.median().item()
        ),
        "mean_rank": (
            ranks.mean().item()
        ),
        "top1_correct_count": (
            top1_correct
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
    lambda_intra,
    seed,
    train_visual,
    train_why_target,
    train_why_labels,
    train_task_labels,
    val_visual,
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

    model = IntentionMLP(
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    global_positive_mask = (
        build_global_positive_mask(
            train_why_labels,
            device,
        )
    )

    (
        same_task_mask,
        intra_positive_mask,
        valid_intra_query,
    ) = build_same_task_masks(
        train_why_labels,
        train_task_labels,
        device,
    )

    num_valid_intra = int(
        valid_intra_query
        .sum()
        .item()
    )

    best_state = None
    best_epoch = -1
    best_mrr = -1.0
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

        global_nce = (
            multi_positive_infonce(
                pred,
                train_why_target,
                global_positive_mask,
                args.temperature,
            )
        )

        cosine_loss = (
            1.0
            - (
                pred
                * train_why_target
            ).sum(dim=-1).mean()
        )

        global_loss = (
            global_nce
            + args.cosine_weight
            * cosine_loss
        )

        if lambda_intra > 0:
            intra_loss = (
                intra_task_hard_negative_loss(
                    pred=pred,
                    target=train_why_target,
                    same_task_mask=same_task_mask,
                    positive_mask=intra_positive_mask,
                    valid_query=valid_intra_query,
                    temperature=(
                        args.intra_temperature
                    ),
                    topk=args.hard_negative_k,
                )
            )
        else:
            intra_loss = torch.zeros(
                (),
                device=device,
            )

        total_loss = (
            global_loss
            + lambda_intra
            * intra_loss
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
            val_metrics = evaluate(
                model,
                val_visual,
                val_why_target,
                val_rows,
            )

            history.append(
                {
                    "epoch": epoch,
                    "total_loss": (
                        total_loss.item()
                    ),
                    "global_loss": (
                        global_loss.item()
                    ),
                    "intra_loss": (
                        intra_loss.item()
                    ),
                    "val": (
                        val_metrics
                    ),
                }
            )

            if (
                val_metrics["MRR"]
                > best_mrr
            ):
                best_mrr = (
                    val_metrics["MRR"]
                )

                best_epoch = epoch

                best_state = (
                    clone_state_dict(
                        model
                    )
                )

    if best_state is None:
        raise RuntimeError(
            "No checkpoint selected."
        )

    model.load_state_dict(
        best_state
    )

    final_metrics = evaluate(
        model,
        val_visual,
        val_why_target,
        val_rows,
    )

    return {
        "lambda_intra": (
            lambda_intra
        ),
        "seed": seed,
        "best_epoch": (
            best_epoch
        ),
        "best_val": (
            final_metrics
        ),
        "history": (
            history
        ),
        "state_dict": (
            best_state
        ),
        "num_valid_intra_queries": (
            num_valid_intra
        ),
    }


# ============================================================
# Summary
# ============================================================

def mean_std(
    values,
):
    x = np.asarray(
        values,
        dtype=np.float64,
    )

    return (
        float(
            x.mean()
        ),
        float(
            x.std(
                ddof=1
            )
            if len(x) > 1
            else 0.0
        ),
    )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path(
            "/media/dhqg/d1/"
            "datasets/egointent/"
            "cache/stage_a_v0"
        ),
    )

    parser.add_argument(
        "--why-pca",
        type=Path,
        default=Path(
            "/media/dhqg/d1/"
            "vlia_outputs/"
            "stage_a_clean_v1/"
            "why_pca_960_to_256.pt"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/media/dhqg/d1/"
            "vlia_outputs/"
            "intra_task_intention_v1"
        ),
    )

    parser.add_argument(
        "--lambdas",
        nargs="+",
        type=float,
        default=[
            0.0,
            0.3,
            1.0,
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
        "--intra-temperature",
        type=float,
        default=0.07,
    )

    parser.add_argument(
        "--hard-negative-k",
        type=int,
        default=16,
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
        load_stage_a_split(
            args.cache_root,
            "train",
        )
    )

    val_rows = (
        load_stage_a_split(
            args.cache_root,
            "val",
        )
    )

    (
        train_visual,
        train_why_target,
        train_why_labels,
        train_task_labels,
    ) = stack_split(
        train_rows,
        why_transform,
        device,
    )

    (
        val_visual,
        val_why_target,
        _,
        _,
    ) = stack_split(
        val_rows,
        why_transform,
        device,
    )

    print("=" * 80)
    print(
        "INTRA-TASK INTENTION DISCRIMINATION"
    )
    print("=" * 80)
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
            set(
                train_task_labels
            )
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
        "lambdas:",
        args.lambdas,
    )
    print(
        "hard-negative-k:",
        args.hard_negative_k,
    )
    print(
        "seeds:",
        args.seeds,
    )

    all_results = []

    for lambda_intra in args.lambdas:
        print()
        print("#" * 80)
        print(
            "LAMBDA_INTRA =",
            lambda_intra,
        )
        print("#" * 80)

        condition_dir = (
            args.output_dir
            / (
                f"lambda_"
                f"{lambda_intra:g}"
            )
        )

        condition_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for seed in args.seeds:
            result = train_one(
                lambda_intra=(
                    lambda_intra
                ),
                seed=seed,
                train_visual=(
                    train_visual
                ),
                train_why_target=(
                    train_why_target
                ),
                train_why_labels=(
                    train_why_labels
                ),
                train_task_labels=(
                    train_task_labels
                ),
                val_visual=(
                    val_visual
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
                    "lambda_intra": (
                        lambda_intra
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
                            "state_dict"
                        ]
                    ),
                    "num_valid_intra_queries": (
                        result[
                            "num_valid_intra_queries"
                        ]
                    ),
                    "config": (
                        vars(args)
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
                "lambda_intra": (
                    lambda_intra
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
                "num_valid_intra_queries": (
                    result[
                        "num_valid_intra_queries"
                    ]
                ),
            }

            all_results.append(
                compact
            )

            print(
                f"seed={seed} "
                f"best_epoch="
                f"{result['best_epoch']} "
                f"MRR="
                f"{result['best_val']['MRR']:.6f} "
                f"Top1="
                f"{result['best_val']['Top1']:.6f} "
                f"same_task_wrong_fraction="
                f"{result['best_val']['same_task_wrong_fraction']:.4f}"
            )

    summary = {}

    for lambda_intra in args.lambdas:
        rows = [
            x
            for x in all_results
            if float(
                x["lambda_intra"]
            )
            == float(
                lambda_intra
            )
        ]

        mrr = [
            x["best_val"]["MRR"]
            for x in rows
        ]

        top1 = [
            x["best_val"]["Top1"]
            for x in rows
        ]

        same_frac = [
            x[
                "best_val"
            ][
                "same_task_wrong_fraction"
            ]
            for x in rows
        ]

        mrr_m, mrr_s = (
            mean_std(
                mrr
            )
        )

        top1_m, top1_s = (
            mean_std(
                top1
            )
        )

        sf_m, sf_s = (
            mean_std(
                same_frac
            )
        )

        summary[
            str(
                lambda_intra
            )
        ] = {
            "mrr_values": (
                mrr
            ),
            "mrr_mean": (
                mrr_m
            ),
            "mrr_sample_std": (
                mrr_s
            ),
            "top1_values": (
                top1
            ),
            "top1_mean": (
                top1_m
            ),
            "top1_sample_std": (
                top1_s
            ),
            "same_task_wrong_fraction_values": (
                same_frac
            ),
            "same_task_wrong_fraction_mean": (
                sf_m
            ),
            "same_task_wrong_fraction_sample_std": (
                sf_s
            ),
        }

    with open(
        args.output_dir
        / "summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "config": (
                    vars(args)
                ),
                "summary": (
                    summary
                ),
                "results": (
                    all_results
                ),
            },
            f,
            indent=2,
            default=str,
        )

    print()
    print("=" * 80)
    print(
        "FINAL COMPARISON"
    )
    print("=" * 80)

    for lambda_intra in args.lambdas:
        row = summary[
            str(
                lambda_intra
            )
        ]

        print(
            f"lambda={lambda_intra:g} | "
            f"MRR "
            f"{row['mrr_mean']:.6f} ± "
            f"{row['mrr_sample_std']:.6f} | "
            f"Top1 "
            f"{row['top1_mean']:.6f} ± "
            f"{row['top1_sample_std']:.6f} | "
            f"same-task-error-frac "
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
