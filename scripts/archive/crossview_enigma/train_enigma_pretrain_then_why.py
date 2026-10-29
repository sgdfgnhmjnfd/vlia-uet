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
        mean = torch.zeros(960)

    mean = mean.squeeze().float()

    if mean.numel() != 960:
        raise RuntimeError(
            f"Unexpected PCA mean size: {mean.numel()}"
        )

    mean = mean.reshape(1, 960)
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
    video_ids = {str(x) for x in video_ids}
    grouped = defaultdict(dict)

    for p in sorted(cache_root.rglob("*.pt")):
        x = torch.load(p, map_location="cpu")

        video_uid = str(x["video_uid"])

        if video_uid not in video_ids:
            continue

        feat = x["frame_features"].float()

        if feat.ndim != 2 or feat.shape[-1] != 960:
            raise RuntimeError(
                f"Bad ENIGMA feature shape {tuple(feat.shape)} in {p}"
            )

        feat = feat.mean(dim=0)

        grouped[str(x["pair_group_id"])][str(x["view_type"])] = {
            "feat": feat,
            "video_uid": video_uid,
        }

    pairs = []

    for pair_id, views in sorted(grouped.items()):
        if "ego" not in views or "exo" not in views:
            continue

        pairs.append(
            {
                "pair_group_id": pair_id,
                "ego": views["ego"]["feat"],
                "exo": views["exo"]["feat"],
            }
        )

    if not pairs:
        raise RuntimeError(
            f"No complete ENIGMA pairs for {sorted(video_ids)}"
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

class PretrainFinetuneEncoder(nn.Module):
    """
    Stage 1:
        shared trunk + CV head are pretrained on ENIGMA.

    Stage 2:
        WHY head is trained on EgoIntent.
        Trunk can either:
          - be finetuned from the cross-view initialization, or
          - be frozen for an additional diagnostic.
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
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.cv_head = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

        self.why_head = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def shared(self, x):
        return self.trunk(x)

    def encode_cv(self, x):
        z = self.cv_head(
            self.shared(x)
        )

        return F.normalize(
            z,
            dim=-1,
        )

    def encode_why(self, x):
        z = self.why_head(
            self.shared(x)
        )

        return F.normalize(
            z,
            dim=-1,
        )


# ============================================================
# Losses
# ============================================================

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

    return {
        "Top1": (ranks <= 1).float().mean().item(),
        "Top3": (ranks <= 3).float().mean().item(),
        "Top5": (ranks <= 5).float().mean().item(),
        "MRR": (1.0 / ranks).mean().item(),
        "median_rank": ranks.median().item(),
        "mean_rank": ranks.mean().item(),
    }


@torch.no_grad()
def evaluate_crossview(
    model,
    ego,
    exo,
):
    model.eval()

    z_ego = model.encode_cv(
        ego
    )

    z_exo = model.encode_cv(
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

        return {
            "R@1": (ranks <= 1).float().mean().item(),
            "R@3": (ranks <= 3).float().mean().item(),
            "R@5": (ranks <= 5).float().mean().item(),
            "MRR": (1.0 / ranks).mean().item(),
        }

    return {
        "e2x": one_direction(sim),
        "x2e": one_direction(sim.T),
    }


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
        float(x.std(ddof=1))
        if len(x) > 1
        else 0.0
    )

    return mean, std


def clone_state_dict(module):
    return {
        k: v.detach().cpu().clone()
        for k, v in module.state_dict().items()
    }


# ============================================================
# Stage 1: ENIGMA pretraining
# ============================================================

def pretrain_crossview(
    seed,
    enigma_train_ego,
    enigma_train_exo,
    enigma_val_ego,
    enigma_val_exo,
    args,
):
    set_seed(seed)

    model = PretrainFinetuneEncoder(
        hidden_dim=args.hidden_dim,
        output_dim=256,
        dropout=args.dropout,
    ).to(enigma_train_ego.device)

    optimizer = torch.optim.AdamW(
        list(model.trunk.parameters())
        + list(model.cv_head.parameters()),
        lr=args.cv_lr,
        weight_decay=args.weight_decay,
    )

    best_state = None
    best_epoch = -1
    best_score = -1.0
    history = []

    for epoch in range(
        1,
        args.cv_epochs + 1,
    ):
        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        z_ego = model.encode_cv(
            enigma_train_ego
        )

        z_exo = model.encode_cv(
            enigma_train_exo
        )

        loss = symmetric_pair_infonce(
            z_ego,
            z_exo,
            args.cv_temperature,
        )

        loss.backward()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                list(model.trunk.parameters())
                + list(model.cv_head.parameters()),
                args.grad_clip,
            )

        optimizer.step()

        if (
            epoch == 1
            or epoch % args.cv_eval_every == 0
            or epoch == args.cv_epochs
        ):
            val_cv = evaluate_crossview(
                model,
                enigma_val_ego,
                enigma_val_exo,
            )

            score = 0.5 * (
                val_cv["e2x"]["MRR"]
                + val_cv["x2e"]["MRR"]
            )

            print(
                f"[CV PRETRAIN] "
                f"seed={seed} "
                f"epoch={epoch:03d} "
                f"loss={loss.item():.5f} "
                f"val_mrr="
                f"{val_cv['e2x']['MRR']:.4f}/"
                f"{val_cv['x2e']['MRR']:.4f}"
            )

            history.append(
                {
                    "epoch": epoch,
                    "loss": loss.item(),
                    "val_cv": val_cv,
                    "score": score,
                }
            )

            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_state = clone_state_dict(
                    model
                )

    if best_state is None:
        raise RuntimeError(
            "No CV checkpoint selected."
        )

    model.load_state_dict(
        best_state
    )

    final_cv = evaluate_crossview(
        model,
        enigma_val_ego,
        enigma_val_exo,
    )

    return {
        "model": model,
        "best_state": best_state,
        "best_epoch": best_epoch,
        "best_crossview": final_cv,
        "history": history,
    }


# ============================================================
# Stage 2: WHY finetuning
# ============================================================

def finetune_why(
    seed,
    pretrained_state,
    mode,
    train_visual,
    train_why_target,
    train_why_texts,
    val_visual,
    val_why_target,
    val_why_texts,
    args,
):
    set_seed(seed)

    model = PretrainFinetuneEncoder(
        hidden_dim=args.hidden_dim,
        output_dim=256,
        dropout=args.dropout,
    ).to(train_visual.device)

    # Load the pretrained trunk + CV head.
    model.load_state_dict(
        pretrained_state,
        strict=False,
    )

    # Reinitialize WHY head so it cannot inherit accidental state.
    for module in model.why_head.modules():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()

    if mode == "finetune":
        params = (
            list(model.trunk.parameters())
            + list(model.why_head.parameters())
        )

        for p in model.trunk.parameters():
            p.requires_grad = True

    elif mode == "freeze_trunk":
        for p in model.trunk.parameters():
            p.requires_grad = False

        params = list(
            model.why_head.parameters()
        )

    else:
        raise ValueError(
            f"Unknown mode: {mode}"
        )

    optimizer = torch.optim.AdamW(
        params,
        lr=args.why_lr,
        weight_decay=args.weight_decay,
    )

    positive_mask = build_positive_mask(
        train_why_texts,
        train_visual.device,
    )

    best_state = None
    best_epoch = -1
    best_val_mrr = -1.0
    history = []

    for epoch in range(
        1,
        args.why_epochs + 1,
    ):
        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        pred = model.encode_why(
            train_visual
        )

        loss_nce = multi_positive_infonce(
            pred,
            train_why_target,
            positive_mask,
            args.why_temperature,
        )

        cosine_loss = (
            1.0
            - (
                pred
                * train_why_target
            ).sum(dim=-1).mean()
        )

        loss = (
            loss_nce
            + args.cosine_weight
            * cosine_loss
        )

        loss.backward()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                params,
                args.grad_clip,
            )

        optimizer.step()

        if (
            epoch == 1
            or epoch % args.why_eval_every == 0
            or epoch == args.why_epochs
        ):
            val_why = evaluate_why(
                model,
                val_visual,
                val_why_target,
                val_why_texts,
            )

            print(
                f"[WHY {mode}] "
                f"seed={seed} "
                f"epoch={epoch:03d} "
                f"loss={loss.item():.5f} "
                f"VAL_MRR={val_why['MRR']:.6f}"
            )

            history.append(
                {
                    "epoch": epoch,
                    "loss": loss.item(),
                    "val_why": val_why,
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

                best_state = clone_state_dict(
                    model
                )

    if best_state is None:
        raise RuntimeError(
            "No WHY checkpoint selected."
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

    return {
        "best_state": best_state,
        "best_epoch": best_epoch,
        "best_val_why": final_why,
        "history": history,
    }


# ============================================================
# Baseline: random init WHY-only
# ============================================================

def train_random_init_why(
    seed,
    train_visual,
    train_why_target,
    train_why_texts,
    val_visual,
    val_why_target,
    val_why_texts,
    args,
):
    set_seed(seed)

    model = PretrainFinetuneEncoder(
        hidden_dim=args.hidden_dim,
        output_dim=256,
        dropout=args.dropout,
    ).to(train_visual.device)

    params = (
        list(model.trunk.parameters())
        + list(model.why_head.parameters())
    )

    optimizer = torch.optim.AdamW(
        params,
        lr=args.why_lr,
        weight_decay=args.weight_decay,
    )

    positive_mask = build_positive_mask(
        train_why_texts,
        train_visual.device,
    )

    best_state = None
    best_epoch = -1
    best_val_mrr = -1.0

    for epoch in range(
        1,
        args.why_epochs + 1,
    ):
        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        pred = model.encode_why(
            train_visual
        )

        loss_nce = multi_positive_infonce(
            pred,
            train_why_target,
            positive_mask,
            args.why_temperature,
        )

        cosine_loss = (
            1.0
            - (
                pred
                * train_why_target
            ).sum(dim=-1).mean()
        )

        loss = (
            loss_nce
            + args.cosine_weight
            * cosine_loss
        )

        loss.backward()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                params,
                args.grad_clip,
            )

        optimizer.step()

        if (
            epoch == 1
            or epoch % args.why_eval_every == 0
            or epoch == args.why_epochs
        ):
            val_why = evaluate_why(
                model,
                val_visual,
                val_why_target,
                val_why_texts,
            )

            if (
                val_why["MRR"]
                > best_val_mrr
            ):
                best_val_mrr = (
                    val_why["MRR"]
                )

                best_epoch = epoch

                best_state = clone_state_dict(
                    model
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

    return {
        "best_epoch": best_epoch,
        "best_val_why": final_why,
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
            "enigma_pretrain_then_why_v1"
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
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2],
    )

    parser.add_argument(
        "--cv-epochs",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--why-epochs",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--cv-eval-every",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--why-eval-every",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--cv-lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--why-lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--cv-temperature",
        type=float,
        default=0.07,
    )

    parser.add_argument(
        "--why-temperature",
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
    print("CROSS-VIEW PRETRAIN -> WHY FINETUNE")
    print("=" * 80)
    print("device:", device)
    print("EgoIntent train:", len(train_rows))
    print("EgoIntent val:", len(val_rows))
    print("ENIGMA train pairs:", len(enigma_train_pairs))
    print("ENIGMA val pairs:", len(enigma_val_pairs))
    print("seeds:", args.seeds)

    all_results = []

    for seed in args.seeds:
        print()
        print("#" * 80)
        print(f"SEED {seed}")
        print("#" * 80)

        seed_dir = (
            args.output_dir
            / f"seed_{seed}"
        )

        seed_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # Baseline
        # ----------------------------------------------------

        baseline = train_random_init_why(
            seed=seed,
            train_visual=train_visual,
            train_why_target=train_why_target,
            train_why_texts=train_why_texts,
            val_visual=val_visual,
            val_why_target=val_why_target,
            val_why_texts=val_why_texts,
            args=args,
        )

        print(
            f"[BASELINE] seed={seed} "
            f"best_epoch={baseline['best_epoch']} "
            f"WHY_MRR={baseline['best_val_why']['MRR']:.6f}"
        )

        # ----------------------------------------------------
        # CV pretraining
        # ----------------------------------------------------

        pretrain = pretrain_crossview(
            seed=seed,
            enigma_train_ego=enigma_train_ego,
            enigma_train_exo=enigma_train_exo,
            enigma_val_ego=enigma_val_ego,
            enigma_val_exo=enigma_val_exo,
            args=args,
        )

        print(
            f"[CV PRETRAIN BEST] seed={seed} "
            f"epoch={pretrain['best_epoch']} "
            f"MRR="
            f"{pretrain['best_crossview']['e2x']['MRR']:.4f}/"
            f"{pretrain['best_crossview']['x2e']['MRR']:.4f}"
        )

        # ----------------------------------------------------
        # Finetune trunk + WHY head
        # ----------------------------------------------------

        finetune = finetune_why(
            seed=seed,
            pretrained_state=pretrain["best_state"],
            mode="finetune",
            train_visual=train_visual,
            train_why_target=train_why_target,
            train_why_texts=train_why_texts,
            val_visual=val_visual,
            val_why_target=val_why_target,
            val_why_texts=val_why_texts,
            args=args,
        )

        # ----------------------------------------------------
        # Frozen-trunk diagnostic
        # ----------------------------------------------------

        freeze = finetune_why(
            seed=seed,
            pretrained_state=pretrain["best_state"],
            mode="freeze_trunk",
            train_visual=train_visual,
            train_why_target=train_why_target,
            train_why_texts=train_why_texts,
            val_visual=val_visual,
            val_why_target=val_why_target,
            val_why_texts=val_why_texts,
            args=args,
        )

        compact = {
            "seed": seed,
            "baseline": baseline,
            "cv_pretrain": {
                "best_epoch": pretrain["best_epoch"],
                "best_crossview": pretrain["best_crossview"],
            },
            "finetune": {
                "best_epoch": finetune["best_epoch"],
                "best_val_why": finetune["best_val_why"],
            },
            "freeze_trunk": {
                "best_epoch": freeze["best_epoch"],
                "best_val_why": freeze["best_val_why"],
            },
        }

        all_results.append(
            compact
        )

        torch.save(
            {
                "seed": seed,
                "cv_best_state": pretrain["best_state"],
                "finetune_best_state": finetune["best_state"],
                "freeze_best_state": freeze["best_state"],
                "results": compact,
                "config": vars(args),
            },
            seed_dir / "results.pt",
        )

        with open(
            seed_dir / "pretrain_history.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                pretrain["history"],
                f,
                indent=2,
            )

        with open(
            seed_dir / "finetune_history.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                finetune["history"],
                f,
                indent=2,
            )

        with open(
            seed_dir / "freeze_history.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                freeze["history"],
                f,
                indent=2,
            )

    # ========================================================
    # Summary
    # ========================================================

    baseline_mrr = [
        x["baseline"]["best_val_why"]["MRR"]
        for x in all_results
    ]

    finetune_mrr = [
        x["finetune"]["best_val_why"]["MRR"]
        for x in all_results
    ]

    freeze_mrr = [
        x["freeze_trunk"]["best_val_why"]["MRR"]
        for x in all_results
    ]

    baseline_top1 = [
        x["baseline"]["best_val_why"]["Top1"]
        for x in all_results
    ]

    finetune_top1 = [
        x["finetune"]["best_val_why"]["Top1"]
        for x in all_results
    ]

    freeze_top1 = [
        x["freeze_trunk"]["best_val_why"]["Top1"]
        for x in all_results
    ]

    cv_e2x = [
        x["cv_pretrain"]["best_crossview"]["e2x"]["MRR"]
        for x in all_results
    ]

    cv_x2e = [
        x["cv_pretrain"]["best_crossview"]["x2e"]["MRR"]
        for x in all_results
    ]

    def summarize(values):
        m, s = metric_mean_std(values)

        return {
            "values": values,
            "mean": m,
            "sample_std": s,
        }

    summary = {
        "baseline_why_mrr": summarize(
            baseline_mrr
        ),
        "pretrain_finetune_why_mrr": summarize(
            finetune_mrr
        ),
        "pretrain_freeze_why_mrr": summarize(
            freeze_mrr
        ),
        "baseline_why_top1": summarize(
            baseline_top1
        ),
        "pretrain_finetune_why_top1": summarize(
            finetune_top1
        ),
        "pretrain_freeze_why_top1": summarize(
            freeze_top1
        ),
        "cv_pretrain_e2x_mrr": summarize(
            cv_e2x
        ),
        "cv_pretrain_x2e_mrr": summarize(
            cv_x2e
        ),
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

    b = summary[
        "baseline_why_mrr"
    ]

    ft = summary[
        "pretrain_finetune_why_mrr"
    ]

    fr = summary[
        "pretrain_freeze_why_mrr"
    ]

    print(
        "Random-init WHY-only: "
        f"{b['mean']:.6f} ± "
        f"{b['sample_std']:.6f} | "
        f"{[round(x, 6) for x in b['values']]}"
    )

    print(
        "CV pretrain -> WHY finetune: "
        f"{ft['mean']:.6f} ± "
        f"{ft['sample_std']:.6f} | "
        f"{[round(x, 6) for x in ft['values']]}"
    )

    print(
        "CV pretrain -> freeze trunk + WHY head: "
        f"{fr['mean']:.6f} ± "
        f"{fr['sample_std']:.6f} | "
        f"{[round(x, 6) for x in fr['values']]}"
    )

    cv1 = summary[
        "cv_pretrain_e2x_mrr"
    ]

    cv2 = summary[
        "cv_pretrain_x2e_mrr"
    ]

    print(
        "CV pretrain held-out MRR: "
        f"E2X {cv1['mean']:.6f} ± "
        f"{cv1['sample_std']:.6f} | "
        f"X2E {cv2['mean']:.6f} ± "
        f"{cv2['sample_std']:.6f}"
    )

    print()
    print(
        "Saved summary:",
        args.output_dir / "summary.json",
    )


if __name__ == "__main__":
    main()
