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
            f"Unexpected PCA component shape in {path}: "
            f"{tuple(comp.shape)}"
        )

    if mean is None:
        mean = torch.zeros(
            960,
            dtype=torch.float32,
        )

    mean = mean.squeeze().float()

    if mean.numel() != 960:
        raise RuntimeError(
            f"Unexpected PCA mean shape in {path}: "
            f"{tuple(mean.shape)}"
        )

    mean = mean.reshape(1, 960)
    basis = basis.reshape(960, 256).float()

    def transform(x: torch.Tensor) -> torch.Tensor:
        return (
            x.float().cpu()
            - mean
        ) @ basis

    return transform


# ============================================================
# Data loading
# ============================================================

def find_pt_files(root: Path):
    return sorted(root.rglob("*.pt"))


def load_stage_a_metadata(
    stage_a_root: Path,
    split: str,
):
    root = stage_a_root / split

    if not root.exists():
        raise FileNotFoundError(
            f"Stage-A split not found: {root}"
        )

    metadata = {}

    for p in find_pt_files(root):
        x = torch.load(
            p,
            map_location="cpu",
            weights_only=False,
        )

        if "sample_id" not in x:
            continue

        sample_id = str(x["sample_id"])

        metadata[sample_id] = {
            "task": str(
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
            ),
            "video_uid": str(
                x.get(
                    "video_uid",
                    "",
                )
            ),
        }

    if not metadata:
        raise RuntimeError(
            f"No Stage-A metadata found in {root}"
        )

    return metadata


def load_structured_split(
    temporal_root: Path,
    stage_a_root: Path,
    split: str,
    what_transform,
    why_transform,
):
    temporal_dir = temporal_root / split

    if not temporal_dir.exists():
        raise FileNotFoundError(
            f"Temporal split not found: {temporal_dir}"
        )

    metadata = load_stage_a_metadata(
        stage_a_root,
        split,
    )

    rows = []

    required = {
        "sample_id",
        "frame_features",
        "what",
        "why",
        "what_feature",
        "why_feature",
    }

    for p in find_pt_files(temporal_dir):
        x = torch.load(
            p,
            map_location="cpu",
            weights_only=False,
        )

        missing = required - set(x.keys())

        if missing:
            raise KeyError(
                f"Missing keys {sorted(missing)} in {p}"
            )

        sample_id = str(x["sample_id"])

        if sample_id not in metadata:
            raise KeyError(
                f"sample_id={sample_id} exists in temporal cache "
                f"but not in Stage-A metadata for split={split}"
            )

        frames = (
            x["frame_features"]
            .detach()
            .float()
            .cpu()
        )

        if (
            frames.ndim != 2
            or frames.shape[-1] != 960
        ):
            raise ValueError(
                f"Invalid frame_features shape for {sample_id}: "
                f"{tuple(frames.shape)}"
            )

        what_960 = (
            x["what_feature"]
            .detach()
            .float()
            .cpu()
            .reshape(1, -1)
        )

        why_960 = (
            x["why_feature"]
            .detach()
            .float()
            .cpu()
            .reshape(1, -1)
        )

        if what_960.shape[-1] != 960:
            raise ValueError(
                f"Invalid WHAT feature shape for {sample_id}: "
                f"{tuple(what_960.shape)}"
            )

        if why_960.shape[-1] != 960:
            raise ValueError(
                f"Invalid WHY feature shape for {sample_id}: "
                f"{tuple(why_960.shape)}"
            )

        task = metadata[sample_id]["task"]

        if not task:
            raise RuntimeError(
                f"Missing task/event label for sample {sample_id}"
            )

        what_target = F.normalize(
            what_transform(what_960).squeeze(0),
            dim=-1,
        )

        why_target = F.normalize(
            why_transform(why_960).squeeze(0),
            dim=-1,
        )

        rows.append(
            {
                "sample_id": sample_id,
                "video_uid": metadata[sample_id]["video_uid"],
                "task": task,
                "what": str(x["what"]),
                "why": str(x["why"]),
                "visual": frames.mean(dim=0),
                "what_target": what_target,
                "why_target": why_target,
            }
        )

    if not rows:
        raise RuntimeError(
            f"No samples loaded from {temporal_dir}"
        )

    return rows


def stack_rows(
    rows,
    device,
):
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
# Masks
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


def build_counterfactual_candidate_mask(
    task_labels,
    what_labels,
    device,
):
    n = len(task_labels)

    mask = torch.zeros(
        (n, n),
        dtype=torch.bool,
        device=device,
    )

    for i in range(n):
        for j in range(n):
            if (
                task_labels[i] == task_labels[j]
                and what_labels[i] != what_labels[j]
            ):
                mask[i, j] = True

    return mask


# ============================================================
# Retrieval loss
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
# Model
# ============================================================

class VisualTrunk(nn.Module):
    def __init__(
        self,
        input_dim=960,
        hidden_dim=512,
        state_dim=256,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_dim,
                state_dim,
            ),
            nn.LayerNorm(
                state_dim
            ),
        )

    def forward(self, x):
        return self.net(x)


class HARModel(nn.Module):
    """
    Hierarchical Action-to-Intention Reasoning.

    Conditions:
      direct:
          visual state -> WHY

      multitask:
          visual state -> WHAT
          visual state -> WHY
          WHAT is auxiliary only

      hierarchical:
          visual state -> predicted WHAT
          [visual state, predicted WHAT] -> WHY

      hierarchical_cf:
          same as hierarchical, plus counterfactual consistency loss
    """

    def __init__(
        self,
        hidden_dim=512,
        state_dim=256,
        semantic_dim=256,
    ):
        super().__init__()

        self.visual_trunk = VisualTrunk(
            input_dim=960,
            hidden_dim=hidden_dim,
            state_dim=state_dim,
        )

        self.what_head = nn.Sequential(
            nn.Linear(
                state_dim,
                semantic_dim,
            ),
            nn.LayerNorm(
                semantic_dim
            ),
        )

        self.direct_why_head = nn.Sequential(
            nn.Linear(
                state_dim,
                semantic_dim,
            ),
            nn.LayerNorm(
                semantic_dim
            ),
        )

        self.reasoner = nn.Sequential(
            nn.Linear(
                state_dim + semantic_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_dim,
                semantic_dim,
            ),
            nn.LayerNorm(
                semantic_dim
            ),
        )

    def encode_visual(self, visual):
        return self.visual_trunk(
            visual
        )

    def predict_what(self, visual_state):
        z = self.what_head(
            visual_state
        )
        return F.normalize(
            z,
            dim=-1,
        )

    def predict_direct_why(
        self,
        visual_state,
    ):
        z = self.direct_why_head(
            visual_state
        )
        return F.normalize(
            z,
            dim=-1,
        )

    def reason_why(
        self,
        visual_state,
        what_repr,
    ):
        z = self.reasoner(
            torch.cat(
                [
                    visual_state,
                    what_repr,
                ],
                dim=-1,
            )
        )

        return F.normalize(
            z,
            dim=-1,
        )


# ============================================================
# Counterfactual sampling/loss
# ============================================================

def select_counterfactual_indices(
    what_pred,
    candidate_mask,
):
    """
    For each query, choose the hardest same-task,
    different-WHAT predicted representation.

    Hardness is based on cosine similarity in predicted WHAT space.
    Selection is discrete/no-gradient, but selected WHAT representations
    remain differentiable when used by the reasoner.
    """
    with torch.no_grad():
        sim = (
            what_pred.detach()
            @ what_pred.detach().T
        )

        sim = sim.masked_fill(
            ~candidate_mask,
            float("-inf"),
        )

        valid = candidate_mask.any(
            dim=1
        )

        indices = torch.argmax(
            sim,
            dim=1,
        )

    return indices, valid


def counterfactual_consistency_loss(
    model,
    visual_state,
    what_pred,
    why_target,
    candidate_mask,
    margin,
):
    cf_idx, valid = (
        select_counterfactual_indices(
            what_pred,
            candidate_mask,
        )
    )

    if not valid.any():
        return torch.zeros(
            (),
            device=visual_state.device,
        )

    pos_why = model.reason_why(
        visual_state,
        what_pred,
    )

    cf_what = what_pred[
        cf_idx
    ]

    neg_why = model.reason_why(
        visual_state,
        cf_what,
    )

    pos_score = (
        pos_why
        * why_target
    ).sum(dim=-1)

    neg_score = (
        neg_why
        * why_target
    ).sum(dim=-1)

    losses = F.relu(
        margin
        - pos_score
        + neg_score
    )

    return losses[
        valid
    ].mean()


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def predict_condition(
    model,
    visual,
    condition,
):
    model.eval()

    visual_state = model.encode_visual(
        visual
    )

    if condition in {
        "direct",
        "multitask",
    }:
        why_pred = (
            model.predict_direct_why(
                visual_state
            )
        )

    elif condition in {
        "hierarchical",
        "hierarchical_cf",
    }:
        what_pred = (
            model.predict_what(
                visual_state
            )
        )

        why_pred = model.reason_why(
            visual_state,
            what_pred,
        )

    else:
        raise ValueError(
            f"Unknown condition: {condition}"
        )

    what_pred = model.predict_what(
        visual_state
    )

    return (
        what_pred,
        why_pred,
    )


@torch.no_grad()
def retrieval_metrics(
    pred,
    target,
    labels,
):
    sim = pred @ target.T

    ranking = torch.argsort(
        sim,
        dim=1,
        descending=True,
    )

    ranks = []

    for i, label in enumerate(labels):
        positive = {
            j
            for j, x in enumerate(labels)
            if x == label
        }

        rank = None

        for r, j in enumerate(
            ranking[i].tolist(),
            start=1,
        ):
            if j in positive:
                rank = r
                break

        ranks.append(rank)

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
def why_error_breakdown(
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

        pred_row = rows[j]

        if pred_row["why"] == row["why"]:
            correct += 1

        elif pred_row["task"] == row["task"]:
            same_task_wrong += 1

        else:
            cross_task_wrong += 1

    wrong = (
        same_task_wrong
        + cross_task_wrong
    )

    return {
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
    what_target,
    why_target,
    rows,
    condition,
):
    what_labels = [
        x["what"]
        for x in rows
    ]

    why_labels = [
        x["why"]
        for x in rows
    ]

    what_pred, why_pred = (
        predict_condition(
            model,
            visual,
            condition,
        )
    )

    what_metrics = (
        retrieval_metrics(
            what_pred,
            what_target,
            what_labels,
        )
    )

    why_metrics = (
        retrieval_metrics(
            why_pred,
            why_target,
            why_labels,
        )
    )

    why_metrics.update(
        why_error_breakdown(
            why_pred,
            why_target,
            rows,
        )
    )

    return {
        "what": what_metrics,
        "why": why_metrics,
    }


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


def count_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


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
    set_seed(seed)

    device = train_visual.device

    model = HARModel(
        hidden_dim=args.hidden_dim,
        state_dim=args.state_dim,
        semantic_dim=256,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    train_what_labels = [
        x["what"]
        for x in train_rows
    ]

    train_why_labels = [
        x["why"]
        for x in train_rows
    ]

    train_task_labels = [
        x["task"]
        for x in train_rows
    ]

    what_positive_mask = (
        build_label_mask(
            train_what_labels,
            device,
        )
    )

    why_positive_mask = (
        build_label_mask(
            train_why_labels,
            device,
        )
    )

    cf_candidate_mask = (
        build_counterfactual_candidate_mask(
            train_task_labels,
            train_what_labels,
            device,
        )
    )

    valid_cf = int(
        cf_candidate_mask.any(
            dim=1
        ).sum().item()
    )

    best_mrr = -1.0
    best_state = None
    best_epoch = -1
    best_eval = None
    history = []

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        visual_state = (
            model.encode_visual(
                train_visual
            )
        )

        what_pred = (
            model.predict_what(
                visual_state
            )
        )

        direct_why_pred = (
            model.predict_direct_why(
                visual_state
            )
        )

        if condition in {
            "hierarchical",
            "hierarchical_cf",
        }:
            why_pred = (
                model.reason_why(
                    visual_state,
                    what_pred,
                )
            )
        else:
            why_pred = (
                direct_why_pred
            )

        why_nce = (
            multi_positive_infonce(
                why_pred,
                train_why_target,
                why_positive_mask,
                args.temperature,
            )
        )

        why_cos = (
            cosine_alignment_loss(
                why_pred,
                train_why_target,
            )
        )

        why_loss = (
            why_nce
            + args.cosine_weight
            * why_cos
        )

        if condition == "direct":
            what_loss = torch.zeros(
                (),
                device=device,
            )

        else:
            what_nce = (
                multi_positive_infonce(
                    what_pred,
                    train_what_target,
                    what_positive_mask,
                    args.temperature,
                )
            )

            what_cos = (
                cosine_alignment_loss(
                    what_pred,
                    train_what_target,
                )
            )

            what_loss = (
                what_nce
                + args.cosine_weight
                * what_cos
            )

        if condition == "hierarchical_cf":
            cf_loss = (
                counterfactual_consistency_loss(
                    model=model,
                    visual_state=visual_state,
                    what_pred=what_pred,
                    why_target=train_why_target,
                    candidate_mask=cf_candidate_mask,
                    margin=args.cf_margin,
                )
            )
        else:
            cf_loss = torch.zeros(
                (),
                device=device,
            )

        total_loss = (
            why_loss
            + args.what_weight
            * what_loss
            + args.cf_weight
            * cf_loss
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
                what_target=val_what_target,
                why_target=val_why_target,
                rows=val_rows,
                condition=condition,
            )

            row = {
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
                "loss_cf": float(
                    cf_loss.item()
                ),
                "val": val_eval,
            }

            history.append(row)

            current_mrr = (
                val_eval["why"]["MRR"]
            )

            if current_mrr > best_mrr:
                best_mrr = current_mrr
                best_epoch = epoch
                best_eval = val_eval
                best_state = (
                    clone_state_dict(
                        model
                    )
                )

    if best_state is None:
        raise RuntimeError(
            "No best checkpoint selected."
        )

    return {
        "condition": condition,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val": best_eval,
        "model_state_dict": best_state,
        "history": history,
        "trainable_parameters": (
            count_parameters(model)
        ),
        "num_valid_counterfactual_queries": (
            valid_cf
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
            x.std(ddof=1)
            if len(x) > 1
            else 0.0
        ),
    )


def summarize_condition(rows):
    why_mrr = [
        x["best_val"]["why"]["MRR"]
        for x in rows
    ]

    why_top1 = [
        x["best_val"]["why"]["Top1"]
        for x in rows
    ]

    what_mrr = [
        x["best_val"]["what"]["MRR"]
        for x in rows
    ]

    same_task_frac = [
        x["best_val"]["why"][
            "same_task_wrong_fraction"
        ]
        for x in rows
    ]

    why_mrr_m, why_mrr_s = (
        mean_std(why_mrr)
    )

    why_top1_m, why_top1_s = (
        mean_std(why_top1)
    )

    what_mrr_m, what_mrr_s = (
        mean_std(what_mrr)
    )

    same_m, same_s = (
        mean_std(same_task_frac)
    )

    return {
        "why_mrr_values": why_mrr,
        "why_mrr_mean": why_mrr_m,
        "why_mrr_sample_std": why_mrr_s,
        "why_top1_values": why_top1,
        "why_top1_mean": why_top1_m,
        "why_top1_sample_std": why_top1_s,
        "what_mrr_values": what_mrr,
        "what_mrr_mean": what_mrr_m,
        "what_mrr_sample_std": what_mrr_s,
        "same_task_wrong_fraction_values": (
            same_task_frac
        ),
        "same_task_wrong_fraction_mean": (
            same_m
        ),
        "same_task_wrong_fraction_sample_std": (
            same_s
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
            "har_intention_v1"
        ),
    )

    parser.add_argument(
        "--conditions",
        nargs="+",
        default=[
            "direct",
            "multitask",
            "hierarchical",
            "hierarchical_cf",
        ],
        choices=[
            "direct",
            "multitask",
            "hierarchical",
            "hierarchical_cf",
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
        "--cf-weight",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--cf-margin",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--state-dim",
        type=int,
        default=256,
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
        args.temporal_cache,
        args.stage_a_cache,
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
        load_structured_split(
            temporal_root=(
                args.temporal_cache
            ),
            stage_a_root=(
                args.stage_a_cache
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
        load_structured_split(
            temporal_root=(
                args.temporal_cache
            ),
            stage_a_root=(
                args.stage_a_cache
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

    print("=" * 88)
    print(
        "HAR: HIERARCHICAL ACTION-TO-INTENTION REASONING"
    )
    print("=" * 88)

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
        "cf_weight:",
        args.cf_weight,
    )

    print(
        "cf_margin:",
        args.cf_margin,
    )

    all_results = []

    for condition in args.conditions:
        print()
        print("#" * 88)
        print(
            "CONDITION:",
            condition,
        )
        print("#" * 88)

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

            checkpoint = {
                "condition": condition,
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
                "trainable_parameters": (
                    result[
                        "trainable_parameters"
                    ]
                ),
                "num_valid_counterfactual_queries": (
                    result[
                        "num_valid_counterfactual_queries"
                    ]
                ),
                "config": vars(args),
            }

            torch.save(
                checkpoint,
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
                "condition": condition,
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
                "trainable_parameters": (
                    result[
                        "trainable_parameters"
                    ]
                ),
                "num_valid_counterfactual_queries": (
                    result[
                        "num_valid_counterfactual_queries"
                    ]
                ),
            }

            all_results.append(
                compact
            )

            print(
                f"seed={seed} | "
                f"epoch={result['best_epoch']} | "
                f"WHY MRR="
                f"{result['best_val']['why']['MRR']:.6f} | "
                f"WHY Top1="
                f"{result['best_val']['why']['Top1']:.6f} | "
                f"WHAT MRR="
                f"{result['best_val']['what']['MRR']:.6f} | "
                f"same-task-error="
                f"{result['best_val']['why']['same_task_wrong_fraction']:.4f}"
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
        ] = summarize_condition(
            rows
        )

    payload = {
        "config": {
            **vars(args),
            "temporal_cache": str(
                args.temporal_cache
            ),
            "stage_a_cache": str(
                args.stage_a_cache
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
    print("=" * 88)
    print(
        "FINAL COMPARISON"
    )
    print("=" * 88)

    for condition in args.conditions:
        row = summary[
            condition
        ]

        print(
            f"{condition:16s} | "
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
