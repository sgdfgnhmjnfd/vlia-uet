from __future__ import annotations

import argparse
import json
import math
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

        if visual.shape != (960,):
            raise ValueError(
                f"Bad visual shape for {sample_id}: "
                f"{tuple(visual.shape)}"
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

        what_target = F.normalize(
            what_transform(
                what_960
            ).squeeze(0),
            dim=-1,
        )

        why_target = F.normalize(
            why_transform(
                why_960
            ).squeeze(0),
            dim=-1,
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
                "what_target": what_target,
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

    what_target = torch.stack(
        [x["what_target"] for x in rows]
    ).to(device)

    why_target = torch.stack(
        [x["why_target"] for x in rows]
    ).to(device)

    return visual, what_target, why_target


# ============================================================
# K-means prototypes in train WHAT space only
# ============================================================

@torch.no_grad()
def spherical_kmeans(
    x: torch.Tensor,
    k: int,
    iterations: int,
    seed: int,
):
    """
    Spherical k-means on normalized WHAT embeddings.
    Prototypes are learned ONLY from training WHAT targets.
    """
    if x.ndim != 2:
        raise ValueError(
            f"Expected [N,D], got {tuple(x.shape)}"
        )

    if k > x.shape[0]:
        raise ValueError(
            f"k={k} > number of samples={x.shape[0]}"
        )

    x = F.normalize(
        x,
        dim=-1,
    )

    g = torch.Generator(
        device=x.device
    )
    g.manual_seed(seed)

    init_idx = torch.randperm(
        x.shape[0],
        generator=g,
        device=x.device,
    )[:k]

    centers = x[
        init_idx
    ].clone()

    for _ in range(iterations):
        sim = x @ centers.T
        assign = torch.argmax(
            sim,
            dim=1,
        )

        new_centers = []

        for c in range(k):
            mask = assign == c

            if mask.any():
                center = x[
                    mask
                ].mean(dim=0)
            else:
                replacement = torch.randint(
                    low=0,
                    high=x.shape[0],
                    size=(1,),
                    generator=g,
                    device=x.device,
                ).item()

                center = x[
                    replacement
                ]

            new_centers.append(
                F.normalize(
                    center,
                    dim=0,
                )
            )

        new_centers = torch.stack(
            new_centers,
            dim=0,
        )

        if torch.allclose(
            centers,
            new_centers,
            atol=1e-6,
        ):
            centers = new_centers
            break

        centers = new_centers

    sim = x @ centers.T
    assign = torch.argmax(
        sim,
        dim=1,
    )

    counts = torch.bincount(
        assign,
        minlength=k,
    )

    return centers, assign, counts


# ============================================================
# Masks / losses
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


def prototype_soft_targets(
    what_target,
    prototypes,
    temperature,
):
    """
    Distribution of each ground-truth WHAT embedding over prototypes.
    """
    logits = (
        what_target
        @ prototypes.T
    ) / temperature

    return F.softmax(
        logits,
        dim=-1,
    )


def soft_ce_loss(
    logits,
    target_probs,
):
    return -(
        target_probs
        * F.log_softmax(
            logits,
            dim=-1,
        )
    ).sum(dim=-1).mean()


# ============================================================
# Models
# ============================================================

class StrongDirectWhy(nn.Module):
    """
    Strong WHY baseline:
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


class ExactWhatHead(nn.Module):
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


class SAPRModel(nn.Module):
    """
    Semantic Action Prototype Reasoning.

    direct:
        visual -> WHY

    multitask:
        visual -> WHY
        visual -> exact WHAT (auxiliary only)

    prototype_reasoning:
        visual -> prototype distribution
        soft weighted semantic prototype
        direct WHY + gated residual reasoning

    prototype_confidence:
        same as prototype_reasoning, but residual contribution is
        multiplied by normalized prototype confidence:
            confidence = 1 - H(p)/log(K)
    """

    def __init__(
        self,
        prototypes: torch.Tensor,
        hidden_dim=512,
        visual_state_dim=256,
    ):
        super().__init__()

        self.register_buffer(
            "prototypes",
            F.normalize(
                prototypes.detach().clone(),
                dim=-1,
            ),
        )

        k = prototypes.shape[0]

        self.direct_why = StrongDirectWhy(
            hidden_dim=hidden_dim,
            output_dim=256,
        )

        self.exact_what = ExactWhatHead(
            hidden_dim=hidden_dim,
            output_dim=256,
        )

        self.prototype_logits = nn.Sequential(
            nn.Linear(
                960,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_dim,
                k,
            ),
        )

        self.visual_state = nn.Sequential(
            nn.Linear(
                960,
                visual_state_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(
                visual_state_dim,
            ),
        )

        fusion_dim = (
            visual_state_dim
            + 256
        )

        self.reasoner = nn.Sequential(
            nn.Linear(
                fusion_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_dim,
                256,
            ),
            nn.LayerNorm(
                256,
            ),
        )

        self.gate = nn.Sequential(
            nn.Linear(
                fusion_dim,
                256,
            ),
            nn.Sigmoid(),
        )

    def predict_direct(
        self,
        visual,
    ):
        return self.direct_why(
            visual
        )

    def predict_exact_what(
        self,
        visual,
    ):
        return self.exact_what(
            visual
        )

    def prototype_distribution(
        self,
        visual,
        temperature,
    ):
        logits = self.prototype_logits(
            visual
        )

        probs = F.softmax(
            logits / temperature,
            dim=-1,
        )

        return logits, probs

    def prototype_state(
        self,
        probs,
    ):
        state = (
            probs
            @ self.prototypes
        )

        return F.normalize(
            state,
            dim=-1,
        )

    def confidence_from_probs(
        self,
        probs,
    ):
        k = probs.shape[-1]

        entropy = -(
            probs.clamp_min(
                1e-8
            ).log()
            * probs
        ).sum(dim=-1)

        confidence = (
            1.0
            - entropy
            / math.log(k)
        )

        return confidence.clamp(
            min=0.0,
            max=1.0,
        )

    def predict_sapr(
        self,
        visual,
        proto_temperature,
        confidence_scaled,
    ):
        direct = self.predict_direct(
            visual
        )

        proto_logits, probs = (
            self.prototype_distribution(
                visual,
                proto_temperature,
            )
        )

        proto_state = (
            self.prototype_state(
                probs
            )
        )

        visual_state = (
            self.visual_state(
                visual
            )
        )

        fused = torch.cat(
            [
                visual_state,
                proto_state,
            ],
            dim=-1,
        )

        residual = self.reasoner(
            fused
        )

        residual = F.normalize(
            residual,
            dim=-1,
        )

        learned_gate = self.gate(
            fused
        )

        confidence = (
            self.confidence_from_probs(
                probs
            )
        )

        if confidence_scaled:
            effective_gate = (
                learned_gate
                * confidence[:, None]
            )
        else:
            effective_gate = (
                learned_gate
            )

        final = F.normalize(
            direct
            + effective_gate
            * residual,
            dim=-1,
        )

        return {
            "why": final,
            "direct": direct,
            "proto_logits": proto_logits,
            "proto_probs": probs,
            "proto_state": proto_state,
            "learned_gate": learned_gate,
            "effective_gate": effective_gate,
            "confidence": confidence,
        }


# ============================================================
# Evaluation
# ============================================================

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
        positives = {
            j
            for j, candidate_label
            in enumerate(labels)
            if candidate_label == label
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
    sim = why_pred @ why_target.T

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
            ranking[i, 0].item()
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
    proto_temperature,
):
    model.eval()

    exact_what = (
        model.predict_exact_what(
            visual
        )
    )

    if condition in {
        "strong_direct",
        "multitask_strong",
    }:
        why_pred = (
            model.predict_direct(
                visual
            )
        )

        proto_metrics = {
            "proto_acc": None,
            "confidence_mean": None,
            "gate_mean": None,
            "effective_gate_mean": None,
        }

    elif condition in {
        "prototype_reasoning",
        "prototype_confidence",
    }:
        out = model.predict_sapr(
            visual=visual,
            proto_temperature=(
                proto_temperature
            ),
            confidence_scaled=(
                condition
                == "prototype_confidence"
            ),
        )

        why_pred = out["why"]

        target_proto = torch.argmax(
            what_target
            @ model.prototypes.T,
            dim=-1,
        )

        pred_proto = torch.argmax(
            out["proto_probs"],
            dim=-1,
        )

        proto_metrics = {
            "proto_acc": float(
                (
                    pred_proto
                    == target_proto
                ).float().mean().item()
            ),
            "confidence_mean": float(
                out["confidence"]
                .mean()
                .item()
            ),
            "gate_mean": float(
                out["learned_gate"]
                .mean()
                .item()
            ),
            "effective_gate_mean": float(
                out["effective_gate"]
                .mean()
                .item()
            ),
        }

    else:
        raise ValueError(
            f"Unknown condition: {condition}"
        )

    what_metrics = retrieval_metrics(
        exact_what,
        what_target,
        [x["what"] for x in rows],
    )

    why_metrics = retrieval_metrics(
        why_pred,
        why_target,
        [x["why"] for x in rows],
    )

    why_metrics.update(
        error_breakdown(
            why_pred,
            why_target,
            rows,
        )
    )

    return {
        "what": what_metrics,
        "why": why_metrics,
        **proto_metrics,
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
    prototypes,
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

    model = SAPRModel(
        prototypes=prototypes,
        hidden_dim=args.hidden_dim,
        visual_state_dim=(
            args.visual_state_dim
        ),
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

    proto_target_probs = (
        prototype_soft_targets(
            what_target=train_what_target,
            prototypes=prototypes,
            temperature=(
                args.proto_target_temperature
            ),
        )
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

        direct_why = (
            model.predict_direct(
                train_visual
            )
        )

        exact_what = (
            model.predict_exact_what(
                train_visual
            )
        )

        if condition == "strong_direct":
            why_pred = direct_why

            what_loss = torch.zeros(
                (),
                device=device,
            )

            proto_loss = torch.zeros(
                (),
                device=device,
            )

            confidence_reg = torch.zeros(
                (),
                device=device,
            )

        elif condition == "multitask_strong":
            why_pred = direct_why

            what_nce = (
                multi_positive_infonce(
                    exact_what,
                    train_what_target,
                    what_mask,
                    args.temperature,
                )
            )

            what_cos = (
                cosine_alignment_loss(
                    exact_what,
                    train_what_target,
                )
            )

            what_loss = (
                what_nce
                + args.cosine_weight
                * what_cos
            )

            proto_loss = torch.zeros(
                (),
                device=device,
            )

            confidence_reg = torch.zeros(
                (),
                device=device,
            )

        elif condition in {
            "prototype_reasoning",
            "prototype_confidence",
        }:
            out = model.predict_sapr(
                visual=train_visual,
                proto_temperature=(
                    args.proto_temperature
                ),
                confidence_scaled=(
                    condition
                    == "prototype_confidence"
                ),
            )

            why_pred = out["why"]

            what_loss = torch.zeros(
                (),
                device=device,
            )

            proto_loss = soft_ce_loss(
                out["proto_logits"],
                proto_target_probs,
            )

            if condition == "prototype_confidence":
                # Mild anti-collapse regularizer:
                # keep average confidence from becoming trivially zero.
                confidence_reg = (
                    F.relu(
                        args.min_confidence
                        - out["confidence"]
                    ).mean()
                )
            else:
                confidence_reg = torch.zeros(
                    (),
                    device=device,
                )

        else:
            raise ValueError(
                f"Unknown condition: {condition}"
            )

        why_nce = (
            multi_positive_infonce(
                why_pred,
                train_why_target,
                why_mask,
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

        total_loss = (
            why_loss
            + args.what_weight
            * what_loss
            + args.proto_weight
            * proto_loss
            + args.confidence_reg_weight
            * confidence_reg
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
                proto_temperature=(
                    args.proto_temperature
                ),
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
                    "loss_proto": float(
                        proto_loss.item()
                    ),
                    "loss_confidence_reg": float(
                        confidence_reg.item()
                    ),
                    "val": val_eval,
                }
            )

            mrr = val_eval[
                "why"
            ][
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

    if best_state is None:
        raise RuntimeError(
            "No checkpoint selected."
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
    fields = {
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

    optional = {
        "proto_acc": [
            x["best_val"]["proto_acc"]
            for x in rows
            if x["best_val"]["proto_acc"]
            is not None
        ],
        "confidence_mean": [
            x["best_val"]["confidence_mean"]
            for x in rows
            if x["best_val"]["confidence_mean"]
            is not None
        ],
        "gate_mean": [
            x["best_val"]["gate_mean"]
            for x in rows
            if x["best_val"]["gate_mean"]
            is not None
        ],
        "effective_gate_mean": [
            x["best_val"]["effective_gate_mean"]
            for x in rows
            if x["best_val"]["effective_gate_mean"]
            is not None
        ],
    }

    fields.update(
        {
            k: v
            for k, v in optional.items()
            if v
        }
    )

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
            "sapr_intention_v1"
        ),
    )

    parser.add_argument(
        "--conditions",
        nargs="+",
        default=[
            "strong_direct",
            "multitask_strong",
            "prototype_reasoning",
            "prototype_confidence",
        ],
        choices=[
            "strong_direct",
            "multitask_strong",
            "prototype_reasoning",
            "prototype_confidence",
        ],
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2],
    )

    parser.add_argument(
        "--num-prototypes",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--kmeans-iterations",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--kmeans-seed",
        type=int,
        default=123,
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
        "--proto-weight",
        type=float,
        default=0.3,
    )

    parser.add_argument(
        "--proto-temperature",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--proto-target-temperature",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--confidence-reg-weight",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--visual-state-dim",
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

    train_rows = load_stage_a_split(
        stage_a_root=args.stage_a_cache,
        temporal_root=args.temporal_cache,
        split="train",
        what_transform=what_transform,
        why_transform=why_transform,
    )

    val_rows = load_stage_a_split(
        stage_a_root=args.stage_a_cache,
        temporal_root=args.temporal_cache,
        split="val",
        what_transform=what_transform,
        why_transform=why_transform,
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

    prototypes, assignments, counts = (
        spherical_kmeans(
            x=train_what_target,
            k=args.num_prototypes,
            iterations=(
                args.kmeans_iterations
            ),
            seed=args.kmeans_seed,
        )
    )

    torch.save(
        {
            "prototypes": (
                prototypes.detach()
                .cpu()
            ),
            "assignments": (
                assignments.detach()
                .cpu()
            ),
            "counts": (
                counts.detach()
                .cpu()
            ),
            "num_prototypes": (
                args.num_prototypes
            ),
            "kmeans_seed": (
                args.kmeans_seed
            ),
        },
        args.output_dir
        / "prototypes.pt",
    )

    print("=" * 96)
    print(
        "SAPR: SEMANTIC ACTION PROTOTYPE REASONING"
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
        "num_prototypes:",
        args.num_prototypes,
    )

    print(
        "prototype counts:",
        counts.tolist(),
    )

    print(
        "proto_weight:",
        args.proto_weight,
    )

    print(
        "proto_temperature:",
        args.proto_temperature,
    )

    print(
        "proto_target_temperature:",
        args.proto_target_temperature,
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
                prototypes=prototypes,
                train_visual=train_visual,
                train_what_target=(
                    train_what_target
                ),
                train_why_target=(
                    train_why_target
                ),
                train_rows=train_rows,
                val_visual=val_visual,
                val_what_target=(
                    val_what_target
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
                    "prototypes": (
                        prototypes.detach()
                        .cpu()
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
                    result["history"],
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

            extras = ""

            if v[
                "proto_acc"
            ] is not None:
                extras = (
                    f" | proto_acc="
                    f"{v['proto_acc']:.4f}"
                    f" | conf="
                    f"{v['confidence_mean']:.4f}"
                    f" | gate="
                    f"{v['gate_mean']:.4f}"
                    f" | eff_gate="
                    f"{v['effective_gate_mean']:.4f}"
                )

            print(
                f"seed={seed} | "
                f"epoch={result['best_epoch']} | "
                f"WHY MRR="
                f"{v['why']['MRR']:.6f} | "
                f"WHY Top1="
                f"{v['why']['Top1']:.6f} | "
                f"WHAT MRR="
                f"{v['what']['MRR']:.6f} | "
                f"same-task-error="
                f"{v['why']['same_task_wrong_fraction']:.4f}"
                f"{extras}"
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
        "prototype_counts": (
            counts.detach()
            .cpu()
            .tolist()
        ),
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

        extras = ""

        if "proto_acc_mean" in row:
            extras += (
                f" | proto_acc "
                f"{row['proto_acc_mean']:.4f} ± "
                f"{row['proto_acc_sample_std']:.4f}"
            )

        if "confidence_mean_mean" in row:
            extras += (
                f" | conf "
                f"{row['confidence_mean_mean']:.4f} ± "
                f"{row['confidence_mean_sample_std']:.4f}"
            )

        if "effective_gate_mean_mean" in row:
            extras += (
                f" | eff_gate "
                f"{row['effective_gate_mean_mean']:.4f} ± "
                f"{row['effective_gate_mean_sample_std']:.4f}"
            )

        print(
            f"{condition:22s} | "
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
            f"{extras}"
        )

    print()
    print(
        "Saved:",
        args.output_dir
        / "summary.json",
    )


if __name__ == "__main__":
    main()
