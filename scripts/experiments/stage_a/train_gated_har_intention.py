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
                f"Missing structured WHAT for sample {sample_id}"
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

        if why_960.shape[-1] != 960:
            raise ValueError(
                f"Bad WHY shape for {sample_id}"
            )

        if what_960.shape[-1] != 960:
            raise ValueError(
                f"Bad WHAT shape for {sample_id}"
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

    return (
        visual,
        what_target,
        why_target,
    )


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


def build_counterfactual_mask(
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
# Strong baseline + gated residual reasoning
# ============================================================

class StrongDirectWhy(nn.Module):
    """
    Exact WHY architecture used by the stronger controlled baseline:
        960 -> 512 -> GELU -> 256 -> LayerNorm -> L2
    """

    def __init__(
        self,
        input_dim=960,
        hidden_dim=512,
        output_dim=256,
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


class WhatPredictor(nn.Module):
    def __init__(
        self,
        input_dim=960,
        hidden_dim=512,
        output_dim=256,
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


class GatedResidualHAR(nn.Module):
    """
    Strong direct WHY path is always preserved.

    direct_why = f_direct(v)
    what       = f_what(v)
    reason     = f_reason([v_proj, what])
    gate       = sigmoid(f_gate([v_proj, what]))

    final WHY = normalize(direct_why + gate * reason)

    This avoids forcing WHY through an unreliable WHAT bottleneck.
    """

    def __init__(
        self,
        hidden_dim=512,
        visual_state_dim=256,
        semantic_dim=256,
    ):
        super().__init__()

        self.direct_why = StrongDirectWhy(
            input_dim=960,
            hidden_dim=hidden_dim,
            output_dim=semantic_dim,
        )

        self.what_predictor = WhatPredictor(
            input_dim=960,
            hidden_dim=hidden_dim,
            output_dim=semantic_dim,
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
            + semantic_dim
        )

        self.reasoner = nn.Sequential(
            nn.Linear(
                fusion_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_dim,
                semantic_dim,
            ),
            nn.LayerNorm(
                semantic_dim,
            ),
        )

        self.gate = nn.Sequential(
            nn.Linear(
                fusion_dim,
                semantic_dim,
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

    def predict_what(
        self,
        visual,
    ):
        return self.what_predictor(
            visual
        )

    def get_visual_state(
        self,
        visual,
    ):
        return self.visual_state(
            visual
        )

    def reason_from_what(
        self,
        visual_state,
        what_repr,
    ):
        fused = torch.cat(
            [
                visual_state,
                what_repr,
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

        gate = self.gate(
            fused
        )

        return residual, gate

    def predict_gated(
        self,
        visual,
        what_override=None,
    ):
        direct = self.predict_direct(
            visual
        )

        what_pred = self.predict_what(
            visual
        )

        visual_state = (
            self.get_visual_state(
                visual
            )
        )

        what_used = (
            what_pred
            if what_override is None
            else what_override
        )

        residual, gate = (
            self.reason_from_what(
                visual_state,
                what_used,
            )
        )

        final = F.normalize(
            direct
            + gate * residual,
            dim=-1,
        )

        return (
            final,
            direct,
            what_pred,
            gate,
        )


# ============================================================
# Counterfactual loss
# ============================================================

def select_cf_indices(
    what_pred,
    cf_mask,
):
    with torch.no_grad():
        sim = (
            what_pred.detach()
            @ what_pred.detach().T
        )

        sim = sim.masked_fill(
            ~cf_mask,
            float("-inf"),
        )

        valid = cf_mask.any(
            dim=1
        )

        idx = torch.argmax(
            sim,
            dim=1,
        )

    return idx, valid


def counterfactual_loss(
    model,
    visual,
    what_pred,
    why_target,
    cf_mask,
    margin,
):
    cf_idx, valid = (
        select_cf_indices(
            what_pred,
            cf_mask,
        )
    )

    if not valid.any():
        return torch.zeros(
            (),
            device=visual.device,
        )

    pos_final, _, _, _ = (
        model.predict_gated(
            visual
        )
    )

    cf_what = what_pred[
        cf_idx
    ]

    neg_final, _, _, _ = (
        model.predict_gated(
            visual,
            what_override=cf_what,
        )
    )

    pos_score = (
        pos_final
        * why_target
    ).sum(dim=-1)

    neg_score = (
        neg_final
        * why_target
    ).sum(dim=-1)

    loss = F.relu(
        margin
        - pos_score
        + neg_score
    )

    return loss[
        valid
    ].mean()


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
            for j, gallery_label
            in enumerate(labels)
            if gallery_label == label
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
            ranking[i, 0].item()
        )

        pred_row = rows[j]

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
    condition,
):
    model.eval()

    if condition in {
        "strong_direct",
        "multitask_strong",
    }:
        why_pred = (
            model.predict_direct(
                visual
            )
        )

        what_pred = (
            model.predict_what(
                visual
            )
        )

        gate_mean = None

    elif condition in {
        "gated_residual",
        "gated_residual_cf",
    }:
        (
            why_pred,
            _,
            what_pred,
            gate,
        ) = model.predict_gated(
            visual
        )

        gate_mean = float(
            gate.mean().item()
        )

    else:
        raise ValueError(
            f"Unknown condition: {condition}"
        )

    what_metrics = retrieval_metrics(
        what_pred,
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
        "what": (
            what_metrics
        ),
        "why": (
            why_metrics
        ),
        "gate_mean": (
            gate_mean
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

    model = GatedResidualHAR(
        hidden_dim=args.hidden_dim,
        visual_state_dim=(
            args.visual_state_dim
        ),
        semantic_dim=256,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    what_labels = [
        x["what"]
        for x in train_rows
    ]

    why_labels = [
        x["why"]
        for x in train_rows
    ]

    task_labels = [
        x["task"]
        for x in train_rows
    ]

    what_pos_mask = (
        build_label_mask(
            what_labels,
            device,
        )
    )

    why_pos_mask = (
        build_label_mask(
            why_labels,
            device,
        )
    )

    cf_mask = (
        build_counterfactual_mask(
            task_labels,
            what_labels,
            device,
        )
    )

    num_valid_cf = int(
        cf_mask.any(
            dim=1
        ).sum().item()
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

        what_pred = (
            model.predict_what(
                train_visual
            )
        )

        if condition in {
            "gated_residual",
            "gated_residual_cf",
        }:
            (
                why_pred,
                direct_why,
                what_pred,
                gate,
            ) = model.predict_gated(
                train_visual
            )
        else:
            why_pred = (
                direct_why
            )

            gate = None

        why_nce = (
            multi_positive_infonce(
                why_pred,
                train_why_target,
                why_pos_mask,
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

        if condition == "strong_direct":
            what_loss = torch.zeros(
                (),
                device=device,
            )

        else:
            what_nce = (
                multi_positive_infonce(
                    what_pred,
                    train_what_target,
                    what_pos_mask,
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

        if condition == "gated_residual_cf":
            cf_loss = (
                counterfactual_loss(
                    model=model,
                    visual=train_visual,
                    what_pred=what_pred,
                    why_target=train_why_target,
                    cf_mask=cf_mask,
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
                    "loss_cf": float(
                        cf_loss.item()
                    ),
                    "gate_train_mean": (
                        float(
                            gate.mean().item()
                        )
                        if gate is not None
                        else None
                    ),
                    "val": (
                        val_eval
                    ),
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
        "best_epoch": (
            best_epoch
        ),
        "best_val": (
            best_eval
        ),
        "model_state_dict": (
            best_state
        ),
        "history": (
            history
        ),
        "num_valid_cf_queries": (
            num_valid_cf
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

    gate_values = [
        x["best_val"]["gate_mean"]
        for x in rows
        if x["best_val"]["gate_mean"]
        is not None
    ]

    out = {}

    for name, values in metrics.items():
        m, s = mean_std(
            values
        )

        out[
            f"{name}_values"
        ] = values

        out[
            f"{name}_mean"
        ] = m

        out[
            f"{name}_sample_std"
        ] = s

    if gate_values:
        m, s = mean_std(
            gate_values
        )

        out[
            "gate_mean_values"
        ] = gate_values

        out[
            "gate_mean_mean"
        ] = m

        out[
            "gate_mean_sample_std"
        ] = s

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
            "gated_har_intention_v1"
        ),
    )

    parser.add_argument(
        "--conditions",
        nargs="+",
        default=[
            "strong_direct",
            "multitask_strong",
            "gated_residual",
            "gated_residual_cf",
        ],
        choices=[
            "strong_direct",
            "multitask_strong",
            "gated_residual",
            "gated_residual_cf",
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

    print("=" * 90)
    print(
        "GATED RESIDUAL ACTION-TO-INTENTION REASONING"
    )
    print("=" * 90)

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
        print("#" * 90)
        print(
            "CONDITION:",
            condition,
        )
        print("#" * 90)

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
                    "num_valid_cf_queries": (
                        result[
                            "num_valid_cf_queries"
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
                "num_valid_cf_queries": (
                    result[
                        "num_valid_cf_queries"
                    ]
                ),
            }

            all_results.append(
                compact
            )

            gate_value = (
                result[
                    "best_val"
                ][
                    "gate_mean"
                ]
            )

            gate_str = (
                f"{gate_value:.4f}"
                if gate_value is not None
                else "n/a"
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
                f"{result['best_val']['why']['same_task_wrong_fraction']:.4f} | "
                f"gate={gate_str}"
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
    print("=" * 90)
    print(
        "FINAL COMPARISON"
    )
    print("=" * 90)

    for condition in args.conditions:
        row = summary[
            condition
        ]

        gate_text = ""

        if "gate_mean_mean" in row:
            gate_text = (
                f" | gate "
                f"{row['gate_mean_mean']:.4f} ± "
                f"{row['gate_mean_sample_std']:.4f}"
            )

        print(
            f"{condition:20s} | "
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
            f"{gate_text}"
        )

    print()
    print(
        "Saved:",
        args.output_dir
        / "summary.json",
    )


if __name__ == "__main__":
    main()
