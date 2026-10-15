"""
Temporal-Guided Goal Disambiguation (v2)
=========================================

Key architectural ideas:
1. Frame-level temporal attention for WHAT prediction (not mean-pool).
2. Guided WHAT → WHY path with gated residual (same as hardneg architecture).
3. Same-task hard-negative contrastive objective on WHAT.
4. Multi-positive InfoNCE + cosine alignment on WHY.

The critical insight is that the current model uses mean-pooled visual features
(960-D) as input, which is the SAME representation the static baseline uses.
By instead using frame-level features (8×960) with learned temporal weighting,
we can extract better WHAT predictions — and the Oracle WHAT diagnostic
(MRR=0.107) proves that better WHAT directly translates to better WHY.

Conditions:
  mean_pool_direct   : mean(frames) → direct WHY  (current static baseline)
  temporal_direct     : temporal_attn(frames) → direct WHY
  temporal_guided     : temporal_attn(frames) → WHAT; [state, WHAT] → guided WHY
  temporal_hardneg    : temporal_guided + same-task WHAT hard negatives
  oracle_temporal     : temporal_attn(frames) for state; GT WHAT → guided WHY
"""

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
    obj = torch.load(path, map_location="cpu", weights_only=False)

    mean = None
    comp = None

    if torch.is_tensor(obj):
        comp = obj.float()
    elif isinstance(obj, dict):
        mean_keys = ["mean", "pca_mean", "mu", "center"]
        comp_keys = [
            "components", "components_", "pca_components",
            "projection", "proj", "V", "basis",
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
        raise RuntimeError(f"Could not infer PCA components from {path}")

    comp = comp.squeeze()
    if comp.shape == (960, 256):
        basis = comp
    elif comp.shape == (256, 960):
        basis = comp.T
    else:
        raise RuntimeError(f"Unexpected PCA shape: {tuple(comp.shape)}")

    if mean is None:
        mean = torch.zeros(960, dtype=torch.float32)
    mean = mean.squeeze().float()
    if mean.numel() != 960:
        raise RuntimeError(f"Unexpected PCA mean size: {mean.numel()}")

    mean = mean.reshape(1, 960)
    basis = basis.reshape(960, 256).float()

    def transform(x: torch.Tensor) -> torch.Tensor:
        return (x.float().cpu() - mean) @ basis

    return transform


# ============================================================
# Data loading — now loads frame_features (8×960)
# ============================================================

def load_temporal_split(
    temporal_root: Path,
    split: str,
    what_transform,
    why_transform,
):
    """Load samples with per-frame features and structured labels."""
    root = temporal_root / split
    if not root.exists():
        raise FileNotFoundError(f"Temporal split not found: {root}")

    rows = []
    for p in sorted(root.rglob("*.pt")):
        x = torch.load(p, map_location="cpu", weights_only=False)
        sample_id = str(x.get("sample_id", p.stem))

        frame_features = x["frame_features"].detach().float().cpu()  # (T, 960)
        if frame_features.ndim != 2 or frame_features.shape[1] != 960:
            raise ValueError(
                f"Bad frame_features shape for {sample_id}: "
                f"{tuple(frame_features.shape)}"
            )

        what_960 = x["what_feature"].detach().float().cpu().reshape(1, -1)
        why_960 = x["why_feature"].detach().float().cpu().reshape(1, -1)

        task = str(x.get("task", x.get("event", x.get("activity", ""))))
        if not task:
            raise RuntimeError(f"Missing task label for {sample_id}")

        what_target = F.normalize(
            what_transform(what_960).squeeze(0), dim=-1
        )
        why_target = F.normalize(
            why_transform(why_960).squeeze(0), dim=-1
        )

        rows.append({
            "sample_id": sample_id,
            "task": task,
            "what": str(x["what"]),
            "why": str(x["why"]),
            "frame_features": frame_features,        # (T, 960)
            "mean_visual": frame_features.mean(0),    # (960,) for backward compat
            "what_target": what_target,               # (256,)
            "why_target": why_target,                 # (256,)
        })

    if not rows:
        raise RuntimeError(f"No samples loaded from {root}")
    return rows


def stack_temporal_rows(rows, device):
    frames = torch.stack([x["frame_features"] for x in rows]).to(device)   # (N, T, 960)
    mean_vis = torch.stack([x["mean_visual"] for x in rows]).to(device)     # (N, 960)
    what_target = torch.stack([x["what_target"] for x in rows]).to(device)  # (N, 256)
    why_target = torch.stack([x["why_target"] for x in rows]).to(device)    # (N, 256)
    return frames, mean_vis, what_target, why_target


# ============================================================
# Masks / losses (same as before)
# ============================================================

def build_label_mask(labels, device):
    n = len(labels)
    mask = torch.zeros((n, n), dtype=torch.bool, device=device)
    groups = defaultdict(list)
    for i, label in enumerate(labels):
        groups[label].append(i)
    for idxs in groups.values():
        idx = torch.tensor(idxs, dtype=torch.long, device=device)
        mask[idx[:, None], idx[None, :]] = True
    return mask


def multi_positive_infonce(pred, target, positive_mask, temperature):
    logits = (pred @ target.T) / temperature
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    pos_count = positive_mask.sum(dim=1).clamp_min(1)
    return -(log_prob * positive_mask.float()).sum(dim=1).div(pos_count).mean()


def cosine_alignment_loss(pred, target):
    return (1.0 - (pred * target).sum(dim=-1)).mean()


def same_task_hard_negative_loss(
    pred, target, positive_mask, task_mask, margin=0.10,
):
    sim = pred @ target.T
    neg_mask = task_mask & (~positive_mask)
    pos_sim = sim.masked_fill(~positive_mask, float("-inf")).max(dim=1).values
    neg_sim = sim.masked_fill(~neg_mask, float("-inf")).max(dim=1).values
    valid = torch.isfinite(pos_sim) & torch.isfinite(neg_sim)
    if not valid.any():
        return torch.zeros((), dtype=pred.dtype, device=pred.device)
    return F.relu(margin + neg_sim[valid] - pos_sim[valid]).mean()


# ============================================================
# Model: Temporal-Guided Goal Disambiguator
# ============================================================

class TemporalAttentionPool(nn.Module):
    """
    Lightweight learned temporal attention pooling.
    Input: (B, T, D) → Output: (B, D)

    Instead of mean-pooling all frames equally, learns to weight frames
    based on their content. Early frames showing the grasp approach
    should get different weights than later frames showing the
    manipulation outcome.
    """
    def __init__(self, d_model=960, n_heads=4, d_ff=512):
        super().__init__()
        # Learnable query for temporal pooling
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Single-layer cross-attention: query attends to frame sequence
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,
            dropout=0.1,
        )
        self.norm1 = nn.LayerNorm(d_model)

        # Positional encoding for 8 frames
        self.pos_embed = nn.Parameter(torch.randn(1, 8, d_model) * 0.02)

        # FFN after attention
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, frames):
        """
        Args:
            frames: (B, T, 960) frame-level features

        Returns:
            pooled: (B, 960) temporally-attended feature
            attn_weights: (B, 1, T) attention weights over frames
        """
        B, T, D = frames.shape

        # Add positional encoding
        kv = frames + self.pos_embed[:, :T, :]

        # Cross-attention: learnable query attends to frame sequence
        q = self.query.expand(B, -1, -1)  # (B, 1, D)
        attn_out, attn_weights = self.attn(q, kv, kv)  # (B, 1, D)

        # Residual connection with query
        x = self.norm1(q + self.dropout(attn_out))

        # FFN
        x = self.norm2(x + self.dropout(self.ffn(x)))

        pooled = x.squeeze(1)  # (B, D)
        return pooled, attn_weights


class ProjectionHead(nn.Module):
    def __init__(self, input_dim=960, hidden_dim=512, output_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class TemporalGuidedModel(nn.Module):
    """
    Temporal-Guided Goal Disambiguation Model.

    Key differences from EgoIntentModel:
    1. Uses frame-level features (8×960) instead of mean-pooled (960).
    2. Learned temporal attention pooling instead of naive mean.
    3. Temporal context feeds WHAT prediction, which guides WHY.

    Conditions:
      mean_pool_direct   : mean(frames) → ProjectionHead → WHY (static baseline)
      temporal_direct     : TemporalAttn(frames) → ProjectionHead → WHY
      temporal_guided     : TemporalAttn(frames) → WHAT; [state, WHAT] → WHY
      temporal_hardneg    : temporal_guided + same-task WHAT hard negatives
      oracle_temporal     : TemporalAttn(frames) for state; GT WHAT → WHY
    """

    def __init__(self, hidden_dim=512, visual_state_dim=256):
        super().__init__()

        # Temporal attention pooling (frames → single vector)
        self.temporal_pool = TemporalAttentionPool(
            d_model=960,
            n_heads=4,
            d_ff=hidden_dim,
        )

        # Direct WHY from mean-pooled features (static baseline)
        self.direct_why_head = ProjectionHead(
            input_dim=960,
            hidden_dim=hidden_dim,
            output_dim=256,
        )

        # Direct WHY from temporal-attended features
        self.temporal_why_head = ProjectionHead(
            input_dim=960,
            hidden_dim=hidden_dim,
            output_dim=256,
        )

        # WHAT predictor from temporal-attended features
        self.what_head = ProjectionHead(
            input_dim=960,
            hidden_dim=hidden_dim,
            output_dim=256,
        )

        # Visual state encoder
        self.visual_state = nn.Sequential(
            nn.Linear(960, visual_state_dim),
            nn.GELU(),
            nn.LayerNorm(visual_state_dim),
        )

        # Guided WHY: fuse visual state + predicted WHAT
        fusion_dim = visual_state_dim + 256
        self.reasoner = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
        )

        self.gate = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.Sigmoid(),
        )

    def _guided_path(self, temporal_feat, what_vec):
        """Shared guided WHY computation."""
        direct = self.temporal_why_head(temporal_feat)
        visual_state = self.visual_state(temporal_feat)
        fused = torch.cat([visual_state, what_vec], dim=-1)
        residual = F.normalize(self.reasoner(fused), dim=-1)
        gate = self.gate(fused)
        why = F.normalize(direct + gate * residual, dim=-1)
        return {
            "why": why,
            "direct": direct,
            "residual": residual,
            "gate": gate,
        }

    def forward_mean_pool_direct(self, mean_visual):
        """Static baseline: mean-pool → direct WHY."""
        why = self.direct_why_head(mean_visual)
        return {"why": why, "what": None, "gate": None}

    def forward_temporal_direct(self, frames):
        """Temporal attention → direct WHY (no WHAT guidance)."""
        temporal_feat, attn_w = self.temporal_pool(frames)
        why = self.temporal_why_head(temporal_feat)
        what = self.what_head(temporal_feat)
        return {"why": why, "what": what, "gate": None, "attn": attn_w}

    def forward_temporal_guided(self, frames):
        """Temporal attention → predicted WHAT → guided WHY."""
        temporal_feat, attn_w = self.temporal_pool(frames)
        predicted_what = self.what_head(temporal_feat)
        out = self._guided_path(temporal_feat, predicted_what)
        out["what"] = predicted_what
        out["attn"] = attn_w
        return out

    def forward_oracle_temporal(self, frames, oracle_what):
        """Temporal attention for state; GT WHAT → guided WHY."""
        temporal_feat, attn_w = self.temporal_pool(frames)
        out = self._guided_path(temporal_feat, oracle_what)
        out["what"] = None
        out["attn"] = attn_w
        return out


# ============================================================
# Metrics
# ============================================================

def retrieval_metrics(pred, target, labels):
    sim = pred @ target.T
    groups = defaultdict(list)
    for i, label in enumerate(labels):
        groups[label].append(i)

    ranks = []
    for i, label in enumerate(labels):
        correct_indices = set(groups[label])
        sorted_j = torch.argsort(sim[i], descending=True).tolist()
        for rank, j in enumerate(sorted_j, start=1):
            if j in correct_indices:
                ranks.append(rank)
                break

    ranks_t = torch.tensor(ranks, dtype=torch.float32, device=pred.device)
    return {
        "Top1": (ranks_t <= 1).float().mean().item(),
        "Top3": (ranks_t <= 3).float().mean().item(),
        "Top5": (ranks_t <= 5).float().mean().item(),
        "MRR": (1.0 / ranks_t).mean().item(),
        "median_rank": ranks_t.median().item(),
        "mean_rank": ranks_t.mean().item(),
    }


@torch.no_grad()
def error_breakdown(why_pred, why_target, rows):
    sim = why_pred @ why_target.T
    ranking = torch.argsort(sim, dim=1, descending=True)

    correct = 0
    same_task_wrong = 0
    cross_task_wrong = 0
    same_task_ranks = []

    for i, row in enumerate(rows):
        j = int(ranking[i, 0].item())
        pred_row = rows[j]

        if pred_row["why"] == row["why"]:
            correct += 1
        elif pred_row["task"] == row["task"]:
            same_task_wrong += 1
        else:
            cross_task_wrong += 1

        # Within-task disambiguation ranking
        same_task_indices = [
            k for k, r in enumerate(rows) if r["task"] == row["task"]
        ]
        if same_task_indices:
            same_task_sims = sim[i, same_task_indices]
            sub_order = torch.argsort(same_task_sims, descending=True)
            for sub_rank, local_k in enumerate(sub_order.tolist(), start=1):
                global_k = same_task_indices[local_k]
                if rows[global_k]["why"] == row["why"]:
                    same_task_ranks.append(sub_rank)
                    break

    wrong = same_task_wrong + cross_task_wrong
    total = len(rows)
    st_t = (
        torch.tensor(same_task_ranks, dtype=torch.float32, device=why_pred.device)
        if same_task_ranks
        else torch.tensor([1.0], device=why_pred.device)
    )

    return {
        "top1_correct_count": correct,
        "same_task_wrong_count": same_task_wrong,
        "cross_task_wrong_count": cross_task_wrong,
        "same_task_wrong_fraction": (
            same_task_wrong / wrong if wrong > 0 else 0.0
        ),
        "cross_task_accuracy": (
            (correct + same_task_wrong) / total if total > 0 else 0.0
        ),
        "same_task_disambig_mrr": (1.0 / st_t).mean().item(),
        "same_task_disambig_top1": (st_t <= 1).float().mean().item(),
    }


# ============================================================
# Evaluate
# ============================================================

@torch.no_grad()
def evaluate(model, frames, mean_vis, what_target, why_target, rows, condition):
    model.eval()

    if condition == "mean_pool_direct":
        out = model.forward_mean_pool_direct(mean_vis)
        what_pred = None
        why_pred = out["why"]
        gate_mean = None

    elif condition == "temporal_direct":
        out = model.forward_temporal_direct(frames)
        what_pred = out["what"]
        why_pred = out["why"]
        gate_mean = None

    elif condition in {"temporal_guided", "temporal_hardneg"}:
        out = model.forward_temporal_guided(frames)
        what_pred = out["what"]
        why_pred = out["why"]
        gate_mean = float(out["gate"].mean().item())

    elif condition == "oracle_temporal":
        out = model.forward_oracle_temporal(frames, what_target)
        what_pred = None
        why_pred = out["why"]
        gate_mean = float(out["gate"].mean().item())

    else:
        raise ValueError(f"Unknown condition: {condition}")

    what_metrics = (
        None if what_pred is None
        else retrieval_metrics(
            what_pred, what_target, [x["what"] for x in rows]
        )
    )

    why_metrics = retrieval_metrics(
        why_pred, why_target, [x["why"] for x in rows]
    )
    why_metrics.update(error_breakdown(why_pred, why_target, rows))

    return {
        "what": what_metrics,
        "why": why_metrics,
        "gate_mean": gate_mean,
    }


# ============================================================
# Train one condition × seed
# ============================================================

def train_one(
    condition,
    train_rows,
    val_rows,
    train_frames,
    train_mean_vis,
    train_what_target,
    train_why_target,
    val_frames,
    val_mean_vis,
    val_what_target,
    val_why_target,
    seed,
    args,
    device,
):
    set_seed(seed)

    model = TemporalGuidedModel(
        hidden_dim=args.hidden_dim,
        visual_state_dim=args.visual_state_dim,
    ).to(device)

    # Determine trainable parameters based on condition
    if condition == "mean_pool_direct":
        trainable_params = list(model.direct_why_head.parameters())

    elif condition == "temporal_direct":
        trainable_params = (
            list(model.temporal_pool.parameters())
            + list(model.temporal_why_head.parameters())
            + list(model.what_head.parameters())
        )

    elif condition in {"temporal_guided", "temporal_hardneg"}:
        trainable_params = list(model.parameters())

    elif condition == "oracle_temporal":
        trainable_params = (
            list(model.temporal_pool.parameters())
            + list(model.temporal_why_head.parameters())
            + list(model.visual_state.parameters())
            + list(model.reasoner.parameters())
            + list(model.gate.parameters())
        )

    else:
        raise ValueError(f"Unknown condition: {condition}")

    n_params = sum(p.numel() for p in trainable_params)
    print(f"  Trainable parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Warm-up + cosine decay scheduler
    warmup_epochs = min(10, args.epochs // 10)
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, args.epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    why_mask = build_label_mask([x["why"] for x in train_rows], device)
    what_mask = build_label_mask([x["what"] for x in train_rows], device)
    task_mask = build_label_mask([x["task"] for x in train_rows], device)

    best_mrr = -1.0
    best_epoch = -1
    best_eval = None
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        what_hard = torch.zeros((), device=device)

        if condition == "mean_pool_direct":
            out = model.forward_mean_pool_direct(train_mean_vis)
            why_pred = out["why"]
            what_loss = torch.zeros((), device=device)

        elif condition == "temporal_direct":
            out = model.forward_temporal_direct(train_frames)
            why_pred = out["why"]
            what_pred = out["what"]

            what_nce = multi_positive_infonce(
                what_pred, train_what_target, what_mask, args.temperature,
            )
            what_cos = cosine_alignment_loss(what_pred, train_what_target)
            what_loss = what_nce + args.cosine_weight * what_cos

        elif condition in {"temporal_guided", "temporal_hardneg"}:
            out = model.forward_temporal_guided(train_frames)
            why_pred = out["why"]
            what_pred = out["what"]

            what_nce = multi_positive_infonce(
                what_pred, train_what_target, what_mask, args.temperature,
            )
            what_cos = cosine_alignment_loss(what_pred, train_what_target)

            if condition == "temporal_hardneg":
                what_hard = same_task_hard_negative_loss(
                    what_pred, train_what_target,
                    what_mask, task_mask,
                    margin=args.what_hard_margin,
                )

            what_loss = (
                what_nce
                + args.cosine_weight * what_cos
                + args.what_hard_weight * what_hard
            )

        elif condition == "oracle_temporal":
            out = model.forward_oracle_temporal(
                train_frames, train_what_target,
            )
            why_pred = out["why"]
            what_loss = torch.zeros((), device=device)

        else:
            raise ValueError(f"Unknown condition: {condition}")

        why_nce = multi_positive_infonce(
            why_pred, train_why_target, why_mask, args.temperature,
        )
        why_cos = cosine_alignment_loss(why_pred, train_why_target)
        why_loss = why_nce + args.cosine_weight * why_cos

        total_loss = why_loss + args.what_weight * what_loss

        total_loss.backward()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)

        optimizer.step()
        scheduler.step()

        if (
            epoch == 1
            or epoch % args.eval_every == 0
            or epoch == args.epochs
        ):
            eval_result = evaluate(
                model,
                val_frames,
                val_mean_vis,
                val_what_target,
                val_why_target,
                val_rows,
                condition,
            )

            why_mrr = eval_result["why"]["MRR"]
            why_top1 = eval_result["why"]["Top1"]
            what_mrr = (
                eval_result["what"]["MRR"]
                if eval_result["what"] is not None
                else 0.0
            )
            st_mrr = eval_result["why"].get("same_task_disambig_mrr", 0.0)

            history.append({
                "epoch": epoch,
                "loss": float(total_loss.item()),
                "val_why_mrr": why_mrr,
                "val_why_top1": why_top1,
                "val_what_mrr": what_mrr,
                "val_st_mrr": st_mrr,
            })

            print(
                f"[{condition}] seed={seed} epoch={epoch:03d}/{args.epochs}"
                f" | loss={total_loss.item():.6f}"
                f" | WHY MRR={why_mrr:.6f}"
                f" | WHY Top1={why_top1:.6f}"
                f" | WHAT MRR={what_mrr:.6f}"
                f" | ST-MRR={st_mrr:.6f}"
            )

            if why_mrr > best_mrr:
                best_mrr = why_mrr
                best_epoch = epoch
                best_eval = eval_result
                best_state = {
                    k: v.cpu().clone()
                    for k, v in model.state_dict().items()
                }

    st_err = best_eval["why"].get("same_task_wrong_fraction", 0.0)
    st_disambig_mrr = best_eval["why"].get("same_task_disambig_mrr", 0.0)
    st_disambig_top1 = best_eval["why"].get("same_task_disambig_top1", 0.0)

    print(
        f"BEST | condition={condition} | seed={seed}"
        f" | epoch={best_epoch}"
        f" | WHY MRR={best_eval['why']['MRR']:.6f}"
        f" | WHY Top1={best_eval['why']['Top1']:.6f}"
        f" | WHY Top3={best_eval['why']['Top3']:.6f}"
        f" | WHY Top5={best_eval['why']['Top5']:.6f}"
        f" | WHAT MRR={best_eval['what']['MRR'] if best_eval['what'] else 0.0:.6f}"
        f" | same-task-err={st_err:.4f}"
        f" | ST-disambig-MRR={st_disambig_mrr:.6f}"
        f" | ST-disambig-Top1={st_disambig_top1:.6f}"
    )

    return {
        "condition": condition,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val": best_eval,
        "state_dict": best_state,
        "history": history,
    }


# ============================================================
# Summary helpers
# ============================================================

def mean_std(values):
    arr = np.array(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def summarize(rows):
    fields = {
        "why_mrr": [x["best_val"]["why"]["MRR"] for x in rows],
        "why_top1": [x["best_val"]["why"]["Top1"] for x in rows],
        "why_top3": [x["best_val"]["why"]["Top3"] for x in rows],
        "why_top5": [x["best_val"]["why"]["Top5"] for x in rows],
        "what_mrr": [
            x["best_val"]["what"]["MRR"]
            for x in rows
            if x["best_val"]["what"] is not None
        ],
        "same_task_wrong_fraction": [
            x["best_val"]["why"]["same_task_wrong_fraction"]
            for x in rows
        ],
        "cross_task_accuracy": [
            x["best_val"]["why"]["cross_task_accuracy"]
            for x in rows
        ],
        "same_task_disambig_mrr": [
            x["best_val"]["why"]["same_task_disambig_mrr"]
            for x in rows
        ],
        "same_task_disambig_top1": [
            x["best_val"]["why"]["same_task_disambig_top1"]
            for x in rows
        ],
        "gate_mean": [
            x["best_val"]["gate_mean"]
            for x in rows
            if x["best_val"]["gate_mean"] is not None
        ],
    }

    out = {}
    for name, values in fields.items():
        if not values:
            continue
        mean, std = mean_std(values)
        out[f"{name}_values"] = values
        out[f"{name}_mean"] = mean
        out[f"{name}_sample_std"] = std

    return out


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Temporal-Guided Goal Disambiguation: "
            "frame-level temporal attention + WHAT→WHY guided architecture."
        )
    )

    parser.add_argument(
        "--temporal-cache",
        type=Path,
        default=Path(
            "/media/dhqg/d/datasets/"
            "egointent/cache/temporal_intention_v1"
        ),
    )

    parser.add_argument(
        "--what-pca",
        type=Path,
        default=Path(
            "/media/dhqg/d/vlia_outputs/"
            "structured_intention_v1/"
            "what_pca_960_to_256.pt"
        ),
    )

    parser.add_argument(
        "--why-pca",
        type=Path,
        default=Path(
            "/media/dhqg/d/vlia_outputs/"
            "stage_a_clean_v1/"
            "why_pca_960_to_256.pt"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/media/dhqg/d/vlia_outputs/"
            "temporal_guided_v2"
        ),
    )

    parser.add_argument(
        "--conditions",
        nargs="+",
        default=[
            "mean_pool_direct",
            "temporal_direct",
            "temporal_guided",
            "temporal_hardneg",
            "oracle_temporal",
        ],
        choices=[
            "mean_pool_direct",
            "temporal_direct",
            "temporal_guided",
            "temporal_hardneg",
            "oracle_temporal",
        ],
    )

    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--cosine-weight", type=float, default=0.1)
    parser.add_argument("--what-weight", type=float, default=0.3)
    parser.add_argument("--what-hard-weight", type=float, default=0.3)
    parser.add_argument("--what-hard-margin", type=float, default=0.10)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--visual-state-dim", type=int, default=256)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    device = torch.device(args.device)

    what_transform = load_pca_transform(args.what_pca)
    why_transform = load_pca_transform(args.why_pca)

    train_rows = load_temporal_split(
        args.temporal_cache, "train", what_transform, why_transform,
    )
    val_rows = load_temporal_split(
        args.temporal_cache, "val", what_transform, why_transform,
    )

    train_frames, train_mean_vis, train_what_target, train_why_target = (
        stack_temporal_rows(train_rows, device)
    )
    val_frames, val_mean_vis, val_what_target, val_why_target = (
        stack_temporal_rows(val_rows, device)
    )

    train_tasks = sorted(set(x["task"] for x in train_rows))
    val_tasks = sorted(set(x["task"] for x in val_rows))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 96)
    print("TEMPORAL-GUIDED GOAL DISAMBIGUATION v2")
    print("=" * 96)
    print(f"device: {device}")
    print(f"train samples: {len(train_rows)}")
    print(f"val samples: {len(val_rows)}")
    print(f"train frames shape: {tuple(train_frames.shape)}")
    print(f"val frames shape: {tuple(val_frames.shape)}")
    print(f"train tasks: {train_tasks}")
    print(f"val tasks: {val_tasks}")
    print(f"conditions: {args.conditions}")
    print(f"seeds: {args.seeds}")
    print(f"epochs: {args.epochs}")
    print(f"lr: {args.lr}")
    print(f"what_weight: {args.what_weight}")
    print(f"what_hard_weight: {args.what_hard_weight}")
    print(f"what_hard_margin: {args.what_hard_margin}")
    print("=" * 96)

    all_results = []

    for condition in args.conditions:
        print()
        print("#" * 96)
        print(f"CONDITION: {condition}")
        print("#" * 96)

        condition_results = []

        for seed in args.seeds:
            result = train_one(
                condition=condition,
                train_rows=train_rows,
                val_rows=val_rows,
                train_frames=train_frames,
                train_mean_vis=train_mean_vis,
                train_what_target=train_what_target,
                train_why_target=train_why_target,
                val_frames=val_frames,
                val_mean_vis=val_mean_vis,
                val_what_target=val_what_target,
                val_why_target=val_why_target,
                seed=seed,
                args=args,
                device=device,
            )

            # Save checkpoint
            seed_dir = args.output_dir / condition / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)

            torch.save(
                {
                    "condition": condition,
                    "seed": seed,
                    "best_epoch": result["best_epoch"],
                    "best_val": result["best_val"],
                    "model_state_dict": result["state_dict"],
                    "config": {
                        k: str(v) if isinstance(v, Path) else v
                        for k, v in vars(args).items()
                    },
                },
                seed_dir / "best.pt",
            )

            with open(seed_dir / "history.json", "w", encoding="utf-8") as f:
                json.dump(result["history"], f, indent=2)

            condition_results.append({
                "condition": condition,
                "seed": seed,
                "best_epoch": result["best_epoch"],
                "best_val": result["best_val"],
            })

        all_results.extend(condition_results)

    # Build summary
    summary = {}
    for condition in args.conditions:
        rows = [x for x in all_results if x["condition"] == condition]
        summary[condition] = summarize(rows)

    # Save summary
    with open(args.output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment": "temporal_guided_v2",
                "config": {
                    k: str(v) if isinstance(v, Path) else v
                    for k, v in vars(args).items()
                },
                "summary": summary,
                "results": all_results,
            },
            f,
            indent=2,
            default=str,
        )

    # Print final comparison
    print()
    print("=" * 96)
    print("FINAL COMPARISON")
    print("=" * 96)

    for condition in args.conditions:
        s = summary[condition]
        why_mrr = s.get("why_mrr_mean", 0.0)
        why_std = s.get("why_mrr_sample_std", 0.0)
        top1 = s.get("why_top1_mean", 0.0)
        top1_s = s.get("why_top1_sample_std", 0.0)
        what_mrr = s.get("what_mrr_mean", 0.0)
        what_s = s.get("what_mrr_sample_std", 0.0)
        st_err = s.get("same_task_wrong_fraction_mean", 0.0)
        st_s = s.get("same_task_wrong_fraction_sample_std", 0.0)
        st_mrr = s.get("same_task_disambig_mrr_mean", 0.0)
        st_m_s = s.get("same_task_disambig_mrr_sample_std", 0.0)

        print(
            f"{condition:22s}"
            f" | WHY MRR {why_mrr:.6f} ± {why_std:.6f}"
            f" | WHY Top1 {top1:.6f} ± {top1_s:.6f}"
            f" | WHAT MRR {what_mrr:.6f} ± {what_s:.6f}"
            f" | ST-err {st_err:.4f} ± {st_s:.4f}"
            f" | ST-disambig-MRR {st_mrr:.6f} ± {st_m_s:.6f}"
        )

    print()
    print(f"Saved: {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
