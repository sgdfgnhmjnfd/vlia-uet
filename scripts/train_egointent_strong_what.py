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
        raise RuntimeError(
            f"Unexpected PCA component shape: {tuple(comp.shape)}"
        )

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
# Data
# ============================================================

def load_structured_feature_map(temporal_root: Path, split: str):
    root = temporal_root / split

    if not root.exists():
        raise FileNotFoundError(f"Temporal split not found: {root}")

    out = {}

    for p in sorted(root.rglob("*.pt")):
        x = torch.load(p, map_location="cpu", weights_only=False)
        sample_id = str(x.get("sample_id", p.stem))

        if "what_feature" not in x or "what" not in x:
            raise KeyError(f"Missing WHAT data in {p}")

        out[sample_id] = {
            "what": str(x["what"]),
            "what_feature": x["what_feature"].detach().float().cpu(),
        }

    if not out:
        raise RuntimeError(f"No structured samples found in {root}")

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
        raise FileNotFoundError(f"Stage-A split not found: {root}")

    structured = load_structured_feature_map(temporal_root, split)
    rows = []

    for p in sorted(root.rglob("*.pt")):
        x = torch.load(p, map_location="cpu", weights_only=False)
        sample_id = str(x["sample_id"])

        if sample_id not in structured:
            raise KeyError(f"Missing WHAT data for sample {sample_id}")

        visual = x["visual_feature"].detach().float().cpu()
        why_960 = x["why_feature"].detach().float().cpu().reshape(1, -1)
        what_960 = structured[sample_id]["what_feature"].reshape(1, -1)

        if visual.shape != (960,):
            raise ValueError(
                f"Bad visual shape for {sample_id}: {tuple(visual.shape)}"
            )

        if why_960.shape != (1, 960):
            raise ValueError(
                f"Bad WHY feature shape for {sample_id}: {tuple(why_960.shape)}"
            )

        if what_960.shape != (1, 960):
            raise ValueError(
                f"Bad WHAT feature shape for {sample_id}: {tuple(what_960.shape)}"
            )

        task = str(
            x.get(
                "task",
                x.get("event", x.get("activity", "")),
            )
        )

        if not task:
            raise RuntimeError(f"Missing task label for {sample_id}")

        what_target = F.normalize(
            what_transform(what_960).squeeze(0), dim=-1
        )
        why_target = F.normalize(
            why_transform(why_960).squeeze(0), dim=-1
        )

        rows.append(
            {
                "sample_id": sample_id,
                "video_uid": str(x.get("video_uid", "")),
                "task": task,
                "what": structured[sample_id]["what"],
                "why": str(x["why"]),
                "visual": visual,
                "what_target": what_target,
                "why_target": why_target,
            }
        )

    if not rows:
        raise RuntimeError(f"No samples loaded from {root}")

    return rows


def stack_rows(rows, device):
    visual = torch.stack([x["visual"] for x in rows]).to(device)
    what_target = torch.stack([x["what_target"] for x in rows]).to(device)
    why_target = torch.stack([x["why_target"] for x in rows]).to(device)
    return visual, what_target, why_target


# ============================================================
# Masks / losses
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

    log_prob = logits - torch.logsumexp(
        logits, dim=1, keepdim=True
    )

    pos_count = positive_mask.sum(dim=1).clamp_min(1)

    return -(
        log_prob * positive_mask.float()
    ).sum(dim=1).div(pos_count).mean()


def cosine_alignment_loss(pred, target):
    return (1.0 - (pred * target).sum(dim=-1)).mean()


# ============================================================
# EgoIntent model
# ============================================================

class ProjectionHead(nn.Module):
    def __init__(self, hidden_dim=512, output_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(960, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class StrongWhatHead(nn.Module):
    """
    Higher-capacity WHAT predictor used only by strong_what_guided.

    This changes only the WHAT prediction branch. The direct WHY head,
    visual-state branch, WHY reasoner and gate are kept identical to the
    previous intention_guided experiment.
    """

    def __init__(
        self,
        hidden_dim=512,
        output_dim=256,
        expansion_dim=768,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(960, expansion_dim),
            nn.LayerNorm(expansion_dim),
            nn.GELU(),
            nn.Linear(expansion_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class EgoIntentModel(nn.Module):
    """
    Controlled comparison:

    direct_why:
        visual -> direct WHY

    intention_guided:
        visual -> predicted WHAT
        visual -> visual state
        [visual state, predicted WHAT] -> residual WHY
        final WHY = normalize(direct WHY + gate * residual WHY)

    strong_what_guided:
        visual -> STRONG predicted WHAT
        visual -> visual state
        [visual state, strong predicted WHAT] -> residual WHY
        final WHY = normalize(direct WHY + gate * residual WHY)

    oracle_intention:
        visual -> visual state
        [visual state, GT WHAT] -> residual WHY
        final WHY = normalize(direct WHY + gate * residual WHY)

    IMPORTANT:
        intention_guided uses only PREDICTED WHAT at validation.
        oracle_intention uses GT WHAT only as a privileged diagnostic input.
        It is NOT a deployable condition and must not be interpreted as such.
    """

    def __init__(
        self,
        hidden_dim=512,
        visual_state_dim=256,
    ):
        super().__init__()

        # Shared direct WHY path. This is the complete direct_why baseline.
        self.why_head = ProjectionHead(
            hidden_dim=hidden_dim,
            output_dim=256,
        )

        # Explicit intention predictor.
        self.what_head = ProjectionHead(
            hidden_dim=hidden_dim,
            output_dim=256,
        )

        self.visual_state = nn.Sequential(
            nn.Linear(960, visual_state_dim),
            nn.GELU(),
            nn.LayerNorm(visual_state_dim),
        )

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

        # Added last on purpose: this preserves RNG initialization of every
        # pre-existing module in the controlled intention_guided condition.
        self.strong_what_head = StrongWhatHead(
            hidden_dim=hidden_dim,
            output_dim=256,
            expansion_dim=768,
        )

    def predict_direct_why(self, visual):
        return self.why_head(visual)

    def predict_what(self, visual):
        return self.what_head(visual)

    def predict_intention_guided(self, visual):
        direct = self.predict_direct_why(visual)
        predicted_what = self.predict_what(visual)

        visual_state = self.visual_state(visual)
        fused = torch.cat(
            [visual_state, predicted_what],
            dim=-1,
        )

        residual = F.normalize(
            self.reasoner(fused),
            dim=-1,
        )
        gate = self.gate(fused)

        why = F.normalize(
            direct + gate * residual,
            dim=-1,
        )

        return {
            "why": why,
            "direct": direct,
            "what": predicted_what,
            "residual": residual,
            "gate": gate,
        }

    def predict_strong_what(self, visual):
        return self.strong_what_head(visual)

    def predict_strong_what_guided(self, visual):
        direct = self.predict_direct_why(visual)
        predicted_what = self.predict_strong_what(visual)

        visual_state = self.visual_state(visual)
        fused = torch.cat(
            [visual_state, predicted_what],
            dim=-1,
        )

        residual = F.normalize(
            self.reasoner(fused),
            dim=-1,
        )
        gate = self.gate(fused)

        why = F.normalize(
            direct + gate * residual,
            dim=-1,
        )

        return {
            "why": why,
            "direct": direct,
            "what": predicted_what,
            "residual": residual,
            "gate": gate,
        }

    def predict_oracle_intention(self, visual, oracle_what):
        """
        Diagnostic upper-bound path.

        oracle_what is the ground-truth WHAT target in the same normalized
        256-D PCA space used to train/evaluate the predicted WHAT branch.
        No gradient is required through oracle_what itself.
        """
        direct = self.predict_direct_why(visual)

        visual_state = self.visual_state(visual)
        fused = torch.cat(
            [visual_state, oracle_what],
            dim=-1,
        )

        residual = F.normalize(
            self.reasoner(fused),
            dim=-1,
        )
        gate = self.gate(fused)

        why = F.normalize(
            direct + gate * residual,
            dim=-1,
        )

        return {
            "why": why,
            "direct": direct,
            "what": None,
            "residual": residual,
            "gate": gate,
        }


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def retrieval_metrics(pred, target, labels):
    sim = pred @ target.T
    ranking = torch.argsort(sim, dim=1, descending=True)

    ranks = []

    for i, label in enumerate(labels):
        positives = {
            j
            for j, candidate_label in enumerate(labels)
            if candidate_label == label
        }

        rank = None
        for r, j in enumerate(ranking[i].tolist(), start=1):
            if j in positives:
                rank = r
                break

        if rank is None:
            raise RuntimeError(
                f"No positive retrieval target for sample index {i}"
            )

        ranks.append(rank)

    ranks_t = torch.tensor(
        ranks,
        dtype=torch.float32,
        device=pred.device,
    )

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

    for i, row in enumerate(rows):
        j = int(ranking[i, 0].item())
        pred_row = rows[j]

        if pred_row["why"] == row["why"]:
            correct += 1
        elif pred_row["task"] == row["task"]:
            same_task_wrong += 1
        else:
            cross_task_wrong += 1

    wrong = same_task_wrong + cross_task_wrong

    return {
        "top1_correct_count": correct,
        "same_task_wrong_count": same_task_wrong,
        "cross_task_wrong_count": cross_task_wrong,
        "same_task_wrong_fraction": (
            same_task_wrong / wrong if wrong > 0 else 0.0
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

    if condition == "direct_why":
        what_pred = model.predict_what(visual)
        why_pred = model.predict_direct_why(visual)
        gate_mean = None

    elif condition == "intention_guided":
        out = model.predict_intention_guided(visual)
        what_pred = out["what"]
        why_pred = out["why"]
        gate_mean = float(out["gate"].mean().item())

    elif condition == "strong_what_guided":
        out = model.predict_strong_what_guided(visual)
        what_pred = out["what"]
        why_pred = out["why"]
        gate_mean = float(out["gate"].mean().item())

    elif condition == "oracle_intention":
        out = model.predict_oracle_intention(
            visual,
            what_target,
        )
        what_pred = None
        why_pred = out["why"]
        gate_mean = float(out["gate"].mean().item())

    else:
        raise ValueError(f"Unknown condition: {condition}")

    what_metrics = (
        None
        if what_pred is None
        else retrieval_metrics(
            what_pred,
            what_target,
            [x["what"] for x in rows],
        )
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
        "gate_mean": gate_mean,
    }


# ============================================================
# Training
# ============================================================

def clone_state_dict(model):
    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
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
    set_seed(seed)
    device = train_visual.device

    model = EgoIntentModel(
        hidden_dim=args.hidden_dim,
        visual_state_dim=args.visual_state_dim,
    ).to(device)

    if condition == "direct_why":
        # Exact direct baseline: only the direct WHY branch is optimized.
        trainable_params = list(model.why_head.parameters())

    elif condition == "intention_guided":
        # Exact previous guided condition: do NOT optimize strong_what_head.
        trainable_params = (
            list(model.why_head.parameters())
            + list(model.what_head.parameters())
            + list(model.visual_state.parameters())
            + list(model.reasoner.parameters())
            + list(model.gate.parameters())
        )

    elif condition == "strong_what_guided":
        # Only the WHAT predictor capacity changes relative to intention_guided.
        trainable_params = (
            list(model.why_head.parameters())
            + list(model.strong_what_head.parameters())
            + list(model.visual_state.parameters())
            + list(model.reasoner.parameters())
            + list(model.gate.parameters())
        )

    elif condition == "oracle_intention":
        # Oracle WHAT is a fixed privileged input. Do not train what_head.
        trainable_params = (
            list(model.why_head.parameters())
            + list(model.visual_state.parameters())
            + list(model.reasoner.parameters())
            + list(model.gate.parameters())
        )

    else:
        raise ValueError(f"Unknown condition: {condition}")

    optimizer = torch.optim.AdamW(
        trainable_params,
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

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        if condition == "direct_why":
            why_pred = model.predict_direct_why(
                train_visual
            )
            what_loss = torch.zeros(
                (),
                device=device,
            )

        elif condition == "intention_guided":
            out = model.predict_intention_guided(
                train_visual
            )
            why_pred = out["why"]
            what_pred = out["what"]

            what_nce = multi_positive_infonce(
                what_pred,
                train_what_target,
                what_mask,
                args.temperature,
            )
            what_cos = cosine_alignment_loss(
                what_pred,
                train_what_target,
            )
            what_loss = (
                what_nce
                + args.cosine_weight * what_cos
            )

        elif condition == "strong_what_guided":
            out = model.predict_strong_what_guided(
                train_visual
            )
            why_pred = out["why"]
            what_pred = out["what"]

            what_nce = multi_positive_infonce(
                what_pred,
                train_what_target,
                what_mask,
                args.temperature,
            )
            what_cos = cosine_alignment_loss(
                what_pred,
                train_what_target,
            )
            what_loss = (
                what_nce
                + args.cosine_weight * what_cos
            )

        elif condition == "oracle_intention":
            out = model.predict_oracle_intention(
                train_visual,
                train_what_target,
            )
            why_pred = out["why"]
            what_loss = torch.zeros(
                (),
                device=device,
            )

        else:
            raise ValueError(f"Unknown condition: {condition}")

        why_nce = multi_positive_infonce(
            why_pred,
            train_why_target,
            why_mask,
            args.temperature,
        )
        why_cos = cosine_alignment_loss(
            why_pred,
            train_why_target,
        )
        why_loss = (
            why_nce
            + args.cosine_weight * why_cos
        )

        total_loss = (
            why_loss
            + args.what_weight * what_loss
        )

        total_loss.backward()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                trainable_params,
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

            record = {
                "epoch": epoch,
                "loss_total": float(total_loss.item()),
                "loss_why": float(why_loss.item()),
                "loss_what": float(what_loss.item()),
                "val": val_eval,
            }
            history.append(record)

            mrr = val_eval["why"]["MRR"]
            gate_text = (
                ""
                if val_eval["gate_mean"] is None
                else f" | gate={val_eval['gate_mean']:.4f}"
            )

            what_text = (
                "WHAT MRR=N/A"
                if val_eval["what"] is None
                else f"WHAT MRR={val_eval['what']['MRR']:.6f}"
            )

            print(
                f"[{condition}] seed={seed} "
                f"epoch={epoch:03d}/{args.epochs} | "
                f"loss={total_loss.item():.6f} | "
                f"WHY MRR={mrr:.6f} | "
                f"WHY Top1={val_eval['why']['Top1']:.6f} | "
                f"{what_text}"
                f"{gate_text}"
            )

            if mrr > best_mrr:
                best_mrr = mrr
                best_epoch = epoch
                best_eval = val_eval
                best_state = clone_state_dict(model)

    if best_state is None:
        raise RuntimeError("No checkpoint selected.")

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
    x = np.asarray(values, dtype=np.float64)
    return (
        float(x.mean()),
        float(x.std(ddof=1) if len(x) > 1 else 0.0),
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
        "why_top3": [
            x["best_val"]["why"]["Top3"]
            for x in rows
        ],
        "why_top5": [
            x["best_val"]["why"]["Top5"]
            for x in rows
        ],
        "what_mrr": [
            x["best_val"]["what"]["MRR"]
            for x in rows
            if x["best_val"]["what"] is not None
        ],
        "same_task_wrong_fraction": [
            x["best_val"]["why"]["same_task_wrong_fraction"]
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
            "EgoIntent diagnostic: test whether a stronger WHAT predictor closes the gap to oracle-WHAT-guided WHY reasoning."
        )
    )

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
            "egointent/cache/temporal_intention_v1"
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
            "egointent_strong_what_v1"
        ),
    )

    parser.add_argument(
        "--conditions",
        nargs="+",
        default=[
            "direct_why",
            "intention_guided",
            "strong_what_guided",
            "oracle_intention",
        ],
        choices=[
            "direct_why",
            "intention_guided",
            "strong_what_guided",
            "oracle_intention",
        ],
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 456],
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
    device = torch.device(args.device)

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

    what_transform = load_pca_transform(
        args.what_pca
    )
    why_transform = load_pca_transform(
        args.why_pca
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

    print("=" * 96)
    print("EGOINTENT STRONG-WHAT GUIDED DIAGNOSTIC")
    print("=" * 96)
    print("device:", device)
    print("train samples:", len(train_rows))
    print("val samples:", len(val_rows))
    print(
        "train tasks:",
        sorted({x["task"] for x in train_rows}),
    )
    print(
        "val tasks:",
        sorted({x["task"] for x in val_rows}),
    )
    print("conditions:", args.conditions)
    print("seeds:", args.seeds)
    print("epochs:", args.epochs)
    print("lr:", args.lr)
    print("what_weight:", args.what_weight)

    audit_model = EgoIntentModel(
        hidden_dim=args.hidden_dim,
        visual_state_dim=args.visual_state_dim,
    )
    old_what_params = sum(
        p.numel() for p in audit_model.what_head.parameters()
    )
    strong_what_params = sum(
        p.numel() for p in audit_model.strong_what_head.parameters()
    )
    del audit_model

    print("original WHAT-head params:", old_what_params)
    print("strong WHAT-head params:", strong_what_params)
    print("=" * 96)

    all_results = []

    for condition in args.conditions:
        print()
        print("#" * 96)
        print("CONDITION:", condition)
        print("#" * 96)

        condition_dir = (
            args.output_dir / condition
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
                train_what_target=train_what_target,
                train_why_target=train_why_target,
                train_rows=train_rows,
                val_visual=val_visual,
                val_what_target=val_what_target,
                val_why_target=val_why_target,
                val_rows=val_rows,
                args=args,
            )

            seed_dir = (
                condition_dir / f"seed_{seed}"
            )
            seed_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            torch.save(
                {
                    "experiment": "egointent_strong_what",
                    "condition": condition,
                    "seed": seed,
                    "best_epoch": result["best_epoch"],
                    "best_val": result["best_val"],
                    "model_state_dict": result[
                        "model_state_dict"
                    ],
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

            compact = {
                "condition": condition,
                "seed": seed,
                "best_epoch": result["best_epoch"],
                "best_val": result["best_val"],
            }
            all_results.append(compact)

            v = result["best_val"]

            gate_text = (
                ""
                if v["gate_mean"] is None
                else f" | gate={v['gate_mean']:.4f}"
            )

            what_text = (
                "WHAT MRR=N/A"
                if v["what"] is None
                else f"WHAT MRR={v['what']['MRR']:.6f}"
            )

            print(
                f"BEST | condition={condition} | "
                f"seed={seed} | "
                f"epoch={result['best_epoch']} | "
                f"WHY MRR={v['why']['MRR']:.6f} | "
                f"WHY Top1={v['why']['Top1']:.6f} | "
                f"WHY Top3={v['why']['Top3']:.6f} | "
                f"WHY Top5={v['why']['Top5']:.6f} | "
                f"{what_text} | "
                f"same-task-error="
                f"{v['why']['same_task_wrong_fraction']:.4f}"
                f"{gate_text}"
            )

    summary = {}

    for condition in args.conditions:
        condition_rows = [
            x
            for x in all_results
            if x["condition"] == condition
        ]
        summary[condition] = summarize(
            condition_rows
        )

    payload = {
        "experiment": "egointent_strong_what",
        "config": {
            **vars(args),
            "stage_a_cache": str(args.stage_a_cache),
            "temporal_cache": str(args.temporal_cache),
            "what_pca": str(args.what_pca),
            "why_pca": str(args.why_pca),
            "output_dir": str(args.output_dir),
        },
        "summary": summary,
        "results": all_results,
    }

    with open(
        args.output_dir / "summary.json",
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
    print("FINAL COMPARISON")
    print("=" * 96)

    for condition in args.conditions:
        row = summary[condition]

        gate_text = ""
        if "gate_mean_mean" in row:
            gate_text = (
                f" | gate "
                f"{row['gate_mean_mean']:.4f} ± "
                f"{row['gate_mean_sample_std']:.4f}"
            )

        what_text = (
            "WHAT MRR N/A"
            if "what_mrr_mean" not in row
            else (
                f"WHAT MRR "
                f"{row['what_mrr_mean']:.6f} ± "
                f"{row['what_mrr_sample_std']:.6f}"
            )
        )

        print(
            f"{condition:18s} | "
            f"WHY MRR "
            f"{row['why_mrr_mean']:.6f} ± "
            f"{row['why_mrr_sample_std']:.6f} | "
            f"WHY Top1 "
            f"{row['why_top1_mean']:.6f} ± "
            f"{row['why_top1_sample_std']:.6f} | "
            f"WHY Top3 "
            f"{row['why_top3_mean']:.6f} ± "
            f"{row['why_top3_sample_std']:.6f} | "
            f"WHY Top5 "
            f"{row['why_top5_mean']:.6f} ± "
            f"{row['why_top5_sample_std']:.6f} | "
            f"{what_text} | "
            f"same-task-error "
            f"{row['same_task_wrong_fraction_mean']:.4f} ± "
            f"{row['same_task_wrong_fraction_sample_std']:.4f}"
            f"{gate_text}"
        )

    print()
    print(
        "Saved:",
        args.output_dir / "summary.json",
    )


if __name__ == "__main__":
    main()
