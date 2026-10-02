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
        mean = torch.zeros(960, dtype=torch.float32)

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
# Cache loading
# ============================================================

def find_temporal_feature(x, path):
    candidates = [
        "frame_features",
        "temporal_features",
        "visual_features",
        "frames_feature",
    ]

    for key in candidates:
        if key in x and torch.is_tensor(x[key]):
            feat = x[key].float()
            if feat.ndim == 2 and feat.shape[-1] == 960:
                return feat

    tensor_keys = {
        k: tuple(v.shape)
        for k, v in x.items()
        if torch.is_tensor(v)
    }

    raise RuntimeError(
        "Could not find an [T,960] temporal feature tensor in "
        f"{path}. Tensor keys: {tensor_keys}"
    )


def load_stage_a_metadata(stage_a_root: Path, split: str):
    root = stage_a_root / split

    if not root.exists():
        raise FileNotFoundError(
            f"Missing Stage-A split: {root}"
        )

    metadata = {}

    for p in sorted(root.glob("*.pt")):
        x = torch.load(p, map_location="cpu")

        sample_id = str(x["sample_id"])

        metadata[sample_id] = {
            "sample_id": sample_id,
            "video_uid": str(x["video_uid"]),
            "task": str(x.get("task", "")),
            "why": str(x["why"]),
            "why_feature": x["why_feature"].float(),
        }

    if not metadata:
        raise RuntimeError(
            f"No Stage-A metadata found in {root}"
        )

    return metadata


def load_temporal_split(
    temporal_root: Path,
    stage_a_root: Path,
    split: str,
):
    root = temporal_root / split

    if not root.exists():
        raise FileNotFoundError(
            f"Missing temporal split: {root}"
        )

    metadata = load_stage_a_metadata(
        stage_a_root,
        split,
    )

    rows = []
    unmatched = []

    files = sorted(root.rglob("*.pt"))

    for p in files:
        x = torch.load(
            p,
            map_location="cpu",
        )

        sample_id = str(
            x.get(
                "sample_id",
                p.stem,
            )
        )

        if sample_id not in metadata:
            unmatched.append(sample_id)
            continue

        feat = find_temporal_feature(
            x,
            p,
        )

        if feat.shape[0] < 2:
            raise RuntimeError(
                f"Need at least 2 frames in {p}, got {tuple(feat.shape)}"
            )

        meta = metadata[sample_id]

        rows.append(
            {
                "sample_id": sample_id,
                "video_uid": meta["video_uid"],
                "task": meta["task"],
                "why": meta["why"],
                "why_feature": meta["why_feature"],
                "frame_features": feat,
            }
        )

    if not rows:
        raise RuntimeError(
            f"No matched temporal samples found in {root}"
        )

    if unmatched:
        print(
            f"WARNING: {len(unmatched)} temporal samples could not "
            "be matched to Stage-A metadata."
        )
        print("First unmatched IDs:", unmatched[:10])

    lengths = sorted(
        set(
            int(x["frame_features"].shape[0])
            for x in rows
        )
    )

    print(
        f"{split}: matched={len(rows)} "
        f"temporal_files={len(files)} "
        f"sequence_lengths={lengths}"
    )

    return rows


def stack_split(
    rows,
    why_transform,
    device,
):
    lengths = {
        int(x["frame_features"].shape[0])
        for x in rows
    }

    if len(lengths) != 1:
        raise RuntimeError(
            "This experiment expects a fixed temporal length. "
            f"Observed lengths: {sorted(lengths)}"
        )

    frames = torch.stack(
        [
            x["frame_features"]
            for x in rows
        ]
    ).to(device)

    why_960 = torch.stack(
        [
            x["why_feature"]
            for x in rows
        ]
    )

    why_target = why_transform(
        why_960
    ).to(device)

    why_target = F.normalize(
        why_target,
        dim=-1,
    )

    why_texts = [
        x["why"]
        for x in rows
    ]

    return frames, why_target, why_texts


# ============================================================
# Models
# ============================================================

class MeanPoolBaseline(nn.Module):
    """
    Controlled static baseline:
        [B,T,960] -> mean over T -> MLP -> 256
    """

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
            nn.Dropout(dropout),
            nn.Linear(
                hidden_dim,
                output_dim,
            ),
            nn.LayerNorm(output_dim),
        )

    def forward(self, frames):
        pooled = frames.mean(dim=1)
        z = self.net(pooled)

        return F.normalize(
            z,
            dim=-1,
        )


class TemporalGRUEncoder(nn.Module):
    """
    Causal temporal encoder.

    Frame features:
        [B,T,960]
            -> frame projection
            -> unidirectional GRU
            -> one 256-d prediction per observed prefix

    prefix_predictions[:, t] only uses frames <= t.
    """

    def __init__(
        self,
        input_dim=960,
        frame_dim=256,
        hidden_dim=256,
        output_dim=256,
        dropout=0.0,
    ):
        super().__init__()

        self.frame_proj = nn.Sequential(
            nn.Linear(
                input_dim,
                frame_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(frame_dim),
        )

        self.gru = nn.GRU(
            input_size=frame_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.why_head = nn.Sequential(
            nn.Linear(
                hidden_dim,
                output_dim,
            ),
            nn.LayerNorm(output_dim),
        )

    def forward_prefixes(self, frames):
        x = self.frame_proj(
            frames
        )

        h, _ = self.gru(
            x
        )

        h = self.dropout(
            h
        )

        z = self.why_head(
            h
        )

        return F.normalize(
            z,
            dim=-1,
        )

    def forward(self, frames):
        prefix_z = self.forward_prefixes(
            frames
        )

        return prefix_z[:, -1]


# ============================================================
# Losses
# ============================================================

def build_positive_mask(
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


def cosine_target_loss(
    pred,
    target,
):
    return (
        1.0
        - (
            pred
            * target
        ).sum(dim=-1)
    ).mean()


def progression_loss(
    prefix_z,
    why_target,
    prefix_indices,
    margin,
):
    """
    Encourage WHY alignment to improve as more frames are observed.

    sim_k = cosine(z_prefix_k, target_why)

    Desired:
        sim_{k+1} >= sim_k + margin

    This does not use future frames as input to an earlier prefix.
    Each prefix embedding is causal.
    """
    sims = []

    for idx in prefix_indices:
        z = prefix_z[:, idx]

        sim = (
            z
            * why_target
        ).sum(dim=-1)

        sims.append(sim)

    losses = []

    for a, b in zip(
        sims[:-1],
        sims[1:],
    ):
        losses.append(
            F.relu(
                a - b + margin
            ).mean()
        )

    if not losses:
        return torch.zeros(
            (),
            device=prefix_z.device,
        )

    return torch.stack(
        losses
    ).mean()


def prefix_supervision_loss(
    prefix_z,
    why_target,
    positive_mask,
    prefix_indices,
    prefix_weights,
    temperature,
    cosine_weight,
):
    total = torch.zeros(
        (),
        device=prefix_z.device,
    )

    weight_sum = 0.0

    for idx, weight in zip(
        prefix_indices,
        prefix_weights,
    ):
        pred = prefix_z[:, idx]

        loss_nce = (
            multi_positive_infonce(
                pred,
                why_target,
                positive_mask,
                temperature,
            )
        )

        loss_cos = (
            cosine_target_loss(
                pred,
                why_target,
            )
        )

        loss = (
            loss_nce
            + cosine_weight
            * loss_cos
        )

        total = (
            total
            + weight * loss
        )

        weight_sum += weight

    if weight_sum <= 0:
        return torch.zeros(
            (),
            device=prefix_z.device,
        )

    return total / weight_sum


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def retrieval_metrics(
    pred,
    why_target,
    why_texts,
):
    sim = pred @ why_target.T

    ranking = torch.argsort(
        sim,
        dim=1,
        descending=True,
    )

    ranks = []
    positive_scores = []
    hardest_negatives = []

    for i, query_why in enumerate(
        why_texts
    ):
        positive_idx = [
            j
            for j, gallery_why
            in enumerate(why_texts)
            if gallery_why == query_why
        ]

        negative_idx = [
            j
            for j, gallery_why
            in enumerate(why_texts)
            if gallery_why != query_why
        ]

        positive_set = set(
            positive_idx
        )

        first_rank = None

        for rank, j in enumerate(
            ranking[i].tolist(),
            start=1,
        ):
            if j in positive_set:
                first_rank = rank
                break

        ranks.append(
            first_rank
        )

        positive_scores.append(
            sim[i, positive_idx].max()
        )

        hardest_negatives.append(
            sim[i, negative_idx].max()
        )

    ranks = torch.tensor(
        ranks,
        dtype=torch.float32,
        device=pred.device,
    )

    positive_scores = torch.stack(
        positive_scores
    )

    hardest_negatives = torch.stack(
        hardest_negatives
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
        "positive_cosine_mean": (
            positive_scores
            .mean()
            .item()
        ),
        "hardest_negative_cosine_mean": (
            hardest_negatives
            .mean()
            .item()
        ),
        "positive_minus_hardest_margin": (
            positive_scores
            - hardest_negatives
        ).mean().item(),
    }


@torch.no_grad()
def evaluate_final(
    model,
    frames,
    why_target,
    why_texts,
):
    model.eval()

    pred = model(
        frames
    )

    return retrieval_metrics(
        pred,
        why_target,
        why_texts,
    )


@torch.no_grad()
def evaluate_prefixes(
    model,
    frames,
    why_target,
    why_texts,
    prefix_indices,
):
    if not isinstance(
        model,
        TemporalGRUEncoder,
    ):
        return {}

    model.eval()

    prefix_z = (
        model.forward_prefixes(
            frames
        )
    )

    result = {}

    for idx in prefix_indices:
        result[str(idx + 1)] = (
            retrieval_metrics(
                prefix_z[:, idx],
                why_target,
                why_texts,
            )
        )

    return result


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


def train_condition(
    condition,
    seed,
    train_frames,
    train_why_target,
    train_why_texts,
    val_frames,
    val_why_target,
    val_why_texts,
    args,
):
    set_seed(seed)

    device = train_frames.device
    seq_len = train_frames.shape[1]

    if condition == "mean_pool":
        model = MeanPoolBaseline(
            hidden_dim=args.static_hidden_dim,
            output_dim=256,
            dropout=args.dropout,
        ).to(device)

    elif condition in {
        "gru_final",
        "gru_progression",
    }:
        model = TemporalGRUEncoder(
            frame_dim=args.frame_dim,
            hidden_dim=args.gru_hidden_dim,
            output_dim=256,
            dropout=args.dropout,
        ).to(device)

    else:
        raise ValueError(
            f"Unknown condition: {condition}"
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    positive_mask = build_positive_mask(
        train_why_texts,
        device,
    )

    prefix_indices = [
        min(
            max(0, x - 1),
            seq_len - 1,
        )
        for x in args.prefix_frames
    ]

    # Remove duplicates while preserving order.
    prefix_indices = list(
        dict.fromkeys(
            prefix_indices
        )
    )

    if prefix_indices[-1] != seq_len - 1:
        prefix_indices.append(
            seq_len - 1
        )

    aux_prefix_indices = (
        prefix_indices[:-1]
    )

    if len(aux_prefix_indices) > 0:
        if len(args.prefix_weights) == len(
            aux_prefix_indices
        ):
            prefix_weights = list(
                args.prefix_weights
            )
        else:
            # Increasing default weights toward the current/final frame.
            prefix_weights = [
                (i + 1)
                / len(aux_prefix_indices)
                for i in range(
                    len(aux_prefix_indices)
                )
            ]
    else:
        prefix_weights = []

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

        if condition == "mean_pool":
            final_pred = model(
                train_frames
            )

            loss_final_nce = (
                multi_positive_infonce(
                    final_pred,
                    train_why_target,
                    positive_mask,
                    args.temperature,
                )
            )

            loss_final_cos = (
                cosine_target_loss(
                    final_pred,
                    train_why_target,
                )
            )

            loss_final = (
                loss_final_nce
                + args.cosine_weight
                * loss_final_cos
            )

            loss_prefix = torch.zeros(
                (),
                device=device,
            )

            loss_prog = torch.zeros(
                (),
                device=device,
            )

        else:
            prefix_z = (
                model.forward_prefixes(
                    train_frames
                )
            )

            final_pred = (
                prefix_z[:, -1]
            )

            loss_final_nce = (
                multi_positive_infonce(
                    final_pred,
                    train_why_target,
                    positive_mask,
                    args.temperature,
                )
            )

            loss_final_cos = (
                cosine_target_loss(
                    final_pred,
                    train_why_target,
                )
            )

            loss_final = (
                loss_final_nce
                + args.cosine_weight
                * loss_final_cos
            )

            if condition == "gru_progression":
                loss_prefix = (
                    prefix_supervision_loss(
                        prefix_z,
                        train_why_target,
                        positive_mask,
                        aux_prefix_indices,
                        prefix_weights,
                        args.temperature,
                        args.cosine_weight,
                    )
                )

                loss_prog = (
                    progression_loss(
                        prefix_z,
                        train_why_target,
                        prefix_indices,
                        args.progression_margin,
                    )
                )
            else:
                loss_prefix = torch.zeros(
                    (),
                    device=device,
                )

                loss_prog = torch.zeros(
                    (),
                    device=device,
                )

        loss = (
            loss_final
            + args.prefix_weight
            * loss_prefix
            + args.progression_weight
            * loss_prog
        )

        loss.backward()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.grad_clip,
            )

        optimizer.step()

        should_eval = (
            epoch == 1
            or epoch % args.eval_every == 0
            or epoch == args.epochs
        )

        if should_eval:
            val_metrics = evaluate_final(
                model,
                val_frames,
                val_why_target,
                val_why_texts,
            )

            print(
                f"[{condition}] "
                f"seed={seed} "
                f"epoch={epoch:03d} "
                f"loss={loss.item():.5f} "
                f"final={loss_final.item():.5f} "
                f"prefix={loss_prefix.item():.5f} "
                f"prog={loss_prog.item():.5f} "
                f"VAL_MRR={val_metrics['MRR']:.6f}"
            )

            history.append(
                {
                    "epoch": epoch,
                    "loss": loss.item(),
                    "loss_final": (
                        loss_final.item()
                    ),
                    "loss_prefix": (
                        loss_prefix.item()
                    ),
                    "loss_progression": (
                        loss_prog.item()
                    ),
                    "val": val_metrics,
                }
            )

            if (
                val_metrics["MRR"]
                > best_val_mrr
            ):
                best_val_mrr = (
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

    final_val = evaluate_final(
        model,
        val_frames,
        val_why_target,
        val_why_texts,
    )

    prefix_val = evaluate_prefixes(
        model,
        val_frames,
        val_why_target,
        val_why_texts,
        prefix_indices,
    )

    return {
        "condition": condition,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val": final_val,
        "prefix_val": prefix_val,
        "history": history,
        "state_dict": best_state,
        "prefix_indices": [
            x + 1
            for x in prefix_indices
        ],
    }


# ============================================================
# Summary
# ============================================================

def metric_mean_std(values):
    arr = np.asarray(
        values,
        dtype=np.float64,
    )

    mean = float(
        arr.mean()
    )

    std = (
        float(
            arr.std(ddof=1)
        )
        if len(arr) > 1
        else 0.0
    )

    return mean, std


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
            "temporal_progression_v1"
        ),
    )

    parser.add_argument(
        "--conditions",
        nargs="+",
        default=[
            "mean_pool",
            "gru_final",
            "gru_progression",
        ],
        choices=[
            "mean_pool",
            "gru_final",
            "gru_progression",
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
        "--prefix-weight",
        type=float,
        default=0.3,
    )

    parser.add_argument(
        "--progression-weight",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--progression-margin",
        type=float,
        default=0.02,
    )

    parser.add_argument(
        "--prefix-frames",
        nargs="+",
        type=int,
        default=[
            2,
            4,
            6,
            8,
        ],
        help=(
            "Observed frame counts used for progression evaluation. "
            "For the default 8-frame cache this means 2/4/6/8 frames."
        ),
    )

    parser.add_argument(
        "--prefix-weights",
        nargs="*",
        type=float,
        default=[
            0.25,
            0.5,
            0.75,
        ],
        help=(
            "Auxiliary weights for non-final prefixes. "
            "If length does not match, increasing weights are generated."
        ),
    )

    parser.add_argument(
        "--static-hidden-dim",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--frame-dim",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--gru-hidden-dim",
        type=int,
        default=256,
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
        load_temporal_split(
            args.temporal_cache,
            args.stage_a_cache,
            "train",
        )
    )

    val_rows = (
        load_temporal_split(
            args.temporal_cache,
            args.stage_a_cache,
            "val",
        )
    )

    (
        train_frames,
        train_why_target,
        train_why_texts,
    ) = stack_split(
        train_rows,
        why_transform,
        device,
    )

    (
        val_frames,
        val_why_target,
        val_why_texts,
    ) = stack_split(
        val_rows,
        why_transform,
        device,
    )

    print()
    print("=" * 80)
    print(
        "TEMPORAL PROGRESSION INTENTION EXPERIMENT"
    )
    print("=" * 80)
    print("device:", device)
    print(
        "train samples:",
        len(train_rows),
    )
    print(
        "val samples:",
        len(val_rows),
    )
    print(
        "frame tensor:",
        tuple(train_frames.shape),
    )
    print(
        "conditions:",
        args.conditions,
    )
    print(
        "seeds:",
        args.seeds,
    )

    all_results = []

    for condition in args.conditions:
        print()
        print("#" * 80)
        print(
            f"CONDITION: {condition}"
        )
        print("#" * 80)

        condition_dir = (
            args.output_dir
            / condition
        )

        condition_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for seed in args.seeds:
            result = train_condition(
                condition=condition,
                seed=seed,
                train_frames=train_frames,
                train_why_target=train_why_target,
                train_why_texts=train_why_texts,
                val_frames=val_frames,
                val_why_target=val_why_target,
                val_why_texts=val_why_texts,
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
                    "condition": condition,
                    "seed": seed,
                    "best_epoch": (
                        result[
                            "best_epoch"
                        ]
                    ),
                    "model_state_dict": (
                        result[
                            "state_dict"
                        ]
                    ),
                    "best_val": (
                        result[
                            "best_val"
                        ]
                    ),
                    "prefix_val": (
                        result[
                            "prefix_val"
                        ]
                    ),
                    "prefix_indices": (
                        result[
                            "prefix_indices"
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
                "prefix_val": (
                    result[
                        "prefix_val"
                    ]
                ),
                "prefix_indices": (
                    result[
                        "prefix_indices"
                    ]
                ),
            }

            all_results.append(
                compact
            )

            print()
            print(
                f"FINAL | {condition} | seed={seed}"
            )
            print(
                "best_epoch:",
                result["best_epoch"],
            )
            print(
                "WHY val:",
                result["best_val"],
            )

            if result["prefix_val"]:
                print(
                    "Prefix MRR:",
                    {
                        k: round(
                            v["MRR"],
                            6,
                        )
                        for k, v
                        in result[
                            "prefix_val"
                        ].items()
                    },
                )

    summary = {}

    for condition in args.conditions:
        rows = [
            x
            for x in all_results
            if x["condition"]
            == condition
        ]

        mrr = [
            x["best_val"]["MRR"]
            for x in rows
        ]

        top1 = [
            x["best_val"]["Top1"]
            for x in rows
        ]

        top3 = [
            x["best_val"]["Top3"]
            for x in rows
        ]

        mrr_m, mrr_s = (
            metric_mean_std(
                mrr
            )
        )

        top1_m, top1_s = (
            metric_mean_std(
                top1
            )
        )

        top3_m, top3_s = (
            metric_mean_std(
                top3
            )
        )

        prefix_summary = {}

        prefix_keys = sorted(
            {
                k
                for x in rows
                for k in x[
                    "prefix_val"
                ].keys()
            },
            key=lambda x: int(x),
        )

        for k in prefix_keys:
            values = [
                x["prefix_val"][k]["MRR"]
                for x in rows
                if k in x["prefix_val"]
            ]

            pm, ps = (
                metric_mean_std(
                    values
                )
            )

            prefix_summary[k] = {
                "values": values,
                "mean": pm,
                "sample_std": ps,
            }

        summary[condition] = {
            "mrr_values": mrr,
            "mrr_mean": mrr_m,
            "mrr_sample_std": mrr_s,
            "top1_values": top1,
            "top1_mean": top1_m,
            "top1_sample_std": (
                top1_s
            ),
            "top3_values": top3,
            "top3_mean": top3_m,
            "top3_sample_std": (
                top3_s
            ),
            "prefix_mrr": (
                prefix_summary
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

    for condition in args.conditions:
        row = summary[
            condition
        ]

        print(
            f"{condition:16s} | "
            f"MRR "
            f"{row['mrr_mean']:.6f} ± "
            f"{row['mrr_sample_std']:.6f} | "
            f"Top1 "
            f"{row['top1_mean']:.6f} ± "
            f"{row['top1_sample_std']:.6f} | "
            f"Top3 "
            f"{row['top3_mean']:.6f} ± "
            f"{row['top3_sample_std']:.6f}"
        )

        if row["prefix_mrr"]:
            prefix_text = " | ".join(
                (
                    f"{k}f="
                    f"{v['mean']:.6f}±"
                    f"{v['sample_std']:.6f}"
                )
                for k, v
                in row[
                    "prefix_mrr"
                ].items()
            )

            print(
                "  prefix MRR:",
                prefix_text,
            )

    print()
    print(
        "Saved summary:",
        args.output_dir
        / "summary.json",
    )


if __name__ == "__main__":
    main()
