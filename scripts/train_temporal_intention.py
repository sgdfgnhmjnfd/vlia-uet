from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from models.temporal_intention_encoder import (
    TemporalIntentionEncoder,
)
from models.mean_pool_intention_encoder import (
    MeanPoolIntentionEncoder,
)
from models.temporal_attention_intention_encoder import (
    TemporalAttentionIntentionEncoder,
)
DEFAULT_TEMPORAL_CACHE_ROOT = Path(
    "/media/dhqg/d1/datasets/egointent/"
    "cache/temporal_intention_v1"
)

DEFAULT_STAGE_A_CACHE_ROOT = Path(
    "/media/dhqg/d1/datasets/egointent/"
    "cache/stage_a_v0"
)

DEFAULT_PCA_PATH = Path(
    "/media/dhqg/d1/vlia_outputs/"
    "stage_a_clean_v1/"
    "why_pca_960_to_256.pt"
)

DEFAULT_OUTPUT_ROOT = Path(
    "/media/dhqg/d1/vlia_outputs/"
    "temporal_intention_v1"
)


# ============================================================
# Utilities
# ============================================================


def normalize_text(text: str) -> str:
    return " ".join(
        str(text)
        .strip()
        .lower()
        .split()
    )


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(
    obj,
    path: Path,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            obj,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


# ============================================================
# PCA target projection
# ============================================================


class WhyPCAProjector:
    def __init__(
        self,
        path: Path,
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"PCA artifact not found: {path}"
            )

        artifact = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        self.mean = artifact[
            "mean"
        ].float()

        self.components = artifact[
            "components"
        ].float()

        self.input_dim = int(
            artifact.get(
                "input_dim",
                self.mean.shape[-1],
            )
        )

        self.output_dim = int(
            artifact.get(
                "output_dim",
                self.components.shape[-1],
            )
        )

        if tuple(
            self.mean.shape
        ) != (
            1,
            self.input_dim,
        ):
            raise ValueError(
                "Unexpected PCA mean shape: "
                f"{tuple(self.mean.shape)}"
            )

        if tuple(
            self.components.shape
        ) != (
            self.input_dim,
            self.output_dim,
        ):
            raise ValueError(
                "Unexpected PCA component shape: "
                f"{tuple(self.components.shape)}"
            )

    def __call__(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                "Unexpected PCA input dimension: "
                f"{x.shape[-1]}, "
                f"expected {self.input_dim}."
            )

        return (
            x.float()
            - self.mean
        ) @ self.components


# ============================================================
# WHY target lookup
# ============================================================


def build_stage_a_index(
    root: Path,
    split: str,
):
    """
    Build sample_id -> Stage-A cache path.

    Supports both:

        stage_a_v0/train/*.pt

    and legacy:

        stage_a_v0/samples/train/*.pt
    """

    candidate_dirs = [
        root / split,
        root / "samples" / split,
    ]

    paths = []

    for directory in candidate_dirs:
        if directory.exists():
            paths.extend(
                sorted(
                    directory.glob("*.pt")
                )
            )

    if not paths:
        raise FileNotFoundError(
            "No Stage-A cache files found "
            f"for split={split} under {root}"
        )

    index = {}

    for path in paths:
        try:
            record = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )
        except Exception:
            continue

        sample_id = record.get(
            "sample_id"
        )

        if sample_id is None:
            continue

        index[
            str(sample_id)
        ] = path

    if not index:
        raise RuntimeError(
            "No usable Stage-A samples "
            f"found for split={split}."
        )

    return index


def load_why_feature(
    path: Path,
):
    record = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if "why_feature" in record:
        feature = record[
            "why_feature"
        ]

    elif "why" in record and isinstance(
        record["why"],
        torch.Tensor,
    ):
        feature = record[
            "why"
        ]

    else:
        raise KeyError(
            "Stage-A record does not contain "
            f"'why_feature': {path}"
        )

    feature = (
        feature
        .detach()
        .float()
        .cpu()
        .reshape(-1)
    )

    if feature.shape[0] != 960:
        raise ValueError(
            "Unexpected WHY feature shape "
            f"{tuple(feature.shape)} "
            f"in {path}"
        )

    if not torch.isfinite(
        feature
    ).all():
        raise ValueError(
            f"Non-finite WHY feature: {path}"
        )

    return feature


# ============================================================
# Dataset
# ============================================================


class TemporalIntentionDataset(
    Dataset
):
    def __init__(
        self,
        temporal_root: Path,
        stage_a_root: Path,
        pca_projector: WhyPCAProjector,
        split: str,
    ):
        self.split = split

        temporal_dir = (
            temporal_root
            / split
        )

        if not temporal_dir.exists():
            raise FileNotFoundError(
                "Temporal cache directory "
                f"not found: {temporal_dir}"
            )

        self.paths = sorted(
            temporal_dir.glob(
                "*.pt"
            )
        )

        if not self.paths:
            raise RuntimeError(
                "No temporal cache samples "
                f"found in {temporal_dir}"
            )

        self.stage_a_index = (
            build_stage_a_index(
                stage_a_root,
                split,
            )
        )

        self.pca = pca_projector

        self.records = []

        missing_targets = []

        for path in self.paths:
            record = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )

            sample_id = str(
                record["sample_id"]
            )

            if sample_id not in (
                self.stage_a_index
            ):
                missing_targets.append(
                    sample_id
                )

                continue

            self.records.append(
                {
                    "path": path,
                    "sample_id": (
                        sample_id
                    ),
                }
            )

        if missing_targets:
            preview = (
                missing_targets[:10]
            )

            raise RuntimeError(
                "Missing Stage-A WHY targets "
                f"for {len(missing_targets)} "
                "temporal samples. "
                f"Examples: {preview}"
            )

        if not self.records:
            raise RuntimeError(
                "Temporal dataset is empty."
            )

    def __len__(self):
        return len(
            self.records
        )

    def __getitem__(
        self,
        index,
    ):
        entry = self.records[
            index
        ]

        temporal_record = (
            torch.load(
                entry["path"],
                map_location="cpu",
                weights_only=False,
            )
        )

        sample_id = (
            entry["sample_id"]
        )

        frame_features = (
            temporal_record[
                "frame_features"
            ]
            .float()
        )

        if (
            frame_features.ndim != 2
            or frame_features.shape[-1]
            != 960
        ):
            raise ValueError(
                "Invalid temporal feature "
                f"shape for {sample_id}: "
                f"{tuple(frame_features.shape)}"
            )

        if not torch.isfinite(
            frame_features
        ).all():
            raise ValueError(
                "Non-finite temporal feature "
                f"for {sample_id}"
            )

        stage_a_path = (
            self.stage_a_index[
                sample_id
            ]
        )

        why_feature = (
            load_why_feature(
                stage_a_path
            )
        )

        why_target = (
            self.pca(
                why_feature.unsqueeze(
                    0
                )
            )
            .squeeze(0)
        )

        if not torch.isfinite(
            why_target
        ).all():
            raise ValueError(
                "Non-finite projected WHY "
                f"target for {sample_id}"
            )

        why_text = str(
            temporal_record[
                "why"
            ]
        )

        return {
            "sample_id":
                sample_id,

            "frame_features":
                frame_features,

            "why_target":
                why_target,

            "why":
                why_text,

            "why_normalized":
                normalize_text(
                    why_text
                ),
        }


def collate_batch(
    batch,
):
    frame_features = torch.stack(
        [
            x["frame_features"]
            for x in batch
        ],
        dim=0,
    )

    why_target = torch.stack(
        [
            x["why_target"]
            for x in batch
        ],
        dim=0,
    )

    return {
        "sample_id": [
            x["sample_id"]
            for x in batch
        ],

        "frame_features":
            frame_features,

        "why_target":
            why_target,

        "why": [
            x["why"]
            for x in batch
        ],

        "why_normalized": [
            x["why_normalized"]
            for x in batch
        ],
    }


# ============================================================
# Multi-positive loss
# ============================================================


def build_positive_mask(
    labels,
    device,
):
    batch_size = len(
        labels
    )

    mask = torch.zeros(
        (
            batch_size,
            batch_size,
        ),
        dtype=torch.bool,
        device=device,
    )

    for i in range(
        batch_size
    ):
        for j in range(
            batch_size
        ):
            if labels[i] == labels[j]:
                mask[
                    i,
                    j,
                ] = True

    return mask


def multi_positive_infonce(
    predicted,
    target,
    labels,
    temperature,
):
    predicted = F.normalize(
        predicted,
        dim=-1,
    )

    target = F.normalize(
        target,
        dim=-1,
    )

    logits = (
        predicted
        @ target.transpose(
            0,
            1,
        )
    ) / temperature

    positive_mask = (
        build_positive_mask(
            labels,
            logits.device,
        )
    )

    log_denominator = (
        torch.logsumexp(
            logits,
            dim=1,
        )
    )

    positive_logits = (
        logits.masked_fill(
            ~positive_mask,
            float("-inf"),
        )
    )

    log_numerator = (
        torch.logsumexp(
            positive_logits,
            dim=1,
        )
    )

    loss = -(
        log_numerator
        - log_denominator
    ).mean()

    return loss


def cosine_alignment_loss(
    predicted,
    target,
):
    return (
        1.0
        - F.cosine_similarity(
            predicted,
            target,
            dim=-1,
        )
    ).mean()


# ============================================================
# Retrieval metrics
# ============================================================


@torch.inference_mode()
def retrieval_metrics(
    predictions,
    targets,
    labels,
):
    predictions = F.normalize(
        predictions.float(),
        dim=-1,
    )

    targets = F.normalize(
        targets.float(),
        dim=-1,
    )

    similarities = (
        predictions
        @ targets.transpose(
            0,
            1,
        )
    )

    num_samples = (
        similarities.shape[0]
    )

    top1_hits = 0
    top3_hits = 0

    reciprocal_ranks = []
    margins = []

    for i in range(
        num_samples
    ):
        query_label = labels[
            i
        ]

        positive_mask = torch.tensor(
            [
                label
                == query_label
                for label in labels
            ],
            dtype=torch.bool,
            device=(
                similarities.device
            ),
        )

        negative_mask = (
            ~positive_mask
        )

        row = similarities[
            i
        ]

        ranking = torch.argsort(
            row,
            descending=True,
        )

        ranked_positive = (
            positive_mask[
                ranking
            ]
        )

        positive_positions = (
            torch.nonzero(
                ranked_positive,
                as_tuple=False,
            )
            .reshape(-1)
        )

        if len(
            positive_positions
        ) == 0:
            raise RuntimeError(
                "Query has no positive "
                "retrieval target."
            )

        first_rank = int(
            positive_positions[
                0
            ].item()
        ) + 1

        reciprocal_ranks.append(
            1.0 / first_rank
        )

        if ranked_positive[
            :1
        ].any():
            top1_hits += 1

        if ranked_positive[
            :3
        ].any():
            top3_hits += 1

        best_positive = (
            row[
                positive_mask
            ]
            .max()
        )

        if negative_mask.any():
            best_negative = (
                row[
                    negative_mask
                ]
                .max()
            )

            margins.append(
                (
                    best_positive
                    - best_negative
                ).item()
            )

    metrics = {
        "top1":
            top1_hits
            / num_samples,

        "top3":
            top3_hits
            / num_samples,

        "mrr":
            float(
                np.mean(
                    reciprocal_ranks
                )
            ),

        "margin":
            float(
                np.mean(
                    margins
                )
            )
            if margins
            else float("nan"),

        "mean_cosine":
            float(
                F.cosine_similarity(
                    predictions,
                    targets,
                    dim=-1,
                )
                .mean()
                .item()
            ),
    }

    return metrics


# ============================================================
# Evaluation
# ============================================================


@torch.inference_mode()
def evaluate(
    model,
    loader,
    device,
):
    model.eval()

    all_predictions = []
    all_targets = []
    all_labels = []

    for batch in loader:
        frame_features = (
            batch[
                "frame_features"
            ]
            .to(
                device,
                non_blocking=True,
            )
        )

        targets = (
            batch[
                "why_target"
            ]
            .to(
                device,
                non_blocking=True,
            )
        )

        predictions = model(
            frame_features
        )

        if isinstance(
            predictions,
            dict,
        ):
            predictions = (
                predictions[
                    "z_int"
                ]
            )

        all_predictions.append(
            predictions.detach().cpu()
        )

        all_targets.append(
            targets.detach().cpu()
        )

        all_labels.extend(
            batch[
                "why_normalized"
            ]
        )

    predictions = torch.cat(
        all_predictions,
        dim=0,
    )

    targets = torch.cat(
        all_targets,
        dim=0,
    )

    return retrieval_metrics(
        predictions,
        targets,
        all_labels,
    )


# ============================================================
# Checkpoint
# ============================================================


def save_best_artifact(
    model,
    output_dir: Path,
    args,
    epoch,
    metrics,
):
    artifact = {
        "model_type":
            type(model).__name__,

        "encoder":
            args.encoder,

        "epoch":
            epoch,

        "model_state_dict":
            model.state_dict(),

        "config": {
            "visual_dim":
                args.visual_dim,

            "model_dim":
                args.model_dim,

            "intention_dim":
                args.intention_dim,

            "num_layers":
                args.num_layers,

            "num_heads":
                args.num_heads,

            "max_frames":
                args.max_frames,

            "dropout":
                args.dropout,
        },

        "training": {
            "temperature":
                args.temperature,

            "cosine_weight":
                args.cosine_weight,

            "seed":
                args.seed,
        },

        "val_metrics":
            metrics,
    }

    torch.save(
        artifact,
        (
            output_dir
            / "intention_encoder.pt"
        ),
    )


# ============================================================
# CLI
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--temporal-cache-root",
        type=Path,
        default=(
            DEFAULT_TEMPORAL_CACHE_ROOT
        ),
    )

    parser.add_argument(
        "--stage-a-cache-root",
        type=Path,
        default=(
            DEFAULT_STAGE_A_CACHE_ROOT
        ),
    )

    parser.add_argument(
        "--pca-path",
        type=Path,
        default=DEFAULT_PCA_PATH,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
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
        "--visual-dim",
        type=int,
        default=960,
    )

    parser.add_argument(
        "--model-dim",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--intention-dim",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--num-heads",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--encoder",
        choices=[
            "mean_pool",
            "temporal_attention",
            "temporal_transformer",
        ],
        default="temporal_transformer",
    )
    return parser.parse_args()


# ============================================================
# Main
# ============================================================


def main():
    args = parse_args()

    set_seed(
        args.seed
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    output_dir = (
        args.output_root
        / args.encoder
        / f"seed_{args.seed}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print(
        "TEMPORAL INTENTION TRAINING"
    )
    print("=" * 72)

    print(
        "encoder:",
        args.encoder,
    )

    print(
        "device:",
        device,
    )

    print(
        "seed:",
        args.seed,
    )

    print(
        "temporal cache:",
        args.temporal_cache_root,
    )

    print(
        "stage-a cache:",
        args.stage_a_cache_root,
    )

    print(
        "PCA:",
        args.pca_path,
    )

    print(
        "output:",
        output_dir,
    )

    print()

    pca_projector = (
        WhyPCAProjector(
            args.pca_path
        )
    )

    if (
        pca_projector.output_dim
        != args.intention_dim
    ):
        raise ValueError(
            "PCA output dimension does not "
            "match intention dimension: "
            f"{pca_projector.output_dim} "
            "vs "
            f"{args.intention_dim}"
        )

    train_dataset = (
        TemporalIntentionDataset(
            temporal_root=(
                args.temporal_cache_root
            ),
            stage_a_root=(
                args.stage_a_cache_root
            ),
            pca_projector=(
                pca_projector
            ),
            split="train",
        )
    )

    val_dataset = (
        TemporalIntentionDataset(
            temporal_root=(
                args.temporal_cache_root
            ),
            stage_a_root=(
                args.stage_a_cache_root
            ),
            pca_projector=(
                pca_projector
            ),
            split="val",
        )
    )

    print(
        "train samples:",
        len(train_dataset),
    )

    print(
        "val samples:",
        len(val_dataset),
    )

    train_generator = (
        torch.Generator()
    )

    train_generator.manual_seed(
        args.seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=train_generator,
        num_workers=args.num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
        collate_fn=collate_batch,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
        collate_fn=collate_batch,
        drop_last=False,
    )
    if args.encoder == "mean_pool":
        model = MeanPoolIntentionEncoder(
            visual_dim=args.visual_dim,
            model_dim=args.model_dim,
            intention_dim=(
                args.intention_dim
            ),
        )

    elif args.encoder == "temporal_attention":
        model = TemporalAttentionIntentionEncoder(
            visual_dim=args.visual_dim,
            model_dim=args.model_dim,
            intention_dim=(
                args.intention_dim
            ),
            max_frames=args.max_frames,
            dropout=args.dropout,
        )

    elif (
        args.encoder
        == "temporal_transformer"
    ):
        model = TemporalIntentionEncoder(
            visual_dim=args.visual_dim,
            model_dim=args.model_dim,
            intention_dim=(
                args.intention_dim
            ),
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            max_frames=args.max_frames,
            dropout=args.dropout,
        )

    else:
        raise ValueError(
            f"Unknown encoder: "
            f"{args.encoder}"
        )

    model = model.to(
        device
    )

    optimizer = (
        torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=(
                args.weight_decay
            ),
        )
    )

    scheduler = (
        torch.optim.lr_scheduler
        .CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
        )
    )

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print()
    print(
        "parameters:",
        parameter_count,
    )

    print(
        "trainable:",
        trainable_count,
    )

    print()

    history = []

    best_mrr = -math.inf
    best_epoch = -1
    best_metrics = None

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        model.train()

        epoch_losses = []
        epoch_contrastive = []
        epoch_cosine = []

        for batch in train_loader:
            frame_features = (
                batch[
                    "frame_features"
                ]
                .to(
                    device,
                    non_blocking=True,
                )
            )

            targets = (
                batch[
                    "why_target"
                ]
                .to(
                    device,
                    non_blocking=True,
                )
            )

            labels = batch[
                "why_normalized"
            ]

            optimizer.zero_grad(
                set_to_none=True
            )

            predictions = model(
                frame_features
            )

            if isinstance(
                predictions,
                dict,
            ):
                predictions = (
                    predictions[
                        "z_int"
                    ]
                )

            contrastive_loss = (
                multi_positive_infonce(
                    predictions,
                    targets,
                    labels,
                    temperature=(
                        args.temperature
                    ),
                )
            )

            cosine_loss = (
                cosine_alignment_loss(
                    predictions,
                    targets,
                )
            )

            loss = (
                contrastive_loss
                + args.cosine_weight
                * cosine_loss
            )

            if not torch.isfinite(
                loss
            ):
                raise RuntimeError(
                    "Non-finite training loss."
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            epoch_losses.append(
                float(
                    loss.item()
                )
            )

            epoch_contrastive.append(
                float(
                    contrastive_loss.item()
                )
            )

            epoch_cosine.append(
                float(
                    cosine_loss.item()
                )
            )

        scheduler.step()

        val_metrics = evaluate(
            model,
            val_loader,
            device,
        )

        train_loss = float(
            np.mean(
                epoch_losses
            )
        )

        train_contrastive = float(
            np.mean(
                epoch_contrastive
            )
        )

        train_cosine = float(
            np.mean(
                epoch_cosine
            )
        )

        lr = float(
            optimizer.param_groups[
                0
            ][
                "lr"
            ]
        )

        row = {
            "epoch":
                epoch,

            "train_loss":
                train_loss,

            "train_contrastive":
                train_contrastive,

            "train_cosine":
                train_cosine,

            "lr":
                lr,

            "val_top1":
                val_metrics[
                    "top1"
                ],

            "val_top3":
                val_metrics[
                    "top3"
                ],

            "val_mrr":
                val_metrics[
                    "mrr"
                ],

            "val_margin":
                val_metrics[
                    "margin"
                ],

            "val_mean_cosine":
                val_metrics[
                    "mean_cosine"
                ],
        }

        history.append(
            row
        )

        print(
            f"epoch={epoch:03d} "
            f"loss={train_loss:.6f} "
            f"contrastive="
            f"{train_contrastive:.6f} "
            f"cos="
            f"{train_cosine:.6f} "
            f"| "
            f"top1="
            f"{val_metrics['top1']:.4f} "
            f"top3="
            f"{val_metrics['top3']:.4f} "
            f"mrr="
            f"{val_metrics['mrr']:.4f} "
            f"margin="
            f"{val_metrics['margin']:.4f} "
            f"cos="
            f"{val_metrics['mean_cosine']:.4f}",
            flush=True,
        )

        save_json(
            history,
            output_dir
            / "history.json",
        )

        if (
            val_metrics[
                "mrr"
            ]
            > best_mrr
        ):
            best_mrr = (
                val_metrics[
                    "mrr"
                ]
            )

            best_epoch = epoch

            best_metrics = dict(
                val_metrics
            )

            save_best_artifact(
                model=model,
                output_dir=(
                    output_dir
                ),
                args=args,
                epoch=epoch,
                metrics=(
                    val_metrics
                ),
            )

            print(
                "  -> NEW BEST "
                f"MRR={best_mrr:.6f}",
                flush=True,
            )

    summary = {
        "encoder":
            args.encoder,

        "seed":
            args.seed,

        "best_epoch":
            best_epoch,

        "best_val":
            best_metrics,

        "train_samples":
            len(
                train_dataset
            ),

        "val_samples":
            len(
                val_dataset
            ),

        "model": {
            "encoder":
                args.encoder,

            "visual_dim":
                args.visual_dim,

            "model_dim":
                args.model_dim,

            "intention_dim":
                args.intention_dim,

            "num_layers":
                args.num_layers,

            "num_heads":
                args.num_heads,

            "max_frames":
                args.max_frames,

            "dropout":
                args.dropout,

            "parameters":
                parameter_count,

            "trainable_parameters":
                trainable_count,
        },

        "training": {
            "epochs":
                args.epochs,

            "batch_size":
                args.batch_size,

            "lr":
                args.lr,

            "weight_decay":
                args.weight_decay,

            "temperature":
                args.temperature,

            "cosine_weight":
                args.cosine_weight,
        },

        "artifacts": {
            "encoder":
                str(
                    output_dir
                    / "intention_encoder.pt"
                ),

            "history":
                str(
                    output_dir
                    / "history.json"
                ),
        },
    }

    save_json(
        summary,
        output_dir
        / "summary.json",
    )

    print()
    print("=" * 72)
    print("TRAINING COMPLETE")
    print("=" * 72)

    print(
        "best_epoch:",
        best_epoch,
    )

    print(
        "best_mrr:",
        best_mrr,
    )

    print(
        "best_metrics:",
        best_metrics,
    )

    print(
        "artifact:",
        output_dir
        / "intention_encoder.pt",
    )


if __name__ == "__main__":
    main()