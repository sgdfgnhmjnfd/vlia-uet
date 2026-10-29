from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from models.structured_temporal_intention_encoder import (
    StructuredTemporalIntentionEncoder,
)


DEFAULT_TEMPORAL_CACHE_ROOT = Path(
    "/media/dhqg/d1/datasets/egointent/"
    "cache/temporal_intention_v1"
)

DEFAULT_WHY_PCA_PATH = Path(
    "/media/dhqg/d1/vlia_outputs/"
    "stage_a_clean_v1/"
    "why_pca_960_to_256.pt"
)

DEFAULT_WHAT_PCA_PATH = Path(
    "/media/dhqg/d1/vlia_outputs/"
    "structured_intention_v1/"
    "what_pca_960_to_256.pt"
)

DEFAULT_NEXT_PCA_PATH = Path(
    "/media/dhqg/d1/vlia_outputs/"
    "structured_intention_v1/"
    "next_pca_960_to_256.pt"
)

DEFAULT_OUTPUT_ROOT = Path(
    "/media/dhqg/d1/vlia_outputs/"
    "structured_intention_v1"
)


# ============================================================
# Utilities
# ============================================================


def normalize_text(
    text: str,
) -> str:
    return " ".join(
        str(text)
        .strip()
        .lower()
        .split()
    )


def set_seed(
    seed: int,
):
    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


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
# PCA projection
# ============================================================


class PCAProjector:
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

        self.path = path

        self.mean = (
            artifact[
                "mean"
            ]
            .detach()
            .float()
            .cpu()
        )

        self.components = (
            artifact[
                "components"
            ]
            .detach()
            .float()
            .cpu()
        )

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

        self.fit_split = str(
            artifact.get(
                "fit_split",
                "unknown",
            )
        )

        if tuple(
            self.mean.shape
        ) != (
            1,
            self.input_dim,
        ):
            raise ValueError(
                "Unexpected PCA mean shape "
                f"in {path}: "
                f"{tuple(self.mean.shape)}"
            )

        if tuple(
            self.components.shape
        ) != (
            self.input_dim,
            self.output_dim,
        ):
            raise ValueError(
                "Unexpected PCA component shape "
                f"in {path}: "
                f"{tuple(self.components.shape)}"
            )

        if not torch.isfinite(
            self.mean
        ).all():
            raise ValueError(
                f"Non-finite PCA mean: {path}"
            )

        if not torch.isfinite(
            self.components
        ).all():
            raise ValueError(
                f"Non-finite PCA components: {path}"
            )

    def __call__(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                "Unexpected PCA input dimension: "
                f"{x.shape[-1]}, expected "
                f"{self.input_dim}."
            )

        return (
            x.float()
            - self.mean
        ) @ self.components


# ============================================================
# Dataset
# ============================================================


class StructuredTemporalIntentionDataset(
    Dataset
):
    def __init__(
        self,
        temporal_root: Path,
        split: str,
        what_pca: PCAProjector,
        why_pca: PCAProjector,
        next_pca: PCAProjector,
    ):
        super().__init__()

        self.split = split

        self.what_pca = what_pca
        self.why_pca = why_pca
        self.next_pca = next_pca

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

        self.records = []

        required_keys = {
            "sample_id",
            "frame_features",
            "what",
            "why",
            "next",
            "what_feature",
            "why_feature",
            "next_feature",
        }

        for path in self.paths:
            record = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )

            missing = (
                required_keys
                - set(
                    record.keys()
                )
            )

            if missing:
                raise KeyError(
                    f"Missing keys {sorted(missing)} "
                    f"in {path}"
                )

            sample_id = str(
                record[
                    "sample_id"
                ]
            )

            self.records.append(
                {
                    "path": path,
                    "sample_id": sample_id,
                }
            )

        if not self.records:
            raise RuntimeError(
                "Structured temporal dataset "
                "is empty."
            )

    def __len__(
        self,
    ):
        return len(
            self.records
        )

    @staticmethod
    def _load_feature(
        record,
        key: str,
        sample_id: str,
    ):
        feature = (
            record[
                key
            ]
            .detach()
            .float()
            .cpu()
            .reshape(-1)
        )

        if feature.shape != (
            960,
        ):
            raise ValueError(
                f"Invalid {key} shape for "
                f"{sample_id}: "
                f"{tuple(feature.shape)}"
            )

        if not torch.isfinite(
            feature
        ).all():
            raise ValueError(
                f"Non-finite {key} for "
                f"{sample_id}"
            )

        return feature

    def __getitem__(
        self,
        index,
    ):
        entry = self.records[
            index
        ]

        record = torch.load(
            entry[
                "path"
            ],
            map_location="cpu",
            weights_only=False,
        )

        sample_id = (
            entry[
                "sample_id"
            ]
        )

        frame_features = (
            record[
                "frame_features"
            ]
            .detach()
            .float()
            .cpu()
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

        what_feature = (
            self._load_feature(
                record,
                "what_feature",
                sample_id,
            )
        )

        why_feature = (
            self._load_feature(
                record,
                "why_feature",
                sample_id,
            )
        )

        next_feature = (
            self._load_feature(
                record,
                "next_feature",
                sample_id,
            )
        )

        what_target = (
            self.what_pca(
                what_feature.unsqueeze(
                    0
                )
            )
            .squeeze(0)
        )

        why_target = (
            self.why_pca(
                why_feature.unsqueeze(
                    0
                )
            )
            .squeeze(0)
        )

        next_target = (
            self.next_pca(
                next_feature.unsqueeze(
                    0
                )
            )
            .squeeze(0)
        )

        for name, target in [
            (
                "WHAT",
                what_target,
            ),
            (
                "WHY",
                why_target,
            ),
            (
                "NEXT",
                next_target,
            ),
        ]:
            if target.shape != (
                256,
            ):
                raise ValueError(
                    f"Invalid projected {name} "
                    f"shape for {sample_id}: "
                    f"{tuple(target.shape)}"
                )

            if not torch.isfinite(
                target
            ).all():
                raise ValueError(
                    f"Non-finite projected "
                    f"{name} target for "
                    f"{sample_id}"
                )

        what_text = str(
            record[
                "what"
            ]
        )

        why_text = str(
            record[
                "why"
            ]
        )

        next_text = str(
            record[
                "next"
            ]
        )

        return {
            "sample_id":
                sample_id,

            "frame_features":
                frame_features,

            "what_target":
                what_target,

            "why_target":
                why_target,

            "next_target":
                next_target,

            "what":
                what_text,

            "why":
                why_text,

            "next":
                next_text,

            "what_normalized":
                normalize_text(
                    what_text
                ),

            "why_normalized":
                normalize_text(
                    why_text
                ),

            "next_normalized":
                normalize_text(
                    next_text
                ),
        }


def collate_batch(
    batch,
):
    return {
        "sample_id": [
            x[
                "sample_id"
            ]
            for x in batch
        ],

        "frame_features":
            torch.stack(
                [
                    x[
                        "frame_features"
                    ]
                    for x in batch
                ],
                dim=0,
            ),

        "what_target":
            torch.stack(
                [
                    x[
                        "what_target"
                    ]
                    for x in batch
                ],
                dim=0,
            ),

        "why_target":
            torch.stack(
                [
                    x[
                        "why_target"
                    ]
                    for x in batch
                ],
                dim=0,
            ),

        "next_target":
            torch.stack(
                [
                    x[
                        "next_target"
                    ]
                    for x in batch
                ],
                dim=0,
            ),

        "what_normalized": [
            x[
                "what_normalized"
            ]
            for x in batch
        ],

        "why_normalized": [
            x[
                "why_normalized"
            ]
            for x in batch
        ],

        "next_normalized": [
            x[
                "next_normalized"
            ]
            for x in batch
        ],
    }


# ============================================================
# Multi-positive objective
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

    return -(
        log_numerator
        - log_denominator
    ).mean()


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


def branch_loss(
    predicted,
    target,
    labels,
    temperature,
    cosine_weight,
):
    contrastive = (
        multi_positive_infonce(
            predicted,
            target,
            labels,
            temperature=temperature,
        )
    )

    cosine = (
        cosine_alignment_loss(
            predicted,
            target,
        )
    )

    total = (
        contrastive
        + cosine_weight
        * cosine
    )

    return (
        total,
        contrastive,
        cosine,
    )


# ============================================================
# WHY retrieval metrics
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
        similarities.shape[
            0
        ]
    )

    top1_hits = 0
    top3_hits = 0

    reciprocal_ranks = []
    margins = []

    for i in range(
        num_samples
    ):
        query_label = (
            labels[
                i
            ]
        )

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

        row = (
            similarities[
                i
            ]
        )

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

        first_rank = (
            int(
                positive_positions[
                    0
                ].item()
            )
            + 1
        )

        reciprocal_ranks.append(
            1.0
            / first_rank
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

    return {
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
            else float(
                "nan"
            ),

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


# ============================================================
# Evaluation
# ============================================================


@torch.inference_mode()
def evaluate(
    model,
    loader,
    device,
):
    """
    Evaluate ONLY the primary WHY branch.

    WHAT and NEXT are auxiliary supervision and are intentionally
    excluded from checkpoint selection.
    """

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

        why_target = (
            batch[
                "why_target"
            ]
            .to(
                device,
                non_blocking=True,
            )
        )

        outputs = model(
            frame_features
        )

        why_pred = outputs[
            "z_int"
        ]

        all_predictions.append(
            why_pred
            .detach()
            .cpu()
        )

        all_targets.append(
            why_target
            .detach()
            .cpu()
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
            type(
                model
            ).__name__,

        "encoder":
            "structured_temporal_attention",

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

            "what_weight":
                args.what_weight,

            "next_weight":
                args.next_weight,

            "seed":
                args.seed,
        },

        "targets": {
            "what_pca":
                str(
                    args.what_pca
                ),

            "why_pca":
                str(
                    args.why_pca
                ),

            "next_pca":
                str(
                    args.next_pca
                ),

            "why_target_source":
                "stage_a_v0_exact",
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
    parser = argparse.ArgumentParser(
        description=(
            "Train structured WHAT-WHY-NEXT "
            "temporal intention prediction."
        )
    )

    parser.add_argument(
        "--temporal-cache-root",
        type=Path,
        default=(
            DEFAULT_TEMPORAL_CACHE_ROOT
        ),
    )

    parser.add_argument(
        "--what-pca",
        type=Path,
        default=(
            DEFAULT_WHAT_PCA_PATH
        ),
    )

    parser.add_argument(
        "--why-pca",
        type=Path,
        default=(
            DEFAULT_WHY_PCA_PATH
        ),
    )

    parser.add_argument(
        "--next-pca",
        type=Path,
        default=(
            DEFAULT_NEXT_PCA_PATH
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            DEFAULT_OUTPUT_ROOT
        ),
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
        "--what-weight",
        type=float,
        default=0.3,
    )

    parser.add_argument(
        "--next-weight",
        type=float,
        default=0.3,
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

    return parser.parse_args()


# ============================================================
# Main
# ============================================================


def main():
    args = parse_args()

    if args.what_weight < 0:
        raise ValueError(
            "--what-weight must be >= 0"
        )

    if args.next_weight < 0:
        raise ValueError(
            "--next-weight must be >= 0"
        )

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
        / "temporal_attention"
        / f"seed_{args.seed}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 72
    )

    print(
        "STRUCTURED WHAT-WHY-NEXT "
        "TEMPORAL INTENTION TRAINING"
    )

    print(
        "=" * 72
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
        "WHAT PCA:",
        args.what_pca,
    )

    print(
        "WHY PCA:",
        args.why_pca,
    )

    print(
        "NEXT PCA:",
        args.next_pca,
    )

    print(
        "WHAT weight:",
        args.what_weight,
    )

    print(
        "NEXT weight:",
        args.next_weight,
    )

    print(
        "output:",
        output_dir,
    )

    print()

    what_pca = PCAProjector(
        args.what_pca
    )

    why_pca = PCAProjector(
        args.why_pca
    )

    next_pca = PCAProjector(
        args.next_pca
    )

    for name, projector in [
        (
            "WHAT",
            what_pca,
        ),
        (
            "WHY",
            why_pca,
        ),
        (
            "NEXT",
            next_pca,
        ),
    ]:
        if projector.input_dim != (
            args.visual_dim
        ):
            raise ValueError(
                f"{name} PCA input "
                f"dimension "
                f"{projector.input_dim} "
                "does not match "
                f"{args.visual_dim}."
            )

        if projector.output_dim != (
            args.intention_dim
        ):
            raise ValueError(
                f"{name} PCA output "
                f"dimension "
                f"{projector.output_dim} "
                "does not match "
                f"{args.intention_dim}."
            )

        if projector.fit_split != (
            "train"
        ):
            raise ValueError(
                f"{name} PCA was not "
                "fit on train split: "
                f"{projector.fit_split}"
            )

    train_dataset = (
        StructuredTemporalIntentionDataset(
            temporal_root=(
                args.temporal_cache_root
            ),
            split="train",
            what_pca=what_pca,
            why_pca=why_pca,
            next_pca=next_pca,
        )
    )

    val_dataset = (
        StructuredTemporalIntentionDataset(
            temporal_root=(
                args.temporal_cache_root
            ),
            split="val",
            what_pca=what_pca,
            why_pca=why_pca,
            next_pca=next_pca,
        )
    )

    print(
        "train samples:",
        len(
            train_dataset
        ),
    )

    print(
        "val samples:",
        len(
            val_dataset
        ),
    )

    train_generator = (
        torch.Generator()
    )

    train_generator.manual_seed(
        args.seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=(
            args.batch_size
        ),
        shuffle=True,
        generator=train_generator,
        num_workers=(
            args.num_workers
        ),
        pin_memory=(
            device.type
            == "cuda"
        ),
        collate_fn=collate_batch,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=(
            args.batch_size
        ),
        shuffle=False,
        num_workers=(
            args.num_workers
        ),
        pin_memory=(
            device.type
            == "cuda"
        ),
        collate_fn=collate_batch,
        drop_last=False,
    )

    model = (
        StructuredTemporalIntentionEncoder(
            visual_dim=(
                args.visual_dim
            ),
            model_dim=(
                args.model_dim
            ),
            intention_dim=(
                args.intention_dim
            ),
            max_frames=(
                args.max_frames
            ),
            dropout=(
                args.dropout
            ),
        )
        .to(
            device
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=(
            args.weight_decay
        ),
    )

    scheduler = (
        torch.optim.lr_scheduler
        .CosineAnnealingLR(
            optimizer,
            T_max=(
                args.epochs
            ),
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

    best_mrr = (
        -math.inf
    )

    best_epoch = -1
    best_metrics = None

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        model.train()

        epoch_total_losses = []
        epoch_why_losses = []
        epoch_what_losses = []
        epoch_next_losses = []

        epoch_why_contrastive = []
        epoch_what_contrastive = []
        epoch_next_contrastive = []

        epoch_why_cosine = []
        epoch_what_cosine = []
        epoch_next_cosine = []

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

            what_target = (
                batch[
                    "what_target"
                ]
                .to(
                    device,
                    non_blocking=True,
                )
            )

            why_target = (
                batch[
                    "why_target"
                ]
                .to(
                    device,
                    non_blocking=True,
                )
            )

            next_target = (
                batch[
                    "next_target"
                ]
                .to(
                    device,
                    non_blocking=True,
                )
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            outputs = model(
                frame_features
            )

            what_pred = outputs[
                "what_pred"
            ]

            why_pred = outputs[
                "why_pred"
            ]

            next_pred = outputs[
                "next_pred"
            ]

            (
                why_loss,
                why_contrastive,
                why_cosine,
            ) = branch_loss(
                predicted=why_pred,
                target=why_target,
                labels=(
                    batch[
                        "why_normalized"
                    ]
                ),
                temperature=(
                    args.temperature
                ),
                cosine_weight=(
                    args.cosine_weight
                ),
            )

            (
                what_loss,
                what_contrastive,
                what_cosine,
            ) = branch_loss(
                predicted=what_pred,
                target=what_target,
                labels=(
                    batch[
                        "what_normalized"
                    ]
                ),
                temperature=(
                    args.temperature
                ),
                cosine_weight=(
                    args.cosine_weight
                ),
            )

            (
                next_loss,
                next_contrastive,
                next_cosine,
            ) = branch_loss(
                predicted=next_pred,
                target=next_target,
                labels=(
                    batch[
                        "next_normalized"
                    ]
                ),
                temperature=(
                    args.temperature
                ),
                cosine_weight=(
                    args.cosine_weight
                ),
            )

            loss = (
                why_loss
                + args.what_weight
                * what_loss
                + args.next_weight
                * next_loss
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

            epoch_total_losses.append(
                float(
                    loss.item()
                )
            )

            epoch_why_losses.append(
                float(
                    why_loss.item()
                )
            )

            epoch_what_losses.append(
                float(
                    what_loss.item()
                )
            )

            epoch_next_losses.append(
                float(
                    next_loss.item()
                )
            )

            epoch_why_contrastive.append(
                float(
                    why_contrastive.item()
                )
            )

            epoch_what_contrastive.append(
                float(
                    what_contrastive.item()
                )
            )

            epoch_next_contrastive.append(
                float(
                    next_contrastive.item()
                )
            )

            epoch_why_cosine.append(
                float(
                    why_cosine.item()
                )
            )

            epoch_what_cosine.append(
                float(
                    what_cosine.item()
                )
            )

            epoch_next_cosine.append(
                float(
                    next_cosine.item()
                )
            )

        scheduler.step()

        val_metrics = evaluate(
            model,
            val_loader,
            device,
        )

        train_total_loss = float(
            np.mean(
                epoch_total_losses
            )
        )

        train_why_loss = float(
            np.mean(
                epoch_why_losses
            )
        )

        train_what_loss = float(
            np.mean(
                epoch_what_losses
            )
        )

        train_next_loss = float(
            np.mean(
                epoch_next_losses
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

            "train_total_loss":
                train_total_loss,

            "train_why_loss":
                train_why_loss,

            "train_what_loss":
                train_what_loss,

            "train_next_loss":
                train_next_loss,

            "train_why_contrastive":
                float(
                    np.mean(
                        epoch_why_contrastive
                    )
                ),

            "train_what_contrastive":
                float(
                    np.mean(
                        epoch_what_contrastive
                    )
                ),

            "train_next_contrastive":
                float(
                    np.mean(
                        epoch_next_contrastive
                    )
                ),

            "train_why_cosine":
                float(
                    np.mean(
                        epoch_why_cosine
                    )
                ),

            "train_what_cosine":
                float(
                    np.mean(
                        epoch_what_cosine
                    )
                ),

            "train_next_cosine":
                float(
                    np.mean(
                        epoch_next_cosine
                    )
                ),

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
            f"loss={train_total_loss:.6f} "
            f"why={train_why_loss:.6f} "
            f"what={train_what_loss:.6f} "
            f"next={train_next_loss:.6f} "
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
                "  -> NEW BEST WHY "
                f"MRR={best_mrr:.6f}",
                flush=True,
            )

    summary = {
        "encoder":
            "structured_temporal_attention",

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
            "visual_dim":
                args.visual_dim,

            "model_dim":
                args.model_dim,

            "intention_dim":
                args.intention_dim,

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

            "what_weight":
                args.what_weight,

            "next_weight":
                args.next_weight,

            "checkpoint_metric":
                "val_why_mrr",
        },

        "targets": {
            "what_pca":
                str(
                    args.what_pca
                ),

            "why_pca":
                str(
                    args.why_pca
                ),

            "next_pca":
                str(
                    args.next_pca
                ),

            "why_target_source":
                "stage_a_v0_exact",

            "next_used_as_input":
                False,
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

    print(
        "=" * 72
    )

    print(
        "TRAINING COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        "best_epoch:",
        best_epoch,
    )

    print(
        "best_why_mrr:",
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