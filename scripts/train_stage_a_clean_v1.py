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
from torch.utils.data import DataLoader, Dataset


def normalize_why(text: str) -> str:
    return " ".join(text.strip().lower().split())


class CachedDataset(Dataset):
    def __init__(self, path: Path):
        data = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        self.visual = data["visual"].float()
        self.why = data["why"].float()
        self.whys = data["whys"]
        self.sample_ids = data["sample_ids"]

        n = len(self.sample_ids)

        assert self.visual.shape == (n, 960)
        assert self.why.shape == (n, 960)

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        return {
            "visual": self.visual[index],
            "why_960": self.why[index],
            "why_text": self.whys[index],
            "sample_id": self.sample_ids[index],
        }


class CleanIntentionEncoderV1(nn.Module):
    """
    Clean Stage-A predictor.

    Predictor input:
        visual only

    Internal alignment representation:
        960-D

    Final VLIA intention representation:
        256-D
    """

    def __init__(self):
        super().__init__()

        self.fusion = nn.Sequential(
            nn.Linear(2880, 1024),
            nn.GELU(),
            nn.Linear(1024, 960),
            nn.LayerNorm(960),
        )

        self.intention_projection = nn.Sequential(
            nn.Linear(960, 256),
            nn.LayerNorm(256),
        )

    def forward(self, visual):
        zero = torch.zeros_like(visual)

        x = torch.cat(
            [
                visual,
                zero,
                zero,
            ],
            dim=-1,
        )

        z_align = self.fusion(x)

        z_int = self.intention_projection(
            z_align
        )

        return z_int


def fit_pca(train_why, output_dim=256):
    """
    PCA is fitted only on training WHY embeddings.
    Validation WHY never influences the PCA basis.
    """

    x = train_why.float()

    mean = x.mean(
        dim=0,
        keepdim=True,
    )

    centered = x - mean

    # Full SVD is fine for 556 x 960.
    _, _, vh = torch.linalg.svd(
        centered,
        full_matrices=False,
    )

    components = vh[:output_dim].T.contiguous()
    # [960, 256]

    return mean, components


def pca_transform(
    x,
    mean,
    components,
):
    z = (
        x.float()
        - mean
    ) @ components

    return z


def build_positive_mask(
    why_texts,
    device,
):
    normalized = [
        normalize_why(x)
        for x in why_texts
    ]

    groups = defaultdict(list)

    for i, text in enumerate(
        normalized
    ):
        groups[text].append(i)

    n = len(normalized)

    mask = torch.zeros(
        n,
        n,
        dtype=torch.bool,
        device=device,
    )

    for indices in groups.values():
        idx = torch.tensor(
            indices,
            device=device,
            dtype=torch.long,
        )

        mask[
            idx[:, None],
            idx[None, :],
        ] = True

    return mask


def multi_positive_infonce(
    pred,
    target,
    why_texts,
    temperature,
):
    pred = F.normalize(
        pred.float(),
        dim=-1,
    )

    target = F.normalize(
        target.float(),
        dim=-1,
    )

    logits = (
        pred @ target.T
    ) / temperature

    positive_mask = build_positive_mask(
        why_texts,
        logits.device,
    )

    log_prob = (
        logits
        - torch.logsumexp(
            logits,
            dim=1,
            keepdim=True,
        )
    )

    positive_log_prob = torch.logsumexp(
        log_prob.masked_fill(
            ~positive_mask,
            float("-inf"),
        ),
        dim=1,
    )

    return (
        -positive_log_prob.mean()
    )


def cosine_loss(
    pred,
    target,
):
    return (
        1.0
        - F.cosine_similarity(
            pred,
            target,
            dim=-1,
        )
    ).mean()


@torch.no_grad()
def retrieval_metrics(
    pred,
    target,
    why_texts,
):
    pred = F.normalize(
        pred.float(),
        dim=-1,
    )

    target = F.normalize(
        target.float(),
        dim=-1,
    )

    sim = pred @ target.T

    positive_mask = build_positive_mask(
        why_texts,
        sim.device,
    )

    ranked = torch.argsort(
        sim,
        dim=1,
        descending=True,
    )

    ranked_positive = torch.gather(
        positive_mask,
        1,
        ranked,
    )

    top1 = (
        ranked_positive[:, :1]
        .any(dim=1)
        .float()
        .mean()
    )

    top3 = (
        ranked_positive[:, :3]
        .any(dim=1)
        .float()
        .mean()
    )

    first_positive_rank = (
        ranked_positive.float()
        .argmax(dim=1)
        + 1
    )

    mrr = (
        1.0
        / first_positive_rank.float()
    ).mean()

    positive_sim = (
        sim.masked_fill(
            ~positive_mask,
            float("-inf"),
        )
        .max(dim=1)
        .values
    )

    negative_sim = (
        sim.masked_fill(
            positive_mask,
            float("-inf"),
        )
        .max(dim=1)
        .values
    )

    margin = (
        positive_sim
        - negative_sim
    ).mean()

    diagonal_cosine = (
        F.cosine_similarity(
            pred,
            target,
            dim=-1,
        ).mean()
    )

    return {
        "top1": top1.item(),
        "top3": top3.item(),
        "mrr": mrr.item(),
        "margin": margin.item(),
        "cosine": diagonal_cosine.item(),
    }


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    pca_mean,
    pca_components,
):
    model.eval()

    predictions = []
    targets = []
    texts = []

    for batch in loader:
        visual = batch[
            "visual"
        ].to(device)

        why_960 = batch[
            "why_960"
        ].to(device)

        target_256 = pca_transform(
            why_960,
            pca_mean,
            pca_components,
        )

        pred_256 = model(
            visual
        )

        predictions.append(
            pred_256
        )

        targets.append(
            target_256
        )

        texts.extend(
            list(
                batch["why_text"]
            )
        )

    predictions = torch.cat(
        predictions,
        dim=0,
    )

    targets = torch.cat(
        targets,
        dim=0,
    )

    return retrieval_metrics(
        predictions,
        targets,
        texts,
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(
            "/media/dhqg/d1/datasets/"
            "egointent/cache/stage_a_v0"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/media/dhqg/d1/vlia_outputs/"
            "stage_a_clean_v1"
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
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
        "--lambda-cos",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


def main():
    args = parse_args()

    set_seed(
        args.seed
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    train_dataset = CachedDataset(
        args.cache_dir
        / "train.pt"
    )

    val_dataset = CachedDataset(
        args.cache_dir
        / "val.pt"
    )

    print(
        "Fitting train-only "
        "WHY PCA 960 -> 256..."
    )

    pca_mean_cpu, pca_components_cpu = (
        fit_pca(
            train_dataset.why,
            output_dim=256,
        )
    )

    torch.save(
        {
            "mean": pca_mean_cpu,
            "components": (
                pca_components_cpu
            ),
            "input_dim": 960,
            "output_dim": 256,
            "fit_split": "train",
        },
        args.output_dir
        / "why_pca_960_to_256.pt",
    )

    pca_mean = (
        pca_mean_cpu
        .to(device)
    )

    pca_components = (
        pca_components_cpu
        .to(device)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = (
        CleanIntentionEncoderV1()
        .to(device)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_mrr = float("-inf")
    best_epoch = -1
    best_val = None

    history = []

    print()
    print(
        "=== CLEAN STAGE-A "
        "INTENTION V1 ==="
    )

    print(
        f"device: {device}"
    )

    print(
        f"train samples: "
        f"{len(train_dataset)}"
    )

    print(
        f"val samples: "
        f"{len(val_dataset)}"
    )

    print(
        "predictor input: visual only"
    )

    print(
        "target space: "
        "train-PCA WHY 256-D"
    )

    print()

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        model.train()

        for batch in train_loader:
            visual = (
                batch["visual"]
                .to(device)
            )

            why_960 = (
                batch["why_960"]
                .to(device)
            )

            why_texts = (
                batch["why_text"]
            )

            with torch.no_grad():
                target_256 = (
                    pca_transform(
                        why_960,
                        pca_mean,
                        pca_components,
                    )
                )

            optimizer.zero_grad(
                set_to_none=True
            )

            pred_256 = model(
                visual
            )

            contrastive = (
                multi_positive_infonce(
                    pred_256,
                    target_256,
                    why_texts,
                    temperature=(
                        args.temperature
                    ),
                )
            )

            alignment = cosine_loss(
                pred_256,
                target_256,
            )

            loss = (
                contrastive
                + args.lambda_cos
                * alignment
            )

            loss.backward()

            optimizer.step()

        train_metrics = evaluate(
            model,
            train_eval_loader,
            device,
            pca_mean,
            pca_components,
        )

        val_metrics = evaluate(
            model,
            val_loader,
            device,
            pca_mean,
            pca_components,
        )

        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
            }
        )

        print(
            f"epoch {epoch:03d} | "
            f"train top1 "
            f"{train_metrics['top1']:.4f} | "
            f"train mrr "
            f"{train_metrics['mrr']:.4f} | "
            f"val top1 "
            f"{val_metrics['top1']:.4f} | "
            f"top3 "
            f"{val_metrics['top3']:.4f} | "
            f"mrr "
            f"{val_metrics['mrr']:.4f} | "
            f"margin "
            f"{val_metrics['margin']:.4f} | "
            f"cos "
            f"{val_metrics['cosine']:.4f}"
        )

        if (
            val_metrics["mrr"]
            > best_mrr
        ):
            best_mrr = (
                val_metrics["mrr"]
            )

            best_epoch = epoch
            best_val = val_metrics

            torch.save(
                {
                    "epoch": epoch,
                    "model": (
                        model.state_dict()
                    ),
                    "val_metrics": (
                        val_metrics
                    ),
                    "train_metrics": (
                        train_metrics
                    ),
                    "config": {
                        "input": "visual_only",
                        "intention_dim": 256,
                        "target": (
                            "train_pca_why"
                        ),
                        "lr": args.lr,
                        "batch_size": (
                            args.batch_size
                        ),
                        "temperature": (
                            args.temperature
                        ),
                        "lambda_cos": (
                            args.lambda_cos
                        ),
                        "seed": args.seed,
                    },
                },
                args.output_dir
                / "best.pt",
            )

    with (
        args.output_dir
        / "history.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            history,
            f,
            indent=2,
        )

    summary = {
        "best_epoch": best_epoch,
        "best_val_mrr": best_mrr,
        "best_val_metrics": best_val,
        "input": "visual_only",
        "intention_dim": 256,
        "target": "train_pca_why",
        "seed": args.seed,
    }

    with (
        args.output_dir
        / "summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print(
        "=== CLEAN TRAINING COMPLETE ==="
    )

    print(
        f"best epoch: {best_epoch}"
    )

    print(
        f"best val MRR: "
        f"{best_mrr:.6f}"
    )

    print(
        "best val metrics:",
        best_val,
    )

    print(
        "checkpoint:",
        args.output_dir
        / "best.pt",
    )

    print(
        "PCA:",
        args.output_dir
        / "why_pca_960_to_256.pt",
    )


if __name__ == "__main__":
    main()