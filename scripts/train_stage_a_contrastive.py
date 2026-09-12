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


class CachedStageADataset(Dataset):
    def __init__(self, path: Path):
        data = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        self.sample_ids = data["sample_ids"]
        self.visual = data["visual"].float()
        self.task = data["task"].float()
        self.history = data["history"].float()
        self.why = data["why"].float()
        self.whys = data["whys"]

        n = len(self.sample_ids)

        assert self.visual.shape == (n, 960)
        assert self.task.shape == (n, 960)
        assert self.history.shape == (n, 960)
        assert self.why.shape == (n, 960)

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        x = torch.cat(
            [
                self.visual[index],
                self.task[index],
                self.history[index],
            ],
            dim=-1,
        )

        return {
            "x": x,
            "why_feature": self.why[index],
            "why_text": self.whys[index],
            "sample_id": self.sample_ids[index],
        }


class StageAFusion(nn.Module):
    def __init__(self):
        super().__init__()

        self.fusion = nn.Sequential(
            nn.Linear(2880, 1024),
            nn.GELU(),
            nn.Linear(1024, 960),
            nn.LayerNorm(960),
        )

    def forward(self, x):
        return self.fusion(x)


def normalize_why(text: str) -> str:
    return " ".join(
        text.strip().lower().split()
    )


def build_positive_mask(
    why_texts,
    device,
):
    normalized = [
        normalize_why(x)
        for x in why_texts
    ]

    n = len(normalized)

    mask = torch.zeros(
        n,
        n,
        dtype=torch.bool,
        device=device,
    )

    groups = defaultdict(list)

    for i, text in enumerate(normalized):
        groups[text].append(i)

    for indices in groups.values():
        idx = torch.tensor(
            indices,
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
    why_texts,
    temperature=0.07,
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

    positive_mask = (
        build_positive_mask(
            why_texts,
            logits.device,
        )
    )

    log_prob = (
        logits
        - torch.logsumexp(
            logits,
            dim=1,
            keepdim=True,
        )
    )

    positive_log_prob = (
        torch.logsumexp(
            log_prob.masked_fill(
                ~positive_mask,
                float("-inf"),
            ),
            dim=1,
        )
    )

    loss = (
        -positive_log_prob.mean()
    )

    return loss


def cosine_alignment_loss(
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

    positive_mask = (
        build_positive_mask(
            why_texts,
            sim.device,
        )
    )

    ranked = torch.argsort(
        sim,
        dim=1,
        descending=True,
    )

    ranked_positive = torch.gather(
        positive_mask,
        dim=1,
        index=ranked,
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

    positive_similarity = (
        sim.masked_fill(
            ~positive_mask,
            float("-inf"),
        )
        .max(dim=1)
        .values
    )

    negative_similarity = (
        sim.masked_fill(
            positive_mask,
            float("-inf"),
        )
        .max(dim=1)
        .values
    )

    margin = (
        positive_similarity
        - negative_similarity
    ).mean()

    diagonal_cosine = (
        F.cosine_similarity(
            pred,
            target,
            dim=-1,
        )
        .mean()
    )

    return {
        "top1": float(
            top1.item()
        ),
        "top3": float(
            top3.item()
        ),
        "mrr": float(
            mrr.item()
        ),
        "margin": float(
            margin.item()
        ),
        "cosine": float(
            diagonal_cosine.item()
        ),
    }


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    temperature,
    lambda_cos,
):
    model.eval()

    all_pred = []
    all_target = []
    all_why_text = []

    total_loss = 0.0
    total_count = 0

    for batch in loader:
        x = batch["x"].to(
            device
        )

        target = (
            batch["why_feature"]
            .to(device)
        )

        why_texts = (
            batch["why_text"]
        )

        pred = model(x)

        contrastive = (
            multi_positive_infonce(
                pred,
                target,
                why_texts,
                temperature=temperature,
            )
        )

        cosine = (
            cosine_alignment_loss(
                pred,
                target,
            )
        )

        loss = (
            contrastive
            + lambda_cos * cosine
        )

        batch_size = x.shape[0]

        total_loss += (
            loss.item()
            * batch_size
        )

        total_count += batch_size

        all_pred.append(
            pred.detach()
        )

        all_target.append(
            target.detach()
        )

        all_why_text.extend(
            list(why_texts)
        )

    pred = torch.cat(
        all_pred,
        dim=0,
    )

    target = torch.cat(
        all_target,
        dim=0,
    )

    metrics = retrieval_metrics(
        pred,
        target,
        all_why_text,
    )

    metrics["loss"] = (
        total_loss
        / total_count
    )

    return metrics


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
            "stage_a_contrastive_v1"
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

    set_seed(args.seed)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    train_dataset = CachedStageADataset(
        args.cache_dir / "train.pt"
    )

    val_dataset = CachedStageADataset(
        args.cache_dir / "val.pt"
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

    model = StageAFusion().to(
        device
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_mrr = float("-inf")
    best_epoch = -1

    history = []

    print(
        "=== STAGE-A CONTRASTIVE V1 ==="
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
        f"epochs: {args.epochs}"
    )

    print(
        f"batch size: "
        f"{args.batch_size}"
    )

    print(
        f"lr: {args.lr}"
    )

    print(
        f"temperature: "
        f"{args.temperature}"
    )

    print(
        f"lambda_cos: "
        f"{args.lambda_cos}"
    )

    print()

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        model.train()

        running_loss = 0.0
        running_count = 0

        for batch in train_loader:
            x = batch["x"].to(
                device
            )

            target = (
                batch["why_feature"]
                .to(device)
            )

            why_texts = (
                batch["why_text"]
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            pred = model(x)

            contrastive = (
                multi_positive_infonce(
                    pred,
                    target,
                    why_texts,
                    temperature=(
                        args.temperature
                    ),
                )
            )

            cosine = (
                cosine_alignment_loss(
                    pred,
                    target,
                )
            )

            loss = (
                contrastive
                + args.lambda_cos
                * cosine
            )

            loss.backward()

            optimizer.step()

            batch_size = (
                x.shape[0]
            )

            running_loss += (
                loss.item()
                * batch_size
            )

            running_count += (
                batch_size
            )

        train_metrics = evaluate(
            model,
            train_eval_loader,
            device,
            temperature=(
                args.temperature
            ),
            lambda_cos=(
                args.lambda_cos
            ),
        )

        val_metrics = evaluate(
            model,
            val_loader,
            device,
            temperature=(
                args.temperature
            ),
            lambda_cos=(
                args.lambda_cos
            ),
        )

        record = {
            "epoch": epoch,
            "train": (
                train_metrics
            ),
            "val": (
                val_metrics
            ),
        }

        history.append(
            record
        )

        print(
            f"epoch {epoch:03d} | "
            f"train cos "
            f"{train_metrics['cosine']:.4f} | "
            f"train top1 "
            f"{train_metrics['top1']:.4f} | "
            f"train mrr "
            f"{train_metrics['mrr']:.4f} | "
            f"val cos "
            f"{val_metrics['cosine']:.4f} | "
            f"val top1 "
            f"{val_metrics['top1']:.4f} | "
            f"top3 "
            f"{val_metrics['top3']:.4f} | "
            f"mrr "
            f"{val_metrics['mrr']:.4f} | "
            f"margin "
            f"{val_metrics['margin']:.4f}"
        )

        if (
            val_metrics["mrr"]
            > best_val_mrr
        ):
            best_val_mrr = (
                val_metrics["mrr"]
            )

            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "model": (
                        model.state_dict()
                    ),
                    "optimizer": (
                        optimizer.state_dict()
                    ),
                    "train_metrics": (
                        train_metrics
                    ),
                    "val_metrics": (
                        val_metrics
                    ),
                    "config": {
                        "lr": args.lr,
                        "batch_size": (
                            args.batch_size
                        ),
                        "weight_decay": (
                            args.weight_decay
                        ),
                        "temperature": (
                            args.temperature
                        ),
                        "lambda_cos": (
                            args.lambda_cos
                        ),
                        "seed": (
                            args.seed
                        ),
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
        "best_epoch": (
            best_epoch
        ),
        "best_val_mrr": (
            best_val_mrr
        ),
        "train_samples": (
            len(train_dataset)
        ),
        "val_samples": (
            len(val_dataset)
        ),
        "temperature": (
            args.temperature
        ),
        "lambda_cos": (
            args.lambda_cos
        ),
        "seed": (
            args.seed
        ),
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
        "=== TRAINING COMPLETE ==="
    )

    print(
        f"best epoch: "
        f"{best_epoch}"
    )

    print(
        f"best val MRR: "
        f"{best_val_mrr:.6f}"
    )

    print(
        f"checkpoint: "
        f"{args.output_dir / 'best.pt'}"
    )


if __name__ == "__main__":
    main()