from __future__ import annotations

import argparse
import json
import random
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

        y = self.why[index]

        return {
            "x": x,
            "why": y,
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


def cosine_loss(pred, target):
    return (
        1.0
        - F.cosine_similarity(
            pred,
            target,
            dim=-1,
        )
    ).mean()


@torch.no_grad()
def retrieval_metrics(pred, target):
    pred = F.normalize(
        pred.float(),
        dim=-1,
    )

    target = F.normalize(
        target.float(),
        dim=-1,
    )

    similarity = pred @ target.T

    n = similarity.shape[0]

    correct = torch.arange(
        n,
        device=similarity.device,
    )

    ranked = torch.argsort(
        similarity,
        dim=1,
        descending=True,
    )

    top1 = (
        ranked[:, :1]
        == correct[:, None]
    ).any(dim=1).float().mean()

    top3 = (
        ranked[:, :3]
        == correct[:, None]
    ).any(dim=1).float().mean()

    matches = (
        ranked
        == correct[:, None]
    )

    ranks = (
        matches.float().argmax(dim=1)
        + 1
    )

    mrr = (
        1.0 / ranks.float()
    ).mean()

    positive = similarity[
        torch.arange(n),
        correct,
    ]

    mask = torch.eye(
        n,
        dtype=torch.bool,
        device=similarity.device,
    )

    negatives = similarity.masked_fill(
        mask,
        float("-inf"),
    )

    hardest_negative = negatives.max(
        dim=1
    ).values

    margin = (
        positive
        - hardest_negative
    ).mean()

    return {
        "top1": float(top1.item()),
        "top3": float(top3.item()),
        "mrr": float(mrr.item()),
        "cosine": float(
            positive.mean().item()
        ),
        "margin": float(
            margin.item()
        ),
    }


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
):
    model.eval()

    all_pred = []
    all_target = []

    total_loss = 0.0
    total_count = 0

    for batch in loader:
        x = batch["x"].to(device)
        why = batch["why"].to(device)

        pred = model(x)

        loss = cosine_loss(
            pred,
            why,
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
            why.detach()
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
            "stage_a_cached_v0"
        ),
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
        default=1e-3,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
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

    model = StageAFusion().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history = []

    best_val_cosine = float("-inf")
    best_epoch = -1

    print(
        "=== STAGE-A CACHED TRAINING ==="
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

            why = batch["why"].to(
                device
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            pred = model(x)

            loss = cosine_loss(
                pred,
                why,
            )

            loss.backward()

            optimizer.step()

            batch_size = x.shape[0]

            running_loss += (
                loss.item()
                * batch_size
            )

            running_count += (
                batch_size
            )

        train_step_loss = (
            running_loss
            / running_count
        )

        train_metrics = evaluate(
            model,
            train_eval_loader,
            device,
        )

        val_metrics = evaluate(
            model,
            val_loader,
            device,
        )

        record = {
            "epoch": epoch,
            "train_step_loss": (
                train_step_loss
            ),
            "train": train_metrics,
            "val": val_metrics,
        }

        history.append(record)

        print(
            f"epoch {epoch:03d} | "
            f"train loss "
            f"{train_metrics['loss']:.4f} | "
            f"train cos "
            f"{train_metrics['cosine']:.4f} | "
            f"val loss "
            f"{val_metrics['loss']:.4f} | "
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
            val_metrics["cosine"]
            > best_val_cosine
        ):
            best_val_cosine = (
                val_metrics["cosine"]
            )

            best_epoch = epoch

            checkpoint = {
                "epoch": epoch,
                "model": (
                    model.state_dict()
                ),
                "optimizer": (
                    optimizer.state_dict()
                ),
                "val_metrics": (
                    val_metrics
                ),
                "train_metrics": (
                    train_metrics
                ),
                "config": {
                    "lr": args.lr,
                    "batch_size": (
                        args.batch_size
                    ),
                    "weight_decay": (
                        args.weight_decay
                    ),
                    "seed": args.seed,
                },
            }

            torch.save(
                checkpoint,
                args.output_dir
                / "best.pt",
            )

    history_path = (
        args.output_dir
        / "history.json"
    )

    with history_path.open(
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
        "best_val_cosine": (
            best_val_cosine
        ),
        "train_samples": (
            len(train_dataset)
        ),
        "val_samples": (
            len(val_dataset)
        ),
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
        "=== TRAINING COMPLETE ==="
    )
    print(
        f"best epoch: {best_epoch}"
    )
    print(
        f"best val cosine: "
        f"{best_val_cosine:.4f}"
    )
    print(
        f"checkpoint: "
        f"{args.output_dir / 'best.pt'}"
    )


if __name__ == "__main__":
    main()