from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.train_stage_a_contrastive import (
    CachedStageADataset,
    StageAFusion,
    cosine_alignment_loss,
    multi_positive_infonce,
    retrieval_metrics,
)


VALID_MODES = {
    "visual",
    "task",
    "history",
    "visual_task",
    "visual_history",
    "task_history",
    "all",
}


def mask_inputs(x, mode):
    """
    x = [visual(960), task(960), history(960)]

    We keep the same 2880-D architecture for every ablation
    and zero out unavailable modalities. This keeps model
    capacity directly comparable across runs.
    """

    visual = x[:, 0:960]
    task = x[:, 960:1920]
    history = x[:, 1920:2880]

    zero_visual = torch.zeros_like(visual)
    zero_task = torch.zeros_like(task)
    zero_history = torch.zeros_like(history)

    if mode == "visual":
        parts = [
            visual,
            zero_task,
            zero_history,
        ]

    elif mode == "task":
        parts = [
            zero_visual,
            task,
            zero_history,
        ]

    elif mode == "history":
        parts = [
            zero_visual,
            zero_task,
            history,
        ]

    elif mode == "visual_task":
        parts = [
            visual,
            task,
            zero_history,
        ]

    elif mode == "visual_history":
        parts = [
            visual,
            zero_task,
            history,
        ]

    elif mode == "task_history":
        parts = [
            zero_visual,
            task,
            history,
        ]

    elif mode == "all":
        parts = [
            visual,
            task,
            history,
        ]

    else:
        raise ValueError(
            f"Unknown mode: {mode}"
        )

    return torch.cat(
        parts,
        dim=-1,
    )


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    mode,
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
        x = batch["x"].to(device)

        x = mask_inputs(
            x,
            mode,
        )

        target = (
            batch["why_feature"]
            .to(device)
        )

        why_texts = batch[
            "why_text"
        ]

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
        "--mode",
        required=True,
        choices=sorted(
            VALID_MODES
        ),
    )

    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(
            "/media/dhqg/d1/datasets/"
            "egointent/cache/stage_a_v0"
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/media/dhqg/d1/vlia_outputs/"
            "stage_a_ablation_v1"
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=35,
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

    output_dir = (
        args.output_root
        / args.mode
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    train_dataset = (
        CachedStageADataset(
            args.cache_dir
            / "train.pt"
        )
    )

    val_dataset = (
        CachedStageADataset(
            args.cache_dir
            / "val.pt"
        )
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

    best_val_mrr = float(
        "-inf"
    )

    best_epoch = -1
    best_val_metrics = None

    history = []

    print(
        "=== STAGE-A INPUT ABLATION ==="
    )
    print(
        f"mode: {args.mode}"
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
    print()

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        model.train()

        for batch in train_loader:
            x = (
                batch["x"]
                .to(device)
            )

            x = mask_inputs(
                x,
                args.mode,
            )

            target = (
                batch[
                    "why_feature"
                ]
                .to(device)
            )

            why_texts = batch[
                "why_text"
            ]

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

        train_metrics = evaluate(
            model=model,
            loader=train_eval_loader,
            device=device,
            mode=args.mode,
            temperature=(
                args.temperature
            ),
            lambda_cos=(
                args.lambda_cos
            ),
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            mode=args.mode,
            temperature=(
                args.temperature
            ),
            lambda_cos=(
                args.lambda_cos
            ),
        )

        history.append(
            {
                "epoch": epoch,
                "train": (
                    train_metrics
                ),
                "val": (
                    val_metrics
                ),
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

            best_val_metrics = (
                val_metrics
            )

            torch.save(
                {
                    "epoch": epoch,
                    "mode": (
                        args.mode
                    ),
                    "model": (
                        model.state_dict()
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
                output_dir
                / "best.pt",
            )

    summary = {
        "mode": args.mode,
        "best_epoch": (
            best_epoch
        ),
        "best_val_mrr": (
            best_val_mrr
        ),
        "best_val_metrics": (
            best_val_metrics
        ),
    }

    with (
        output_dir
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

    with (
        output_dir
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
        "=== ABLATION COMPLETE ==="
    )
    print(
        f"mode: {args.mode}"
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
        "best val metrics:",
        best_val_metrics,
    )


if __name__ == "__main__":
    main()