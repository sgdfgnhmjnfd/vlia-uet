import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset


DEFAULT_CACHE_ROOT = Path(
    "/media/dhqg/d1/datasets/egointent/cache/stage_a_v0"
)

DEFAULT_FUSION_CHECKPOINT = Path(
    "/media/dhqg/d1/vlia_outputs/"
    "stage_a_v1_contrastive/visual/best_fusion.pt"
)

DEFAULT_OUTPUT_DIR = Path(
    "/media/dhqg/d1/vlia_outputs/"
    "stage_a_v1_projection_256"
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cache-root",
        type=Path,
        default=DEFAULT_CACHE_ROOT,
    )

    parser.add_argument(
        "--fusion-checkpoint",
        type=Path,
        default=DEFAULT_FUSION_CHECKPOINT,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
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
        "--temperature",
        type=float,
        default=0.07,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CachedProjectionDataset(Dataset):
    def __init__(
        self,
        cache_dir,
        pca_mean,
        pca_components,
    ):
        self.files = sorted(
            Path(cache_dir).glob("*.pt")
        )

        if not self.files:
            raise FileNotFoundError(
                f"No cache files found in {cache_dir}"
            )

        self.pca_mean = pca_mean
        self.pca_components = pca_components

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        record = torch.load(
            self.files[index],
            map_location="cpu",
            weights_only=False,
        )

        visual = record[
            "visual_feature"
        ].float()

        why_960 = record[
            "why_feature"
        ].float()

        why_256 = (
            why_960 - self.pca_mean
        ) @ self.pca_components

        return {
            "visual": visual,
            "why_256": why_256.float(),
            "why": record["why"],
            "sample_id": record["sample_id"],
        }


class StageAFusion(nn.Module):
    def __init__(
        self,
        input_dim=960,
        hidden_dim=1024,
        output_dim=960,
    ):
        super().__init__()

        self.fusion = nn.Sequential(
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
        return self.fusion(x)


class IntentionProjection(nn.Module):
    """
    Same architecture as
    IntentionEncoderV0.intention_projection.
    """

    def __init__(
        self,
        input_dim=960,
        intention_dim=256,
    ):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(
                input_dim,
                intention_dim,
            ),
            nn.LayerNorm(
                intention_dim,
            ),
        )

    def forward(self, x):
        return self.projection(x)


def load_why_features(cache_dir):
    features = []

    for path in sorted(
        Path(cache_dir).glob("*.pt")
    ):
        record = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        features.append(
            record["why_feature"].float()
        )

    if not features:
        raise RuntimeError(
            f"No WHY features in {cache_dir}"
        )

    return torch.stack(
        features,
        dim=0,
    )


def fit_pca(
    train_features,
    output_dim=256,
):
    mean = train_features.mean(
        dim=0
    )

    centered = (
        train_features
        - mean.unsqueeze(0)
    )

    _, _, vh = torch.linalg.svd(
        centered,
        full_matrices=False,
    )

    components = (
        vh[:output_dim]
        .T
        .contiguous()
    )

    return (
        mean.float(),
        components.float(),
    )


def build_positive_mask(
    why_texts,
    device,
):
    n = len(why_texts)

    mask = torch.zeros(
        n,
        n,
        dtype=torch.bool,
        device=device,
    )

    for i in range(n):
        for j in range(n):
            if why_texts[i] == why_texts[j]:
                mask[i, j] = True

    return mask


def multi_positive_contrastive_loss(
    pred,
    target,
    why_texts,
    temperature,
):
    pred = F.normalize(
        pred,
        dim=-1,
    )

    target = F.normalize(
        target,
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

    positive_counts = (
        positive_mask.sum(dim=1)
        .clamp(min=1)
    )

    positive_log_prob = (
        log_prob
        * positive_mask.float()
    ).sum(dim=1) / positive_counts

    return -positive_log_prob.mean()


@torch.no_grad()
def evaluate(
    fusion,
    projection,
    loader,
    device,
):
    fusion.eval()
    projection.eval()

    all_pred = []
    all_target = []
    all_why = []

    for batch in loader:
        visual = batch[
            "visual"
        ].to(device)

        target = batch[
            "why_256"
        ].to(device)

        z_align = fusion(
            visual
        )

        z_int = projection(
            z_align
        )

        all_pred.append(
            z_int.cpu()
        )

        all_target.append(
            target.cpu()
        )

        all_why.extend(
            batch["why"]
        )

    pred = torch.cat(
        all_pred,
        dim=0,
    )

    target = torch.cat(
        all_target,
        dim=0,
    )

    pred = F.normalize(
        pred.float(),
        dim=-1,
    )

    target = F.normalize(
        target.float(),
        dim=-1,
    )

    positive_cosine = (
        pred * target
    ).sum(dim=-1)

    similarity = (
        pred @ target.T
    )

    n = similarity.shape[0]

    top1_hits = 0
    top3_hits = 0
    reciprocal_ranks = []
    margins = []

    for i in range(n):
        query_label = all_why[i]

        scores = similarity[i]

        ranking = torch.argsort(
            scores,
            descending=True,
        )

        ranked_labels = [
            all_why[j]
            for j in ranking.tolist()
        ]

        if ranked_labels[0] == query_label:
            top1_hits += 1

        if query_label in ranked_labels[:3]:
            top3_hits += 1

        rank_found = None

        for rank, label in enumerate(
            ranked_labels,
            start=1,
        ):
            if label == query_label:
                rank_found = rank
                break

        if rank_found is not None:
            reciprocal_ranks.append(
                1.0 / rank_found
            )

        positive_indices = [
            j
            for j in range(n)
            if all_why[j] == query_label
        ]

        negative_indices = [
            j
            for j in range(n)
            if all_why[j] != query_label
        ]

        best_positive = scores[
            positive_indices
        ].max().item()

        if negative_indices:
            hardest_negative = scores[
                negative_indices
            ].max().item()

            margins.append(
                best_positive
                - hardest_negative
            )

    return {
        "cosine_mean": (
            positive_cosine.mean().item()
        ),
        "cosine_median": (
            positive_cosine.median().item()
        ),
        "top1": (
            top1_hits / n
        ),
        "top3": (
            top3_hits / n
        ),
        "mrr": float(
            np.mean(reciprocal_ranks)
        ),
        "margin_mean": float(
            np.mean(margins)
        ),
        "margin_median": float(
            np.median(margins)
        ),
        "unique_why": len(
            set(all_why)
        ),
    }


def train_one_epoch(
    fusion,
    projection,
    loader,
    optimizer,
    device,
    temperature,
):
    fusion.eval()
    projection.train()

    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        visual = batch[
            "visual"
        ].to(device)

        target = batch[
            "why_256"
        ].to(device)

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.no_grad():
            z_align = fusion(
                visual
            )

        z_int = projection(
            z_align
        )

        loss = (
            multi_positive_contrastive_loss(
                pred=z_int,
                target=target,
                why_texts=batch["why"],
                temperature=temperature,
            )
        )

        loss.backward()
        optimizer.step()

        batch_size = (
            visual.shape[0]
        )

        total_loss += (
            loss.item()
            * batch_size
        )

        total_samples += (
            batch_size
        )

    return (
        total_loss
        / total_samples
    )


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

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("device:", device)
    print(
        "fusion checkpoint:",
        args.fusion_checkpoint,
    )

    # --------------------------------------------------
    # Fit PCA using TRAIN WHY embeddings only.
    # --------------------------------------------------

    train_why = load_why_features(
        args.cache_root / "train"
    )

    pca_mean, pca_components = fit_pca(
        train_why,
        output_dim=256,
    )

    print(
        "PCA mean:",
        tuple(pca_mean.shape),
    )

    print(
        "PCA components:",
        tuple(pca_components.shape),
    )

    # Persist the exact semantic bridge.
    pca_path = (
        args.output_dir
        / "why_pca_256.pt"
    )

    torch.save(
        {
            "mean": pca_mean,
            "components": (
                pca_components
            ),
            "fit_split": "train",
            "input_dim": 960,
            "output_dim": 256,
        },
        pca_path,
    )

    train_dataset = (
        CachedProjectionDataset(
            args.cache_root / "train",
            pca_mean=pca_mean,
            pca_components=(
                pca_components
            ),
        )
    )

    val_dataset = (
        CachedProjectionDataset(
            args.cache_root / "val",
            pca_mean=pca_mean,
            pca_components=(
                pca_components
            ),
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

    # --------------------------------------------------
    # Load the frozen visual-only Stage-A fusion.
    # --------------------------------------------------

    fusion = StageAFusion(
        input_dim=960,
    ).to(device)

    checkpoint = torch.load(
        args.fusion_checkpoint,
        map_location=device,
        weights_only=False,
    )

    fusion.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    fusion.eval()

    for param in fusion.parameters():
        param.requires_grad = False

    # --------------------------------------------------
    # Train ONLY 960 -> 256 intention projection.
    # --------------------------------------------------

    projection = (
        IntentionProjection()
        .to(device)
    )

    optimizer = torch.optim.AdamW(
        projection.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    trainable_params = sum(
        p.numel()
        for p in projection.parameters()
        if p.requires_grad
    )

    print(
        "train samples:",
        len(train_dataset),
    )

    print(
        "val samples:",
        len(val_dataset),
    )

    print(
        "projection trainable parameters:",
        trainable_params,
    )

    print(
        "temperature:",
        args.temperature,
    )

    print()

    initial_train = evaluate(
        fusion,
        projection,
        train_eval_loader,
        device,
    )

    initial_val = evaluate(
        fusion,
        projection,
        val_loader,
        device,
    )

    print(
        "initial train:",
        initial_train,
    )

    print(
        "initial val:",
        initial_val,
    )

    print()

    best_mrr = float(
        "-inf"
    )

    best_epoch = 0
    history = []

    checkpoint_path = (
        args.output_dir
        / "best_projection.pt"
    )

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        train_loss = train_one_epoch(
            fusion=fusion,
            projection=projection,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            temperature=(
                args.temperature
            ),
        )

        val_metrics = evaluate(
            fusion,
            projection,
            val_loader,
            device,
        )

        record = {
            "epoch": epoch,
            "train_loss": (
                train_loss
            ),
            **{
                f"val_{k}": v
                for k, v
                in val_metrics.items()
            },
        }

        history.append(
            record
        )

        if (
            val_metrics["mrr"]
            > best_mrr
        ):
            best_mrr = (
                val_metrics["mrr"]
            )

            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "projection_state_dict": (
                        projection.state_dict()
                    ),
                    "metrics": record,
                    "fusion_checkpoint": str(
                        args.fusion_checkpoint
                    ),
                    "pca_path": str(
                        pca_path
                    ),
                    "intention_dim": 256,
                },
                checkpoint_path,
            )

        if (
            epoch == 1
            or epoch % 5 == 0
            or epoch == args.epochs
        ):
            print(
                f"epoch {epoch:03d} | "
                f"loss {train_loss:.4f} | "
                f"val cos "
                f"{val_metrics['cosine_mean']:.4f} | "
                f"top1 "
                f"{val_metrics['top1']:.4f} | "
                f"top3 "
                f"{val_metrics['top3']:.4f} | "
                f"MRR "
                f"{val_metrics['mrr']:.4f} | "
                f"margin "
                f"{val_metrics['margin_mean']:.4f}"
            )

    best_checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    projection.load_state_dict(
        best_checkpoint[
            "projection_state_dict"
        ]
    )

    final_train = evaluate(
        fusion,
        projection,
        train_eval_loader,
        device,
    )

    final_val = evaluate(
        fusion,
        projection,
        val_loader,
        device,
    )

    results = {
        "best_epoch": (
            best_epoch
        ),
        "selection_metric": (
            "val_mrr"
        ),
        "fusion_frozen": True,
        "fusion_checkpoint": str(
            args.fusion_checkpoint
        ),
        "pca_fit_split": "train",
        "pca_output_dim": 256,
        "pca_path": str(
            pca_path
        ),
        "initial": {
            "train": (
                initial_train
            ),
            "val": (
                initial_val
            ),
        },
        "final": {
            "train": (
                final_train
            ),
            "val": (
                final_val
            ),
        },
        "history": (
            history
        ),
    }

    results_path = (
        args.output_dir
        / "results.json"
    )

    with open(
        results_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    print()
    print(
        "best epoch:",
        best_epoch,
    )

    print()

    print("FINAL TRAIN")
    for key, value in (
        final_train.items()
    ):
        print(
            key,
            "=",
            value,
        )

    print()

    print("FINAL VAL")
    for key, value in (
        final_val.items()
    ):
        print(
            key,
            "=",
            value,
        )

    print()

    print(
        "PCA:",
        pca_path,
    )

    print(
        "projection:",
        checkpoint_path,
    )

    print(
        "results:",
        results_path,
    )

    print("PASS")


if __name__ == "__main__":
    main()