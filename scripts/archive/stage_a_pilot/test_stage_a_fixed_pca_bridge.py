from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


CACHE_ROOT = Path(
    "/media/dhqg/d1/datasets/egointent/cache/stage_a_v0"
)

FUSION_CHECKPOINT = Path(
    "/media/dhqg/d1/vlia_outputs/"
    "stage_a_v1_contrastive/visual/best_fusion.pt"
)

PCA_PATH = Path(
    "/media/dhqg/d1/vlia_outputs/"
    "stage_a_v1_projection_256/why_pca_256.pt"
)


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


def load_split(split):
    visual = []
    why_960 = []
    labels = []

    for path in sorted(
        (CACHE_ROOT / split).glob("*.pt")
    ):
        record = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        visual.append(
            record["visual_feature"].float()
        )

        why_960.append(
            record["why_feature"].float()
        )

        labels.append(
            record["why"]
        )

    return (
        torch.stack(visual),
        torch.stack(why_960),
        labels,
    )


def evaluate_retrieval(
    pred,
    target,
    labels,
):
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

    n = len(labels)

    top1 = 0
    top3 = 0
    reciprocal_ranks = []
    margins = []

    for i in range(n):
        scores = similarity[i]
        query_label = labels[i]

        ranking = torch.argsort(
            scores,
            descending=True,
        )

        ranked_labels = [
            labels[j]
            for j in ranking.tolist()
        ]

        if ranked_labels[0] == query_label:
            top1 += 1

        if query_label in ranked_labels[:3]:
            top3 += 1

        for rank, label in enumerate(
            ranked_labels,
            start=1,
        ):
            if label == query_label:
                reciprocal_ranks.append(
                    1.0 / rank
                )
                break

        positive_indices = [
            j
            for j in range(n)
            if labels[j] == query_label
        ]

        negative_indices = [
            j
            for j in range(n)
            if labels[j] != query_label
        ]

        best_positive = scores[
            positive_indices
        ].max().item()

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
            top1 / n
        ),
        "top3": (
            top3 / n
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
    }


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("device:", device)

    fusion = StageAFusion().to(
        device
    )

    checkpoint = torch.load(
        FUSION_CHECKPOINT,
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

    pca = torch.load(
        PCA_PATH,
        map_location="cpu",
        weights_only=False,
    )

    pca_mean = (
        pca["mean"]
        .float()
        .to(device)
    )

    pca_components = (
        pca["components"]
        .float()
        .to(device)
    )

    for split in [
        "train",
        "val",
    ]:
        visual, why_960, labels = (
            load_split(split)
        )

        visual = visual.to(
            device
        )

        why_960 = why_960.to(
            device
        )

        with torch.no_grad():
            z_align = fusion(
                visual
            )

            # Fixed semantic PCA bridge.
            z_int = (
                z_align
                - pca_mean.unsqueeze(0)
            ) @ pca_components

            why_256 = (
                why_960
                - pca_mean.unsqueeze(0)
            ) @ pca_components

        metrics_960 = evaluate_retrieval(
            z_align.cpu(),
            why_960.cpu(),
            labels,
        )

        metrics_256 = evaluate_retrieval(
            z_int.cpu(),
            why_256.cpu(),
            labels,
        )

        print()
        print(
            f"{split.upper()} 960-D"
        )

        for key, value in (
            metrics_960.items()
        ):
            print(
                key,
                "=",
                value,
            )

        print()
        print(
            f"{split.upper()} FIXED PCA 256-D"
        )

        for key, value in (
            metrics_256.items()
        ):
            print(
                key,
                "=",
                value,
            )

    print()
    print("PASS")


if __name__ == "__main__":
    main()