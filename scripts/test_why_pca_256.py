from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


CACHE_ROOT = Path(
    "/media/dhqg/d1/datasets/egointent/cache/stage_a_v0"
)


def load_split(split):
    features = []
    labels = []

    for path in sorted(
        (CACHE_ROOT / split).glob("*.pt")
    ):
        record = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        features.append(
            record["why_feature"].float()
        )

        labels.append(
            record["why"]
        )

    return (
        torch.stack(features),
        labels,
    )


def fit_pca(
    train_features,
    output_dim=256,
):
    mean = train_features.mean(
        dim=0,
        keepdim=True,
    )

    centered = (
        train_features - mean
    )

    _, _, vh = torch.linalg.svd(
        centered,
        full_matrices=False,
    )

    components = vh[
        :output_dim
    ].T.contiguous()

    return mean, components


def transform(
    features,
    mean,
    components,
):
    return (
        features - mean
    ) @ components


def retrieval_metrics(
    features,
    labels,
):
    x = F.normalize(
        features.float(),
        dim=-1,
    )

    similarity = x @ x.T

    n = len(labels)

    top1 = 0
    top3 = 0
    reciprocal_ranks = []

    for i in range(n):
        scores = similarity[i].clone()

        # Do not retrieve the query itself.
        scores[i] = -float("inf")

        ranking = torch.argsort(
            scores,
            descending=True,
        )

        ranked_labels = [
            labels[j]
            for j in ranking.tolist()
        ]

        target = labels[i]

        if ranked_labels[0] == target:
            top1 += 1

        if target in ranked_labels[:3]:
            top3 += 1

        rank_found = None

        for rank, label in enumerate(
            ranked_labels,
            start=1,
        ):
            if label == target:
                rank_found = rank
                break

        if rank_found is not None:
            reciprocal_ranks.append(
                1.0 / rank_found
            )

    return {
        "top1": top1 / n,
        "top3": top3 / n,
        "mrr": float(
            np.mean(reciprocal_ranks)
        ),
    }


def geometry_stats(features):
    x = F.normalize(
        features.float(),
        dim=-1,
    )

    similarity = x @ x.T

    n = x.shape[0]

    mask = ~torch.eye(
        n,
        dtype=torch.bool,
    )

    pairwise = similarity[
        mask
    ]

    return {
        "mean_pairwise_cosine": (
            pairwise.mean().item()
        ),
        "median_pairwise_cosine": (
            pairwise.median().item()
        ),
    }


def main():
    train_x, train_labels = load_split(
        "train"
    )

    val_x, val_labels = load_split(
        "val"
    )

    print(
        "train:",
        train_x.shape,
    )

    print(
        "val:",
        val_x.shape,
    )

    mean, components = fit_pca(
        train_x,
        output_dim=256,
    )

    train_256 = transform(
        train_x,
        mean,
        components,
    )

    val_256 = transform(
        val_x,
        mean,
        components,
    )

    print()
    print("RAW 960-D VAL GEOMETRY")
    print(
        geometry_stats(
            val_x
        )
    )

    print()
    print("PCA 256-D VAL GEOMETRY")
    print(
        geometry_stats(
            val_256
        )
    )

    print()
    print(
        "RAW 960-D VAL RETRIEVAL"
    )
    print(
        retrieval_metrics(
            val_x,
            val_labels,
        )
    )

    print()
    print(
        "PCA 256-D VAL RETRIEVAL"
    )
    print(
        retrieval_metrics(
            val_256,
            val_labels,
        )
    )

    explained = (
        torch.var(
            train_256,
            dim=0,
        ).sum()
        /
        torch.var(
            train_x,
            dim=0,
        ).sum()
    )

    print()
    print(
        "variance retained:",
        explained.item(),
    )

    print("PASS")


if __name__ == "__main__":
    main()