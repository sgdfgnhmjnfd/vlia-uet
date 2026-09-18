from __future__ import annotations

import argparse
from pathlib import Path

import torch


DEFAULT_CACHE_ROOT = Path(
    "/media/dhqg/d1/datasets/egointent/"
    "cache/temporal_intention_v1"
)

DEFAULT_OUTPUT_ROOT = Path(
    "/media/dhqg/d1/vlia_outputs/"
    "structured_intention_v1"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fit train-only PCA projections for "
            "WHAT and NEXT semantic targets."
        )
    )

    parser.add_argument(
        "--cache-root",
        type=Path,
        default=DEFAULT_CACHE_ROOT,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )

    parser.add_argument(
        "--output-dim",
        type=int,
        default=256,
    )

    return parser.parse_args()


def load_feature_matrix(
    cache_dir: Path,
    feature_key: str,
):
    paths = sorted(
        cache_dir.glob("*.pt")
    )

    if not paths:
        raise RuntimeError(
            f"No .pt files found in {cache_dir}"
        )

    features = []
    sample_ids = []

    for path in paths:
        record = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        if feature_key not in record:
            raise KeyError(
                f"Missing '{feature_key}' in {path}"
            )

        feature = record[
            feature_key
        ].detach().float().cpu()

        if feature.shape != (960,):
            raise ValueError(
                f"Unexpected {feature_key} shape "
                f"{tuple(feature.shape)} "
                f"in {path}"
            )

        if not torch.isfinite(
            feature
        ).all():
            raise ValueError(
                f"Non-finite values in "
                f"{feature_key}: {path}"
            )

        features.append(
            feature
        )

        sample_ids.append(
            str(
                record[
                    "sample_id"
                ]
            )
        )

    matrix = torch.stack(
        features,
        dim=0,
    )

    return (
        matrix,
        sample_ids,
    )


def fit_pca(
    x: torch.Tensor,
    output_dim: int,
):
    if x.ndim != 2:
        raise ValueError(
            f"Expected 2-D matrix, got {x.shape}"
        )

    num_samples, input_dim = x.shape

    if output_dim > input_dim:
        raise ValueError(
            f"output_dim={output_dim} "
            f"> input_dim={input_dim}"
        )

    if output_dim > num_samples:
        raise ValueError(
            f"output_dim={output_dim} "
            f"> num_samples={num_samples}"
        )

    mean = x.mean(
        dim=0,
        keepdim=True,
    )

    centered = (
        x - mean
    )

    # torch.pca_lowrank returns V with shape
    # [input_dim, q].
    _, _, components = torch.pca_lowrank(
        centered,
        q=output_dim,
        center=False,
    )

    components = (
        components[
            :,
            :output_dim
        ]
        .contiguous()
        .float()
        .cpu()
    )

    return {
        "mean":
            mean.float().cpu(),

        "components":
            components,

        "input_dim":
            input_dim,

        "output_dim":
            output_dim,

        "fit_split":
            "train",
    }


def transform(
    x: torch.Tensor,
    pca,
):
    mean = pca[
        "mean"
    ]

    components = pca[
        "components"
    ]

    return (
        (x - mean)
        @ components
    )


def explained_energy_ratio(
    x: torch.Tensor,
    projected: torch.Tensor,
):
    centered = (
        x
        - x.mean(
            dim=0,
            keepdim=True,
        )
    )

    total_energy = (
        centered.pow(2)
        .sum()
    )

    projected_energy = (
        projected.pow(2)
        .sum()
    )

    if total_energy <= 0:
        return 0.0

    return float(
        (
            projected_energy
            / total_energy
        ).item()
    )


def validate_pca(
    name: str,
    x: torch.Tensor,
    pca,
):
    mean = pca[
        "mean"
    ]

    components = pca[
        "components"
    ]

    output_dim = pca[
        "output_dim"
    ]

    if mean.shape != (
        1,
        960,
    ):
        raise ValueError(
            f"{name}: bad mean shape "
            f"{tuple(mean.shape)}"
        )

    if components.shape != (
        960,
        output_dim,
    ):
        raise ValueError(
            f"{name}: bad components shape "
            f"{tuple(components.shape)}"
        )

    projected = transform(
        x,
        pca,
    )

    if projected.shape != (
        x.shape[0],
        output_dim,
    ):
        raise ValueError(
            f"{name}: bad projected shape "
            f"{tuple(projected.shape)}"
        )

    if not torch.isfinite(
        projected
    ).all():
        raise ValueError(
            f"{name}: projected features "
            "contain non-finite values"
        )

    gram = (
        components.T
        @ components
    )

    identity = torch.eye(
        output_dim,
        dtype=gram.dtype,
    )

    orthogonality_error = float(
        (
            gram - identity
        )
        .abs()
        .max()
        .item()
    )

    energy_ratio = (
        explained_energy_ratio(
            x,
            projected,
        )
    )

    print(
        f"{name}:"
    )

    print(
        "  input:",
        tuple(
            x.shape
        ),
    )

    print(
        "  mean:",
        tuple(
            mean.shape
        ),
    )

    print(
        "  components:",
        tuple(
            components.shape
        ),
    )

    print(
        "  projected:",
        tuple(
            projected.shape
        ),
    )

    print(
        "  max orthogonality error:",
        orthogonality_error,
    )

    print(
        "  retained energy:",
        energy_ratio,
    )

    print(
        "  projected mean norm:",
        float(
            projected.mean(
                dim=0
            ).norm()
        ),
    )


def save_pca(
    pca,
    path: Path,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = (
        path.with_suffix(
            ".tmp"
        )
    )

    torch.save(
        pca,
        temp_path,
    )

    temp_path.replace(
        path
    )


def main():
    args = parse_args()

    train_dir = (
        args.cache_root
        / "train"
    )

    if not train_dir.exists():
        raise FileNotFoundError(
            train_dir
        )

    print("=" * 72)
    print(
        "FIT STRUCTURED TARGET PCA"
    )
    print("=" * 72)

    print(
        "train cache:",
        train_dir,
    )

    print(
        "output root:",
        args.output_root,
    )

    print(
        "output dim:",
        args.output_dim,
    )

    print()

    what_features, what_ids = (
        load_feature_matrix(
            train_dir,
            "what_feature",
        )
    )

    next_features, next_ids = (
        load_feature_matrix(
            train_dir,
            "next_feature",
        )
    )

    if what_ids != next_ids:
        raise RuntimeError(
            "WHAT/NEXT sample order mismatch"
        )

    print(
        "train samples:",
        len(
            what_ids
        ),
    )

    print(
        "WHAT matrix:",
        tuple(
            what_features.shape
        ),
    )

    print(
        "NEXT matrix:",
        tuple(
            next_features.shape
        ),
    )

    print()

    print(
        "Fitting WHAT PCA..."
    )

    what_pca = fit_pca(
        what_features,
        args.output_dim,
    )

    validate_pca(
        "WHAT",
        what_features,
        what_pca,
    )

    print()

    print(
        "Fitting NEXT PCA..."
    )

    next_pca = fit_pca(
        next_features,
        args.output_dim,
    )

    validate_pca(
        "NEXT",
        next_features,
        next_pca,
    )

    print()

    what_path = (
        args.output_root
        / "what_pca_960_to_256.pt"
    )

    next_path = (
        args.output_root
        / "next_pca_960_to_256.pt"
    )

    save_pca(
        what_pca,
        what_path,
    )

    save_pca(
        next_pca,
        next_path,
    )

    print(
        "saved WHAT PCA:",
        what_path,
    )

    print(
        "saved NEXT PCA:",
        next_path,
    )

    print()
    print(
        "PASS"
    )


if __name__ == "__main__":
    main()