from pathlib import Path
import argparse
import math

import torch
import torch.nn.functional as F


# ============================================================
# Loading
# ============================================================

def load_pairs(root: Path, video_ids):
    video_ids = set(str(v) for v in video_ids)

    pairs = {}

    for p in sorted(root.rglob("*.pt")):
        x = torch.load(p, map_location="cpu")

        video_uid = str(x["video_uid"])

        if video_uid not in video_ids:
            continue

        pair_id = x["pair_group_id"]
        view = x["view_type"]

        feat = x["frame_features"].float()  # [8, 960]

        if tuple(feat.shape) != (8, 960):
            raise RuntimeError(
                f"Unexpected feature shape {feat.shape}: {p}"
            )

        item = {
            "feat": feat,
            "sample_id": x["sample_id"],
            "pair_group_id": pair_id,
            "video_uid": video_uid,
            "phase": str(x["phase"]),
            "what": str(x.get("what", "")),
        }

        pairs.setdefault(pair_id, {})
        pairs[pair_id][view] = item

    pairs = {
        k: v
        for k, v in pairs.items()
        if "ego" in v and "exo" in v
    }

    return pairs


# ============================================================
# Similarity functions
# ============================================================

def mean_pool_similarity(a, b):
    """
    a, b: [T, D]

    Mean-pool clip first, then cosine.
    This reproduces our previous diagnostic.
    """

    a = a.mean(dim=0)
    b = b.mean(dim=0)

    a = F.normalize(a, dim=0)
    b = F.normalize(b, dim=0)

    return torch.dot(a, b)


def diagonal_temporal_similarity(a, b):
    """
    Compare normalized frame i in ego against frame i in exo.

    Since ENIGMA ego/exo segments use the same temporal interval
    and both are uniformly sampled to 8 positions, this tests
    whether corresponding temporal positions carry alignment.
    """

    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)

    sim = a @ b.T  # [T, T]

    return sim.diag().mean()


def symmetric_max_similarity(a, b):
    """
    Soft temporal correspondence without requiring exact
    frame-to-frame synchronization.

    For each ego frame:
        find best matching exo frame.

    For each exo frame:
        find best matching ego frame.

    Average both directions.
    """

    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)

    sim = a @ b.T  # [T, T]

    ego_to_exo = sim.max(dim=1).values.mean()
    exo_to_ego = sim.max(dim=0).values.mean()

    return 0.5 * (ego_to_exo + exo_to_ego)


# ============================================================
# Similarity matrix
# ============================================================

def build_similarity_matrix(
    ego,
    exo,
    similarity_fn,
):
    n = len(ego)

    result = torch.empty(
        (n, n),
        dtype=torch.float32,
    )

    for i in range(n):
        if (i + 1) % 50 == 0 or i == n - 1:
            print(
                f"  computing row {i + 1}/{n}",
                flush=True,
            )

        for j in range(n):
            result[i, j] = similarity_fn(
                ego[i]["feat"],
                exo[j]["feat"],
            )

    return result


# ============================================================
# Exact-pair retrieval
# ============================================================

def exact_pair_metrics(sim):
    n = sim.shape[0]

    ranking = torch.argsort(
        sim,
        dim=1,
        descending=True,
    )

    ranks = []

    for i in range(n):
        rank = (
            (ranking[i] == i)
            .nonzero(as_tuple=False)
            .item()
            + 1
        )

        ranks.append(rank)

    ranks = torch.tensor(
        ranks,
        dtype=torch.float32,
    )

    positive = sim.diag()

    eye = torch.eye(
        n,
        dtype=torch.bool,
    )

    negatives = sim[~eye].view(n, n - 1)

    hardest_negative = negatives.max(dim=1).values
    mean_negative = negatives.mean(dim=1)

    return {
        "queries": n,
        "R@1": (ranks <= 1).float().mean().item(),
        "R@3": (ranks <= 3).float().mean().item(),
        "R@5": (ranks <= 5).float().mean().item(),
        "R@10": (ranks <= 10).float().mean().item(),
        "MRR": (1.0 / ranks).mean().item(),
        "median_rank": ranks.median().item(),
        "mean_rank": ranks.mean().item(),
        "positive_score_mean": positive.mean().item(),
        "mean_negative_score": mean_negative.mean().item(),
        "hardest_negative_score_mean": (
            hardest_negative.mean().item()
        ),
        "positive_minus_mean_negative": (
            positive - mean_negative
        ).mean().item(),
        "positive_minus_hardest_negative": (
            positive - hardest_negative
        ).mean().item(),
    }


# ============================================================
# Multi-positive semantic retrieval
# ============================================================

def multipositive_metrics(
    sim,
    queries,
    gallery,
):
    ranking = torch.argsort(
        sim,
        dim=1,
        descending=True,
    )

    ranks = []
    positive_counts = []

    for i, q in enumerate(queries):
        positive_indices = [
            j
            for j, g in enumerate(gallery)
            if q["phase"] == g["phase"]
        ]

        if not positive_indices:
            continue

        positive_counts.append(
            len(positive_indices)
        )

        positive_set = set(positive_indices)

        first_positive_rank = None

        for rank, idx in enumerate(
            ranking[i].tolist(),
            start=1,
        ):
            if idx in positive_set:
                first_positive_rank = rank
                break

        ranks.append(first_positive_rank)

    ranks = torch.tensor(
        ranks,
        dtype=torch.float32,
    )

    return {
        "queries": len(ranks),
        "mean_positive_count": (
            sum(positive_counts)
            / len(positive_counts)
        ),
        "R@1": (ranks <= 1).float().mean().item(),
        "R@3": (ranks <= 3).float().mean().item(),
        "R@5": (ranks <= 5).float().mean().item(),
        "R@10": (ranks <= 10).float().mean().item(),
        "MRR": (1.0 / ranks).mean().item(),
        "median_rank": ranks.median().item(),
        "mean_rank": ranks.mean().item(),
    }


# ============================================================
# Random references
# ============================================================

def log_comb(n, k):
    if k < 0 or k > n:
        return float("-inf")

    return (
        math.lgamma(n + 1)
        - math.lgamma(k + 1)
        - math.lgamma(n - k + 1)
    )


def exact_random_metrics(n):
    harmonic = sum(
        1.0 / r
        for r in range(1, n + 1)
    )

    return {
        "R@1": 1.0 / n,
        "R@3": min(3.0 / n, 1.0),
        "R@5": min(5.0 / n, 1.0),
        "R@10": min(10.0 / n, 1.0),
        "MRR": harmonic / n,
    }


def multipositive_random_metrics(
    queries,
    gallery,
):
    n = len(gallery)

    all_r1 = []
    all_r3 = []
    all_r5 = []
    all_r10 = []
    all_mrr = []

    positive_counts = []

    for q in queries:
        p = sum(
            1
            for g in gallery
            if q["phase"] == g["phase"]
        )

        if p == 0:
            continue

        positive_counts.append(p)

        def recall_at(k):
            k = min(k, n)

            if n - k < p:
                return 1.0

            no_positive = math.exp(
                log_comb(n - k, p)
                - log_comb(n, p)
            )

            return 1.0 - no_positive

        all_r1.append(recall_at(1))
        all_r3.append(recall_at(3))
        all_r5.append(recall_at(5))
        all_r10.append(recall_at(10))

        denom = log_comb(n, p)

        expected_mrr = 0.0

        for r in range(
            1,
            n - p + 2,
        ):
            prob = math.exp(
                log_comb(
                    n - r,
                    p - 1,
                )
                - denom
            )

            expected_mrr += prob / r

        all_mrr.append(expected_mrr)

    def mean(x):
        return sum(x) / len(x)

    return {
        "queries": len(all_mrr),
        "mean_positive_count": mean(
            positive_counts
        ),
        "R@1": mean(all_r1),
        "R@3": mean(all_r3),
        "R@5": mean(all_r5),
        "R@10": mean(all_r10),
        "MRR": mean(all_mrr),
    }


# ============================================================
# Printing
# ============================================================

def print_metrics(title, metrics):
    print()
    print(title)
    print("-" * len(title))

    for k, v in metrics.items():

        if k == "queries":
            print(f"{k}: {v}")

        elif "rank" in k:
            print(f"{k}: {v:.2f}")

        elif k == "mean_positive_count":
            print(f"{k}: {v:.2f}")

        else:
            print(f"{k}: {v:.6f}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path(
            "/media/dhqg/d1/datasets/"
            "enigma360/cache/crossview_v1/train"
        ),
    )

    parser.add_argument(
        "--video-ids",
        nargs="+",
        default=[
            "54",
            "55",
            "56",
            "57",
            "58",
        ],
    )

    args = parser.parse_args()

    pairs = load_pairs(
        args.cache_root,
        args.video_ids,
    )

    pair_ids = sorted(pairs.keys())

    ego = [
        pairs[k]["ego"]
        for k in pair_ids
    ]

    exo = [
        pairs[k]["exo"]
        for k in pair_ids
    ]

    print("cache_root:", args.cache_root)
    print("videos:", args.video_ids)
    print("complete pairs:", len(pair_ids))
    print("feature shape:", ego[0]["feat"].shape)

    exact_random = exact_random_metrics(
        len(pair_ids)
    )

    semantic_random = (
        multipositive_random_metrics(
            ego,
            exo,
        )
    )

    methods = [
        (
            "MEAN POOL",
            mean_pool_similarity,
        ),
        (
            "DIAGONAL TEMPORAL",
            diagonal_temporal_similarity,
        ),
        (
            "SYMMETRIC MAX TEMPORAL",
            symmetric_max_similarity,
        ),
    ]

    for name, similarity_fn in methods:

        print()
        print("=" * 70)
        print(name)
        print("=" * 70)

        sim = build_similarity_matrix(
            ego,
            exo,
            similarity_fn,
        )

        # -----------------------------------------------------
        # Exact pair
        # -----------------------------------------------------

        ego_to_exo_exact = exact_pair_metrics(
            sim
        )

        exo_to_ego_exact = exact_pair_metrics(
            sim.T
        )

        print_metrics(
            f"{name} | EGO -> EXO | EXACT PAIR",
            ego_to_exo_exact,
        )

        print_metrics(
            f"{name} | EXO -> EGO | EXACT PAIR",
            exo_to_ego_exact,
        )

        # -----------------------------------------------------
        # Same semantic action / phase
        # -----------------------------------------------------

        ego_to_exo_phase = (
            multipositive_metrics(
                sim,
                ego,
                exo,
            )
        )

        exo_to_ego_phase = (
            multipositive_metrics(
                sim.T,
                exo,
                ego,
            )
        )

        print_metrics(
            f"{name} | EGO -> EXO | SAME ACTION/PHASE",
            ego_to_exo_phase,
        )

        print_metrics(
            f"{name} | EXO -> EGO | SAME ACTION/PHASE",
            exo_to_ego_phase,
        )

    # ========================================================
    # Random references
    # ========================================================

    print()
    print("=" * 70)
    print("RANDOM REFERENCES")
    print("=" * 70)

    print_metrics(
        "RANDOM | EXACT PAIR",
        exact_random,
    )

    print_metrics(
        "RANDOM | SAME ACTION/PHASE",
        semantic_random,
    )


if __name__ == "__main__":
    main()