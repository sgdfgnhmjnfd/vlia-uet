from pathlib import Path
import argparse
import math

import torch
import torch.nn.functional as F


def load_samples(root: Path, video_ids):
    ego = []
    exo = []

    video_ids = set(str(v) for v in video_ids)

    for p in sorted(root.rglob("*.pt")):
        x = torch.load(p, map_location="cpu")

        if str(x["video_uid"]) not in video_ids:
            continue

        feat = x["frame_features"].float().mean(dim=0)
        feat = F.normalize(feat, dim=0)

        item = {
            "feat": feat,
            "pair_group_id": x["pair_group_id"],
            "video_uid": str(x["video_uid"]),
            "phase": str(x["phase"]),
            "what": str(x.get("what", "")),
            "sample_id": x["sample_id"],
        }

        if x["view_type"] == "ego":
            ego.append(item)
        elif x["view_type"] == "exo":
            exo.append(item)

    return ego, exo


def multi_positive_metrics(queries, gallery, positive_fn):
    qf = torch.stack([x["feat"] for x in queries])
    gf = torch.stack([x["feat"] for x in gallery])

    sim = qf @ gf.T
    ranking = torch.argsort(sim, dim=1, descending=True)

    ranks = []

    for i, q in enumerate(queries):
        positives = [
            j for j, g in enumerate(gallery)
            if positive_fn(q, g)
        ]

        if not positives:
            continue

        positive_set = set(positives)

        rank = None

        for r, j in enumerate(
            ranking[i].tolist(),
            start=1,
        ):
            if j in positive_set:
                rank = r
                break

        if rank is not None:
            ranks.append(rank)

    if not ranks:
        raise RuntimeError("No valid queries with positive samples.")

    ranks = torch.tensor(
        ranks,
        dtype=torch.float32,
    )

    return {
        "queries": len(ranks),
        "R@1": (ranks <= 1).float().mean().item(),
        "R@3": (ranks <= 3).float().mean().item(),
        "R@5": (ranks <= 5).float().mean().item(),
        "R@10": (ranks <= 10).float().mean().item(),
        "MRR": (1.0 / ranks).mean().item(),
        "median_rank": ranks.median().item(),
        "mean_rank": ranks.mean().item(),
    }


def log_comb(n, k):
    if k < 0 or k > n:
        return float("-inf")

    return (
        math.lgamma(n + 1)
        - math.lgamma(k + 1)
        - math.lgamma(n - k + 1)
    )


def random_metrics(queries, gallery, positive_fn):
    """
    Exact expected random retrieval baseline for a multi-positive setting.

    For each query:
        N = gallery size
        P = number of valid positives

    R@K:
        probability that at least one positive appears
        in the first K positions of a random ranking.

    MRR:
        expected reciprocal rank of the first positive.
    """

    N = len(gallery)

    r1 = []
    r3 = []
    r5 = []
    r10 = []
    mrr = []

    positive_counts = []

    for q in queries:
        P = sum(
            1
            for g in gallery
            if positive_fn(q, g)
        )

        if P == 0:
            continue

        positive_counts.append(P)

        def expected_recall_at(k):
            k = min(k, N)

            # Impossible to avoid all positives.
            if N - k < P:
                return 1.0

            log_no_positive = (
                log_comb(N - k, P)
                - log_comb(N, P)
            )

            no_positive = math.exp(log_no_positive)

            return 1.0 - no_positive

        r1.append(expected_recall_at(1))
        r3.append(expected_recall_at(3))
        r5.append(expected_recall_at(5))
        r10.append(expected_recall_at(10))

        # Distribution of minimum positive rank:
        #
        # P(R_min = r)
        # =
        # C(N-r, P-1) / C(N, P)
        #
        # for r = 1, ..., N-P+1
        denom = log_comb(N, P)

        expected_mrr = 0.0

        for r in range(1, N - P + 2):
            log_prob = (
                log_comb(N - r, P - 1)
                - denom
            )

            prob = math.exp(log_prob)

            expected_mrr += prob / r

        mrr.append(expected_mrr)

    if not mrr:
        raise RuntimeError(
            "No valid queries for random baseline."
        )

    def mean(xs):
        return sum(xs) / len(xs)

    return {
        "queries": len(mrr),
        "mean_positive_count": mean(positive_counts),
        "min_positive_count": min(positive_counts),
        "max_positive_count": max(positive_counts),
        "R@1": mean(r1),
        "R@3": mean(r3),
        "R@5": mean(r5),
        "R@10": mean(r10),
        "MRR": mean(mrr),
    }


def print_metrics(title, metrics):
    print(f"\n{title}")
    print("-" * len(title))

    for k, v in metrics.items():
        if k in {
            "queries",
            "min_positive_count",
            "max_positive_count",
        }:
            print(f"{k}: {v}")

        elif "rank" in k:
            print(f"{k}: {v:.2f}")

        elif k == "mean_positive_count":
            print(f"{k}: {v:.2f}")

        else:
            print(f"{k}: {v:.6f}")


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

    ego, exo = load_samples(
        args.cache_root,
        args.video_ids,
    )

    print("cache_root:", args.cache_root)
    print("videos:", args.video_ids)

    print("\nego samples:", len(ego))
    print("exo samples:", len(exo))

    print(
        "unique ego phases:",
        len(set(x["phase"] for x in ego)),
    )

    print(
        "unique exo phases:",
        len(set(x["phase"] for x in exo)),
    )

    # =========================================================
    # Positive definitions
    # =========================================================

    phase_fn = lambda q, g: (
        q["phase"] == g["phase"]
    )

    video_phase_fn = lambda q, g: (
        q["phase"] == g["phase"]
        and q["video_uid"] == g["video_uid"]
    )

    # =========================================================
    # Actual frozen-feature retrieval
    # =========================================================

    ego_to_exo_phase = multi_positive_metrics(
        ego,
        exo,
        phase_fn,
    )

    exo_to_ego_phase = multi_positive_metrics(
        exo,
        ego,
        phase_fn,
    )

    ego_to_exo_video_phase = multi_positive_metrics(
        ego,
        exo,
        video_phase_fn,
    )

    exo_to_ego_video_phase = multi_positive_metrics(
        exo,
        ego,
        video_phase_fn,
    )

    print_metrics(
        "EGO -> EXO | SAME ACTION/PHASE",
        ego_to_exo_phase,
    )

    print_metrics(
        "EXO -> EGO | SAME ACTION/PHASE",
        exo_to_ego_phase,
    )

    print_metrics(
        "EGO -> EXO | SAME VIDEO + ACTION",
        ego_to_exo_video_phase,
    )

    print_metrics(
        "EXO -> EGO | SAME VIDEO + ACTION",
        exo_to_ego_video_phase,
    )

    # =========================================================
    # Exact random multi-positive baselines
    # =========================================================

    random_ego_to_exo_phase = random_metrics(
        ego,
        exo,
        phase_fn,
    )

    random_exo_to_ego_phase = random_metrics(
        exo,
        ego,
        phase_fn,
    )

    random_ego_to_exo_video_phase = random_metrics(
        ego,
        exo,
        video_phase_fn,
    )

    random_exo_to_ego_video_phase = random_metrics(
        exo,
        ego,
        video_phase_fn,
    )

    print_metrics(
        "RANDOM | EGO -> EXO | SAME ACTION/PHASE",
        random_ego_to_exo_phase,
    )

    print_metrics(
        "RANDOM | EXO -> EGO | SAME ACTION/PHASE",
        random_exo_to_ego_phase,
    )

    print_metrics(
        "RANDOM | EGO -> EXO | SAME VIDEO + ACTION",
        random_ego_to_exo_video_phase,
    )

    print_metrics(
        "RANDOM | EXO -> EGO | SAME VIDEO + ACTION",
        random_exo_to_ego_video_phase,
    )

    # =========================================================
    # Gain over random
    # =========================================================

    def print_gain(
        title,
        actual,
        random_ref,
    ):
        print(f"\n{title}")
        print("-" * len(title))

        for metric in [
            "R@1",
            "R@3",
            "R@5",
            "R@10",
            "MRR",
        ]:
            a = actual[metric]
            r = random_ref[metric]

            abs_gain = a - r

            if r > 0:
                ratio = a / r
            else:
                ratio = float("inf")

            print(
                f"{metric}: "
                f"actual={a:.6f} | "
                f"random={r:.6f} | "
                f"delta={abs_gain:+.6f} | "
                f"x{ratio:.2f}"
            )

    print_gain(
        "GAIN | EGO -> EXO | SAME ACTION/PHASE",
        ego_to_exo_phase,
        random_ego_to_exo_phase,
    )

    print_gain(
        "GAIN | EXO -> EGO | SAME ACTION/PHASE",
        exo_to_ego_phase,
        random_exo_to_ego_phase,
    )

    print_gain(
        "GAIN | EGO -> EXO | SAME VIDEO + ACTION",
        ego_to_exo_video_phase,
        random_ego_to_exo_video_phase,
    )

    print_gain(
        "GAIN | EXO -> EGO | SAME VIDEO + ACTION",
        exo_to_ego_video_phase,
        random_exo_to_ego_video_phase,
    )


if __name__ == "__main__":
    main()