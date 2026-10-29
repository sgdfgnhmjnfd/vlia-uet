from pathlib import Path
import argparse
from collections import defaultdict

import torch
import torch.nn.functional as F


def load_pairs(root: Path):
    pair_dict = defaultdict(dict)

    for p in sorted(root.rglob("*.pt")):
        x = torch.load(p, map_location="cpu")

        pair_id = x["pair_group_id"]
        view_type = x["view_type"]
        feat = x["frame_features"].float()  # [8, 960]

        # First sanity check: simple temporal mean pooling
        feat = feat.mean(dim=0)  # [960]
        feat = F.normalize(feat, dim=-1)

        pair_dict[pair_id][view_type] = {
            "feat": feat,
            "sample_id": x["sample_id"],
            "video_uid": str(x["video_uid"]),
            "phase": str(x["phase"]),
            "what": x.get("what"),
        }

    valid = {
        k: v
        for k, v in pair_dict.items()
        if "ego" in v and "exo" in v
    }

    return valid


def retrieval_metrics(query_feats, gallery_feats):
    sim = query_feats @ gallery_feats.T
    n = sim.shape[0]

    # positive is same row index because arrays are built from same ordered pair IDs
    target = torch.arange(n)

    ranking = torch.argsort(sim, dim=1, descending=True)

    ranks = []
    for i in range(n):
        rank = (ranking[i] == target[i]).nonzero(as_tuple=False).item() + 1
        ranks.append(rank)

    ranks = torch.tensor(ranks, dtype=torch.float32)

    metrics = {
        "R@1": (ranks <= 1).float().mean().item(),
        "R@3": (ranks <= 3).float().mean().item(),
        "R@5": (ranks <= 5).float().mean().item(),
        "MRR": (1.0 / ranks).mean().item(),
        "median_rank": ranks.median().item(),
        "mean_rank": ranks.mean().item(),
    }

    pos = sim.diag()

    eye = torch.eye(n, dtype=torch.bool)
    neg = sim[~eye].view(n, n - 1)

    hardest_neg = neg.max(dim=1).values
    mean_neg = neg.mean(dim=1)

    metrics["positive_cosine_mean"] = pos.mean().item()
    metrics["hardest_negative_cosine_mean"] = hardest_neg.mean().item()
    metrics["mean_negative_cosine"] = mean_neg.mean().item()
    metrics["positive_minus_hardest_margin"] = (
        pos - hardest_neg
    ).mean().item()
    metrics["positive_minus_mean_negative"] = (
        pos - mean_neg
    ).mean().item()

    return metrics


def print_metrics(name, metrics):
    print(f"\n{name}")
    print("-" * len(name))

    for k, v in metrics.items():
        if "rank" in k:
            print(f"{k}: {v:.2f}")
        else:
            print(f"{k}: {v:.6f}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path(
            "/media/dhqg/d1/datasets/enigma360/cache/crossview_v1/train"
        ),
    )

    parser.add_argument(
        "--video-ids",
        nargs="*",
        default=["54", "55", "56", "57", "58"],
    )

    args = parser.parse_args()

    pairs = load_pairs(args.cache_root)

    selected = {
        k: v
        for k, v in pairs.items()
        if v["ego"]["video_uid"] in set(args.video_ids)
    }

    pair_ids = sorted(selected.keys())

    print("cache_root:", args.cache_root)
    print("videos:", args.video_ids)
    print("complete_pairs:", len(pair_ids))

    if not pair_ids:
        raise RuntimeError("No complete pairs found.")

    ego = torch.stack(
        [selected[k]["ego"]["feat"] for k in pair_ids]
    )

    exo = torch.stack(
        [selected[k]["exo"]["feat"] for k in pair_ids]
    )

    print("ego features:", tuple(ego.shape))
    print("exo features:", tuple(exo.shape))

    ego_to_exo = retrieval_metrics(ego, exo)
    exo_to_ego = retrieval_metrics(exo, ego)

    print_metrics("EGO -> EXO", ego_to_exo)
    print_metrics("EXO -> EGO", exo_to_ego)

    n = len(pair_ids)

    random_r1 = 1.0 / n
    random_r3 = min(3.0 / n, 1.0)
    random_r5 = min(5.0 / n, 1.0)

    # Expected MRR for random permutation:
    # H_n / n
    harmonic = sum(1.0 / i for i in range(1, n + 1))
    random_mrr = harmonic / n

    print("\nRANDOM REFERENCE")
    print("----------------")
    print(f"R@1: {random_r1:.6f}")
    print(f"R@3: {random_r3:.6f}")
    print(f"R@5: {random_r5:.6f}")
    print(f"MRR: {random_mrr:.6f}")


if __name__ == "__main__":
    main()