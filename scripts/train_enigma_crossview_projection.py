from pathlib import Path
import argparse
import json
import math
import random
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Data loading
# ============================================================

@dataclass
class PairSample:
    pair_group_id: str
    video_uid: str
    phase: str
    what: str
    ego_feat: torch.Tensor
    exo_feat: torch.Tensor


def load_pairs(cache_root: Path, video_ids):
    """
    Load complete ego/exo pairs from cached ENIGMA features.

    Each cached file contains frame_features with shape [8, 960].
    We mean-pool across time here so this experiment isolates whether
    a lightweight learned projection can align the two viewpoints.
    """
    video_ids = {str(v) for v in video_ids}
    grouped = {}

    for p in sorted(cache_root.rglob("*.pt")):
        x = torch.load(p, map_location="cpu")

        video_uid = str(x["video_uid"])
        if video_uid not in video_ids:
            continue

        pair_id = str(x["pair_group_id"])
        view_type = str(x["view_type"])

        feat = x["frame_features"].float()

        if feat.ndim != 2 or feat.shape[-1] != 960:
            raise RuntimeError(
                f"Unexpected feature shape {tuple(feat.shape)} in {p}"
            )

        # Keep this diagnostic simple and directly comparable to
        # the previous raw mean-pool retrieval baseline.
        feat = feat.mean(dim=0)  # [960]

        item = {
            "feat": feat,
            "video_uid": video_uid,
            "phase": str(x["phase"]),
            "what": str(x.get("what", "")),
        }

        grouped.setdefault(pair_id, {})
        grouped[pair_id][view_type] = item

    pairs = []

    for pair_id, views in sorted(grouped.items()):
        if "ego" not in views or "exo" not in views:
            continue

        ego = views["ego"]
        exo = views["exo"]

        if ego["video_uid"] != exo["video_uid"]:
            raise RuntimeError(
                f"Video mismatch inside pair {pair_id}"
            )

        if ego["phase"] != exo["phase"]:
            raise RuntimeError(
                f"Phase mismatch inside pair {pair_id}"
            )

        pairs.append(
            PairSample(
                pair_group_id=pair_id,
                video_uid=ego["video_uid"],
                phase=ego["phase"],
                what=ego["what"],
                ego_feat=ego["feat"],
                exo_feat=exo["feat"],
            )
        )

    return pairs


def stack_pairs(pairs, device):
    ego = torch.stack([p.ego_feat for p in pairs]).to(device)
    exo = torch.stack([p.exo_feat for p in pairs]).to(device)

    phases = [p.phase for p in pairs]
    pair_ids = [p.pair_group_id for p in pairs]
    video_uids = [p.video_uid for p in pairs]

    return ego, exo, phases, pair_ids, video_uids


# ============================================================
# Model
# ============================================================

class CrossViewProjection(nn.Module):
    """
    Small shared projection head:
        960 -> hidden -> 256 -> LayerNorm -> L2 normalize

    The same head is used for ego and exo so that the diagnostic
    tests whether a shared low-dimensional representation can become
    viewpoint-aligned without finetuning the frozen SmolVLM backbone.
    """

    def __init__(
        self,
        input_dim=960,
        hidden_dim=512,
        output_dim=256,
        dropout=0.0,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x):
        z = self.net(x)
        z = F.normalize(z, dim=-1)
        return z


# ============================================================
# Contrastive objective
# ============================================================

def symmetric_infonce(z_ego, z_exo, temperature):
    """
    Exact pair i <-> i is the positive.
    Every other pair in the batch is a negative.

    Full-batch training is used by default because the pilot only has
    279 train pairs, making the complete similarity matrix inexpensive.
    """
    logits = (z_ego @ z_exo.T) / temperature
    target = torch.arange(
        logits.shape[0],
        device=logits.device,
    )

    loss_e2x = F.cross_entropy(logits, target)
    loss_x2e = F.cross_entropy(logits.T, target)

    return 0.5 * (loss_e2x + loss_x2e)


# ============================================================
# Retrieval metrics
# ============================================================

def exact_pair_metrics(sim):
    """
    sim: [N, N], with exact positive on the diagonal.
    """
    n = sim.shape[0]

    ranking = torch.argsort(
        sim,
        dim=1,
        descending=True,
    )

    target = torch.arange(
        n,
        device=sim.device,
    )

    ranks = []

    for i in range(n):
        rank = (
            (ranking[i] == target[i])
            .nonzero(as_tuple=False)
            .item()
            + 1
        )
        ranks.append(rank)

    ranks = torch.tensor(
        ranks,
        dtype=torch.float32,
        device=sim.device,
    )

    positive = sim.diag()

    eye = torch.eye(
        n,
        dtype=torch.bool,
        device=sim.device,
    )

    negative = sim[~eye].view(n, n - 1)

    mean_negative = negative.mean(dim=1)
    hardest_negative = negative.max(dim=1).values

    return {
        "R@1": (ranks <= 1).float().mean().item(),
        "R@3": (ranks <= 3).float().mean().item(),
        "R@5": (ranks <= 5).float().mean().item(),
        "R@10": (ranks <= 10).float().mean().item(),
        "MRR": (1.0 / ranks).mean().item(),
        "median_rank": ranks.median().item(),
        "mean_rank": ranks.mean().item(),
        "positive_cosine_mean": positive.mean().item(),
        "mean_negative_cosine": mean_negative.mean().item(),
        "hardest_negative_cosine_mean": hardest_negative.mean().item(),
        "positive_minus_mean_negative": (
            positive - mean_negative
        ).mean().item(),
        "positive_minus_hardest_negative": (
            positive - hardest_negative
        ).mean().item(),
    }


def multi_positive_metrics(sim, query_phases, gallery_phases):
    """
    Same-action / same-phase retrieval.

    Any gallery sample with the same phase is considered positive.
    """
    ranking = torch.argsort(
        sim,
        dim=1,
        descending=True,
    )

    ranks = []
    positive_counts = []

    for i, phase in enumerate(query_phases):
        positives = [
            j
            for j, gallery_phase in enumerate(gallery_phases)
            if gallery_phase == phase
        ]

        if not positives:
            continue

        positive_set = set(positives)
        positive_counts.append(len(positives))

        first_rank = None

        for rank, j in enumerate(
            ranking[i].tolist(),
            start=1,
        ):
            if j in positive_set:
                first_rank = rank
                break

        if first_rank is not None:
            ranks.append(first_rank)

    ranks = torch.tensor(
        ranks,
        dtype=torch.float32,
        device=sim.device,
    )

    return {
        "queries": len(ranks),
        "mean_positive_count": (
            sum(positive_counts) / len(positive_counts)
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


def multi_positive_random_metrics(
    query_phases,
    gallery_phases,
):
    n = len(gallery_phases)

    all_r1 = []
    all_r3 = []
    all_r5 = []
    all_r10 = []
    all_mrr = []
    positive_counts = []

    for phase in query_phases:
        p = sum(
            1
            for gallery_phase in gallery_phases
            if gallery_phase == phase
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
                log_comb(n - r, p - 1)
                - denom
            )

            expected_mrr += prob / r

        all_mrr.append(expected_mrr)

    def mean(xs):
        return sum(xs) / len(xs)

    return {
        "queries": len(all_mrr),
        "mean_positive_count": mean(positive_counts),
        "R@1": mean(all_r1),
        "R@3": mean(all_r3),
        "R@5": mean(all_r5),
        "R@10": mean(all_r10),
        "MRR": mean(all_mrr),
    }


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    ego,
    exo,
    phases,
):
    model.eval()

    z_ego = model(ego)
    z_exo = model(exo)

    sim = z_ego @ z_exo.T

    exact_e2x = exact_pair_metrics(sim)
    exact_x2e = exact_pair_metrics(sim.T)

    action_e2x = multi_positive_metrics(
        sim,
        phases,
        phases,
    )

    action_x2e = multi_positive_metrics(
        sim.T,
        phases,
        phases,
    )

    return {
        "exact_e2x": exact_e2x,
        "exact_x2e": exact_x2e,
        "action_e2x": action_e2x,
        "action_x2e": action_x2e,
    }


# ============================================================
# Printing helpers
# ============================================================

def print_metric_block(title, metrics):
    print()
    print(title)
    print("-" * len(title))

    for key, value in metrics.items():
        if key == "queries":
            print(f"{key}: {value}")
        elif key == "mean_positive_count":
            print(f"{key}: {value:.2f}")
        elif "rank" in key:
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value:.6f}")


def print_eval(prefix, result):
    print_metric_block(
        f"{prefix} | EXACT | EGO -> EXO",
        result["exact_e2x"],
    )

    print_metric_block(
        f"{prefix} | EXACT | EXO -> EGO",
        result["exact_x2e"],
    )

    print_metric_block(
        f"{prefix} | SAME ACTION | EGO -> EXO",
        result["action_e2x"],
    )

    print_metric_block(
        f"{prefix} | SAME ACTION | EXO -> EGO",
        result["action_x2e"],
    )


# ============================================================
# Training one seed
# ============================================================

def train_one_seed(
    seed,
    train_data,
    val_data,
    args,
    output_dir,
):
    set_seed(seed)

    device = torch.device(args.device)

    train_ego, train_exo, train_phases, _, _ = stack_pairs(
        train_data,
        device,
    )

    val_ego, val_exo, val_phases, _, _ = stack_pairs(
        val_data,
        device,
    )

    model = CrossViewProjection(
        input_dim=960,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print()
    print("=" * 80)
    print(f"SEED {seed}")
    print("=" * 80)
    print("train pairs:", len(train_data))
    print("val pairs:", len(val_data))
    print(
        "trainable parameters:",
        sum(p.numel() for p in model.parameters()),
    )

    # Raw, untrained projection diagnostic.
    # This is NOT the same as raw 960-d feature retrieval, but it helps
    # show how much training changes the randomly initialized head.
    init_train = evaluate(
        model,
        train_ego,
        train_exo,
        train_phases,
    )

    init_val = evaluate(
        model,
        val_ego,
        val_exo,
        val_phases,
    )

    print_eval(
        f"SEED {seed} | INIT TRAIN",
        init_train,
    )

    print_eval(
        f"SEED {seed} | INIT VAL",
        init_val,
    )

    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()

        optimizer.zero_grad(set_to_none=True)

        z_ego = model(train_ego)
        z_exo = model(train_exo)

        loss = symmetric_infonce(
            z_ego,
            z_exo,
            args.temperature,
        )

        loss.backward()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.grad_clip,
            )

        optimizer.step()

        if (
            epoch == 1
            or epoch % args.eval_every == 0
            or epoch == args.epochs
        ):
            train_result = evaluate(
                model,
                train_ego,
                train_exo,
                train_phases,
            )

            val_result = evaluate(
                model,
                val_ego,
                val_exo,
                val_phases,
            )

            print(
                f"\n[seed={seed}] "
                f"epoch={epoch:04d} "
                f"loss={loss.item():.6f} "
                f"train_exact_mrr="
                f"{train_result['exact_e2x']['MRR']:.4f}/"
                f"{train_result['exact_x2e']['MRR']:.4f} "
                f"val_exact_mrr="
                f"{val_result['exact_e2x']['MRR']:.4f}/"
                f"{val_result['exact_x2e']['MRR']:.4f}"
            )

            history.append(
                {
                    "epoch": epoch,
                    "loss": loss.item(),
                    "train": train_result,
                    "val": val_result,
                }
            )

    final_train = evaluate(
        model,
        train_ego,
        train_exo,
        train_phases,
    )

    final_val = evaluate(
        model,
        val_ego,
        val_exo,
        val_phases,
    )

    print_eval(
        f"SEED {seed} | FINAL TRAIN",
        final_train,
    )

    print_eval(
        f"SEED {seed} | FINAL VAL",
        final_val,
    )

    seed_dir = output_dir / f"seed_{seed}"
    seed_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "seed": seed,
            "model_state_dict": model.state_dict(),
            "config": vars(args),
            "train_video_ids": args.train_video_ids,
            "val_video_ids": args.val_video_ids,
            "train_pairs": len(train_data),
            "val_pairs": len(val_data),
            "final_train": final_train,
            "final_val": final_val,
        },
        seed_dir / "projection_final.pt",
    )

    with open(
        seed_dir / "history.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            history,
            f,
            indent=2,
        )

    return {
        "seed": seed,
        "final_train": final_train,
        "final_val": final_val,
    }


# ============================================================
# Multi-seed summary
# ============================================================

def summarize_seed_results(results):
    def values(path):
        out = []

        for result in results:
            x = result

            for key in path:
                x = x[key]

            out.append(float(x))

        return out

    summary_specs = {
        "val_exact_e2x_MRR": [
            "final_val",
            "exact_e2x",
            "MRR",
        ],
        "val_exact_x2e_MRR": [
            "final_val",
            "exact_x2e",
            "MRR",
        ],
        "val_action_e2x_MRR": [
            "final_val",
            "action_e2x",
            "MRR",
        ],
        "val_action_x2e_MRR": [
            "final_val",
            "action_x2e",
            "MRR",
        ],
        "val_exact_e2x_R@1": [
            "final_val",
            "exact_e2x",
            "R@1",
        ],
        "val_exact_x2e_R@1": [
            "final_val",
            "exact_x2e",
            "R@1",
        ],
        "val_exact_e2x_margin": [
            "final_val",
            "exact_e2x",
            "positive_minus_hardest_negative",
        ],
        "val_exact_x2e_margin": [
            "final_val",
            "exact_x2e",
            "positive_minus_hardest_negative",
        ],
    }

    summary = {}

    print()
    print("=" * 80)
    print("MULTI-SEED FINAL SUMMARY")
    print("=" * 80)

    for name, path in summary_specs.items():
        xs = values(path)

        mean = float(np.mean(xs))

        if len(xs) > 1:
            std = float(
                np.std(
                    xs,
                    ddof=1,
                )
            )
        else:
            std = 0.0

        summary[name] = {
            "values": xs,
            "mean": mean,
            "sample_std": std,
        }

        values_str = ", ".join(
            f"{x:.6f}"
            for x in xs
        )

        print(
            f"{name}: "
            f"{mean:.6f} ± {std:.6f} "
            f"| [{values_str}]"
        )

    return summary


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
        "--output-dir",
        type=Path,
        default=Path(
            "/media/dhqg/d1/vlia_outputs/"
            "enigma_crossview_projection_v1"
        ),
    )

    parser.add_argument(
        "--train-video-ids",
        nargs="+",
        default=["54", "55", "56"],
    )

    parser.add_argument(
        "--val-video-ids",
        nargs="+",
        default=["57", "58"],
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2],
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--eval-every",
        type=int,
        default=25,
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
        "--hidden-dim",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--output-dim",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--device",
        type=str,
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )

    args = parser.parse_args()

    train_video_ids = {
        str(x)
        for x in args.train_video_ids
    }

    val_video_ids = {
        str(x)
        for x in args.val_video_ids
    }

    overlap = train_video_ids & val_video_ids

    if overlap:
        raise RuntimeError(
            f"Train/val video overlap detected: {sorted(overlap)}"
        )

    train_data = load_pairs(
        args.cache_root,
        args.train_video_ids,
    )

    val_data = load_pairs(
        args.cache_root,
        args.val_video_ids,
    )

    if not train_data:
        raise RuntimeError("No train pairs found.")

    if not val_data:
        raise RuntimeError("No validation pairs found.")

    print("cache_root:", args.cache_root)
    print("output_dir:", args.output_dir)
    print("device:", args.device)
    print("train_video_ids:", args.train_video_ids)
    print("val_video_ids:", args.val_video_ids)
    print("train_pairs:", len(train_data))
    print("val_pairs:", len(val_data))

    print(
        "train_unique_phases:",
        len(set(p.phase for p in train_data)),
    )

    print(
        "val_unique_phases:",
        len(set(p.phase for p in val_data)),
    )

    print(
        "phase_overlap:",
        len(
            set(p.phase for p in train_data)
            & set(p.phase for p in val_data)
        ),
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Random reference for the held-out split
    # --------------------------------------------------------

    val_phases = [p.phase for p in val_data]

    random_exact = exact_random_metrics(
        len(val_data)
    )

    random_action = multi_positive_random_metrics(
        val_phases,
        val_phases,
    )

    print_metric_block(
        "VAL RANDOM | EXACT PAIR",
        random_exact,
    )

    print_metric_block(
        "VAL RANDOM | SAME ACTION",
        random_action,
    )

    # --------------------------------------------------------
    # Train all seeds
    # --------------------------------------------------------

    all_results = []

    for seed in args.seeds:
        result = train_one_seed(
            seed=seed,
            train_data=train_data,
            val_data=val_data,
            args=args,
            output_dir=args.output_dir,
        )

        all_results.append(result)

    summary = summarize_seed_results(
        all_results
    )

    final_payload = {
        "config": vars(args),
        "train_pairs": len(train_data),
        "val_pairs": len(val_data),
        "random_val_exact": random_exact,
        "random_val_action": random_action,
        "seed_results": all_results,
        "summary": summary,
    }

    with open(
        args.output_dir / "summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            final_payload,
            f,
            indent=2,
            default=str,
        )

    print()
    print("TRAINING COMPLETE")
    print("summary:", args.output_dir / "summary.json")


if __name__ == "__main__":
    main()
