#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import torch

import train_egointent_hierarchical_fixed as base


@torch.no_grad()
def score_metrics(sim, labels):
    ranking = torch.argsort(sim, dim=1, descending=True)
    ranks = []

    for i, label in enumerate(labels):
        positives = {
            j
            for j, candidate_label in enumerate(labels)
            if candidate_label == label
        }

        rank = None
        for r, j in enumerate(ranking[i].tolist(), start=1):
            if j in positives:
                rank = r
                break

        if rank is None:
            raise RuntimeError(
                f"No positive retrieval target for sample index {i}"
            )

        ranks.append(rank)

    ranks_t = torch.tensor(
        ranks,
        dtype=torch.float32,
        device=sim.device,
    )

    return {
        "Top1": (ranks_t <= 1).float().mean().item(),
        "Top3": (ranks_t <= 3).float().mean().item(),
        "Top5": (ranks_t <= 5).float().mean().item(),
        "MRR": (1.0 / ranks_t).mean().item(),
        "median_rank": ranks_t.median().item(),
        "mean_rank": ranks_t.mean().item(),
    }


@torch.no_grad()
def score_error_breakdown(sim, rows):
    ranking = torch.argsort(sim, dim=1, descending=True)

    correct = 0
    same_task_wrong = 0
    cross_task_wrong = 0

    for i, row in enumerate(rows):
        j = int(ranking[i, 0].item())
        pred_row = rows[j]

        if pred_row["why"] == row["why"]:
            correct += 1
        elif pred_row["task"] == row["task"]:
            same_task_wrong += 1
        else:
            cross_task_wrong += 1

    wrong = same_task_wrong + cross_task_wrong

    return {
        "top1_correct_count": correct,
        "same_task_wrong_count": same_task_wrong,
        "cross_task_wrong_count": cross_task_wrong,
        "same_task_wrong_fraction": (
            same_task_wrong / wrong if wrong > 0 else 0.0
        ),
    }


@torch.no_grad()
def get_scores(model, visual, why_target, condition):
    model.eval()

    if condition == "what_hardneg_guided":
        out = model.predict_intention_guided(visual)
    elif condition == "hierarchical_guided":
        out = model.predict_hierarchical_guided(visual)
    else:
        raise ValueError(condition)

    why_pred = out["why"]
    return why_pred @ why_target.T


def restore_model(state_dict, args, device):
    model = base.EgoIntentModel(
        hidden_dim=args.hidden_dim,
        visual_state_dim=args.visual_state_dim,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def mean_std(values):
    t = torch.tensor(values, dtype=torch.float64)
    return (
        float(t.mean().item()),
        float(t.std(unbiased=True).item())
        if len(values) > 1
        else 0.0,
    )


def summarize(rows):
    out = {}
    for key in [
        "MRR",
        "Top1",
        "Top3",
        "Top5",
        "same_task_wrong_fraction",
    ]:
        vals = [r[key] for r in rows]
        m, s = mean_std(vals)
        out[key + "_mean"] = m
        out[key + "_std"] = s
    return out


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train the best WHAT-hard-negative model and the hierarchical "
            "model independently, then test score-level late fusion."
        )
    )

    parser.add_argument(
        "--stage-a-cache",
        type=Path,
        default=Path(
            "/media/dhqg/d1/datasets/egointent/cache/stage_a_v0"
        ),
    )
    parser.add_argument(
        "--temporal-cache",
        type=Path,
        default=Path(
            "/media/dhqg/d1/datasets/egointent/cache/temporal_intention_v1"
        ),
    )
    parser.add_argument(
        "--what-pca",
        type=Path,
        default=Path(
            "/media/dhqg/d1/vlia_outputs/structured_intention_v1/"
            "what_pca_960_to_256.pt"
        ),
    )
    parser.add_argument(
        "--why-pca",
        type=Path,
        default=Path(
            "/media/dhqg/d1/vlia_outputs/stage_a_clean_v1/"
            "why_pca_960_to_256.pt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/media/dhqg/d1/vlia_outputs/egointent_late_fusion_v1"
        ),
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 456],
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--cosine-weight", type=float, default=0.1)
    parser.add_argument("--what-weight", type=float, default=0.3)
    parser.add_argument("--what-hard-weight", type=float, default=0.3)
    parser.add_argument("--what-hard-margin", type=float, default=0.10)
    parser.add_argument("--task-context-weight", type=float, default=0.10)
    parser.add_argument("--task-temperature", type=float, default=0.10)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--visual-state-dim", type=int, default=256)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[0.10, 0.20, 0.30, 0.40, 0.50],
        help=(
            "Hierarchical score weight. "
            "Fused score=(1-alpha)*base + alpha*hierarchical."
        ),
    )

    args = parser.parse_args()
    device = torch.device(args.device)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    what_transform = base.load_pca_transform(args.what_pca)
    why_transform = base.load_pca_transform(args.why_pca)

    train_rows = base.load_stage_a_split(
        stage_a_root=args.stage_a_cache,
        temporal_root=args.temporal_cache,
        split="train",
        what_transform=what_transform,
        why_transform=why_transform,
    )
    val_rows = base.load_stage_a_split(
        stage_a_root=args.stage_a_cache,
        temporal_root=args.temporal_cache,
        split="val",
        what_transform=what_transform,
        why_transform=why_transform,
    )

    (
        train_visual,
        train_what_target,
        train_why_target,
    ) = base.stack_rows(train_rows, device)

    (
        val_visual,
        val_what_target,
        val_why_target,
    ) = base.stack_rows(val_rows, device)

    labels = [x["why"] for x in val_rows]

    print("=" * 96)
    print("EGOINTENT SCORE-LEVEL LATE FUSION")
    print("=" * 96)
    print("device:", device)
    print("train samples:", len(train_rows))
    print("val samples:", len(val_rows))
    print("seeds:", args.seeds)
    print("alphas:", args.alphas)
    print("=" * 96)

    all_rows = []

    for seed in args.seeds:
        print()
        print("#" * 96)
        print("SEED:", seed)
        print("#" * 96)

        base_result = base.train_one(
            condition="what_hardneg_guided",
            seed=seed,
            train_visual=train_visual,
            train_what_target=train_what_target,
            train_why_target=train_why_target,
            train_rows=train_rows,
            val_visual=val_visual,
            val_what_target=val_what_target,
            val_why_target=val_why_target,
            val_rows=val_rows,
            args=args,
        )

        hier_result = base.train_one(
            condition="hierarchical_guided",
            seed=seed,
            train_visual=train_visual,
            train_what_target=train_what_target,
            train_why_target=train_why_target,
            train_rows=train_rows,
            val_visual=val_visual,
            val_what_target=val_what_target,
            val_why_target=val_why_target,
            val_rows=val_rows,
            args=args,
        )

        base_model = restore_model(
            base_result["model_state_dict"],
            args,
            device,
        )
        hier_model = restore_model(
            hier_result["model_state_dict"],
            args,
            device,
        )

        sim_base = get_scores(
            base_model,
            val_visual,
            val_why_target,
            "what_hardneg_guided",
        )
        sim_hier = get_scores(
            hier_model,
            val_visual,
            val_why_target,
            "hierarchical_guided",
        )

        # Include alpha=0 and 1 diagnostics automatically.
        sweep = sorted(set([0.0, 1.0] + args.alphas))

        for alpha in sweep:
            sim = (
                (1.0 - alpha) * sim_base
                + alpha * sim_hier
            )

            metrics = score_metrics(sim, labels)
            err = score_error_breakdown(sim, val_rows)

            row = {
                "seed": seed,
                "alpha": alpha,
                **metrics,
                **err,
            }
            all_rows.append(row)

            print(
                f"FUSION | seed={seed} | alpha={alpha:.2f} | "
                f"WHY MRR={metrics['MRR']:.6f} | "
                f"Top1={metrics['Top1']:.6f} | "
                f"Top3={metrics['Top3']:.6f} | "
                f"Top5={metrics['Top5']:.6f} | "
                f"same-task-error="
                f"{err['same_task_wrong_fraction']:.4f}"
            )

    summary = {}
    sweep = sorted(set([0.0, 1.0] + args.alphas))

    print()
    print("=" * 96)
    print("FINAL FUSION SWEEP")
    print("=" * 96)

    for alpha in sweep:
        rows = [
            r for r in all_rows
            if abs(r["alpha"] - alpha) < 1e-12
        ]
        s = summarize(rows)
        summary[str(alpha)] = s

        print(
            f"alpha={alpha:.2f} | "
            f"WHY MRR {s['MRR_mean']:.6f} ± {s['MRR_std']:.6f} | "
            f"WHY Top1 {s['Top1_mean']:.6f} ± {s['Top1_std']:.6f} | "
            f"WHY Top3 {s['Top3_mean']:.6f} ± {s['Top3_std']:.6f} | "
            f"WHY Top5 {s['Top5_mean']:.6f} ± {s['Top5_std']:.6f} | "
            f"same-task-error "
            f"{s['same_task_wrong_fraction_mean']:.4f} ± "
            f"{s['same_task_wrong_fraction_std']:.4f}"
        )

    # Pick by mean WHY MRR only; report, don't hide other metrics.
    best_alpha = max(
        sweep,
        key=lambda a: summary[str(a)]["MRR_mean"],
    )

    print()
    print(
        f"BEST_ALPHA_BY_MRR={best_alpha:.2f} | "
        f"MRR={summary[str(best_alpha)]['MRR_mean']:.6f}"
    )

    payload = {
        "experiment": "egointent_score_late_fusion",
        "config": {
            **vars(args),
            "stage_a_cache": str(args.stage_a_cache),
            "temporal_cache": str(args.temporal_cache),
            "what_pca": str(args.what_pca),
            "why_pca": str(args.why_pca),
            "output_dir": str(args.output_dir),
        },
        "summary": summary,
        "best_alpha_by_mrr": best_alpha,
        "rows": all_rows,
    }

    with open(
        args.output_dir / "late_fusion_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(payload, f, indent=2, default=str)


if __name__ == "__main__":
    main()
