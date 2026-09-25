#!/usr/bin/env python3
import argparse
from pathlib import Path
import json
import torch

import train_egointent_softmrr as exp


def restore_model(ckpt_path, device, hidden_dim=512, visual_state_dim=256):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = exp.EgoIntentModel(
        hidden_dim=hidden_dim,
        visual_state_dim=visual_state_dim,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def guided_scores(model, visual, why_target):
    out = model.predict_intention_guided(visual)
    pred = out["why"]
    return pred @ why_target.T


def positive_sets(rows):
    labels = [r["why"] for r in rows]
    sets = []
    for i, lab in enumerate(labels):
        sets.append({j for j, x in enumerate(labels) if x == lab})
    return sets


def metrics_from_scores(scores, rows):
    pos_sets = positive_sets(rows)
    ranking = torch.argsort(scores, dim=1, descending=True)

    ranks = []
    top1_correct = []
    same_task_wrong = 0
    cross_task_wrong = 0

    for i in range(scores.shape[0]):
        rank = None
        for r, j in enumerate(ranking[i].tolist(), start=1):
            if j in pos_sets[i]:
                rank = r
                break
        ranks.append(rank)

        j0 = int(ranking[i, 0].item())
        ok = j0 in pos_sets[i]
        top1_correct.append(ok)

        if not ok:
            if rows[j0]["task"] == rows[i]["task"]:
                same_task_wrong += 1
            else:
                cross_task_wrong += 1

    ranks = torch.tensor(ranks, dtype=torch.float32)
    wrong = same_task_wrong + cross_task_wrong

    return {
        "MRR": float((1.0 / ranks).mean().item()),
        "Top1": float((ranks <= 1).float().mean().item()),
        "Top3": float((ranks <= 3).float().mean().item()),
        "Top5": float((ranks <= 5).float().mean().item()),
        "same_task_error": (
            float(same_task_wrong / wrong) if wrong > 0 else 0.0
        ),
        "top1_correct_mask": torch.tensor(top1_correct, dtype=torch.bool),
    }


def mean_std(vals):
    x = torch.tensor(vals, dtype=torch.float64)
    return float(x.mean()), float(x.std(unbiased=True)) if len(vals) > 1 else 0.0


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--base-root",
        type=Path,
        required=True,
        help="Root containing what_hardneg_guided/seed_*/best.pt",
    )
    p.add_argument(
        "--soft-root",
        type=Path,
        required=True,
        help="Root containing softmrr_guided/seed_*/best.pt",
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[42,123,456])

    p.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50],
    )

    p.add_argument(
        "--stage-a-cache",
        type=Path,
        default=Path("/media/dhqg/d1/datasets/egointent/cache/stage_a_v0"),
    )
    p.add_argument(
        "--temporal-cache",
        type=Path,
        default=Path("/media/dhqg/d1/datasets/egointent/cache/temporal_intention_v1"),
    )
    p.add_argument(
        "--what-pca",
        type=Path,
        default=Path(
            "/media/dhqg/d1/vlia_outputs/structured_intention_v1/"
            "what_pca_960_to_256.pt"
        ),
    )
    p.add_argument(
        "--why-pca",
        type=Path,
        default=Path(
            "/media/dhqg/d1/vlia_outputs/stage_a_clean_v1/"
            "why_pca_960_to_256.pt"
        ),
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("/media/dhqg/d1/vlia_outputs/hardneg_softmrr_fusion.json"),
    )

    args = p.parse_args()
    device = torch.device(args.device)

    what_transform = exp.load_pca_transform(args.what_pca)
    why_transform = exp.load_pca_transform(args.why_pca)

    val_rows = exp.load_stage_a_split(
        stage_a_root=args.stage_a_cache,
        temporal_root=args.temporal_cache,
        split="val",
        what_transform=what_transform,
        why_transform=why_transform,
    )
    val_visual, val_what_target, val_why_target = exp.stack_rows(
        val_rows, device
    )

    alphas = sorted(set([0.0, 1.0] + list(args.alphas)))
    all_rows = []

    print("=" * 100)
    print("HARDNEG + SOFTMRR CHECKPOINT FUSION")
    print("=" * 100)

    for seed in args.seeds:
        base_ckpt = (
            args.base_root
            / "what_hardneg_guided"
            / f"seed_{seed}"
            / "best.pt"
        )
        soft_ckpt = (
            args.soft_root
            / "softmrr_guided"
            / f"seed_{seed}"
            / "best.pt"
        )

        if not base_ckpt.exists():
            raise FileNotFoundError(base_ckpt)
        if not soft_ckpt.exists():
            raise FileNotFoundError(soft_ckpt)

        base_model, _ = restore_model(base_ckpt, device)
        soft_model, soft_meta = restore_model(soft_ckpt, device)

        sb = guided_scores(base_model, val_visual, val_why_target)
        ss = guided_scores(soft_model, val_visual, val_why_target)

        mb = metrics_from_scores(sb, val_rows)
        ms = metrics_from_scores(ss, val_rows)

        bmask = mb["top1_correct_mask"]
        smask = ms["top1_correct_mask"]

        baseline_wrong = ~bmask
        rescue = baseline_wrong & smask
        regress = bmask & (~smask)
        disagreement = bmask ^ smask

        n_base_wrong = int(baseline_wrong.sum())
        rescue_rate = (
            float(rescue.sum()) / n_base_wrong
            if n_base_wrong > 0 else 0.0
        )

        print()
        print(f"SEED {seed}")
        print(
            f"error overlap | rescue={int(rescue.sum())} | "
            f"regress={int(regress.sum())} | "
            f"top1-disagreement={int(disagreement.sum())} | "
            f"rescue-rate={rescue_rate:.4f}"
        )

        for alpha in alphas:
            score = (1.0-alpha)*sb + alpha*ss
            m = metrics_from_scores(score, val_rows)
            row = {
                "seed": seed,
                "alpha": alpha,
                "MRR": m["MRR"],
                "Top1": m["Top1"],
                "Top3": m["Top3"],
                "Top5": m["Top5"],
                "same_task_error": m["same_task_error"],
                "rescue_rate_soft_vs_base": rescue_rate,
                "rescue_count": int(rescue.sum()),
                "regress_count": int(regress.sum()),
                "disagreement_count": int(disagreement.sum()),
            }
            all_rows.append(row)

    print()
    print("=" * 100)
    print("FINAL CHECKPOINT FUSION SWEEP")
    print("=" * 100)

    summary = {}
    for alpha in alphas:
        rows = [x for x in all_rows if x["alpha"] == alpha]
        s = {}
        for key in ["MRR","Top1","Top3","Top5","same_task_error"]:
            m, sd = mean_std([x[key] for x in rows])
            s[key] = {"mean": m, "std": sd}
        summary[str(alpha)] = s

        print(
            f"alpha={alpha:.2f} | "
            f"WHY MRR {s['MRR']['mean']:.6f} ± {s['MRR']['std']:.6f} | "
            f"Top1 {s['Top1']['mean']:.6f} ± {s['Top1']['std']:.6f} | "
            f"Top3 {s['Top3']['mean']:.6f} ± {s['Top3']['std']:.6f} | "
            f"Top5 {s['Top5']['mean']:.6f} ± {s['Top5']['std']:.6f} | "
            f"same-task-error {s['same_task_error']['mean']:.4f}"
        )

    best_alpha = max(
        alphas,
        key=lambda a: summary[str(a)]["MRR"]["mean"],
    )

    print()
    print(
        f"BEST_ALPHA_BY_MRR={best_alpha:.2f} | "
        f"MRR={summary[str(best_alpha)]['MRR']['mean']:.6f}"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_alpha_by_mrr": best_alpha,
                "summary": summary,
                "rows": all_rows,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
