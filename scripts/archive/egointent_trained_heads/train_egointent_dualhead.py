#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import train_egointent_softmrr as base


class DualHeadEgoIntent(base.EgoIntentModel):
    """
    Shared visual + WHAT pathway, two separate WHY branches:

      Head A: standard guided WHY (InfoNCE + cosine)
      Head B: ranking WHY (InfoNCE + cosine + Soft-MRR)

    Evaluation fuses their similarity scores with a fixed alpha.
    """

    def __init__(self, hidden_dim=512, visual_state_dim=256):
        super().__init__(
            hidden_dim=hidden_dim,
            visual_state_dim=visual_state_dim,
        )

        self.rank_why_head = base.ProjectionHead(
            hidden_dim=hidden_dim,
            output_dim=256,
        )

        fusion_dim = visual_state_dim + 256

        self.rank_reasoner = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
        )

        self.rank_gate = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.Sigmoid(),
        )

    def predict_dual(self, visual):
        predicted_what = self.predict_what(visual)
        visual_state = self.visual_state(visual)

        fused = torch.cat(
            [visual_state, predicted_what],
            dim=-1,
        )

        # Head A: original best guided pathway.
        direct_a = self.predict_direct_why(visual)
        residual_a = F.normalize(
            self.reasoner(fused),
            dim=-1,
        )
        gate_a = self.gate(fused)
        why_a = F.normalize(
            direct_a + gate_a * residual_a,
            dim=-1,
        )

        # Head B: separate ranking-oriented pathway.
        direct_b = self.rank_why_head(visual)
        residual_b = F.normalize(
            self.rank_reasoner(fused),
            dim=-1,
        )
        gate_b = self.rank_gate(fused)
        why_b = F.normalize(
            direct_b + gate_b * residual_b,
            dim=-1,
        )

        return {
            "what": predicted_what,
            "why_hard": why_a,
            "why_rank": why_b,
            "gate_hard": gate_a,
            "gate_rank": gate_b,
        }


@torch.no_grad()
def metrics_from_similarity(sim, labels):
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
                f"No positive retrieval target for sample {i}"
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
def error_breakdown_from_similarity(sim, rows):
    ranking = torch.argsort(
        sim,
        dim=1,
        descending=True,
    )

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
def evaluate(
    model,
    visual,
    what_target,
    why_target,
    rows,
    fusion_alpha,
):
    model.eval()

    out = model.predict_dual(visual)

    what_metrics = base.retrieval_metrics(
        out["what"],
        what_target,
        [x["what"] for x in rows],
    )

    sim_hard = out["why_hard"] @ why_target.T
    sim_rank = out["why_rank"] @ why_target.T

    sim = (
        (1.0 - fusion_alpha) * sim_hard
        + fusion_alpha * sim_rank
    )

    why_metrics = metrics_from_similarity(
        sim,
        [x["why"] for x in rows],
    )
    why_metrics.update(
        error_breakdown_from_similarity(
            sim,
            rows,
        )
    )

    return {
        "what": what_metrics,
        "why": why_metrics,
        "gate_mean": float(
            out["gate_hard"].mean().item()
        ),
        "rank_gate_mean": float(
            out["gate_rank"].mean().item()
        ),
    }


def clone_state_dict(model):
    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }


def train_one(
    seed,
    train_visual,
    train_what_target,
    train_why_target,
    train_rows,
    val_visual,
    val_what_target,
    val_why_target,
    val_rows,
    args,
):
    base.set_seed(seed)
    device = train_visual.device

    model = DualHeadEgoIntent(
        hidden_dim=args.hidden_dim,
        visual_state_dim=args.visual_state_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    why_mask = base.build_label_mask(
        [x["why"] for x in train_rows],
        device,
    )
    what_mask = base.build_label_mask(
        [x["what"] for x in train_rows],
        device,
    )
    task_mask = base.build_label_mask(
        [x["task"] for x in train_rows],
        device,
    )

    best_mrr = -1.0
    best_epoch = -1
    best_eval = None
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        out = model.predict_dual(
            train_visual
        )

        what_pred = out["what"]
        why_hard = out["why_hard"]
        why_rank = out["why_rank"]

        # Shared WHAT objective: retain the best WHAT hard-negative setup.
        what_nce = base.multi_positive_infonce(
            what_pred,
            train_what_target,
            what_mask,
            args.temperature,
        )
        what_cos = base.cosine_alignment_loss(
            what_pred,
            train_what_target,
        )
        what_hard = base.same_task_hard_negative_loss(
            what_pred,
            train_what_target,
            what_mask,
            task_mask,
            margin=args.what_hard_margin,
        )
        what_loss = (
            what_nce
            + args.cosine_weight * what_cos
            + args.what_hard_weight * what_hard
        )

        # Head A: preserve original hard-neg-guided WHY training.
        hard_nce = base.multi_positive_infonce(
            why_hard,
            train_why_target,
            why_mask,
            args.temperature,
        )
        hard_cos = base.cosine_alignment_loss(
            why_hard,
            train_why_target,
        )
        hard_loss = (
            hard_nce
            + args.cosine_weight * hard_cos
        )

        # Head B: separate listwise-ranking branch.
        rank_nce = base.multi_positive_infonce(
            why_rank,
            train_why_target,
            why_mask,
            args.temperature,
        )
        rank_cos = base.cosine_alignment_loss(
            why_rank,
            train_why_target,
        )
        rank_softmrr = base.soft_mrr_ranking_loss(
            why_rank,
            train_why_target,
            why_mask,
            rank_temperature=args.rank_temperature,
            positive_temperature=args.rank_positive_temperature,
        )

        rank_loss = (
            rank_nce
            + args.cosine_weight * rank_cos
            + args.softmrr_weight * rank_softmrr
        )

        total_loss = (
            hard_loss
            + args.what_weight * what_loss
            + args.rank_head_weight * rank_loss
        )

        total_loss.backward()

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
            val_eval = evaluate(
                model=model,
                visual=val_visual,
                what_target=val_what_target,
                why_target=val_why_target,
                rows=val_rows,
                fusion_alpha=args.fusion_alpha,
            )

            mrr = val_eval["why"]["MRR"]

            history.append(
                {
                    "epoch": epoch,
                    "loss_total": float(total_loss.item()),
                    "loss_hard_head": float(hard_loss.item()),
                    "loss_rank_head": float(rank_loss.item()),
                    "loss_rank_softmrr": float(rank_softmrr.item()),
                    "loss_what": float(what_loss.item()),
                    "val": val_eval,
                }
            )

            print(
                f"[dualhead] seed={seed} "
                f"epoch={epoch:03d}/{args.epochs} | "
                f"loss={total_loss.item():.6f} | "
                f"WHY MRR={mrr:.6f} | "
                f"Top1={val_eval['why']['Top1']:.6f} | "
                f"Top3={val_eval['why']['Top3']:.6f} | "
                f"Top5={val_eval['why']['Top5']:.6f} | "
                f"WHAT MRR={val_eval['what']['MRR']:.6f} | "
                f"gateA={val_eval['gate_mean']:.4f} | "
                f"gateB={val_eval['rank_gate_mean']:.4f}"
            )

            if mrr > best_mrr:
                best_mrr = mrr
                best_epoch = epoch
                best_eval = val_eval
                best_state = clone_state_dict(
                    model
                )

    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val": best_eval,
        "model_state_dict": best_state,
        "history": history,
    }


def mean_std(values):
    x = np.asarray(
        values,
        dtype=np.float64,
    )
    return (
        float(x.mean()),
        float(
            x.std(ddof=1)
            if len(x) > 1
            else 0.0
        ),
    )


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--stage-a-cache",
        type=Path,
        default=Path(
            "/media/dhqg/d1/datasets/"
            "egointent/cache/stage_a_v0"
        ),
    )
    p.add_argument(
        "--temporal-cache",
        type=Path,
        default=Path(
            "/media/dhqg/d1/datasets/"
            "egointent/cache/temporal_intention_v1"
        ),
    )
    p.add_argument(
        "--what-pca",
        type=Path,
        default=Path(
            "/media/dhqg/d1/vlia_outputs/"
            "structured_intention_v1/"
            "what_pca_960_to_256.pt"
        ),
    )
    p.add_argument(
        "--why-pca",
        type=Path,
        default=Path(
            "/media/dhqg/d1/vlia_outputs/"
            "stage_a_clean_v1/"
            "why_pca_960_to_256.pt"
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/media/dhqg/d1/vlia_outputs/"
            "egointent_dualhead_v1"
        ),
    )

    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42,123,456],
    )
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--cosine-weight", type=float, default=0.1)
    p.add_argument("--what-weight", type=float, default=0.3)
    p.add_argument("--what-hard-weight", type=float, default=0.30)
    p.add_argument("--what-hard-margin", type=float, default=0.10)

    # Use the Soft-MRR setting that gave the useful complementary head.
    p.add_argument("--softmrr-weight", type=float, default=0.10)
    p.add_argument("--rank-temperature", type=float, default=0.05)
    p.add_argument(
        "--rank-positive-temperature",
        type=float,
        default=0.05,
    )

    # External checkpoint fusion found alpha=0.30 best.
    p.add_argument(
        "--fusion-alpha",
        type=float,
        default=0.30,
    )
    p.add_argument(
        "--rank-head-weight",
        type=float,
        default=1.0,
    )

    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--visual-state-dim", type=int, default=256)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument(
        "--device",
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )

    args = p.parse_args()
    device = torch.device(args.device)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    what_transform = base.load_pca_transform(
        args.what_pca
    )
    why_transform = base.load_pca_transform(
        args.why_pca
    )

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

    train_visual, train_what_target, train_why_target = (
        base.stack_rows(
            train_rows,
            device,
        )
    )
    val_visual, val_what_target, val_why_target = (
        base.stack_rows(
            val_rows,
            device,
        )
    )

    print("=" * 96)
    print("EGOINTENT DUAL-HEAD HARDNEG + SOFTMRR")
    print("=" * 96)
    print("device:", device)
    print("seeds:", args.seeds)
    print("fusion_alpha:", args.fusion_alpha)
    print("rank_head_weight:", args.rank_head_weight)
    print("softmrr_weight:", args.softmrr_weight)
    print("=" * 96)

    results = []

    for seed in args.seeds:
        result = train_one(
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

        seed_dir = (
            args.output_dir
            / f"seed_{seed}"
        )
        seed_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            {
                "experiment": "egointent_dualhead",
                "seed": seed,
                "best_epoch": result["best_epoch"],
                "best_val": result["best_val"],
                "model_state_dict": result["model_state_dict"],
                "config": {
                    **vars(args),
                    "stage_a_cache": str(args.stage_a_cache),
                    "temporal_cache": str(args.temporal_cache),
                    "what_pca": str(args.what_pca),
                    "why_pca": str(args.why_pca),
                    "output_dir": str(args.output_dir),
                },
            },
            seed_dir / "best.pt",
        )

        with open(
            seed_dir / "history.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                result["history"],
                f,
                indent=2,
            )

        results.append(result)
        v = result["best_val"]

        print(
            f"BEST | seed={seed} | "
            f"epoch={result['best_epoch']} | "
            f"WHY MRR={v['why']['MRR']:.6f} | "
            f"WHY Top1={v['why']['Top1']:.6f} | "
            f"WHY Top3={v['why']['Top3']:.6f} | "
            f"WHY Top5={v['why']['Top5']:.6f} | "
            f"WHAT MRR={v['what']['MRR']:.6f} | "
            f"same-task-error="
            f"{v['why']['same_task_wrong_fraction']:.4f} | "
            f"gateA={v['gate_mean']:.4f} | "
            f"gateB={v['rank_gate_mean']:.4f}"
        )

    fields = {
        "MRR": [
            r["best_val"]["why"]["MRR"]
            for r in results
        ],
        "Top1": [
            r["best_val"]["why"]["Top1"]
            for r in results
        ],
        "Top3": [
            r["best_val"]["why"]["Top3"]
            for r in results
        ],
        "Top5": [
            r["best_val"]["why"]["Top5"]
            for r in results
        ],
        "WHAT_MRR": [
            r["best_val"]["what"]["MRR"]
            for r in results
        ],
        "same_task_error": [
            r["best_val"]["why"]["same_task_wrong_fraction"]
            for r in results
        ],
        "gateA": [
            r["best_val"]["gate_mean"]
            for r in results
        ],
        "gateB": [
            r["best_val"]["rank_gate_mean"]
            for r in results
        ],
    }

    summary = {}
    for key, vals in fields.items():
        m, s = mean_std(vals)
        summary[key] = {
            "mean": m,
            "std": s,
        }

    print()
    print("=" * 96)
    print("FINAL COMPARISON")
    print("=" * 96)
    print(
        "dualhead_guided | "
        f"WHY MRR {summary['MRR']['mean']:.6f} ± "
        f"{summary['MRR']['std']:.6f} | "
        f"WHY Top1 {summary['Top1']['mean']:.6f} ± "
        f"{summary['Top1']['std']:.6f} | "
        f"WHY Top3 {summary['Top3']['mean']:.6f} ± "
        f"{summary['Top3']['std']:.6f} | "
        f"WHY Top5 {summary['Top5']['mean']:.6f} ± "
        f"{summary['Top5']['std']:.6f} | "
        f"WHAT MRR {summary['WHAT_MRR']['mean']:.6f} ± "
        f"{summary['WHAT_MRR']['std']:.6f} | "
        f"same-task-error "
        f"{summary['same_task_error']['mean']:.4f} ± "
        f"{summary['same_task_error']['std']:.4f} | "
        f"gateA {summary['gateA']['mean']:.4f} ± "
        f"{summary['gateA']['std']:.4f} | "
        f"gateB {summary['gateB']['mean']:.4f} ± "
        f"{summary['gateB']['std']:.4f}"
    )

    with open(
        args.output_dir / "summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "experiment": "egointent_dualhead",
                "summary": summary,
                "fusion_alpha": args.fusion_alpha,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
