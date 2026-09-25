#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import train_egointent_softmrr as base


class FrozenResidualReranker(nn.Module):
    """
    Frozen best WHAT-hardneg guided model + small residual WHY corrector.

    The base model is never updated. The corrector starts at zero, so the
    initial model is exactly the frozen baseline.
    """

    def __init__(self, base_model, hidden_dim=256, residual_scale=0.20):
        super().__init__()
        self.base_model = base_model
        self.residual_scale = residual_scale

        for p in self.base_model.parameters():
            p.requires_grad = False

        # Input = base WHY (256) + predicted WHAT (256).
        self.corrector = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 256),
        )

        # Exact baseline initialization: final correction = 0.
        nn.init.zeros_(self.corrector[-1].weight)
        nn.init.zeros_(self.corrector[-1].bias)

    def train(self, mode=True):
        super().train(mode)
        # Keep frozen base deterministic/eval-like.
        self.base_model.eval()
        return self

    def forward(self, visual):
        with torch.no_grad():
            out = self.base_model.predict_intention_guided(visual)
            base_why = out["why"].detach()
            pred_what = out["what"].detach()
            base_gate = out["gate"].detach()

        x = torch.cat([base_why, pred_what], dim=-1)
        correction = self.corrector(x)

        final_why = F.normalize(
            base_why + self.residual_scale * correction,
            dim=-1,
        )

        return {
            "why": final_why,
            "base_why": base_why,
            "what": pred_what,
            "correction": correction,
            "gate": base_gate,
        }


def load_base_model(ckpt_path, args, device):
    ckpt = torch.load(ckpt_path, map_location=device)

    model = base.EgoIntentModel(
        hidden_dim=args.hidden_dim,
        visual_state_dim=args.visual_state_dim,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    return model, ckpt


@torch.no_grad()
def evaluate(model, visual, what_target, why_target, rows):
    model.eval()
    out = model(visual)

    why_metrics = base.retrieval_metrics(
        out["why"],
        why_target,
        [x["why"] for x in rows],
    )

    what_metrics = base.retrieval_metrics(
        out["what"],
        what_target,
        [x["what"] for x in rows],
    )

    # Same-task error analysis.
    sim = out["why"] @ why_target.T
    ranking = torch.argsort(sim, dim=1, descending=True)

    same_task_wrong = 0
    cross_task_wrong = 0

    for i, row in enumerate(rows):
        j = int(ranking[i, 0].item())
        pred_row = rows[j]

        if pred_row["why"] != row["why"]:
            if pred_row["task"] == row["task"]:
                same_task_wrong += 1
            else:
                cross_task_wrong += 1

    wrong = same_task_wrong + cross_task_wrong
    same_task_error = (
        same_task_wrong / wrong if wrong > 0 else 0.0
    )

    return {
        "why": why_metrics,
        "what": what_metrics,
        "same_task_error": same_task_error,
        "correction_norm": float(
            out["correction"].norm(dim=-1).mean().item()
        ),
        "gate_mean": float(out["gate"].mean().item()),
    }


def clone_state_dict(model):
    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
        if not k.startswith("base_model.")
    }


def train_one(
    seed,
    base_ckpt_path,
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

    frozen_base, base_meta = load_base_model(
        base_ckpt_path,
        args,
        device,
    )

    model = FrozenResidualReranker(
        base_model=frozen_base,
        hidden_dim=args.rerank_hidden_dim,
        residual_scale=args.residual_scale,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.corrector.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    why_mask = base.build_label_mask(
        [x["why"] for x in train_rows],
        device,
    )

    # Confirm exact-baseline initialization before any update.
    initial_eval = evaluate(
        model,
        val_visual,
        val_what_target,
        val_why_target,
        val_rows,
    )

    print(
        f"INIT | seed={seed} | "
        f"WHY MRR={initial_eval['why']['MRR']:.6f} | "
        f"Top1={initial_eval['why']['Top1']:.6f} | "
        f"correction_norm={initial_eval['correction_norm']:.6f}"
    )

    best_mrr = initial_eval["why"]["MRR"]
    best_epoch = 0
    best_eval = initial_eval
    best_state = clone_state_dict(model)
    history = [{
        "epoch": 0,
        "loss": None,
        "val": initial_eval,
    }]

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        out = model(train_visual)
        why_pred = out["why"]
        correction = out["correction"]

        softmrr = base.soft_mrr_ranking_loss(
            why_pred,
            train_why_target,
            why_mask,
            rank_temperature=args.rank_temperature,
            positive_temperature=args.rank_positive_temperature,
        )

        # Small stabilizer: retain alignment to GT WHY while the frozen
        # baseline remains the anchor.
        cos = base.cosine_alignment_loss(
            why_pred,
            train_why_target,
        )

        # Penalize unnecessarily large corrections.
        residual_reg = correction.pow(2).mean()

        loss = (
            args.softmrr_weight * softmrr
            + args.cosine_weight * cos
            + args.residual_reg_weight * residual_reg
        )

        loss.backward()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.corrector.parameters(),
                args.grad_clip,
            )

        optimizer.step()

        if (
            epoch == 1
            or epoch % args.eval_every == 0
            or epoch == args.epochs
        ):
            val_eval = evaluate(
                model,
                val_visual,
                val_what_target,
                val_why_target,
                val_rows,
            )

            history.append({
                "epoch": epoch,
                "loss": float(loss.item()),
                "softmrr": float(softmrr.item()),
                "cos": float(cos.item()),
                "residual_reg": float(residual_reg.item()),
                "val": val_eval,
            })

            print(
                f"[frozen_reranker] seed={seed} "
                f"epoch={epoch:03d}/{args.epochs} | "
                f"loss={loss.item():.6f} | "
                f"WHY MRR={val_eval['why']['MRR']:.6f} | "
                f"Top1={val_eval['why']['Top1']:.6f} | "
                f"Top3={val_eval['why']['Top3']:.6f} | "
                f"Top5={val_eval['why']['Top5']:.6f} | "
                f"corr_norm={val_eval['correction_norm']:.4f}"
            )

            if val_eval["why"]["MRR"] > best_mrr:
                best_mrr = val_eval["why"]["MRR"]
                best_epoch = epoch
                best_eval = val_eval
                best_state = clone_state_dict(model)

    return {
        "seed": seed,
        "base_ckpt": str(base_ckpt_path),
        "base_best_val": base_meta.get("best_val"),
        "best_epoch": best_epoch,
        "best_val": best_eval,
        "corrector_state_dict": best_state,
        "history": history,
    }


def mean_std(vals):
    x = np.asarray(vals, dtype=np.float64)
    return (
        float(x.mean()),
        float(x.std(ddof=1) if len(x) > 1 else 0.0),
    )


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--base-root",
        type=Path,
        default=Path(
            "/media/dhqg/d1/vlia_outputs/"
            "egointent_softmrr_v1"
        ),
        help=(
            "Root containing "
            "what_hardneg_guided/seed_<seed>/best.pt"
        ),
    )
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
            "egointent_frozen_reranker_v1"
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

    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--visual-state-dim", type=int, default=256)
    p.add_argument("--rerank-hidden-dim", type=int, default=256)

    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--residual-scale", type=float, default=0.20)
    p.add_argument("--softmrr-weight", type=float, default=1.0)
    p.add_argument("--cosine-weight", type=float, default=0.05)
    p.add_argument("--residual-reg-weight", type=float, default=1e-3)

    p.add_argument("--rank-temperature", type=float, default=0.05)
    p.add_argument(
        "--rank-positive-temperature",
        type=float,
        default=0.05,
    )

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

    train_visual, train_what_target, train_why_target = (
        base.stack_rows(train_rows, device)
    )
    val_visual, val_what_target, val_why_target = (
        base.stack_rows(val_rows, device)
    )

    print("=" * 96)
    print("EGOINTENT FROZEN-BASE RESIDUAL RERANKER")
    print("=" * 96)
    print("base_root:", args.base_root)
    print("seeds:", args.seeds)
    print("residual_scale:", args.residual_scale)
    print("softmrr_weight:", args.softmrr_weight)
    print("cosine_weight:", args.cosine_weight)
    print("residual_reg_weight:", args.residual_reg_weight)
    print("=" * 96)

    results = []

    for seed in args.seeds:
        base_ckpt = (
            args.base_root
            / "what_hardneg_guided"
            / f"seed_{seed}"
            / "best.pt"
        )
        if not base_ckpt.exists():
            raise FileNotFoundError(base_ckpt)

        result = train_one(
            seed=seed,
            base_ckpt_path=base_ckpt,
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

        seed_dir = args.output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "experiment": "egointent_frozen_reranker",
                "seed": seed,
                "base_ckpt": result["base_ckpt"],
                "best_epoch": result["best_epoch"],
                "best_val": result["best_val"],
                "corrector_state_dict": result["corrector_state_dict"],
                "config": {
                    **vars(args),
                    "base_root": str(args.base_root),
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
            json.dump(result["history"], f, indent=2)

        results.append(result)

        v = result["best_val"]
        print(
            f"BEST | seed={seed} | "
            f"epoch={result['best_epoch']} | "
            f"WHY MRR={v['why']['MRR']:.6f} | "
            f"Top1={v['why']['Top1']:.6f} | "
            f"Top3={v['why']['Top3']:.6f} | "
            f"Top5={v['why']['Top5']:.6f} | "
            f"WHAT MRR={v['what']['MRR']:.6f} | "
            f"same-task-error={v['same_task_error']:.4f} | "
            f"corr_norm={v['correction_norm']:.4f}"
        )

    fields = {
        "MRR": [r["best_val"]["why"]["MRR"] for r in results],
        "Top1": [r["best_val"]["why"]["Top1"] for r in results],
        "Top3": [r["best_val"]["why"]["Top3"] for r in results],
        "Top5": [r["best_val"]["why"]["Top5"] for r in results],
        "WHAT_MRR": [r["best_val"]["what"]["MRR"] for r in results],
        "same_task_error": [r["best_val"]["same_task_error"] for r in results],
        "correction_norm": [r["best_val"]["correction_norm"] for r in results],
    }

    summary = {}
    for key, vals in fields.items():
        m, s = mean_std(vals)
        summary[key] = {"mean": m, "std": s}

    print()
    print("=" * 96)
    print("FINAL COMPARISON")
    print("=" * 96)
    print(
        "frozen_reranker | "
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
        f"corr_norm "
        f"{summary['correction_norm']['mean']:.4f} ± "
        f"{summary['correction_norm']['std']:.4f}"
    )

    with open(
        args.output_dir / "summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "experiment": "egointent_frozen_reranker",
                "summary": summary,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
