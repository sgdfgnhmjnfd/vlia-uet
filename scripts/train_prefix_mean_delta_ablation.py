from pathlib import Path
import argparse
import json
import random
from collections import defaultdict

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
# PCA loader
# ============================================================

def load_pca_transform(path: Path):
    obj = torch.load(path, map_location="cpu")

    mean = None
    comp = None

    if torch.is_tensor(obj):
        comp = obj.float()

    elif isinstance(obj, dict):
        mean_keys = ["mean", "pca_mean", "mu", "center"]
        comp_keys = [
            "components",
            "components_",
            "pca_components",
            "projection",
            "proj",
            "V",
            "basis",
        ]

        for k in mean_keys:
            if k in obj and torch.is_tensor(obj[k]):
                mean = obj[k].float()
                break

        for k in comp_keys:
            if k in obj and torch.is_tensor(obj[k]):
                comp = obj[k].float()
                break

        if comp is None:
            for container_key in ["state_dict", "pca", "transform"]:
                nested = obj.get(container_key)
                if not isinstance(nested, dict):
                    continue

                for k in mean_keys:
                    if k in nested and torch.is_tensor(nested[k]):
                        mean = nested[k].float()
                        break

                for k in comp_keys:
                    if k in nested and torch.is_tensor(nested[k]):
                        comp = nested[k].float()
                        break

                if comp is not None:
                    break

    if comp is None:
        raise RuntimeError(
            f"Could not infer PCA components from {path}"
        )

    comp = comp.squeeze()

    if comp.shape == (960, 256):
        basis = comp
    elif comp.shape == (256, 960):
        basis = comp.T
    else:
        raise RuntimeError(
            f"Unexpected PCA component shape: {tuple(comp.shape)}"
        )

    if mean is None:
        mean = torch.zeros(960, dtype=torch.float32)

    mean = mean.squeeze().float()

    if mean.numel() != 960:
        raise RuntimeError(
            f"Unexpected PCA mean size: {mean.numel()}"
        )

    mean = mean.reshape(1, 960)
    basis = basis.reshape(960, 256).float()

    def transform(x):
        return (x.float().cpu() - mean) @ basis

    return transform


# ============================================================
# Data loading
# ============================================================

def load_stage_a_metadata(stage_a_root: Path, split: str):
    root = stage_a_root / split
    rows = {}

    for p in sorted(root.glob("*.pt")):
        x = torch.load(p, map_location="cpu")

        sample_id = str(x["sample_id"])

        rows[sample_id] = {
            "why": str(x["why"]),
            "why_feature": x["why_feature"].float(),
        }

    if not rows:
        raise RuntimeError(f"No Stage-A files found in {root}")

    return rows


def find_temporal_feature(x, path):
    for key in [
        "frame_features",
        "temporal_features",
        "visual_features",
        "frames_feature",
    ]:
        if key in x and torch.is_tensor(x[key]):
            feat = x[key].float()

            if feat.ndim == 2 and feat.shape[-1] == 960:
                return feat

    tensor_keys = {
        k: tuple(v.shape)
        for k, v in x.items()
        if torch.is_tensor(v)
    }

    raise RuntimeError(
        f"Could not find [T,960] feature in {path}. "
        f"Tensor keys: {tensor_keys}"
    )


def load_temporal_split(
    temporal_root: Path,
    stage_a_root: Path,
    split: str,
):
    temporal_dir = temporal_root / split
    metadata = load_stage_a_metadata(
        stage_a_root,
        split,
    )

    rows = []

    for p in sorted(temporal_dir.rglob("*.pt")):
        x = torch.load(p, map_location="cpu")

        sample_id = str(
            x.get("sample_id", p.stem)
        )

        if sample_id not in metadata:
            continue

        feat = find_temporal_feature(
            x,
            p,
        )

        rows.append(
            {
                "sample_id": sample_id,
                "frames": feat,
                "why": metadata[sample_id]["why"],
                "why_feature": metadata[sample_id]["why_feature"],
            }
        )

    if not rows:
        raise RuntimeError(
            f"No temporal samples matched in {temporal_dir}"
        )

    lengths = sorted(
        {
            int(r["frames"].shape[0])
            for r in rows
        }
    )

    if len(lengths) != 1:
        raise RuntimeError(
            f"Expected fixed sequence length, got {lengths}"
        )

    return rows


def stack_split(rows, why_transform, device):
    frames = torch.stack(
        [r["frames"] for r in rows]
    ).to(device)

    why_960 = torch.stack(
        [r["why_feature"] for r in rows]
    )

    why_256 = why_transform(
        why_960
    ).to(device)

    why_256 = F.normalize(
        why_256,
        dim=-1,
    )

    why_texts = [
        r["why"]
        for r in rows
    ]

    return frames, why_256, why_texts


# ============================================================
# Feature constructions
# ============================================================

def build_prefix_mean(frames, prefix_len):
    """
    frames: [B,T,960]
    Use only first prefix_len frames.
    """
    return frames[:, :prefix_len].mean(dim=1)


def build_mean_delta(frames):
    """
    Lightweight ordered temporal representation.

    mean:
        overall appearance / scene semantics

    delta:
        late-half mean - early-half mean

    This keeps coarse temporal direction without a recurrent model.
    """
    t = frames.shape[1]

    if t < 4:
        raise RuntimeError(
            "mean+delta requires at least 4 frames"
        )

    mid = t // 2

    overall = frames.mean(dim=1)
    early = frames[:, :mid].mean(dim=1)
    late = frames[:, mid:].mean(dim=1)

    delta = late - early

    return torch.cat(
        [overall, delta],
        dim=-1,
    )


# ============================================================
# Model
# ============================================================

class RetrievalMLP(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=512,
        output_dim=256,
        dropout=0.0,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_dim,
                output_dim,
            ),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x):
        z = self.net(x)

        return F.normalize(
            z,
            dim=-1,
        )


# ============================================================
# Loss
# ============================================================

def build_positive_mask(labels, device):
    n = len(labels)

    mask = torch.zeros(
        (n, n),
        dtype=torch.bool,
        device=device,
    )

    groups = defaultdict(list)

    for i, label in enumerate(labels):
        groups[label].append(i)

    for idxs in groups.values():
        idx = torch.tensor(
            idxs,
            dtype=torch.long,
            device=device,
        )

        mask[
            idx[:, None],
            idx[None, :],
        ] = True

    return mask


def multi_positive_infonce(
    pred,
    target,
    positive_mask,
    temperature,
):
    logits = (
        pred @ target.T
    ) / temperature

    log_prob = (
        logits
        - torch.logsumexp(
            logits,
            dim=1,
            keepdim=True,
        )
    )

    pos_count = (
        positive_mask
        .sum(dim=1)
        .clamp_min(1)
    )

    loss = -(
        log_prob
        * positive_mask.float()
    ).sum(dim=1) / pos_count

    return loss.mean()


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    features,
    why_target,
    why_texts,
):
    model.eval()

    pred = model(
        features
    )

    sim = pred @ why_target.T

    ranking = torch.argsort(
        sim,
        dim=1,
        descending=True,
    )

    ranks = []

    for i, query_why in enumerate(
        why_texts
    ):
        positive_set = {
            j
            for j, gallery_why
            in enumerate(why_texts)
            if gallery_why == query_why
        }

        first_rank = None

        for rank, j in enumerate(
            ranking[i].tolist(),
            start=1,
        ):
            if j in positive_set:
                first_rank = rank
                break

        ranks.append(
            first_rank
        )

    ranks = torch.tensor(
        ranks,
        dtype=torch.float32,
        device=features.device,
    )

    return {
        "Top1": (
            ranks <= 1
        ).float().mean().item(),
        "Top3": (
            ranks <= 3
        ).float().mean().item(),
        "Top5": (
            ranks <= 5
        ).float().mean().item(),
        "MRR": (
            1.0 / ranks
        ).mean().item(),
        "median_rank": (
            ranks.median().item()
        ),
        "mean_rank": (
            ranks.mean().item()
        ),
    }


# ============================================================
# Training one run
# ============================================================

def train_one(
    name,
    train_features,
    val_features,
    train_why_target,
    val_why_target,
    train_why_texts,
    val_why_texts,
    seed,
    args,
):
    set_seed(seed)

    input_dim = int(
        train_features.shape[-1]
    )

    model = RetrievalMLP(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=256,
        dropout=args.dropout,
    ).to(train_features.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    positive_mask = build_positive_mask(
        train_why_texts,
        train_features.device,
    )

    best_state = None
    best_epoch = -1
    best_mrr = -1.0
    history = []

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        pred = model(
            train_features
        )

        loss_nce = (
            multi_positive_infonce(
                pred,
                train_why_target,
                positive_mask,
                args.temperature,
            )
        )

        loss_cos = (
            1.0
            - (
                pred
                * train_why_target
            ).sum(dim=-1).mean()
        )

        loss = (
            loss_nce
            + args.cosine_weight
            * loss_cos
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
            metrics = evaluate(
                model,
                val_features,
                val_why_target,
                val_why_texts,
            )

            history.append(
                {
                    "epoch": epoch,
                    "loss": loss.item(),
                    "val": metrics,
                }
            )

            if metrics["MRR"] > best_mrr:
                best_mrr = metrics["MRR"]
                best_epoch = epoch

                best_state = {
                    k: (
                        v.detach()
                        .cpu()
                        .clone()
                    )
                    for k, v
                    in model.state_dict().items()
                }

    model.load_state_dict(
        best_state
    )

    final_metrics = evaluate(
        model,
        val_features,
        val_why_target,
        val_why_texts,
    )

    print(
        f"{name:18s} "
        f"seed={seed} "
        f"best_epoch={best_epoch:3d} "
        f"MRR={final_metrics['MRR']:.6f} "
        f"Top1={final_metrics['Top1']:.6f}"
    )

    return {
        "name": name,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val": final_metrics,
        "history": history,
        "state_dict": best_state,
    }


# ============================================================
# Summary helper
# ============================================================

def mean_std(values):
    arr = np.asarray(
        values,
        dtype=np.float64,
    )

    return (
        float(arr.mean()),
        float(
            arr.std(ddof=1)
            if len(arr) > 1
            else 0.0
        ),
    )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--temporal-cache",
        type=Path,
        default=Path(
            "/media/dhqg/d1/datasets/"
            "egointent/cache/"
            "temporal_intention_v1"
        ),
    )

    parser.add_argument(
        "--stage-a-cache",
        type=Path,
        default=Path(
            "/media/dhqg/d1/datasets/"
            "egointent/cache/"
            "stage_a_v0"
        ),
    )

    parser.add_argument(
        "--why-pca",
        type=Path,
        default=Path(
            "/media/dhqg/d1/vlia_outputs/"
            "stage_a_clean_v1/"
            "why_pca_960_to_256.pt"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/media/dhqg/d1/vlia_outputs/"
            "prefix_mean_delta_v1"
        ),
    )

    parser.add_argument(
        "--prefixes",
        nargs="+",
        type=int,
        default=[
            2,
            4,
            6,
            8,
        ],
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[
            0,
            1,
            2,
        ],
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--eval-every",
        type=int,
        default=1,
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
        "--cosine-weight",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=512,
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

    device = torch.device(
        args.device
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    why_transform = load_pca_transform(
        args.why_pca
    )

    train_rows = load_temporal_split(
        args.temporal_cache,
        args.stage_a_cache,
        "train",
    )

    val_rows = load_temporal_split(
        args.temporal_cache,
        args.stage_a_cache,
        "val",
    )

    (
        train_frames,
        train_why_target,
        train_why_texts,
    ) = stack_split(
        train_rows,
        why_transform,
        device,
    )

    (
        val_frames,
        val_why_target,
        val_why_texts,
    ) = stack_split(
        val_rows,
        why_transform,
        device,
    )

    seq_len = int(
        train_frames.shape[1]
    )

    prefixes = [
        p
        for p in args.prefixes
        if 1 <= p <= seq_len
    ]

    if seq_len not in prefixes:
        prefixes.append(
            seq_len
        )

    prefixes = sorted(
        set(prefixes)
    )

    print("=" * 80)
    print("PREFIX MEAN vs MEAN+DELTA")
    print("=" * 80)
    print(
        "train:",
        tuple(train_frames.shape),
    )
    print(
        "val:",
        tuple(val_frames.shape),
    )
    print(
        "prefixes:",
        prefixes,
    )
    print(
        "seeds:",
        args.seeds,
    )

    experiments = {}

    for p in prefixes:
        name = f"mean_{p}f"

        experiments[name] = {
            "train": build_prefix_mean(
                train_frames,
                p,
            ),
            "val": build_prefix_mean(
                val_frames,
                p,
            ),
        }

    experiments["mean_delta_8f"] = {
        "train": build_mean_delta(
            train_frames
        ),
        "val": build_mean_delta(
            val_frames
        ),
    }

    all_results = []

    for name, feat in experiments.items():
        print()
        print("#" * 80)
        print(
            f"EXPERIMENT: {name}"
        )
        print("#" * 80)

        exp_dir = (
            args.output_dir
            / name
        )

        exp_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for seed in args.seeds:
            result = train_one(
                name=name,
                train_features=feat["train"],
                val_features=feat["val"],
                train_why_target=train_why_target,
                val_why_target=val_why_target,
                train_why_texts=train_why_texts,
                val_why_texts=val_why_texts,
                seed=seed,
                args=args,
            )

            seed_dir = (
                exp_dir
                / f"seed_{seed}"
            )

            seed_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            torch.save(
                {
                    "name": name,
                    "seed": seed,
                    "best_epoch": (
                        result[
                            "best_epoch"
                        ]
                    ),
                    "best_val": (
                        result[
                            "best_val"
                        ]
                    ),
                    "model_state_dict": (
                        result[
                            "state_dict"
                        ]
                    ),
                    "config": vars(args),
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

            all_results.append(
                {
                    "name": name,
                    "seed": seed,
                    "best_epoch": (
                        result[
                            "best_epoch"
                        ]
                    ),
                    "best_val": (
                        result[
                            "best_val"
                        ]
                    ),
                }
            )

    summary = {}

    for name in experiments:
        rows = [
            x
            for x in all_results
            if x["name"] == name
        ]

        mrr_values = [
            x["best_val"]["MRR"]
            for x in rows
        ]

        top1_values = [
            x["best_val"]["Top1"]
            for x in rows
        ]

        top3_values = [
            x["best_val"]["Top3"]
            for x in rows
        ]

        mrr_m, mrr_s = mean_std(
            mrr_values
        )

        top1_m, top1_s = mean_std(
            top1_values
        )

        top3_m, top3_s = mean_std(
            top3_values
        )

        summary[name] = {
            "mrr_values": mrr_values,
            "mrr_mean": mrr_m,
            "mrr_sample_std": mrr_s,
            "top1_values": top1_values,
            "top1_mean": top1_m,
            "top1_sample_std": top1_s,
            "top3_values": top3_values,
            "top3_mean": top3_m,
            "top3_sample_std": top3_s,
        }

    with open(
        args.output_dir / "summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "config": vars(args),
                "summary": summary,
                "results": all_results,
            },
            f,
            indent=2,
            default=str,
        )

    print()
    print("=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)

    for name in experiments:
        row = summary[name]

        print(
            f"{name:18s} | "
            f"MRR "
            f"{row['mrr_mean']:.6f} ± "
            f"{row['mrr_sample_std']:.6f} | "
            f"Top1 "
            f"{row['top1_mean']:.6f} ± "
            f"{row['top1_sample_std']:.6f} | "
            f"Top3 "
            f"{row['top3_mean']:.6f} ± "
            f"{row['top3_sample_std']:.6f}"
        )

    print()
    print(
        "Saved summary:",
        args.output_dir / "summary.json",
    )


if __name__ == "__main__":
    main()
