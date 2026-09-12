from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class StageAFusion(nn.Module):
    def __init__(self):
        super().__init__()

        self.fusion = nn.Sequential(
            nn.Linear(2880, 1024),
            nn.GELU(),
            nn.Linear(1024, 960),
            nn.LayerNorm(960),
        )

    def forward(self, x):
        return self.fusion(x)


def normalize_why(text):
    return " ".join(
        text.strip().lower().split()
    )


def build_positive_mask(whys, device):
    normalized = [
        normalize_why(w)
        for w in whys
    ]

    groups = defaultdict(list)

    for i, why in enumerate(normalized):
        groups[why].append(i)

    n = len(whys)

    mask = torch.zeros(
        n,
        n,
        dtype=torch.bool,
        device=device,
    )

    for indices in groups.values():
        idx = torch.tensor(
            indices,
            device=device,
        )

        mask[
            idx[:, None],
            idx[None, :],
        ] = True

    return mask


@torch.no_grad()
def multi_positive_metrics(
    pred,
    target,
    whys,
):
    pred = F.normalize(
        pred.float(),
        dim=-1,
    )

    target = F.normalize(
        target.float(),
        dim=-1,
    )

    sim = pred @ target.T

    n = sim.shape[0]

    positive_mask = build_positive_mask(
        whys,
        sim.device,
    )

    ranked = torch.argsort(
        sim,
        dim=1,
        descending=True,
    )

    ranked_positive = torch.gather(
        positive_mask,
        dim=1,
        index=ranked,
    )

    top1 = (
        ranked_positive[:, :1]
        .any(dim=1)
        .float()
        .mean()
    )

    top3 = (
        ranked_positive[:, :3]
        .any(dim=1)
        .float()
        .mean()
    )

    first_positive_rank = (
        ranked_positive.float()
        .argmax(dim=1)
        + 1
    )

    mrr = (
        1.0
        / first_positive_rank.float()
    ).mean()

    positive_similarity = (
        sim.masked_fill(
            ~positive_mask,
            float("-inf"),
        )
        .max(dim=1)
        .values
    )

    negative_similarity = (
        sim.masked_fill(
            positive_mask,
            float("-inf"),
        )
        .max(dim=1)
        .values
    )

    margin = (
        positive_similarity
        - negative_similarity
    ).mean()

    return {
        "top1": top1.item(),
        "top3": top3.item(),
        "mrr": mrr.item(),
        "positive_cosine": (
            positive_similarity
            .mean()
            .item()
        ),
        "margin": margin.item(),
    }


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(
            "/media/dhqg/d1/datasets/"
            "egointent/cache/stage_a_v0/"
            "val.pt"
        ),
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/media/dhqg/d1/vlia_outputs/"
            "stage_a_cached_v0/best.pt"
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    data = torch.load(
        args.cache,
        map_location="cpu",
        weights_only=False,
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    model = StageAFusion().to(device)

    model.load_state_dict(
        checkpoint["model"]
    )

    model.eval()

    x = torch.cat(
        [
            data["visual"],
            data["task"],
            data["history"],
        ],
        dim=-1,
    ).to(device)

    why_feature = (
        data["why"]
        .float()
        .to(device)
    )

    with torch.no_grad():
        pred = model(x)

    metrics = (
        multi_positive_metrics(
            pred,
            why_feature,
            data["whys"],
        )
    )

    print(
        "=== MULTI-POSITIVE "
        "STAGE-A EVAL ==="
    )

    print(
        "samples:",
        len(data["whys"]),
    )

    print(
        "checkpoint epoch:",
        checkpoint["epoch"],
    )

    for key, value in metrics.items():
        print(
            f"{key}: {value:.6f}"
        )


if __name__ == "__main__":
    main()