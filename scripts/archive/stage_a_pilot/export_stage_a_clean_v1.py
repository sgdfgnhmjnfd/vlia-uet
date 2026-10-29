import json
from pathlib import Path

import torch


CHECKPOINT = Path(
    "/media/dhqg/d1/vlia_outputs/"
    "stage_a_clean_v1/best.pt"
)

OUTPUT = Path(
    "/media/dhqg/d1/vlia_outputs/"
    "stage_a_clean_v1/intention_encoder_v1.pt"
)


checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=False,
)

state = checkpoint["model"]

required_prefixes = (
    "fusion.",
    "intention_projection.",
)

export_state = {
    key: value
    for key, value in state.items()
    if key.startswith(required_prefixes)
}

assert export_state, "No Stage-A head weights found."

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

artifact = {
    "state_dict": export_state,
    "architecture": {
        "predictor_input": "visual_only",
        "visual_feature_dim": 960,
        "alignment_dim": 960,
        "intention_dim": 256,
        "fusion": "2880->1024->960",
    },
    "training": {
        "target": "train_only_pca_why_960_to_256",
        "objective": (
            "multi_positive_infonce"
            "+0.1_cosine"
        ),
        "checkpoint_selection": "val_mrr",
        "best_epoch": checkpoint["epoch"],
        "val_metrics": checkpoint["val_metrics"],
    },
    "note": (
        "PCA is used only to construct Stage-A supervision. "
        "It is not required during Predicted VLIA inference."
    ),
}

torch.save(
    artifact,
    OUTPUT,
)

print("saved:", OUTPUT)
print("weights:")
for key in export_state:
    print(" ", key)

print()
print("best epoch:", checkpoint["epoch"])
print(
    "val metrics:",
    checkpoint["val_metrics"],
)