import gc
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path("/home/dhqg/vlia-uet")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from lerobot.configs import PreTrainedConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.collate import lerobot_collate_fn

from policies.smolvla.configuration_smolvla import VLIASmolVLAConfig
from policies.smolvla.modeling_smolvla import VLIASmolVLAPolicy
from vlia_data.libero_oracle_dataset import LiberoOracleSubset


PRETRAINED = "lerobot/smolvla_base"

RENAME_MAP = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.image2": "observation.images.camera2",
}


def clone_vlia_config(base_cfg):
    kwargs = {}

    for field in base_cfg.__dataclass_fields__.values():
        if field.init:
            kwargs[field.name] = getattr(base_cfg, field.name)

    return VLIASmolVLAConfig(
        **kwargs,
        use_intention_token=True,
        intention_dim=960,
    )


def build_preprocessor(cfg, stats):
    overrides = {
        "device_processor": {
            "device": cfg.device,
        },
        "normalizer_processor": {
            "stats": stats,
            "features": {
                **cfg.input_features,
                **cfg.output_features,
            },
            "norm_map": cfg.normalization_mapping,
        },
        "rename_observations_processor": {
            "rename_map": RENAME_MAP,
        },
    }

    preprocessor, _ = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=PRETRAINED,
        dataset_stats=stats,
        preprocessor_overrides=overrides,
    )

    return preprocessor


def prepare_raw_batch(batch, camera_keys):
    for key in camera_keys:
        if key in batch and batch[key].dtype == torch.uint8:
            batch[key] = batch[key].float() / 255.0

    return batch


def main():
    torch.manual_seed(0)

    print("=" * 72)
    print("MATCHED BASELINE / ORACLE TRAIN STEP")
    print("=" * 72)

    print("\n=== CONFIG ===")

    cfg = PreTrainedConfig.from_pretrained(
        pretrained_name_or_path=PRETRAINED,
    )

    if not isinstance(cfg, SmolVLAConfig):
        raise TypeError(type(cfg))

    cfg.num_vlm_layers = 16
    cfg.resize_imgs_with_padding = (512, 512)
    cfg.chunk_size = 50
    cfg.n_action_steps = 50

    print("device:", cfg.device)
    print("chunk_size:", cfg.chunk_size)
    print("resize:", cfg.resize_imgs_with_padding)

    print("\n=== DATASET ===")

    meta = LeRobotDatasetMetadata(
        "lerobot/libero"
    )

    delta_timestamps = resolve_delta_timestamps(
        cfg,
        meta,
    )

    base = LeRobotDataset(
        "lerobot/libero",
        delta_timestamps=delta_timestamps,
        video_backend="torchcodec",
        return_uint8=True,
    )

    baseline_ds = LiberoOracleSubset(
        base_dataset=base,
        add_intention=False,
        semantic_action_padding=True,
    )

    oracle_ds = LiberoOracleSubset(
        base_dataset=base,
        add_intention=True,
        semantic_action_padding=True,
    )

    collate_fn = (
        lerobot_collate_fn
        if base.meta.has_language_columns
        else None
    )

    baseline_loader = DataLoader(
        baseline_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    oracle_loader = DataLoader(
        oracle_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    baseline_raw = next(iter(baseline_loader))
    oracle_raw = next(iter(oracle_loader))

    assert torch.equal(
        baseline_raw["action"],
        oracle_raw["action"],
    )

    assert torch.equal(
        baseline_raw["action_is_pad"],
        oracle_raw["action_is_pad"],
    )

    print("matched raw targets: PASS")

    print("\n=== PREPROCESS ===")

    preprocessor = build_preprocessor(
        cfg,
        base.meta.stats,
    )

    baseline_raw = prepare_raw_batch(
        baseline_raw,
        base.meta.camera_keys,
    )

    oracle_raw = prepare_raw_batch(
        oracle_raw,
        base.meta.camera_keys,
    )

    baseline_batch = preprocessor(
        baseline_raw
    )

    # Preserve custom semantic feature if the loaded
    # processor implementation ever drops unknown keys.
    if "intention" not in oracle_raw:
        raise RuntimeError(
            "Raw Oracle batch lost intention unexpectedly"
        )

    oracle_intention = oracle_raw["intention"]

    oracle_batch = preprocessor(
        oracle_raw
    )

    if "intention" not in oracle_batch:
        oracle_batch["intention"] = (
            oracle_intention.to(cfg.device)
        )

    assert "intention" not in baseline_batch
    assert oracle_batch["intention"].shape == (1, 960)

    assert torch.allclose(
        baseline_batch["action"],
        oracle_batch["action"],
    )

    assert torch.equal(
        baseline_batch["action_is_pad"],
        oracle_batch["action_is_pad"],
    )

    print("matched processed targets: PASS")

    print("\n=== BASELINE FORWARD / BACKWARD ===")

    baseline = SmolVLAPolicy.from_pretrained(
        PRETRAINED,
        config=cfg,
        strict=False,
    )

    baseline.train()

    baseline.zero_grad(
        set_to_none=True
    )

    torch.manual_seed(1234)

    baseline_loss, baseline_info = baseline(
        baseline_batch
    )

    print(
        "baseline loss:",
        float(
            baseline_loss.detach().cpu()
        ),
    )

    assert torch.isfinite(
        baseline_loss
    )

    baseline_loss.backward()

    baseline_grad = 0.0

    for p in baseline.parameters():
        if p.grad is not None:
            baseline_grad += float(
                p.grad.detach().norm().cpu()
            )

    print(
        "baseline grad sum:",
        baseline_grad,
    )

    assert baseline_grad > 0

    del baseline
    gc.collect()
    torch.cuda.empty_cache()

    print("\n=== ORACLE FORWARD / BACKWARD ===")

    vlia_cfg = clone_vlia_config(
        cfg
    )

    oracle = VLIASmolVLAPolicy.from_pretrained(
        PRETRAINED,
        config=vlia_cfg,
        strict=False,
    )

    oracle.train()

    oracle.zero_grad(
        set_to_none=True
    )

    torch.manual_seed(1234)

    oracle_loss, oracle_info = oracle(
        oracle_batch
    )

    print(
        "oracle loss:",
        float(
            oracle_loss.detach().cpu()
        ),
    )

    assert torch.isfinite(
        oracle_loss
    )

    oracle_loss.backward()

    adapter_grad = 0.0

    for p in (
        oracle.model.intention_adapter.parameters()
    ):
        if p.grad is not None:
            adapter_grad += float(
                p.grad.detach().norm().cpu()
            )

    print(
        "adapter grad sum:",
        adapter_grad,
    )

    assert adapter_grad > 0

    print("\n=== FINAL ===")

    print(
        "baseline loss:",
        float(
            baseline_loss.detach().cpu()
        ),
    )

    print(
        "oracle loss:",
        float(
            oracle_loss.detach().cpu()
        ),
    )

    print(
        "adapter grad sum:",
        adapter_grad,
    )

    print()
    print(
        "MATCHED TRAIN STEP: PASS"
    )


if __name__ == "__main__":
    main()