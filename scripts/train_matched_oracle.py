from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


from lerobot.configs import PreTrainedConfig
from lerobot.datasets.dataset_metadata import (
    LeRobotDatasetMetadata,
)
from lerobot.datasets.factory import (
    resolve_delta_timestamps,
)
from lerobot.datasets.lerobot_dataset import (
    LeRobotDataset,
)
from lerobot.policies.factory import (
    make_pre_post_processors,
)
from lerobot.policies.smolvla.configuration_smolvla import (
    SmolVLAConfig,
)
from lerobot.policies.smolvla.modeling_smolvla import (
    SmolVLAPolicy,
)
from lerobot.utils.collate import (
    lerobot_collate_fn,
)

from policies.smolvla.configuration_smolvla import (
    VLIASmolVLAConfig,
)
from policies.smolvla.modeling_smolvla import (
    VLIASmolVLAPolicy,
)
from vlia_data.libero_oracle_dataset import (
    LiberoOracleSubset,
)


PRETRAINED = "lerobot/smolvla_base"

RENAME_MAP = {
    "observation.images.image":
        "observation.images.camera1",
    "observation.images.image2":
        "observation.images.camera2",
}

DEFAULT_OUTPUT_ROOT = Path(
    "/media/dhqg/d1/vlia_outputs/"
    "oracle_matched"
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--condition",
        required=True,
        choices=[
            "baseline",
            "oracle",
        ],
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=25000,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--min-lr",
        type=float,
        default=2.5e-6,
    )

    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--save-freq",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--log-freq",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clone_vlia_config(base_cfg):
    kwargs = {}

    for field in (
        base_cfg.__dataclass_fields__.values()
    ):
        if field.init:
            kwargs[field.name] = getattr(
                base_cfg,
                field.name,
            )

    return VLIASmolVLAConfig(
        **kwargs,
        use_intention_token=True,
        intention_dim=960,
    )


def build_base_config():
    cfg = PreTrainedConfig.from_pretrained(
        pretrained_name_or_path=PRETRAINED,
    )

    if not isinstance(
        cfg,
        SmolVLAConfig,
    ):
        raise TypeError(
            f"Expected SmolVLAConfig, "
            f"got {type(cfg)}"
        )

    # Freeze matched experiment protocol.
    cfg.num_vlm_layers = 16
    cfg.resize_imgs_with_padding = (
        512,
        512,
    )
    cfg.chunk_size = 50
    cfg.n_action_steps = 50

    return cfg


def build_preprocessor(
    cfg,
    stats,
):
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
            "norm_map":
                cfg.normalization_mapping,
        },
        "rename_observations_processor": {
            "rename_map": RENAME_MAP,
        },
    }

    preprocessor, _ = (
        make_pre_post_processors(
            policy_cfg=cfg,
            pretrained_path=PRETRAINED,
            dataset_stats=stats,
            preprocessor_overrides=overrides,
        )
    )

    return preprocessor


def prepare_raw_batch(
    batch,
    camera_keys,
):
    for key in camera_keys:
        if (
            key in batch
            and batch[key].dtype
            == torch.uint8
        ):
            batch[key] = (
                batch[key].float()
                / 255.0
            )

    return batch


def preserve_oracle_intention(
    raw_batch,
    processed_batch,
    device,
):
    if "intention" not in raw_batch:
        raise RuntimeError(
            "Oracle raw batch does not "
            "contain intention."
        )

    if "intention" not in processed_batch:
        processed_batch[
            "intention"
        ] = raw_batch[
            "intention"
        ].to(device)

    return processed_batch


def lr_scale(
    step,
    total_steps,
    warmup_steps,
    min_lr_ratio,
):
    """
    Linear warmup followed by cosine decay.
    """

    if (
        warmup_steps > 0
        and step < warmup_steps
    ):
        return max(
            1e-8,
            float(step + 1)
            / float(warmup_steps),
        )

    if total_steps <= warmup_steps:
        return 1.0

    progress = (
        step - warmup_steps
    ) / (
        total_steps - warmup_steps
    )

    progress = min(
        max(progress, 0.0),
        1.0,
    )

    cosine = (
        0.5
        * (
            1.0
            + math.cos(
                math.pi * progress
            )
        )
    )

    return (
        min_lr_ratio
        + (
            1.0
            - min_lr_ratio
        )
        * cosine
    )


def save_checkpoint(
    output_dir,
    step,
    policy,
    optimizer,
    scheduler,
    args,
    running_info,
):
    ckpt_dir = (
        output_dir
        / f"checkpoint_{step:06d}"
    )

    ckpt_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Hugging Face / LeRobot style weights.
    policy.save_pretrained(
        ckpt_dir
        / "pretrained_model"
    )

    torch.save(
        {
            "step": step,
            "optimizer":
                optimizer.state_dict(),
            "scheduler":
                scheduler.state_dict(),
            "args": vars(args),
            "running_info":
                running_info,
        },
        ckpt_dir
        / "training_state.pt",
    )

    print(
        f"[checkpoint] {ckpt_dir}"
    )


def main():
    args = parse_args()

    if args.dry_run:
        args.steps = 2
        args.batch_size = min(
            args.batch_size,
            2,
        )
        args.num_workers = 0
        args.save_freq = 1
        args.log_freq = 1

    set_seed(
        args.seed
    )

    output_dir = (
        args.output_root
        / args.condition
        / f"seed_{args.seed}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print(
        "MATCHED SMOLVLA / ORACLE VLIA TRAINING"
    )
    print("=" * 72)
    print(
        "condition:",
        args.condition,
    )
    print(
        "output:",
        output_dir,
    )
    print(
        "seed:",
        args.seed,
    )
    print(
        "steps:",
        args.steps,
    )
    print(
        "batch_size:",
        args.batch_size,
    )
    print(
        "lr:",
        args.lr,
    )
    print(
        "warmup_steps:",
        args.warmup_steps,
    )
    print()

    # ---------------------------------------------------------
    # Config
    # ---------------------------------------------------------

    base_cfg = build_base_config()

    if args.condition == "baseline":
        policy_cfg = base_cfg
    else:
        policy_cfg = clone_vlia_config(
            base_cfg
        )

    print(
        "device:",
        policy_cfg.device,
    )
    print(
        "num_vlm_layers:",
        policy_cfg.num_vlm_layers,
    )
    print(
        "resize:",
        policy_cfg.resize_imgs_with_padding,
    )
    print(
        "chunk_size:",
        policy_cfg.chunk_size,
    )
    print()

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    print(
        "Building matched LIBERO-10 "
        "canonical dataset..."
    )

    meta = LeRobotDatasetMetadata(
        "lerobot/libero"
    )

    delta_timestamps = (
        resolve_delta_timestamps(
            base_cfg,
            meta,
        )
    )

    base_dataset = LeRobotDataset(
        "lerobot/libero",
        delta_timestamps=(
            delta_timestamps
        ),
        video_backend="torchcodec",
        return_uint8=True,
    )

    dataset = LiberoOracleSubset(
        base_dataset=base_dataset,
        add_intention=(
            args.condition
            == "oracle"
        ),
        semantic_action_padding=True,
    )

    if len(dataset) != 88302:
        raise RuntimeError(
            f"Expected 88302 canonical "
            f"frames, got {len(dataset)}"
        )

    if dataset.num_episodes != 335:
        raise RuntimeError(
            f"Expected 335 canonical "
            f"episodes, got "
            f"{dataset.num_episodes}"
        )

    print(
        "canonical frames:",
        len(dataset),
    )
    print(
        "canonical episodes:",
        dataset.num_episodes,
    )

    collate_fn = (
        lerobot_collate_fn
        if base_dataset.meta
        .has_language_columns
        else None
    )

    # Separate generator ensures sample order is
    # determined only by --seed.
    loader_generator = (
        torch.Generator()
    )
    loader_generator.manual_seed(
        args.seed
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=(
            args.num_workers
        ),
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=True,
        persistent_workers=(
            args.num_workers > 0
        ),
        generator=loader_generator,
    )

    # ---------------------------------------------------------
    # Preprocessor
    # ---------------------------------------------------------

    preprocessor = (
        build_preprocessor(
            policy_cfg,
            base_dataset.meta.stats,
        )
    )

    # ---------------------------------------------------------
    # Policy
    # ---------------------------------------------------------

    print()
    print(
        "Loading pretrained policy..."
    )

    # Reset seed here deliberately. This makes
    # shared pretrained initialization independent
    # of dataset/DataLoader construction.
    set_seed(
        args.seed
    )

    if args.condition == "baseline":
        policy = (
            SmolVLAPolicy
            .from_pretrained(
                PRETRAINED,
                config=policy_cfg,
                strict=False,
            )
        )
    else:
        policy = (
            VLIASmolVLAPolicy
            .from_pretrained(
                PRETRAINED,
                config=policy_cfg,
                strict=False,
            )
        )

    policy.train()

    # ---------------------------------------------------------
    # Optimizer / scheduler
    # ---------------------------------------------------------

    trainable_params = [
        p
        for p in policy.parameters()
        if p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=(
            args.weight_decay
        ),
    )

    min_lr_ratio = (
        args.min_lr
        / args.lr
    )

    scheduler = (
        torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda s: (
                lr_scale(
                    s,
                    args.steps,
                    args.warmup_steps,
                    min_lr_ratio,
                )
            ),
        )
    )

    n_trainable = sum(
        p.numel()
        for p in trainable_params
    )

    print(
        "trainable parameters:",
        n_trainable,
    )

    if args.condition == "oracle":
        adapter_params = sum(
            p.numel()
            for p in (
                policy.model
                .intention_adapter
                .parameters()
            )
        )

        print(
            "intention adapter params:",
            adapter_params,
        )

        if adapter_params != 922560:
            raise RuntimeError(
                "Unexpected intention "
                "adapter parameter count."
            )

    # ---------------------------------------------------------
    # Save protocol
    # ---------------------------------------------------------

    protocol = {
        "condition":
            args.condition,
        "dataset":
            "lerobot/libero",
        "canonical_frames":
            88302,
        "canonical_episodes":
            335,
        "semantic_action_padding":
            True,
        "pretrained":
            PRETRAINED,
        "num_vlm_layers":
            16,
        "resize":
            [512, 512],
        "chunk_size":
            50,
        "n_action_steps":
            50,
        "seed":
            args.seed,
        "steps":
            args.steps,
        "batch_size":
            args.batch_size,
        "optimizer":
            "AdamW",
        "lr":
            args.lr,
        "weight_decay":
            args.weight_decay,
        "scheduler":
            "linear_warmup_cosine_decay",
        "warmup_steps":
            args.warmup_steps,
        "min_lr":
            args.min_lr,
    }

    with (
        output_dir
        / "protocol.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            protocol,
            f,
            indent=2,
        )

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    step = 0
    epoch = 0

    loss_window = []
    log_history = []

    start_time = time.time()

    print()
    print("=" * 72)
    print("TRAINING")
    print("=" * 72)

    while step < args.steps:
        epoch += 1

        for raw_batch in loader:
            if step >= args.steps:
                break

            raw_batch = (
                prepare_raw_batch(
                    raw_batch,
                    base_dataset.meta
                    .camera_keys,
                )
            )

            # Preserve custom oracle vector in case
            # processor drops unknown keys.
            oracle_intention = None

            if args.condition == "oracle":
                if (
                    "intention"
                    not in raw_batch
                ):
                    raise RuntimeError(
                        "Oracle batch missing "
                        "intention."
                    )

                oracle_intention = (
                    raw_batch[
                        "intention"
                    ]
                )

            batch = preprocessor(
                raw_batch
            )

            if args.condition == "oracle":
                if (
                    "intention"
                    not in batch
                ):
                    batch[
                        "intention"
                    ] = (
                        oracle_intention
                        .to(
                            policy_cfg.device
                        )
                    )

                if (
                    batch["intention"]
                    .shape[-1]
                    != 960
                ):
                    raise RuntimeError(
                        "Invalid oracle "
                        "intention shape: "
                        f"{tuple(batch['intention'].shape)}"
                    )
            else:
                if "intention" in batch:
                    raise RuntimeError(
                        "Baseline batch "
                        "unexpectedly contains "
                        "intention."
                    )

            optimizer.zero_grad(
                set_to_none=True
            )

            # Important matched stochasticity:
            # baseline and oracle use the same
            # per-step random seed for flow time,
            # noise, and stochastic layers.
            torch.manual_seed(
                args.seed
                + step
                + 100000
            )

            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(
                    args.seed
                    + step
                    + 100000
                )

            loss, info = policy(
                batch
            )

            if not torch.isfinite(
                loss
            ):
                raise RuntimeError(
                    f"Non-finite loss "
                    f"at step {step}: "
                    f"{loss}"
                )

            loss.backward()

            optimizer.step()
            scheduler.step()

            step += 1

            loss_value = float(
                loss.detach().cpu()
            )

            loss_window.append(
                loss_value
            )

            if len(loss_window) > 100:
                loss_window.pop(0)

            if (
                step == 1
                or step
                % args.log_freq
                == 0
            ):
                elapsed = (
                    time.time()
                    - start_time
                )

                avg_loss = (
                    sum(loss_window)
                    / len(loss_window)
                )

                current_lr = (
                    optimizer
                    .param_groups[0]
                    ["lr"]
                )

                record = {
                    "step": step,
                    "epoch": epoch,
                    "loss":
                        loss_value,
                    "loss_100":
                        avg_loss,
                    "lr":
                        current_lr,
                    "elapsed_s":
                        elapsed,
                }

                if (
                    args.condition
                    == "oracle"
                ):
                    adapter_grad = 0.0

                    for p in (
                        policy.model
                        .intention_adapter
                        .parameters()
                    ):
                        if p.grad is not None:
                            adapter_grad += (
                                float(
                                    p.grad
                                    .detach()
                                    .norm()
                                    .cpu()
                                )
                            )

                    record[
                        "adapter_grad_sum"
                    ] = adapter_grad

                log_history.append(
                    record
                )

                msg = (
                    f"step {step:06d} | "
                    f"epoch {epoch:03d} | "
                    f"loss "
                    f"{loss_value:.6f} | "
                    f"loss100 "
                    f"{avg_loss:.6f} | "
                    f"lr "
                    f"{current_lr:.8f}"
                )

                if (
                    args.condition
                    == "oracle"
                ):
                    msg += (
                        " | adapter_grad "
                        f"{record['adapter_grad_sum']:.6f}"
                    )

                print(msg)

            if (
                step % args.save_freq
                == 0
                or step == args.steps
            ):
                save_checkpoint(
                    output_dir=output_dir,
                    step=step,
                    policy=policy,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    args=args,
                    running_info={
                        "epoch": epoch,
                        "loss":
                            loss_value,
                    },
                )

                with (
                    output_dir
                    / "history.json"
                ).open(
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(
                        log_history,
                        f,
                        indent=2,
                    )

    total_time = (
        time.time()
        - start_time
    )

    summary = {
        "condition":
            args.condition,
        "steps":
            step,
        "epochs_seen":
            epoch,
        "final_loss":
            loss_value,
        "total_time_s":
            total_time,
        "output_dir":
            str(output_dir),
    }

    with (
        output_dir
        / "summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print("=" * 72)
    print("TRAINING COMPLETE")
    print("=" * 72)
    print(
        json.dumps(
            summary,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()