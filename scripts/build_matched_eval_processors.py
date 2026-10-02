from pathlib import Path

from lerobot.configs import PreTrainedConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

PRETRAINED = "lerobot/smolvla_base"

RENAME_MAP = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.image2": "observation.images.camera2",
}

OUT = Path(
    "/media/dhqg/d2/vlia_outputs/oracle_matched_full/"
    "baseline/seed_0/checkpoint_025000/pretrained_model_eval_matched"
)

cfg = PreTrainedConfig.from_pretrained(
    pretrained_name_or_path=PRETRAINED
)

if not isinstance(cfg, SmolVLAConfig):
    raise TypeError(type(cfg))

cfg.num_vlm_layers = 16
cfg.resize_imgs_with_padding = (512, 512)
cfg.chunk_size = 50
cfg.n_action_steps = 50
cfg.device = "cuda"

meta = LeRobotDatasetMetadata("lerobot/libero")
stats = meta.stats

print("stats keys:", list(stats.keys()))

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

preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=cfg,
    pretrained_path=PRETRAINED,
    dataset_stats=stats,
    preprocessor_overrides=overrides,
)

OUT.mkdir(parents=True, exist_ok=True)

preprocessor.save_pretrained(OUT)
postprocessor.save_pretrained(OUT)

print("saved to:", OUT)
