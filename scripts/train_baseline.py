"""[GIAI ĐOẠN 1] Train SmolVLA baseline nhẹ trên LIBERO spatial.

Cấu hình tinh gọn:
  - num_vlm_layers = 16  (dùng 16 lớp đầu của SmolVLM2-500M-Video-Instruct)
  - resize_imgs_with_padding = (224, 224)  → SigLIP cho ~64 visual tokens/ảnh
  - steps = 25_000, batch_size = 32

Cách dùng:
    python scripts/train_baseline.py
    python scripts/train_baseline.py --steps 10000 --batch_size 16
    python scripts/train_baseline.py --dry_run   # chạy 1 step kiểm tra
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

LEROBOT_PYTHON = "/home/dhqg/anaconda3/envs/lerobot/bin/python"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "tasks" / "phase1_baseline.yaml"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase1_baseline"


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_lerobot_config(cfg: dict, overrides: dict) -> dict:
    """Tạo YAML config theo đúng format draccus của lerobot.

    Lerobot parser xử lý 'policy.path' đặc biệt (dùng from_pretrained).
    Khi thấy policy.path, parser sẽ:
      1. Extract 'path' → dùng from_pretrained
      2. Flatten các fields còn lại trong policy block → _config_yaml_overrides["policy"]
      3. Truyền chúng vào from_pretrained qua cli_overrides

    => Tất cả policy fields (kể cả optimizer/scheduler) phải nằm trong policy block.
    """
    policy = cfg.get("policy", {})
    dataset = cfg.get("dataset", {})
    training = cfg.get("training", {})
    for k, v in overrides.items():
        if v is not None:
            training[k] = v

    rename_map = dataset.get("rename_map", {})

    return {
        "policy": {
            "path": policy.get("path", "lerobot/smolvla_base"),
            # Layer skipping
            "num_vlm_layers": policy.get("num_vlm_layers", 16),
            # Visual tokens: 224×224 → SigLIP patch 14×14 → ~64 tokens/ảnh
            "resize_imgs_with_padding": [224, 224],
            # Optimizer / scheduler — phải nằm ở đây để được flatten thành cli_overrides
            "optimizer_lr": training.get("optimizer_lr", 1e-4),
            "scheduler_warmup_steps": training.get("scheduler_warmup_steps", 1000),
            "scheduler_decay_steps": training.get("scheduler_decay_steps", 30000),
            "scheduler_decay_lr": training.get("scheduler_decay_lr", 2.5e-6),
            "push_to_hub": False,
        },
        "dataset": {
            "repo_id": dataset.get("repo_id", "HuggingFaceVLA/libero"),
        },
        "wandb": {
            "enable": False,
        },
        "steps": training.get("steps", 25000),
        "batch_size": training.get("batch_size", 32),
        "save_freq": training.get("save_freq", 500),
        "rename_map": rename_map,
        "output_dir": str(OUTPUT_DIR),
    }


def main():
    parser = argparse.ArgumentParser(description="Train SmolVLA Phase 1 Baseline")
    parser.add_argument("--steps", type=int, default=None, help="Override số steps train")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--dry_run", action="store_true",
                        help="Chạy 1 step để kiểm tra (không train thật)")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH,
                        help="Đường dẫn file config YAML")
    args = parser.parse_args()

    cfg = load_config(args.config)

    overrides = {}
    if args.steps is not None:
        overrides["steps"] = args.steps
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.dry_run:
        overrides["steps"] = 1
        overrides["save_freq"] = 1
        print("[DRY RUN] Chạy 1 step để kiểm tra pipeline...")

    lerobot_cfg = make_lerobot_config(cfg, overrides)
    training = cfg.get("training", {})

    # Nếu thư mục output đã có và không phải dry-run, cần set resume=True
    if OUTPUT_DIR.exists() and not args.dry_run:
        lerobot_cfg["resume"] = True
    elif args.dry_run:
        # Dry run output tạm
        lerobot_cfg["output_dir"] = str(OUTPUT_DIR / "dry_run")

    # Ghi config tạm để truyền qua --config_path
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(lerobot_cfg, tmp, default_flow_style=False)
        tmp_config_path = tmp.name

    train_cmd = [
        LEROBOT_PYTHON, "-m", "lerobot.scripts.lerobot_train",
        f"--config_path={tmp_config_path}",
    ]

    print("=" * 60)
    print("GIAI ĐOẠN 1 — Train SmolVLA Baseline nhẹ")
    print("=" * 60)
    print(f"Config gốc:  {args.config}")
    print(f"Config tạm:  {tmp_config_path}")
    print(f"Output dir:  {lerobot_cfg['output_dir']}")
    print(f"Steps:       {lerobot_cfg['steps']} (dry_run={args.dry_run})")
    print(f"Batch size:  {lerobot_cfg['batch_size']}")
    print(f"VLM layers:  {lerobot_cfg['policy']['num_vlm_layers']}")
    print(f"Img size:    224×224 → ~64 visual tokens/ảnh")
    print("=" * 60)
    print("Lệnh:")
    print("  " + " \\\n    ".join(train_cmd))
    print("=" * 60)

    result = subprocess.run(train_cmd, cwd=str(PROJECT_ROOT))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
