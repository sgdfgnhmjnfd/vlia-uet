"""Script đánh giá (Eval) policy trên LIBERO environment.

Cách dùng:
    python scripts/eval_policy.py 
    # Hoặc trỏ cụ thể checkpoint:
    python scripts/eval_policy.py --ckpt outputs/phase1_baseline/checkpoints/000500/pretrained_model
"""

import argparse
import subprocess
import sys
from pathlib import Path

LEROBOT_PYTHON = "/home/dhqg/anaconda3/envs/lerobot/bin/python"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase1_baseline"


def get_latest_checkpoint() -> Path:
    """Tự động tìm checkpoint mới nhất."""
    ckpt_dir = OUTPUT_DIR / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Chưa có checkpoint nào trong {ckpt_dir}!")
    
    # Checkpoints format: 000500, 001000...
    ckpts = sorted([d for d in ckpt_dir.iterdir() if d.is_dir() and d.name.isdigit()])
    if not ckpts:
        raise FileNotFoundError(f"Thư mục {ckpt_dir} trống!")
    
    latest_ckpt = ckpts[-1] / "pretrained_model"
    return latest_ckpt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Đường dẫn đến folder pretrained_model")
    parser.add_argument("--episodes", type=int, default=10, help="Số episodes eval")
    parser.add_argument("--task", type=str, default="libero_10", help="Tập task của LIBERO")
    args = parser.parse_args()

    # Xác định policy path
    if args.ckpt is None:
        try:
            policy_path = get_latest_checkpoint()
        except FileNotFoundError as e:
            print(f"Lỗi: {e}")
            print("Model có vẻ chưa chạy tới mốc lưu checkpoint đầu tiên (500 steps).")
            print("Hãy đợi thêm vài phút nữa để có checkpoint nhé!")
            sys.exit(1)
    else:
        policy_path = Path(args.ckpt).resolve()

    print("=" * 60)
    print("ĐÁNH GIÁ (EVAL) MODEL")
    print(f"Policy:   {policy_path}")
    print(f"Môi trường: LIBERO ({args.task})")
    print(f"Số lượt:  {args.episodes} episodes")
    print("=" * 60)

    # Lệnh eval chuẩn của lerobot
    eval_cmd = [
        LEROBOT_PYTHON, "-m", "lerobot.scripts.lerobot_eval",
        "-p", str(policy_path),
        "--env.type", "libero",
        f"--env.name={args.task}",
        f"--eval.n_episodes={args.episodes}",
        "--eval.batch_size=1",  # Chạy 1 env để tránh ngốn RAM
        # Rename map để trùng với camera model train
        "--rename_map", '{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'
    ]

    print("Lệnh chạy:")
    print(" ".join(eval_cmd))
    print("=" * 60)
    
    result = subprocess.run(eval_cmd, cwd=str(PROJECT_ROOT))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
