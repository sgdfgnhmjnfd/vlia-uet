"""[GIAI ĐOẠN 2] Sinh nhãn ý định (What/Why/Next) bằng Qwen2.5-VL-3B.

Ví dụ:
    python scripts/generate_intention_labels.py \\
        --video-dir datasets/libero_rollouts \\
        --out datasets/intention_labels/phase2_labels.jsonl
"""

import argparse
from envs.intention_labeler import generate_intention_labels

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--video-dir", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    generate_intention_labels(args.video_dir, args.out)
