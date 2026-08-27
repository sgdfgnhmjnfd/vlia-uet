"""[GIAI ĐOẠN 1] Train SmolVLA baseline nhẹ trên LIBERO spatial.

Tương ứng lệnh gốc:
    python -m lerobot.scripts.lerobot_train \\
      --policy.path=lerobot/smolvla_base \\
      --dataset.repo_id=HuggingFaceVLA/libero \\
      --steps=25000 --batch_size=32 --save_freq=500 \\
      --config config/tasks/phase1_baseline.yaml

TODO: đọc config/tasks/phase1_baseline.yaml, build args, gọi lerobot_train.
"""

if __name__ == "__main__":
    raise NotImplementedError
