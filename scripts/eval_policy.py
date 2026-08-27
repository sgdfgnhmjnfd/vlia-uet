"""[GIAI ĐOẠN 5] Đánh giá & so sánh: SmolVLA gốc vs Text-based VLIA vs Token-based VLIA.

Ví dụ (giống lerobot_eval, mở rộng chạy cả 3 checkpoint và gộp bảng so sánh):
    python scripts/eval_policy.py \\
        --policy.path outputs/phase4_intention_token/checkpoints/last/pretrained_model \\
        --env.task libero_10 \\
        --eval.n_episodes 10 \\
        --suites-config config/suites/libero_suites.yaml
"""

if __name__ == "__main__":
    raise NotImplementedError
