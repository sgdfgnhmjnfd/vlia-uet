"""Giai đoạn 2 — Sinh nhãn ý định (What / Why / Next) bằng VLM ngoài (Qwen2.5-VL-3B).

Quét qua video rollout LIBERO, xuất nhãn dạng JSONL:
    {"episode_id": ..., "frame": ..., "what": ..., "why": ..., "next": ...}
"""

VLM_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


def generate_intention_labels(video_path: str, out_path: str):
    """TODO: load VLM_MODEL_ID, chạy inference theo từng đoạn video, ghi JSONL ra out_path."""
    raise NotImplementedError
