"""Kiến trúc PyTorch SmolVLA mở rộng — chèn Intention Token vào Prefix Embedding.

Kiến trúc gốc SmolVLA:
    Image (16x960) + Language (48x960) + State (1x960) -> Prefix Embedding (97x960)
    -> SmolVLM (16 layers) -> Cross-Attention Expert -> Flow Matching (10 steps) -> Action (50x32)

Giai đoạn 4 (VLIA): chèn thêm 1 Intention Token (960D) vào Prefix Embedding
    -> Prefix Embedding (98x960)
Token này được huấn luyện để căn chỉnh (align) với không gian nhãn ý định
đã học ở Giai đoạn 3 (Text-based Intention baseline).

TODO: import SmolVLAPolicy gốc từ lerobot, override forward() / prepare_inputs()
để chèn intention token trước khi đưa vào SmolVLM backbone.
"""


class IntentionToken:
    """Placeholder cho learned intention token (960D), gắn vào đầu prefix embedding."""

    def __init__(self, dim: int = 960):
        self.dim = dim
        raise NotImplementedError
