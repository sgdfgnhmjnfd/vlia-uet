"""Dataclass cấu hình SmolVLA mở rộng cho VLIA (Giai đoạn 4).

Kế thừa từ lerobot SmolVLAConfig gốc, thêm các trường liên quan Intention Token.
"""

from dataclasses import dataclass


@dataclass
class VLIASmolVLAConfig:
    # --- Kế thừa từ SmolVLA gốc (điền lại theo lerobot khi implement) ---
    # ...

    # --- Mở rộng cho Intention Token (Giai đoạn 4) ---
    use_intention_token: bool = False
    intention_token_dim: int = 960
