"""VLIA configuration extending LeRobot SmolVLA."""

from dataclasses import dataclass

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig


@dataclass
class VLIASmolVLAConfig(SmolVLAConfig):
    """SmolVLA config with an optional semantic intention representation."""

    use_intention_token: bool = True
    intention_dim: int = 256
