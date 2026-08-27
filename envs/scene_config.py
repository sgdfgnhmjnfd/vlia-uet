"""Parser nạp TaskConfig từ file YAML trong config/tasks/."""

import yaml
from pathlib import Path


def load_task_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
