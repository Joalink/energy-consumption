from pathlib import Path

import yaml

CONFIG_PATH = Path("config/training_config.yaml")


def load_config() -> dict:
    with open(Path(CONFIG_PATH)) as f:
        return yaml.safe_load(f)
