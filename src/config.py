"""OpenAlex 配置加载器 — 从 openalex_config.json 读取。

用法:
    from src.config import load_config
    cfg = load_config()
    print(cfg.openalex_mailto)
"""

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "openalex_config.json"


@dataclass
class Config:
    """OpenAlex configuration from openalex_config.json."""

    openalex_mailto: str = ""
    openalex_api_key: str = ""


def load_config(config_path: Path | None = None) -> Config:
    """Load OpenAlex config from JSON file.

    Args:
        config_path: Path to openalex_config.json. Defaults to project root.

    Returns:
        Config instance with values from the JSON file (or defaults if file not found).
    """
    path = config_path or DEFAULT_CONFIG_PATH
    cfg = Config()

    if not path.exists():
        return cfg

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cfg.openalex_mailto = data.get("mailto", "")
        cfg.openalex_api_key = data.get("api_key", "")
    except (json.JSONDecodeError, OSError):
        pass

    return cfg
