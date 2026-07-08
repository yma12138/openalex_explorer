"""Tests for config loader (OpenAlex config from JSON)."""

import json
import sys
import tempfile
from pathlib import Path


def test_load_config_returns_openalex_secrets():
    """Config load should read OpenAlex secrets from JSON."""
    with tempfile.TemporaryDirectory() as tmp:
        config_data = {
            "mailto": "test@example.com",
            "api_key": "test-key-123",
        }
        config_path = Path(tmp) / "openalex_config.json"
        with open(config_path, "w") as f:
            json.dump(config_data, f)

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from src.config import load_config

        cfg = load_config(config_path=config_path)
        assert cfg.openalex_mailto == "test@example.com"
        assert cfg.openalex_api_key == "test-key-123"


def test_load_config_defaults():
    """Config should not crash when called without args."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.config import load_config

    cfg = load_config()
    assert cfg is not None


def test_config_file_not_found():
    """If openalex_config.json doesn't exist, keys should be empty."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.config import load_config

    cfg = load_config(config_path=Path("/nonexistent/path/openalex_config.json"))
    assert cfg.openalex_mailto == ""
    assert cfg.openalex_api_key == ""
