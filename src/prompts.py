"""System prompt 加载器，支持多语言。

语言配置在 config/languages.json 中定义。
新增语言只需在 prompts.json 中添加 {name}_{suffix} 键。
"""

import json
from pathlib import Path

_PROMPTS_PATH = Path(__file__).resolve().parent / "agents" / "prompts.json"
_LANG_PATH = Path(__file__).resolve().parent.parent / "config" / "languages.json"
_CACHE: dict[str, str] = {}
_META_CACHE: dict | None = None
_LANG_CACHE: dict | None = None


def _load_languages() -> dict:
    global _LANG_CACHE
    if _LANG_CACHE is None:
        if not _LANG_PATH.exists():
            raise FileNotFoundError(f"Languages file not found: {_LANG_PATH}")
        with open(_LANG_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        lookup = {lang["code"]: lang for lang in raw.get("languages", [])}
        _LANG_CACHE = {"default": raw.get("default", "zh-hans"), "lookup": lookup}
    return _LANG_CACHE


def _load_all() -> dict:
    global _META_CACHE
    if _META_CACHE is None:
        if not _PROMPTS_PATH.exists():
            raise FileNotFoundError(f"Prompts file not found: {_PROMPTS_PATH}")
        with open(_PROMPTS_PATH, "r", encoding="utf-8") as f:
            _META_CACHE = json.load(f)
    assert _META_CACHE is not None
    return _META_CACHE


def load_prompt(name: str, lang: str = "zh-hans") -> str:
    """加载指定语言的 system prompt。

    Args:
        name: 键名，如 "relevance", "quality", "review_section"。
        lang: 语言代码 "zh-hans", "zh-hant", "en" 等。

    Returns:
        Prompt 文本内容。
    """
    cache_key = f"{name}_{lang}"
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    data = _load_all()
    lang_info = _load_languages()
    lookup = lang_info["lookup"]
    info = lookup.get(lang, {})
    suffix = info.get("suffix", "")

    # Try suffixed key first: {name}{suffix}
    suffixed = f"{name}{suffix}"
    if suffixed in data:
        text = data[suffixed]["system_prompt"].strip()
        _CACHE[cache_key] = text
        return text

    # Fallback to base key
    if name in data:
        text = data[name]["system_prompt"].strip()
        _CACHE[cache_key] = text
        return text

    # Fallback to default language
    fallback = info.get("fallback", lang_info["default"])
    if fallback != lang:
        return load_prompt(name, fallback)

    raise KeyError(
        f"Prompt not found: {name} (lang={lang}), available: {list(data.keys())}"
    )


def list_prompts() -> list[dict]:
    data = _load_all()
    return [
        {"name": k, "description": v.get("description", "")} for k, v in data.items()
    ]
