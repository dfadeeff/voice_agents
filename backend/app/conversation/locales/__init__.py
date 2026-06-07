"""Locale system for voice agent prompts."""

from __future__ import annotations

import importlib
from types import ModuleType

_cache: dict[str, ModuleType] = {}


def get_locale(lang: str = "de") -> ModuleType:
    if lang not in _cache:
        _cache[lang] = importlib.import_module(f"app.conversation.locales.{lang}")
    return _cache[lang]
