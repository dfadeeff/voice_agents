"""Large-language-model providers.

Add a vendor by writing a builder and adding one ``_BUILDERS`` entry. Vendor SDK
imports stay inside each builder.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app.config import Settings

LlmBuilder = Callable[[Settings], Any]


def _ollama_max_tokens(settings: Settings) -> int:
    """Completion cap for the local model. ``ollama_max_tokens`` overrides; 0 means
    pick by family — qwen3 "thinking" models need headroom for dropped reasoning."""
    if settings.ollama_max_tokens:
        return settings.ollama_max_tokens
    return 400 if "qwen3" in settings.ollama_model else 150


def _build_ollama(settings: Settings) -> Any:
    from pipecat.services.ollama.llm import OLLamaLLMService

    return OLLamaLLMService(
        settings=OLLamaLLMService.Settings(
            model=settings.ollama_model,
            max_tokens=_ollama_max_tokens(settings),
            temperature=settings.llm_temperature,
        ),
        base_url=f"{settings.ollama_base_url}/v1",
    )


def _build_openai(settings: Settings) -> Any:
    from pipecat.services.openai.llm import OpenAILLMService

    return OpenAILLMService(api_key=settings.openai_api_key, model=settings.openai_model)


_BUILDERS: dict[str, LlmBuilder] = {
    "ollama": _build_ollama,
    "openai": _build_openai,
}


def create_llm(settings: Settings) -> Any:
    try:
        builder = _BUILDERS[settings.llm_provider]
    except KeyError:
        raise ValueError(
            f"Unknown LLM provider {settings.llm_provider!r}; known: {sorted(_BUILDERS)}"
        ) from None
    return builder(settings)
