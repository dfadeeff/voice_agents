"""LLM-assisted spoken-email extraction (cloud-first, opt-in).

Spoken email over phone-quality audio is the one field where a model reliably
beats regex: it can reassemble a chunked, noisy STT string ("Lang M at gmail
punkt com") into "langm@gmail.com". The deterministic regex (manager._parse_email)
stays the local default and runs first for free; this is only invoked as a rescue
when regex fails and an LLM is configured.

Uses the OpenAI-compatible chat API, so it works against both OpenAI (cloud) and
Ollama's /v1 endpoint (local), selected from Settings.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable

from app.config import Settings

logger = logging.getLogger(__name__)

_EMAIL_RE = re.compile(r"[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}")

_SYSTEM_PROMPT = (
    "You convert a spoken email dictation into a single email address. The text was "
    "produced by speech-to-text over a noisy phone line, so expect: spaces inside the "
    "local part that should be joined ('lang m' -> 'langm'), spoken connectors "
    "('at'/'ät' -> @, 'punkt'/'dot'/'point' -> .), filler/lead-in words to drop "
    "('ja', 'es ist', 'meine E-Mail-Adresse ist'), and minor mishearings. "
    "If a caller name is given, the local part is usually derived from it, so prefer "
    "the spelling that matches the name when the dictation is a close match "
    "(e.g. name 'Leon Sigmar' + heard 'sigma' -> 'sigmar'). "
    "Reply with ONLY the most likely email address in lowercase and nothing else. "
    "If the text clearly contains no email at all, reply with the single word NONE."
)

EmailExtractor = Callable[[str, str], Awaitable[str | None]]


def make_email_extractor(settings: Settings) -> EmailExtractor | None:
    """Build an async email extractor, or None when LLM assist is disabled.

    Enabled when ``llm_assist`` is set, or implicitly for the cloud (OpenAI)
    provider. Local Ollama stays regex-only unless explicitly enabled.
    """
    enabled = settings.llm_assist or settings.llm_provider == "openai"
    if not enabled:
        return None

    from openai import AsyncOpenAI

    if settings.llm_provider == "openai":
        client = AsyncOpenAI(api_key=settings.openai_api_key)
        model = settings.openai_model
    else:
        client = AsyncOpenAI(base_url=f"{settings.ollama_base_url}/v1", api_key="ollama")
        model = settings.ollama_model

    async def extract(text: str, name_hint: str = "") -> str | None:
        user = f"Caller name: {name_hint}\nDictation: {text}" if name_hint else text
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                temperature=0,
                max_tokens=30,
            )
            out = (resp.choices[0].message.content or "").strip().lower()
        except Exception as e:  # network/model errors must not break the call
            logger.warning("LLM email extraction failed: %s", e)
            return None
        if out.startswith("none"):
            return None
        m = _EMAIL_RE.search(out)
        return m.group(0) if m else None

    logger.info(
        "LLM email extraction enabled (provider=%s, model=%s)", settings.llm_provider, model
    )
    return extract
