"""LLM-assisted matter-type classification (cloud-first, opt-in).

`matter_type` is an open-ended slot: the caller describes their issue in free
speech and we map it to a coarse label for triage. Keyword matching
(manager._MATTER_KEYWORDS) is the local default and runs first; it works when the
caller echoes an offered option ("Kündigung der Wohnung" → eviction) but misses
free phrasings ("ich wurde aus meiner Wohnung geworfen"). This LLM rescue maps the
answer to one of the area's valid labels when the keywords miss — same pattern as
the spoken-email rescue. The verbatim answer is always kept in matter_summary
regardless, so nothing is lost even if classification falls back to "other".
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from app.config import Settings

logger = logging.getLogger(__name__)

# Valid coarse labels per area (mirrors the options the agent offers and
# manager._MATTER_KEYWORDS). "other" is always acceptable.
MATTER_TYPES: dict[str, list[str]] = {
    "employment": ["dismissal", "warning", "wages", "contract", "other"],
    "tenancy": ["eviction", "deposit", "rent_increase", "repairs", "other"],
    "traffic": ["accident", "damage", "insurance", "other"],
}

# (area, caller_text) -> one valid label, or None.
MatterClassifier = Callable[[str, str], Awaitable[str | None]]


def make_matter_classifier(settings: Settings) -> MatterClassifier | None:
    """Build an async matter-type classifier, or None when LLM assist is disabled.

    Enabled when ``llm_assist`` is set, or implicitly for the cloud (OpenAI)
    provider. Local Ollama stays keyword-only unless explicitly enabled.
    """
    if not (settings.llm_assist or settings.llm_provider == "openai"):
        return None

    from openai import AsyncOpenAI

    if settings.llm_provider == "openai":
        client = AsyncOpenAI(api_key=settings.openai_api_key)
        model = settings.openai_model
    else:
        client = AsyncOpenAI(base_url=f"{settings.ollama_base_url}/v1", api_key="ollama")
        model = settings.ollama_model

    async def classify(area: str, text: str) -> str | None:
        labels = MATTER_TYPES.get(area)
        if not labels:
            return None
        prompt = (
            "You classify a law-firm caller's described matter into exactly one label. "
            f"The legal area is '{area}'. Valid labels: {', '.join(labels)}. "
            "Reply with ONLY one label from that list — no other words."
        )
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": text},
                ],
                temperature=0,
                max_tokens=8,
            )
            out = (resp.choices[0].message.content or "").strip().lower()
        except Exception as e:  # network/model errors must not break the call
            logger.warning("LLM matter classification failed: %s", e)
            return None
        # Accept only a valid label for this area (guards against chatter).
        for label in labels:
            if label in out:
                return label
        return None

    logger.info(
        "LLM matter classification enabled (provider=%s, model=%s)", settings.llm_provider, model
    )
    return classify
