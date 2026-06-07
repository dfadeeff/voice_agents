"""System prompts: per-phase constraints for the LLM.

Each phase gives the LLM a narrow task. The full call flow is NOT in the prompt —
the state machine in flow.py controls transitions, not the LLM.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.config import Settings
from app.conversation.flow import CALLBACK_REQUIRED_FIELDS, CONTACT_FIELDS
from app.conversation.locales import get_locale
from app.models.schemas import CallPhase

if TYPE_CHECKING:
    from app.conversation.state import ConversationState

_settings = Settings()
_is_qwen3 = "qwen3" in _settings.ollama_model

# Backward-compat aliases — English constants for existing imports/tests
_en = get_locale("en")
PREAMBLE = _en.PREAMBLE
PHASE_PROMPTS = _en.PHASE_PROMPTS
EMPLOYMENT_FRAGMENT = _en.EMPLOYMENT_FRAGMENT
TENANCY_FRAGMENT = _en.TENANCY_FRAGMENT
FRAGMENTS = _en.FRAGMENTS
SYSTEM_PROMPT_TOOLS = _en.SYSTEM_PROMPT_TOOLS
SYSTEM_PROMPT_LOCAL = _en.SYSTEM_PROMPT_LOCAL
SYSTEM_PROMPT_BASE = SYSTEM_PROMPT_TOOLS


def get_system_prompt_base(lang: str = "de") -> str:
    return get_locale(lang).SYSTEM_PROMPT_TOOLS


def build_system_prompt(state: ConversationState, lang: str = "de") -> str:
    """Build a phase-specific system prompt from current conversation state."""
    locale = get_locale(lang)
    parts = [locale.PREAMBLE]

    if state.callback_requested and state.phase in (CallPhase.CAPTURE, CallPhase.CONFIRMATION):
        callback_prompts = getattr(locale, "CALLBACK_PROMPTS", {})
        phase_prompt = callback_prompts.get(state.phase, "")
        target = state.target_person or ("das Kanzleiteam" if lang == "de" else "the team")
        phase_prompt = phase_prompt.replace("{target_person}", target)
    else:
        phase_prompt = locale.PHASE_PROMPTS.get(state.phase, "")

    if phase_prompt:
        task_label = "DEINE AKTUELLE AUFGABE" if lang == "de" else "YOUR CURRENT TASK"
        parts.append(f"\n{task_label}:\n{phase_prompt}")

    if state.phase == CallPhase.CAPTURE:
        fields = CALLBACK_REQUIRED_FIELDS if state.callback_requested else CONTACT_FIELDS
        missing = []
        unconfirmed = []
        for field in fields:
            entity = state.entities.get(field)
            if not entity:
                missing.append(field)
            elif not entity.confirmed:
                unconfirmed.append(f"{field} (heard: '{entity.value}', needs confirmation)")
        if missing:
            label = "FEHLENDE FELDER" if lang == "de" else "MISSING FIELDS"
            parts.append(f"\n{label}: {', '.join(missing)}")
        if unconfirmed:
            label = "BESTÄTIGUNG NÖTIG" if lang == "de" else "NEEDS CONFIRMATION"
            parts.append(f"\n{label}: {', '.join(unconfirmed)}")

    if not state.callback_requested:
        fragment = locale.FRAGMENTS.get(state.legal_area.value, "")
        if fragment and state.phase in (
            CallPhase.ROUTING,
            CallPhase.INFORMATION,
            CallPhase.CAPTURE,
            CallPhase.BOOKING,
        ):
            parts.append(fragment)

    prompt = "\n".join(parts)
    if _is_qwen3:
        prompt += "\n/no_think"
    return prompt
