"""Deterministic narration for state-determined steps (the narration split).

For callback steps the next utterance is a pure function of state, so we speak it
straight from a template and bypass the LLM. This removes the single-call
"decide + speak" coupling that lets a 7B model narrate actions it didn't take —
no invented phone numbers, no claimed appointment times, no rambling. Open turns
(routing, qualification, booking negotiation) still go to the LLM.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from app.conversation.locales import get_locale
from app.models.schemas import CallPhase, LegalArea

if TYPE_CHECKING:
    from app.conversation.state import ConversationState

# Caller denies the area-confirmation → let the LLM re-route instead of repeating.
_AREA_DENIAL_RE = re.compile(
    r"\b(?:nein|nö|nee|stimmt\s+nicht|nicht\s+richtig|falsch|no|wrong|incorrect)\b",
    re.IGNORECASE,
)


def _last_user_text(state: ConversationState) -> str:
    for msg in reversed(state.messages):
        if msg.get("role") == "user" and msg.get("content"):
            return str(msg["content"])
    return ""


def scripted_line(state: ConversationState, lang: str = "de") -> str | None:
    """Return the deterministic next line for a structured step, or None for the LLM.

    Covers the traffic qualification steps (area confirmation, insurance) and the
    callback contact steps — the turns where the matter is fully state-determined,
    so the model never generates them and cannot leak a tool call or hallucinate.
    The phone is left raw (e.g. "+4915159832614"); the pre-TTS layer expands it.
    """
    scripts = getattr(get_locale(lang), "SCRIPTED", {})
    if not scripts:
        return None

    ents = state.entities

    # Qualification intake (before any handoff): area confirmation, then — for
    # traffic — the insurance/claim number.
    if state.phase == CallPhase.QUALIFICATION and state.legal_area != LegalArea.UNKNOWN:
        if "matter_type" not in ents:
            if _AREA_DENIAL_RE.search(_last_user_text(state)):
                return None  # caller denied the area → LLM re-routes
            return scripts.get(f"{state.legal_area.value}_confirm")
        if (
            state.legal_area == LegalArea.TRAFFIC
            and not state.insurance_resolved
            and "insurance_number" not in ents
        ):
            return scripts.get("traffic_insurance")
        return None

    if not state.callback_requested:
        return None

    name = state.entities.get("name")
    phone = state.entities.get("phone")

    if state.phase in (CallPhase.CAPTURE, CallPhase.ESCALATION):
        if not (name and name.value):
            return scripts["ask_name"]
        if not phone:
            return scripts["ask_phone"]
        if not phone.confirmed:
            return scripts["confirm_phone"].format(phone=phone.value)
        return None

    if state.phase == CallPhase.CONFIRMATION:
        person = state.target_person or scripts.get("team", "")
        return scripts["callback_done"].format(person=person)

    return None
