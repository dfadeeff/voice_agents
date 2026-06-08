"""Deterministic narration for state-determined steps (the narration split).

For callback steps the next utterance is a pure function of state, so we speak it
straight from a template and bypass the LLM. This removes the single-call
"decide + speak" coupling that lets a 7B model narrate actions it didn't take —
no invented phone numbers, no claimed appointment times, no rambling. Open turns
(routing, qualification, booking negotiation) still go to the LLM.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.conversation.locales import get_locale
from app.models.schemas import CallPhase

if TYPE_CHECKING:
    from app.conversation.state import ConversationState


def scripted_line(state: ConversationState, lang: str = "de") -> str | None:
    """Return the deterministic next line for a callback step, or None for the LLM.

    The phone is left raw (e.g. "+4915159832614"); the pre-TTS layer expands it to
    spoken digits.
    """
    if not state.callback_requested:
        return None
    scripts = getattr(get_locale(lang), "SCRIPTED", {})
    if not scripts:
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
