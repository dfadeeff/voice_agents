"""Accuracy-critical fast-path overrides (the narration split).

The LLM drives the conversation — asking questions, acknowledging, transitioning.
This module overrides the LLM only when exact data must be read back or presented:
email/phone confirmations, slot offers from the DB, and booking/callback done.
Deterministic guards in ConversationManager catch data the LLM misses.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.conversation.locales import get_locale
from app.models.schemas import CallPhase

if TYPE_CHECKING:
    from app.conversation.state import ConversationState


def _last_agent_text(state: ConversationState) -> str:
    for msg in reversed(state.messages):
        if msg.get("role") == "assistant" and msg.get("content"):
            return str(msg["content"])
    return ""


def _spoken_phone(value: str, lang: str) -> str:
    """German callers expect their national number read back (0151…), not +49."""
    if lang == "de" and value.startswith("+49"):
        return "0" + value[3:]
    return value


_MONTHS = {
    "de": [
        "",
        "Januar",
        "Februar",
        "März",
        "April",
        "Mai",
        "Juni",
        "Juli",
        "August",
        "September",
        "Oktober",
        "November",
        "Dezember",
    ],
    "en": [
        "",
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ],
}


_ORDINALS_DE = [
    "",
    "ersten",
    "zweiten",
    "dritten",
    "vierten",
    "fünften",
    "sechsten",
    "siebten",
    "achten",
    "neunten",
    "zehnten",
    "elften",
    "zwölften",
    "dreizehnten",
    "vierzehnten",
    "fünfzehnten",
    "sechzehnten",
    "siebzehnten",
    "achtzehnten",
    "neunzehnten",
    "zwanzigsten",
    "einundzwanzigsten",
    "zweiundzwanzigsten",
    "dreiundzwanzigsten",
    "vierundzwanzigsten",
    "fünfundzwanzigsten",
    "sechsundzwanzigsten",
    "siebenundzwanzigsten",
    "achtundzwanzigsten",
    "neunundzwanzigsten",
    "dreißigsten",
    "einunddreißigsten",
]


def _fmt_date(date: str, lang: str) -> str:
    try:
        _, month, day = date.split("-")
    except ValueError:
        return date
    months = _MONTHS.get(lang, _MONTHS["en"])
    name = months[int(month)]
    if lang == "de":
        d = int(day)
        if d < len(_ORDINALS_DE):
            return f"{_ORDINALS_DE[d]} {name}"
        return f"{d} {name}"
    return f"{name} {int(day)}"


def _fmt_time(time: str, lang: str) -> str:
    hh, _, mm = time.partition(":")
    hour = int(hh)
    if lang == "de":
        if mm in ("00", ""):
            return f"{hour} Uhr"
        return f"{hour} Uhr {int(mm)}"
    return time


def _fmt_slot(slot: dict, lang: str) -> str:
    date = _fmt_date(slot["date"], lang)
    time = _fmt_time(slot["time"], lang)
    if lang == "de":
        return f"{date} um {time}"
    return f"{date} at {time}"


def _fmt_slots(slots: list[dict], lang: str) -> str:
    parts = [_fmt_slot(s, lang) for s in slots]
    if len(parts) <= 1:
        return parts[0] if parts else ""
    joiner = " oder " if lang == "de" else " or "
    return ", ".join(parts[:-1]) + joiner + parts[-1]


def scripted_line(state: ConversationState, lang: str = "de") -> str | None:
    """Return a deterministic line only for accuracy-critical steps, else None.

    The LLM drives conversation (asking questions, transitions, acknowledgements).
    Fast-path overrides fire only where exact data must be read back or presented:
    email/phone confirmations, slot offers, and booking/callback confirmations.
    Deterministic guards in ConversationManager catch data the LLM misses.
    """
    scripts = getattr(get_locale(lang), "SCRIPTED", {})
    if not scripts:
        return None

    ents = state.entities
    callback = state.callback_requested
    email = ents.get("email")
    phone = ents.get("phone")

    # Accuracy-critical read-backs: email and phone must be exact.
    if state.phase in (CallPhase.CAPTURE, CallPhase.ESCALATION):
        if not callback and email and not email.confirmed:
            return scripts["confirm_email"].format(email=email.value)
        if phone and not phone.confirmed:
            return scripts["confirm_phone"].format(phone=_spoken_phone(phone.value, lang))

    # Booking: present available slots (the manager fetches them and books the choice).
    if state.phase == CallPhase.BOOKING and not state.booking_confirmed:
        if state.offered_slots:
            return scripts["slot_offer"].format(options=_fmt_slots(state.offered_slots, lang))
        return scripts["no_slots"]

    if not callback and state.phase == CallPhase.CONFIRMATION and state.booked_slot:
        last_agent = _last_agent_text(state)
        if "gebucht" in last_agent or "booked" in last_agent:
            return scripts.get("goodbye")
        slot = state.booked_slot
        return scripts["booking_done"].format(
            date=_fmt_date(slot["date"], lang),
            time=_fmt_time(slot["time"], lang),
        )

    if callback and state.phase == CallPhase.CONFIRMATION:
        # Say the closing only once, even if the caller adds "Tschüss" afterwards.
        last_agent = _last_agent_text(state)
        if "Wiederhören" in last_agent or "Goodbye" in last_agent:
            return None
        person = state.target_person or scripts.get("team", "")
        return scripts["callback_done"].format(person=person, time=state.preferred_time or "")

    return None
