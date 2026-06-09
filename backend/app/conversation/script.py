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
    lawyer = slot.get("lawyer_name", "")
    if lang == "de":
        s = f"{date} um {time}"
        if lawyer:
            s += f" bei {lawyer}"
        return s
    s = f"{date} at {time}"
    if lawyer:
        s += f" with {lawyer}"
    return s


def _fmt_slots(slots: list[dict], lang: str) -> str:
    parts = [_fmt_slot(s, lang) for s in slots]
    if len(parts) <= 1:
        return parts[0] if parts else ""
    joiner = " oder " if lang == "de" else " or "
    return ", ".join(parts[:-1]) + joiner + parts[-1]


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

    callback = state.callback_requested
    name = ents.get("name")
    email = ents.get("email")
    phone = ents.get("phone")

    if state.phase in (CallPhase.CAPTURE, CallPhase.ESCALATION):
        if not (name and name.value):
            return scripts["ask_name"] if callback else scripts["ask_name_booking"]
        # Booking asks for an email (with spell-back) but it is optional — a phone
        # number is enough, so the caller can decline. A callback skips email.
        if not callback:
            if not email and not state.email_skipped:
                return scripts["ask_email"]
            if email and not email.confirmed:
                return scripts["confirm_email"].format(email=email.value)
        if not phone:
            return scripts["ask_phone"]
        if not phone.confirmed:
            return scripts["confirm_phone"].format(phone=_spoken_phone(phone.value, lang))
        # Callback: capture a preferred call-back time after the contact details.
        if callback and not state.preferred_time:
            return scripts["ask_callback_time"]
        return None  # all done → CONFIRMATION (callback) or BOOKING (booking)

    # Booking: present available slots (the manager fetches them and books the choice).
    if state.phase == CallPhase.BOOKING and not state.booking_confirmed:
        if state.offered_slots:
            return scripts["slot_offer"].format(options=_fmt_slots(state.offered_slots, lang))
        return scripts["no_slots"]

    if not callback and state.phase == CallPhase.CONFIRMATION and state.booked_slot:
        slot = state.booked_slot
        return scripts["booking_done"].format(
            date=_fmt_date(slot["date"], lang),
            time=_fmt_time(slot["time"], lang),
            lawyer=slot.get("lawyer_name", ""),
        )

    if callback and state.phase == CallPhase.CONFIRMATION:
        # Say the closing only once, even if the caller adds "Tschüss" afterwards.
        last_agent = _last_agent_text(state)
        if "Wiederhören" in last_agent or "Goodbye" in last_agent:
            return None
        person = state.target_person or scripts.get("team", "")
        return scripts["callback_done"].format(person=person, time=state.preferred_time or "")

    return None
