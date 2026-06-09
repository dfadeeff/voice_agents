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


def compute_prompt(state: ConversationState, lang: str = "de") -> tuple[str | None, str | None]:
    """Compute the next deterministic question/line and what it asks for.

    Returns ``(spoken_line, awaiting)``. The whole data-collection spine
    (qualification → name/email/phone → slot → confirmation, plus the callback
    branch) is driven from here so the local LLM never gets a turn it can drift
    on. Returns ``(None, None)`` for ROUTING and INFORMATION, where the LLM
    legitimately drives, and for GREETING (spoken on connect).

    ``awaiting`` is the datum the line asks for; ConversationManager stores it on
    the state and the reply parser dispatches on it (no regex on prior speech).
    """
    scripts = getattr(get_locale(lang), "SCRIPTED", {})
    if not scripts:
        return None, None

    ents = state.entities
    name = ents.get("name")
    email = ents.get("email")
    phone = ents.get("phone")
    phase = state.phase

    # ----- Callback / human-handoff branch: name → phone → callback time -----
    if state.callback_requested:
        if phase in (CallPhase.CAPTURE, CallPhase.ESCALATION):
            if not (name and name.confirmed):
                return scripts["ask_name"], "name"
            if phone and not phone.confirmed:
                return scripts["confirm_phone"].format(phone=_spoken_phone(phone.value, lang)), (
                    "phone_confirm"
                )
            if not phone:
                return scripts["ask_phone"], "phone"
            if not state.preferred_time:
                return scripts["ask_callback_time"], "callback_time"
            return None, None
        if phase == CallPhase.CONFIRMATION:
            # Say the closing only once, even if the caller adds "Tschüss" after.
            last_agent = _last_agent_text(state)
            if "Wiederhören" in last_agent or "oodbye" in last_agent:
                return None, None
            person = state.target_person or scripts.get("team", "")
            done = scripts["callback_done"].format(person=person, time=state.preferred_time or "")
            return done, None
        return None, None

    # ----- Booking / intake branch -----
    area = state.legal_area.value

    if phase == CallPhase.QUALIFICATION:
        if "matter_type" not in ents:
            return scripts.get(f"{area}_confirm"), "matter_type"
        if area == "traffic":
            if "insurance_number" not in ents and not state.insurance_resolved:
                return scripts.get("traffic_insurance"), "insurance"
        elif "matter_details" not in ents:
            return scripts.get(f"{area}_details"), "matter_details"
        return None, None

    if phase == CallPhase.CAPTURE:
        if not name:
            return scripts["ask_name_booking"], "name"
        if not name.confirmed:
            return scripts["confirm_name"].format(name=name.value), "name_confirm"
        if email and not email.confirmed:
            return scripts["confirm_email"].format(email=email.value), "email_confirm"
        if not email and not state.email_skipped:
            if state.email_misheard:
                return scripts["ask_email_not_understood"], "email"
            return scripts["ask_email"], "email"
        if phone and not phone.confirmed:
            return scripts["confirm_phone"].format(phone=_spoken_phone(phone.value, lang)), (
                "phone_confirm"
            )
        if not phone:
            return scripts["ask_phone"], "phone"
        return None, None

    if phase == CallPhase.BOOKING and not state.booking_confirmed:
        if state.offered_slots:
            return scripts["slot_offer"].format(options=_fmt_slots(state.offered_slots, lang)), (
                "slot"
            )
        return scripts["no_slots"], None

    if phase == CallPhase.CONFIRMATION and state.booked_slot:
        last_agent = _last_agent_text(state)
        if "gebucht" in last_agent or "booked" in last_agent:
            return scripts.get("goodbye"), None
        slot = state.booked_slot
        return scripts["booking_done"].format(
            date=_fmt_date(slot["date"], lang),
            time=_fmt_time(slot["time"], lang),
        ), None

    return None, None


def scripted_line(state: ConversationState, lang: str = "de") -> str | None:
    """Back-compat accessor: the spoken line only (without the ``awaiting`` tag)."""
    return compute_prompt(state, lang)[0]
