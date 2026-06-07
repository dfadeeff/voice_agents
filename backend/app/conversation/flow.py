"""Deterministic state machine for call flow control.

next_phase() is a projection from accumulated state to phase — it looks at
what data has been collected and returns where we should be. Code controls
transitions; the LLM handles language understanding and natural phrasing.
"""

from app.models.schemas import CallerIntent, CallPhase, LegalArea

CONTACT_FIELDS = ("name", "email", "phone")

PHASE_TOOLS: dict[CallPhase, list[str]] = {
    CallPhase.GREETING: [],
    CallPhase.ROUTING: ["route_call", "request_handoff"],
    CallPhase.QUALIFICATION: ["capture_caller_details", "request_handoff"],
    CallPhase.INFORMATION: ["route_call", "request_handoff"],
    CallPhase.CAPTURE: ["capture_caller_details", "confirm_caller_detail", "request_handoff"],
    CallPhase.BOOKING: ["check_availability", "book_consultation", "request_handoff"],
    CallPhase.CONFIRMATION: [],
    CallPhase.ESCALATION: ["request_handoff"],
}


def all_contacts_confirmed(entities: dict) -> bool:
    for f in CONTACT_FIELDS:
        entity = entities.get(f)
        if not entity or not entity.confirmed:
            return False
    return True


def next_phase(state) -> CallPhase:
    """Compute the correct phase from accumulated state.

    This is a pure projection — it doesn't care what the current phase is,
    only what data exists. Makes it idempotent and robust to out-of-order
    tool calls (e.g. caller gives name + email + legal issue in one sentence).
    """
    if state.escalation_requested or state.misunderstanding_streak >= 3:
        return CallPhase.ESCALATION

    if state.booking_confirmed:
        return CallPhase.CONFIRMATION

    if state.caller_intent == CallerIntent.UNKNOWN:
        if state.turn_count < 1:
            return CallPhase.GREETING
        return CallPhase.ROUTING

    if state.legal_area == LegalArea.UNKNOWN:
        return CallPhase.ROUTING

    if state.caller_intent == CallerIntent.GENERAL_INFO:
        return CallPhase.INFORMATION

    if "matter_type" not in state.entities:
        return CallPhase.QUALIFICATION

    if not all_contacts_confirmed(state.entities):
        return CallPhase.CAPTURE

    return CallPhase.BOOKING
