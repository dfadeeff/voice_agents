"""Deterministic state machine for call flow control.

next_phase() is a projection from accumulated state to phase — it looks at
what data has been collected and returns where we should be. Code controls
transitions; the LLM handles language understanding and natural phrasing.
"""

from app.models.schemas import CallerIntent, CallPhase, LegalArea

REQUIRED_FIELDS = ("name", "email", "phone")

PHASE_TOOLS: dict[CallPhase, list[str]] = {
    CallPhase.GREETING: [],
    CallPhase.INTENT_DETECTION: ["classify_caller_intent", "escalate_to_human"],
    CallPhase.ROUTING: ["classify_legal_area", "escalate_to_human"],
    CallPhase.INFORMATION: ["classify_caller_intent", "escalate_to_human"],
    CallPhase.CAPTURE: ["extract_caller_details", "escalate_to_human"],
    CallPhase.BOOKING: ["check_availability", "book_consultation", "escalate_to_human"],
    CallPhase.CONFIRMATION: [],
    CallPhase.ESCALATION: ["escalate_to_human"],
    CallPhase.FAREWELL: [],
}


def all_required_confirmed(entities: dict) -> bool:
    for field in REQUIRED_FIELDS:
        entity = entities.get(field)
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
        return CallPhase.INTENT_DETECTION

    if state.legal_area == LegalArea.UNKNOWN:
        return CallPhase.ROUTING

    if state.caller_intent == CallerIntent.GENERAL_INFO:
        return CallPhase.INFORMATION

    if not all_required_confirmed(state.entities):
        return CallPhase.CAPTURE

    return CallPhase.BOOKING
