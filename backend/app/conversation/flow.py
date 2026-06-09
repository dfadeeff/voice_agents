"""Deterministic state machine for call flow control.

next_phase() is a projection from accumulated state to phase — it looks at
what data has been collected and returns where we should be. Code controls
transitions; the LLM handles language understanding and natural phrasing.
"""

from app.models.schemas import CallerIntent, CallPhase, LegalArea

CONTACT_FIELDS = ("name", "email", "phone")
CALLBACK_REQUIRED_FIELDS = ("name", "phone")

PHASE_TOOLS: dict[CallPhase, list[str]] = {
    CallPhase.GREETING: [],
    # ROUTING is the one place the local LLM drives: it interprets messy free
    # speech into intent + legal area (keyword fallback backs it up).
    CallPhase.ROUTING: ["route_call", "request_handoff"],
    # QUALIFICATION, CAPTURE and BOOKING are driven by the scripted spine
    # (next_prompt fast-paths every turn), so the LLM never runs here — it only
    # keeps the handoff escape hatch.
    CallPhase.QUALIFICATION: ["request_handoff"],
    CallPhase.INFORMATION: ["route_call", "request_handoff"],
    CallPhase.CAPTURE: ["request_handoff"],
    CallPhase.BOOKING: ["request_handoff"],
    CallPhase.CONFIRMATION: [],
    # Non-callback escalation (out-of-scope area, repeated misunderstanding) is
    # still LLM-driven, so it keeps the capture tools.
    CallPhase.ESCALATION: ["capture_caller_details", "confirm_caller_detail", "request_handoff"],
}


def all_contacts_confirmed(state) -> bool:
    """Booking needs name + phone confirmed; email is optional (a phone number is
    enough), so it counts as resolved once given or explicitly skipped."""
    ents = state.entities

    def confirmed(field: str) -> bool:
        entity = ents.get(field)
        return bool(entity and entity.confirmed)

    if not (confirmed("name") and confirmed("phone")):
        return False
    return confirmed("email") or state.email_skipped


def callback_contacts_confirmed(entities: dict) -> bool:
    for f in CALLBACK_REQUIRED_FIELDS:
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
    if state.misunderstanding_streak >= 3:
        return CallPhase.ESCALATION

    if state.escalation_requested:
        return CallPhase.ESCALATION

    if state.booking_confirmed:
        return CallPhase.CONFIRMATION

    if state.callback_requested:
        # Name + phone, then a preferred callback time, then confirm.
        if callback_contacts_confirmed(state.entities) and state.preferred_time:
            return CallPhase.CONFIRMATION
        return CallPhase.CAPTURE

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

    # Traffic: the insurance/claim number must be asked before moving on — either
    # a number was captured or the caller was asked and had none. Other areas use
    # the matter_details follow-up instead.
    if state.legal_area == LegalArea.TRAFFIC:
        if "insurance_number" not in state.entities and not state.insurance_resolved:
            return CallPhase.QUALIFICATION
    elif "matter_details" not in state.entities:
        return CallPhase.QUALIFICATION

    if not all_contacts_confirmed(state):
        return CallPhase.CAPTURE

    return CallPhase.BOOKING
