"""Structured call summary for handoff and evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.conversation.state import ConversationState


def _entity_value(state: ConversationState, field: str) -> str | None:
    entity = state.entities.get(field)
    return entity.value if entity else None


def generate_call_summary(state: ConversationState) -> dict:
    booking = None
    if state.booking_confirmed and state.booked_slot:
        slot = state.booked_slot
        booking = {
            "date": slot.get("date"),
            "time": slot.get("time"),
            "lawyer": slot.get("lawyer_name"),
        }

    escalation = None
    if state.escalation_requested:
        escalation = {
            "reason": state.escalation_reason or "unknown",
            "summary": state.escalation_summary or "",
        }

    return {
        "call_id": state.call_id,
        "caller_name": _entity_value(state, "name"),
        "email": _entity_value(state, "email"),
        "phone": _entity_value(state, "phone"),
        "legal_area": state.legal_area.value,
        "issue_summary": _entity_value(state, "matter_description"),
        "booking": booking,
        "escalation": escalation,
        "turn_count": state.turn_count,
        "phase": state.phase.value,
    }
