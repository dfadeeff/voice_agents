"""Tests for the deterministic state machine (flow.py)."""

from app.conversation.flow import CONTACT_FIELDS, PHASE_TOOLS, all_contacts_confirmed, next_phase
from app.conversation.state import ConversationState
from app.models.schemas import CallerIntent, CallPhase, ExtractedEntity, LegalArea


def _state(**kwargs) -> ConversationState:
    return ConversationState(call_id="test", **kwargs)


def _confirmed_entity(field, value="test"):
    return ExtractedEntity(field_name=field, value=value, confidence=0.95, confirmed=True)


def _unconfirmed_entity(field, value="test"):
    return ExtractedEntity(field_name=field, value=value, confidence=0.5, confirmed=False)


class TestNextPhase:
    def test_greeting_when_no_turns(self):
        assert next_phase(_state()) == CallPhase.GREETING

    def test_routing_after_first_turn(self):
        assert next_phase(_state(turn_count=1)) == CallPhase.ROUTING

    def test_routing_stays_if_intent_unknown(self):
        s = _state(turn_count=3, caller_intent=CallerIntent.UNKNOWN)
        assert next_phase(s) == CallPhase.ROUTING

    def test_routing_when_intent_known_area_unknown(self):
        s = _state(
            turn_count=2,
            caller_intent=CallerIntent.BOOK_CONSULTATION,
            legal_area=LegalArea.UNKNOWN,
        )
        assert next_phase(s) == CallPhase.ROUTING

    def test_information_for_general_info(self):
        s = _state(
            turn_count=2,
            caller_intent=CallerIntent.GENERAL_INFO,
            legal_area=LegalArea.EMPLOYMENT,
        )
        assert next_phase(s) == CallPhase.INFORMATION

    def test_qualification_when_matter_type_missing(self):
        s = _state(
            turn_count=2,
            caller_intent=CallerIntent.BOOK_CONSULTATION,
            legal_area=LegalArea.EMPLOYMENT,
        )
        assert next_phase(s) == CallPhase.QUALIFICATION

    def test_capture_when_matter_type_set_fields_missing(self):
        s = _state(
            turn_count=3,
            caller_intent=CallerIntent.BOOK_CONSULTATION,
            legal_area=LegalArea.EMPLOYMENT,
            entities={"matter_type": _confirmed_entity("matter_type", "dismissal")},
        )
        assert next_phase(s) == CallPhase.CAPTURE

    def test_capture_when_fields_unconfirmed(self):
        s = _state(
            turn_count=3,
            caller_intent=CallerIntent.BOOK_CONSULTATION,
            legal_area=LegalArea.TENANCY,
            entities={
                "matter_type": _confirmed_entity("matter_type", "deposit"),
                "name": _confirmed_entity("name"),
                "phone": _unconfirmed_entity("phone"),
            },
        )
        assert next_phase(s) == CallPhase.CAPTURE

    def test_booking_when_all_confirmed(self):
        s = _state(
            turn_count=7,
            caller_intent=CallerIntent.BOOK_CONSULTATION,
            legal_area=LegalArea.EMPLOYMENT,
            entities={
                "matter_type": _confirmed_entity("matter_type", "dismissal"),
                "name": _confirmed_entity("name", "John Smith"),
                "email": _confirmed_entity("email", "john@example.com"),
                "phone": _confirmed_entity("phone", "+1234567890"),
            },
        )
        assert next_phase(s) == CallPhase.BOOKING

    def test_confirmation_when_booked(self):
        s = _state(
            turn_count=8,
            caller_intent=CallerIntent.BOOK_CONSULTATION,
            legal_area=LegalArea.EMPLOYMENT,
            booking_confirmed=True,
        )
        assert next_phase(s) == CallPhase.CONFIRMATION

    def test_escalation_when_requested(self):
        s = _state(turn_count=3, escalation_requested=True)
        assert next_phase(s) == CallPhase.ESCALATION

    def test_escalation_on_misunderstanding_streak(self):
        s = _state(turn_count=5, misunderstanding_streak=3)
        assert next_phase(s) == CallPhase.ESCALATION

    def test_escalation_overrides_normal_transition(self):
        s = _state(
            turn_count=5,
            caller_intent=CallerIntent.BOOK_CONSULTATION,
            legal_area=LegalArea.EMPLOYMENT,
            entities={
                "matter_type": _confirmed_entity("matter_type", "dismissal"),
                "name": _confirmed_entity("name"),
                "email": _confirmed_entity("email"),
                "phone": _confirmed_entity("phone"),
            },
            escalation_requested=True,
        )
        assert next_phase(s) == CallPhase.ESCALATION

    def test_escalation_overrides_booking(self):
        s = _state(booking_confirmed=True, escalation_requested=True)
        assert next_phase(s) == CallPhase.ESCALATION

    def test_extra_entities_dont_affect_required(self):
        s = _state(
            turn_count=3,
            caller_intent=CallerIntent.BOOK_CONSULTATION,
            legal_area=LegalArea.EMPLOYMENT,
            entities={
                "matter_type": _confirmed_entity("matter_type", "dismissal"),
                "name": _confirmed_entity("name"),
                "matter_description": _confirmed_entity("matter_description"),
            },
        )
        assert next_phase(s) == CallPhase.CAPTURE


class TestAllContactsConfirmed:
    def test_empty_entities(self):
        assert all_contacts_confirmed({}) is False

    def test_partial_entities(self):
        entities = {"name": _confirmed_entity("name")}
        assert all_contacts_confirmed(entities) is False

    def test_all_confirmed(self):
        entities = {
            "name": _confirmed_entity("name"),
            "email": _confirmed_entity("email"),
            "phone": _confirmed_entity("phone"),
        }
        assert all_contacts_confirmed(entities) is True

    def test_one_unconfirmed(self):
        entities = {
            "name": _confirmed_entity("name"),
            "phone": _unconfirmed_entity("phone"),
        }
        assert all_contacts_confirmed(entities) is False


class TestContactFields:
    def test_contact_fields_are_name_email_phone(self):
        assert CONTACT_FIELDS == ("name", "email", "phone")


class TestPhaseTools:
    def test_greeting_has_no_tools(self):
        assert PHASE_TOOLS[CallPhase.GREETING] == []

    def test_routing_tools(self):
        tools = PHASE_TOOLS[CallPhase.ROUTING]
        assert "route_call" in tools
        assert "request_handoff" in tools

    def test_qualification_tools(self):
        tools = PHASE_TOOLS[CallPhase.QUALIFICATION]
        assert "capture_caller_details" in tools
        assert "request_handoff" in tools

    def test_capture_tools(self):
        tools = PHASE_TOOLS[CallPhase.CAPTURE]
        assert "capture_caller_details" in tools
        assert "confirm_caller_detail" in tools
        assert "request_handoff" in tools
        assert "book_consultation" not in tools

    def test_booking_tools(self):
        tools = PHASE_TOOLS[CallPhase.BOOKING]
        assert "check_availability" in tools
        assert "book_consultation" in tools
        assert "request_handoff" in tools

    def test_confirmation_has_no_tools(self):
        assert PHASE_TOOLS[CallPhase.CONFIRMATION] == []

    def test_information_tools(self):
        tools = PHASE_TOOLS[CallPhase.INFORMATION]
        assert "route_call" in tools
        assert "request_handoff" in tools

    def test_all_phases_covered(self):
        for phase in CallPhase:
            assert phase in PHASE_TOOLS
