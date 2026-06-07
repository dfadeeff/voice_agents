"""Tests for all tool handlers — route, extraction, booking, handoff."""

import pytest
from app.models.schemas import CallerIntent, CallPhase, LegalArea


class TestRouteCall:
    @pytest.mark.asyncio
    async def test_book_consultation_employment(self, registry, conversation):
        conversation.add_user_message("I was fired last week, I need a consultation")
        result = await registry.execute(
            "route_call",
            {
                "intent": "book_consultation",
                "legal_area": "employment",
                "matter_summary": "unfair dismissal",
            },
            conversation,
        )
        assert result["status"] == "routed"
        assert result["intent"] == "book_consultation"
        assert result["legal_area"] == "employment"
        assert conversation.state.caller_intent == CallerIntent.BOOK_CONSULTATION
        assert conversation.state.legal_area == LegalArea.EMPLOYMENT
        assert conversation.state.matter_summary == "unfair dismissal"
        assert conversation.state.phase == CallPhase.QUALIFICATION

    @pytest.mark.asyncio
    async def test_general_info(self, registry, conversation):
        conversation.add_user_message("I have a question about employment law")
        result = await registry.execute(
            "route_call",
            {"intent": "general_info", "legal_area": "employment"},
            conversation,
        )
        assert result["status"] == "routed"
        assert result["intent"] == "general_info"
        assert conversation.state.caller_intent == CallerIntent.GENERAL_INFO
        assert conversation.state.phase == CallPhase.INFORMATION

    @pytest.mark.asyncio
    async def test_unknown_area_escalates(self, registry, conversation):
        conversation.add_user_message("I need immigration help")
        result = await registry.execute(
            "route_call",
            {"intent": "book_consultation", "legal_area": "unknown"},
            conversation,
        )
        assert result["status"] == "unknown_area"
        assert conversation.state.escalation_requested is True
        assert conversation.state.phase == CallPhase.ESCALATION

    @pytest.mark.asyncio
    async def test_invalid_area_becomes_unknown(self, registry, conversation):
        conversation.add_user_message("I need criminal defense")
        result = await registry.execute(
            "route_call",
            {"intent": "book_consultation", "legal_area": "criminal"},
            conversation,
        )
        assert result["status"] == "unknown_area"
        assert conversation.state.legal_area == LegalArea.UNKNOWN

    @pytest.mark.asyncio
    async def test_empty_intent_returns_need_more_info(self, registry, conversation):
        result = await registry.execute(
            "route_call", {"intent": "", "legal_area": "employment"}, conversation
        )
        assert result["status"] == "need_more_info"

    @pytest.mark.asyncio
    async def test_missing_intent_returns_need_more_info(self, registry, conversation):
        result = await registry.execute("route_call", {"legal_area": "employment"}, conversation)
        assert result["status"] == "need_more_info"

    @pytest.mark.asyncio
    async def test_invalid_intent_becomes_unknown(self, registry, conversation):
        conversation.add_user_message("test")
        result = await registry.execute(
            "route_call",
            {"intent": "complain_loudly", "legal_area": "employment"},
            conversation,
        )
        assert result["status"] == "routed"
        assert conversation.state.caller_intent == CallerIntent.UNKNOWN


class TestCaptureCallerDetails:
    @pytest.mark.asyncio
    async def test_name_auto_confirms(self, registry, conversation):
        conversation.add_user_message("My name is John Smith")
        result = await registry.execute(
            "capture_caller_details",
            {"name": "John Smith"},
            conversation,
        )
        assert "name" in result["stored"]
        assert conversation.state.entities["name"].confirmed is True

    @pytest.mark.asyncio
    async def test_low_confidence_name_requires_confirmation(self, registry, conversation):
        conversation.add_user_message("My name is Jane Smith")
        conversation.state.last_transcription_confidence = 0.42

        result = await registry.execute(
            "capture_caller_details",
            {"name": "Jane Smith"},
            conversation,
        )

        assert conversation.state.entities["name"].confidence == 0.42
        assert conversation.state.entities["name"].confirmed is False
        assert result["needs_confirmation"][0]["field"] == "name"
        assert result["needs_confirmation"][0]["confidence"] == 0.42

    @pytest.mark.asyncio
    async def test_email_needs_confirmation(self, registry, conversation):
        conversation.add_user_message("john at example dot com")
        result = await registry.execute(
            "capture_caller_details",
            {"email": "john@example.com"},
            conversation,
        )
        assert "email" in result["stored"]
        assert not result["all_confirmed"]
        assert any(c["field"] == "email" for c in result["needs_confirmation"])
        assert not conversation.state.entities["email"].confirmed

    @pytest.mark.asyncio
    async def test_phone_needs_confirmation(self, registry, conversation):
        conversation.add_user_message("+1234567890")
        result = await registry.execute(
            "capture_caller_details",
            {"phone": "+1234567890"},
            conversation,
        )
        assert "phone" in result["stored"]
        assert not result["all_confirmed"]
        assert any(c["field"] == "phone" for c in result["needs_confirmation"])

    @pytest.mark.asyncio
    async def test_matter_type_auto_confirms(self, registry, conversation):
        conversation.add_user_message("It's about a dismissal")
        result = await registry.execute(
            "capture_caller_details",
            {"matter_type": "dismissal"},
            conversation,
        )
        assert "matter_type" in result["stored"]
        assert result["all_confirmed"]
        assert conversation.state.entities["matter_type"].confirmed is True

    @pytest.mark.asyncio
    async def test_empty_args(self, registry, conversation):
        result = await registry.execute("capture_caller_details", {}, conversation)
        assert result["stored"] == []
        assert result["all_confirmed"]

    @pytest.mark.asyncio
    async def test_non_string_values_ignored(self, registry, conversation):
        conversation.add_user_message("test")
        result = await registry.execute(
            "capture_caller_details",
            {"name": "John", "debug_mode": True},
            conversation,
        )
        assert "name" in result["stored"]
        assert "debug_mode" not in result["stored"]

    @pytest.mark.asyncio
    async def test_invalid_email_format_rejected(self, registry, conversation):
        conversation.add_user_message("my email is not-an-email")
        result = await registry.execute(
            "capture_caller_details",
            {"email": "not-an-email"},
            conversation,
        )
        assert "email" not in result["stored"]
        assert len(result.get("format_errors", [])) == 1
        assert result["format_errors"][0]["field"] == "email"

    @pytest.mark.asyncio
    async def test_valid_email_accepted(self, registry, conversation):
        conversation.add_user_message("fadejeff at gmail dot com")
        result = await registry.execute(
            "capture_caller_details",
            {"email": "fadejeff@gmail.com"},
            conversation,
        )
        assert "email" in result["stored"]

    @pytest.mark.asyncio
    async def test_invalid_phone_rejected(self, registry, conversation):
        conversation.add_user_message("my number is abc")
        result = await registry.execute(
            "capture_caller_details",
            {"phone": "abc"},
            conversation,
        )
        assert "phone" not in result["stored"]
        assert len(result.get("format_errors", [])) == 1


class TestConfirmCallerDetail:
    @pytest.mark.asyncio
    async def test_accept_existing_email(self, registry, conversation):
        conversation.add_user_message("john@example.com")
        await registry.execute(
            "capture_caller_details", {"email": "john@example.com"}, conversation
        )
        assert not conversation.state.entities["email"].confirmed

        result = await registry.execute(
            "confirm_caller_detail",
            {"field": "email", "confirmed_value": "john@example.com", "status": "accepted"},
            conversation,
        )
        assert result["status"] == "confirmed"
        assert result["was_corrected"] is False
        assert conversation.state.entities["email"].confirmed is True

    @pytest.mark.asyncio
    async def test_correct_email(self, registry, conversation):
        conversation.add_user_message("jon@example.com")
        await registry.execute("capture_caller_details", {"email": "jon@example.com"}, conversation)

        result = await registry.execute(
            "confirm_caller_detail",
            {"field": "email", "confirmed_value": "john@example.com", "status": "corrected"},
            conversation,
        )
        assert result["status"] == "confirmed"
        assert result["was_corrected"] is True
        assert conversation.state.entities["email"].value == "john@example.com"
        assert conversation.state.entities["email"].confirmed is True

    @pytest.mark.asyncio
    async def test_accept_phone(self, registry, conversation):
        conversation.add_user_message("+1234567890")
        await registry.execute("capture_caller_details", {"phone": "+1234567890"}, conversation)

        result = await registry.execute(
            "confirm_caller_detail",
            {"field": "phone", "confirmed_value": "+1234567890", "status": "accepted"},
            conversation,
        )
        assert result["status"] == "confirmed"
        assert conversation.state.entities["phone"].confirmed is True

    @pytest.mark.asyncio
    async def test_correct_phone(self, registry, conversation):
        conversation.add_user_message("+1234567890")
        await registry.execute("capture_caller_details", {"phone": "+1234567890"}, conversation)

        result = await registry.execute(
            "confirm_caller_detail",
            {"field": "phone", "confirmed_value": "+1234567891", "status": "corrected"},
            conversation,
        )
        assert result["status"] == "confirmed"
        assert result["was_corrected"] is True
        assert conversation.state.entities["phone"].value == "+1234567891"

    @pytest.mark.asyncio
    async def test_error_on_missing_field(self, registry, conversation):
        result = await registry.execute(
            "confirm_caller_detail",
            {"field": "email", "confirmed_value": "test@test.com", "status": "accepted"},
            conversation,
        )
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_error_on_invalid_status(self, registry, conversation):
        result = await registry.execute(
            "confirm_caller_detail",
            {"field": "email", "confirmed_value": "test@test.com", "status": "maybe"},
            conversation,
        )
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_invalid_correction_rejected(self, registry, conversation):
        conversation.add_user_message("test@test.com")
        await registry.execute("capture_caller_details", {"email": "test@test.com"}, conversation)

        result = await registry.execute(
            "confirm_caller_detail",
            {"field": "email", "confirmed_value": "not-valid", "status": "corrected"},
            conversation,
        )
        assert result["status"] == "invalid"


class TestCheckAvailability:
    @pytest.mark.asyncio
    async def test_available_slots(self, registry, conversation):
        result = await registry.execute(
            "check_availability",
            {"date": "2026-06-10", "legal_area": "employment"},
            conversation,
        )
        assert result["available"] is True
        assert len(result["slots"]) > 0
        assert result["slots"][0]["lawyer"] == "Sarah Chen"

    @pytest.mark.asyncio
    async def test_stores_offered_slot_ids(self, registry, conversation):
        result = await registry.execute(
            "check_availability",
            {"date": "2026-06-10", "legal_area": "employment"},
            conversation,
        )
        assert len(conversation.state.offered_slot_ids) > 0
        assert conversation.state.offered_slot_ids == [s["id"] for s in result["slots"]]

    @pytest.mark.asyncio
    async def test_no_slots_returns_alternatives(self, registry, conversation):
        result = await registry.execute(
            "check_availability",
            {"date": "2026-06-15"},
            conversation,
        )
        assert result["available"] is False
        assert "alternatives" in result

    @pytest.mark.asyncio
    async def test_time_preference_morning(self, registry, conversation):
        result = await registry.execute(
            "check_availability",
            {"date": "2026-06-10", "time_preference": "morning"},
            conversation,
        )
        assert result["available"] is True
        for slot in result["slots"]:
            assert slot["time"] < "12:00"

    @pytest.mark.asyncio
    async def test_time_preference_afternoon(self, registry, conversation):
        result = await registry.execute(
            "check_availability",
            {
                "date": "2026-06-10",
                "time_preference": "afternoon",
                "legal_area": "tenancy",
            },
            conversation,
        )
        assert result["available"] is True
        for slot in result["slots"]:
            assert slot["time"] >= "12:00"


class TestBookConsultation:
    @pytest.mark.asyncio
    async def test_booking_blocked_without_confirmed_entities(self, registry, conversation):
        conversation.add_user_message("John Smith")
        conversation.store_entity("name", "John Smith", 0.9)
        result = await registry.execute(
            "book_consultation",
            {"slot_id": 1, "caller_name": "John Smith"},
            conversation,
        )
        assert result["status"] == "blocked"

    @pytest.mark.asyncio
    async def test_successful_booking(self, registry, conversation):
        conversation.add_user_message("details")
        for field in ("name", "email", "phone"):
            conversation.store_entity(field, f"test_{field}", 0.9)
            conversation.confirm_entity(field)

        result = await registry.execute(
            "book_consultation",
            {"slot_id": 1, "caller_name": "test_name"},
            conversation,
        )
        assert result["status"] == "booked"
        assert conversation.state.booking_confirmed is True
        assert conversation.state.phase == CallPhase.CONFIRMATION

    @pytest.mark.asyncio
    async def test_rejects_unoffered_slot(self, registry, conversation):
        conversation.add_user_message("details")
        for field in ("name", "email", "phone"):
            conversation.store_entity(field, f"test_{field}", 0.9)
            conversation.confirm_entity(field)

        await registry.execute(
            "check_availability",
            {"date": "2026-06-10", "legal_area": "employment"},
            conversation,
        )
        result = await registry.execute(
            "book_consultation",
            {"slot_id": 999, "caller_name": "test_name"},
            conversation,
        )
        assert result["status"] == "error"
        assert "not offered" in result["message"]

    @pytest.mark.asyncio
    async def test_double_booking_fails(self, registry, conversation):
        conversation.add_user_message("details")
        for field in ("name", "email", "phone"):
            conversation.store_entity(field, f"test_{field}", 0.9)
            conversation.confirm_entity(field)

        await registry.execute(
            "book_consultation",
            {"slot_id": 1, "caller_name": "test_name"},
            conversation,
        )
        result = await registry.execute(
            "book_consultation",
            {"slot_id": 1, "caller_name": "other_person"},
            conversation,
        )
        assert result["status"] == "error"
        assert "no longer available" in result["message"]

    @pytest.mark.asyncio
    async def test_missing_slot_id(self, registry, conversation):
        conversation.add_user_message("details")
        for field in ("name", "email", "phone"):
            conversation.store_entity(field, f"test_{field}", 0.9)
            conversation.confirm_entity(field)

        result = await registry.execute("book_consultation", {"caller_name": "test"}, conversation)
        assert result["status"] == "error"


class TestRequestHandoff:
    @pytest.mark.asyncio
    async def test_handoff_sets_state(self, registry, conversation):
        result = await registry.execute(
            "request_handoff",
            {"reason": "caller_requested_human", "summary": "Wants to speak to lawyer"},
            conversation,
        )
        assert result["status"] == "handoff_requested"
        assert result["mode"] == "callback"
        assert conversation.state.escalation_requested is True
        assert conversation.state.phase == CallPhase.ESCALATION

    @pytest.mark.asyncio
    async def test_handoff_includes_context(self, registry, conversation):
        conversation.add_user_message("I was dismissed")
        conversation.set_route(
            CallerIntent.BOOK_CONSULTATION, LegalArea.EMPLOYMENT, "unfair dismissal"
        )
        conversation.store_entity("name", "Jane Doe", 0.9)

        result = await registry.execute(
            "request_handoff",
            {"reason": "caller_requested_human"},
            conversation,
        )
        ctx = result["context_for_human"]
        assert ctx["legal_area"] == "employment"
        assert ctx["matter_summary"] == "unfair dismissal"
        assert ctx["caller_details"]["name"] == "Jane Doe"
        assert ctx["call_id"] == conversation.state.call_id
