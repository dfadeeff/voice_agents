"""Tests for all tool handlers — intent, routing, extraction, booking, escalation."""

import pytest

from app.models.schemas import CallerIntent, CallPhase, LegalArea


class TestClassifyCallerIntent:
    @pytest.mark.asyncio
    async def test_book_consultation(self, registry, conversation):
        result = await registry.execute(
            "classify_caller_intent",
            {"intent": "book_consultation"},
            conversation,
        )
        assert result["status"] == "classified"
        assert result["intent"] == "book_consultation"
        assert conversation.state.caller_intent == CallerIntent.BOOK_CONSULTATION

    @pytest.mark.asyncio
    async def test_general_info(self, registry, conversation):
        result = await registry.execute(
            "classify_caller_intent",
            {"intent": "general_info"},
            conversation,
        )
        assert result["status"] == "classified"
        assert result["intent"] == "general_info"

    @pytest.mark.asyncio
    async def test_empty_intent_returns_need_more_info(self, registry, conversation):
        result = await registry.execute(
            "classify_caller_intent", {"intent": ""}, conversation
        )
        assert result["status"] == "need_more_info"

    @pytest.mark.asyncio
    async def test_missing_intent_returns_need_more_info(self, registry, conversation):
        result = await registry.execute(
            "classify_caller_intent", {}, conversation
        )
        assert result["status"] == "need_more_info"

    @pytest.mark.asyncio
    async def test_invalid_intent_becomes_unknown(self, registry, conversation):
        result = await registry.execute(
            "classify_caller_intent",
            {"intent": "complain_loudly"},
            conversation,
        )
        assert result["status"] == "classified"
        assert conversation.state.caller_intent == CallerIntent.UNKNOWN


class TestClassifyLegalArea:
    @pytest.mark.asyncio
    async def test_employment(self, registry, conversation):
        result = await registry.execute(
            "classify_legal_area",
            {"legal_area": "employment"},
            conversation,
        )
        assert result["status"] == "routed"
        assert result["legal_area"] == "employment"
        assert conversation.state.legal_area == LegalArea.EMPLOYMENT
        assert conversation.state.phase == CallPhase.ROUTING

    @pytest.mark.asyncio
    async def test_tenancy(self, registry, conversation):
        result = await registry.execute(
            "classify_legal_area",
            {"legal_area": "tenancy"},
            conversation,
        )
        assert result["status"] == "routed"
        assert result["legal_area"] == "tenancy"

    @pytest.mark.asyncio
    async def test_unknown_area_triggers_escalation(self, registry, conversation):
        result = await registry.execute(
            "classify_legal_area",
            {"legal_area": "unknown"},
            conversation,
        )
        assert result["status"] == "unknown_area"
        assert conversation.state.phase == CallPhase.ESCALATION

    @pytest.mark.asyncio
    async def test_invalid_area_becomes_unknown(self, registry, conversation):
        result = await registry.execute(
            "classify_legal_area",
            {"legal_area": "criminal"},
            conversation,
        )
        assert result["status"] == "unknown_area"
        assert conversation.state.legal_area == LegalArea.UNKNOWN


class TestExtractCallerDetails:
    @pytest.mark.asyncio
    async def test_high_confidence_auto_confirms(
        self, registry, conversation_with_high_confidence
    ):
        ctx = conversation_with_high_confidence
        result = await registry.execute(
            "extract_caller_details",
            {"name": "John Smith", "email": "john@example.com"},
            ctx,
        )
        assert "name" in result["stored"]
        assert ctx.state.entities["name"].confirmed is True

    @pytest.mark.asyncio
    async def test_low_confidence_needs_confirmation(
        self, registry, conversation_with_low_confidence
    ):
        ctx = conversation_with_low_confidence
        result = await registry.execute(
            "extract_caller_details",
            {"name": "Siobhan Murphy"},
            ctx,
        )
        assert not result["all_confirmed"]
        assert len(result["needs_confirmation"]) == 1
        assert result["needs_confirmation"][0]["field"] == "name"
        assert not ctx.state.entities["name"].confirmed

    @pytest.mark.asyncio
    async def test_non_contact_fields_skip_confirmation(
        self, registry, conversation_with_low_confidence
    ):
        ctx = conversation_with_low_confidence
        result = await registry.execute(
            "extract_caller_details",
            {"matter_description": "unfair dismissal"},
            ctx,
        )
        assert result["all_confirmed"]
        assert "matter_description" in result["stored"]

    @pytest.mark.asyncio
    async def test_empty_args(self, registry, conversation):
        result = await registry.execute(
            "extract_caller_details", {}, conversation
        )
        assert result["stored"] == []
        assert result["all_confirmed"]

    @pytest.mark.asyncio
    async def test_non_string_values_ignored(self, registry, conversation):
        conversation.add_user_message("test")
        result = await registry.execute(
            "extract_caller_details",
            {"name": "John", "debug_mode": True},
            conversation,
        )
        assert "name" in result["stored"]
        assert "debug_mode" not in result["stored"]


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
    async def test_booking_blocked_without_confirmed_entities(
        self, registry, conversation
    ):
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

        result = await registry.execute(
            "book_consultation", {"caller_name": "test"}, conversation
        )
        assert result["status"] == "error"


class TestEscalateToHuman:
    @pytest.mark.asyncio
    async def test_escalation_sets_state(self, registry, conversation):
        result = await registry.execute(
            "escalate_to_human",
            {"reason": "caller_requested_human", "summary": "Wants to speak to lawyer"},
            conversation,
        )
        assert result["status"] == "escalating"
        assert conversation.state.escalation_requested is True
        assert conversation.state.phase == CallPhase.ESCALATION

    @pytest.mark.asyncio
    async def test_escalation_includes_context(self, registry, conversation):
        conversation.set_legal_area(LegalArea.EMPLOYMENT)
        conversation.add_user_message("I need a human")
        conversation.store_entity("name", "Jane Doe", 0.9)

        result = await registry.execute(
            "escalate_to_human",
            {"reason": "caller_requested_human"},
            conversation,
        )
        ctx = result["context_for_human"]
        assert ctx["legal_area"] == "employment"
        assert ctx["caller_details"]["name"] == "Jane Doe"
        assert ctx["call_id"] == conversation.state.call_id
