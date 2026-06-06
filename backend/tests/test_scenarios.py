"""Scenario-level tests proving the agent handles required user stories.

These test the full state machine + tool pipeline for realistic call flows,
not individual tools in isolation (that's test_tools.py).
"""

import pytest
from app.conversation.manager import ConversationManager
from app.models.schemas import CallerIntent, CallPhase, LegalArea


@pytest.fixture
def ctx():
    return ConversationManager(call_id="scenario-test")


class TestEmploymentRoutingScenario:
    """Caller with an employment issue routes correctly and reaches capture."""

    @pytest.mark.asyncio
    async def test_employment_booking_flow(self, registry, ctx):
        ctx.add_user_message("I was fired last week and I think it was unfair.")
        assert ctx.state.phase == CallPhase.INTENT_DETECTION

        await registry.execute(
            "classify_caller_intent",
            {"intent": "book_consultation"},
            ctx,
        )
        assert ctx.state.phase == CallPhase.ROUTING

        await registry.execute(
            "classify_legal_area",
            {"legal_area": "employment"},
            ctx,
        )
        assert ctx.state.phase == CallPhase.CAPTURE
        assert ctx.state.legal_area == LegalArea.EMPLOYMENT


class TestTenancyRoutingScenario:
    """Caller with a tenancy issue routes correctly."""

    @pytest.mark.asyncio
    async def test_tenancy_routing(self, registry, ctx):
        ctx.add_user_message("My landlord is refusing to return my deposit.")

        await registry.execute(
            "classify_caller_intent",
            {"intent": "book_consultation"},
            ctx,
        )
        await registry.execute(
            "classify_legal_area",
            {"legal_area": "tenancy"},
            ctx,
        )
        assert ctx.state.legal_area == LegalArea.TENANCY
        assert ctx.state.phase == CallPhase.CAPTURE


class TestUnknownAreaEscalationScenario:
    """Caller with an unsupported legal area (family, criminal, immigration) escalates."""

    @pytest.mark.asyncio
    async def test_family_law_escalates(self, registry, ctx):
        ctx.add_user_message("I need help with child custody after my divorce.")

        await registry.execute(
            "classify_caller_intent",
            {"intent": "book_consultation"},
            ctx,
        )
        result = await registry.execute(
            "classify_legal_area",
            {"legal_area": "unknown"},
            ctx,
        )
        assert result["status"] == "unknown_area"
        assert ctx.state.escalation_requested is True
        assert ctx.state.phase == CallPhase.ESCALATION

    @pytest.mark.asyncio
    async def test_invalid_area_string_escalates(self, registry, ctx):
        ctx.add_user_message("I need criminal defense.")

        await registry.execute(
            "classify_caller_intent",
            {"intent": "book_consultation"},
            ctx,
        )
        result = await registry.execute(
            "classify_legal_area",
            {"legal_area": "criminal"},
            ctx,
        )
        assert result["status"] == "unknown_area"
        assert ctx.state.phase == CallPhase.ESCALATION


class TestHumanHandoffScenario:
    """Caller explicitly asks for a human at any point."""

    @pytest.mark.asyncio
    async def test_immediate_handoff(self, registry, ctx):
        ctx.add_user_message("I want to speak to a person.")

        result = await registry.execute(
            "escalate_to_human",
            {"reason": "caller_requested_human", "summary": "Wants to speak to a lawyer"},
            ctx,
        )
        assert result["status"] == "escalating"
        assert ctx.state.phase == CallPhase.ESCALATION

    @pytest.mark.asyncio
    async def test_handoff_mid_flow(self, registry, ctx):
        ctx.add_user_message("I was dismissed")
        await registry.execute(
            "classify_caller_intent",
            {"intent": "book_consultation"},
            ctx,
        )
        assert ctx.state.phase == CallPhase.ROUTING

        ctx.add_user_message("Actually, can I just talk to someone?")
        result = await registry.execute(
            "escalate_to_human",
            {"reason": "caller_requested_human"},
            ctx,
        )
        assert ctx.state.phase == CallPhase.ESCALATION
        assert result["context_for_human"]["caller_details"] == {}


class TestLowConfidenceEmailScenario:
    """Email with low STT confidence requires verbal confirmation."""

    @pytest.mark.asyncio
    async def test_low_confidence_email_needs_confirmation(
        self, registry, conversation_with_low_confidence
    ):
        ctx = conversation_with_low_confidence
        ctx.set_intent(CallerIntent.BOOK_CONSULTATION)
        ctx.set_legal_area(LegalArea.EMPLOYMENT)

        result = await registry.execute(
            "extract_caller_details",
            {"email": "fadejeff@gmail.com"},
            ctx,
        )
        assert not result["all_confirmed"]
        assert len(result["needs_confirmation"]) == 1
        assert result["needs_confirmation"][0]["field"] == "email"
        assert not ctx.state.entities["email"].confirmed
        assert ctx.state.phase == CallPhase.CAPTURE


class TestUnavailableSlotScenario:
    """Caller wants a time that's not available — agent offers alternatives."""

    @pytest.mark.asyncio
    async def test_unavailable_slot_offers_alternatives(self, registry, ctx):
        ctx.add_user_message("details")
        ctx.set_intent(CallerIntent.BOOK_CONSULTATION)
        ctx.set_legal_area(LegalArea.EMPLOYMENT)
        for field in ("name", "email", "phone"):
            ctx.store_entity(field, f"test_{field}", 0.9)
            ctx.confirm_entity(field)

        assert ctx.state.phase == CallPhase.BOOKING

        result = await registry.execute(
            "check_availability",
            {"date": "2026-06-15"},
            ctx,
        )
        assert result["available"] is False
        assert "alternatives" in result


class TestFullBookingScenario:
    """Complete happy path: greeting → routing → capture → booking → confirmation."""

    @pytest.mark.asyncio
    async def test_full_happy_path(self, registry, ctx):
        assert ctx.state.phase == CallPhase.GREETING

        ctx.add_user_message("Hi, I need help with an employment issue.")
        assert ctx.state.phase == CallPhase.INTENT_DETECTION

        await registry.execute(
            "classify_caller_intent",
            {"intent": "book_consultation"},
            ctx,
        )
        assert ctx.state.phase == CallPhase.ROUTING

        await registry.execute(
            "classify_legal_area",
            {"legal_area": "employment"},
            ctx,
        )
        assert ctx.state.phase == CallPhase.CAPTURE

        ctx.add_user_message("My name is John Smith")
        await registry.execute(
            "extract_caller_details",
            {"name": "John Smith"},
            ctx,
        )

        ctx.add_user_message("john at example dot com")
        await registry.execute(
            "extract_caller_details",
            {"email": "john@example.com"},
            ctx,
        )

        ctx.add_user_message("plus one two three four five six seven eight nine zero")
        await registry.execute(
            "extract_caller_details",
            {"phone": "+1234567890"},
            ctx,
        )

        for field in ("name", "email", "phone"):
            ctx.confirm_entity(field)

        assert ctx.state.phase == CallPhase.BOOKING

        avail = await registry.execute(
            "check_availability",
            {"date": "2026-06-10", "legal_area": "employment"},
            ctx,
        )
        assert avail["available"] is True
        slot_id = avail["slots"][0]["id"]

        result = await registry.execute(
            "book_consultation",
            {"slot_id": slot_id, "caller_name": "John Smith"},
            ctx,
        )
        assert result["status"] == "booked"
        assert ctx.state.booking_confirmed is True
        assert ctx.state.phase == CallPhase.CONFIRMATION


class TestGeneralInfoScenario:
    """Caller wants information, not a booking."""

    @pytest.mark.asyncio
    async def test_general_info_flow(self, registry, ctx):
        ctx.add_user_message("I just have a question about employment law.")

        await registry.execute(
            "classify_caller_intent",
            {"intent": "general_info"},
            ctx,
        )
        assert ctx.state.caller_intent == CallerIntent.GENERAL_INFO
        assert ctx.state.phase == CallPhase.ROUTING

        await registry.execute(
            "classify_legal_area",
            {"legal_area": "employment"},
            ctx,
        )
        assert ctx.state.phase == CallPhase.INFORMATION


class TestMisunderstandingEscalation:
    """3+ consecutive misunderstandings trigger automatic escalation."""

    def test_misunderstanding_streak_escalates(self, ctx):
        ctx.add_user_message("Hello")
        ctx.state.misunderstanding_streak = 3
        ctx.advance_phase()
        assert ctx.state.phase == CallPhase.ESCALATION


class TestEmailValidation:
    """Invalid email format is rejected, not stored."""

    @pytest.mark.asyncio
    async def test_invalid_email_rejected(self, registry, ctx):
        ctx.add_user_message("my email is not-an-email")
        ctx.set_intent(CallerIntent.BOOK_CONSULTATION)
        ctx.set_legal_area(LegalArea.EMPLOYMENT)

        result = await registry.execute(
            "extract_caller_details",
            {"email": "not-an-email"},
            ctx,
        )
        assert "email" not in result["stored"]
        assert len(result.get("format_errors", [])) == 1
