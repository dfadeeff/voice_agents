"""Scenario-level tests proving the agent handles required user stories.

These test the full state machine + tool pipeline for realistic call flows,
not individual tools in isolation (that's test_tools.py).
"""

import pytest
from app.conversation.manager import ConversationManager
from app.models.schemas import CallerIntent, CallPhase, LegalArea


@pytest.fixture
def ctx():
    return ConversationManager(call_id="scenario-test", lang="en")


class TestEmploymentRoutingScenario:
    """Caller with an employment issue routes correctly through qualification."""

    @pytest.mark.asyncio
    async def test_employment_booking_flow(self, registry, ctx):
        ctx.add_user_message("I was fired last week and I think it was unfair.")
        assert ctx.state.phase == CallPhase.QUALIFICATION
        assert ctx.state.legal_area == LegalArea.EMPLOYMENT

        await registry.execute(
            "capture_caller_details",
            {"matter_type": "dismissal"},
            ctx,
        )
        assert ctx.state.phase == CallPhase.QUALIFICATION

        ctx.add_user_message("Yes, the deadline is in two weeks")
        await registry.execute(
            "capture_caller_details",
            {"matter_details": "deadline in 2 weeks"},
            ctx,
        )
        assert ctx.state.phase == CallPhase.CAPTURE


class TestTenancyRoutingScenario:
    """Caller with a tenancy issue routes correctly."""

    @pytest.mark.asyncio
    async def test_tenancy_routing(self, registry, ctx):
        ctx.add_user_message("My landlord is refusing to return my deposit.")

        await registry.execute(
            "route_call",
            {
                "intent": "book_consultation",
                "legal_area": "tenancy",
                "matter_summary": "deposit dispute",
            },
            ctx,
        )
        assert ctx.state.legal_area == LegalArea.TENANCY
        assert ctx.state.phase == CallPhase.QUALIFICATION


class TestUnknownAreaEscalationScenario:
    """Caller with an unsupported legal area escalates."""

    @pytest.mark.asyncio
    async def test_family_law_escalates(self, registry, ctx):
        ctx.add_user_message("I need help with child custody after my divorce.")

        result = await registry.execute(
            "route_call",
            {"intent": "book_consultation", "legal_area": "unknown"},
            ctx,
        )
        assert result["status"] == "unknown_area"
        assert ctx.state.escalation_requested is True
        assert ctx.state.phase == CallPhase.ESCALATION

    @pytest.mark.asyncio
    async def test_invalid_area_string_escalates(self, registry, ctx):
        ctx.add_user_message("I need criminal defense.")

        result = await registry.execute(
            "route_call",
            {"intent": "book_consultation", "legal_area": "criminal"},
            ctx,
        )
        assert result["status"] == "unknown_area"
        assert ctx.state.phase == CallPhase.ESCALATION


class TestHumanHandoffScenario:
    """Caller explicitly asks for a human at any point."""

    def test_named_person_request_enters_callback_capture(self):
        ctx = ConversationManager(call_id="named-person", lang="de")

        ctx.add_user_message("Ich möchte bitte Frau Landau sprechen.")

        assert ctx.state.phase == CallPhase.CAPTURE
        assert ctx.state.callback_requested is True
        assert ctx.state.target_person == "Landau"
        assert ctx.state.escalation_requested is False

    @pytest.mark.asyncio
    async def test_immediate_handoff(self, registry, ctx):
        ctx.add_user_message("I want to speak to a person.")

        result = await registry.execute(
            "request_handoff",
            {"reason": "caller_requested_human", "summary": "Wants to speak to a lawyer"},
            ctx,
        )
        assert result["status"] == "handoff_requested"
        assert result["mode"] == "callback"
        assert ctx.state.phase == CallPhase.ESCALATION

    @pytest.mark.asyncio
    async def test_handoff_mid_flow(self, registry, ctx):
        ctx.add_user_message("I was dismissed")
        await registry.execute(
            "route_call",
            {"intent": "book_consultation", "legal_area": "employment"},
            ctx,
        )
        assert ctx.state.phase == CallPhase.QUALIFICATION

        ctx.add_user_message("Actually, can I just talk to someone?")
        result = await registry.execute(
            "request_handoff",
            {"reason": "caller_requested_human"},
            ctx,
        )
        assert ctx.state.phase == CallPhase.ESCALATION
        assert result["context_for_human"]["caller_details"] == {}


class TestEmailConfirmationScenario:
    """Email always requires explicit confirmation via confirm_caller_detail."""

    @pytest.mark.asyncio
    async def test_email_capture_then_confirm(self, registry, ctx):
        ctx.add_user_message("I was dismissed")
        ctx.set_route(CallerIntent.BOOK_CONSULTATION, LegalArea.EMPLOYMENT)
        await registry.execute("capture_caller_details", {"matter_type": "dismissal"}, ctx)

        result = await registry.execute(
            "capture_caller_details",
            {"email": "fadejeff@gmail.com"},
            ctx,
        )
        assert not result["all_confirmed"]
        assert any(c["field"] == "email" for c in result["needs_confirmation"])
        assert not ctx.state.entities["email"].confirmed

        confirm_result = await registry.execute(
            "confirm_caller_detail",
            {"field": "email", "confirmed_value": "fadejeff@gmail.com", "status": "accepted"},
            ctx,
        )
        assert confirm_result["status"] == "confirmed"
        assert ctx.state.entities["email"].confirmed is True

    @pytest.mark.asyncio
    async def test_email_correction_flow(self, registry, ctx):
        """Caller corrects a misheard email."""
        ctx.add_user_message("details")
        ctx.set_route(CallerIntent.BOOK_CONSULTATION, LegalArea.EMPLOYMENT)
        await registry.execute("capture_caller_details", {"matter_type": "dismissal"}, ctx)

        await registry.execute(
            "capture_caller_details",
            {"email": "fade_jeff@gmail.com"},
            ctx,
        )
        assert not ctx.state.entities["email"].confirmed

        result = await registry.execute(
            "confirm_caller_detail",
            {"field": "email", "confirmed_value": "fadejeff@gmail.com", "status": "corrected"},
            ctx,
        )
        assert result["status"] == "confirmed"
        assert result["was_corrected"] is True
        assert ctx.state.entities["email"].value == "fadejeff@gmail.com"
        assert ctx.state.entities["email"].confirmed is True


class TestUnavailableSlotScenario:
    """Caller wants a time that's not available — agent offers alternatives."""

    @pytest.mark.asyncio
    async def test_unavailable_slot_offers_alternatives(self, registry, ctx):
        ctx.add_user_message("details")
        ctx.set_route(CallerIntent.BOOK_CONSULTATION, LegalArea.EMPLOYMENT)
        ctx.store_entity("matter_type", "dismissal", 1.0)
        ctx.confirm_entity("matter_type")
        ctx.store_entity("matter_details", "deadline soon", 1.0)
        ctx.confirm_entity("matter_details")
        for field in ("name", "email", "phone"):
            ctx.store_entity(field, f"test_{field}", 0.9)
            ctx.confirm_entity(field)
        ctx.advance_phase()

        assert ctx.state.phase == CallPhase.BOOKING

        result = await registry.execute(
            "check_availability",
            {"date": "2026-06-15"},
            ctx,
        )
        assert result["available"] is False
        assert "alternatives" in result


class TestFullBookingScenario:
    """Complete happy path: greeting -> routing -> qualification
    -> capture -> booking -> confirmation."""

    @pytest.mark.asyncio
    async def test_full_happy_path(self, registry, ctx):
        assert ctx.state.phase == CallPhase.GREETING

        ctx.add_user_message("Hi, I need help with an employment issue.")
        assert ctx.state.phase == CallPhase.ROUTING

        await registry.execute(
            "route_call",
            {
                "intent": "book_consultation",
                "legal_area": "employment",
                "matter_summary": "unfair dismissal",
            },
            ctx,
        )
        assert ctx.state.phase == CallPhase.QUALIFICATION

        ctx.add_user_message("It's about a dismissal")
        await registry.execute(
            "capture_caller_details",
            {"matter_type": "dismissal"},
            ctx,
        )
        assert ctx.state.phase == CallPhase.QUALIFICATION

        ctx.add_user_message("Yes, the deadline is in about two weeks")
        await registry.execute(
            "capture_caller_details",
            {"matter_details": "deadline approaching in 2 weeks"},
            ctx,
        )
        assert ctx.state.phase == CallPhase.CAPTURE

        ctx.add_user_message("My name is John Smith")
        await registry.execute("capture_caller_details", {"name": "John Smith"}, ctx)

        ctx.add_user_message("john at example dot com")
        await registry.execute("capture_caller_details", {"email": "john@example.com"}, ctx)

        ctx.add_user_message("yes that's correct")
        await registry.execute(
            "confirm_caller_detail",
            {"field": "email", "confirmed_value": "john@example.com", "status": "accepted"},
            ctx,
        )

        ctx.add_user_message("plus one two three four five six seven eight nine zero")
        await registry.execute("capture_caller_details", {"phone": "+1234567890"}, ctx)

        ctx.add_user_message("yes")
        await registry.execute(
            "confirm_caller_detail",
            {"field": "phone", "confirmed_value": "+1234567890", "status": "accepted"},
            ctx,
        )

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
            "route_call",
            {"intent": "general_info", "legal_area": "employment"},
            ctx,
        )
        assert ctx.state.caller_intent == CallerIntent.GENERAL_INFO
        assert ctx.state.phase == CallPhase.INFORMATION


class TestMisunderstandingEscalation:
    """3+ consecutive misunderstandings trigger automatic escalation."""

    def test_misunderstanding_streak_escalates(self, ctx):
        ctx.add_user_message("Hello")
        ctx.state.misunderstanding_streak = 3
        ctx.advance_phase()
        assert ctx.state.phase == CallPhase.ESCALATION

    @pytest.mark.asyncio
    async def test_format_errors_increment_streak(self, registry, ctx):
        ctx.add_user_message("details")
        ctx.set_route(CallerIntent.BOOK_CONSULTATION, LegalArea.EMPLOYMENT)
        await registry.execute("capture_caller_details", {"matter_type": "dismissal"}, ctx)

        for i in range(3):
            ctx.add_user_message(f"bad email {i}")
            await registry.execute(
                "capture_caller_details",
                {"email": f"not-an-email-{i}"},
                ctx,
            )

        assert ctx.state.misunderstanding_streak >= 3
        assert ctx.state.phase == CallPhase.ESCALATION

    def test_successful_route_resets_streak(self, ctx):
        ctx.add_user_message("Hello")
        ctx.record_misunderstanding()
        ctx.record_misunderstanding()
        assert ctx.state.misunderstanding_streak == 2
        ctx.set_route(CallerIntent.BOOK_CONSULTATION, LegalArea.EMPLOYMENT)
        assert ctx.state.misunderstanding_streak == 0


class TestEmailValidation:
    """Invalid email format is rejected, not stored."""

    @pytest.mark.asyncio
    async def test_invalid_email_rejected(self, registry, ctx):
        ctx.add_user_message("my email is not-an-email")
        ctx.set_route(CallerIntent.BOOK_CONSULTATION, LegalArea.EMPLOYMENT)
        await registry.execute("capture_caller_details", {"matter_type": "dismissal"}, ctx)

        result = await registry.execute(
            "capture_caller_details",
            {"email": "not-an-email"},
            ctx,
        )
        assert "email" not in result["stored"]
        assert len(result.get("format_errors", [])) == 1


class TestOfferedSlotSafety:
    """book_consultation rejects slots that weren't offered by check_availability."""

    @pytest.mark.asyncio
    async def test_unoffered_slot_rejected(self, registry, ctx):
        ctx.add_user_message("details")
        ctx.set_route(CallerIntent.BOOK_CONSULTATION, LegalArea.EMPLOYMENT)
        ctx.store_entity("matter_type", "dismissal", 1.0)
        ctx.confirm_entity("matter_type")
        ctx.store_entity("matter_details", "deadline soon", 1.0)
        ctx.confirm_entity("matter_details")
        for field in ("name", "email", "phone"):
            ctx.store_entity(field, f"test_{field}", 0.9)
            ctx.confirm_entity(field)
        ctx.advance_phase()

        await registry.execute(
            "check_availability",
            {"date": "2026-06-10", "legal_area": "employment"},
            ctx,
        )
        assert len(ctx.state.offered_slot_ids) > 0

        result = await registry.execute(
            "book_consultation",
            {"slot_id": 999, "caller_name": "test_name"},
            ctx,
        )
        assert result["status"] == "error"
        assert "not offered" in result["message"]


class TestAutoRouting:
    """Keyword-based auto-routing when LLM skips route_call."""

    def test_employment_auto_route_de(self):
        ctx = ConversationManager(call_id="auto-de", lang="de")
        ctx.add_user_message("Ich wurde letzte Woche gekündigt.")
        assert ctx.state.legal_area == LegalArea.EMPLOYMENT
        assert ctx.state.caller_intent == CallerIntent.BOOK_CONSULTATION
        assert ctx.state.phase == CallPhase.QUALIFICATION

    def test_tenancy_auto_route_de(self):
        ctx = ConversationManager(call_id="auto-de", lang="de")
        ctx.add_user_message("Mein Vermieter will die Kaution nicht zurückgeben.")
        assert ctx.state.legal_area == LegalArea.TENANCY
        assert ctx.state.phase == CallPhase.QUALIFICATION

    def test_traffic_auto_route_de(self):
        ctx = ConversationManager(call_id="auto-de", lang="de")
        ctx.add_user_message("Ich hatte einen Unfall auf der Autobahn.")
        assert ctx.state.legal_area == LegalArea.TRAFFIC
        assert ctx.state.phase == CallPhase.QUALIFICATION

    def test_employment_auto_route_en(self):
        ctx = ConversationManager(call_id="auto-en", lang="en")
        ctx.add_user_message("I was fired last week.")
        assert ctx.state.legal_area == LegalArea.EMPLOYMENT
        assert ctx.state.phase == CallPhase.QUALIFICATION

    def test_tenancy_auto_route_en(self):
        ctx = ConversationManager(call_id="auto-en", lang="en")
        ctx.add_user_message("My landlord won't return my deposit.")
        assert ctx.state.legal_area == LegalArea.TENANCY
        assert ctx.state.phase == CallPhase.QUALIFICATION

    def test_traffic_auto_route_en(self):
        ctx = ConversationManager(call_id="auto-en", lang="en")
        ctx.add_user_message("I had a car accident yesterday.")
        assert ctx.state.legal_area == LegalArea.TRAFFIC
        assert ctx.state.phase == CallPhase.QUALIFICATION

    def test_ambiguous_stays_in_routing(self):
        ctx = ConversationManager(call_id="ambiguous", lang="de")
        ctx.add_user_message("Ich habe ein Problem.")
        assert ctx.state.legal_area == LegalArea.UNKNOWN
        assert ctx.state.phase == CallPhase.ROUTING

    def test_multi_area_stays_in_routing(self):
        ctx = ConversationManager(call_id="multi", lang="de")
        ctx.add_user_message("Mein Arbeitgeber hat einen Unfall verursacht.")
        assert ctx.state.legal_area == LegalArea.UNKNOWN
        assert ctx.state.phase == CallPhase.ROUTING
