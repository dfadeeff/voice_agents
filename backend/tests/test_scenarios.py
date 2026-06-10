"""Scenario-level tests proving the agent handles required user stories.

These test the full state machine + tool pipeline for realistic call flows,
not individual tools in isolation (that's test_tools.py).
"""

import pytest

from app.conversation.email_capture import EmailCapture
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


class TestDeterministicContactCapture:
    """LLM-skip fallback: name/phone are stored in code so data still persists.

    Regression for a live call where the model narrated capturing contact details
    but never called capture_caller_details, so nothing reached the database.
    """

    def test_name_captured_without_tool_call_in_callback(self):
        ctx = ConversationManager(call_id="det-name", lang="de")
        ctx.add_user_message("Bitte rufen Sie mich zurück.")
        assert ctx.state.phase == CallPhase.CAPTURE

        ctx.add_user_message("Mein Name ist Herbert Wilhelm")

        assert "name" in ctx.state.entities
        assert ctx.state.entities["name"].value == "Herbert Wilhelm"

    def test_phone_captured_without_tool_call(self):
        ctx = ConversationManager(call_id="det-phone", lang="de")
        ctx.add_user_message("Bitte rufen Sie mich zurück.")

        ctx.add_user_message("0151 693 67234")

        assert "phone" in ctx.state.entities
        assert ctx.state.entities["phone"].value == "+4915169367234"

    def test_partial_phone_fragment_not_stored(self):
        ctx = ConversationManager(call_id="det-partial", lang="de")
        ctx.add_user_message("Bitte rufen Sie mich zurück.")

        ctx.add_user_message("0151")

        assert "phone" not in ctx.state.entities

    def test_no_contact_capture_during_qualification(self):
        """Name must NOT be captured while still qualifying the matter."""
        ctx = ConversationManager(call_id="det-qual", lang="de")
        ctx.add_user_message("Ich hatte einen Unfall")
        assert ctx.state.phase == CallPhase.QUALIFICATION

        ctx.add_user_message("Mein Name ist Herbert Wilhelm")

        assert "name" not in ctx.state.entities

    def test_english_name_capture(self):
        ctx = ConversationManager(call_id="det-en", lang="en")
        ctx.add_user_message("Could you call me back, please?")
        assert ctx.state.phase == CallPhase.CAPTURE

        ctx.add_user_message("My name is Herbert Wilhelm")

        assert ctx.state.entities["name"].value == "Herbert Wilhelm"

    def test_bare_name_captured_after_agent_asks(self):
        """Regression: caller answers 'Daniel Stein' with no 'Mein Name ist' trigger."""
        ctx = ConversationManager(call_id="det-bare", lang="de")
        ctx.add_user_message("Bitte rufen Sie mich zurück.")
        ctx.next_prompt()  # scripted name ask → awaiting == "name"
        ctx.add_user_message("Daniel Stein")
        assert ctx.state.entities["name"].value == "Daniel Stein"

    def test_bare_reply_non_name_not_captured(self):
        ctx = ConversationManager(call_id="det-bare2", lang="de")
        ctx.add_user_message("Bitte rufen Sie mich zurück.")
        ctx.next_prompt()  # awaiting == "name"
        ctx.add_user_message("Ja gerne")
        assert "name" not in ctx.state.entities


class TestDeterministicMatterType:
    """matter_type backfill when the caller front-loads details or the LLM skips."""

    def test_routing_turn_does_not_capture_matter_type(self):
        """Regression: the area-confirmation question must still be asked."""
        ctx = ConversationManager(call_id="mt-route", lang="de")
        ctx.add_user_message("Ich hatte einen Unfall")
        assert ctx.state.phase == CallPhase.QUALIFICATION
        assert "matter_type" not in ctx.state.entities

    def test_matter_type_backfilled_on_confirm_turn(self):
        ctx = ConversationManager(call_id="mt-confirm", lang="de")
        ctx.add_user_message("Ich hatte einen Unfall")
        ctx.add_user_message("Ja, ein Verkehrsunfall")
        assert ctx.state.entities["matter_type"].value == "accident"

    def test_matter_type_backfilled_when_frontloaded_with_handoff(self):
        """The live-call regression: matter + person request in one breath."""
        ctx = ConversationManager(call_id="mt-frontload", lang="de")
        ctx.add_user_message("Ich hatte einen Unfall")
        ctx.add_user_message(
            "Es war ein Verkehrsunfall, ich habe keine Versicherungsnummer "
            "und ich möchte mit Herrn Schulz sprechen."
        )
        assert ctx.state.entities["matter_type"].value == "accident"
        # A named-person request now routes to a booking (with that lawyer), not a callback.
        assert ctx.state.target_person == "Herr Schulz"
        assert ctx.state.callback_requested is False

    def test_employment_matter_type(self):
        ctx = ConversationManager(call_id="mt-emp", lang="de")
        ctx.add_user_message("Mein Arbeitgeber hat mir gekündigt")
        ctx.add_user_message("Ja, eine Kündigung")
        assert ctx.state.entities["matter_type"].value == "dismissal"

    def test_accident_wins_over_insurance_keyword(self):
        """'Unfall' + 'Versicherung' in one sentence resolves to accident."""
        ctx = ConversationManager(call_id="mt-prio", lang="de")
        ctx.add_user_message("Ich hatte einen Unfall")
        ctx.add_user_message("Ein Verkehrsunfall, die Versicherung macht Probleme")
        assert ctx.state.entities["matter_type"].value == "accident"

    def test_versicherungsnummer_mention_does_not_become_matter_type(self):
        """Regression: mentioning 'Versicherungsnummer' must not set matter_type=insurance
        when the original complaint was an accident."""
        ctx = ConversationManager(call_id="mt-vsn", lang="de")
        ctx.add_user_message("Ich hatte einen Unfall")
        ctx.add_user_message("Ja, ich habe eine Versicherungsnummer")
        assert ctx.state.entities["matter_type"].value == "accident"

    def test_bare_yes_confirmation_captures_from_complaint(self):
        """Regression: 'Ja, das ist korrekt' must capture the type so the agent
        does not re-ask the same area-confirmation question."""
        ctx = ConversationManager(call_id="mt-yes", lang="de")
        ctx.add_user_message("Ich hatte ein Unfall")
        assert "matter_type" not in ctx.state.entities  # confirmation still asked
        ctx.add_user_message("Ja, das ist korrekt.")
        assert ctx.state.entities["matter_type"].value == "accident"

    def test_negative_confirmation_captures_nothing(self):
        ctx = ConversationManager(call_id="mt-no", lang="de")
        ctx.add_user_message("Ich hatte ein Unfall")
        ctx.add_user_message("Nein, das stimmt nicht")
        assert "matter_type" not in ctx.state.entities

    def test_tenancy_kuendigung_maps_to_eviction(self):
        """Regression: the tenancy question offers 'Kündigung der Wohnung', so that
        answer must map to a matter type (eviction) — not loop forever."""
        ctx = ConversationManager(call_id="mt-evict", lang="de")
        ctx.add_user_message("Ich habe ein Problem mit meiner Wohnung")  # → tenancy
        ctx.next_prompt()  # area_confirm, awaiting matter_type
        ctx.add_user_message("Kündigung der Wohnung")
        assert ctx.state.entities["matter_type"].value == "eviction"

    def test_unrecognized_matter_falls_back_to_other(self):
        """Loop guard: an unrecognised matter answer is recorded as 'other' after a
        couple of tries instead of re-asking the same question forever."""
        ctx = ConversationManager(call_id="mt-other", lang="de")
        ctx.add_user_message("Ich habe ein Problem mit meiner Wohnung")  # → tenancy
        ctx.next_prompt()
        ctx.add_user_message("Das ist schwer zu erklären")  # unmatched (try 1)
        assert "matter_type" not in ctx.state.entities
        ctx.next_prompt()
        ctx.add_user_message("Es ist kompliziert")  # unmatched (try 2 → other)
        assert ctx.state.entities["matter_type"].value == "other"

    def test_llm_classifier_rescues_free_phrased_matter(self):
        """When keywords miss, the LLM rescue maps a free phrasing to a label."""
        import asyncio

        async def fake_clf(_area, _text):
            return "eviction"

        ctx = ConversationManager(call_id="mt-llm", lang="de")
        ctx.set_matter_classifier(fake_clf)
        ctx.add_user_message("Ich habe ein Problem mit meiner Wohnung")  # → tenancy
        ctx.next_prompt()  # awaiting matter_type
        garbled = "Ich wurde aus meiner Wohnung geworfen"  # no keyword hit
        ctx.add_user_message(garbled)
        assert "matter_type" not in ctx.state.entities  # keywords missed
        asyncio.run(ctx.resolve_matter_if_pending(garbled))
        assert ctx.state.entities["matter_type"].value == "eviction"


class TestContextAwareInsuranceCapture:
    """insurance_number is captured only when the agent just asked for it."""

    def _traffic_ctx(self):
        """Traffic call advanced to the scripted insurance step (awaiting=='insurance')."""
        ctx = ConversationManager(call_id="ins", lang="de")
        ctx.add_user_message("Ich hatte einen Unfall")
        ctx.next_prompt()  # area_confirm, awaiting matter_type
        ctx.add_user_message("Ja, ein Verkehrsunfall")
        ctx.next_prompt()  # traffic_insurance, awaiting insurance
        return ctx

    def test_captured_after_agent_asks(self):
        ctx = self._traffic_ctx()
        ctx.add_user_message("Ja, meine Versicherungsnummer ist VS 4455 6677")
        assert ctx.state.entities["insurance_number"].value == "VS44556677"

    def test_digits_not_misread_as_phone(self):
        ctx = self._traffic_ctx()
        ctx.add_user_message("Ja, 9988776655")
        assert ctx.state.entities["insurance_number"].value == "9988776655"
        assert "phone" not in ctx.state.entities

    def test_negative_reply_stores_nothing(self):
        ctx = self._traffic_ctx()
        ctx.add_user_message("Nee leider nicht")
        assert "insurance_number" not in ctx.state.entities

    def test_not_captured_when_not_awaiting_insurance(self):
        """A bare number must NOT become insurance_number unless we asked for it."""
        ctx = self._traffic_ctx()
        ctx.state.awaiting = None  # we are not on the insurance question
        ctx.add_user_message("Vorgang 123456")
        assert "insurance_number" not in ctx.state.entities

    def test_phone_still_works_when_phone_asked(self):
        ctx = ConversationManager(call_id="ins-phone", lang="de")
        ctx.add_user_message("Bitte rufen Sie mich zurück.")
        ctx.next_prompt()  # ask name, awaiting name
        ctx.add_user_message("Max Mustermann")
        ctx.next_prompt()  # ask phone, awaiting phone
        ctx.add_user_message("0151 598 32614")
        assert ctx.state.entities["phone"].value == "+4915159832614"
        assert "insurance_number" not in ctx.state.entities


class TestEnforcedInsuranceStep:
    """Traffic qualification cannot skip the insurance question (state-gated)."""

    def test_phase_stays_in_qualification_until_insurance_resolved(self):
        ctx = ConversationManager(call_id="enf1", lang="de")
        ctx.add_user_message("Ich hatte einen Unfall")
        ctx.add_user_message("Ja, das ist korrekt.")
        assert ctx.state.entities["matter_type"].value == "accident"
        # Gated: matter_type known but insurance not yet asked → still qualifying.
        assert ctx.state.phase == CallPhase.QUALIFICATION
        assert ctx.state.insurance_resolved is False

    def test_full_traffic_booking_path(self):
        ctx = ConversationManager(call_id="enf2", lang="de")
        ctx.add_user_message("Ich hatte einen Unfall")
        ctx.next_prompt()  # area_confirm, awaiting matter_type
        ctx.add_user_message("Ja, das ist korrekt.")
        ctx.next_prompt()  # traffic_insurance, awaiting insurance
        ctx.add_user_message("Ja, VS 4455 6677")
        assert ctx.state.entities["insurance_number"].value == "VS44556677"
        ctx.next_prompt()  # confirm_insurance read-back, awaiting insurance_confirm
        ctx.add_user_message("Ja, korrekt")  # confirm the number
        assert ctx.state.phase == CallPhase.CAPTURE
        for f, v in [("name", "Daniel Steinmeier"), ("email", "d@example.com")]:
            ctx.store_entity(f, v, 0.95)
            ctx.confirm_entity(f)
        ctx.store_entity("phone", "+4915112345678", 0.95)
        ctx.confirm_entity("phone")
        assert ctx.state.phase == CallPhase.BOOKING

    def test_spoken_digit_words_insurance_captured(self):
        # Regression: STT renders a spoken reference as digit *words*
        # ("F fünf vier zwei sechs ..."), which the extractor missed entirely.
        ctx = ConversationManager(call_id="enf-spoken", lang="de")
        ctx.add_user_message("Ich hatte einen Unfall")
        ctx.next_prompt()  # area_confirm, awaiting matter_type
        ctx.add_user_message("Ja, ein Verkehrsunfall")
        ctx.next_prompt()  # traffic_insurance, awaiting insurance
        ctx.add_user_message("F fünf vier zwei sechs acht neun drei vier sieben")
        assert ctx.state.entities["insurance_number"].value == "F542689347"
        ctx.next_prompt()  # read-back, awaiting insurance_confirm
        ctx.add_user_message("Ja")  # confirm
        assert ctx.state.phase == CallPhase.CAPTURE

    def test_caller_without_insurance_number_still_advances(self):
        ctx = ConversationManager(call_id="enf3", lang="de")
        ctx.add_user_message("Ich hatte einen Unfall")
        ctx.next_prompt()  # area_confirm, awaiting matter_type
        ctx.add_user_message("Ja, genau")
        ctx.next_prompt()  # traffic_insurance, awaiting insurance
        ctx.add_user_message("Nein, leider noch keine")
        assert ctx.state.insurance_resolved is True
        assert "insurance_number" not in ctx.state.entities
        assert ctx.state.phase == CallPhase.CAPTURE

    def test_unparseable_insurance_answers_do_not_loop_forever(self):
        # Caller never reads any digits ("Versicherungsnummer", "F", …) → the step
        # must cap the loop and proceed rather than re-ask forever.
        ctx = ConversationManager(call_id="enf4", lang="de")
        ctx.add_user_message("Ich hatte einen Autounfall")
        ctx.next_prompt()  # area_confirm, awaiting matter_type
        ctx.add_user_message("Verkehrsunfall")
        ctx.next_prompt()  # traffic_insurance, awaiting insurance
        for _ in range(4):  # _MAX_INSURANCE_ATTEMPTS — no digits ever given
            ctx.add_user_message("Versicherungsnummer")
        assert ctx.state.insurance_resolved is True
        assert "insurance_number" not in ctx.state.entities
        assert ctx.state.phase == CallPhase.CAPTURE

    def test_insurance_dictated_piecewise_across_turns_is_accumulated(self):
        # The live regression: the number is spoken in short bursts with pauses,
        # each chunk too short on its own — they must accumulate into one reference.
        ctx = ConversationManager(call_id="enf5", lang="de")
        ctx.add_user_message("Ich hatte einen Autounfall")
        ctx.next_prompt()  # area_confirm, awaiting matter_type
        ctx.add_user_message("Verkehrsunfall")
        ctx.next_prompt()  # traffic_insurance, awaiting insurance
        ctx.add_user_message("Ich habe eine Versicherungsnummer F vier")  # "F4"
        assert ctx.state.insurance_resolved is False  # still collecting
        ctx.add_user_message("fünf vier")  # "54" → buffer "F454"
        assert ctx.state.insurance_resolved is False
        ctx.add_user_message("sechs acht neun")  # "689" → buffer "F454689" (≥5)
        assert ctx.state.entities["insurance_number"].value == "F454689"
        # Now read back for confirmation — caller can still add more or confirm.
        assert ctx.state.entities["insurance_number"].confirmed is False
        ctx.next_prompt()  # read-back, awaiting insurance_confirm
        ctx.add_user_message("Ja, korrekt")
        assert ctx.state.entities["insurance_number"].confirmed is True
        assert ctx.state.phase == CallPhase.CAPTURE

    def test_insurance_continued_during_readback(self):
        # The exact live bug: caller pauses after "…acht neun", the number is read
        # back, and the caller continues "drei vier sieben" — it must be appended,
        # not treated as a new field, and re-read back before confirming.
        ctx = ConversationManager(call_id="enf6", lang="de")
        ctx.add_user_message("Ich hatte einen Autounfall")
        ctx.next_prompt()
        ctx.add_user_message("Verkehrsunfall")
        ctx.next_prompt()
        ctx.add_user_message("Versicherungsnummer f fünf vier zwei sechs acht neun")  # F542689
        line = ctx.next_prompt()  # read-back, awaiting insurance_confirm
        assert ctx.state.awaiting == "insurance_confirm"
        assert "F, 5, 4, 2, 6, 8, 9" in line
        ctx.add_user_message("drei vier sieben")  # continuation, not a new turn
        ctx.next_prompt()  # re-read-back
        assert ctx.state.entities["insurance_number"].value == "F542689347"
        ctx.add_user_message("Ja, korrekt")
        assert ctx.state.entities["insurance_number"].confirmed is True
        assert ctx.state.phase == CallPhase.CAPTURE


class TestHumanHandoffScenario:
    """Caller explicitly asks for a human at any point."""

    def test_named_person_request_routes_to_booking(self):
        ctx = ConversationManager(call_id="named-person", lang="de")

        ctx.add_user_message("Ich möchte bitte Frau Landau sprechen.")

        # Books a consultation with that lawyer (not a callback / escalation).
        assert ctx.state.target_person == "Frau Landau"
        assert ctx.state.caller_intent == CallerIntent.BOOK_CONSULTATION
        assert ctx.state.callback_requested is False
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
        assert ctx.state.callback_requested is True
        assert ctx.state.phase == CallPhase.CAPTURE

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
        assert ctx.state.callback_requested is True
        assert ctx.state.phase == CallPhase.CAPTURE
        assert result["context_for_human"]["caller_details"] == {}


class TestEmailConfirmationScenario:
    """Email always requires explicit confirmation via confirm_caller_detail."""

    @pytest.mark.asyncio
    async def test_email_capture_then_confirm(self, registry, ctx):
        ctx.add_user_message("I was dismissed")
        ctx.set_route(CallerIntent.BOOK_CONSULTATION, LegalArea.EMPLOYMENT)
        await registry.execute("capture_caller_details", {"matter_type": "dismissal"}, ctx)
        await registry.execute(
            "capture_caller_details", {"matter_details": "deadline in 2 weeks"}, ctx
        )
        assert ctx.state.phase == CallPhase.CAPTURE

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
            "capture_caller_details", {"matter_details": "deadline in 2 weeks"}, ctx
        )

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
        await registry.execute(
            "capture_caller_details", {"matter_details": "deadline in 2 weeks"}, ctx
        )

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
        await registry.execute(
            "capture_caller_details", {"matter_details": "deadline in 2 weeks"}, ctx
        )

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


class TestTrafficInsuranceCapture:
    """Traffic-specific capture: insurance/claim number stored."""

    @pytest.mark.asyncio
    async def test_insurance_number_stored(self, registry, ctx):
        ctx.set_route(CallerIntent.BOOK_CONSULTATION, LegalArea.TRAFFIC)
        await registry.execute("capture_caller_details", {"matter_type": "accident"}, ctx)
        assert "matter_type" in ctx.state.entities

        result = await registry.execute(
            "capture_caller_details",
            {"insurance_number": "VS-2026-12345"},
            ctx,
        )
        assert "insurance_number" in result["stored"]
        assert ctx.state.entities["insurance_number"].value == "VS-2026-12345"


class TestCaseReferenceCapture:
    """Existing client provides Aktenzeichen (case reference)."""

    @pytest.mark.asyncio
    async def test_case_reference_stored(self, registry, ctx):
        ctx.set_route(CallerIntent.BOOK_CONSULTATION, LegalArea.EMPLOYMENT)
        await registry.execute("capture_caller_details", {"matter_type": "dismissal"}, ctx)
        await registry.execute(
            "capture_caller_details", {"matter_details": "deadline in 2 weeks"}, ctx
        )
        assert ctx.state.phase == CallPhase.CAPTURE

        result = await registry.execute(
            "capture_caller_details",
            {"case_reference": "44/24"},
            ctx,
        )
        assert "case_reference" in result["stored"]
        assert ctx.state.entities["case_reference"].value == "44/24"
        assert ctx.state.entities["case_reference"].confirmed is True


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

    def test_multi_area_queues_disambiguation(self):
        """≥2 matched areas → record candidates + ask a scripted disambiguation,
        instead of stalling in ROUTING where the LLM could hallucinate a booking."""
        ctx = ConversationManager(call_id="multi", lang="de")
        ctx.add_user_message("Mein Arbeitgeber hat einen Unfall verursacht.")
        assert ctx.state.legal_area == LegalArea.UNKNOWN
        assert ctx.state.phase == CallPhase.ROUTING
        assert set(ctx.state.area_options) == {"employment", "traffic"}
        line = ctx.next_prompt()  # scripted disambiguation, LLM skipped
        assert line is not None and ctx.state.awaiting == "area"

    def test_ambiguous_routing_disambiguates_then_proceeds(self):
        """The live-call regression: 'Mietvertrag gekündigt' (tenancy + employment)
        must NOT stall in ROUTING; a follow-up question resolves the area in code."""
        ctx = ConversationManager(call_id="disambig", lang="de")
        ctx.add_user_message("Mein Mietvertrag wurde gekündigt und ich will Frau Hofmann sprechen.")
        assert ctx.state.legal_area == LegalArea.UNKNOWN
        assert set(ctx.state.area_options) == {"employment", "tenancy"}
        assert ctx.state.target_person == "Frau Hofmann"  # person request still recorded

        line = ctx.next_prompt()
        assert ctx.state.awaiting == "area"
        assert "Arbeitsrecht" in line and "Mietrecht" in line

        ctx.add_user_message("Es geht um Mietrecht.")
        assert ctx.state.legal_area == LegalArea.TENANCY
        assert ctx.state.area_options == []
        assert ctx.state.phase == CallPhase.QUALIFICATION  # moved on deterministically

    def test_future_tense_booking_claim_is_blocked(self):
        """The hallucinated 'wird … einen Termin … buchen' must be guarded when no
        booking is confirmed; the genuine 'ich vereinbare gerne …' offer is not."""
        from app.pipeline.processors import _guard_false_booking

        hallucination = "Frau Hofmann wird übermorgen um 10 Uhr einen Termin für Sie buchen."
        assert _guard_false_booking(hallucination, booking_confirmed=False) != hallucination

        offer = "Ich vereinbare gerne einen Beratungstermin mit ihr für Sie."
        assert _guard_false_booking(offer, booking_confirmed=False) == offer


class TestNarrationSplit:
    """Scripted spine: the whole callback capture is driven from templates."""

    def test_full_callback_flow(self):
        """name → phone → read-back → callback time → done, all scripted."""
        ctx = ConversationManager(call_id="ns", lang="de")
        ctx.add_user_message("Herr Schulz soll mich bitte zurückrufen.")
        line = ctx.next_prompt()
        assert line is not None and ctx.state.awaiting == "name"
        ctx.add_user_message("Daniel Stein")
        line = ctx.next_prompt()
        assert line is not None and ctx.state.awaiting == "phone"
        ctx.add_user_message("0151 598 32614")
        line = ctx.next_prompt()
        assert line is not None and "015159832614" in line  # accuracy-critical read-back
        assert ctx.state.awaiting == "phone_confirm"
        ctx.add_user_message("Ja, das stimmt")
        assert ctx.state.entities["phone"].confirmed is True
        assert ctx.state.entities["phone"].value == "+4915159832614"
        line = ctx.next_prompt()
        assert line is not None and ctx.state.awaiting == "callback_time"
        ctx.add_user_message("Morgen Vormittag")
        done = ctx.next_prompt()
        assert done is not None
        assert "Herr Schulz" in done
        assert "Morgen Vormittag" in done

    def test_denial_deletes_phone(self):
        """Phone denial deletes the entity; the scripted spine re-asks."""
        ctx = ConversationManager(call_id="ns2", lang="de")
        ctx.add_user_message("Bitte rufen Sie mich zurück.")
        ctx.next_prompt()  # ask name
        ctx.add_user_message("Daniel Stein")
        ctx.next_prompt()  # ask phone
        ctx.add_user_message("0151 598 32614")
        ctx.next_prompt()  # confirm_phone, awaiting phone_confirm
        ctx.add_user_message("Nein, das ist falsch")
        assert "phone" not in ctx.state.entities
        ctx.next_prompt()
        assert ctx.state.awaiting == "phone"  # re-asks

    def test_traffic_qualification_is_scripted(self):
        """Area confirmation / matter-type question is scripted, not LLM-driven."""
        ctx = ConversationManager(call_id="ns3", lang="de")
        ctx.add_user_message("Ich hatte einen Unfall")
        line = ctx.next_prompt()
        assert line is not None and ctx.state.awaiting == "matter_type"

    def test_ambiguous_routing_uses_llm(self):
        ctx = ConversationManager(call_id="ns4", lang="de")
        ctx.add_user_message("Ich habe da ein Problem.")
        assert ctx.next_prompt() is None  # stays ROUTING → LLM


class TestBookingCaptureSplit:
    """Scripted spine: name → email → phone with exact read-backs for email/phone."""

    def test_confirmations_are_scripted(self):
        ctx = ConversationManager(call_id="bc", lang="de")
        ctx.set_route(CallerIntent.BOOK_CONSULTATION, LegalArea.TRAFFIC)
        ctx.store_entity("matter_type", "accident", 1.0)
        ctx.confirm_entity("matter_type")
        ctx.state.insurance_resolved = True
        ctx.advance_phase()
        assert ctx.state.phase == CallPhase.CAPTURE
        line = ctx.next_prompt()
        assert line is not None and ctx.state.awaiting == "name"
        ctx.add_user_message("Daniel Stein")
        line = ctx.next_prompt()
        assert line is not None and ctx.state.awaiting == "email"
        ctx.add_user_message("max at gmail punkt com")
        line = ctx.next_prompt()
        assert line is not None and "max@gmail.com" in line  # accuracy-critical read-back
        assert ctx.state.awaiting == "email_confirm"
        ctx.add_user_message("Ja, stimmt")
        line = ctx.next_prompt()
        assert line is not None and ctx.state.awaiting == "phone"
        ctx.add_user_message("0151 598 32614")
        line = ctx.next_prompt()
        assert line is not None and "015159832614" in line  # accuracy-critical read-back
        assert ctx.state.awaiting == "phone_confirm"
        ctx.add_user_message("Ja, das stimmt")
        ctx.next_prompt()
        assert ctx.state.phase == CallPhase.BOOKING
        assert ctx.state.entities["email"].value == "max@gmail.com"
        assert ctx.state.entities["phone"].confirmed is True

    def test_email_denial_deletes_entity(self):
        ctx = ConversationManager(call_id="bc2", lang="de")
        ctx.set_route(CallerIntent.BOOK_CONSULTATION, LegalArea.EMPLOYMENT)
        ctx.store_entity("matter_type", "dismissal", 1.0)
        ctx.confirm_entity("matter_type")
        ctx.store_entity("matter_details", "Frist", 1.0)
        ctx.confirm_entity("matter_details")
        ctx.store_entity("name", "Anna Schmidt", 0.9)
        ctx.confirm_entity("name")
        assert ctx.state.phase == CallPhase.CAPTURE
        line = ctx.next_prompt()
        assert line is not None and ctx.state.awaiting == "email"
        ctx.add_user_message("anna at example punkt de")
        line = ctx.next_prompt()
        assert line is not None and ctx.state.awaiting == "email_confirm"
        ctx.add_user_message("Nein, das ist falsch")
        assert "email" not in ctx.state.entities
        ctx.next_prompt()
        assert ctx.state.awaiting == "email"  # re-asks


class TestEmailSpelling:
    """After repeated read-back rejections, offer to spell the local part."""

    def _parse(self):
        from app.conversation.email_parse import parse_spelled_local

        return parse_spelled_local

    def test_parse_phonetic_and_double(self):
        assert self._parse()("R wie Richard, I, doppel T, E, R") == "ritter"

    def test_parse_letter_names_stt_renders_as_words(self):
        # Deepgram renders spelled German letters as little words.
        assert self._parse()("er i te te e er") == "ritter"

    def test_parse_bare_letters(self):
        assert self._parse()("r i t t e r") == "ritter"

    def test_parse_double_english(self):
        assert self._parse()("l e o, double n") == "leonn"

    def test_merged_word_is_left_to_llm(self):
        # A single merged token isn't spelling we can trust — defer to the LLM.
        assert self._parse()("ritter") is None

    def _to_email_confirm(self, ctx):
        ctx.set_route(CallerIntent.BOOK_CONSULTATION, LegalArea.EMPLOYMENT)
        for f, v in [("matter_type", "dismissal"), ("matter_details", "Frist")]:
            ctx.store_entity(f, v, 1.0)
            ctx.confirm_entity(f)
        ctx.store_entity("name", "Reiner Ritter", 0.9)
        ctx.confirm_entity("name")
        assert ctx.state.phase == CallPhase.CAPTURE
        ctx.next_prompt()  # awaiting email

    def test_two_rejections_switch_to_spelling_then_capture(self):
        ctx = ConversationManager(call_id="spell1", lang="de")
        self._to_email_confirm(ctx)
        # First mishear + rejection.
        ctx.add_user_message("rita at gmail punkt com")
        ctx.next_prompt()
        assert ctx.state.awaiting == "email_confirm"
        ctx.add_user_message("Nein")
        ctx.next_prompt()
        assert ctx.state.awaiting == "email"  # one rejection → just re-ask
        # Second mishear + rejection → spelling mode.
        ctx.add_user_message("rietterre at gmail punkt com")
        ctx.next_prompt()
        assert ctx.state.awaiting == "email_confirm"
        ctx.add_user_message("Nein, falsch")
        line = ctx.next_prompt()
        assert ctx.state.awaiting == "email_spell"
        assert line is not None and "buchstab" in line.lower()
        assert ctx.state.email_spelling is True
        assert "email" not in ctx.state.entities
        # Caller spells; domain (gmail.com) carries over from the rejected read-back.
        ctx.add_user_message("R wie Richard, I, doppel T, E, R")
        line = ctx.next_prompt()
        assert ctx.state.awaiting == "email_confirm"
        assert line is not None and "ritter@gmail.com" in line
        ctx.add_user_message("Ja, korrekt")
        ctx.next_prompt()
        assert ctx.state.entities["email"].value == "ritter@gmail.com"
        assert ctx.state.entities["email"].confirmed is True

    @pytest.mark.asyncio
    async def test_llm_fallback_when_letters_dont_parse(self):
        ctx = ConversationManager(call_id="spell2", lang="de")

        async def fake_extractor(text, name_hint=""):
            return "ritter@gmail.com"

        ctx.set_email_extractor(fake_extractor)
        self._to_email_confirm(ctx)
        ctx.state.email_spelling = True
        ctx.state.email_domain = "gmail.com"
        ctx.next_prompt()
        assert ctx.state.awaiting == "email_spell"
        # STT merged the spelled letters into one word → deterministic parser defers.
        ctx.add_user_message("ritter")
        assert "email" not in ctx.state.entities
        await ctx.resolve_email_if_pending("ritter")
        assert ctx.state.entities["email"].value == "ritter@gmail.com"

    @pytest.mark.asyncio
    async def test_spelling_skips_email_after_max_attempts(self):
        ctx = ConversationManager(call_id="spell3", lang="de")
        self._to_email_confirm(ctx)
        ctx.state.email_spelling = True
        for _ in range(EmailCapture.MAX_SPELL_ATTEMPTS):
            ctx.next_prompt()
            assert ctx.state.awaiting == "email_spell"
            ctx.add_user_message("ähm keine Ahnung")
            await ctx.resolve_email_if_pending("ähm keine Ahnung")
        assert ctx.state.email_skipped is True
        ctx.next_prompt()  # recompute after the skip (as the pipeline does)
        assert ctx.state.awaiting == "phone"  # moved on to phone


class TestRequestedTimeParsing:
    """Callers name a time with or without 'Uhr'."""

    def _parse(self):
        from app.conversation.time_parse import parse_requested_time

        return parse_requested_time

    def test_uhr_with_minute(self):
        assert self._parse()("Haben Sie vierzehn Uhr dreißig?") == "14:30"

    def test_minute_without_uhr(self):
        # Real call: "vierzehn dreißig" (no "Uhr") must still mean 14:30.
        assert self._parse()("Haben Sie Zeit vierzehn dreißig?") == "14:30"

    def test_digit_hour_without_uhr(self):
        assert self._parse()("um 14 30") == "14:30"

    def test_full_hour(self):
        assert self._parse()("dreizehn Uhr") == "13:00"

    def test_any_hour_with_half_hour_without_uhr(self):
        # Not hardcoded to 14:30 — every offered half-hour parses without "Uhr".
        assert self._parse()("neun dreißig") == "09:30"
        assert self._parse()("zehn dreißig") == "10:30"

    def test_bare_hour_is_a_time_in_booking(self):
        # Only called when the caller is choosing a slot, so a bare hour is the
        # chosen time on the hour ("um 15" → 15 Uhr), not a stray number.
        assert self._parse()("um 15") == "15:00"
        assert self._parse()("vierzehn") == "14:00"
        assert self._parse()("Können wir um neun?") == "09:00"


class TestPhoneSplitDictation:
    """A phone dictated in bursts accumulates instead of confirming a fragment."""

    def _to_phone(self, ctx):
        ctx.set_route(CallerIntent.BOOK_CONSULTATION, LegalArea.EMPLOYMENT)
        for f, v in [("matter_type", "dismissal"), ("matter_details", "Frist")]:
            ctx.store_entity(f, v, 1.0)
            ctx.confirm_entity(f)
        ctx.store_entity("name", "Max Mustermann", 0.9)
        ctx.confirm_entity("name")
        ctx.state.email_skipped = True
        ctx.advance_phase()
        ctx.next_prompt()
        assert ctx.state.awaiting == "phone"

    def test_short_fragment_not_read_back(self):
        ctx = ConversationManager(call_id="ph1", lang="de")
        self._to_phone(ctx)
        ctx.add_user_message("plus vier neun eins fünf eins fünf sieben acht")  # 8 digits
        assert "phone" not in ctx.state.entities  # too short to confirm
        ctx.next_prompt()
        assert ctx.state.awaiting == "phone"  # still asking

    def test_split_dictation_accumulates_then_confirms(self):
        ctx = ConversationManager(call_id="ph2", lang="de")
        self._to_phone(ctx)
        ctx.add_user_message("plus vier neun eins fünf eins fünf sieben acht")
        ctx.next_prompt()
        ctx.add_user_message("drei eins sechs eins fünf")  # the rest, after a pause
        ctx.next_prompt()
        assert ctx.state.entities["phone"].value == "+4915157831615"
        assert ctx.state.awaiting == "phone_confirm"

    def test_more_digits_during_confirm_extend_the_number(self):
        ctx = ConversationManager(call_id="ph3", lang="de")
        self._to_phone(ctx)
        ctx.add_user_message(
            "null eins fünf eins fünf sieben acht drei eins"
        )  # 015157831 → 10 digits
        ctx.next_prompt()
        assert ctx.state.awaiting == "phone_confirm"
        ctx.add_user_message("sechs eins fünf")  # caller kept going
        ctx.next_prompt()
        assert ctx.state.entities["phone"].value == "+4915157831615"
        assert ctx.state.awaiting == "phone_confirm"

    def test_denial_clears_buffer_for_fresh_restatement(self):
        ctx = ConversationManager(call_id="ph4", lang="de")
        self._to_phone(ctx)
        ctx.add_user_message("null eins fünf eins fünf sieben acht drei eins sechs eins fünf")
        ctx.next_prompt()
        assert ctx.state.awaiting == "phone_confirm"
        ctx.add_user_message("Nein, das ist falsch")
        assert "phone" not in ctx.state.entities
        assert ctx.state.phone_buffer == ""
        ctx.next_prompt()
        assert ctx.state.awaiting == "phone"


class TestDeterministicBooking:
    """Slot selection is done in code (no LLM): offer, choose, book, confirm."""

    def _calendar(self, times):
        import asyncio
        import sqlite3
        import tempfile

        from app.services.calendar import CalendarService

        db = tempfile.mktemp(suffix=".db")
        cal = CalendarService(db_path=db)
        asyncio.run(cal.init_db())
        conn = sqlite3.connect(db)
        for t in times:
            conn.execute(
                "INSERT INTO slots (date,time,legal_area,lawyer_name,is_booked) VALUES (?,?,?,?,0)",
                ("2026-06-15", t, "employment", "Sarah Mitchell"),
            )
        conn.commit()
        conn.close()
        return cal

    def _ready_to_book(self, cal):
        ctx = ConversationManager(call_id="bk", lang="de", calendar=cal)
        ctx.set_route(CallerIntent.BOOK_CONSULTATION, LegalArea.EMPLOYMENT)
        for f, v in [("matter_type", "dismissal"), ("matter_details", "Frist"), ("name", "Anna")]:
            ctx.store_entity(f, v, 1.0)
            ctx.confirm_entity(f)
        ctx.state.email_skipped = True
        ctx.advance_phase()  # → CAPTURE (only phone missing)
        # Capture + confirm the phone via real turns so the BOOKING transition
        # (and the slot fetch inside add_user_message) fires like in a live call.
        ctx.next_prompt()  # ask_phone, awaiting phone
        ctx.add_user_message("0151 598 32614")
        ctx.next_prompt()  # confirm_phone, awaiting phone_confirm
        ctx.add_user_message("Ja, das stimmt")
        return ctx

    def test_offer_choose_book_confirm(self):
        cal = self._calendar(["09:00", "09:30", "10:00"])
        ctx = self._ready_to_book(cal)
        assert ctx.state.phase == CallPhase.BOOKING
        line = ctx.next_prompt()
        assert "9 Uhr" in line  # slots offered deterministically
        assert ctx.state.awaiting == "slot"
        ctx.add_user_message("Die erste passt")
        assert ctx.state.booking_confirmed is True
        assert ctx.state.phase == CallPhase.CONFIRMATION
        assert "gebucht" in ctx.next_prompt()

    def test_unavailable_offers_alternatives(self):
        cal = self._calendar(["09:00", "09:30", "10:00", "14:00", "14:30"])
        ctx = self._ready_to_book(cal)
        first = ctx.next_prompt()
        ctx.add_user_message("Die passen mir nicht")
        alts = ctx.next_prompt()
        assert "14 Uhr" in alts and alts != first  # different slots offered

    def test_choose_by_time(self):
        cal = self._calendar(["09:00", "14:00"])
        ctx = self._ready_to_book(cal)
        ctx.next_prompt()  # offer slots, awaiting slot
        ctx.add_user_message("14 Uhr bitte")
        assert ctx.state.booked_slot["time"] == "14:00"

    def test_requested_time_outside_offer_is_booked(self):
        # The offer shows the first three (9 / 9:30 / 10), but 13:00 also exists.
        # "Können wir dreizehn Uhr machen?" must book it, not repeat the offer.
        cal = self._calendar(["09:00", "09:30", "10:00", "13:00"])
        ctx = self._ready_to_book(cal)
        line = ctx.next_prompt()
        assert "13 Uhr" not in line  # 13:00 was not among the offered slots
        ctx.add_user_message("Können wir dreizehn Uhr machen?")
        assert ctx.state.booking_confirmed is True
        assert ctx.state.booked_slot["time"] == "13:00"

    def test_requested_time_unavailable_apologises_and_reoffers(self):
        # 18:00 is not in the calendar at all → don't loop; apologise + re-offer.
        cal = self._calendar(["09:00", "09:30", "10:00"])
        ctx = self._ready_to_book(cal)
        ctx.next_prompt()  # offer slots, awaiting slot
        ctx.add_user_message("Können wir achtzehn Uhr machen?")
        assert ctx.state.booking_confirmed is False
        line = ctx.next_prompt()
        assert "18 Uhr" in line  # names the unavailable time
        assert "9 Uhr" in line  # and lists the real alternatives

    def test_decline_excludes_time_across_lawyers(self):
        """Regression: with two lawyers per slot, declining a time must not
        re-offer the same time via the other lawyer's slot."""
        import asyncio
        import sqlite3
        import tempfile

        from app.services.calendar import CalendarService

        db = tempfile.mktemp(suffix=".db")
        cal = CalendarService(db_path=db)
        asyncio.run(cal.init_db())
        conn = sqlite3.connect(db)
        for t in ["09:00", "09:30", "10:00", "14:00", "14:30"]:
            for lawyer in ("Weber", "Hoffmann"):  # two slots per time
                conn.execute(
                    "INSERT INTO slots (date,time,legal_area,lawyer_name,is_booked)"
                    " VALUES (?,?,?,?,0)",
                    ("2026-06-15", t, "employment", lawyer),
                )
        conn.commit()
        conn.close()

        ctx = self._ready_to_book(cal)
        first = ctx.next_prompt()
        assert "9 Uhr" in first
        ctx.add_user_message("Die passen mir nicht")
        second = ctx.next_prompt()
        assert "9 Uhr" not in second  # the declined times are gone, not re-offered
        assert "14 Uhr" in second


class TestFullScriptedTrafficBooking:
    """End-to-end regression for the live call that faked a booking.

    Drives the whole scripted spine on a real SQLite calendar and asserts a real
    slot is booked and the caller row is persisted — the thing that silently did
    NOT happen before (phase stuck in QUALIFICATION, no phone/email, no booking).
    """

    def _calendar(self, times):
        import asyncio
        import sqlite3
        import tempfile

        from app.services.calendar import CalendarService

        db = tempfile.mktemp(suffix=".db")
        cal = CalendarService(db_path=db)
        asyncio.run(cal.init_db())
        conn = sqlite3.connect(db)
        for t in times:
            conn.execute(
                "INSERT INTO slots (date,time,legal_area,lawyer_name,is_booked) VALUES (?,?,?,?,0)",
                ("2026-06-15", t, "traffic", "Sarah Mitchell"),
            )
        conn.commit()
        conn.close()
        return cal

    def test_traffic_call_books_a_real_slot(self):
        cal = self._calendar(["09:00", "14:00"])
        ctx = ConversationManager(call_id="e2e-traffic", lang="de", calendar=cal)

        ctx.add_user_message("Ich hatte einen Autounfall")  # auto-route → traffic
        assert ctx.state.legal_area == LegalArea.TRAFFIC

        ctx.next_prompt()  # area_confirm, awaiting matter_type
        ctx.add_user_message("Verkehrsunfall")
        assert ctx.state.entities["matter_type"].value == "accident"

        ctx.next_prompt()  # traffic_insurance, awaiting insurance
        ctx.add_user_message("Ja, Versicherungsnummer F62314759")
        assert ctx.state.entities["insurance_number"].value == "F62314759"
        ctx.next_prompt()  # confirm_insurance read-back, awaiting insurance_confirm
        ctx.add_user_message("Ja, korrekt")  # confirm the number
        assert ctx.state.phase == CallPhase.CAPTURE

        ctx.next_prompt()  # ask_name, awaiting name
        ctx.add_user_message("Mein Name ist Felix Lang")
        assert ctx.state.entities["name"].value == "Felix Lang"

        ctx.next_prompt()  # ask_email, awaiting email
        ctx.add_user_message("felix at gmail punkt com")
        ctx.next_prompt()  # confirm_email, awaiting email_confirm
        ctx.add_user_message("Ja, stimmt")
        assert ctx.state.entities["email"].value == "felix@gmail.com"

        ctx.next_prompt()  # ask_phone, awaiting phone
        ctx.add_user_message("0151 598 32614")
        ctx.next_prompt()  # confirm_phone, awaiting phone_confirm
        ctx.add_user_message("Ja, das stimmt")
        assert ctx.state.phase == CallPhase.BOOKING

        offer = ctx.next_prompt()  # slot_offer, awaiting slot
        assert "9 Uhr" in offer
        ctx.add_user_message("Die erste passt")
        assert ctx.state.booking_confirmed is True
        assert ctx.state.booked_slot["time"] == "09:00"

        # The slot is really booked in the DB (no double-book).
        import asyncio

        remaining = asyncio.run(cal.get_available_slots("traffic"))
        assert all(s["time"] != "09:00" for s in remaining)

    def test_books_with_requested_lawyer(self):
        """Story 4 handoff: asking for Frau Hoffmann books a consultation WITH her —
        her slots are offered and the confirmation names her."""
        import asyncio
        import sqlite3
        import tempfile

        from app.services.calendar import CalendarService

        db = tempfile.mktemp(suffix=".db")
        cal = CalendarService(db_path=db)
        asyncio.run(cal.init_db())
        conn = sqlite3.connect(db)
        for t in ["09:00", "09:30", "10:00"]:
            for lawyer in ("Lisa Hoffmann", "Michael Weber"):
                conn.execute(
                    "INSERT INTO slots (date,time,legal_area,lawyer_name,is_booked)"
                    " VALUES (?,?,?,?,0)",
                    ("2026-06-15", t, "traffic", lawyer),
                )
        conn.commit()
        conn.close()

        ctx = ConversationManager(call_id="e2e-lawyer", lang="de", calendar=cal)
        ctx.add_user_message("Ich möchte mit Frau Hoffmann sprechen.")
        assert ctx.state.target_person == "Frau Hoffmann"
        assert ctx.state.callback_requested is False

        # The matter comes next → keyword-routes to traffic (intent already 'book').
        ctx.add_user_message("Es geht um einen Autounfall.")
        assert ctx.state.legal_area == LegalArea.TRAFFIC

        # Fast-forward qualification + name/email deterministically; capture the
        # phone via real turns so the BOOKING transition (and slot fetch) fire.
        ctx.state.insurance_resolved = True
        ctx.store_entity("matter_type", "accident", 1.0)
        ctx.confirm_entity("matter_type")
        ctx.store_entity("name", "Felix Lang", 1.0)
        ctx.confirm_entity("name")
        ctx.state.email_skipped = True
        ctx.advance_phase()
        assert ctx.state.phase == CallPhase.CAPTURE
        ctx.next_prompt()  # ask_phone
        ctx.add_user_message("0151 598 32614")
        ctx.next_prompt()  # confirm_phone
        ctx.add_user_message("Ja, das stimmt")
        assert ctx.state.phase == CallPhase.BOOKING

        offer = ctx.next_prompt()  # only Frau Hoffmann's slots are offered
        assert offer is not None and "9 Uhr" in offer
        assert all("Hoffmann" in s["lawyer_name"] for s in ctx.state.offered_slots)

        ctx.add_user_message("Die erste passt")
        assert ctx.state.booking_confirmed is True
        assert "Hoffmann" in ctx.state.booked_slot["lawyer_name"]
        done = ctx.next_prompt()  # confirmation names the requested lawyer
        assert done is not None and "Frau Hoffmann" in done

    def test_email_misheard_triggers_reask(self):
        ctx = ConversationManager(call_id="e2e-email", lang="de")
        ctx.set_route(CallerIntent.BOOK_CONSULTATION, LegalArea.EMPLOYMENT)
        ctx.store_entity("matter_type", "dismissal", 1.0)
        ctx.confirm_entity("matter_type")
        ctx.store_entity("matter_details", "Frist", 1.0)
        ctx.confirm_entity("matter_details")
        ctx.store_entity("name", "Anna Schmidt", 1.0)
        ctx.confirm_entity("name")

        first = ctx.next_prompt()  # ask_email
        assert ctx.state.awaiting == "email"
        ctx.add_user_message("Wie bitte? Können Sie das wiederholen?")  # no parseable email
        reask = ctx.next_prompt()  # should be the "didn't catch it" variant
        assert ctx.state.awaiting == "email"
        assert reask != first
        assert "nicht verstanden" in reask

    def test_low_confidence_name_is_confirmed(self):
        ctx = ConversationManager(call_id="e2e-name", lang="de")
        ctx.set_route(CallerIntent.BOOK_CONSULTATION, LegalArea.EMPLOYMENT)
        ctx.store_entity("matter_type", "dismissal", 1.0)
        ctx.confirm_entity("matter_type")
        ctx.store_entity("matter_details", "Frist", 1.0)
        ctx.confirm_entity("matter_details")

        ctx.next_prompt()  # ask_name, awaiting name
        ctx.set_transcription_confidence(0.55)  # noisy line
        ctx.add_user_message("Daniel Stein")
        # Stored but unconfirmed → a read-back is required.
        assert ctx.state.entities["name"].confirmed is False
        line = ctx.next_prompt()
        assert ctx.state.awaiting == "name_confirm"
        assert "Daniel Stein" in line
        ctx.add_user_message("Ja, genau")
        assert ctx.state.entities["name"].confirmed is True


class TestSpokenEmailParsing:
    """Spoken email is the hardest field: chunked local parts and graceful skip."""

    def test_chunked_local_part_is_joined(self):
        from app.conversation.email_parse import parse_email

        # Regression: "Lang M at gmail.com" used to collapse to "m@gmail.com"
        # because the space before the @-token dropped the "lang" prefix.
        assert parse_email("Lang M at gmail.com") == "langm@gmail.com"
        assert parse_email("Lang M at gmail punkt com") == "langm@gmail.com"
        assert parse_email("meine email ist lang m at gmail punkt com") == "langm@gmail.com"
        assert parse_email("max punkt mueller at gmail punkt com") == "max.mueller@gmail.com"
        # Non-emails must still yield nothing (so we re-ask, not mis-store).
        assert parse_email("Gmail.com") is None
        assert parse_email("Wie bitte?") is None

    def test_spoken_at_between_dots_does_not_leave_stray_dot(self):
        from app.conversation.email_parse import parse_email

        # Regression: Whisper renders a spoken "at" between dots
        # ("baum.at.gmail.com"), which used to parse to "baum.@gmail.com" — an
        # invalid local part. The dot touching the @ must be collapsed away.
        assert parse_email("baum.at.gmail.com") == "baum@gmail.com"
        assert parse_email("baum.at gmail.com") == "baum@gmail.com"
        assert parse_email("baum at gmail punkt com") == "baum@gmail.com"

    def test_leadin_es_ist_is_stripped(self):
        from app.conversation.email_parse import parse_email

        # Regression: "Ja, es ist Sigmar at ..." leaked the lead-in into the local
        # part as "esistsigmar@..." because "es" wasn't a recognised filler word.
        assert parse_email("Ja, es ist Sigmar at Gmail dot com.") == "sigmar@gmail.com"

    def test_name_anchoring_snaps_near_miss_local_part(self):
        from app.conversation.email_parse import anchor_email_to_name

        # STT dropped a letter: heard "sigma", caller's name is "Leon Sigmar".
        assert anchor_email_to_name("sigma@gmail.com", "Leon Sigmar") == "sigmar@gmail.com"
        # Exact match and genuinely different addresses are left untouched.
        assert anchor_email_to_name("sigmar@gmail.com", "Leon Sigmar") == "sigmar@gmail.com"
        assert anchor_email_to_name("leon.legal@gmail.com", "Leon Sigmar") == "leon.legal@gmail.com"
        assert anchor_email_to_name("x@gmail.com", "") == "x@gmail.com"  # no name → no-op

    def _email_ready_ctx(self, call_id):
        ctx = ConversationManager(call_id=call_id, lang="de")
        ctx.set_route(CallerIntent.BOOK_CONSULTATION, LegalArea.EMPLOYMENT)
        for f, v in [
            ("matter_type", "dismissal"),
            ("matter_details", "Frist"),
            ("name", "Albert Klein"),
        ]:
            ctx.store_entity(f, v, 1.0)
            ctx.confirm_entity(f)
        ctx.advance_phase()
        ctx.next_prompt()  # awaiting email
        return ctx

    def test_split_email_is_accumulated_across_turns(self):
        # The live regression: caller dictates "Klein at Hotmail" then (after a
        # pause that split the turn) "Punkt de" — the halves must reassemble.
        ctx = self._email_ready_ctx("split-email")
        ctx.add_user_message("Klein at Hotmail")
        assert "email" not in ctx.state.entities  # incomplete — still collecting
        ctx.add_user_message("Punkt de")
        assert ctx.state.entities["email"].value == "klein@hotmail.de"

    def test_rescue_does_not_invent_tld_without_spoken_suffix(self):
        import asyncio

        async def fake_llm(_text, _name=""):
            return "klein@hotmail.com"  # the model's ".com" guess

        ctx = self._email_ready_ctx("no-tld")
        ctx.set_email_extractor(fake_llm)
        ctx.add_user_message("Klein at Hotmail")  # no "punkt/dot" spoken yet
        asyncio.run(ctx.resolve_email_if_pending("Klein at Hotmail"))
        assert "email" not in ctx.state.entities  # must NOT lock in hotmail.com

    def test_anchoring_applied_when_capturing_email_after_name(self):
        ctx = ConversationManager(call_id="anchor-e2e", lang="de")
        ctx.set_route(CallerIntent.BOOK_CONSULTATION, LegalArea.TRAFFIC)
        ctx.store_entity("matter_type", "accident", 1.0)
        ctx.confirm_entity("matter_type")
        ctx.state.insurance_resolved = True
        ctx.store_entity("name", "Leon Sigmar", 0.95)
        ctx.confirm_entity("name")
        ctx.advance_phase()
        ctx.next_prompt()  # ask_email → awaiting email (name already captured)
        ctx.add_user_message("Sigma at Gmail dot com")  # STT dropped the 'r'
        assert ctx.state.entities["email"].value == "sigmar@gmail.com"

    def test_llm_rescue_recovers_email_regex_missed(self):
        """When regex fails and an extractor is configured, the LLM rescue stores it."""
        import asyncio

        async def fake_llm(_text, _name=""):
            return "langm@gmail.com"

        ctx = ConversationManager(call_id="e2e-llm-email", lang="de")
        ctx.set_email_extractor(fake_llm)
        ctx.set_route(CallerIntent.BOOK_CONSULTATION, LegalArea.EMPLOYMENT)
        ctx.store_entity("matter_type", "dismissal", 1.0)
        ctx.confirm_entity("matter_type")
        ctx.store_entity("matter_details", "Frist", 1.0)
        ctx.confirm_entity("matter_details")
        ctx.store_entity("name", "Anna", 1.0)
        ctx.confirm_entity("name")

        ctx.next_prompt()  # ask_email, awaiting email
        # A spoken suffix ("punkt com") is present, so the rescue is allowed to run.
        garbled = "ähm Lang M Gmail punkt com irgendwas"
        ctx.add_user_message(garbled)
        assert "email" not in ctx.state.entities  # regex couldn't parse it
        asyncio.run(ctx.resolve_email_if_pending(garbled))
        assert ctx.state.entities["email"].value == "langm@gmail.com"
        line = ctx.next_prompt()
        assert ctx.state.awaiting == "email_confirm"
        assert "langm@gmail.com" in line

    def test_llm_rescue_is_noop_without_extractor(self):
        """Local default: no extractor → resolve_email_if_pending does nothing."""
        import asyncio

        ctx = ConversationManager(call_id="e2e-no-llm", lang="de")
        ctx.set_route(CallerIntent.BOOK_CONSULTATION, LegalArea.EMPLOYMENT)
        ctx.store_entity("matter_type", "dismissal", 1.0)
        ctx.confirm_entity("matter_type")
        ctx.store_entity("matter_details", "Frist", 1.0)
        ctx.confirm_entity("matter_details")
        ctx.store_entity("name", "Anna", 1.0)
        ctx.confirm_entity("name")

        ctx.next_prompt()
        ctx.add_user_message("Lang M Gmail irgendwas")
        asyncio.run(ctx.resolve_email_if_pending("Lang M Gmail irgendwas"))
        assert "email" not in ctx.state.entities  # stays regex-only

    def test_email_skipped_after_repeated_misses(self):
        ctx = ConversationManager(call_id="e2e-email-skip", lang="de")
        ctx.set_route(CallerIntent.BOOK_CONSULTATION, LegalArea.EMPLOYMENT)
        ctx.store_entity("matter_type", "dismissal", 1.0)
        ctx.confirm_entity("matter_type")
        ctx.store_entity("matter_details", "Frist", 1.0)
        ctx.confirm_entity("matter_details")
        ctx.store_entity("name", "Anna Schmidt", 1.0)
        ctx.confirm_entity("name")

        for _ in range(3):
            ctx.next_prompt()  # asks / re-asks email
            ctx.add_user_message("was bitte?")  # never parseable
        # After the cap: email skipped, moves on to the phone ask (no infinite loop).
        assert ctx.state.email_skipped is True
        line = ctx.next_prompt()
        assert ctx.state.awaiting == "phone"
        assert line is not None
