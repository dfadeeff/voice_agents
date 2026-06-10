"""Comprehensive German-language tests.

Covers: prompts for every phase, TTS preprocessing, false booking guard
(including state-aware bypass), PreTTSSanitizer streaming, booking validation
for missing fields, and a full German booking scenario with tool calls.
"""

import pytest

from app.conversation.manager import ConversationManager
from app.conversation.prompts import build_system_prompt
from app.conversation.state import ConversationState
from app.models.schemas import CallerIntent, CallPhase, ExtractedEntity, LegalArea
from app.pipeline.processors import (
    PreTTSSanitizer,
    _guard_false_booking,
    _tts_preprocess,
)


def _confirmed(field, value="test"):
    return ExtractedEntity(field_name=field, value=value, confidence=0.95, confirmed=True)


def _unconfirmed(field, value="test"):
    return ExtractedEntity(field_name=field, value=value, confidence=0.5, confirmed=False)


# ---------------------------------------------------------------------------
# German prompt content per phase
# ---------------------------------------------------------------------------


class TestGermanPrompts:
    """Verify every phase prompt contains the right German keywords."""

    def test_greeting_prompt(self):
        state = ConversationState(call_id="t")
        prompt = build_system_prompt(state, lang="de")
        assert "Claudia" in prompt
        assert "KI-Anrufannahme" in prompt
        assert "Kanzlei" in prompt

    def test_routing_prompt(self):
        state = ConversationState(call_id="t")
        state.phase = CallPhase.ROUTING
        state.turn_count = 1
        prompt = build_system_prompt(state, lang="de")
        assert "Rechtsgebiet" in prompt
        assert "route_call" in prompt

    def test_qualification_prompt_employment(self):
        state = ConversationState(call_id="t")
        state.phase = CallPhase.QUALIFICATION
        state.legal_area = LegalArea.EMPLOYMENT
        prompt = build_system_prompt(state, lang="de")
        assert "Kündigung" in prompt
        assert "capture_caller_details" in prompt
        assert "matter_type" in prompt

    def test_qualification_prompt_tenancy(self):
        state = ConversationState(call_id="t")
        state.phase = CallPhase.QUALIFICATION
        state.legal_area = LegalArea.TENANCY
        prompt = build_system_prompt(state, lang="de")
        assert "Kaution" in prompt
        assert "Vermieter" in prompt

    def test_qualification_prompt_traffic(self):
        state = ConversationState(call_id="t")
        state.phase = CallPhase.QUALIFICATION
        state.legal_area = LegalArea.TRAFFIC
        prompt = build_system_prompt(state, lang="de")
        assert "Unfall" in prompt or "Verkehrsunfall" in prompt
        assert "Versicherung" in prompt

    def test_qualification_details_prompt(self):
        state = ConversationState(call_id="t")
        state.phase = CallPhase.QUALIFICATION
        state.legal_area = LegalArea.EMPLOYMENT
        state.entities = {
            "matter_type": ExtractedEntity(
                field_name="matter_type", value="dismissal", confidence=0.9, confirmed=True
            )
        }
        prompt = build_system_prompt(state, lang="de")
        assert "Frist" in prompt
        assert "matter_details" in prompt

    def test_capture_prompt_shows_missing_fields(self):
        state = ConversationState(call_id="t")
        state.phase = CallPhase.CAPTURE
        state.caller_intent = CallerIntent.BOOK_CONSULTATION
        state.legal_area = LegalArea.EMPLOYMENT
        prompt = build_system_prompt(state, lang="de")
        assert "FEHLENDE FELDER" in prompt
        assert "name" in prompt
        assert "email" in prompt
        assert "phone" in prompt

    def test_capture_prompt_shows_unconfirmed_fields(self):
        state = ConversationState(call_id="t")
        state.phase = CallPhase.CAPTURE
        state.caller_intent = CallerIntent.BOOK_CONSULTATION
        state.legal_area = LegalArea.EMPLOYMENT
        state.entities = {
            "name": _confirmed("name", "Max Müller"),
            "email": _unconfirmed("email", "max@example.com"),
        }
        prompt = build_system_prompt(state, lang="de")
        assert "BESTÄTIGUNG NÖTIG" in prompt
        assert "max@example.com" in prompt
        assert "phone" in prompt  # still missing

    def test_capture_prompt_mentions_email_spelling(self):
        state = ConversationState(call_id="t")
        state.phase = CallPhase.CAPTURE
        prompt = build_system_prompt(state, lang="de")
        assert "buchstab" in prompt.lower()

    def test_booking_prompt(self):
        state = ConversationState(call_id="t")
        state.phase = CallPhase.BOOKING
        prompt = build_system_prompt(state, lang="de")
        assert "book_consultation" in prompt
        assert "check_availability" in prompt
        assert "gebucht" in prompt.lower() or "Terminwunsch" in prompt

    def test_confirmation_prompt(self):
        state = ConversationState(call_id="t")
        state.phase = CallPhase.CONFIRMATION
        prompt = build_system_prompt(state, lang="de")
        assert "Termin" in prompt or "Datum" in prompt

    def test_escalation_prompt(self):
        state = ConversationState(call_id="t")
        state.phase = CallPhase.ESCALATION
        prompt = build_system_prompt(state, lang="de")
        assert "Anliegen" in prompt
        assert "NICHT live verbinden" in prompt

    def test_information_prompt(self):
        state = ConversationState(call_id="t")
        state.phase = CallPhase.INFORMATION
        prompt = build_system_prompt(state, lang="de")
        assert "Rechtsberatung" in prompt or "Informationen" in prompt

    def test_preamble_forbids_legal_advice(self):
        state = ConversationState(call_id="t")
        prompt = build_system_prompt(state, lang="de")
        assert "NIEMALS" in prompt
        assert "Rechtsberatung" in prompt

    def test_preamble_forbids_false_booking(self):
        state = ConversationState(call_id="t")
        prompt = build_system_prompt(state, lang="de")
        assert "gebucht" in prompt.lower() or "bestätigt" in prompt.lower()
        assert "Termin" in prompt

    def test_preamble_forbids_connection(self):
        state = ConversationState(call_id="t")
        prompt = build_system_prompt(state, lang="de")
        assert "KANNST NIEMANDEN VERBINDEN" in prompt

    def test_preamble_forbids_tool_name_disclosure(self):
        state = ConversationState(call_id="t")
        prompt = build_system_prompt(state, lang="de")
        assert "Funktionsnamen" in prompt or "NIEMALS" in prompt

    def test_employment_fragment_included_in_routing(self):
        state = ConversationState(call_id="t")
        state.phase = CallPhase.ROUTING
        state.legal_area = LegalArea.EMPLOYMENT
        prompt = build_system_prompt(state, lang="de")
        assert "ARBEITSRECHT" in prompt
        assert "Kündigungsschutz" in prompt

    def test_tenancy_fragment_included_in_routing(self):
        state = ConversationState(call_id="t")
        state.phase = CallPhase.ROUTING
        state.legal_area = LegalArea.TENANCY
        prompt = build_system_prompt(state, lang="de")
        assert "MIETRECHT" in prompt

    def test_traffic_fragment_included_in_routing(self):
        state = ConversationState(call_id="t")
        state.phase = CallPhase.ROUTING
        state.legal_area = LegalArea.TRAFFIC
        prompt = build_system_prompt(state, lang="de")
        assert "VERKEHRSRECHT" in prompt

    def test_no_think_only_for_qwen3(self):
        from app.conversation.prompts import _is_qwen3

        state = ConversationState(call_id="t")
        prompt = build_system_prompt(state, lang="de")
        if _is_qwen3:
            assert prompt.endswith("/no_think")
        else:
            assert "/no_think" not in prompt


# ---------------------------------------------------------------------------
# German TTS preprocessing
# ---------------------------------------------------------------------------


class TestGermanTTSPreprocessing:
    def test_email_at_becomes_at(self):
        result = _tts_preprocess("max@example.com", lang="de")
        assert "at" in result
        assert "@" not in result

    def test_email_dot_becomes_punkt(self):
        result = _tts_preprocess("max@example.com", lang="de")
        assert "Punkt" in result

    def test_phone_digits_separated(self):
        result = _tts_preprocess("017612345678", lang="de")
        assert "0" in result
        assert "1" in result
        assert "," in result

    def test_phone_with_plus_prefix(self):
        result = _tts_preprocess("+49 176 12345678", lang="de")
        assert "plus" in result

    def test_sentences_glued_without_space_are_separated(self):
        # Scripted lines concatenate sentences ("Tag!Hier", "können.Wobei"); TTS
        # needs a space so it pauses between them instead of slurring the junction.
        out = _tts_preprocess("Guten Tag!Hier ist Claudia. Kanzlei.Ich helfe.", lang="de")
        assert "Tag! Hier" in out
        assert "Kanzlei. Ich" in out

    def test_normal_text_unchanged(self):
        text = "Verstanden, es geht um eine Kündigung."
        assert _tts_preprocess(text, lang="de") == text

    def test_think_tags_stripped(self):
        result = _tts_preprocess("<think>reasoning</think>Hallo", lang="de")
        assert "think" not in result
        assert "Hallo" in result

    def test_cjk_stripped(self):
        result = _tts_preprocess("你好 Guten Tag", lang="de")
        assert "你好" not in result
        assert "Guten Tag" in result

    def test_phone_trailing_period_stripped(self):
        result = _tts_preprocess("Ihre Nummer ist 151-597-30614.", lang="de")
        assert result.endswith("4")
        assert not result.endswith("4.")

    def test_english_email_uses_dot_not_punkt(self):
        result = _tts_preprocess("john@example.com", lang="en")
        assert "dot" in result
        assert "Punkt" not in result


# ---------------------------------------------------------------------------
# False booking guard — state-aware
# ---------------------------------------------------------------------------


class TestFalseBookingStateAware:
    def test_blocks_when_not_confirmed(self):
        bad = "Ihr Termin ist gebucht."
        result = _guard_false_booking(bad, booking_confirmed=False)
        assert "gebucht" not in result.lower()
        assert "Terminwunsch" in result

    def test_allows_when_booking_confirmed(self):
        text = "Ihr Termin ist gebucht."
        result = _guard_false_booking(text, booking_confirmed=True)
        assert result == text

    def test_allows_bestätigt_when_confirmed(self):
        text = "Ihr Termin ist bestätigt für Montag um 9 Uhr."
        result = _guard_false_booking(text, booking_confirmed=True)
        assert result == text

    def test_blocks_bestätigt_when_not_confirmed(self):
        text = "Ihr Termin ist bestätigt für Montag um 9 Uhr."
        result = _guard_false_booking(text, booking_confirmed=False)
        assert "bestätigt" not in result.lower()

    def test_default_is_not_confirmed(self):
        text = "Termin steht."
        result = _guard_false_booking(text)
        assert "steht" not in result.lower() or "Terminwunsch" in result

    def test_blocks_ich_habe_termin_gebucht(self):
        text = "Ich habe Ihnen einen Termin gebucht."
        result = _guard_false_booking(text, booking_confirmed=False)
        assert "gebucht" not in result.lower()

    def test_allows_terminwunsch_always(self):
        text = "Ich nehme Ihren Terminwunsch auf."
        assert _guard_false_booking(text, booking_confirmed=False) == text
        assert _guard_false_booking(text, booking_confirmed=True) == text


# ---------------------------------------------------------------------------
# PreTTSSanitizer streaming simulation
# ---------------------------------------------------------------------------


class TestPreTTSSanitizerStreaming:
    """Simulate chunk-by-chunk LLM output through the sanitizer."""

    def _make(self, tool_names=None):
        return PreTTSSanitizer(lang="de", tool_names=tool_names)

    def test_think_then_text_across_three_chunks(self):
        san = self._make()
        assert san._strip_think("<think>") == ""
        assert san._in_think is True
        assert san._strip_think("internal reasoning") == ""
        assert san._strip_think("</think>Guten Tag") == "Guten Tag"
        assert san._in_think is False

    def test_think_and_text_in_single_chunk(self):
        san = self._make()
        result = san._strip_think("<think>r</think>Hallo")
        assert result == "Hallo"
        assert san._in_think is False

    def test_multiple_think_blocks(self):
        san = self._make()
        result = san._strip_think("<think>a</think>Ja<think>b</think>, genau.")
        assert result == "Ja, genau."

    def test_tool_name_in_chunk(self):
        san = self._make(["route_call"])
        assert san._tool_re is not None
        assert san._tool_re.search("route call")

    def test_json_leak_stripped(self):
        from app.pipeline.processors import _JSON_LEAK_RE

        assert _JSON_LEAK_RE.search('{"legal_area": "employment"}')

    def test_normal_german_text_passes_through(self):
        san = self._make(["route_call", "request_handoff"])
        text = "Ich verstehe, es geht um eine Kündigung."
        assert san._strip_think(text) == text
        assert san._tool_re is not None
        assert san._tool_re.sub("", text) == text


# ---------------------------------------------------------------------------
# Booking validation — missing fields
# ---------------------------------------------------------------------------


class TestBookingValidation:
    @pytest.fixture()
    def ctx_de(self):
        return ConversationManager(call_id="booking-test", lang="de")

    @pytest.mark.asyncio
    async def test_booking_blocked_when_fields_missing(self, registry, ctx_de):
        ctx = ctx_de
        ctx.add_user_message("Ich brauche einen Termin")
        ctx.set_route(CallerIntent.BOOK_CONSULTATION, LegalArea.EMPLOYMENT)
        ctx.store_entity("name", "Max Müller", 0.95)
        ctx.confirm_entity("name")

        result = await registry.execute(
            "book_consultation",
            {"slot_id": 1, "caller_name": "Max Müller"},
            ctx,
        )
        assert result["status"] == "blocked"
        # Email is optional; only the phone number is required (plus name).
        assert "phone" in result["missing_fields"]
        assert "email" not in result["missing_fields"]

    @pytest.mark.asyncio
    async def test_booking_not_blocked_by_optional_email(self, registry, ctx_de):
        """Name + phone confirmed is enough — an unconfirmed email does not block."""
        ctx = ctx_de
        ctx.add_user_message("Ich brauche einen Termin")
        ctx.set_route(CallerIntent.BOOK_CONSULTATION, LegalArea.EMPLOYMENT)
        ctx.store_entity("name", "Test", 0.95)
        ctx.confirm_entity("name")
        ctx.store_entity("phone", "+4915112345678", 0.9)
        ctx.confirm_entity("phone")
        ctx.store_entity("email", "test@example.com", 0.9)  # unconfirmed

        result = await registry.execute(
            "book_consultation",
            {"slot_id": 1, "caller_name": "Test"},
            ctx,
        )
        assert result["status"] != "blocked"

    @pytest.mark.asyncio
    async def test_booking_proceeds_when_all_confirmed(self, registry, ctx_de):
        ctx = ctx_de
        ctx.add_user_message("Termin bitte")
        ctx.set_route(CallerIntent.BOOK_CONSULTATION, LegalArea.EMPLOYMENT)
        for f in ("name", "email", "phone"):
            ctx.store_entity(f, f"test_{f}", 0.95)
            ctx.confirm_entity(f)
        ctx.store_entity("matter_type", "dismissal", 1.0)
        ctx.confirm_entity("matter_type")
        ctx.store_entity("matter_details", "deadline soon", 1.0)
        ctx.confirm_entity("matter_details")
        ctx.advance_phase()

        avail = await registry.execute(
            "check_availability",
            {"date": "2026-06-10", "legal_area": "employment"},
            ctx,
        )
        assert avail["available"] is True
        slot_id = avail["slots"][0]["id"]

        result = await registry.execute(
            "book_consultation",
            {"slot_id": slot_id, "caller_name": "test_name"},
            ctx,
        )
        assert result["status"] == "booked"


# ---------------------------------------------------------------------------
# Full German booking scenario
# ---------------------------------------------------------------------------


class TestGermanFullBookingScenario:
    @pytest.fixture()
    def ctx_de(self):
        return ConversationManager(call_id="de-scenario", lang="de")

    @pytest.mark.asyncio
    async def test_german_happy_path(self, registry, ctx_de):
        ctx = ctx_de
        assert ctx.state.phase == CallPhase.GREETING

        ctx.add_user_message("Hallo, ich wurde letzte Woche gekündigt.")
        assert ctx.state.phase == CallPhase.QUALIFICATION
        assert ctx.state.legal_area == LegalArea.EMPLOYMENT

        ctx.add_user_message("Es geht um eine Kündigung")
        await registry.execute("capture_caller_details", {"matter_type": "dismissal"}, ctx)
        assert ctx.state.phase == CallPhase.QUALIFICATION

        ctx.add_user_message("Ja, die Frist läuft in zwei Wochen ab")
        await registry.execute(
            "capture_caller_details", {"matter_details": "deadline in 2 weeks"}, ctx
        )
        assert ctx.state.phase == CallPhase.CAPTURE

        ctx.add_user_message("Mein Name ist Max Müller")
        await registry.execute("capture_caller_details", {"name": "Max Müller"}, ctx)

        ctx.add_user_message("max punkt mueller at gmail punkt com")
        await registry.execute("capture_caller_details", {"email": "max.mueller@gmail.com"}, ctx)

        ctx.add_user_message("Ja, das stimmt")
        await registry.execute(
            "confirm_caller_detail",
            {"field": "email", "confirmed_value": "max.mueller@gmail.com", "status": "accepted"},
            ctx,
        )

        ctx.add_user_message("null eins sieben sechs eins zwei drei vier fünf sechs sieben acht")
        await registry.execute("capture_caller_details", {"phone": "017612345678"}, ctx)

        ctx.add_user_message("Ja")
        await registry.execute(
            "confirm_caller_detail",
            {"field": "phone", "confirmed_value": "017612345678", "status": "accepted"},
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
            {"slot_id": slot_id, "caller_name": "Max Müller"},
            ctx,
        )
        assert result["status"] == "booked"
        assert ctx.state.booking_confirmed is True
        assert ctx.state.phase == CallPhase.CONFIRMATION

    @pytest.mark.asyncio
    async def test_german_tenancy_flow(self, registry, ctx_de):
        ctx = ctx_de
        ctx.add_user_message("Mein Vermieter will mich rauswerfen.")
        assert ctx.state.phase == CallPhase.QUALIFICATION
        assert ctx.state.legal_area == LegalArea.TENANCY

        await registry.execute("capture_caller_details", {"matter_type": "eviction"}, ctx)
        assert ctx.state.phase == CallPhase.QUALIFICATION

        ctx.add_user_message("Nein, noch nicht schriftlich")
        await registry.execute(
            "capture_caller_details", {"matter_details": "not yet in writing"}, ctx
        )
        assert ctx.state.phase == CallPhase.CAPTURE

        for field, value in [
            ("name", "Anna Schmidt"),
            ("email", "anna@example.com"),
            ("phone", "+4917612345678"),
        ]:
            await registry.execute("capture_caller_details", {field: value}, ctx)
            if field in ("email", "phone"):
                await registry.execute(
                    "confirm_caller_detail",
                    {"field": field, "confirmed_value": value, "status": "accepted"},
                    ctx,
                )

        assert ctx.state.phase == CallPhase.BOOKING

    @pytest.mark.asyncio
    async def test_german_escalation_mid_flow(self, registry, ctx_de):
        ctx = ctx_de
        ctx.add_user_message("Ich möchte mit jemandem sprechen.")

        result = await registry.execute(
            "request_handoff",
            {"reason": "caller_requested_human", "summary": "Möchte einen Anwalt sprechen"},
            ctx,
        )
        assert result["status"] == "handoff_requested"
        assert result["mode"] == "callback"
        assert ctx.state.callback_requested is True
        assert ctx.state.phase == CallPhase.CAPTURE


# ---------------------------------------------------------------------------
# German email confirmation via confirm_caller_detail
# ---------------------------------------------------------------------------


class TestGermanEmailConfirmation:
    @pytest.mark.asyncio
    async def test_email_needs_confirmation_first_call(self, registry):
        ctx = ConversationManager(call_id="de-email-test", lang="de")
        ctx.add_user_message("ameliesommer at gmail punkt com")
        ctx.set_route(CallerIntent.BOOK_CONSULTATION, LegalArea.EMPLOYMENT)
        await registry.execute("capture_caller_details", {"matter_type": "dismissal"}, ctx)
        await registry.execute(
            "capture_caller_details", {"matter_details": "deadline in 2 weeks"}, ctx
        )

        result = await registry.execute(
            "capture_caller_details",
            {"email": "ameliesommer@gmail.com"},
            ctx,
        )
        assert not result["all_confirmed"]
        assert any(c["field"] == "email" for c in result["needs_confirmation"])
        assert not ctx.state.entities["email"].confirmed

    @pytest.mark.asyncio
    async def test_email_confirms_via_confirm_tool(self, registry):
        ctx = ConversationManager(call_id="de-email-confirm", lang="de")
        ctx.add_user_message("max at example punkt com")
        ctx.set_route(CallerIntent.BOOK_CONSULTATION, LegalArea.EMPLOYMENT)
        await registry.execute("capture_caller_details", {"matter_type": "dismissal"}, ctx)
        await registry.execute(
            "capture_caller_details", {"matter_details": "deadline in 2 weeks"}, ctx
        )

        await registry.execute("capture_caller_details", {"email": "max@example.com"}, ctx)
        assert not ctx.state.entities["email"].confirmed

        ctx.add_user_message("ja, das stimmt")
        result = await registry.execute(
            "confirm_caller_detail",
            {"field": "email", "confirmed_value": "max@example.com", "status": "accepted"},
            ctx,
        )
        assert result["status"] == "confirmed"
        assert ctx.state.entities["email"].confirmed

    @pytest.mark.asyncio
    async def test_email_corrected_via_confirm_tool(self, registry):
        ctx = ConversationManager(call_id="de-email-correct", lang="de")
        ctx.add_user_message("max at example punkt com")
        ctx.set_route(CallerIntent.BOOK_CONSULTATION, LegalArea.EMPLOYMENT)

        await registry.execute("capture_caller_details", {"email": "mex@example.com"}, ctx)

        result = await registry.execute(
            "confirm_caller_detail",
            {"field": "email", "confirmed_value": "max@example.com", "status": "corrected"},
            ctx,
        )
        assert result["status"] == "confirmed"
        assert result["was_corrected"] is True
        assert ctx.state.entities["email"].value == "max@example.com"
        assert ctx.state.entities["email"].confirmed
