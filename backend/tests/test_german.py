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
        assert "Guten Tag" in prompt
        assert "Kanzlei" in prompt
        assert "Anliegen" in prompt

    def test_intent_detection_prompt(self):
        state = ConversationState(call_id="t")
        state.phase = CallPhase.INTENT_DETECTION
        state.turn_count = 1
        prompt = build_system_prompt(state, lang="de")
        assert "classify_caller_intent" in prompt
        assert "Empfang" in prompt

    def test_routing_prompt(self):
        state = ConversationState(call_id="t")
        state.phase = CallPhase.ROUTING
        state.caller_intent = CallerIntent.BOOK_CONSULTATION
        prompt = build_system_prompt(state, lang="de")
        assert "Rechtsgebiet" in prompt
        assert "classify_legal_area" in prompt
        assert "employment" in prompt or "Arbeitsrecht" in prompt

    def test_intake_prompt(self):
        state = ConversationState(call_id="t")
        state.phase = CallPhase.INTAKE
        state.legal_area = LegalArea.EMPLOYMENT
        prompt = build_system_prompt(state, lang="de")
        assert "complete_intake" in prompt
        assert "Zusammenfassung" in prompt or "Sachverhalt" in prompt

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
        assert "Buchstabier" in prompt or "buchstabier" in prompt

    def test_conflict_check_prompt(self):
        state = ConversationState(call_id="t")
        state.phase = CallPhase.CONFLICT_CHECK
        prompt = build_system_prompt(state, lang="de")
        assert "Unternehmen" in prompt or "Arbeitgeber" in prompt
        assert "Rechtsschutzversicherung" in prompt
        assert "record_conflict_info" in prompt

    def test_additional_info_prompt(self):
        state = ConversationState(call_id="t")
        state.phase = CallPhase.ADDITIONAL_INFO
        prompt = build_system_prompt(state, lang="de")
        assert "record_additional_info" in prompt

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
        assert "Teammitglied" in prompt or "weiterleite" in prompt

    def test_farewell_prompt(self):
        state = ConversationState(call_id="t")
        state.phase = CallPhase.FAREWELL
        prompt = build_system_prompt(state, lang="de")
        assert "Gute" in prompt or "Danke" in prompt or "bedanke" in prompt

    def test_preamble_forbids_legal_advice(self):
        state = ConversationState(call_id="t")
        prompt = build_system_prompt(state, lang="de")
        assert "NIEMALS" in prompt
        assert "Rechtsberatung" in prompt

    def test_preamble_forbids_false_booking(self):
        state = ConversationState(call_id="t")
        prompt = build_system_prompt(state, lang="de")
        assert "gebucht" in prompt.lower() or "bestätigt" in prompt.lower()
        assert "Terminwunsch" in prompt

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
        # digits should be comma-separated
        assert "," in result

    def test_phone_with_plus_prefix(self):
        result = _tts_preprocess("+49 176 12345678", lang="de")
        assert "plus" in result

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

    def test_english_email_uses_dot_not_punkt(self):
        result = _tts_preprocess("john@example.com", lang="en")
        assert "dot" in result
        assert "Punkt" not in result


# ---------------------------------------------------------------------------
# False booking guard — state-aware (Priority 2)
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
        san = self._make(["classify_legal_area"])
        assert san._tool_re is not None
        assert san._tool_re.sub("", "classify legal area") == ""

    def test_json_leak_stripped(self):
        from app.pipeline.processors import _JSON_LEAK_RE

        assert _JSON_LEAK_RE.search('{"legal_area": "employment"}')

    def test_normal_german_text_passes_through(self):
        san = self._make(["classify_legal_area", "escalate_to_human"])
        text = "Ich verstehe, es geht um eine Kündigung."
        assert san._strip_think(text) == text
        assert san._tool_re is not None
        assert san._tool_re.sub("", text) == text


# ---------------------------------------------------------------------------
# Booking validation — missing fields (Priority 3)
# ---------------------------------------------------------------------------


class TestBookingValidation:
    @pytest.fixture()
    def ctx_de(self):
        return ConversationManager(call_id="booking-test", lang="de")

    @pytest.mark.asyncio
    async def test_booking_blocked_when_fields_missing(self, registry, ctx_de):
        ctx = ctx_de
        ctx.add_user_message("Ich brauche einen Termin")
        ctx.set_intent(CallerIntent.BOOK_CONSULTATION)
        ctx.set_legal_area(LegalArea.EMPLOYMENT)
        ctx.state.intake_complete = True
        ctx.store_entity("name", "Max Müller", 0.95)
        ctx.confirm_entity("name")
        # email and phone are missing

        result = await registry.execute(
            "book_consultation",
            {"slot_id": 1, "caller_name": "Max Müller"},
            ctx,
        )
        assert result["status"] == "blocked"
        assert "email" in result["missing_fields"]
        assert "phone" in result["missing_fields"]

    @pytest.mark.asyncio
    async def test_booking_blocked_when_email_unconfirmed(self, registry, ctx_de):
        ctx = ctx_de
        ctx.add_user_message("Ich brauche einen Termin")
        ctx.set_intent(CallerIntent.BOOK_CONSULTATION)
        ctx.set_legal_area(LegalArea.EMPLOYMENT)
        ctx.state.intake_complete = True
        for f in ("name", "email", "phone"):
            ctx.store_entity(f, f"test_{f}", 0.95)
        ctx.confirm_entity("name")
        ctx.confirm_entity("phone")
        # email NOT confirmed

        result = await registry.execute(
            "book_consultation",
            {"slot_id": 1, "caller_name": "Test"},
            ctx,
        )
        assert result["status"] == "blocked"
        assert result["missing_fields"] == []
        assert "email" in result["unconfirmed_fields"]

    @pytest.mark.asyncio
    async def test_booking_proceeds_when_all_confirmed(self, registry, ctx_de):
        ctx = ctx_de
        ctx.add_user_message("Termin bitte")
        ctx.set_intent(CallerIntent.BOOK_CONSULTATION)
        ctx.set_legal_area(LegalArea.EMPLOYMENT)
        ctx.state.intake_complete = True
        for f in ("name", "email", "phone"):
            ctx.store_entity(f, f"test_{f}", 0.95)
            ctx.confirm_entity(f)
        ctx.state.employer_name = "TestCorp"
        ctx.state.has_legal_insurance = False
        ctx.state.additional_notes = ""
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
# Full German booking scenario — mirrors TestFullBookingScenario in English
# ---------------------------------------------------------------------------


class TestGermanFullBookingScenario:
    @pytest.fixture()
    def ctx_de(self):
        return ConversationManager(call_id="de-scenario", lang="de")

    @pytest.mark.asyncio
    async def test_german_happy_path(self, registry, ctx_de):
        ctx = ctx_de
        assert ctx.state.phase == CallPhase.GREETING

        # Caller describes issue
        ctx.add_user_message("Hallo, ich wurde letzte Woche gekündigt.")
        assert ctx.state.phase == CallPhase.INTENT_DETECTION

        # Intent detected
        await registry.execute(
            "classify_caller_intent",
            {"intent": "book_consultation"},
            ctx,
        )
        assert ctx.state.phase == CallPhase.ROUTING

        # Legal area classified
        await registry.execute(
            "classify_legal_area",
            {"legal_area": "employment"},
            ctx,
        )
        assert ctx.state.phase == CallPhase.INTAKE
        assert ctx.state.legal_area == LegalArea.EMPLOYMENT

        # Intake summary
        ctx.add_user_message("Fristlose Kündigung nach 5 Jahren ohne Abmahnung.")
        result = await registry.execute(
            "complete_intake",
            {"summary": "Fristlose Kündigung nach 5 Jahren, keine vorherige Abmahnung"},
            ctx,
        )
        assert result["status"] == "intake_complete"
        assert ctx.state.phase == CallPhase.CAPTURE

        # Name capture
        ctx.add_user_message("Mein Name ist Max Müller")
        await registry.execute(
            "extract_caller_details",
            {"name": "Max Müller"},
            ctx,
        )

        # Email capture
        ctx.add_user_message("max punkt mueller at gmail punkt com")
        await registry.execute(
            "extract_caller_details",
            {"email": "max.mueller@gmail.com"},
            ctx,
        )

        # Phone capture
        ctx.add_user_message("null eins sieben sechs eins zwei drei vier fünf sechs sieben acht")
        await registry.execute(
            "extract_caller_details",
            {"phone": "017612345678"},
            ctx,
        )

        # Confirm all entities
        for field in ("name", "email", "phone"):
            ctx.confirm_entity(field)

        assert ctx.state.phase == CallPhase.CONFLICT_CHECK

        # Conflict check
        ctx.add_user_message("Ich arbeite bei Siemens und habe Rechtsschutz")
        result = await registry.execute(
            "record_conflict_info",
            {"employer_name": "Siemens", "has_legal_insurance": True},
            ctx,
        )
        assert result["status"] == "recorded"
        assert ctx.state.phase == CallPhase.ADDITIONAL_INFO

        # Additional info
        ctx.add_user_message("Nichts weiter.")
        result = await registry.execute(
            "record_additional_info",
            {"notes": ""},
            ctx,
        )
        assert result["status"] == "recorded"
        assert ctx.state.phase == CallPhase.BOOKING

        # Check availability and book
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
    async def test_german_tenancy_flow_skips_conflict_check(self, registry, ctx_de):
        ctx = ctx_de
        ctx.add_user_message("Mein Vermieter will mich rauswerfen.")
        await registry.execute("classify_caller_intent", {"intent": "book_consultation"}, ctx)
        await registry.execute("classify_legal_area", {"legal_area": "tenancy"}, ctx)
        assert ctx.state.phase == CallPhase.INTAKE

        ctx.add_user_message("Kündigung wegen Eigenbedarf.")
        await registry.execute(
            "complete_intake",
            {"summary": "Kündigung wegen Eigenbedarf"},
            ctx,
        )
        assert ctx.state.phase == CallPhase.CAPTURE

        for field, value in [
            ("name", "Anna Schmidt"),
            ("email", "anna@example.com"),
            ("phone", "+4917612345678"),
        ]:
            await registry.execute("extract_caller_details", {field: value}, ctx)
            ctx.confirm_entity(field)

        # Tenancy skips conflict check, goes to additional_info
        assert ctx.state.phase == CallPhase.ADDITIONAL_INFO

    @pytest.mark.asyncio
    async def test_german_escalation_mid_flow(self, registry, ctx_de):
        ctx = ctx_de
        ctx.add_user_message("Ich möchte mit jemandem sprechen.")

        result = await registry.execute(
            "escalate_to_human",
            {"reason": "caller_requested_human", "summary": "Möchte einen Anwalt sprechen"},
            ctx,
        )
        assert result["status"] == "escalating"
        assert ctx.state.phase == CallPhase.ESCALATION


# ---------------------------------------------------------------------------
# German email low-confidence scenario — the uncertainty path
# ---------------------------------------------------------------------------


class TestGermanEmailUncertaintyPath:
    @pytest.mark.asyncio
    async def test_low_confidence_email_needs_confirmation(self, registry):
        word_infos = [
            _make_word_info("ameliesommer", 0.0, 1.0, 0.4),
            _make_word_info("gmail", 1.0, 1.5, 0.6),
            _make_word_info("com", 1.5, 2.0, 0.8),
        ]
        ctx = ConversationManager(call_id="de-email-test", lang="de")
        ctx.add_user_message("ameliesommer gmail com", word_infos)
        ctx.set_intent(CallerIntent.BOOK_CONSULTATION)
        ctx.set_legal_area(LegalArea.EMPLOYMENT)
        ctx.state.intake_complete = True

        result = await registry.execute(
            "extract_caller_details",
            {"email": "ameliesommer@gmail.com"},
            ctx,
        )
        assert not result["all_confirmed"]
        assert any(c["field"] == "email" for c in result["needs_confirmation"])
        assert not ctx.state.entities["email"].confirmed

    @pytest.mark.asyncio
    async def test_high_confidence_email_auto_confirms(self, registry):
        word_infos = [
            _make_word_info("max@example.com", 0.0, 1.2, 0.95),
        ]
        ctx = ConversationManager(call_id="de-email-hi", lang="de")
        ctx.add_user_message("max@example.com", word_infos)
        ctx.set_intent(CallerIntent.BOOK_CONSULTATION)
        ctx.set_legal_area(LegalArea.EMPLOYMENT)
        ctx.state.intake_complete = True

        result = await registry.execute(
            "extract_caller_details",
            {"email": "max@example.com"},
            ctx,
        )
        assert result["all_confirmed"]
        assert ctx.state.entities["email"].confirmed


def _make_word_info(word, start, end, confidence):
    from app.models.schemas import WordInfo

    return WordInfo(word=word, start_time=start, end_time=end, confidence=confidence)
