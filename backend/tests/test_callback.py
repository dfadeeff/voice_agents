"""Regression tests for the callback/contact-capture flow.

Tests the fix for: agent prematurely closing after getting name but before
collecting phone number in named-person callback requests.
"""

import pytest
from app.conversation.flow import (
    CALLBACK_REQUIRED_FIELDS,
    callback_contacts_confirmed,
    next_phase,
)
from app.conversation.manager import ConversationManager
from app.conversation.phone import normalize_phone_text
from app.conversation.policy import extract_target_person
from app.conversation.prompts import build_system_prompt
from app.conversation.state import ConversationState
from app.models.schemas import CallPhase, ExtractedEntity


def _confirmed(field, value="test"):
    return ExtractedEntity(field_name=field, value=value, confidence=0.95, confirmed=True)


def _unconfirmed(field, value="test"):
    return ExtractedEntity(field_name=field, value=value, confidence=0.5, confirmed=False)


def _state(**kwargs):
    return ConversationState(call_id="test", **kwargs)


# ---------------------------------------------------------------------------
# Callback state machine
# ---------------------------------------------------------------------------


class TestCallbackFlow:
    def test_callback_required_fields_are_name_and_phone(self):
        assert CALLBACK_REQUIRED_FIELDS == ("name", "phone")

    def test_person_request_enters_capture_not_escalation(self):
        ctx = ConversationManager(call_id="cb-test", lang="de")
        ctx.add_user_message("Ich möchte bitte Herrn Schmid sprechen.")
        assert ctx.state.phase == CallPhase.CAPTURE
        assert ctx.state.callback_requested is True
        assert ctx.state.target_person == "Herr Schmid"
        assert ctx.state.escalation_requested is False

    def test_callback_requires_phone_after_name(self):
        """The exact regression: after name, must ask for phone, not close."""
        ctx = ConversationManager(call_id="cb-test", lang="de")
        ctx.add_user_message("Ich möchte bitte Herrn Schmid sprechen.")
        assert ctx.state.phase == CallPhase.CAPTURE

        ctx.store_entity("name", "Max Mustermann", 0.95)
        ctx.confirm_entity("name")

        assert ctx.state.phase == CallPhase.CAPTURE
        assert not ctx.state.entities.get("phone")

    def test_callback_confirms_after_name_phone_and_time(self):
        ctx = ConversationManager(call_id="cb-test", lang="de")
        ctx.add_user_message("Ich möchte bitte Herrn Schmid sprechen.")

        ctx.store_entity("name", "Max Mustermann", 0.95)
        ctx.confirm_entity("name")

        ctx.store_entity("phone", "+49176123456", 0.9)
        ctx.confirm_entity("phone")
        # Name + phone alone now wait for a preferred callback time.
        assert ctx.state.phase == CallPhase.CAPTURE

        ctx.state.preferred_time = "morgen Vormittag"
        ctx.advance_phase()
        assert ctx.state.phase == CallPhase.CONFIRMATION

    def test_callback_does_not_require_email(self):
        s = _state(
            callback_requested=True,
            target_person="Schmid",
            turn_count=3,
            preferred_time="morgen Vormittag",
            entities={
                "name": _confirmed("name", "Max Mustermann"),
                "phone": _confirmed("phone", "+49176123456"),
            },
        )
        assert next_phase(s) == CallPhase.CONFIRMATION

    def test_callback_stays_in_capture_without_phone(self):
        s = _state(
            callback_requested=True,
            target_person="Schmid",
            turn_count=2,
            entities={
                "name": _confirmed("name", "Max Mustermann"),
            },
        )
        assert next_phase(s) == CallPhase.CAPTURE

    def test_callback_stays_in_capture_with_unconfirmed_phone(self):
        s = _state(
            callback_requested=True,
            target_person="Schmid",
            turn_count=3,
            entities={
                "name": _confirmed("name", "Max Mustermann"),
                "phone": _unconfirmed("phone", "+49176123456"),
            },
        )
        assert next_phase(s) == CallPhase.CAPTURE

    def test_misunderstanding_escalation_overrides_callback(self):
        s = _state(
            callback_requested=True,
            target_person="Schmid",
            turn_count=5,
            misunderstanding_streak=3,
        )
        assert next_phase(s) == CallPhase.ESCALATION


class TestCallbackContactsConfirmed:
    def test_empty_entities(self):
        assert callback_contacts_confirmed({}) is False

    def test_name_only(self):
        assert callback_contacts_confirmed({"name": _confirmed("name")}) is False

    def test_phone_only(self):
        assert callback_contacts_confirmed({"phone": _confirmed("phone")}) is False

    def test_name_and_phone_confirmed(self):
        entities = {
            "name": _confirmed("name"),
            "phone": _confirmed("phone"),
        }
        assert callback_contacts_confirmed(entities) is True

    def test_unconfirmed_phone_fails(self):
        entities = {
            "name": _confirmed("name"),
            "phone": _unconfirmed("phone"),
        }
        assert callback_contacts_confirmed(entities) is False


# ---------------------------------------------------------------------------
# Full callback scenario with tools
# ---------------------------------------------------------------------------


class TestCallbackScenarioWithTools:
    @pytest.mark.asyncio
    async def test_named_person_callback_happy_path(self, registry):
        ctx = ConversationManager(call_id="cb-happy", lang="de")

        ctx.add_user_message("Ich möchte bitte Herrn Schmid sprechen.")
        assert ctx.state.phase == CallPhase.CAPTURE
        assert ctx.state.callback_requested is True
        assert ctx.state.target_person == "Herr Schmid"

        ctx.add_user_message("Ja, Max Mustermann")
        await registry.execute("capture_caller_details", {"name": "Max Mustermann"}, ctx)
        assert ctx.state.phase == CallPhase.CAPTURE
        assert "name" in ctx.state.entities

        ctx.add_user_message(
            "plus vier neun eins sieben sechs fünf sechs acht drei vier zwei null fünf sieben"
        )
        await registry.execute("capture_caller_details", {"phone": "+4917656834205"}, ctx)
        assert "phone" in ctx.state.entities

        ctx.add_user_message("Ja, das stimmt")
        await registry.execute(
            "confirm_caller_detail",
            {"field": "phone", "confirmed_value": "+4917656834205", "status": "accepted"},
            ctx,
        )
        # Name + phone done → scheduling step asks for a callback time.
        assert ctx.state.phase == CallPhase.CAPTURE

        line = ctx.next_prompt()  # scripted callback-time ask → awaiting == "callback_time"
        assert line is not None and ctx.state.awaiting == "callback_time"
        ctx.add_user_message("Morgen um 10 Uhr")
        assert ctx.state.preferred_time == "Morgen um 10 Uhr"
        assert ctx.state.phase == CallPhase.CONFIRMATION

    @pytest.mark.asyncio
    async def test_no_premature_close_after_name_only(self, registry):
        """Regression: system must not say 'weiterleiten' before phone is captured."""
        ctx = ConversationManager(call_id="cb-no-close", lang="de")

        ctx.add_user_message("Ich möchte bitte Herrn Schmid sprechen.")
        ctx.add_user_message("Max Mustermann")
        await registry.execute("capture_caller_details", {"name": "Max Mustermann"}, ctx)

        assert ctx.state.phase == CallPhase.CAPTURE
        assert ctx.state.entities.get("phone") is None

        prompt = build_system_prompt(ctx.state, lang="de")
        assert "phone" in prompt.lower() or "telefon" in prompt.lower()

    @pytest.mark.asyncio
    async def test_generic_person_request_callback(self, registry):
        """'Mit einem Anwalt sprechen' also enters callback flow."""
        ctx = ConversationManager(call_id="cb-generic", lang="de")
        ctx.add_user_message("Ich möchte mit einem Anwalt sprechen.")

        assert ctx.state.callback_requested is True
        assert ctx.state.target_person is None
        assert ctx.state.phase == CallPhase.CAPTURE

    @pytest.mark.asyncio
    async def test_handoff_tool_still_works_during_callback(self, registry):
        """LLM can still explicitly escalate during callback capture."""
        ctx = ConversationManager(call_id="cb-escalate", lang="de")
        ctx.add_user_message("Ich möchte bitte Herrn Schmid sprechen.")
        assert ctx.state.phase == CallPhase.CAPTURE

        result = await registry.execute(
            "request_handoff",
            {"reason": "caller_frustrated", "summary": "Caller is very upset"},
            ctx,
        )
        assert result["status"] == "handoff_requested"
        assert ctx.state.phase == CallPhase.ESCALATION


# ---------------------------------------------------------------------------
# Target person extraction
# ---------------------------------------------------------------------------


class TestExtractTargetPerson:
    def test_herr_schmid(self):
        assert extract_target_person("Ich möchte Herrn Schmid sprechen", "de") == "Herr Schmid"

    def test_frau_landau(self):
        result = extract_target_person("Ich möchte bitte Frau Landau sprechen.", "de")
        assert result == "Frau Landau"

    def test_no_person(self):
        assert extract_target_person("Ich möchte einen Termin", "de") is None

    def test_english_mr(self):
        assert extract_target_person("I'd like to speak to Mr Smith", "en") == "Mr Smith"

    def test_english_ms(self):
        assert extract_target_person("Can I talk to Ms Johnson please?", "en") == "Ms Johnson"

    def test_generic_request_no_name(self):
        assert extract_target_person("Ich möchte mit einem Anwalt sprechen", "de") is None


# ---------------------------------------------------------------------------
# Callback prompts
# ---------------------------------------------------------------------------


class TestCallbackPrompts:
    def test_callback_capture_prompt_mentions_target_person(self):
        state = _state(
            callback_requested=True,
            target_person="Schmid",
            phase=CallPhase.CAPTURE,
            turn_count=1,
        )
        prompt = build_system_prompt(state, lang="de")
        assert "Schmid" in prompt
        assert "Telefonnummer" in prompt
        assert "email" not in prompt.lower() or "keine e-mail" in prompt.lower()

    def test_callback_capture_prompt_no_email_in_missing_fields(self):
        state = _state(
            callback_requested=True,
            target_person="Schmid",
            phase=CallPhase.CAPTURE,
            turn_count=2,
        )
        prompt = build_system_prompt(state, lang="de")
        fields_section = prompt.split("FEHLENDE FELDER")[-1] if "FEHLENDE FELDER" in prompt else ""
        assert "email" not in fields_section

    def test_callback_confirmation_prompt(self):
        state = _state(
            callback_requested=True,
            target_person="Schmid",
            phase=CallPhase.CONFIRMATION,
            turn_count=4,
            entities={
                "name": _confirmed("name", "Max Mustermann"),
                "phone": _confirmed("phone", "+49176123456"),
            },
        )
        prompt = build_system_prompt(state, lang="de")
        assert "Schmid" in prompt
        assert "Rückrufwunsch" in prompt

    def test_callback_prompt_uses_team_when_no_target(self):
        state = _state(
            callback_requested=True,
            target_person=None,
            phase=CallPhase.CAPTURE,
            turn_count=1,
        )
        prompt = build_system_prompt(state, lang="de")
        assert "Kanzleiteam" in prompt

    def test_normal_capture_still_requires_email(self):
        """Non-callback flow still requires email."""
        state = _state(
            phase=CallPhase.CAPTURE,
            turn_count=3,
        )
        prompt = build_system_prompt(state, lang="de")
        assert "FEHLENDE FELDER" in prompt
        missing = prompt.split("FEHLENDE FELDER:")[-1].split("\n")[0]
        assert "email" in missing


# ---------------------------------------------------------------------------
# Phone normalization
# ---------------------------------------------------------------------------


class TestPhoneNormalization:
    def test_german_digit_words(self):
        assert normalize_phone_text("vier neun eins sieben sechs") == "49176"

    def test_plus_prefix(self):
        assert normalize_phone_text("Plus vier neun eins sieben sechs") == "+49176"

    def test_full_german_phone(self):
        result = normalize_phone_text(
            "plus vier neun eins sieben sechs fünf sechs acht drei vier zwei null fünf sieben"
        )
        assert result == "+49176568342057"

    def test_digits_pass_through(self):
        assert normalize_phone_text("+49 176 12345678") == "+4917612345678"

    def test_commas_handled(self):
        assert normalize_phone_text("4,9,5,6,8") == "49568"

    def test_zero_prefix_becomes_plus49(self):
        assert normalize_phone_text("017612345678") == "+4917612345678"

    def test_49_prefix_gets_plus(self):
        assert normalize_phone_text("4917612345678") == "+4917612345678"

    def test_mixed_words_and_digits(self):
        result = normalize_phone_text("plus 49 eins sieben sechs 12345")
        assert result == "+4917612345"

    def test_empty_input(self):
        assert normalize_phone_text("") == ""

    def test_no_digits(self):
        assert normalize_phone_text("hallo wie geht es") == ""

    def test_zwo_variant(self):
        assert normalize_phone_text("zwo drei vier") == "234"

    def test_fuenf_variant(self):
        assert normalize_phone_text("fuenf sechs sieben") == "567"
