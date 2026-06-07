"""Tests for conversation state and manager."""

import json

from app.conversation.prompts import (
    FRAGMENTS,
    SYSTEM_PROMPT_BASE,
    SYSTEM_PROMPT_LOCAL,
    SYSTEM_PROMPT_TOOLS,
    build_system_prompt,
)
from app.conversation.state import ConversationState
from app.models.schemas import CallerIntent, CallPhase, ExtractedEntity, LegalArea


class TestConversationState:
    def test_initial_state(self, call_id):
        state = ConversationState(call_id=call_id)
        assert state.call_id == call_id
        assert state.phase == CallPhase.GREETING
        assert state.caller_intent == CallerIntent.UNKNOWN
        assert state.legal_area == LegalArea.UNKNOWN
        assert state.entities == {}
        assert state.turn_count == 0
        assert not state.escalation_requested
        assert not state.booking_confirmed
        assert state.offered_slot_ids == []
        assert state.matter_summary is None

    def test_to_dict_roundtrip(self, call_id):
        state = ConversationState(call_id=call_id)
        state.phase = CallPhase.ROUTING
        state.caller_intent = CallerIntent.BOOK_CONSULTATION
        state.legal_area = LegalArea.EMPLOYMENT
        state.entities["name"] = ExtractedEntity(
            field_name="name", value="John", confidence=0.9, confirmed=True
        )

        d = state.to_dict()
        assert d["call_id"] == call_id
        assert d["phase"] == "routing"
        assert d["caller_intent"] == "book_consultation"
        assert d["legal_area"] == "employment"
        assert d["entities"]["name"]["value"] == "John"
        assert d["entities"]["name"]["confirmed"] is True

    def test_to_json_is_valid(self, call_id):
        state = ConversationState(call_id=call_id)
        j = state.to_json()
        parsed = json.loads(j)
        assert parsed["call_id"] == call_id


class TestConversationManager:
    def test_init_has_system_message(self, conversation):
        messages = conversation.get_messages()
        assert len(messages) == 1
        assert messages[0]["role"] == "system"
        assert "receptionist" in messages[0]["content"].lower()

    def test_add_user_message_increments_turn(self, conversation):
        conversation.add_user_message("Hello")
        assert conversation.state.turn_count == 1
        conversation.add_user_message("I need help")
        assert conversation.state.turn_count == 2

    def test_add_user_message_advances_phase(self, conversation):
        assert conversation.state.phase == CallPhase.GREETING
        conversation.add_user_message("Hello")
        assert conversation.state.phase == CallPhase.ROUTING

    def test_add_assistant_message(self, conversation):
        conversation.add_assistant_message("How can I help?")
        messages = conversation.get_messages()
        assert messages[-1]["role"] == "assistant"
        assert messages[-1]["content"] == "How can I help?"

    def test_add_tool_call(self, conversation):
        tool_calls = [{"id": "tc_1", "function": {"name": "test", "arguments": "{}"}}]
        conversation.add_tool_call("Let me check", tool_calls)
        msg = conversation.get_messages()[-1]
        assert msg["role"] == "assistant"
        assert msg["tool_calls"] == tool_calls

    def test_add_tool_result(self, conversation):
        conversation.add_tool_result("tc_1", "test_tool", {"status": "ok"})
        msg = conversation.get_messages()[-1]
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "tc_1"
        assert msg["name"] == "test_tool"
        assert json.loads(msg["content"]) == {"status": "ok"}

    def test_set_route_advances_to_qualification(self, conversation):
        conversation.add_user_message("I was fired unfairly")
        conversation.set_route(
            CallerIntent.BOOK_CONSULTATION, LegalArea.EMPLOYMENT, "unfair dismissal"
        )
        assert conversation.state.caller_intent == CallerIntent.BOOK_CONSULTATION
        assert conversation.state.legal_area == LegalArea.EMPLOYMENT
        assert conversation.state.matter_summary == "unfair dismissal"
        assert conversation.state.phase == CallPhase.QUALIFICATION

    def test_set_route_general_info_goes_to_information(self, conversation):
        conversation.add_user_message("I have a question")
        conversation.set_route(CallerIntent.GENERAL_INFO, LegalArea.EMPLOYMENT)
        assert conversation.state.phase == CallPhase.INFORMATION

    def test_store_entity(self, conversation):
        conversation.add_user_message("My name is John")
        entity = conversation.store_entity("name", "John", 0.9)
        assert entity.field_name == "name"
        assert entity.value == "John"
        assert entity.confidence == 0.9
        assert not entity.confirmed

    def test_confirm_entity(self, conversation):
        conversation.add_user_message("John")
        conversation.store_entity("name", "John", 0.9)
        conversation.confirm_entity("name")
        assert conversation.state.entities["name"].confirmed is True

    def test_confirm_nonexistent_entity_is_noop(self, conversation):
        conversation.confirm_entity("nonexistent")

    def test_update_and_confirm_entity(self, conversation):
        conversation.add_user_message("test@example.com")
        conversation.store_entity("email", "tset@example.com", 0.8)
        conversation.update_and_confirm_entity("email", "test@example.com")
        assert conversation.state.entities["email"].value == "test@example.com"
        assert conversation.state.entities["email"].confirmed is True

    def test_update_and_confirm_entity_creates_if_missing(self, conversation):
        conversation.update_and_confirm_entity("phone", "+1234567890")
        assert conversation.state.entities["phone"].value == "+1234567890"
        assert conversation.state.entities["phone"].confirmed is True

    def test_available_tools_change_with_phase(self, conversation):
        assert conversation.get_available_tools() == []
        conversation.add_user_message("Hello")
        assert "route_call" in conversation.get_available_tools()

    def test_record_misunderstanding_increments(self, conversation):
        conversation.record_misunderstanding()
        assert conversation.state.misunderstanding_streak == 1
        conversation.record_misunderstanding()
        assert conversation.state.misunderstanding_streak == 2

    def test_reset_misunderstanding_streak(self, conversation):
        conversation.record_misunderstanding()
        conversation.record_misunderstanding()
        conversation.reset_misunderstanding_streak()
        assert conversation.state.misunderstanding_streak == 0


class TestPrompts:
    def test_tools_prompt_is_natural(self):
        assert "receptionist" in SYSTEM_PROMPT_TOOLS
        assert "NEVER" in SYSTEM_PROMPT_TOOLS

    def test_local_prompt_has_no_tool_references(self):
        assert "capture_caller_details" not in SYSTEM_PROMPT_LOCAL
        assert "route_call" not in SYSTEM_PROMPT_LOCAL

    def test_fragments_exist_for_supported_areas(self):
        assert "employment" in FRAGMENTS
        assert "tenancy" in FRAGMENTS
        assert "traffic" in FRAGMENTS

    def test_base_prompt_is_tools_prompt(self):
        assert SYSTEM_PROMPT_BASE is SYSTEM_PROMPT_TOOLS

    def test_build_system_prompt_greeting(self):
        state = ConversationState(call_id="test")
        prompt = build_system_prompt(state, lang="en")
        assert "Thank you for calling" in prompt
        assert "MISSING FIELDS" not in prompt

    def test_build_system_prompt_greeting_de(self):
        state = ConversationState(call_id="test")
        prompt = build_system_prompt(state, lang="de")
        assert "Claudia" in prompt
        assert "FEHLENDE FELDER" not in prompt

    def test_build_system_prompt_capture_shows_missing(self):
        state = ConversationState(call_id="test")
        state.phase = CallPhase.CAPTURE
        state.caller_intent = CallerIntent.BOOK_CONSULTATION
        state.legal_area = LegalArea.EMPLOYMENT
        prompt = build_system_prompt(state, lang="en")
        assert "MISSING FIELDS" in prompt
        assert "name" in prompt
        assert "email" in prompt
        assert "phone" in prompt

    def test_build_system_prompt_capture_shows_missing_de(self):
        state = ConversationState(call_id="test")
        state.phase = CallPhase.CAPTURE
        state.caller_intent = CallerIntent.BOOK_CONSULTATION
        state.legal_area = LegalArea.EMPLOYMENT
        prompt = build_system_prompt(state, lang="de")
        assert "FEHLENDE FELDER" in prompt

    def test_build_system_prompt_includes_legal_fragment(self):
        state = ConversationState(call_id="test")
        state.phase = CallPhase.INFORMATION
        state.legal_area = LegalArea.TENANCY
        prompt = build_system_prompt(state, lang="en")
        assert "TENANCY LAW" in prompt

    def test_build_system_prompt_includes_legal_fragment_de(self):
        state = ConversationState(call_id="test")
        state.phase = CallPhase.INFORMATION
        state.legal_area = LegalArea.TENANCY
        prompt = build_system_prompt(state, lang="de")
        assert "MIETRECHT" in prompt
