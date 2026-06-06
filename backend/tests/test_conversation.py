"""Tests for conversation state and manager."""

import json

import pytest

from app.conversation.manager import ConversationManager
from app.conversation.prompts import (
    FRAGMENTS,
    SYSTEM_PROMPT_BASE,
    SYSTEM_PROMPT_LOCAL,
    SYSTEM_PROMPT_TOOLS,
)
from app.conversation.state import ConversationState
from app.models.schemas import CallerIntent, CallPhase, ExtractedEntity, LegalArea, WordInfo


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
        assert messages[0]["content"] == SYSTEM_PROMPT_BASE

    def test_add_user_message_increments_turn(self, conversation):
        conversation.add_user_message("Hello")
        assert conversation.state.turn_count == 1
        conversation.add_user_message("I need help")
        assert conversation.state.turn_count == 2

    def test_add_user_message_stores_word_infos(self, conversation):
        words = [WordInfo(word="hello", start_time=0, end_time=0.5, confidence=0.95)]
        conversation.add_user_message("hello", words)
        assert conversation._word_infos_by_turn[1] == words

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

    def test_set_intent(self, conversation):
        conversation.set_intent(CallerIntent.BOOK_CONSULTATION)
        assert conversation.state.caller_intent == CallerIntent.BOOK_CONSULTATION
        assert conversation.state.phase == CallPhase.INTENT_DETECTION

    def test_set_legal_area_employment(self, conversation):
        conversation.set_legal_area(LegalArea.EMPLOYMENT)
        assert conversation.state.legal_area == LegalArea.EMPLOYMENT
        assert conversation.state.phase == CallPhase.ROUTING
        system_msg = conversation.get_messages()[0]
        assert "employment" in system_msg["content"].lower()
        assert FRAGMENTS["employment"] in system_msg["content"]

    def test_set_legal_area_unknown_no_fragment(self, conversation):
        conversation.set_legal_area(LegalArea.UNKNOWN)
        system_msg = conversation.get_messages()[0]
        assert system_msg["content"] == SYSTEM_PROMPT_BASE

    def test_store_entity(self, conversation):
        conversation.add_user_message("My name is John")
        entity = conversation.store_entity("name", "John", 0.9)
        assert entity.field_name == "name"
        assert entity.value == "John"
        assert entity.confidence == 0.9
        assert not entity.confirmed
        assert conversation.state.phase == CallPhase.CAPTURE

    def test_confirm_entity(self, conversation):
        conversation.add_user_message("John")
        conversation.store_entity("name", "John", 0.9)
        conversation.confirm_entity("name")
        assert conversation.state.entities["name"].confirmed is True

    def test_confirm_nonexistent_entity_is_noop(self, conversation):
        conversation.confirm_entity("nonexistent")

    def test_word_confidence_matches(self, conversation_with_high_confidence):
        ctx = conversation_with_high_confidence
        conf = ctx.get_word_confidence_for_value("John Smith")
        assert conf == 0.92  # min of 0.95 and 0.92

    def test_word_confidence_low(self, conversation_with_low_confidence):
        ctx = conversation_with_low_confidence
        conf = ctx.get_word_confidence_for_value("Siobhan Murphy")
        assert conf == 0.4  # min of 0.4 and 0.9

    def test_word_confidence_no_match_returns_default(self, conversation):
        conversation.add_user_message("hello there")
        conf = conversation.get_word_confidence_for_value("John")
        assert conf == 0.5

    def test_word_confidence_empty_value(self, conversation):
        conf = conversation.get_word_confidence_for_value("")
        assert conf == 0.5

    def test_word_confidence_no_word_infos(self, conversation):
        conversation.add_user_message("John Smith")
        conf = conversation.get_word_confidence_for_value("John")
        assert conf == 0.5


class TestPrompts:
    def test_tools_prompt_references_tools(self):
        assert "extract_caller_details" in SYSTEM_PROMPT_TOOLS

    def test_local_prompt_has_no_tool_references(self):
        assert "extract_caller_details" not in SYSTEM_PROMPT_LOCAL
        assert "classify_caller_intent" not in SYSTEM_PROMPT_LOCAL

    def test_fragments_exist_for_supported_areas(self):
        assert "employment" in FRAGMENTS
        assert "tenancy" in FRAGMENTS

    def test_base_prompt_is_tools_prompt(self):
        assert SYSTEM_PROMPT_BASE is SYSTEM_PROMPT_TOOLS
