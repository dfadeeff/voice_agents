from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

from app.conversation.flow import PHASE_TOOLS, next_phase
from app.conversation.prompts import SYSTEM_PROMPT_BASE, build_system_prompt
from app.conversation.state import ConversationState
from app.models.schemas import (
    CallerIntent,
    CallPhase,
    ExtractedEntity,
    LegalArea,
    WordInfo,
)

logger = logging.getLogger(__name__)


class ConversationManager:
    def __init__(self, call_id: str):
        self.state = ConversationState(call_id=call_id)
        self.state.messages = [{"role": "system", "content": SYSTEM_PROMPT_BASE}]
        self._word_infos_by_turn: dict[int, list[WordInfo]] = {}
        self._llm_context = None
        self._tools_builder: Callable[[list[str]], Any] | None = None

    def set_llm_context(self, context) -> None:
        self._llm_context = context

    def set_tools_builder(self, builder: Callable[[list[str]], Any]) -> None:
        """Set callback that builds a ToolsSchema from a list of tool names."""
        self._tools_builder = builder

    def advance_phase(self) -> CallPhase:
        new_phase = next_phase(self.state)
        if new_phase != self.state.phase:
            old_phase = self.state.phase
            self.state.phase = new_phase
            self._update_llm_context()
            logger.info("Phase advanced: %s → %s", old_phase.value, new_phase.value)
        return self.state.phase

    def _update_llm_context(self) -> None:
        prompt = build_system_prompt(self.state)
        if self.state.messages:
            self.state.messages[0]["content"] = prompt
        if self._llm_context and hasattr(self._llm_context, "messages"):
            ctx_messages = self._llm_context.messages
            if ctx_messages:
                ctx_messages[0]["content"] = prompt
            allowed = self.get_available_tools()
            if self._tools_builder and hasattr(self._llm_context, "set_tools"):
                self._llm_context.set_tools(self._tools_builder(allowed))

    def get_available_tools(self) -> list[str]:
        return PHASE_TOOLS.get(self.state.phase, [])

    def add_user_message(self, text: str, word_infos: list[WordInfo] | None = None) -> None:
        self.state.turn_count += 1
        self.state.messages.append({"role": "user", "content": text})
        if word_infos:
            self._word_infos_by_turn[self.state.turn_count] = word_infos
        self.advance_phase()

    def add_assistant_message(self, text: str) -> None:
        self.state.messages.append({"role": "assistant", "content": text})

    def add_tool_call(self, assistant_text: str, tool_calls: list[dict]) -> None:
        self.state.messages.append(
            {
                "role": "assistant",
                "content": assistant_text or None,
                "tool_calls": tool_calls,
            }
        )

    def add_tool_result(self, tool_call_id: str, name: str, result: dict) -> None:
        self.state.messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": name,
                "content": json.dumps(result),
            }
        )

    def get_messages(self) -> list[dict]:
        return self.state.messages

    def set_intent(self, intent: CallerIntent) -> None:
        self.state.caller_intent = intent
        self.advance_phase()

    def set_legal_area(self, area: LegalArea) -> None:
        self.state.legal_area = area
        self.advance_phase()

    def store_entity(self, field_name: str, value: str, confidence: float) -> ExtractedEntity:
        entity = ExtractedEntity(
            field_name=field_name,
            value=value,
            confidence=confidence,
            source_turn=self.state.turn_count,
        )
        self.state.entities[field_name] = entity
        self.advance_phase()
        return entity

    def confirm_entity(self, field_name: str) -> None:
        if field_name in self.state.entities:
            self.state.entities[field_name].confirmed = True
            self.advance_phase()

    def get_word_confidence_for_value(self, value: str) -> float:
        value_words = value.lower().split()
        if not value_words:
            return 0.5

        current_turn = self.state.turn_count
        word_infos = self._word_infos_by_turn.get(current_turn, [])
        if not word_infos:
            return 0.5

        matched_confidences = []
        for vw in value_words:
            best_match = None
            for wi in word_infos:
                if wi.word.lower().strip(".,!?") == vw:
                    best_match = wi.confidence
                    break
            if best_match is not None:
                matched_confidences.append(best_match)

        if not matched_confidences:
            return 0.5

        return min(matched_confidences)
