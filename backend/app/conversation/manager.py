from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

from app.conversation.flow import PHASE_TOOLS, next_phase
from app.conversation.policy import is_explicit_handoff_request
from app.conversation.prompts import build_system_prompt, get_system_prompt_base
from app.conversation.state import ConversationState
from app.models.schemas import CallerIntent, CallPhase, ExtractedEntity, LegalArea

logger = logging.getLogger(__name__)


class ConversationManager:
    def __init__(self, call_id: str, lang: str = "de"):
        self.lang = lang
        self.state = ConversationState(call_id=call_id)
        self.state.messages = [{"role": "system", "content": get_system_prompt_base(lang)}]
        self._llm_context = None
        self._tools_builder: Callable[[list[str]], Any] | None = None

    def set_llm_context(self, context) -> None:
        self._llm_context = context

    def set_tools_builder(self, builder: Callable[[list[str]], Any]) -> None:
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
        prompt = build_system_prompt(self.state, self.lang)
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

    def add_user_message(self, text: str) -> None:
        self.state.turn_count += 1
        self.state.messages.append({"role": "user", "content": text})
        if is_explicit_handoff_request(text, self.lang):
            self.request_handoff("caller_requested_human", text)
        else:
            self.advance_phase()

    def set_transcription_confidence(self, confidence: float | None) -> None:
        self.state.last_transcription_confidence = confidence

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

    def set_route(self, intent: CallerIntent, area: LegalArea, summary: str | None = None) -> None:
        self.state.caller_intent = intent
        self.state.legal_area = area
        if summary:
            self.state.matter_summary = summary
        self.reset_misunderstanding_streak()
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

    def update_and_confirm_entity(self, field_name: str, value: str) -> None:
        entity = self.state.entities.get(field_name)
        if entity:
            entity.value = value
            entity.confirmed = True
            entity.source_turn = self.state.turn_count
        else:
            entity = ExtractedEntity(
                field_name=field_name,
                value=value,
                confidence=1.0,
                confirmed=True,
                source_turn=self.state.turn_count,
            )
            self.state.entities[field_name] = entity
        self.advance_phase()

    def record_misunderstanding(self) -> None:
        self.state.misunderstanding_streak += 1
        logger.info("Misunderstanding streak: %d", self.state.misunderstanding_streak)
        self.advance_phase()

    def request_handoff(self, reason: str, summary: str = "") -> None:
        self.state.escalation_requested = True
        self.state.escalation_reason = reason
        self.state.escalation_summary = summary
        self.advance_phase()

    def reset_misunderstanding_streak(self) -> None:
        if self.state.misunderstanding_streak > 0:
            self.state.misunderstanding_streak = 0
