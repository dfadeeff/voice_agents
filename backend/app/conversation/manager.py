import json

from app.conversation.prompts import FRAGMENTS, SYSTEM_PROMPT_BASE
from app.conversation.state import ConversationState
from app.models.schemas import (
    CallerIntent,
    CallPhase,
    ExtractedEntity,
    LegalArea,
    WordInfo,
)


class ConversationManager:
    def __init__(self, call_id: str):
        self.state = ConversationState(call_id=call_id)
        self.state.messages = [
            {"role": "system", "content": SYSTEM_PROMPT_BASE}
        ]
        self._word_infos_by_turn: dict[int, list[WordInfo]] = {}

    def add_user_message(
        self, text: str, word_infos: list[WordInfo] | None = None
    ) -> None:
        self.state.turn_count += 1
        self.state.messages.append({"role": "user", "content": text})
        if word_infos:
            self._word_infos_by_turn[self.state.turn_count] = word_infos

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

    def add_tool_result(
        self, tool_call_id: str, name: str, result: dict
    ) -> None:
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
        self.state.phase = CallPhase.INTENT_DETECTION

    def set_legal_area(self, area: LegalArea) -> None:
        self.state.legal_area = area
        self.state.phase = CallPhase.ROUTING
        fragment = FRAGMENTS.get(area.value, "")
        if fragment:
            system_msg = self.state.messages[0]
            system_msg["content"] = SYSTEM_PROMPT_BASE + "\n" + fragment

    def store_entity(
        self, field_name: str, value: str, confidence: float
    ) -> ExtractedEntity:
        entity = ExtractedEntity(
            field_name=field_name,
            value=value,
            confidence=confidence,
            source_turn=self.state.turn_count,
        )
        self.state.entities[field_name] = entity
        self.state.phase = CallPhase.CAPTURE
        return entity

    def confirm_entity(self, field_name: str) -> None:
        if field_name in self.state.entities:
            self.state.entities[field_name].confirmed = True

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