import json
import time
from dataclasses import dataclass, field

from app.models.schemas import CallerIntent, CallPhase, ExtractedEntity, LegalArea


@dataclass
class ConversationState:
    call_id: str
    phase: CallPhase = CallPhase.GREETING
    caller_intent: CallerIntent = CallerIntent.UNKNOWN
    legal_area: LegalArea = LegalArea.UNKNOWN
    matter_summary: str | None = None
    entities: dict[str, ExtractedEntity] = field(default_factory=dict)
    messages: list[dict] = field(default_factory=list)
    turn_count: int = 0
    callback_requested: bool = False
    target_person: str | None = None
    escalation_requested: bool = False
    escalation_reason: str | None = None
    escalation_summary: str | None = None
    booking_confirmed: bool = False
    booked_slot: dict | None = None
    offered_slot_ids: list[int] = field(default_factory=list)
    # Traffic-only: True once the insurance/claim number has been asked and the
    # caller answered (with a number or a clear "none"). Gates qualification so
    # the step cannot be skipped by the model.
    insurance_resolved: bool = False
    # Callback-only: the caller's preferred time to be called back (free text).
    preferred_time: str | None = None
    misunderstanding_streak: int = 0
    last_transcription_confidence: float | None = None
    started_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "call_id": self.call_id,
            "phase": self.phase.value,
            "caller_intent": self.caller_intent.value,
            "legal_area": self.legal_area.value,
            "matter_summary": self.matter_summary,
            "entities": {
                k: {
                    "value": v.value,
                    "confidence": v.confidence,
                    "confirmed": v.confirmed,
                }
                for k, v in self.entities.items()
            },
            "turn_count": self.turn_count,
            "callback_requested": self.callback_requested,
            "target_person": self.target_person,
            "escalation_requested": self.escalation_requested,
            "booking_confirmed": self.booking_confirmed,
            "offered_slot_ids": self.offered_slot_ids,
            "last_transcription_confidence": self.last_transcription_confidence,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())
