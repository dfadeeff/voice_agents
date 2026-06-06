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
    entities: dict[str, ExtractedEntity] = field(default_factory=dict)
    messages: list[dict] = field(default_factory=list)
    turn_count: int = 0
    escalation_requested: bool = False
    booking_confirmed: bool = False
    misunderstanding_streak: int = 0
    started_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "call_id": self.call_id,
            "phase": self.phase.value,
            "caller_intent": self.caller_intent.value,
            "legal_area": self.legal_area.value,
            "entities": {
                k: {
                    "value": v.value,
                    "confidence": v.confidence,
                    "confirmed": v.confirmed,
                }
                for k, v in self.entities.items()
            },
            "turn_count": self.turn_count,
            "escalation_requested": self.escalation_requested,
            "booking_confirmed": self.booking_confirmed,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())
