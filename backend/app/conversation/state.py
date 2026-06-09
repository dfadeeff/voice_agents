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
    # (date, time) pairs the caller declined — excluded from future offers so a
    # decline never re-surfaces the same time via another lawyer's slot.
    declined_slot_times: list[str] = field(default_factory=list)
    # Slots currently offered to the caller (deterministic booking).
    offered_slots: list[dict] = field(default_factory=list)
    # A specific time the caller asked for that wasn't free (e.g. "13 Uhr"); the
    # next slot offer apologises for it before listing the available alternatives.
    unavailable_time: str | None = None
    # How many times the matter-type question was asked without a recognised
    # answer; after a couple of misses we record it as "other" and move on
    # instead of re-asking forever.
    matter_attempts: int = 0
    # Candidate legal areas when the opening utterance matched more than one
    # (e.g. "Mietvertrag gekündigt" → tenancy + employment). Triggers a scripted
    # disambiguation question instead of leaving the call stalled in ROUTING.
    area_options: list[str] = field(default_factory=list)
    # Traffic-only: True once the insurance/claim number has been asked and the
    # caller answered (with a number or a clear "none"). Gates qualification so
    # the step cannot be skipped by the model.
    insurance_resolved: bool = False
    # How many times the insurance step was asked without a usable answer (neither
    # a number nor a clear "no"). The number is optional, so after a couple of
    # misses we proceed without it instead of re-asking forever.
    insurance_attempts: int = 0
    # Accumulates spoken reference fragments across turns — callers dictate a
    # number piece by piece with pauses ("F vier" … "fünf vier"), so each turn's
    # chunk is appended until it forms a complete reference.
    insurance_buffer: str = ""
    # Callback-only: the caller's preferred time to be called back (free text).
    preferred_time: str | None = None
    # Booking: True once email was asked and the caller had none — email is
    # optional, a phone number is enough to book.
    email_skipped: bool = False
    # True when we asked for the email but couldn't parse one from the reply, so
    # the next prompt apologises and asks again (the "didn't catch it" path).
    email_misheard: bool = False
    # Accumulates email-turn text across split utterances ("Klein at Hotmail" |
    # "Punkt de"), re-parsed as a whole each turn so a chunked address assembles.
    email_buffer: str = ""
    # How many times the email was asked but not understood. Spoken email is the
    # hardest field over phone-quality audio; after a few misses we stop looping
    # and skip it (a phone number is enough to book / call back).
    email_attempts: int = 0
    misunderstanding_streak: int = 0
    last_transcription_confidence: float | None = None
    # The specific datum the agent's last scripted question asked for. Set by
    # ConversationManager.next_prompt(); the deterministic reply parser dispatches
    # on this instead of regex-matching the agent's own previous sentence.
    # One of: area_confirm, matter_type, matter_details, insurance, name,
    # name_confirm, email, email_confirm, phone, phone_confirm, slot, callback_time.
    awaiting: str | None = None
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
            "awaiting": self.awaiting,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())
