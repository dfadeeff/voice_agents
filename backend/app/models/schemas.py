from dataclasses import dataclass
from enum import StrEnum


class CallPhase(StrEnum):
    GREETING = "greeting"
    ROUTING = "routing"
    QUALIFICATION = "qualification"
    CAPTURE = "capture"
    BOOKING = "booking"
    CONFIRMATION = "confirmation"
    INFORMATION = "information"
    ESCALATION = "escalation"


class CallerIntent(StrEnum):
    GENERAL_INFO = "general_info"
    BOOK_CONSULTATION = "book_consultation"
    UNKNOWN = "unknown"


class LegalArea(StrEnum):
    EMPLOYMENT = "employment"
    TENANCY = "tenancy"
    TRAFFIC = "traffic"
    UNKNOWN = "unknown"


@dataclass
class ExtractedEntity:
    field_name: str
    value: str
    confidence: float
    confirmed: bool = False
    source_turn: int = 0
