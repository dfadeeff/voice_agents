from dataclasses import dataclass, field
from enum import StrEnum


class CallPhase(StrEnum):
    GREETING = "greeting"
    INTENT_DETECTION = "intent_detection"
    ROUTING = "routing"
    INTAKE = "intake"
    INFORMATION = "information"
    CAPTURE = "capture"
    CONFLICT_CHECK = "conflict_check"
    ADDITIONAL_INFO = "additional_info"
    BOOKING = "booking"
    CONFIRMATION = "confirmation"
    ESCALATION = "escalation"
    FAREWELL = "farewell"


class CallerIntent(StrEnum):
    GENERAL_INFO = "general_info"
    BOOK_CONSULTATION = "book_consultation"
    UNKNOWN = "unknown"


class LegalArea(StrEnum):
    EMPLOYMENT = "employment"
    TENANCY = "tenancy"
    UNKNOWN = "unknown"


@dataclass
class WordInfo:
    word: str
    start_time: float
    end_time: float
    confidence: float


@dataclass
class STTSegment:
    text: str
    is_final: bool
    confidence: float
    words: list[WordInfo] = field(default_factory=list)
    language: str = "en"


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMChunk:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict = field(default_factory=dict)


@dataclass
class AudioChunk:
    data: bytes
    sample_rate: int
    is_last: bool = False


@dataclass
class ExtractedEntity:
    field_name: str
    value: str
    confidence: float
    confirmed: bool = False
    source_turn: int = 0
