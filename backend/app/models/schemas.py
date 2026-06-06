from dataclasses import dataclass, field
from enum import Enum


class CallPhase(str, Enum):
    GREETING = "greeting"
    INTENT_DETECTION = "intent_detection"
    ROUTING = "routing"
    INFORMATION = "information"
    CAPTURE = "capture"
    BOOKING = "booking"
    CONFIRMATION = "confirmation"
    ESCALATION = "escalation"
    FAREWELL = "farewell"


class CallerIntent(str, Enum):
    GENERAL_INFO = "general_info"
    BOOK_CONSULTATION = "book_consultation"
    UNKNOWN = "unknown"


class LegalArea(str, Enum):
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
