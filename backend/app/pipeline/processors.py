"""Custom Pipecat processors for the voice agent pipeline."""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from pipecat.frames.frames import (
    AggregatedTextFrame,
    Frame,
    MetricsFrame,
    TranscriptionFrame,
    TTSTextFrame,
)
from pipecat.metrics.metrics import TTFBMetricsData
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

if TYPE_CHECKING:
    from app.conversation.manager import ConversationManager

logger = logging.getLogger(__name__)

_PHONE_RE = re.compile(r"(?<!\w)[+]?[\d][\d\s\-]{3,}[\d](?!\w)")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.\w+")


def _tts_preprocess(text: str, lang: str = "de") -> str:
    """Make agent text more TTS-friendly.

    - Expand phone-like digit sequences to space-separated digits so TTS
      reads them individually instead of as large numbers.
    - Expand email addresses to spoken form.
    """

    def _expand_phone(m: re.Match) -> str:
        digits = re.sub(r"[^\d]", "", m.group(0))
        prefix = "plus " if m.group(0).startswith("+") else ""
        return prefix + ", ".join(digits)

    def _expand_email(m: re.Match) -> str:
        email = m.group(0)
        if lang == "de":
            return email.replace("@", " at ").replace(".", " Punkt ")
        return email.replace("@", " at ").replace(".", " dot ")

    text = _EMAIL_RE.sub(_expand_email, text)
    text = _PHONE_RE.sub(_expand_phone, text)
    return text


LOGS_DIR = Path("logs")


class CallLogger:
    """Accumulates call transcript entries and writes them to a JSON file on save()."""

    def __init__(self, call_id: str):
        self.call_id = call_id
        self.start_time = datetime.now(UTC).isoformat()
        self.entries: list[dict] = []
        self.ttfa_samples: list[float] = []
        self.last_user_speech_end: float | None = None
        self.component_ttfb: dict[str, list[float]] = {}

    def record_user_speech_end(self) -> None:
        self.last_user_speech_end = time.monotonic()

    def record_first_agent_chunk(self) -> None:
        if self.last_user_speech_end is not None:
            ttfa = time.monotonic() - self.last_user_speech_end
            self.ttfa_samples.append(ttfa)
            logger.info("TTFA: %.0fms", ttfa * 1000)
            self.last_user_speech_end = None

    def record_component_ttfb(self, component: str, ttfb_ms: float) -> None:
        self.component_ttfb.setdefault(component, []).append(ttfb_ms)
        logger.info("%s TTFB: %.0fms", component, ttfb_ms)

    def log(self, role: str, text: str):
        self.entries.append(
            {
                "role": role,
                "text": text,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )

    def save(self):
        if not self.entries:
            return
        LOGS_DIR.mkdir(parents=True, exist_ok=True)

        metrics = {}
        if self.ttfa_samples:
            sorted_samples = sorted(self.ttfa_samples)
            metrics["ttfa_avg_ms"] = round(sum(sorted_samples) / len(sorted_samples) * 1000)
            metrics["ttfa_p50_ms"] = round(sorted_samples[len(sorted_samples) // 2] * 1000)
            metrics["ttfa_samples"] = len(sorted_samples)
            logger.info(
                "Call %s TTFA: avg=%dms p50=%dms (%d samples)",
                self.call_id,
                metrics["ttfa_avg_ms"],
                metrics["ttfa_p50_ms"],
                metrics["ttfa_samples"],
            )

        if self.component_ttfb:
            components = {}
            for name, samples in self.component_ttfb.items():
                sorted_s = sorted(samples)
                comp = {
                    "avg_ms": round(sum(sorted_s) / len(sorted_s)),
                    "p50_ms": round(sorted_s[len(sorted_s) // 2]),
                    "min_ms": round(min(sorted_s)),
                    "max_ms": round(max(sorted_s)),
                    "samples": len(sorted_s),
                }
                components[name] = comp
                logger.info(
                    "Call %s %s: avg=%dms p50=%dms min=%dms max=%dms (%d samples)",
                    self.call_id,
                    name,
                    comp["avg_ms"],
                    comp["p50_ms"],
                    comp["min_ms"],
                    comp["max_ms"],
                    comp["samples"],
                )
            metrics["component_ttfb"] = components

        log_data = {
            "call_id": self.call_id,
            "start_time": self.start_time,
            "end_time": datetime.now(UTC).isoformat(),
            "transcript": self.entries,
            "metrics": metrics,
        }
        path = LOGS_DIR / f"{self.call_id}.json"
        path.write_text(json.dumps(log_data, indent=2))
        logger.info("Call log saved: %s (%d entries)", path, len(self.entries))


class MetricsProcessor(FrameProcessor):
    """Captures Pipecat's per-component TTFB metrics (STT, LLM, TTS)."""

    def __init__(self, call_logger: CallLogger, **kwargs):
        super().__init__(**kwargs)
        self._logger = call_logger

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, MetricsFrame):
            for datum in frame.data:
                if isinstance(datum, TTFBMetricsData):
                    self._logger.record_component_ttfb(datum.processor, datum.value)

        await self.push_frame(frame, direction)


class TranscriptProcessor(FrameProcessor):
    """Sits between STT and UserAggregator. Sends user transcriptions to
    the frontend via the websocket (the output transport only serializes audio).
    Also advances the conversation state machine on each user turn."""

    def __init__(
        self,
        websocket,
        call_logger: CallLogger,
        conversation: ConversationManager,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._ws = websocket
        self._logger = call_logger
        self._conversation = conversation

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame) and frame.text and frame.text.strip():
            text = frame.text.strip()
            logger.info("USER: %s", text)
            self._logger.log("user", text)
            self._logger.record_user_speech_end()

            old_phase = self._conversation.state.phase
            self._conversation.add_user_message(text)
            new_phase = self._conversation.state.phase
            if new_phase != old_phase:
                logger.info("Phase: %s → %s", old_phase.value, new_phase.value)

            try:
                await self._ws.send_json({"type": "user_transcript", "text": text})
            except Exception:
                pass

        await self.push_frame(frame, direction)


class AgentTextProcessor(FrameProcessor):
    """Sits between LLM and TTS. Sends agent text to the frontend and
    preprocesses text for better TTS pronunciation (digits, emails)."""

    def __init__(self, websocket, call_logger: CallLogger, lang: str = "de", **kwargs):
        super().__init__(**kwargs)
        self._ws = websocket
        self._logger = call_logger
        self._lang = lang
        self._first_chunk_this_turn = True

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TTSTextFrame) and frame.text:
            if self._first_chunk_this_turn:
                self._logger.record_first_agent_chunk()
                self._first_chunk_this_turn = False
            original = frame.text
            processed = _tts_preprocess(original, self._lang)
            if processed != original:
                logger.debug("TTS preprocess: %r → %r", original, processed)
                frame = TTSTextFrame(text=processed)

        if (
            isinstance(frame, AggregatedTextFrame)
            and not isinstance(frame, TTSTextFrame)
            and frame.text
        ):
            logger.info("AGENT: %s", frame.text)
            self._logger.log("agent", frame.text)
            self._first_chunk_this_turn = True
            try:
                await self._ws.send_json({"type": "agent_text", "text": frame.text})
            except Exception:
                pass

        await self.push_frame(frame, direction)
