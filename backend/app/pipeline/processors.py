"""Custom Pipecat processors for the voice agent pipeline."""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from pipecat.frames.frames import (
    AggregatedTextFrame,
    Frame,
    TranscriptionFrame,
    TTSTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

if TYPE_CHECKING:
    from app.conversation.manager import ConversationManager

logger = logging.getLogger(__name__)

_PHONE_RE = re.compile(r"(?<!\w)[+]?[\d][\d\s\-]{3,}[\d](?!\w)")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.\w+")


def _tts_preprocess(text: str) -> str:
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
        log_data = {
            "call_id": self.call_id,
            "start_time": self.start_time,
            "end_time": datetime.now(UTC).isoformat(),
            "transcript": self.entries,
        }
        path = LOGS_DIR / f"{self.call_id}.json"
        path.write_text(json.dumps(log_data, indent=2))
        logger.info("Call log saved: %s (%d entries)", path, len(self.entries))


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

    def __init__(self, websocket, call_logger: CallLogger, **kwargs):
        super().__init__(**kwargs)
        self._ws = websocket
        self._logger = call_logger

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TTSTextFrame) and frame.text:
            original = frame.text
            processed = _tts_preprocess(original)
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
            try:
                await self._ws.send_json({"type": "agent_text", "text": frame.text})
            except Exception:
                pass

        await self.push_frame(frame, direction)
