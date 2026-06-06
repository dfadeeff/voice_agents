"""Custom Pipecat processors for the voice agent pipeline."""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from pipecat.frames.frames import (
    AggregatedTextFrame,
    Frame,
    TranscriptionFrame,
    TTSTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

logger = logging.getLogger(__name__)

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
    the frontend via the websocket (the output transport only serializes audio)."""

    def __init__(self, websocket, call_logger: CallLogger, **kwargs):
        super().__init__(**kwargs)
        self._ws = websocket
        self._logger = call_logger

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame) and frame.text and frame.text.strip():
            text = frame.text.strip()
            logger.info("USER: %s", text)
            self._logger.log("user", text)
            try:
                await self._ws.send_json({"type": "user_transcript", "text": text})
            except Exception:
                pass

        await self.push_frame(frame, direction)


class AgentTextProcessor(FrameProcessor):
    """Sits between TTS and output transport. Sends agent text to the frontend.
    Only catches AggregatedTextFrame (not TTSTextFrame) to avoid duplication."""

    def __init__(self, websocket, call_logger: CallLogger, **kwargs):
        super().__init__(**kwargs)
        self._ws = websocket
        self._logger = call_logger

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

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
