"""Local Whisper service with domain bias and transcription confidence."""

import asyncio
import logging
import math
from collections.abc import AsyncGenerator

import numpy as np
from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.services.settings import assert_given
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.utils.time import time_now_iso8601

logger = logging.getLogger(__name__)

_LEGAL_HOTWORDS = (
    "Anwaltskanzlei Arbeitsrecht Mietrecht Kündigung Abmahnung Arbeitgeber "
    "Vermieter Kaution Beratungstermin E-Mail-Adresse Telefonnummer Rückruf"
)

# Spoken emails are the field Whisper-small mangles worst over phone audio: it
# tends to drop the "@gmail" middle and emit just "Steinfeld.com". Biasing the
# decoder toward the connectors and the common German/global mail domains during
# the email turn keeps those tokens in the transcript so the parser can recover
# the address. Only applied when the agent just asked for the email.
_EMAIL_HOTWORDS = (
    "E-Mail-Adresse at ät punkt gmail.com googlemail.com gmx.de gmx.net web.de "
    "outlook.de hotmail.de hotmail.com yahoo.de yahoo.com icloud.com t-online.de"
)
# During phone dictation, bias toward the spoken digit words so they aren't
# rendered as decimals ("5.1, 5.6") or merged into ordinals.
_PHONE_HOTWORDS = (
    "Telefonnummer Handynummer Vorwahl " "null eins zwei drei vier fünf sechs sieben acht neun"
)

_shared_model = None


def preload_whisper(model_size: str = "medium", device: str = "cpu", compute_type: str = "int8"):
    """Load Whisper model once at startup. Called from app.main."""
    global _shared_model
    if _shared_model is None:
        from faster_whisper import WhisperModel

        logger.info("Pre-loading Whisper model '%s' (%s/%s)...", model_size, device, compute_type)
        _shared_model = WhisperModel(model_size, device=device, compute_type=compute_type)
        logger.info("Whisper model ready.")
    return _shared_model


class LocalWhisperSTTService(WhisperSTTService):
    """Faster Whisper tuned for short German legal-intake turns.

    Uses a shared model instance loaded at startup to avoid per-call reload.
    """

    def __init__(self, *, beam_size: int = 1, vad_filter: bool = False, **kwargs):
        super().__init__(**kwargs)
        self._beam_size = beam_size
        self._vad_filter = vad_filter
        self._conversation = None

    def set_conversation(self, conversation) -> None:
        """Attach the conversation manager so the decoder can be biased toward the
        tokens expected for the field the agent just asked for (email/phone)."""
        self._conversation = conversation

    def _hotwords_for_turn(self) -> str:
        """German legal hotwords, plus field-specific bias for the awaited slot."""
        awaiting = getattr(getattr(self._conversation, "state", None), "awaiting", None)
        if awaiting in ("email", "email_confirm"):
            return _LEGAL_HOTWORDS + " " + _EMAIL_HOTWORDS
        if awaiting in ("phone", "phone_confirm"):
            return _LEGAL_HOTWORDS + " " + _PHONE_HOTWORDS
        return _LEGAL_HOTWORDS

    def _load(self):
        if _shared_model is not None:
            self._model = _shared_model
        else:
            super()._load()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        if not self._model:
            yield ErrorFrame("Whisper model not available")
            return

        await self.start_processing_metrics()
        audio_float = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        language = assert_given(self._settings.language)

        beam_size = self._beam_size
        vad_filter = self._vad_filter
        hotwords = self._hotwords_for_turn()

        def transcribe():
            kwargs = {
                "language": language,
                "beam_size": beam_size,
                "temperature": 0.0,
                "condition_on_previous_text": False,
                "vad_filter": vad_filter,
            }
            if vad_filter:
                kwargs["vad_parameters"] = {"min_silence_duration_ms": 300}
            if str(language).lower().endswith("de"):
                kwargs["hotwords"] = hotwords
            segments, info = self._model.transcribe(audio_float, **kwargs)
            return list(segments), info

        try:
            segments, info = await asyncio.to_thread(transcribe)
            threshold = assert_given(self._settings.no_speech_prob)
            accepted = [
                segment
                for segment in segments
                if threshold is None or segment.no_speech_prob < threshold
            ]
            text = " ".join(segment.text.strip() for segment in accepted).strip()
            if accepted:
                confidence = sum(math.exp(segment.avg_logprob) for segment in accepted) / len(
                    accepted
                )
            else:
                confidence = 0.0
        finally:
            await self.stop_processing_metrics()

        if text:
            result = {
                "confidence": round(max(0.0, min(1.0, confidence)), 3),
                "language": getattr(info, "language", str(language)),
            }
            await self._handle_transcription(text, True, language)
            yield TranscriptionFrame(
                text,
                self._user_id,
                time_now_iso8601(),
                language,
                result=result,
            )
