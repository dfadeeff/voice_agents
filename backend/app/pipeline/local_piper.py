"""Local Piper TTS with a short inter-sentence pause.

Pipecat feeds one sentence per ``run_tts`` call (PreTTSSanitizer splits on
sentence boundaries), and Piper emits that sentence's audio with no trailing
silence. So consecutive sentences play back-to-back and run together —
"...der Kanzlei.Ich nehme gerne..." sounds like one breathless utterance.

We append a short silence to each synthesized sentence so the boundary is
audible and the speech sounds natural.

Why this is email-safe: email addresses are expanded to "... at ... Punkt ..."
before TTS (see processors._tts_preprocess), so an email's "." never reaches
Piper as a period. And the pause is added once per sentence (per run_tts call),
not per "." in the text — so "gmail.com" is never split or paused mid-address.
Cloud TTS (ElevenLabs) is unaffected: it renders sentence prosody natively.
"""

import asyncio
import logging
from collections.abc import AsyncGenerator, AsyncIterator

from pipecat.frames.frames import ErrorFrame, Frame
from pipecat.services.piper.tts import PiperTTSService

logger = logging.getLogger(__name__)


class LocalPiperTTSService(PiperTTSService):
    """Piper TTS that pads each synthesized sentence with trailing silence."""

    def __init__(self, *, sentence_pause_ms: int = 180, **kwargs):
        super().__init__(**kwargs)
        self._sentence_pause_ms = max(0, sentence_pause_ms)

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        def async_next(it):
            try:
                return next(it)
            except StopIteration:
                return None

        in_sample_rate = self._voice.config.sample_rate

        async def async_iterator() -> AsyncIterator[bytes]:
            iterator = self._voice.synthesize(text)
            while True:
                item = await asyncio.to_thread(async_next, iterator)
                if item is None:
                    break
                yield item.audio_int16_bytes
            # Trailing silence so the next sentence doesn't butt right up against
            # this one. int16 mono → 2 bytes per sample.
            if self._sentence_pause_ms > 0:
                num_samples = int(in_sample_rate * self._sentence_pause_ms / 1000)
                yield b"\x00\x00" * num_samples

        logger.debug("Piper TTS (pause %dms): %s", self._sentence_pause_ms, text)
        try:
            await self.start_tts_usage_metrics(text)
            async for frame in self._stream_audio_frames_from_iterator(
                async_iterator(),
                in_sample_rate=in_sample_rate,
                context_id=context_id,
            ):
                await self.stop_ttfb_metrics()
                yield frame
        except Exception as e:
            logger.error("Piper TTS exception: %s", e)
            yield ErrorFrame(error=f"Unknown error occurred: {e}")
        finally:
            await self.stop_ttfb_metrics()
